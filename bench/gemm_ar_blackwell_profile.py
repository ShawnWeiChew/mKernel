"""In-kernel profile of gemm_ar_blackwell at a single shape, on one GPU.

Unlike gemm_ar_blackwell_bench.py this measures *inside* the kernel: every warp
records (tag, start, duration) around the points instrumented in
src/gemm_ar_blackwell.cu, so the output answers "which wait is eating the
launch" rather than "how long did the launch take".

Single process, single device. With config::NUM_COMP_SM == NUM_BLOCKS there are
no comm SMs, so the kernel is a plain local GEMM and needs no peers -- build
with INTRA_NUM_DEVICES=1 and the DistBuffers below are just local allocations.

Usage:
    python bench/gemm_ar_blackwell_profile.py --shape 2048 --out trace.json

Requires a .so built from a tree that binds entrypoint_profile. The profiled
kernel is a separate template instantiation and is always compiled in, so no
special build flag is needed.
"""

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402

# Must match ProfilerTag in include/operators/gemm_ar/profiler.h, in order.
TAGS = [
    "SETUP",
    "ISSUE_TMA",
    "ISSUE_MMA",
    "WAIT_TMA",
    "WAIT_MMA",
    "WAIT_MAINLOOP",
    "WAIT_EPILOGUE",
    "EPILOGUE",
]

# Must match config:: in include/operators/gemm_ar/gemm_ar_blackwell.cuh.
# The kernel keys its profiler slot on blockIdx.x * NUM_WARPS + warp_id, so an
# undersized NUM_WARPS here is an out-of-bounds write, not just a bad decode.
NUM_BLOCKS = 148
EPILOGUE_WARPS = 8
PRODUCER_WARPS = 1
CONSUMER_WARPS = 2
# +1 padding warp, so the CTA is whole warpgroups for setmaxnreg.
NUM_WARPS = EPILOGUE_WARPS + PRODUCER_WARPS + CONSUMER_WARPS + 1

WARMUP = 5


def warp_role(warp: int) -> str:
    if warp < EPILOGUE_WARPS:
        return "epilogue"
    if warp < EPILOGUE_WARPS + PRODUCER_WARPS:
        return "producer"
    if warp < EPILOGUE_WARPS + PRODUCER_WARPS + CONSUMER_WARPS:
        return "consumer"
    return "padding"


def decode(profiler: torch.Tensor, num_entries: int):
    """Profiler buffer -> (events, truncated_rows).

    Row layout is [count, (sm_id, tag, start_ns, duration_ns) * count], one row
    per warp, laid out as blockIdx.x * NUM_WARPS + warp_id.
    """
    rows = profiler.cpu().tolist()
    events = []
    truncated = 0

    for row_id, data in enumerate(rows):
        count = data[0]
        if count <= 0:
            continue
        if count >= num_entries:
            # The kernel does not bound-check cnt_, so a row at the cap means
            # entries were dropped and neighbouring rows may be corrupt.
            truncated += 1
            count = num_entries

        block, warp = divmod(row_id, NUM_WARPS)
        role = warp_role(warp)
        for i in range(count):
            sm_id, tag, start, duration = data[1 + i * 4 : 1 + (i + 1) * 4]
            events.append(
                dict(
                    name=TAGS[tag],
                    cat=role,
                    ph="X",
                    # %globaltimer is ns; the trace format wants microseconds.
                    ts=start / 1000.0,
                    dur=duration / 1000.0,
                    pid=sm_id,
                    tid=row_id,
                    args=dict(block=block, warp=warp, role=role, sm=sm_id),
                )
            )

    return events, truncated


def to_trace(events):
    """Zero the clock and attach readable process/thread names."""
    if not events:
        return dict(traceEvents=[], displayTimeUnit="ns")

    offset = min(e["ts"] for e in events)
    for e in events:
        e["ts"] -= offset

    meta = []
    for pid in sorted({e["pid"] for e in events}):
        meta.append(
            dict(name="process_name", ph="M", pid=pid, tid=0, args=dict(name=f"SM {pid}"))
        )
    seen = {}
    for e in events:
        seen.setdefault((e["pid"], e["tid"]), e["args"])
    for (pid, tid), a in sorted(seen.items()):
        meta.append(
            dict(
                name="thread_name",
                ph="M",
                pid=pid,
                tid=tid,
                args=dict(name=f"blk{a['block']} w{a['warp']} {a['role']}"),
            )
        )

    return dict(traceEvents=meta + events, displayTimeUnit="ns")


def summarize(events):
    """Total/mean ns per (role, tag), sorted by total time. This is the table
    that actually says where the kernel is spending itself."""
    agg = {}
    for e in events:
        key = (e["cat"], e["name"])
        tot, n = agg.get(key, (0.0, 0))
        agg[key] = (tot + e["dur"] * 1000.0, n + 1)

    lines = [f"  {'role':<9} {'tag':<15} {'count':>7} {'total us':>11} {'mean ns':>10}"]
    for (role, tag), (total_ns, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
        lines.append(
            f"  {role:<9} {tag:<15} {n:>7} {total_ns / 1000.0:>11.1f} {total_ns / n:>10.1f}"
        )
    return "\n".join(lines)


def span_us(events):
    """Wall span of the instrumented region, max over SMs -- a sanity check
    against the kernel time the bench reports."""
    if not events:
        return 0.0
    return max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, default=2048, help="square M=N=K")
    ap.add_argument("--k", type=int, default=0, help="override K (default: = shape)")
    ap.add_argument("--entries", type=int, default=1000, help="max events recorded per warp")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--out", type=str, default="", help="write the trace JSON here")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    mod = load_module.load("gemm_ar_blackwell")
    if not hasattr(mod, "gemm_ar_intranode_blackwell_profile"):
        print(
            "ERROR: the .so has no gemm_ar_intranode_blackwell_profile binding; "
            "rebuild after adding entrypoint_profile.",
            flush=True,
        )
        return 1

    M = N = args.shape
    K = args.k or args.shape
    print(f"profiling M={M} K={K} N={N}, {args.entries} entries/warp", flush=True)

    torch.manual_seed(42)
    A = torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / (K**0.25)
    B = torch.randn((K, N), device="cuda", dtype=torch.bfloat16) / (K**0.25)

    # local_world_size=1: LocalBroker's barrier completes on its own and every
    # distributed slot resolves to this process's own buffer.
    def dbuf(shape, dtype, multicast):
        return mod.DistBuffer(
            shape,
            dtype=dtype,
            local_rank=args.device,
            local_world_size=1,
            multicast=multicast,
        )

    # A multicast group of one device can be refused by the driver. Nothing in
    # this configuration needs it: with config::NUM_COMP_SM == NUM_BLOCKS no
    # comm SM ever runs, and distributed_tensor_from_buffer copies
    # multicast_ptr_ without checking it, so a null one is inert. Probe once and
    # build every buffer the same way.
    multicast = True
    try:
        C_dbuf = dbuf((M, N), torch.bfloat16, True)
    except Exception as e:  # noqa: BLE001 - driver-dependent, message varies
        print(f"note: 1-device multicast unavailable ({e}); using multicast=False", flush=True)
        multicast = False
        C_dbuf = dbuf((M, N), torch.bfloat16, False)

    barrier = dbuf((2, 1024, 1024), torch.int, multicast)
    C_final = dbuf((M, N), torch.bfloat16, multicast)

    profiler = torch.zeros(
        NUM_BLOCKS * NUM_WARPS, 1 + args.entries * 4, dtype=torch.int64, device="cuda"
    )

    def reset():
        C_dbuf.data_.zero_()
        barrier.data_.zero_()
        C_final.data_.zero_()
        torch.cuda.synchronize()

    run = lambda: mod.gemm_ar_intranode_blackwell_profile(  # noqa: E731
        A, B, C_dbuf, barrier, C_final, profiler, args.entries
    )

    # Warm up on the instrumented kernel itself: it is a different cubin from
    # the fast one, and a cold-icache first launch is exactly the outlier the
    # trace would otherwise be built from.
    for _ in range(WARMUP):
        reset()
        run()
    torch.cuda.synchronize()

    # The kernel only writes the entries it records and never clears stale
    # ones, so the buffer has to be zeroed for the run being kept.
    reset()
    profiler.zero_()
    torch.cuda.synchronize()

    run()
    torch.cuda.synchronize()

    # Correctness guard: a trace of a kernel that computed garbage is a trace of
    # the wrong thing.
    ref = torch.matmul(A, B)
    err = (C_dbuf.data_.float() - ref.float()).abs().max().item()
    print(f"max |C - A@B| = {err:.4f}", flush=True)

    events, truncated = decode(profiler, args.entries)
    if truncated:
        print(
            f"WARNING: {truncated} warp row(s) hit the {args.entries}-entry cap; "
            f"raise --entries (the kernel does not bound-check, so those rows "
            f"overran into their neighbour)",
            flush=True,
        )

    active_sms = len({e["pid"] for e in events})
    print(f"\n{len(events)} events across {active_sms} SMs, "
          f"instrumented span {span_us(events):.1f} us\n", flush=True)
    print(summarize(events), flush=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(to_trace(events)))
        print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB) "
              f"-- open in chrome://tracing or ui.perfetto.dev", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
