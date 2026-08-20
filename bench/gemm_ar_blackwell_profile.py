"""In-kernel timing profile of gemm_ar_blackwell at a single shape, on one GPU.

Unlike gemm_ar_blackwell_bench.py this measures *inside* the kernel: every warp
stamps a (timestamp, event, payload) record at each phase boundary instrumented
in src/gemm_ar_blackwell.cu, and the result is a per-warp Gantt chart. It
answers "who is waiting on whom" rather than "how long did the launch take" --
a question neither Nsight Systems nor Nsight Compute can answer, since both see
one launch as one bar and cannot separate concurrent warp roles inside it.

Needs the instrumented .so, which is a separate build so the shipping one keeps
zero profiling instructions in its SASS:

    make gemm_ar_blackwell_profile
    python bench/gemm_ar_blackwell_profile.py --shape 4096 --out traces/4096.npz

Then re-render offline as often as you like, without a GPU:

    python python/timings.py traces/4096.npz --blocks 0-7

Single process, single device. With config::NUM_COMP_SM == NUM_BLOCKS there are
no comm SMs, so the kernel is a plain local GEMM and needs no peers -- build
with INTRA_NUM_DEVICES=1 and the DistBuffers below are just local allocations.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
import timings  # noqa: E402

WARMUP = 5


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", type=int, default=2048, help="square M=N=K")
    ap.add_argument("--k", type=int, default=0, help="override K (default: = shape)")
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument(
        "--out", type=str, default="traces/gemm_ar_blackwell.npz", help="trace .npz path"
    )
    ap.add_argument("--pdf", type=str, default="", help="Gantt PDF (default: <out>.pdf)")
    ap.add_argument(
        "--blocks",
        type=str,
        default="",
        help="only draw these CTAs, e.g. '0-7'. The .npz always keeps them all.",
    )
    ap.add_argument("--no-plot", action="store_true", help="dump the .npz only")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    mod = load_module.load("gemm_ar_blackwell_profile")
    # The design's load-time probe: these attributes exist only in a build that
    # passed -DPROFILE_TIMINGS.
    if not hasattr(mod, "EVENTS_PER_BLOCK"):
        print(
            "ERROR: this .so was built without -DPROFILE_TIMINGS. "
            "Run `make gemm_ar_blackwell_profile`.",
            flush=True,
        )
        return 1

    num_blocks = mod.TIMING_NUM_BLOCKS
    num_warps = mod.TIMING_NUM_WARPS
    cap = mod.EVENTS_PER_BLOCK
    event_map = dict(mod.TIMING_EVENTS)

    M = N = args.shape
    K = args.k or args.shape
    ring_mb = num_blocks * cap * mod.TIMING_RECORD_SIZE / 1e6
    print(
        f"profiling M={M} K={K} N={N}; ring = {num_blocks} blocks x {cap} events "
        f"({ring_mb:.0f} MB)",
        flush=True,
    )

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

    # Two int64s per 16-byte record -- a shape torch round-trips cheaply. Zeroed,
    # which is what makes timestamp == 0 the "never written" sentinel and lets
    # the host recover each CTA's event count without a head array.
    ring = torch.zeros(num_blocks * cap * 2, dtype=torch.int64, device="cuda")

    def reset():
        C_dbuf.data_.zero_()
        barrier.data_.zero_()
        C_final.data_.zero_()
        torch.cuda.synchronize()

    run = lambda: mod.gemm_ar_intranode_blackwell_profile(  # noqa: E731
        A, B, C_dbuf, barrier, C_final, ring
    )

    # Warm up on the instrumented kernel itself: it is a different cubin from
    # the fast one, and a cold-icache first launch is exactly the outlier the
    # trace would otherwise be built from.
    for _ in range(WARMUP):
        reset()
        run()
    torch.cuda.synchronize()

    # Exactly one iteration into a freshly zeroed ring. Several iterations would
    # reuse payload values within a CTA slot and collide in the pairing key.
    reset()
    ring.zero_()
    torch.cuda.synchronize()

    run()
    torch.cuda.synchronize()

    # Correctness guard: a trace of a kernel that computed garbage is a trace of
    # the wrong thing.
    ref = torch.matmul(A, B)
    err = (C_dbuf.data_.float() - ref.float()).abs().max().item()
    print(f"max |C - A@B| = {err:.4f}", flush=True)

    records, heads = timings.unpack(ring.cpu().numpy(), num_blocks, cap)
    overflowed = int((heads >= cap).sum())
    if overflowed:
        print(
            f"WARNING: {overflowed} CTA(s) filled all {cap} slots; their timelines "
            f"are truncated at the same x-coordinate. Rebuild with "
            f"`make EVENTS_PER_BLOCK={cap * 2} gemm_ar_blackwell_profile`.",
            flush=True,
        )

    print(
        f"\n{records.shape[0]} events across {len(set(records[:, 0].tolist()))} CTAs, "
        f"wall span {timings.wall_span_us(records):.1f} us\n",
        flush=True,
    )
    print(timings.summarize(timings.spans_for_all_roles(records, event_map, num_warps)), flush=True)

    out = timings.save(
        args.out,
        records,
        event_map,
        num_blocks=num_blocks,
        num_warps=num_warps,
        events_per_block=cap,
        overflowed_blocks=overflowed,
        M=M,
        N=N,
        K=K,
        rank=args.device,
        title=f"gemm_ar_blackwell M={M} N={N} K={K}",
    )
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)", flush=True)

    if args.no_plot:
        return 0

    blocks = timings.parse_blocks(args.blocks)
    draw = records if blocks is None else records[np.isin(records[:, 0], sorted(blocks))]
    pdf = Path(args.pdf) if args.pdf else Path(args.out).with_suffix(".pdf")
    written = timings.plot(
        draw, pdf, event_map, num_warps=num_warps, title=f"M={M} N={N} K={K}"
    )
    if written is None:
        print("no spans paired -- nothing to draw", flush=True)
        return 1
    print(f"wrote {written} ({written.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
