"""NVLink/NVSwitch bandwidth sweep for the gemm_ar comm path.

Drives `test_nvswitch_bw` (src/gemm_ar_blackwell.cu, namespace mnvl_bw_test):
the fused kernel's multimem all-reduce loop with the GEMM, the barrier and the
compute SMs stripped out. The question it answers is *what is the smallest
output tile a comm CTA can be handed before the NVSwitch, and not the tile
walk, becomes the bottleneck* — i.e. how few SMs the fused kernel has to give
up to comm.

Launch (one process per GPU, single node):

    python -m torch.distributed.run --standalone --nproc-per-node=8 \
        bench/mnvl_bw_sweep.py

The full default grid is 2*5*6*4*2 = 480 configs per shape. Narrow it while
exploring, e.g.:

    ... bench/mnvl_bw_sweep.py --shapes 8192 --subtile-m 128 \
        --comm-sms 16,24,32 --ar-unroll 8

Work split (mirrors the kernel): device d owns the tile ids congruent to
d mod world_size, so the node together covers the whole matrix exactly once.
That is what makes --mode check meaningful: an indexing bug that skips tiles
would otherwise just look like a faster kernel.

The two phases use different data on purpose. `--mode check` runs on an exact
small-integer pattern so the result can be compared bit-for-bit; the timed runs
fill C_dist with randn, because the real buffer holds GEMM output and that is
what the all-reduce should be measured on.

Each shape also times a plain NCCL all-reduce over the same bytes as a ceiling
(--no-baseline to skip it). Compare the `ar GB/s` columns — those are both just
"matrix bytes per second". The `nvl GB/s` column means different things per row
kind: for `mnvl` rows it is the per-GPU per-direction NVLink estimate, for the
`nccl` row it is nccl-tests busbw, and the two accountings are not the same.

--save-csv columns: kind (mnvl|nccl), M, N, subtile_m, subtile_n, num_comm_sm,
ar_unroll, supergroup_width, repeats, ms, ar_gbps, nvl_gbps. The nccl rows
carry 0 in every tile-config column.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402

DEFAULT_SHAPES = [4096, 8192, 16384, 32768]
DEFAULT_SUBTILE_M = [128, 256]
DEFAULT_SUBTILE_N = [16, 32, 64, 128, 256]
DEFAULT_COMM_SMS = [12, 16, 20, 24, 28, 32]
DEFAULT_AR_UNROLL = [4, 8, 16, 32]
DEFAULT_SUPERGROUP = [4, 8]

# Correctness pattern: value(i, j) = ((i + j) % period + 1) * (rank + 1), so the
# node-wide sum is (value_base) * world*(world+1)/2. `period` is chosen as the
# largest one keeping that sum <= 256, which bf16 represents exactly — that lets
# the check demand bit equality instead of picking a tolerance.
_PERIOD_CANDIDATES = [23, 19, 17, 13, 11, 7, 5, 3]

# Rows compared / filled per step. Keeps the temporaries bounded at the 32768
# shape, where a whole-matrix int32 intermediate would be 4 GiB.
FILL_CHUNK_ROWS = 4096


def parse_int_list(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x.strip()]


def sync_ranks():
    """Drain the local stream, then line every rank up on the host, so each
    timed iteration starts from an idle stream on every GPU."""
    torch.cuda.synchronize()
    dist.barrier()


def median_then_max(samples: list[float]) -> float:
    """Local median over iters, then max over ranks — the slowest rank sets the
    cost of a collective, so that is the number worth reporting."""
    ordered = sorted(float(x) for x in samples)
    t = torch.tensor([ordered[len(ordered) // 2]], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def pattern_period(world: int) -> int:
    tri = world * (world + 1) // 2
    for p in _PERIOD_CANDIDATES:
        if p * tri <= 256:
            return p
    raise ValueError(f"world={world} too large for an exact bf16 correctness pattern")


def _row_table(N: int, period: int, scale: int) -> torch.Tensor:
    """(period, N) bf16 table; row r is the pattern for any matrix row i == r
    (mod period), pre-multiplied by `scale`."""
    cols = torch.arange(N, device="cuda", dtype=torch.int32).unsqueeze(0)
    rows = torch.arange(period, device="cuda", dtype=torch.int32).unsqueeze(1)
    return ((((cols + rows) % period) + 1) * scale).to(torch.bfloat16)


def fill_pattern(dst: torch.Tensor, period: int, scale: int) -> None:
    """Write the position-dependent pattern into `dst` in row chunks.

    Position dependence is the point: a constant fill would pass even if the
    kernel wrote tile A's data into tile B's slot.
    """
    M, N = dst.shape
    table = _row_table(N, period, scale)
    idx = torch.arange(M, device="cuda", dtype=torch.int64) % period
    for s in range(0, M, FILL_CHUNK_ROWS):
        e = min(s + FILL_CHUNK_ROWS, M)
        dst[s:e].copy_(table[idx[s:e]])


def fill_random(dst: torch.Tensor, seed: int) -> None:
    """Fill `dst` with randn in place — no temporary, whatever the shape.

    Timing runs on float data on purpose: the buffer the fused kernel
    all-reduces holds GEMM output, not the small exact integers the
    correctness pattern uses. Seeded per rank so a run is reproducible and the
    devices are not all reducing the same values.
    """
    g = torch.Generator(device=dst.device)
    g.manual_seed(seed)
    dst.normal_(0.0, 1.0, generator=g)


def count_mismatch(observed: torch.Tensor, period: int, scale: int) -> int:
    """Bit-exact mismatch count against the summed pattern, chunked."""
    M, N = observed.shape
    table = _row_table(N, period, scale)
    idx = torch.arange(M, device="cuda", dtype=torch.int64) % period
    bad = 0
    for s in range(0, M, FILL_CHUNK_ROWS):
        e = min(s + FILL_CHUNK_ROWS, M)
        bad += int((observed[s:e] != table[idx[s:e]]).sum().item())
    return bad


def auto_repeats(M: int, N: int, target_mb: int) -> int:
    """Replay count for one launch.

    At 4096x4096 the matrix is only 32 MiB, so a single pass finishes in a few
    microseconds and the measurement is mostly launch overhead. The tile walk
    is idempotent (C_dist is never mutated, multimem.st is a plain store), so
    replaying it inside the kernel is a free way to get a stable sample.
    """
    matrix_mb = M * N * 2 / (1 << 20)
    return max(1, min(64, int(round(target_mb / matrix_mb))))


def bandwidth_gbps(M: int, N: int, world: int, repeats: int, ms: float) -> tuple[float, float]:
    """(all-reduce rate, per-GPU per-direction NVLink rate) in GB/s.

    Per 4-byte bf16_2 unit in the node the switch moves, for the GPU that
    issues it: 4B out (the multimem.st) and 4B in (the reduced ld_reduce
    result). Every *other* GPU also puts 4B on the wire (its contribution to
    the reduction) and takes 4B back (its copy of the broadcast store). Summed
    per GPU over a whole pass:

        egress = ingress = matrix_bytes * (1 + 1/world)

    so `nvl` is the number to hold against the per-direction NVLink peak
    (~900 GB/s per GPU on GB300), while `ar` is the plain "how fast does the
    whole matrix get all-reduced" rate.
    """
    matrix_bytes = M * N * 2 * repeats
    seconds = ms * 1e-3
    ar = matrix_bytes / seconds / 1e9
    nvl = matrix_bytes * (1.0 + 1.0 / world) / seconds / 1e9
    return ar, nvl


def nccl_baseline(M: int, N: int, world: int, repeats: int,
                  warmup: int, iters: int, seed: int) -> tuple[float, float, float]:
    """Time NCCL's own all-reduce on the same bytes. Returns (ms, ar, busbw).

    This is the ceiling to read the sweep against: NCCL picks its own
    algorithm (on an NVSwitch box that is usually NVLS, i.e. the same multimem
    hardware path this kernel drives by hand), so a config that lands near it
    is bandwidth-bound rather than tile-walk-bound.

    `busbw` is the nccl-tests convention, algbw * 2*(n-1)/n, kept so the number
    can be compared against published NCCL figures. It is *not* the same
    accounting as the sweep's `nvl` column — a ring all-reduce and a multimem
    all-reduce put different traffic on the wire for the same payload — so
    compare the `ar` columns, which are both just "matrix bytes per second".
    """
    buf = torch.empty((M, N), device="cuda", dtype=torch.bfloat16)
    fill_random(buf, seed)
    for _ in range(warmup):
        for _ in range(repeats):
            dist.all_reduce(buf)
    sync_ranks()

    events = []
    for _ in range(iters):
        # Refill outside the timed window. all_reduce is in place, so with
        # repeats > 1 the magnitudes grow by a factor of `world` per pass and
        # eventually saturate to inf. That does not change the timing — NVIDIA
        # float throughput is value-independent — but each sample should still
        # start from real data.
        fill_random(buf, seed)
        sync_ranks()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(repeats):
            dist.all_reduce(buf)
        e.record()
        events.append((s, e))
    torch.cuda.synchronize()

    ms = median_then_max([s.elapsed_time(e) for s, e in events])
    matrix_bytes = M * N * 2 * repeats
    algbw = matrix_bytes / (ms * 1e-3) / 1e9
    busbw = algbw * 2.0 * (world - 1) / world

    del buf
    torch.cuda.empty_cache()
    return ms, algbw, busbw


def config_valid(M: int, N: int, subtile_m: int, subtile_n: int, supergroup: int) -> bool:
    """Mirror the kernel-side TORCH_CHECKs so the sweep skips instead of dying.

    The tile walk has no tail handling, and calculate_tile_idx is only a
    bijection when the column count fills whole supergroups.
    """
    if M % subtile_m or N % subtile_n:
        return False
    return (N // subtile_n) % supergroup == 0


def main() -> int:
    p = argparse.ArgumentParser(description="NVLink bandwidth sweep for the gemm_ar comm loop")
    p.add_argument("--mode", choices=["check", "bench", "both"], default="both",
                   help="check: correctness only; bench: timing only; both: check the "
                        "smallest shape, then time everything")
    p.add_argument("--shapes", type=str, default=",".join(map(str, DEFAULT_SHAPES)),
                   help="square problem sizes (M = N)")
    p.add_argument("--subtile-m", type=str, default=",".join(map(str, DEFAULT_SUBTILE_M)))
    p.add_argument("--subtile-n", type=str, default=",".join(map(str, DEFAULT_SUBTILE_N)))
    p.add_argument("--comm-sms", type=str, default=",".join(map(str, DEFAULT_COMM_SMS)))
    p.add_argument("--ar-unroll", type=str, default=",".join(map(str, DEFAULT_AR_UNROLL)))
    p.add_argument("--supergroup-width", type=str, default=",".join(map(str, DEFAULT_SUPERGROUP)))
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--repeats", type=int, default=0,
                   help="in-kernel replays of the tile walk; 0 = auto from --target-mb")
    p.add_argument("--target-mb", type=int, default=512,
                   help="auto-repeat target: matrix bytes touched per launch")
    p.add_argument("--top", type=int, default=10, help="best configs to summarise per shape")
    p.add_argument("--no-baseline", dest="baseline", action="store_false",
                   help="skip the NCCL all-reduce ceiling measured per shape")
    p.add_argument("--seed", type=int, default=42,
                   help="base seed for the randn fill used by the timing runs; "
                        "each rank uses seed + rank")
    p.add_argument("--save-csv", type=str, default=None)
    args = p.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world = int(os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"]))
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", device_id=torch.device(f"cuda:{local_rank}"))
    is_chief = rank == 0

    mod = load_module.load("gemm_ar_blackwell")

    shapes = parse_int_list(args.shapes)
    grid = list(itertools.product(
        parse_int_list(args.subtile_m),
        parse_int_list(args.subtile_n),
        parse_int_list(args.comm_sms),
        parse_int_list(args.ar_unroll),
        parse_int_list(args.supergroup_width),
    ))

    period = pattern_period(world)
    sum_scale = world * (world + 1) // 2

    if is_chief:
        print(f"[mnvl-sweep] world={world} shapes={shapes} "
              f"configs/shape={len(grid)} pattern_period={period}", flush=True)

    rows: list[dict] = []
    ceiling_ar: dict[int, float] = {}
    failures = 0

    for n in shapes:
        M = N = n
        repeats = args.repeats if args.repeats > 0 else auto_repeats(M, N, args.target_mb)

        C_dbuf = mod.DistBuffer((M, N), dtype=torch.bfloat16, local_rank=local_rank,
                                local_world_size=world, multicast=True)
        C_final = mod.DistBuffer((M, N), dtype=torch.bfloat16, local_rank=local_rank,
                                 local_world_size=world, multicast=True)
        C_final.data_.zero_()
        sync_ranks()

        # Correctness is checked on the first shape only: it validates the tile
        # walk, which does not depend on the problem size, and a bit-equality
        # compare over a 32768^2 buffer is not worth paying for per config.
        do_check = args.mode == "check" or (args.mode == "both" and n == shapes[0])
        do_bench = args.mode in ("bench", "both")

        configs = [c for c in grid if config_valid(M, N, c[0], c[1], c[4])]

        if is_chief:
            print(f"\n=== M={M} N={N} repeats={repeats} configs={len(configs)} "
                  f"check={'yes' if do_check else 'no'} ===", flush=True)

        def run_config(cfg, reps):
            subtile_m, subtile_n, comm_sms, ar_unroll, supergroup = cfg
            mod.test_nvswitch_bw(C_dbuf, C_final,
                                 subtile_m=subtile_m, subtile_n=subtile_n,
                                 num_comm_sm=comm_sms, ar_unroll=ar_unroll,
                                 supergroup_width=supergroup, num_repeats=reps)

        # ---- phase 1: correctness, on the exact integer fill pattern ----
        if do_check:
            shape_failures = 0
            fill_pattern(C_dbuf.data_, period, rank + 1)
            sync_ranks()
            for cfg in configs:
                # Zero first so a tile the walk never visits shows up as 0
                # rather than as a stale (correct) value from a prior config.
                C_final.data_.zero_()
                sync_ranks()
                run_config(cfg, 1)  # replays add nothing — the walk is idempotent
                sync_ranks()
                bad = count_mismatch(C_final.data_, period, sum_scale)
                flag = torch.tensor([bad], dtype=torch.int64, device="cuda")
                dist.all_reduce(flag, op=dist.ReduceOp.MAX)
                if int(flag.item()) != 0:
                    failures += 1
                    shape_failures += 1
                    if is_chief:
                        print(f"  [FAIL] sub_m={cfg[0]} sub_n={cfg[1]} sms={cfg[2]} "
                              f"unroll={cfg[3]} sg={cfg[4]}: "
                              f"{int(flag.item())} mismatched elements", flush=True)
            if is_chief and not shape_failures:
                print(f"  [check] all {len(configs)} configs bit-exact", flush=True)

        # ---- phase 2: timing, on randn ----
        if do_bench:
            # The real C_dist holds GEMM output, so time the all-reduce on
            # float data rather than on the correctness pattern's small exact
            # integers. C_dist is never mutated by the kernel, so this fill
            # survives the whole sweep.
            fill_random(C_dbuf.data_, seed=args.seed + rank)
            C_final.data_.zero_()
            sync_ranks()

            if args.baseline:
                nccl_ms, nccl_ar, nccl_bus = nccl_baseline(
                    M, N, world, repeats, args.warmup, args.iters, args.seed + rank)
                rows.append(dict(kind="nccl", M=M, N=N, subtile_m=0, subtile_n=0,
                                 num_comm_sm=0, ar_unroll=0, supergroup_width=0,
                                 repeats=repeats, ms=round(nccl_ms, 5),
                                 ar_gbps=round(nccl_ar, 2), nvl_gbps=round(nccl_bus, 2)))
                ceiling_ar[M] = nccl_ar
                if is_chief:
                    print(f"  [ceiling] nccl all_reduce: {nccl_ms:.4f} ms  "
                          f"{nccl_ar:.1f} GB/s alg  {nccl_bus:.1f} GB/s bus", flush=True)
                    if repeats > 1:
                        # The sweep kernel replays the walk inside ONE launch;
                        # NCCL cannot, so it pays a launch and a rendezvous per
                        # repeat while the sweep pays one for all of them.
                        print(f"  [ceiling] NOTE repeats={repeats}: the sweep amortises "
                              f"launch + rendezvous over all {repeats} passes and NCCL "
                              f"does not, so '% of nccl' flatters the sweep here. "
                              f"Use --repeats 1, or read it off the large shapes.",
                              flush=True)
                dist.barrier()

            if is_chief:
                print(f"{'sub_m':>6} {'sub_n':>6} {'sms':>4} {'unroll':>7} {'sg':>3} "
                      f"{'ms':>9} {'ar GB/s':>10} {'nvl GB/s':>10}", flush=True)

            for cfg in configs:
                subtile_m, subtile_n, comm_sms, ar_unroll, supergroup = cfg

                for _ in range(args.warmup):
                    run_config(cfg, repeats)
                sync_ranks()

                events = []
                for _ in range(args.iters):
                    sync_ranks()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    run_config(cfg, repeats)
                    e.record()
                    events.append((s, e))
                torch.cuda.synchronize()

                ms = median_then_max([s.elapsed_time(e) for s, e in events])
                ar_gbps, nvl_gbps = bandwidth_gbps(M, N, world, repeats, ms)

                rows.append(dict(kind="mnvl", M=M, N=N,
                                 subtile_m=subtile_m, subtile_n=subtile_n,
                                 num_comm_sm=comm_sms, ar_unroll=ar_unroll,
                                 supergroup_width=supergroup, repeats=repeats,
                                 ms=round(ms, 5), ar_gbps=round(ar_gbps, 2),
                                 nvl_gbps=round(nvl_gbps, 2)))
                if is_chief:
                    print(f"{subtile_m:>6} {subtile_n:>6} {comm_sms:>4} {ar_unroll:>7} "
                          f"{supergroup:>3} {ms:>9.4f} {ar_gbps:>10.1f} {nvl_gbps:>10.1f}",
                          flush=True)

            if is_chief:
                shape_rows = sorted((r for r in rows if r["M"] == M and r["kind"] == "mnvl"),
                                    key=lambda r: -r["nvl_gbps"])[:args.top]
                ceil_v = ceiling_ar.get(M)
                print(f"  -- top {len(shape_rows)} by per-GPU NVLink GB/s --", flush=True)
                for r in shape_rows:
                    frac = f"  ({r['ar_gbps'] / ceil_v * 100:5.1f}% of nccl)" if ceil_v else ""
                    print(f"     {r['nvl_gbps']:>8.1f} GB/s nvl  {r['ar_gbps']:>8.1f} GB/s ar"
                          f"{frac}  sub={r['subtile_m']}x{r['subtile_n']} "
                          f"sms={r['num_comm_sm']} unroll={r['ar_unroll']} "
                          f"sg={r['supergroup_width']}", flush=True)

        # Free both multicast buffers before the next shape allocates its own —
        # at 32768 they are 2 GiB each. Every rank has to be past its last
        # kernel before the IPC mappings go away, hence the barrier first.
        sync_ranks()
        del C_dbuf, C_final
        torch.cuda.empty_cache()
        dist.barrier()

    if args.save_csv and is_chief and rows:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n[mnvl-sweep] wrote {len(rows)} rows to {out}", flush=True)

    if is_chief and failures:
        print(f"\n[mnvl-sweep] {failures} config(s) FAILED the correctness check", flush=True)

    dist.destroy_process_group()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
