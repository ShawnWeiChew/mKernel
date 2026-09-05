import argparse
import os
import sys
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
import timings_dump  # noqa: E402
from common import check_close  # noqa: E402


GLOBAL_M = [2048, 4096, 8192, 16384, 32768]
K = 7168

# Eight-way tensor parallel KDA projection width before kernel padding:
#   (4 * 12288 + 96) / 8 + 128 = 6284.
LOGICAL_N = 6284

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20

# In-kernel timing profile (build with `make PROFILE=1 ag-gemm-kda-mla`).
PROFILE_MODULE = "ag_gemm_kda_mla_profile"
DEFAULT_PROFILE_M = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness and performance test for ag_gemm_kda_mla"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="warmup iterations per implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="timed iterations per implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help=(
            "capture an in-kernel timing trace instead of running the "
            "correctness/benchmark sweep (needs `make PROFILE=1 "
            "ag-gemm-kda-mla`)"
        ),
    )
    parser.add_argument(
        "--profile-m",
        type=int,
        default=DEFAULT_PROFILE_M,
        help="global M to profile (default: %(default)s)",
    )
    parser.add_argument(
        "--profile-rank",
        type=int,
        default=0,
        help=(
            "which rank dumps its trace; %%globaltimer is not synchronized "
            "across GPUs, so each rank's file is its own timeline "
            "(default: %(default)s)"
        ),
    )
    parser.add_argument(
        "--profile-out",
        type=str,
        default=None,
        help="output .npz path (default: plots/ag_gemm_kda_mla_trace_rank<N>.npz)",
    )
    parser.add_argument(
        "--no-render",
        action="store_true",
        help="dump the .npz but skip rendering the PDF",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    if args.profile and args.profile_m % 8 != 0:
        parser.error("--profile-m must be divisible by the world size (8)")
    return args


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def logical_n_for(world_size: int) -> int:
    """Projection width before kernel padding, for the ranks we support.

    Shared by the correctness sweep and the profile run so a trace can never
    record a different N than the one it actually ran.
    """
    if world_size == 8:
        return LOGICAL_N
    if world_size == 4:
        return (4 * 12288 + 96) // 4 + 128
    raise RuntimeError(
        f"logical projection width is only defined for 4 or 8 ranks; "
        f"got {world_size}"
    )


def padded_n_for_m(m: int, logical_n: int) -> int:
    col_block = 128 if m < 4096 else 256
    return round_up(logical_n, col_block)


def benchmark_cuda(
    run_once: Callable[[], None], warmup: int, iters: int
) -> float:
    """Return average CUDA time in ms, taking the slowest rank's result."""
    for _ in range(warmup):
        run_once()

    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / iters

    # End-to-end distributed latency is gated by the slowest rank. This
    # reduction is outside the timed region for both implementations.
    rank_ms = torch.tensor(local_ms, device="cuda", dtype=torch.float64)
    dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
    dist.barrier()
    return float(rank_ms.item())


def render_trace(npz_path: str) -> None:
    """Render the freshly dumped trace to a PDF beside it.

    Import-time failure is not fatal: the .npz is the artifact, and it can
    always be rendered later on a machine that has matplotlib.
    """
    try:
        sys.path.insert(0, str(HERE.parent / "plots"))
        import render_timings
    except ImportError as exc:  # pragma: no cover - matplotlib not installed
        print(
            f"  (skipping PDF: {exc}. Render later with "
            f"`python3 plots/render_timings.py {npz_path}`)",
            flush=True,
        )
        return
    render_timings.render_trace(npz_path)


def run_profile(args: argparse.Namespace, rank: int, local_rank: int,
                local_world_size: int) -> int:
    """Capture one instrumented iteration and dump it as a .npz trace.

    Exactly one iteration: the per-CTA head restarts at 0 each launch, so a
    second iteration would overwrite the first and collide on the
    (block, payload) pairing key.
    """
    mod = load_module.load(PROFILE_MODULE)
    if not hasattr(mod, "EVENTS_PER_BLOCK"):
        raise RuntimeError(
            f"{PROFILE_MODULE} was built without -DPROFILE_TIMINGS. "
            "Run `make PROFILE=1 ag-gemm-kda-mla`."
        )

    m = args.profile_m
    local_m = m // local_world_size
    logical_n = logical_n_for(local_world_size)
    padded_n = padded_n_for_m(m, logical_n)
    device = torch.device(f"cuda:{local_rank}")
    is_dumper = rank == args.profile_rank

    torch.manual_seed(42 + rank)
    torch.cuda.manual_seed(42 + rank)
    A_local = torch.randn(
        (local_m, K), device=device, dtype=torch.bfloat16
    ) / (K**0.25)

    A_kernel = mod.DistBuffer(
        (local_m, K),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=local_world_size,
        multicast=True,
    )
    A_kernel.data_.copy_(A_local)
    B_kernel = torch.zeros((K, padded_n), device=device, dtype=torch.bfloat16)
    B_kernel[:, :logical_n].normal_(0.0, K**-0.25)
    C_kernel = torch.zeros((m, padded_n), device=device, dtype=torch.bfloat16)

    # Warm up with a null ring: EMIT short-circuits on a null buffer, so these
    # launches leave the trace untouched while still warming caches and clocks.
    for _ in range(args.warmup):
        mod.ag_gemm_kda_mla(A_kernel, B_kernel, C_kernel, timings_ptr=0)
    torch.cuda.synchronize()
    dist.barrier()

    num_blocks = mod.TIMING_NUM_BLOCKS
    events_per_block = mod.EVENTS_PER_BLOCK
    ring = timings_dump.allocate_ring(num_blocks, events_per_block, device)

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    mod.ag_gemm_kda_mla(
        A_kernel, B_kernel, C_kernel, timings_ptr=ring.data_ptr()
    )
    end.record()
    end.synchronize()
    kernel_ms = start.elapsed_time(end)

    if not is_dumper:
        dist.barrier()
        return 0

    records, heads = timings_dump.unpack_ring(
        ring, num_blocks, events_per_block
    )
    overflowed = int((heads >= events_per_block).sum())

    out_path = args.profile_out or str(
        HERE.parent / "plots" / f"ag_gemm_kda_mla_trace_rank{rank}.npz"
    )
    timings_dump.save_trace(
        out_path,
        records,
        heads,
        dict(mod.TIMING_EVENTS),
        dict(mod.TIMING_ROLES),
        events_per_block,
        num_blocks=num_blocks,
        fine=bool(mod.TIMING_FINE),
        rank=rank,
        world_size=local_world_size,
        problem_m=m,
        local_m=local_m,
        problem_n=logical_n,
        padded_n=padded_n,
        problem_k=K,
        kernel_ms=kernel_ms,
    )
    print(
        f"wrote {out_path}\n"
        f"  M={m} local_m={local_m} padded_n={padded_n} rank={rank}\n"
        f"  {records.shape[0]} events from {int((heads > 0).sum())} CTAs, "
        f"kernel {kernel_ms:.3f} ms",
        flush=True,
    )
    if overflowed:
        print(
            f"  WARNING: {overflowed} CTA(s) hit the {events_per_block}-event "
            f"cap; the tail of their timeline was dropped. Rebuild with a "
            f"larger PROFILE_EVENTS.",
            flush=True,
        )
    if not bool(mod.TIMING_FINE):
        print(
            "  WARNING: this .so was built PROFILE_COARSE=1, so the per-K-step "
            "spans (mma: wait tma, prod: wait stage) are absent. Tile-level "
            "spans alone run back to back and render as a solid band -- "
            "rebuild with plain `make PROFILE=1` to see the waiting.",
            flush=True,
        )

    if not args.no_render:
        render_trace(out_path)

    dist.barrier()
    return 0


def main() -> int:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    world_size = dist.get_world_size()
    is_chief = rank == 0

    # Local name on purpose: the 4-rank layout differs from the module-level
    # 8-rank default. logical_n_for() is the single source of truth so the
    # profile path cannot disagree with the correctness sweep.
    logical_n = logical_n_for(world_size)

    if local_world_size != world_size:
        raise RuntimeError(
            "ag_gemm_kda_mla is an intra-node test and requires "
            "LOCAL_WORLD_SIZE == WORLD_SIZE"
        )

    if args.profile:
        status = run_profile(args, rank, local_rank, local_world_size)
        dist.destroy_process_group()
        return status

    mod = load_module.load("ag_gemm_kda_mla")
    all_correct = True

    for m in GLOBAL_M:
        if m % world_size != 0:
            raise ValueError(f"global M={m} is not divisible by {world_size=}")

        local_m = m // world_size
        padded_n = padded_n_for_m(m, logical_n)

        # Reference tensors retain the original, unpadded problem shapes.
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, logical_n), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, logical_n), device="cuda", dtype=torch.bfloat16
        )

        dist.all_gather_into_tensor(A_ref, A_ref_local)
        torch.mm(A_ref, B_ref, out=C_ref)

        # The modified implementation gets its own tensors. A already meets
        # the kernel's K/row requirements, so only B and C need N padding.
        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)

        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :logical_n].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        dist.barrier()
        mod.ag_gemm_kda_mla(A_kernel, B_kernel, C_kernel)
        torch.cuda.synchronize()

        # Ignore the padded output columns and compare the original 6284-wide
        # result against the unpadded PyTorch reference.
        is_correct = check_close(
            f"ag-gemm-kda-mla M={m} N={logical_n} padded_n={padded_n}",
            C_kernel[:, :logical_n],
            C_ref,
        )
        all_correct = all_correct and is_correct

        if is_chief:
            status = "passed :)" if is_correct else "FAILED :("
            print(
                f"M={m} local_m={local_m} N={logical_n} "
                f"padded_n={padded_n}: {status}",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, B_kernel, C_kernel
        dist.barrier()

    if not all_correct:
        if is_chief:
            print("Correctness checks failed; skipping benchmarks.", flush=True)
        dist.destroy_process_group()
        return 1

    if is_chief:
        print(
            f"All correctness checks passed. Benchmarking with "
            f"warmup={args.warmup}, iters={args.iters}...",
            flush=True,
        )

    # Allocate fresh tensors for the benchmark pass so no performance result
    # is emitted until the complete correctness suite has passed.
    for m in GLOBAL_M:
        local_m = m // world_size
        padded_n = padded_n_for_m(m, logical_n)

        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, logical_n), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, logical_n), device="cuda", dtype=torch.bfloat16
        )

        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)
        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :logical_n].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        def run_baseline() -> None:
            # NCCL all-gather followed by a cuBLAS GEMM. Reusing C_ref keeps
            # output allocation outside the timed region.
            dist.all_gather_into_tensor(A_ref, A_ref_local)
            torch.mm(A_ref, B_ref, out=C_ref)

        def run_kernel() -> None:
            mod.ag_gemm_kda_mla(A_kernel, B_kernel, C_kernel)

        baseline_ms = benchmark_cuda(run_baseline, args.warmup, args.iters)
        kernel_ms = benchmark_cuda(run_kernel, args.warmup, args.iters)
        relative_performance = baseline_ms / kernel_ms

        if is_chief:
            print(
                f"M={m} local_m={local_m} N={logical_n} "
                f"padded_n={padded_n}\n"
                f"  {'cuBLAS + NCCL':<17} {baseline_ms:8.3f} ms  "
                f"(1.000x, 100.0%)\n"
                f"  {'ag_gemm_kda_mla':<17} {kernel_ms:8.3f} ms  "
                f"({relative_performance:6.3f}x, "
                f"{relative_performance * 100:6.1f}% of baseline)",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, B_kernel, C_kernel
        dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
