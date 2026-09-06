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
from common import check_close  # noqa: E402


GLOBAL_M = [2048, 4096, 8192, 16384, 32768]
K = 7168

# Eight-way tensor parallel KDA projection width before kernel padding:
#   (4 * 12288 + 96) / 8 + 128 = 6284.
LOGICAL_N = 6284

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20


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
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


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


def main() -> int:
    # The four-rank configuration selects a different projection width.
    global LOGICAL_N
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

    if world_size == 4:
        LOGICAL_N = (4 * 12288 + 96) // 4 + 128
    elif world_size != 8:
        raise RuntimeError(
            f"This correctness test fixes the logical projection width at "
            f"{LOGICAL_N}, which assumes 8 ranks; got {world_size}."
        )

    if local_world_size != world_size:
        raise RuntimeError(
            "ag_gemm_kda_mla is an intra-node test and requires "
            "LOCAL_WORLD_SIZE == WORLD_SIZE"
        )

    mod = load_module.load("ag_gemm_kda_mla")
    all_correct = True

    for m in GLOBAL_M:
        if m % world_size != 0:
            raise ValueError(f"global M={m} is not divisible by {world_size=}")

        local_m = m // world_size
        padded_n = padded_n_for_m(m, LOGICAL_N)

        # Reference tensors retain the original, unpadded problem shapes.
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, LOGICAL_N), device="cuda", dtype=torch.bfloat16
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
        A_local_buf = torch.empty(
            (m, K), device="cuda", dtype=torch.bfloat16
        )

        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :LOGICAL_N].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        dist.barrier()
        mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B_kernel, C_kernel)
        torch.cuda.synchronize()

        # Ignore the padded output columns and compare the original 6284-wide
        # result against the unpadded PyTorch reference.
        is_correct = check_close(
            f"ag-gemm-kda-mla M={m} N={LOGICAL_N} padded_n={padded_n}",
            C_kernel[:, :LOGICAL_N],
            C_ref,
        )
        all_correct = all_correct and is_correct

        if is_chief:
            status = "passed :)" if is_correct else "FAILED :("
            print(
                f"M={m} local_m={local_m} N={LOGICAL_N} "
                f"padded_n={padded_n}: {status}",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, A_local_buf, B_kernel, C_kernel
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
        padded_n = padded_n_for_m(m, LOGICAL_N)

        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        )

        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)
        A_local_buf = torch.empty(
            (m, K), device="cuda", dtype=torch.bfloat16
        )
        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :LOGICAL_N].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        def run_all_gather() -> None:
            dist.all_gather_into_tensor(A_ref, A_ref_local)

        def run_cublas_logical() -> None:
            torch.mm(A_ref, B_ref, out=C_ref)

        def run_cublas_padded() -> None:
            torch.mm(A_ref, B_kernel, out=C_kernel)

        def run_baseline_logical() -> None:
            run_all_gather()
            run_cublas_logical()

        def run_baseline_padded() -> None:
            run_all_gather()
            run_cublas_padded()

        def run_kernel() -> None:
            mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B_kernel, C_kernel)

        all_gather_ms = benchmark_cuda(run_all_gather, args.warmup, args.iters)
        cublas_logical_ms = benchmark_cuda(
            run_cublas_logical, args.warmup, args.iters
        )
        cublas_padded_ms = benchmark_cuda(
            run_cublas_padded, args.warmup, args.iters
        )
        baseline_logical_ms = benchmark_cuda(
            run_baseline_logical, args.warmup, args.iters
        )
        baseline_padded_ms = benchmark_cuda(
            run_baseline_padded, args.warmup, args.iters
        )
        kernel_ms = benchmark_cuda(run_kernel, args.warmup, args.iters)
        relative_performance = baseline_padded_ms / kernel_ms
        logical_relative_performance = baseline_logical_ms / kernel_ms

        if is_chief:
            print(
                f"M={m} local_m={local_m} N={LOGICAL_N} "
                f"padded_n={padded_n}\n"
                f"  {'NCCL all-gather':<26} {all_gather_ms:8.3f} ms\n"
                f"  {f'cuBLAS N={LOGICAL_N}':<26} {cublas_logical_ms:8.3f} ms\n"
                f"  {f'cuBLAS N={padded_n}':<26} {cublas_padded_ms:8.3f} ms\n"
                f"  {f'cuBLAS + NCCL N={LOGICAL_N}':<26} "
                f"{baseline_logical_ms:8.3f} ms\n"
                f"  {f'cuBLAS + NCCL N={padded_n}':<26} "
                f"{baseline_padded_ms:8.3f} ms  (matched baseline)\n"
                f"  {'ag_gemm_kda_mla':<26} {kernel_ms:8.3f} ms  "
                f"({relative_performance:6.3f}x vs matched, "
                f"{logical_relative_performance:6.3f}x vs logical)",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, A_local_buf, B_kernel, C_kernel
        dist.barrier()

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
