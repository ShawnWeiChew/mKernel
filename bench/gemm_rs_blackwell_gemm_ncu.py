"""Run the single-GPU gemm_rs_blackwell GEMM (RS disabled) once, for Nsight Compute.

    make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell
    ncu --profile-from-start off --set full -o results/gemm-ncu/gemm_m8192 -f \
        python bench/gemm_rs_blackwell_gemm_ncu.py

Same GEMM as gemm_rs_blackwell_gemm_bench.py: C[M,N] = A[M,K] @ B[N,K].T, BF16 in
and out, default M=8192 N=7168 K=12288/8=1536. The kernel is launched a few times
to warm up, checked against cuBLAS, and then launched --profile-launches more times
between cudaProfilerStart/Stop, so `--profile-from-start off` captures only those
launches (no cuBLAS kernels, no warmup). Without ncu it simply runs and exits.
Requires the RS body/signals to remain disabled in src/gemm_rs_blackwell.cu.
"""

import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import load_module


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--m", type=int, default=8192)
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, default=12288 // 8, help="Local GEMM K (default 12288/8)")
    parser.add_argument("--consumer-warps", type=int, choices=(1, 2), default=2,
                        help="Each consumer handles 128 rows per CTA (cluster tile 256 or 512 rows)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=10, help="Unprofiled launches before profiling")
    parser.add_argument("--profile-launches", type=int, default=1, help="Launches inside the profiled range")
    args = parser.parse_args()
    m, n, k = args.m, args.n, args.k
    if m <= 0 or m % (256 * args.consumer_warps):
        parser.error(f"M must be a positive multiple of {256 * args.consumer_warps}")
    if n <= 0 or n % 256 or k <= 0 or k % 64:
        parser.error("N must be a positive multiple of 256; K must be a positive multiple of 64")
    if args.warmup < 0 or args.profile_launches < 1:
        parser.error("--warmup must be nonnegative; --profile-launches must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device must select a visible CUDA device")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This script requires a Blackwell GPU")
    module = load_module.load("gemm_rs_blackwell")
    if not hasattr(module, "gemm_blackwell_consumer_variants"):
        parser.error("Rebuild with: make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell")
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)

    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    output = torch.full((m, n), float("nan"), device="cuda", dtype=torch.bfloat16)

    def launch():
        module.gemm_blackwell_launch(a, b, output, num_consumer_warps=args.consumer_warps)

    launch()
    torch.testing.assert_close(output, a @ b.T, rtol=2e-2, atol=2e-2)
    for _ in range(args.warmup):
        launch()
    torch.cuda.synchronize()

    torch.cuda.cudart().cudaProfilerStart()
    for _ in range(args.profile_launches):
        launch()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()

    print(f"{torch.cuda.get_device_name()}; M={m} N={n} K={k} consumers={args.consumer_warps}: "
          f"matched cuBLAS; {args.profile_launches} profiled launch(es) done", flush=True)


if __name__ == "__main__":
    main()
