"""Compare the GEMM-only gemm_rs_blackwell kernel with cuBLAS (BF16 A @ B.T) on ONE GPU.

Requires the reduce-scatter body/signals in src/gemm_rs_blackwell.cu to be
disabled, as in the current experiment. The exit barrier and PDL launch overlap
are disabled by this script.

    make GPU=blackwell INTRA_NUM_DEVICES=1 gemm-rs-blackwell
    python bench/gemm_rs_blackwell_gemm_bench_single.py

Use --shapes 2048,8192 --n 7168 --k 1536 to select M, N, and K. Since there is
only one rank, K defaults to the PER-RANK K of the multi-GPU benchmark:
12288 / --emulated-world-size (default 8 -> K=1536).
Reports warm-cache CUDA graph timings, excluding allocation.
"""

import argparse
import os
import statistics

# Single-process "distributed" environment; must be set before the config is built.
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")
os.environ.setdefault("LOCAL_WORLD_SIZE", "1")
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29511")

import torch
import torch.distributed as dist

from gemm_rs_blackwell_bench import (
    BlackwellBenchConfig,
    benchmark_cuda,
    capture_cuda_graph,
    load_module,
    make_blackwell_buffers,
    useful_tflops,
)


def check_output(actual, expected, label):
    try:
        torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
    except AssertionError as exc:
        raise RuntimeError(
            f"{label}: GEMM correctness failed. "
            "This benchmark requires RS disabled and checks the FULL local C.\n"
            f"{exc}"
        ) from exc


def benchmark_shape(config, module, m, n, k, args):
    buffers = make_blackwell_buffers(config, m, n, k, module)
    b_transposed = buffers.baseline_b.T

    def cublas():
        torch.mm(buffers.baseline_a, b_transposed, out=buffers.baseline_c)

    def mkernel():
        module.gemm_rs_blackwell_launch(
            buffers.baseline_a, buffers.baseline_b, buffers.c_mkernel,
            buffers.ready_mkernel, buffers.epochs_mkernel,
            with_exit_barrier=False,
        )

    graphs = {}
    try:
        buffers.c_mkernel.data_.fill_(float("nan"))
        cublas()
        mkernel()
        torch.cuda.synchronize()
        check_output(buffers.c_mkernel.data_, buffers.baseline_c, "eager")

        for name, launch in (("cuBLAS", cublas), ("mKernel", mkernel)):
            def batch(launch=launch):
                for _ in range(args.graph_size):
                    launch()

            graphs[name] = capture_cuda_graph(batch)

        # Changing inputs and poisoning C catches stale or partially written output.
        for replay in range(2):
            buffers.baseline_a.neg_()
            buffers.c_mkernel.data_.fill_(float("nan"))
            graphs["cuBLAS"].replay()
            graphs["mKernel"].replay()
            torch.cuda.synchronize()
            check_output(
                buffers.c_mkernel.data_, buffers.baseline_c, f"graph replay {replay + 1}",
            )

        samples = {name: [] for name in graphs}
        for repeat in range(args.repeats):
            # Alternate order to reduce systematic clock/thermal bias.
            order = ("cuBLAS", "mKernel") if repeat % 2 == 0 else ("mKernel", "cuBLAS")
            for name in order:
                ms = benchmark_cuda(graphs[name].replay, args.warmup, args.iters)
                samples[name].append(ms / args.graph_size)
        timings = {name: statistics.median(values) for name, values in samples.items()}
        cu, mk = timings["cuBLAS"], timings["mKernel"]
        print(
            f"M={m:6d} N={n} K={k}  "
            f"cuBLAS={cu * 1000:9.2f} us ({useful_tflops(m, n, k, cu):7.2f} TF/s)  "
            f"mKernel={mk * 1000:9.2f} us ({useful_tflops(m, n, k, mk):7.2f} TF/s)  "
            f"mKernel/cuBLAS={mk / cu:.3f}x time",
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        graphs.clear()
        dist.barrier()  # trivial with one rank; keeps buffer teardown ordering


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="2048,4096,8192,16384,32768", help="Comma-separated M values")
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, help="K; default: 12288 / --emulated-world-size")
    parser.add_argument("--emulated-world-size", type=int, default=8,
                        help="Multi-GPU world size whose per-rank K to emulate (only used for default K)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup graph replays per sample")
    parser.add_argument("--iters", type=int, default=20, help="Timed graph replays per sample")
    parser.add_argument("--graph-size", type=int, default=20, help="GEMMs per graph replay")
    parser.add_argument("--repeats", type=int, default=5, help="Report median of this many samples")
    args = parser.parse_args()

    try:
        shapes = [int(value) for value in args.shapes.split(",")]
    except ValueError:
        parser.error("--shapes must be comma-separated integers")
    k = 12288 // args.emulated_world_size if args.k is None else args.k
    if any(m <= 0 or m % 512 or m % 256 for m in shapes):
        parser.error("M must be positive and divisible by 512")
    if args.n <= 0 or args.n % 256 or k <= 0 or k % 64:
        parser.error("N must be a positive multiple of 256; K must be a positive multiple of 64")
    if args.warmup < 0 or min(args.iters, args.graph_size, args.repeats) < 1:
        parser.error("--warmup must be nonnegative; --iters, --graph-size, --repeats must be positive")
    if int(os.environ["WORLD_SIZE"]) != 1:
        parser.error("This script is single-GPU; use the multi-GPU benchmark with torchrun instead")

    config = BlackwellBenchConfig()
    torch.cuda.set_device(config.local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This benchmark requires Blackwell GPUs")
    module = load_module.load("gemm_rs_blackwell")
    if module.gemm_rs_blackwell_world_size != 1:
        parser.error("Rebuild with INTRA_NUM_DEVICES=1")
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    dist.init_process_group("nccl", device_id=torch.device("cuda", config.local_rank))
    try:
        print(
            f"{torch.cuda.get_device_name()}; single GPU; BF16, FP32 accumulation\n"
            f"GEMM only, no exit barrier; {args.graph_size} GEMMs/graph, "
            f"median of {args.repeats} samples.\n"
            "mKernel/cuBLAS > 1 means mKernel is slower.",
            flush=True,
        )
        for m in shapes:
            benchmark_shape(config, module, m, args.n, k, args)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()