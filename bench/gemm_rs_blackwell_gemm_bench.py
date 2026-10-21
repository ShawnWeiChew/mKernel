"""Single-GPU BF16 GEMM: gemm_rs_blackwell (RS disabled) versus cuBLAS.

    make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell
    python bench/gemm_rs_blackwell_gemm_bench.py --device 0
    python bench/gemm_rs_blackwell_gemm_bench.py --device 1 --emulated-world-size 2

No torchrun or process group. Works on either GPU of a 2-GPU instance, and does
not require the visible GPU count to match INTRA_NUM_DEVICES. K is the LOCAL
GEMM dimension. By default K=12288/8=1536; use --emulated-world-size 2 for
K=6144, or --k to specify it directly. The emulated size only selects K.
Requires the RS body/signals to remain disabled in src/gemm_rs_blackwell.cu.
Reports warm-cache CUDA graph timings, excluding allocation and the exit barrier.
Compare against a separate full GEMM+RS run at matching local dimensions to
assess GEMM's contribution; overlap means the times are not additive.
"""

import argparse
from pathlib import Path
import statistics
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import load_module


def capture(launch, graph_size):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_size):
            launch()
    return graph


def time_graph(graph, args):
    for _ in range(args.warmup):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / (args.iters * args.graph_size)


def benchmark_shape(module, m, n, k, args):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    b_transposed = b.T
    reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    output = torch.full_like(reference, float("nan"))

    def cublas():
        torch.mm(a, b_transposed, out=reference)

    def mkernel():
        module.gemm_blackwell_launch(a, b, output)

    cublas()
    mkernel()
    torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)
    graphs = {}
    try:
        graphs["cuBLAS"] = capture(cublas, args.graph_size)
        graphs["mKernel"] = capture(mkernel, args.graph_size)
        # Changed inputs and poisoned output catch stale or partially written C.
        for _ in range(2):
            a.neg_()
            output.fill_(float("nan"))
            graphs["cuBLAS"].replay()
            graphs["mKernel"].replay()
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

        samples = {name: [] for name in graphs}
        for repeat in range(args.repeats):
            order = ("cuBLAS", "mKernel") if repeat % 2 == 0 else ("mKernel", "cuBLAS")
            for name in order:
                samples[name].append(time_graph(graphs[name], args))
        cu = statistics.median(samples["cuBLAS"])
        mk = statistics.median(samples["mKernel"])
        flops = 2 * m * n * k
        print(
            f"M={m:6d} N={n} K={k}  "
            f"cuBLAS={cu * 1000:9.2f} us ({flops / (cu * 1e9):7.2f} TF/s)  "
            f"mKernel={mk * 1000:9.2f} us ({flops / (mk * 1e9):7.2f} TF/s)  "
            f"mKernel={100 * cu / mk:.1f}% of cuBLAS throughput",
            flush=True,
        )
    finally:
        torch.cuda.synchronize()
        graphs.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="2048,4096,8192,16384,32768", help="Comma-separated M values")
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--k", type=int, help="Local GEMM K; default: 12288 / --emulated-world-size")
    parser.add_argument("--emulated-world-size", type=int, default=8,
                        help="Select default local K; does not launch multiple processes")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=5, help="Warmup graph replays per sample")
    parser.add_argument("--iters", type=int, default=20, help="Timed graph replays per sample")
    parser.add_argument("--graph-size", type=int, default=20, help="GEMMs per graph replay")
    parser.add_argument("--repeats", type=int, default=5, help="Number of samples for median timing")
    args = parser.parse_args()
    try:
        shapes = [int(value) for value in args.shapes.split(",")]
    except ValueError:
        parser.error("--shapes must be comma-separated integers")
    if args.emulated_world_size <= 0 or 12288 % args.emulated_world_size:
        parser.error("--emulated-world-size must be a positive divisor of 12288")
    k = 12288 // args.emulated_world_size if args.k is None else args.k
    if any(m <= 0 or m % 512 for m in shapes):
        parser.error("M must be a positive multiple of 512")
    if args.n <= 0 or args.n % 256 or k <= 0 or k % 64:
        parser.error("N must be a positive multiple of 256; K must be a positive multiple of 64")
    if args.warmup < 0 or min(args.iters, args.graph_size, args.repeats) < 1:
        parser.error("--warmup must be nonnegative; --iters, --graph-size, --repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device must select a visible CUDA device")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This benchmark requires a Blackwell GPU")
    module = load_module.load("gemm_rs_blackwell")
    if not hasattr(module, "gemm_blackwell_launch"):
        parser.error("Rebuild with: make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell")
    # https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_blas_library
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    print(
        f"{torch.cuda.get_device_name()}; device {args.device}; BF16, FP32 accumulation\n"
        f"GEMM only; {args.graph_size} GEMMs/graph; median of {args.repeats} samples.\n"
        "mKernel throughput: 100% matches cuBLAS; higher is faster. Correctness checked before timing.",
        flush=True,
    )
    for m in shapes:
        benchmark_shape(module, m, args.n, k, args)


if __name__ == "__main__":
    main()
