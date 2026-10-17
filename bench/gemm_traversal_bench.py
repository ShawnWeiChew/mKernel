#!/usr/bin/env python3
"""Single-GPU BF16 A[M,K] @ B[N,K].T: sharded traversal vs global vs cuBLAS.

    python bench/gemm_traversal_bench.py --build --json build/gemm_traversal.json
    python bench/gemm_traversal_bench.py --validate-only  # CPU, no torch needed
    python bench/gemm_traversal_bench.py --build-only     # NVCC, no GPU needed

Default M=N=K: 2048, 4096, 8192, 16384, 32768. --n/--k fix those axes.
Both variants of the same kernel use two MMA consumers, 2-CTA clusters, 148 CTAs, four
input stages, and one epilogue warpgroup with a two-buffer staging ring.
Logical devices are row shards in local GPU memory; no torchrun or NCCL.
"""

import argparse
import ctypes
import json
import math
from pathlib import Path
import statistics
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SIZES = (2048, 4096, 8192, 16384, 32768)


def shape_config(args, m):
    n, k = args.n or m, args.k or m
    width = args.supergroup_width or (4 if m <= 4096 else 8)
    return m, n, k, width


def snake(rows, cols, tile, width):
    """The kernels' grouped-column snake, including AG's partial final group."""
    group = tile // (rows * width)
    full_cols = cols // width * width
    if tile < rows * full_cols:
        row = (tile % (rows * width)) // width
        col = group * width + tile % width
    else:
        row, col = divmod(tile - rows * full_cols, cols - full_cols)
        col += full_cols
    return (rows - row - 1 if group % 2 else row), col


def traversal_tiles(m, n, devices, width, rank, sharded):
    """Model the actual producer/epilogue row assignments, in 128x256 units."""
    local_rows = m // devices // 128
    cluster_rows = (local_rows + 3) // 4
    cols = n // 256
    per_device = cluster_rows * cols
    total = per_device * devices
    seen, padded = set(), 0
    for cluster in range(148 // 2):
        for tile in range(cluster, total, 148 // 2):
            if sharded:
                row, col = snake(cluster_rows, cols, tile % per_device, width)
                device = (tile // per_device + rank) % devices
            else:
                global_row, col = snake(cluster_rows * devices, cols, tile, width)
                device, row = divmod(global_row, cluster_rows)
            for consumer in range(2):
                for cta in range(2):
                    local_row = row * 4 + consumer * 2 + cta
                    if local_row >= local_rows:
                        padded += 1
                        continue  # TMA zero-fill loads and discarded stores at shard boundary.
                    output = (device * local_rows + local_row, col)
                    if output in seen or not (0 <= output[0] < m // 128 and 0 <= col < cols):
                        raise AssertionError(f"duplicate/out-of-range tile: {output}")
                    seen.add(output)
    if len(seen) != (m // 128) * cols:
        raise AssertionError(f"missing tiles: got {len(seen)}, expected {(m // 128) * cols}")
    return seen, padded


def validate_traversals(args):
    for m in args.shapes:
        m, n, k, width = shape_config(args, m)
        if m % args.logical_devices or (m // args.logical_devices) % 128:
            raise ValueError("M must divide into logical shards of a multiple of 128 rows")
        expected, expected_padding = traversal_tiles(m, n, args.logical_devices, width, 0, False)
        for rank in range(args.logical_devices):
            actual, padded = traversal_tiles(m, n, args.logical_devices, width, rank, True)
            if actual != expected or padded != expected_padding:
                raise AssertionError(f"traversals disagree for rank {rank}")
        print(f"CPU coverage M={m} N={n} K={k}: {len(expected)} tiles, all "
              f"{args.logical_devices} rank rotations; padded tiles per mode={padded}", flush=True)


class Kernels:
    def __init__(self, path, logical_devices):
        if not path.is_file():
            raise FileNotFoundError(f"{path} is missing; run with --build")
        self.lib = ctypes.CDLL(str(path))
        p, i = ctypes.c_void_p, ctypes.c_int
        signatures = {
            "gemm_bench_logical_devices": [],
            "gemm_bench_launch": [p, p, p, i, i, i, i, i, i, p],
            "gemm_bench_cublas_create": [ctypes.POINTER(p)],
            "gemm_bench_cublas_destroy": [p],
            "gemm_bench_cublas": [p, p, p, p, i, i, i, p],
        }
        for name, argtypes in signatures.items():
            function = getattr(self.lib, name)
            function.argtypes, function.restype = argtypes, i
        if self.lib.gemm_bench_logical_devices() != logical_devices:
            raise RuntimeError("Library logical-device count differs from --logical-devices")
        self.handle = p()
        self.call("gemm_bench_cublas_create", ctypes.byref(self.handle))

    def call(self, name, *args):
        status = getattr(self.lib, name)(*args)
        if status:
            raise RuntimeError(f"{name} failed with {'cuBLAS' if 'cublas' in name else 'CUDA'} "
                               f"status {status}")

    def close(self):
        self.call("gemm_bench_cublas_destroy", self.handle)


def check_output(torch, output, reference, label, args):
    # Chunking avoids allocating several full FP32 matrices at 32768^2.
    bad = torch.zeros((), device=output.device, dtype=torch.int64)
    max_error = torch.zeros((), device=output.device)
    squared_error = torch.zeros((), device=output.device, dtype=torch.float64)
    squared_reference = torch.zeros_like(squared_error)
    for row in range(0, output.shape[0], 128):
        actual = output[row:row + 128].float()
        expected = reference[row:row + 128].float()
        error = (actual - expected).abs()
        bad += (~torch.isclose(actual, expected, rtol=args.rtol, atol=args.atol)).sum()
        max_error = torch.maximum(max_error, error.max())
        squared_error += error.square().sum(dtype=torch.float64)
        squared_reference += expected.square().sum(dtype=torch.float64)
    mismatches = bad.item()
    metrics = {"max_abs_error": max_error.item(),
               "relative_rms": (squared_error / squared_reference.clamp_min(1e-30)).sqrt().item()}
    if mismatches:
        raise AssertionError(f"{label}: {mismatches}/{output.numel()} mismatches; {metrics}")
    return metrics


def benchmark_shape(torch, kernels, args, m):
    m, n, k, width = shape_config(args, m)
    stream = torch.cuda.current_stream()
    # Variance ~1 in C across K sizes, making the fixed BF16 tolerance meaningful.
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16).mul_(1 / math.sqrt(k))
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    reference = torch.empty_like(output)
    pointers = a.data_ptr(), b.data_ptr(), output.data_ptr()

    def ag(rank=args.logical_rank):
        kernels.call("gemm_bench_launch", *pointers, m, n, k, width, rank, 0, stream.cuda_stream)

    def global_gemm():
        kernels.call("gemm_bench_launch", *pointers, m, n, k, width, args.logical_rank, 1,
                     stream.cuda_stream)

    def cublas():
        kernels.call("gemm_bench_cublas", kernels.handle, *pointers, m, n, k, stream.cuda_stream)

    functions = {"device_Mlocal_N": ag, "Mglobal_N": global_gemm, "cuBLAS": cublas}
    kernels.call("gemm_bench_cublas", kernels.handle, a.data_ptr(), b.data_ptr(),
                 reference.data_ptr(), m, n, k, stream.cuda_stream)
    checks = {}
    for name, function in functions.items():
        output.fill_(float("nan"))  # Detect unwritten tiles as well as wrong values.
        function()
        checks[name] = check_output(torch, output, reference, name, args)
    # Also check a different first shard; no other physical GPU is involved.
    if args.logical_devices > 1:
        output.fill_(float("nan"))
        ag((args.logical_rank + 1) % args.logical_devices)
        check_output(torch, output, reference, "AG rotated rank", args)

    graphs = {}
    for name, function in functions.items():
        for _ in range(args.warmup):
            function()
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            for _ in range(args.inner_iters):
                function()
        graphs[name] = graph
        output.fill_(float("nan"))
        graph.replay()
        check_output(torch, output, reference, f"{name} graph replay", args)
    stream.synchronize()

    # Rotate and reverse order across samples to reduce clock/order bias.
    samples = {name: [] for name in functions}
    names = list(functions)
    for sample in range(args.samples):
        offset = sample % len(names)
        order = names[offset:] + names[:offset]
        if (sample // len(names)) % 2:
            order.reverse()
        for name in order:
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record(stream)
            graphs[name].replay()
            end.record(stream)
            end.synchronize()
            samples[name].append(start.elapsed_time(end) / args.inner_iters)

    times = {name: statistics.median(values) for name, values in samples.items()}
    # Report useful FLOPs; do not count the padded MMA work as useful throughput.
    padded_m = math.ceil((m // args.logical_devices) / 512) * 512 * args.logical_devices
    print(f"\nM={m} N={n} K={k}, supergroup={width}, "
          f"both traversal modes executed-row ratio={padded_m / m:.2f}x; correctness PASS", flush=True)
    results = []
    for name, ms in times.items():
        tflops = 2 * m * n * k / (ms * 1e9)
        speedup = times["cuBLAS"] / ms
        print(f"  {name:17s} {ms:9.4f} ms  {tflops:9.2f} TFLOP/s  {speedup:6.3f}x cuBLAS",
              flush=True)
        results.append(dict(kernel=name, m=m, n=n, k=k, supergroup_width=width,
                            median_ms=ms, samples_ms=samples[name], tflops=tflops,
                            speedup_vs_cublas=speedup,
                            executed_row_ratio=padded_m / m if name != "cuBLAS" else 1,
                            **checks[name]))
    # All captures and tensors are released on return, before the next shape.
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", type=int, nargs="+", choices=SIZES, default=list(SIZES))
    parser.add_argument("--n", type=int, choices=SIZES, help="fixed N; default N=M")
    parser.add_argument("--k", type=int, choices=SIZES, help="fixed K; default K=M")
    parser.add_argument("--logical-devices", type=int, choices=(1, 2, 4, 8, 16), default=8)
    parser.add_argument("--logical-rank", type=int, default=0)
    parser.add_argument("--supergroup-width", type=int, choices=(4, 8))
    parser.add_argument("--arch", choices=("sm_100a", "sm_103a"), default="sm_103a")
    parser.add_argument("--cuda-home", default="/usr/local/cuda-13.2")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--inner-iters", type=int, default=10, help="GEMMs per captured graph")
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--rtol", type=float, default=0.02)
    parser.add_argument("--atol", type=float, default=0.01)
    parser.add_argument("--json", type=Path, help="write metadata, correctness errors and timing samples")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--validate-only", action="store_true", help="CPU traversal model; no torch or CUDA needed")
    args = parser.parse_args()
    if not 0 <= args.logical_rank < args.logical_devices:
        parser.error("--logical-rank must be in [0, logical-devices)")
    if min(args.warmup, args.inner_iters, args.samples) < 1:
        parser.error("warmup, inner-iters and samples must be positive")
    if not all(math.isfinite(x) and x >= 0 for x in (args.rtol, args.atol)):
        parser.error("tolerances must be finite and nonnegative")
    validate_traversals(args)
    if args.validate_only:
        return
    if args.build or args.build_only:
        subprocess.run(["make", "-f", "bench/Makefile.gemm_traversal", f"CUDA_ARCH={args.arch}",
                        f"LOGICAL_DEVICES={args.logical_devices}", f"CUDA_HOME={args.cuda_home}"],
                       cwd=ROOT, check=True)
    if args.build_only:
        return

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("GPU benchmarking requires CUDA-enabled PyTorch; --build-only and "
                         "--validate-only work without it.") from exc
    if not torch.cuda.is_available():
        raise SystemExit("GPU benchmarking requires a Blackwell GPU; use --build-only or --validate-only here.")
    torch.cuda.set_device(args.device)
    expected_cc = (10, 3) if args.arch == "sm_103a" else (10, 0)
    if torch.cuda.get_device_capability() != expected_cc:
        raise SystemExit(f"--arch {args.arch} requires compute capability {expected_cc}")
    torch.manual_seed(1234)
    library = ROOT / "build" / f"libgemm_traversal_{args.arch}_d{args.logical_devices}.so"
    kernels = Kernels(library, args.logical_devices)
    props = torch.cuda.get_device_properties(args.device)
    print(f"\n{props.name}, {props.multi_processor_count} SMs; BF16 inputs/output, FP32 accumulation")
    print("A[M,K] @ B[N,K].T; same kernel, traversal-only change; two MMA consumers; "
          "CUDA-graph device timing; useful TFLOP/s")
    metadata = dict(experiment="shared_kernel_traversal_only", gpu=props.name,
                    sm_count=props.multi_processor_count, torch=torch.__version__,
                    torch_cuda=torch.version.cuda, library=str(library),
                    configuration={key: str(value) if isinstance(value, Path) else value
                                   for key, value in vars(args).items()})
    results = []
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    try:
        with torch.inference_mode(), torch.cuda.stream(stream):
            for m in args.shapes:
                results.extend(benchmark_shape(torch, kernels, args, m))
    finally:
        stream.synchronize()
        kernels.close()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(dict(metadata=metadata, results=results), indent=2) + "\n")
        print(f"\nSaved {args.json}")


if __name__ == "__main__":
    main()
