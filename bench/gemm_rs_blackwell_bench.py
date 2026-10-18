"""Single-node BF16 GEMM + reduce-scatter: cuBLAS/NCCL and CUTLASS.

Run with torchrun --standalone --nproc-per-node=8 and set CUTLASS_PATH to
the pinned checkout linked below. K is the per-rank shard of global K=12288.
"""

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import torch
import torch.distributed as dist

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_thirdparty

class BlackwellBenchConfig:
    arch = "blackwell"
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    is_chief = local_rank == 0
    num_nodes = 1

    shapes_to_test = [2048, 4096, 8192, 16384, 32768]
    output_dim = 7168
    split_inner_dim = 12288 // world_size

    # Keys match the pinned CUTLASS example's CLI / run() keyword arguments:
    # https://github.com/NVIDIA/cutlass/blob/dcf215af68a2d08d305076c152a06f201728cd53/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/distributed_gemm_reduce_scatter_blackwell.py#L3009
    cutlass_fixed_config = {
        "use_cuda_graph": True,
        "use_2cta_instrs": True,
        "benchmark_or_test": "benchmark",
        # 2-CTA instructions require an even cluster M; the CLI default is 1.
        "cluster_shape_mn": (2, 1),
    }

    # (raster_order, swizzle_size, mma_tiler_mn)
    cutlass_autotune_configs = [
        ("m", 1, (256, 256)),
        ("m", 2, (256, 256)),
        ("m", 4, (256, 256)),
        ("m", 8, (256, 256)),
        ("n", 1, (256, 256)),
        ("n", 2, (256, 256)),
        ("n", 4, (256, 256)),
        ("n", 8, (256, 256)),
    ]

    # M=2048 also tries the smaller tile.
    cutlass_autotune_configs_2048 = cutlass_autotune_configs + [
        ("m", 1, (128, 128)),
        ("m", 2, (128, 128)),
        ("m", 4, (128, 128)),
        ("m", 8, (128, 128)),
        ("n", 1, (128, 128)),
        ("n", 2, (128, 128)),
        ("n", 4, (128, 128)),
        ("n", 8, (128, 128)),
    ]


@dataclass
class BlackwellBenchVars:
    baseline_a: torch.Tensor | None = None
    baseline_b: torch.Tensor | None = None
    baseline_c: torch.Tensor | None = None
    baseline_c_rs: torch.Tensor | None = None
    baseline_graph: torch.cuda.CUDAGraph | None = None


def useful_tflops(M: int, N: int, K: int, elapsed_time_ms: float) -> float:
    """Per-rank GEMM throughput, with elapsed time including reduce-scatter."""
    return 2 * M * N * K / (elapsed_time_ms * 1.0e9)


def cutlass_config_label(candidate) -> str:
    raster, swizzle, tile = candidate
    return f"raster={raster} swizzle={swizzle} tile={tile[0]}x{tile[1]}"


def cutlass_config_kwargs(config: BlackwellBenchConfig, candidate) -> dict:
    raster, swizzle, tile = candidate
    return {
        **config.cutlass_fixed_config,
        "raster_order": raster,
        "swizzle_size": swizzle,
        "mma_tiler_mn": tile,
    }


def cutlass_available(config: BlackwellBenchConfig) -> bool:
    ok, why = load_thirdparty.cutlass_availability(
        "distributed_gemm_reduce_scatter_blackwell.py"
    )
    vote = torch.tensor([int(ok)], device="cuda")
    dist.all_reduce(vote, op=dist.ReduceOp.MIN)
    if not vote.item() and config.is_chief:
        print(
            f"[skip] CUTLASS GEMM reduce-scatter: {why or 'unavailable on a peer'}",
            flush=True,
        )
    return bool(vote.item())


def make_baseline_buffers(config: BlackwellBenchConfig, M: int, N: int, K: int) -> BlackwellBenchVars:
    if M % config.world_size:
        raise ValueError(f"M={M} must be divisible by world_size={config.world_size}")
    torch.manual_seed(42 + config.local_rank)
    return BlackwellBenchVars(
        baseline_a=torch.randn((M, K), device="cuda", dtype=torch.bfloat16) / K**0.25,
        baseline_b=torch.randn((N, K), device="cuda", dtype=torch.bfloat16) / K**0.25,
        baseline_c=torch.empty((M, N), device="cuda", dtype=torch.bfloat16),
        baseline_c_rs=torch.empty(
            (M // config.world_size, N), device="cuda", dtype=torch.bfloat16,
        ),
    )


def run_baseline(buffers: BlackwellBenchVars):
    torch.mm(buffers.baseline_a, buffers.baseline_b.T, out=buffers.baseline_c)
    dist.reduce_scatter_tensor(buffers.baseline_c_rs, buffers.baseline_c)


def capture_cuda_graph(run_once) -> torch.cuda.CUDAGraph:
    """Warm up and capture one launch; return the graph for replay or checking."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run_once()
    torch.cuda.synchronize()
    dist.barrier()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run_once()
    return graph


def check_correctness_gemm_rs_blackwell(config: BlackwellBenchConfig):
    if not cutlass_available(config):
        return True

    for M in config.shapes_to_test:
        N, K = config.output_dim, config.split_inner_dim
        candidates = (
            config.cutlass_autotune_configs_2048 if M == 2048
            else config.cutlass_autotune_configs
        )
        for candidate in candidates:
            result = load_thirdparty.run_cutlass_reduce_scatter_once(
                m=M, n=N, k=K, config=cutlass_config_kwargs(config, candidate),
                warmup=0, iterations=1, check=True,
            )
            if result is not None and config.is_chief:
                print(
                    f"[correctness] CUTLASS M={M} {cutlass_config_label(candidate)}: PASS",
                    flush=True,
                )
    return True


def gemm_rs_blackwell_prepare(
    config: BlackwellBenchConfig, M: int, N: int, K: int, warmup: int, iters: int,
):
    """Return (fn, name, should_wrap); CUTLASS returns its own timing in ms."""
    buffers = make_baseline_buffers(config, M, N, K)
    
    def run_baseline_for_graph_capture():
        run_baseline(buffers)

    baseline_graph = capture_cuda_graph(run_baseline_for_graph_capture)

    def bench_baseline():
        baseline_graph.replay()

    fns = [(bench_baseline, "baseline", True)]
    if not cutlass_available(config):
        return fns

    candidates = (
        config.cutlass_autotune_configs_2048 if M == 2048
        else config.cutlass_autotune_configs
    )
    timings = []
    for candidate in candidates:
        ms = load_thirdparty.run_cutlass_reduce_scatter_once(
            m=M, n=N, k=K, config=cutlass_config_kwargs(config, candidate),
            warmup=warmup, iterations=iters,
        )
        if ms is not None:
            timings.append((candidate, ms))

    if not timings:
        if config.is_chief:
            print(
                f"[skip] CUTLASS GEMM reduce-scatter: no config ran at M={M} N={N}",
                flush=True,
            )
        return fns

    # Each timing is already the maximum across ranks, so every rank selects
    # the same candidate even when their local fastest configurations differ.
    best_config, _ = min(timings, key=lambda item: item[1])
    if config.is_chief:
        for candidate, ms in sorted(timings, key=lambda item: item[1]):
            mark = " <- best" if candidate == best_config else ""
            print(
                f"  [autotune] {cutlass_config_label(candidate)}: "
                f"{ms:8.3f} ms  {useful_tflops(M, N, K, ms):8.2f} TFLOP/s{mark}",
                flush=True,
            )

    def bench_cutlass():
        return load_thirdparty.run_cutlass_reduce_scatter_once(
            m=M, n=N, k=K, config=cutlass_config_kwargs(config, best_config),
            warmup=warmup, iterations=iters,
        )

    fns.append((bench_cutlass, "cutlass", False))
    return fns


def benchmark_cuda(run_once, warmup: int, iters: int) -> float:
    """Time the supplied launch or graph replay, taking the slowest rank."""
    for _ in range(warmup):
        run_once()
    torch.cuda.synchronize()
    dist.barrier()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    end.synchronize()
    elapsed = torch.tensor([start.elapsed_time(end) / iters], dtype=torch.float64, device="cuda")
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("check", "bench"), default="bench")
    parser.add_argument("--shapes", help="Comma-separated M values")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=30)
    args = parser.parse_args()
    config = BlackwellBenchConfig()
    if args.shapes is not None:
        config.shapes_to_test = [int(value) for value in args.shapes.split(",")]
    if not config.shapes_to_test or any(
        m <= 0 or m % config.world_size for m in config.shapes_to_test
    ):
        parser.error("M values must be positive and divisible by WORLD_SIZE")
    if args.warmup < 0 or args.iters < 1:
        parser.error("--warmup must be nonnegative and --iters must be positive")
    if config.world_size not in (2, 4, 8):
        parser.error("Launch with torchrun using 2, 4, or 8 GPUs on one node")
    if int(os.environ.get("LOCAL_WORLD_SIZE", config.world_size)) != config.world_size:
        parser.error("This benchmark requires a single node")

    torch.cuda.set_device(config.local_rank)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This benchmark requires Blackwell GPUs")
    dist.init_process_group("nccl", device_id=torch.device("cuda", config.local_rank))
    try:
        check_correctness_gemm_rs_blackwell(config)
        if args.mode == "check":
            return
        for M in config.shapes_to_test:
            N, K = config.output_dim, config.split_inner_dim
            fns = gemm_rs_blackwell_prepare(config, M, N, K, args.warmup, args.iters)
            results = {}
            for fn, name, should_wrap in fns:
                ms = (
                    benchmark_cuda(fn, args.warmup, args.iters)
                    if should_wrap else fn()
                )
                if ms is not None:
                    results[name] = ms
                    if config.is_chief:
                        print(
                            f"M={M} N={N} K_local={K} {name}: "
                            f"{ms:.3f} ms  {useful_tflops(M, N, K, ms):.2f} TFLOP/s",
                            flush=True,
                        )
            if config.is_chief and "cutlass" in results:
                print(
                    f"  CUTLASS speedup vs cuBLAS+NCCL: "
                    f"{results['baseline'] / results['cutlass']:.3f}x",
                    flush=True,
                )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
