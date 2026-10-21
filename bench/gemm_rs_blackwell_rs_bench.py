"""Standalone BF16 multimem reduce-scatter versus NCCL on one NVLink node.

    make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell
    torchrun --standalone --nproc-per-node=2 bench/gemm_rs_blackwell_rs_bench.py

Default M/N are all five shapes in gemm_rs_blackwell_bench.py. K does not
change the RS workload. Use --shapes 2048,4096 --n 7168 for a custom sweep.
The kernel uses the fused GEMM's ROW_BLOCK x 256 tiles, one warp per CTA, and
the same CTA count. Two lane layouts are compared with equal bytes in flight:
  lane=32B: the fused RS helper, two strided 16-byte multimem reduction loads
            per 32-byte store, 16 stores per batch;
  lane=16B: one contiguous 16-byte reduction load + 16-byte store per lane
            (each warp instruction covers a 512-byte row), 32 loads per batch.

Both implementations read the SAME BF16 [M,N] local input and write a separate
[M/world_size,N] output. Inputs stay immutable during timing, so graph replays
need no reset copies. Allocation, initial readiness, and correctness checks are
outside timing. NCCL chooses its own algorithm; no algorithm is forced.

Reports data-path-only multimem and multimem + the existing exit epoch barrier.
The latter includes cross-rank completion; raw timings omit that cost. Neither
models GEMM interference, tile-readiness waits, or compute/communication overlap.
These are warm-cache CUDA graph timings, median of max-rank samples.
"""

import argparse
import json
import os
from pathlib import Path
import statistics
import sys

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python"))
import load_module


def sync_ranks():
    torch.cuda.synchronize()
    dist.barrier()


def capture(launch, graph_size):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            launch()
    sync_ranks()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(graph_size):
            launch()
    return graph


def time_graph(graph, args):
    for _ in range(args.warmup):
        graph.replay()
    sync_ranks()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(args.iters):
        graph.replay()
    end.record()
    end.synchronize()
    elapsed = torch.tensor(
        [start.elapsed_time(end) / (args.iters * args.graph_size)],
        dtype=torch.float64, device="cuda",
    )
    dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
    return elapsed.item()


def check_output(output, expected, label, exact=False):
    error = None
    try:
        torch.testing.assert_close(
            output.float(), expected, rtol=0 if exact else 8e-2,
            atol=0 if exact else 8e-2,
        )
    except AssertionError as exc:
        error = str(exc)
    passed = torch.tensor([int(error is None)], device="cuda")
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    if not passed.item():
        raise RuntimeError(f"{label}: RS correctness failed. {error or 'Failure on another rank.'}")


def benchmark_shape(module, m, n, rank, world, args):
    source = module.DistBuffer(
        (m, n), dtype=torch.bfloat16, local_rank=rank,
        local_world_size=world, multicast=True,
    )
    epochs = module.DistBuffer(
        (1,), dtype=torch.int32, local_rank=rank,
        local_world_size=world, multicast=False,
    )
    epochs.data_.zero_()
    output = torch.empty((m // world, n), dtype=torch.bfloat16, device="cuda")
    expected = torch.empty_like(output, dtype=torch.float32)

    def nccl():
        dist.reduce_scatter_tensor(output, source.data_, op=dist.ReduceOp.SUM)

    launches = {"NCCL": nccl}
    for lane_bytes in args.lane_bytes:
        for barrier in (False, True):
            def multimem(barrier=barrier, lane_bytes=lane_bytes):
                module.rs_blackwell_launch(source, output, epochs, barrier, lane_bytes)
            name = f"multimem lane={lane_bytes}B" + (" + exit barrier" if barrier else " raw")
            launches[name] = multimem

    graphs = {}
    graph = None
    try:
        # Small integers sum exactly in BF16, exposing indexing errors without
        # confusing them with NCCL's different reduction order/precision.
        source.data_.copy_(torch.randint(-4, 5, (m, n), device="cuda"))
        sync_ranks()
        fp32_input = source.data_.float()
        dist.reduce_scatter_tensor(expected, fp32_input, op=dist.ReduceOp.SUM)
        for name, launch in launches.items():
            output.fill_(float("nan"))
            launch()
            check_output(output, expected, f"eager {name}", exact=True)
        del fp32_input

        for name, launch in launches.items():
            graphs[name] = capture(launch, args.graph_size)
        # Verify fresh, rank-distinct inputs after capture and again on replay.
        for replay in range(2):
            sync_ranks()
            source.data_.normal_().div_(world**0.5)
            sync_ranks()
            fp32_input = source.data_.float()
            dist.reduce_scatter_tensor(expected, fp32_input, op=dist.ReduceOp.SUM)
            for name, graph in graphs.items():
                output.fill_(float("nan"))
                graph.replay()
                check_output(output, expected, f"replay={replay + 1} {name}")
            del fp32_input

        if args.check_only:
            if rank == 0:
                print(f"M={m} N={n}: all RS variants passed eager/replay checks", flush=True)
            return None

        samples = {name: [] for name in graphs}
        names = list(graphs)
        for repeat in range(args.repeats):
            offset = repeat % len(names)
            for name in names[offset:] + names[:offset]:
                samples[name].append(time_graph(graphs[name], args))
        timings = {name: statistics.median(values) for name, values in samples.items()}
        baseline = timings["NCCL"]
        if rank == 0:
            print(
                f"M={m:6d} N={n} input/rank={m * n * 2 / 2**20:.1f} MiB "
                f"output/rank={m * n * 2 / world / 2**20:.1f} MiB",
                flush=True,
            )
            for name, ms in timings.items():
                print(
                    f"  {name:34s} {ms * 1000:9.2f} us  "
                    f"{100 * baseline / ms:7.1f}% of NCCL throughput  "
                    f"{ms / baseline:.3f}x NCCL time",
                    flush=True,
                )
        return {"m": m, "n": n, "world_size": world, "lane_bytes": args.lane_bytes,
                "ms": timings, "samples_ms": samples}
    finally:
        # Free graphs before their communicator and multicast backing allocations.
        sync_ranks()
        graph = None
        graphs.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--shapes", default="2048,4096,8192,16384,32768", help="Comma-separated M values")
    parser.add_argument("--n", type=int, default=7168)
    parser.add_argument("--lane-bytes", nargs="+", type=int, choices=(16, 32), default=[32, 16],
                        help="32 = two strided 16B loads + 32B store per lane; "
                             "16 = one contiguous 16B load/store per lane")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--graph-size", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--save-json", type=Path)
    args = parser.parse_args()
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world not in (2, 4, 8) or int(os.environ.get("LOCAL_WORLD_SIZE", world)) != world:
        parser.error("Use torchrun on one NVLink node with 2, 4, or 8 GPUs")
    try:
        shapes = [int(value) for value in args.shapes.split(",")]
    except ValueError:
        parser.error("--shapes must contain comma-separated M values")
    args.lane_bytes = list(dict.fromkeys(args.lane_bytes))
    if args.n <= 0 or args.n % 256:
        parser.error("N must be a positive multiple of 256")
    if args.warmup < 0 or min(args.iters, args.graph_size, args.repeats) < 1:
        parser.error("Warmup must be nonnegative; iters, graph-size, and repeats must be positive")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world:
        parser.error(f"This run needs {world} visible CUDA GPUs")
    torch.cuda.set_device(rank)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("Build target requires Blackwell GPUs")
    module = load_module.load("gemm_rs_blackwell")
    if module.gemm_rs_blackwell_world_size != world or not hasattr(module, "rs_blackwell_launch"):
        parser.error(f"Rebuild: make GPU=blackwell INTRA_NUM_DEVICES={world} gemm-rs-blackwell")
    alignment = world * module.rs_blackwell_tile_rows
    if any(m <= 0 or m % alignment for m in shapes):
        parser.error(f"M must be positive and divisible by {alignment} (world_size * RS tile rows)")
    torch.manual_seed(42 + rank)
    dist.init_process_group("nccl", device_id=torch.device("cuda", rank))
    results = []
    try:
        if rank == 0:
            print(
                f"{torch.cuda.get_device_name()}; {world} GPUs; NCCL {torch.cuda.nccl.version()}\n"
                f"BF16 SUM; {module.rs_blackwell_blocks} CTAs, 1 warp/CTA, "
                f"{module.rs_blackwell_tile_rows}x256 tiles; lane layouts {args.lane_bytes} B\n"
                "Raw = data path only; + exit barrier = cross-rank completion included.\n"
                "Immutable inputs; no GEMM/readiness waits; median of max-rank graph timings.\n"
                f"NCCL_ALGO={os.environ.get('NCCL_ALGO', 'auto')} "
                f"NCCL_PROTO={os.environ.get('NCCL_PROTO', 'auto')} "
                f"NCCL_NVLS_ENABLE={os.environ.get('NCCL_NVLS_ENABLE', 'auto')}",
                flush=True,
            )
        for m in shapes:
            result = benchmark_shape(module, m, args.n, rank, world, args)
            if result is not None:
                results.append(result)
    finally:
        dist.destroy_process_group()
    if rank == 0 and args.save_json:
        args.save_json.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
