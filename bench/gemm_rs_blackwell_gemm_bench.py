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
Samples run in a fixed order. Each sample rests --rest seconds, then times one
replay of --graph-size back-to-back GEMMs (no warmup by default) and reports the
per-GEMM average, so every kernel starts from the same thermal/clock state.
Compare against a separate full GEMM+RS run at matching local dimensions to
assess GEMM's contribution; overlap means the times are not additive.
All tile variants are compiled into one extension. --variants names each by its
2-SM (cluster) output tile: 256x256, 512x256 (two consumers), 256x128, or 128x128
(default: 256x256 512x256 128x128). 128x128 uses the M = 128 2-CTA MMA, so each
CTA computes 64x128; the others give each CTA 128 rows per consumer. All use one
accumulator stage per consumer.

Tiles 128 columns wide (mKernel 256x128/128x128, CUTLASS mma Mx128) only run for
M <= --small-tile-max-m (default 4096); 256-wide tiles run at every shape.

Pass --cutlass-dir (or set CUTLASS_LOCAL_DIR) to also time CuTeDSL's
distributed_gemm_reduce_scatter_blackwell.py as a plain GEMM: single-process with
reduce_scatter="none", which compiles out the comm warps and flags. It runs 2-SM
configs on a 2x1 cluster (the MMA pair, like mKernel): mma 256x256 at every shape, and for small M mma
128x128 (128x128 per cluster, 64x128 per CTA, like mKernel's 128x128) and mma
256x128 (128x128 per CTA). Both raster orders, with swizzle fixed at
--cutlass-swizzle (default 8). Each config compiles once per shape.

Pass --tk-dir (or set TK_GEMM_RS_DIR) to also time ThunderKittens' GEMM-only
gemm_rs_b200 build (its _C*.so, built with ARCH=SM103 make). It writes to a
single-process (M, N) TKParallelTensor; --tk-dev-idx picks which rank's tile
rotation to emulate. TK requires CUDA device 0, so for the second GPU use
CUDA_VISIBLE_DEVICES=1 instead of --device 1.
"""

import argparse
import importlib.util
import os
from pathlib import Path
import statistics
import sys
import time
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


def load_tk(tk_dir):
    candidates = sorted(Path(tk_dir).glob("_C*.so"), key=lambda p: p.stat().st_mtime_ns)
    if not candidates:
        raise RuntimeError(f"No _C*.so in {tk_dir}; run `ARCH=SM103 make` there first")
    spec = importlib.util.spec_from_file_location("_C", candidates[-1])
    tk = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tk)
    if not hasattr(tk, "gemm_only"):
        raise RuntimeError(f"{candidates[-1]} lacks gemm_only; rebuild the TK GEMM-only checkout")
    return tk


CUTLASS_KERNEL = ("examples/python/CuTeDSL/cute/blackwell/kernel/distributed/"
                  "distributed_gemm_reduce_scatter_blackwell.py")
# (use_2cta_instrs, mma_tiler_mn, cluster_shape_mn, raster_order): 2-SM benchmark_all
# candidates. mma 256x256 is the full GEMM+RS bench's tile; the 128-wide ones are
# the 128x128 MMA tile and mma 256x128, whose CTA tile (mma M / 2, mma N) is 128x128.
CUTLASS_CONFIGS = [
    (True, mma, cluster, raster)
    for mma in ((256, 256), (128, 128), (256, 128))
    for cluster in ((2, 1),)
    for raster in ("m", "n")
]

# 2-SM output tile -> (num_consumer_warps, col_block, consumer_rows) for gemm_blackwell_launch.
MKERNEL_VARIANTS = {
    "256x256": (1, 256, 128),
    "512x256": (2, 256, 128),
    "256x128": (1, 128, 128),
    "128x128": (1, 128, 64),
}


def runs_at(m, tile_n, args):
    """128-wide tiles are only benchmarked for small M."""
    return tile_n != 128 or m <= args.small_tile_max_m


class CutlassGemms:
    def __init__(self, cutlass_dir, swizzle):
        path = Path(cutlass_dir) / CUTLASS_KERNEL
        if not path.exists():
            raise RuntimeError(f"{path} does not exist")
        spec = importlib.util.spec_from_file_location("cutlass_local_gemm_rs", path)
        self.mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.mod)
        self.swizzle = swizzle

    @staticmethod
    def name(config):
        use_2cta, mma, cluster, raster = config
        return (f"CUTLASS[{'2sm' if use_2cta else '1sm'} mma{mma[0]}x{mma[1]} "
                f"cluster{cluster[0]}x{cluster[1]} raster-{raster}]")

    def launches(self, a, b, c, configs):
        """Compile each config for these tensors; return ({name: launch}, {name: skip reason})."""
        mod = self.mod
        cutlass, cute, utils, cuda = mod.cutlass, mod.cute, mod.utils, mod.cuda
        m, k = a.shape
        n = b.shape[0]

        def view(t):
            # (rows, cols) row-major -> CuTe (rows, cols, L=1) with the batch stride outermost.
            return mod.from_dlpack(t.unsqueeze(0).permute(1, 2, 0), assumed_align=16) \
                .mark_layout_dynamic(leading_dim=1)

        flags = torch.zeros((m // 64) * (n // 64) + 160, dtype=torch.int32, device=a.device)
        flag = mod.from_dlpack(flags, assumed_align=16)
        flag.element_type = cutlass.Int32
        flag = flag.mark_layout_dynamic()
        tensors = dict(a=view(a), b=view(b), c=view(c),
                       # Unused by the GEMM-only kernel, but part of its signature.
                       comm_in_multicast_tensor=view(c), comm_out_unicast_tensor=view(c),
                       barrier_flag_unicast=flag, barrier_flag_multicast=flag,
                       barrier_flag_unicast_per_peer=(flag,))
        launches, skipped = {}, {}
        for config in configs:
            use_2cta, mma, cluster, raster = config
            gemm = mod.Sm100PersistentDenseGemmReduceScatterLDMCKernel(
                cutlass.Float32, cutlass.BFloat16, use_2cta, mma, cluster, True,
                rank_id=0, num_ranks=1, reduce_scatter="none",
                swizzle_size=self.swizzle, raster_order=raster)
            try:
                gemm.can_implement((m, n, k, 1), cutlass.BFloat16, cutlass.BFloat16, "k", "k", "n")
            except mod.testing.CantImplementError as exc:
                skipped[self.name(config)] = str(exc)
                continue
            stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
            compiled = cute.compile(
                gemm, **tensors, stream=stream,
                max_active_clusters=utils.HardwareInfo().get_max_active_clusters(cluster[0] * cluster[1]))

            def launch(compiled=compiled):
                compiled(**tensors, stream=cuda.CUstream(torch.cuda.current_stream().cuda_stream))
            launches[self.name(config)] = launch
        self.flags = flags  # the compiled launches reference this buffer
        return launches, skipped


def benchmark_shape(module, tk, cutlass_gemms, m, n, k, args):
    a = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    b = torch.randn((n, k), device="cuda", dtype=torch.bfloat16) / k**0.25
    b_transposed = b.T
    reference = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
    output = torch.full_like(reference, float("nan"))

    def cublas():
        torch.mm(a, b_transposed, out=reference)

    launches = {"cuBLAS": cublas}
    outputs = {}
    variants = [v for v in args.variants if runs_at(m, MKERNEL_VARIANTS[v][1], args)]
    for variant in variants:
        consumers, cols, rows = MKERNEL_VARIANTS[variant]

        def mkernel(consumers=consumers, cols=cols, rows=rows):
            module.gemm_blackwell_launch(a, b, output, num_consumer_warps=consumers,
                                         col_block=cols, consumer_rows=rows)
        launches[f"mKernel[{variant}]"] = mkernel
        outputs[f"mKernel[{variant}]"] = output
    if tk is not None:
        # Keep the TKParallelTensor alive: c_tk.data_ aliases its VMM allocation.
        c_tk = tk.TKParallelTensor((m, n), dtype=torch.bfloat16, local_rank=torch.cuda.current_device(),
                                   local_world_size=1, multicast=False)

        def thunderkittens():
            tk.gemm_only(a, b, c_tk, args.tk_dev_idx)
        launches["TK"] = thunderkittens
        outputs["TK"] = c_tk.data_
    if cutlass_gemms is not None:
        configs = [c for c in CUTLASS_CONFIGS if runs_at(m, c[1][1], args)]
        cutlass_launches, cutlass_skipped = cutlass_gemms.launches(a, b, output, configs)
        for name, reason in cutlass_skipped.items():
            print(f"  skipping {name}: {reason}", flush=True)
        launches.update(cutlass_launches)
        outputs.update(dict.fromkeys(cutlass_launches, output))

    def check(name):
        try:
            torch.testing.assert_close(outputs[name], reference, rtol=2e-2, atol=2e-2)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}") from exc

    cublas()
    for name, launch in launches.items():
        if name != "cuBLAS":
            outputs[name].fill_(float("nan"))
            launch()
            check(name)
    graphs = {}
    try:
        for name, launch in launches.items():
            graphs[name] = capture(launch, args.graph_size)
        # Changed inputs and poisoned output catch stale or partially written C.
        for _ in range(2):
            a.neg_()
            graphs["cuBLAS"].replay()
            for name, graph in graphs.items():
                if name != "cuBLAS":
                    outputs[name].fill_(float("nan"))
                    graph.replay()
                    check(name)

        samples = {name: [] for name in graphs}
        # Fixed order; every sample starts from the same idle GPU after --rest seconds.
        for _ in range(args.repeats):
            for name, graph in graphs.items():
                time.sleep(args.rest)
                samples[name].append(time_graph(graph, args))
        cu = statistics.median(samples["cuBLAS"])
        flops = 2 * m * n * k
        print(
            f"M={m:6d} N={n} K={k}  "
            f"cuBLAS={cu * 1000:9.2f} us ({flops / (cu * 1e9):7.2f} TF/s)",
            flush=True,
        )
        for variant in variants:
            consumers, cols, rows = MKERNEL_VARIANTS[variant]
            mk = statistics.median(samples[f"mKernel[{variant}]"])
            print(
                f"  mKernel 2sm tile={variant} (CTA {consumers * rows}x{cols}, "
                f"{consumers} consumer{'s' if consumers > 1 else ''})  "
                f"mKernel={mk * 1000:9.2f} us ({flops / (mk * 1e9):7.2f} TF/s)  "
                f"{100 * cu / mk:.1f}% of cuBLAS throughput",
                flush=True,
            )
        if tk is not None:
            tk_ms = statistics.median(samples["TK"])
            print(
                f"  ThunderKittens tile=512x256  "
                f"TK={tk_ms * 1000:9.2f} us ({flops / (tk_ms * 1e9):7.2f} TF/s)  "
                f"{100 * cu / tk_ms:.1f}% of cuBLAS throughput",
                flush=True,
            )
        for name in samples:
            if name.startswith("CUTLASS["):
                ct = statistics.median(samples[name])
                print(
                    f"  {name}  {ct * 1000:9.2f} us ({flops / (ct * 1e9):7.2f} TF/s)  "
                    f"{100 * cu / ct:.1f}% of cuBLAS throughput",
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
    parser.add_argument("--variants", nargs="+", choices=tuple(MKERNEL_VARIANTS),
                        default=["256x256", "512x256", "128x128"],
                        help="mKernel 2-SM output tiles to benchmark (default: 256x256 512x256 128x128)")
    parser.add_argument("--small-tile-max-m", type=int, default=4096,
                        help="Largest M at which 128-wide tiles run (default: 4096)")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cutlass-dir", default=os.environ.get("CUTLASS_LOCAL_DIR"),
                        help="CUTLASS checkout holding the CuTeDSL GEMM+RS example (default: $CUTLASS_LOCAL_DIR)")
    parser.add_argument("--cutlass-swizzle", type=int, default=8, choices=(1, 2, 4, 8),
                        help="Tile-scheduler swizzle size for every CUTLASS config (default: 8)")
    parser.add_argument("--tk-dir", default=os.environ.get("TK_GEMM_RS_DIR"),
                        help="ThunderKittens gemm_rs dir holding the GEMM-only _C*.so (default: $TK_GEMM_RS_DIR)")
    parser.add_argument("--tk-dev-idx", type=int, default=0, choices=range(8),
                        help="Rank whose TK tile rotation to emulate (default: 0)")
    parser.add_argument("--warmup", type=int, default=0, help="Untimed graph replays after each rest")
    parser.add_argument("--iters", type=int, default=1, help="Timed graph replays per sample")
    parser.add_argument("--graph-size", type=int, default=20, help="GEMMs per graph replay")
    parser.add_argument("--repeats", type=int, default=5, help="Number of samples for median timing")
    parser.add_argument("--rest", type=float, default=5.0,
                        help="Idle seconds before each timed sample (default: 5)")
    args = parser.parse_args()
    args.variants = list(dict.fromkeys(args.variants))
    try:
        shapes = [int(value) for value in args.shapes.split(",")]
    except ValueError:
        parser.error("--shapes must be comma-separated integers")
    if args.emulated_world_size <= 0 or 12288 % args.emulated_world_size:
        parser.error("--emulated-world-size must be a positive divisor of 12288")
    k = 12288 // args.emulated_world_size if args.k is None else args.k
    cluster_rows = max(int(variant.split("x")[0]) for variant in args.variants)
    if args.tk_dir:
        cluster_rows = max(cluster_rows, 512)
        if k < 256:
            parser.error("TK requires K >= 256")
    if any(m <= 0 or m % cluster_rows for m in shapes):
        parser.error(f"Selected variants require M to be a positive multiple of {cluster_rows}")
    if args.n <= 0 or args.n % 256 or k <= 0 or k % 64:
        parser.error("N must be a positive multiple of 256; K must be a positive multiple of 64")
    if args.warmup < 0 or args.rest < 0 or min(args.iters, args.graph_size, args.repeats) < 1:
        parser.error("--warmup, --rest must be nonnegative; --iters, --graph-size, --repeats must be positive")
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    if not 0 <= args.device < torch.cuda.device_count():
        parser.error("--device must select a visible CUDA device")
    if args.tk_dir and args.device != 0:
        parser.error("TK requires CUDA device 0; use CUDA_VISIBLE_DEVICES to pick the GPU")
    torch.cuda.set_device(args.device)
    if torch.cuda.get_device_capability()[0] != 10:
        parser.error("This benchmark requires a Blackwell GPU")
    module = load_module.load("gemm_rs_blackwell")
    if not hasattr(module, "gemm_blackwell_tile_variants"):
        parser.error("Rebuild with: make GPU=blackwell INTRA_NUM_DEVICES=2 gemm-rs-blackwell")
    tk = load_tk(args.tk_dir) if args.tk_dir else None
    cutlass_gemms = CutlassGemms(args.cutlass_dir, args.cutlass_swizzle) if args.cutlass_dir else None
    # https://docs.pytorch.org/docs/stable/backends.html#torch.backends.cuda.preferred_blas_library
    torch.backends.cuda.preferred_blas_library("cublas")
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.manual_seed(42)
    print(
        f"{torch.cuda.get_device_name()}; device {args.device}; BF16, FP32 accumulation\n"
        f"mKernel 2-SM tiles: {args.variants}\n"
        f"ThunderKittens: {args.tk_dir or 'off'}\n"
        f"CUTLASS: {args.cutlass_dir or 'off'}"
        f"{f' (swizzle {args.cutlass_swizzle})' if args.cutlass_dir else ''}\n"
        f"128-wide tiles only for M <= {args.small_tile_max_m}\n"
        f"GEMM only; {args.graph_size} GEMMs/graph; median of {args.repeats} samples, "
        f"{args.rest:g} s rest before each.\n"
        "mKernel throughput: 100% matches cuBLAS; higher is faster. Correctness checked before timing.",
        flush=True,
    )
    for m in shapes:
        benchmark_shape(module, tk, cutlass_gemms, m, args.n, k, args)


if __name__ == "__main__":
    main()
