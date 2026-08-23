"""Adapter exposing NVIDIA's CuTeDSL distributed GEMM+AllReduce as a bench condition.

Upstream file:
    cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/
        distributed_gemm_all_reduce_blackwell.py   (pin to the tag matching your
                                                    nvidia-cutlass-dsl wheel)

That example is a standalone script -- its own argparse, its own process-group
setup -- so it is loaded by path and its kernel class is driven directly,
reusing the process group the bench has already initialised. Everything in it
is guarded behind `if __name__ == "__main__"`, so importing it has no side
effects.

Semantics match our kernel: each rank computes a full M x N GEMM from its own
K-slice, then the result is all-reduced. The example's own correctness check
does `dist.all_reduce(local_C)` as the reference, same as ours, so passing
K = N // world_size gives an apples-to-apples comparison.

Point one of these at the file:
    CUTLASS_DGEMM_AR=/path/to/distributed_gemm_all_reduce_blackwell.py
    CUTLASS_PATH=/path/to/cutlass          (a source checkout, not the wheel)

Set CUTLASS_AUTOTUNE=1 to pick the best (swizzle_size, raster_order) per shape
by measurement instead of using the matched defaults.
"""
import importlib.util
import os
from pathlib import Path

_ENV_FILE = "CUTLASS_DGEMM_AR"
_ENV_ROOT = "CUTLASS_PATH"
_ENV_AUTOTUNE = "CUTLASS_AUTOTUNE"
_REL = ("examples/python/CuTeDSL/cute/blackwell/kernel/distributed/"
        "distributed_gemm_all_reduce_blackwell.py")

# Tile geometry is pinned to match our kernel so the comparison is like for
# like: 256x256 MMA tile, 2-CTA cluster, TMA store, multimem load-reduce/store.
MMA_TILER_MN = (256, 256)
CLUSTER_SHAPE_MN = (2, 1)
USE_2CTA_INSTRS = True
USE_TMA_STORE = True
ALL_REDUCE = "LDMCxSTMC"

# Autotune grid. Deliberately only the two scheduling knobs -- widening it to
# mma_tiler/cluster would let CUTLASS win on a different tile shape, which is a
# different (and less informative) comparison than "same tiling, best schedule".
AUTOTUNE_SWIZZLES = (1, 2, 4, 8)
AUTOTUNE_RASTERS = ("m", "n")

_module = None
_tensor_cache = {}


def _locate():
    p = os.environ.get(_ENV_FILE)
    if p:
        return Path(p)
    root = os.environ.get(_ENV_ROOT)
    if root:
        return Path(root) / _REL
    return None


def _load():
    global _module
    if _module is None:
        path = _locate()
        if path is None:
            raise RuntimeError(
                f"set {_ENV_FILE} to distributed_gemm_all_reduce_blackwell.py, "
                f"or {_ENV_ROOT} to a cutlass checkout")
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        spec = importlib.util.spec_from_file_location(
            "cutlass_dgemm_ar_example", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _module = mod
    return _module


def availability():
    """(ok, reason). Cheap enough to call before every shape."""
    try:
        _load()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def matched_swizzle(M):
    """Mirror of our SUPERGROUP_WIDTH heuristic in gemm_ar_blackwell.cuh.

    CUTLASS's swizzle_size and our SUPERGROUP_WIDTH are the same quantity in the
    same units -- how many cluster-columns are grouped before the walk advances
    in M. CUTLASS defaults it to 1 (no grouping), so leaving it alone compares
    our L2-aware tile walk against an unswizzled one.
    """
    return 4 if M <= 4096 else 8


def _tensors(ex, M, N, K, rank, world_size, device):
    """Allocate once per shape and reuse across autotune candidates.

    The flag buffer is worst-case sized upstream ((m/64)*(n/64)+160) precisely
    so it can be shared across candidates, and A/B/C do not depend on the
    schedule -- so reallocating per candidate would just burn several GB at
    M=N=32768 for nothing.
    """
    import cutlass
    key = (M, N, K)
    if key not in _tensor_cache:
        _tensor_cache[key] = ex.allocate_tensors(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            # Our A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major.
            a_major="k",
            b_major="n",
            c_major="n",
            num_workspace=1,
            device=device,
            slot_init_mode="benchmark",
            global_rank=rank,
            local_rank=device,
            world_size=world_size,
        )
    return _tensor_cache[key]


def _compile_one(ex, tensors, *, M, N, K, rank, world_size,
                 swizzle_size, raster_order):
    """Compile one candidate. Returns a launcher, or None if unsupported."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.testing as testing
    import cutlass.utils as utils
    import torch

    kernel = ex.Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel(
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.BFloat16,
        use_2cta_instrs=USE_2CTA_INSTRS,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_tma_store=USE_TMA_STORE,
        rank_id=rank,
        num_ranks=world_size,
        all_reduce=ALL_REDUCE,
        swizzle_size=swizzle_size,
        raster_order=raster_order,
    )

    # Upstream raises CantImplementError for combinations the kernel rejects --
    # notably num_clusters_n % swizzle_size != 0 under raster_order="m".
    try:
        kernel.can_implement(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            a_major="k",
            b_major="n",
            c_major="n",
        )
    except testing.CantImplementError:
        return None

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    kwargs = dict(
        a=tensors["cute_tensor_a_list"][0],
        b=tensors["cute_tensor_b_list"][0],
        c=tensors["cute_tensor_c"],
        comm_in_multicast_tensor=tensors["cute_tensor_comm_in_mc"],
        comm_out_multicast_tensor=tensors["cute_tensor_comm_out_mc"],
        barrier_flag_unicast=tensors["cute_tensor_flag_unicast"],
        barrier_flag_multicast=tensors["cute_tensor_flag_multicast"],
        stream=stream,
    )
    compiled = cute.compile(
        kernel,
        **kwargs,
        max_active_clusters=utils.HardwareInfo().get_max_active_clusters(
            CLUSTER_SHAPE_MN[0] * CLUSTER_SHAPE_MN[1]),
    )

    def launch():
        compiled(**kwargs)

    launch._keepalive = tensors
    launch.config = (swizzle_size, raster_order)
    return launch


def _time(launch, iters=5):
    """Median launch time in ms, agreed across ranks (max, as for a collective)."""
    import torch
    import torch.distributed as dist

    for _ in range(2):
        launch()
    torch.cuda.synchronize()
    dist.barrier()

    samples = []
    for _ in range(iters):
        torch.cuda.synchronize()
        dist.barrier()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        launch()
        e.record()
        samples.append((s, e))
    torch.cuda.synchronize()

    ms = sorted(s.elapsed_time(e) for s, e in samples)[len(samples) // 2]
    t = torch.tensor([ms], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def build(*, M, N, K, rank, world_size, device,
          swizzle_size=None, raster_order="m", autotune=None):
    """Compile for one shape. Returns a zero-arg launcher.

    swizzle_size=None uses matched_swizzle(M), i.e. the same tile-grouping width
    our kernel uses -- without it CUTLASS runs unswizzled and the comparison is
    not like for like.

    autotune=None reads CUTLASS_AUTOTUNE. When on, every (swizzle, raster) pair
    is compiled and timed and the fastest is kept. All ranks evaluate the same
    candidates in the same order and agree on the winner by all-reducing the
    timings, which they must -- these are collectives.
    """
    ex = _load()
    tensors = _tensors(ex, M, N, K, rank, world_size, device)

    if autotune is None:
        autotune = os.environ.get(_ENV_AUTOTUNE, "0") not in ("0", "", "false")

    if not autotune:
        if swizzle_size is None:
            swizzle_size = matched_swizzle(M)
        launch = _compile_one(ex, tensors, M=M, N=N, K=K, rank=rank,
                              world_size=world_size, swizzle_size=swizzle_size,
                              raster_order=raster_order)
        if launch is None:
            raise RuntimeError(
                f"cutlass rejects swizzle_size={swizzle_size} "
                f"raster_order={raster_order} at M={M} N={N}")
        return launch

    best, best_ms = None, float("inf")
    tried = []
    for raster in AUTOTUNE_RASTERS:
        for swz in AUTOTUNE_SWIZZLES:
            cand = _compile_one(ex, tensors, M=M, N=N, K=K, rank=rank,
                                world_size=world_size, swizzle_size=swz,
                                raster_order=raster)
            if cand is None:
                continue
            ms = _time(cand)
            tried.append((swz, raster, ms))
            if ms < best_ms:
                best, best_ms = cand, ms
    if best is None:
        raise RuntimeError(f"no cutlass config is implementable at M={M} N={N}")
    best.autotune_log = tried
    return best
