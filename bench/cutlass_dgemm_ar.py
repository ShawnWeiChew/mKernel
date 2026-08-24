"""Adapter exposing NVIDIA's CuTeDSL distributed GEMM+AllReduce as bench conditions.

Two upstream variants are supported, both under
    cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/

    distributed_gemm_all_reduce_blackwell.py          -> variant "ldmc"
    distributed_gemm_all_reduce_lamport_blackwell.py  -> variant "lamport"

"ldmc" is the NVLS path: multimem.ld_reduce + multimem.st, reduce-scatter
shaped, same primitives as our kernel. "lamport" is the latency-optimised
one-shot path: every rank reads every peer's full partial over plain P2P loads
and reduces in registers, with no barrier at all -- arrival is detected by
spinning on a -0.0 sentinel written into the data itself. That trades wire
bandwidth for a saved round-trip, so at world_size 8 it moves ~(R-1)=7x the
bytes and should only win when the message is small enough for latency to
dominate. Both are worth having in the table for exactly that reason.

Each example is a standalone script -- own argparse, own process-group setup --
so it is loaded by path and its kernel class driven directly, reusing the
process group the bench already initialised. Everything is guarded behind
`if __name__ == "__main__"`, so importing has no side effects.

Semantics match our kernel: each rank computes a full M x N GEMM from its own
K-slice, then the result is all-reduced. Both examples check themselves against
`dist.all_reduce(local_C)`, same reference as ours, so K = N // world_size is
apples to apples.

Point these at the files (or set CUTLASS_PATH to a source checkout and let both
be found relatively):
    CUTLASS_DGEMM_AR=/path/to/distributed_gemm_all_reduce_blackwell.py
    CUTLASS_DGEMM_AR_LAMPORT=/path/to/distributed_gemm_all_reduce_lamport_blackwell.py
    CUTLASS_PATH=/path/to/cutlass          (a source checkout, not the wheel)

Set CUTLASS_AUTOTUNE=1 to pick the best (swizzle_size, raster_order) per shape
per variant by measurement instead of using the matched defaults.
"""
import importlib.util
import os
from pathlib import Path

_ENV_ROOT = "CUTLASS_PATH"
_ENV_AUTOTUNE = "CUTLASS_AUTOTUNE"
_REL_DIR = "examples/python/CuTeDSL/cute/blackwell/kernel/distributed"

# Tile geometry is pinned to match our kernel so the comparison is like for
# like: 256x256 MMA tile, 2-CTA cluster, TMA store. Lamport *requires*
# use_tma_store -- its multicast TMA store is the only producer path that fans
# the epilogue out to peers atomically per 16B, which is what makes the
# data-as-flag scheme sound.
MMA_TILER_MN = (256, 256)
CLUSTER_SHAPE_MN = (2, 1)
USE_2CTA_INSTRS = True
USE_TMA_STORE = True

# Autotune grid. Deliberately only the two scheduling knobs -- widening it to
# mma_tiler/cluster would let CUTLASS win on a different tile shape, which is a
# different (and less informative) comparison than "same tiling, best schedule".
AUTOTUNE_SWIZZLES = (1, 2, 4, 8)
AUTOTUNE_RASTERS = ("m", "n")

VARIANTS = ("ldmc", "lamport")
DEFAULT_VARIANT = "ldmc"

_SPEC = {
    "ldmc": dict(
        env="CUTLASS_DGEMM_AR",
        filename="distributed_gemm_all_reduce_blackwell.py",
        cls="Sm100PersistentDenseGemmAllReduceLDMCxSTMCKernel",
        all_reduce="LDMCxSTMC",
        # One workspace: the LDMC path writes and reduces in place, so there is
        # no slot to rotate.
        num_workspace=1,
    ),
    "lamport": dict(
        env="CUTLASS_DGEMM_AR_LAMPORT",
        filename="distributed_gemm_all_reduce_lamport_blackwell.py",
        cls="Sm100PersistentDenseGemmAllReduceLamportKernel",
        all_reduce="Lamport",
        # NUM_C_BUFFERS = 3 is fixed inside the example's allocate_tensors
        # (ping / pong / cooling) and it asserts num_workspace == that, so the
        # A/B ring rotates in lockstep with the slot rotation.
        num_workspace=3,
    ),
}

_module = {}
_tensor_cache = {}
# Rotation counter per (variant, shape). Shared across autotune candidates on
# purpose -- see _lamport_launcher.
_rotation = {}


def _check(variant):
    if variant not in _SPEC:
        raise ValueError(f"unknown variant {variant!r}, expected one of {VARIANTS}")
    return _SPEC[variant]


def _locate(variant):
    spec = _check(variant)
    p = os.environ.get(spec["env"])
    if p:
        return Path(p)
    root = os.environ.get(_ENV_ROOT)
    if root:
        return Path(root) / _REL_DIR / spec["filename"]
    return None


def _load(variant=DEFAULT_VARIANT):
    if variant not in _module:
        spec = _check(variant)
        path = _locate(variant)
        if path is None:
            raise RuntimeError(
                f"set {spec['env']} to {spec['filename']}, "
                f"or {_ENV_ROOT} to a cutlass checkout")
        if not path.is_file():
            raise FileNotFoundError(f"{path} does not exist")
        loader = importlib.util.spec_from_file_location(
            f"cutlass_dgemm_ar_example_{variant}", path)
        mod = importlib.util.module_from_spec(loader)
        loader.loader.exec_module(mod)
        _module[variant] = mod
    return _module[variant]


def availability(variant=DEFAULT_VARIANT):
    """(ok, reason). Cheap enough to call before every shape."""
    try:
        _load(variant)
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


def _tensors(ex, variant, M, N, K, rank, world_size, device):
    """Allocate once per (variant, shape) and reuse across autotune candidates.

    A/B/C do not depend on the schedule, and upstream sizes the flag buffer for
    the worst case precisely so it can be shared -- so reallocating per
    candidate would just burn several GB at M=N=32768 for nothing.
    """
    import inspect
    import cutlass
    key = (variant, M, N, K)
    if key not in _tensor_cache:
        want = dict(
            mnkl=(M, N, K, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            # Our A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major.
            a_major="k",
            b_major="n",
            c_major="n",
            num_workspace=_SPEC[variant]["num_workspace"],
            device=device,
            # For Lamport this is what arms every one of the three slots with
            # the -0.0 sentinel. "test" arms only slot 0 and fills the rest with
            # random data for its own verifier, which would make the first
            # rotations read stale non-sentinel values.
            slot_init_mode="benchmark",
            global_rank=rank,
            local_rank=device,
            world_size=world_size,
        )
        # The two examples' allocate_tensors do not take the same keywords --
        # lamport's has no `local_rank`, for one -- and upstream is free to add
        # or drop more between tags. Pass the intersection rather than hardcode
        # a per-variant list that silently rots.
        accepted = inspect.signature(ex.allocate_tensors).parameters
        if not any(p.kind is p.VAR_KEYWORD for p in accepted.values()):
            want = {k: v for k, v in want.items() if k in accepted}
        missing = [name for name, p in accepted.items()
                   if p.default is p.empty
                   and p.kind not in (p.VAR_KEYWORD, p.VAR_POSITIONAL)
                   and name not in want]
        if missing:
            raise RuntimeError(
                f"cutlass[{variant}] allocate_tensors needs arguments this "
                f"adapter does not supply: {missing}")
        _tensor_cache[key] = ex.allocate_tensors(**want)
    return _tensor_cache[key]


def release():
    """Drop every cached allocation.

    The bench walks shapes largest-last and holds its own A/B/C alongside, and
    Lamport's footprint is not small: three C slots plus a three-deep A/B ring
    plus the comm-out buffer, which at M=N=32768 is several GB more than the
    LDMC path. Nothing here survives a shape, so hand it back.
    """
    _tensor_cache.clear()
    _rotation.clear()


def _ldmc_kwargs(tensors, stream):
    return dict(
        a=tensors["cute_tensor_a_list"][0],
        b=tensors["cute_tensor_b_list"][0],
        c=tensors["cute_tensor_c"],
        comm_in_multicast_tensor=tensors["cute_tensor_comm_in_mc"],
        comm_out_multicast_tensor=tensors["cute_tensor_comm_out_mc"],
        barrier_flag_unicast=tensors["cute_tensor_flag_unicast"],
        barrier_flag_multicast=tensors["cute_tensor_flag_multicast"],
        stream=stream,
    )


def _lamport_kwargs(tensors, stream, rank, i):
    """Kwargs for rotation index `i`, mirroring upstream's make_kernel_kwargs.

        ping    = i       % 3   this iter's read + write + epilogue target
        pong    = (i + 1) % 3   this iter's clear target (armed for next iter)
        cooling = (i + 2) % 3   untouched, draining
    """
    uc = tensors["cute_tensors_c_uc_per_peer_grouped"]
    mc = tensors["cute_tensors_c_mc_per_peer_grouped"]
    n_buf = len(uc)
    ping, pong = i % n_buf, (i + 1) % n_buf
    return dict(
        a=tensors["cute_tensor_a_list"][i % n_buf],
        b=tensors["cute_tensor_b_list"][i % n_buf],
        c_multicast_tensor=mc[ping][rank],
        comm_in_unicast_tensor_per_peer=uc[ping],
        comm_clear_unicast_tensor_per_peer=uc[pong],
        comm_out_unicast_tensor=tensors["cute_tensor_comm_out_uc"],
        stream=stream,
    )


def _compile_one(ex, tensors, *, variant, M, N, K, rank, world_size,
                 swizzle_size, raster_order):
    """Compile one candidate. Returns a launcher, or None if unsupported."""
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.testing as testing
    import cutlass.utils as utils
    import torch

    spec = _SPEC[variant]
    kernel = getattr(ex, spec["cls"])(
        acc_dtype=cutlass.Float32,
        c_dtype=cutlass.BFloat16,
        use_2cta_instrs=USE_2CTA_INSTRS,
        mma_tiler_mn=MMA_TILER_MN,
        cluster_shape_mn=CLUSTER_SHAPE_MN,
        use_tma_store=USE_TMA_STORE,
        rank_id=rank,
        num_ranks=world_size,
        all_reduce=spec["all_reduce"],
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
    max_active_clusters = utils.HardwareInfo().get_max_active_clusters(
        CLUSTER_SHAPE_MN[0] * CLUSTER_SHAPE_MN[1])

    if variant == "ldmc":
        kwargs = _ldmc_kwargs(tensors, stream)
        compiled = cute.compile(kernel, **kwargs,
                                max_active_clusters=max_active_clusters)

        def launch():
            compiled(**kwargs)
    else:
        launch = _lamport_launcher(
            cute, kernel, tensors, stream, rank,
            max_active_clusters=max_active_clusters,
            rot_key=(variant, M, N, K))

    launch._keepalive = tensors
    launch.config = (swizzle_size, raster_order)
    return launch


def _lamport_launcher(cute, kernel, tensors, stream, rank, *,
                      max_active_clusters, rot_key):
    """Launcher that advances the ping/pong/cooling rotation on every call.

    Rotation is not optional. The Lamport consumer spins while the loaded word
    still equals the -0.0 sentinel, so a slot is only safe to read if it was
    scrubbed by the previous iteration's clear. Calling with a fixed slot would
    hit data left over from the last call, exit the spin immediately on stale
    values, and report a *faster* time for a wrong answer -- the worst possible
    failure mode in a benchmark.

    The counter is keyed by shape and shared across autotune candidates rather
    than restarting per candidate. Candidates share one set of tensors, so
    restarting at ping=0 would land on whichever slot the previous candidate
    left dirty. Continuing the count keeps the invariant "the slot I am about to
    read was cleared one iteration ago" true across candidate boundaries.

    All ranks step the counter in lockstep because they issue the same calls in
    the same order -- which they must anyway, these are collectives.
    """
    n_buf = len(tensors["cute_tensors_c_uc_per_peer_grouped"])
    kwargs_ring = [_lamport_kwargs(tensors, stream, rank, i) for i in range(n_buf)]
    # Pointers are kernel arguments, not baked constants, so one compile covers
    # every rotation; upstream compiles against slot 0 the same way.
    compiled = cute.compile(kernel, **kwargs_ring[0],
                            max_active_clusters=max_active_clusters)
    _rotation.setdefault(rot_key, 0)

    def launch():
        i = _rotation[rot_key]
        compiled(**kwargs_ring[i % n_buf])
        _rotation[rot_key] = i + 1

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


def build(*, M, N, K, rank, world_size, device, variant=DEFAULT_VARIANT,
          swizzle_size=None, raster_order="m", autotune=None):
    """Compile one variant for one shape. Returns a zero-arg launcher.

    swizzle_size=None uses matched_swizzle(M), i.e. the same tile-grouping width
    our kernel uses -- without it CUTLASS runs unswizzled and the comparison is
    not like for like.

    autotune=None reads CUTLASS_AUTOTUNE. When on, every (swizzle, raster) pair
    is compiled and timed and the fastest is kept, per variant -- the two have
    different comm structure and there is no reason for them to prefer the same
    schedule. All ranks evaluate the same candidates in the same order and agree
    on the winner by all-reducing the timings, which they must -- these are
    collectives.
    """
    _check(variant)
    ex = _load(variant)
    tensors = _tensors(ex, variant, M, N, K, rank, world_size, device)

    if autotune is None:
        autotune = os.environ.get(_ENV_AUTOTUNE, "0") not in ("0", "", "false")

    def compile_at(swz, raster):
        return _compile_one(ex, tensors, variant=variant, M=M, N=N, K=K,
                            rank=rank, world_size=world_size,
                            swizzle_size=swz, raster_order=raster)

    if not autotune:
        if swizzle_size is None:
            swizzle_size = matched_swizzle(M)
        launch = compile_at(swizzle_size, raster_order)
        if launch is None:
            raise RuntimeError(
                f"cutlass[{variant}] rejects swizzle_size={swizzle_size} "
                f"raster_order={raster_order} at M={M} N={N}")
        return launch

    best, best_ms = None, float("inf")
    tried = []
    for raster in AUTOTUNE_RASTERS:
        for swz in AUTOTUNE_SWIZZLES:
            cand = compile_at(swz, raster)
            if cand is None:
                continue
            ms = _time(cand)
            tried.append((swz, raster, ms))
            if ms < best_ms:
                best, best_ms = cand, ms
    if best is None:
        raise RuntimeError(
            f"no cutlass[{variant}] config is implementable at M={M} N={N}")
    best.autotune_log = tried
    return best
