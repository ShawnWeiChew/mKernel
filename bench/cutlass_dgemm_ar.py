"""Adapter exposing NVIDIA's CuTeDSL distributed GEMM+AllReduce as a bench condition.

Upstream file:
    cutlass/examples/python/CuTeDSL/cute/blackwell/kernel/distributed/
        distributed_gemm_all_reduce_blackwell.py

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
    CUTLASS_PATH=/path/to/cutlass          (the example is found relative to it)
"""
import importlib.util
import os
from pathlib import Path

_ENV_FILE = "CUTLASS_DGEMM_AR"
_ENV_ROOT = "CUTLASS_PATH"
_REL = ("examples/python/CuTeDSL/cute/blackwell/kernel/distributed/"
        "distributed_gemm_all_reduce_blackwell.py")

# Config chosen to match our kernel: 256x256 MMA tile, 2-CTA cluster, TMA store,
# multimem load-reduce / store (LDMCxSTMC). Changing these makes the comparison
# something other than like-for-like.
MMA_TILER_MN = (256, 256)
CLUSTER_SHAPE_MN = (2, 1)
USE_2CTA_INSTRS = True
USE_TMA_STORE = True
ALL_REDUCE = "LDMCxSTMC"

_module = None


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
    except Exception as exc:  # missing file, missing cutlass, arch mismatch
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def build(*, M, N, K, rank, world_size, device):
    """Allocate + compile for one shape. Returns a zero-arg launcher.

    The launcher enqueues one iteration onto the *current* torch stream, so the
    bench's existing event bracketing times it the same way it times everything
    else.
    """
    ex = _load()
    import cuda.bindings.driver as cuda
    import cutlass
    import cutlass.cute as cute
    import cutlass.utils as utils
    import torch

    tensors = ex.allocate_tensors(
        mnkl=(M, N, K, 1),
        ab_dtype=cutlass.BFloat16,
        c_dtype=cutlass.BFloat16,
        # Our A is (M,K) row-major, B is (K,N) row-major, C is (M,N) row-major.
        a_major="k",
        b_major="n",
        c_major="n",
        # One workspace: our bench reuses a single A/B pair, so rotating
        # buffers here would hand the example a different L2 profile.
        num_workspace=1,
        device=device,
        slot_init_mode="benchmark",
        global_rank=rank,
        local_rank=device,
        world_size=world_size,
    )

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
    )

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

    # allocate_tensors hands back _anchors that must outlive the closure --
    # they own the symmetric-memory allocations the compiled kernel points at.
    launch._keepalive = tensors
    return launch
