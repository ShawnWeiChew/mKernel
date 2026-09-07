import argparse
import importlib.util
import os
import random
import sys
import sysconfig
import time
from pathlib import Path
from typing import Callable

import torch
import torch.distributed as dist


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "python"))
import load_module  # noqa: E402
from common import check_close  # noqa: E402


GLOBAL_M = [2048, 4096, 8192, 16384, 32768]
K = 7168

IS_KDA = False

# Eight-way tensor parallel KDA projection width before kernel padding:
#   (4 * 12288 + 96) / 8 + 128 = 6284.
LOGICAL_N = 6284

DEFAULT_WARMUP = 5
DEFAULT_ITERS = 20
# Idle time between phases and between shapes. A sleep only resets temperature
# once the part is actually idle, so every cooldown drains the stream first.
DEFAULT_COOLDOWN_S = 5.0


########## CUTLASS compatibility layer ##########
_CUTLASS_ENV_ROOT = "CUTLASS_PATH"
_CUTLASS_ENV_EXAMPLE = "CUTLASS_AG_GEMM"
_CUTLASS_ENV_AUTOTUNE = "CUTLASS_AUTOTUNE"
_CUTLASS_RELATIVE_EXAMPLE = (
    "examples/python/CuTeDSL/cute/blackwell/kernel/distributed/"
    "distributed_all_gather_gemm_blackwell.py"
)

# The upstream example does not expose the swizzle/raster controls used by the
# CUTLASS GEMM+AR benchmark. Its useful schedule knob is the MMA N tile. Keep M
# and the cluster shape matched to this kernel's 2-CTA geometry, and tune the two
# N tiles this kernel itself dispatches between.
_CUTLASS_MMA_TILERS = ((256, 128), (256, 256))
_CUTLASS_CLUSTER_SHAPE = (2, 1)
_CUTLASS_MODULE = None


def _env_enabled(name: str, default: bool = True) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in ("", "0", "false", "no", "off")


def _cutlass_example_path() -> Path | None:
    override = os.environ.get(_CUTLASS_ENV_EXAMPLE)
    if override:
        return Path(override).expanduser()
    root = os.environ.get(_CUTLASS_ENV_ROOT)
    if root:
        return Path(root).expanduser() / _CUTLASS_RELATIVE_EXAMPLE
    return None


def _load_cutlass_example():
    global _CUTLASS_MODULE
    if _CUTLASS_MODULE is not None:
        return _CUTLASS_MODULE

    path = _cutlass_example_path()
    if path is None:
        raise RuntimeError(
            f"set {_CUTLASS_ENV_ROOT} to a CUTLASS checkout or "
            f"{_CUTLASS_ENV_EXAMPLE} to {_CUTLASS_RELATIVE_EXAMPLE.rsplit('/', 1)[-1]}"
        )
    if not path.is_file():
        raise FileNotFoundError(f"{path} does not exist")

    spec = importlib.util.spec_from_file_location(
        "cutlass_distributed_all_gather_gemm_blackwell", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "run"):
        raise AttributeError(f"{path} does not define run()")
    _CUTLASS_MODULE = module
    return module


def cutlass_availability() -> tuple[bool, str]:
    try:
        _load_cutlass_example()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def _run_cutlass_once(
    *,
    m: int,
    n: int,
    k: int,
    mma_tiler_mn: tuple[int, int],
    warmup: int,
    iterations: int,
) -> float:
    """Run one upstream configuration and return its max-rank time in ms.

    The pinned upstream helper owns its benchmark loop and returns microseconds.
    It also destroys the caller's process group before returning, which is right
    for its standalone CLI but not for an embedded benchmark. Temporarily make
    that teardown a no-op so all candidates and mKernel share one process group.
    """
    example = _load_cutlass_example()
    import contextlib
    import io

    import cutlass

    # Upstream's standalone __main__ defines this module global after parsing
    # torchrun's rank. Imported run() still references it when constructing
    # streams and walking the ring, but __main__ is not executed by our loader.
    # Supply the same value explicitly for the embedded path.
    example.local_rank = int(os.environ["LOCAL_RANK"])

    destroy_process_group = dist.destroy_process_group
    can_implement = example.PersistentDenseGemmKernel.can_implement

    def quiet_can_implement(*args, **kwargs):
        # Upstream prints MNKL unconditionally from every rank. It is not a
        # tuning result and obscures the per-candidate timing emitted below.
        with contextlib.redirect_stdout(io.StringIO()):
            return can_implement(*args, **kwargs)

    dist.destroy_process_group = lambda *args, **kwargs: None
    example.PersistentDenseGemmKernel.can_implement = quiet_can_implement
    try:
        time_us = example.run(
            mnkl=(m, n, k, 1),
            ab_dtype=cutlass.BFloat16,
            c_dtype=cutlass.BFloat16,
            acc_dtype=cutlass.Float32,
            a_major="k",
            b_major="n",
            c_major="n",
            mma_tiler_mn=mma_tiler_mn,
            cluster_shape_mn=_CUTLASS_CLUSTER_SHAPE,
            use_2cta_instrs=True,
            use_tma_store=True,
            warmup_iterations=warmup,
            iterations=iterations,
            skip_ref_check=True,
            use_cold_l2=False,
        )
    finally:
        example.PersistentDenseGemmKernel.can_implement = can_implement
        dist.destroy_process_group = destroy_process_group
    torch.cuda.synchronize()
    return float(time_us) / 1000.0


def matched_cutlass_tiler(m: int) -> tuple[int, int]:
    return (256, 128 if m < 4096 else 256)


def cutlass_benchmark(
    *, m: int, n: int, k: int, warmup: int, iterations: int
) -> tuple[float, tuple[int, int], list[tuple[tuple[int, int], float]]]:
    """Autotune CUTLASS's MMA tile and benchmark the winning configuration."""
    autotune = _env_enabled(_CUTLASS_ENV_AUTOTUNE)
    candidates = _CUTLASS_MMA_TILERS if autotune else (matched_cutlass_tiler(m),)
    tune_warmup = min(2, warmup)
    tune_iterations = min(5, iterations)

    timings = []
    best_tiler = None
    best_ms = float("inf")
    for tiler in candidates:
        ms = _run_cutlass_once(
            m=m,
            n=n,
            k=k,
            mma_tiler_mn=tiler,
            warmup=tune_warmup if autotune else warmup,
            iterations=tune_iterations if autotune else iterations,
        )
        timings.append((tiler, ms))
        if ms < best_ms:
            best_tiler, best_ms = tiler, ms

    assert best_tiler is not None
    if autotune:
        # The upstream run() builds a private CUDA graph, so it cannot hand its
        # tuned launcher back to us. Rebuild the winner for the full benchmark
        # sample rather than reporting the short tuning measurement.
        best_ms = _run_cutlass_once(
            m=m,
            n=n,
            k=k,
            mma_tiler_mn=best_tiler,
            warmup=warmup,
            iterations=iterations,
        )
    return best_ms, best_tiler, timings


########## ThunderKittens compatibility layer ##########
_TK_ENV_ROOT = "THUNDERKITTENS_PATH"
_TK_ENV_EXTENSION = "THUNDERKITTENS_AG_GEMM"
_TK_ENV_AUTOTUNE = "THUNDERKITTENS_AUTOTUNE"
_TK_ENV_COMM_SMS = "THUNDERKITTENS_NUM_COMM_SMS"
_TK_RELATIVE_DIR = Path("kernels/parallel/ag_gemm")
_TK_COMM_SMS = (2, 4, 8, 16, 32, 64)
_TK_MODULE = None


def _tk_extension_path() -> Path | None:
    override = os.environ.get(_TK_ENV_EXTENSION)
    if override:
        path = Path(override).expanduser()
        if path.is_file() and path.suffix != ".cu":
            return path
        search_dir = path.parent if path.suffix == ".cu" else path
    else:
        root = os.environ.get(_TK_ENV_ROOT)
        if not root:
            return None
        root_path = Path(root).expanduser()
        nested = root_path / _TK_RELATIVE_DIR
        search_dir = nested if nested.is_dir() else root_path

    extension_suffix = sysconfig.get_config_var("EXT_SUFFIX")
    preferred = search_dir / f"_C{extension_suffix}"
    if preferred.is_file():
        return preferred
    candidates = sorted(
        search_dir.glob("_C*.so"), key=lambda path: path.stat().st_mtime_ns
    )
    return candidates[-1] if candidates else preferred


def _load_tk_extension():
    global _TK_MODULE
    if _TK_MODULE is not None:
        return _TK_MODULE

    path = _tk_extension_path()
    if path is None:
        raise RuntimeError(
            f"set {_TK_ENV_ROOT} to a ThunderKittens checkout or "
            f"{_TK_ENV_EXTENSION} to its built AG-GEMM extension"
        )
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} does not exist; build the GPU-compatible ThunderKittens "
            "kernels/parallel/ag_gemm/_C extension first"
        )

    # ThunderKittens names the pybind module `_C`, so the import spec must keep
    # that final component even when the shared object has an ABI-tagged name.
    spec = importlib.util.spec_from_file_location("_C", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create an import spec for {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for symbol in ("TKParallelTensor", "all_gather_matmul"):
        if not hasattr(module, symbol):
            raise AttributeError(f"{path} does not define {symbol}")
    _TK_MODULE = module
    return module


def tk_availability(world_size: int) -> tuple[bool, str]:
    if world_size != 8:
        return False, "upstream ag_gemm_b200.cu hardcodes NUM_DEVICES=8"
    try:
        _load_tk_extension()
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    return True, ""


def make_tk_state(
    *,
    m: int,
    k: int,
    n: int,
    local_rank: int,
    world_size: int,
    a_local: torch.Tensor,
    b: torch.Tensor,
):
    module = _load_tk_extension()
    a = module.TKParallelTensor(
        (m, k),
        dtype=torch.bfloat16,
        local_rank=local_rank,
        local_world_size=world_size,
        multicast=True,
    )
    local_m = m // world_size
    a.data_[local_rank * local_m : (local_rank + 1) * local_m].copy_(a_local)

    # The Blackwell TK kernel declares B as [N, K] and performs AB^T.
    b_transposed = b.T.contiguous()
    c = torch.zeros((m, n), dtype=torch.bfloat16, device=a_local.device)
    barrier = module.TKParallelTensor(
        (2, 1024, 1024),
        dtype=torch.int,
        local_rank=local_rank,
        local_world_size=world_size,
        multicast=True,
    )
    barrier.data_.zero_()
    return {
        "module": module,
        "a": a,
        "b": b_transposed,
        "c": c,
        "barrier": barrier,
    }


def launch_tk(state, num_comm_sms: int) -> None:
    state["module"].all_gather_matmul(
        state["a"],
        state["b"],
        state["c"],
        state["barrier"],
        num_comm_sms,
    )


def configured_tk_comm_sms() -> int:
    value = int(os.environ.get(_TK_ENV_COMM_SMS, "8"))
    if value <= 0 or value >= 148 or value % 2:
        raise ValueError(
            f"{_TK_ENV_COMM_SMS} must be a positive even integer below 148"
        )
    return value


def tune_tk_comm_sms(
    state, *, warmup: int, iterations: int
) -> tuple[int, list[tuple[int, float]]]:
    autotune = _env_enabled(_TK_ENV_AUTOTUNE)
    candidates = _TK_COMM_SMS if autotune else (configured_tk_comm_sms(),)
    tune_warmup = min(2, warmup)
    tune_iterations = min(5, iterations)
    timings = []
    best_comm_sms = None
    best_ms = float("inf")
    for num_comm_sms in candidates:
        ms = benchmark_cuda(
            lambda num_comm_sms=num_comm_sms: launch_tk(state, num_comm_sms),
            tune_warmup if autotune else warmup,
            tune_iterations if autotune else iterations,
        )
        timings.append((num_comm_sms, ms))
        if ms < best_ms:
            best_comm_sms, best_ms = num_comm_sms, ms
    assert best_comm_sms is not None
    return best_comm_sms, timings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Correctness and performance test for ag_gemm_kda_mla"
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=DEFAULT_WARMUP,
        help="warmup iterations per implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--iters",
        type=int,
        default=DEFAULT_ITERS,
        help="timed iterations per implementation (default: %(default)s)",
    )
    parser.add_argument(
        "--cooldown",
        type=float,
        default=DEFAULT_COOLDOWN_S,
        help="idle seconds between phases and shapes (default: %(default)s)",
    )
    parser.add_argument(
        "--shape-seed",
        type=int,
        default=0,
        help="seed for the benchmark shape order; -1 keeps the declared "
             "ascending order (default: %(default)s)",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.iters <= 0:
        parser.error("--iters must be positive")
    return args


def round_up(value: int, multiple: int) -> int:
    return (value + multiple - 1) // multiple * multiple


def gemm_tflops(m: int, n: int, k: int, elapsed_ms: float) -> float:
    """Return GEMM throughput using 2*M*N*K floating-point operations."""
    return 2.0 * m * n * k / (elapsed_ms * 1.0e9)


def padded_n_for_m(m: int, logical_n: int) -> int:
    col_block = 128 if m < 4096 else 256
    return round_up(logical_n, col_block)


def benchmark_cuda(
    run_once: Callable[[], None], warmup: int, iters: int
) -> float:
    """Return average CUDA time in ms, taking the slowest rank's result."""
    for _ in range(warmup):
        run_once()

    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        run_once()
    end.record()
    end.synchronize()
    local_ms = start.elapsed_time(end) / iters

    # End-to-end distributed latency is gated by the slowest rank. This
    # reduction is outside the timed region for both implementations.
    rank_ms = torch.tensor(local_ms, device="cuda", dtype=torch.float64)
    dist.all_reduce(rank_ms, op=dist.ReduceOp.MAX)
    dist.barrier()
    return float(rank_ms.item())


def williams_orders(items):
    """Balanced Latin square (Williams design) over `items`.

    Same construction as bench/gemm_ar_blackwell_bench.py. Returns len(items)
    orderings in which every condition occupies every position exactly once, so
    no implementation is systematically measured first (cold clock) or last (hot
    part). Every row is the first row shifted, which keeps the order identical
    on every rank -- it has to be, because each condition is collective.
    """
    n = len(items)
    first, lo, hi = [], 0, n - 1
    while lo <= hi:
        first.append(lo)
        if lo != hi:
            first.append(hi)
        lo, hi = lo + 1, hi - 1
    return [tuple(items[(v + r) % n] for v in first) for r in range(n)]


def sync_ranks() -> None:
    """Drain the local stream, then line every rank up on the host.

    Each timed iteration then starts from an idle stream on every rank. Without
    it a back-to-back loop hides launch overhead behind the queue and lets ranks
    self-synchronize, which flatters whichever condition is measured that way.
    """
    torch.cuda.synchronize()
    dist.barrier()


def cooldown(seconds: float) -> None:
    """Drain, line the ranks up, then idle so the next phase starts cool."""
    torch.cuda.synchronize()
    dist.barrier()
    if seconds > 0:
        time.sleep(seconds)


def elapsed_ms(samples):
    """Drain (start, end) cuda event pairs into per-iter wall times (ms)."""
    return [s.elapsed_time(e) for s, e in samples]


def median_then_max(samples) -> float:
    """Median over iterations per rank, then the slowest rank.

    Median rather than mean because a single descheduled iteration should not
    move the number; max across ranks because end-to-end latency is gated by
    the slowest rank.
    """
    ordered = sorted(float(x) for x in samples)
    median = ordered[len(ordered) // 2]
    t = torch.tensor([median], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return float(t.item())


def bench_shape_order(shapes, seed: int):
    """Shapes in a fixed but non-monotonic order, identical on every rank.

    Running 2048 -> 32768 in ascending order confounds size with thermal state:
    the largest shape is always measured on the hottest part. A seeded shuffle
    breaks that correlation while staying reproducible across runs and ranks.
    """
    if seed < 0:
        return list(shapes)
    ordered = list(shapes)
    random.Random(seed).shuffle(ordered)
    return ordered


def main() -> int:
    # The four-rank configuration selects a different projection width.
    global LOGICAL_N
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_world_size = int(
        os.environ.get("LOCAL_WORLD_SIZE", os.environ["WORLD_SIZE"])
    )
    torch.cuda.set_device(local_rank)

    dist.init_process_group(
        "nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    world_size = dist.get_world_size()
    is_chief = rank == 0

    assert world_size == 4 or world_size == 8, f"{world_size=} is not 4 or 8"

    if IS_KDA:
        LOGICAL_N = (4 * 12288 + 96) // world_size + 128
    else:
        LOGICAL_N = 576 + 1536 + 12288 // world_size

    if local_world_size != world_size:
        raise RuntimeError(
            "ag_gemm_kda_mla is an intra-node test and requires "
            "LOCAL_WORLD_SIZE == WORLD_SIZE"
        )

    mod = load_module.load("ag_gemm_kda_mla")
    all_correct = True

    for m in GLOBAL_M:
        if m % world_size != 0:
            raise ValueError(f"global M={m} is not divisible by {world_size=}")

        local_m = m // world_size
        padded_n = padded_n_for_m(m, LOGICAL_N)

        # Reference tensors retain the original, unpadded problem shapes.
        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        )

        dist.all_gather_into_tensor(A_ref, A_ref_local)
        torch.mm(A_ref, B_ref, out=C_ref)

        # The modified implementation gets its own tensors. A already meets
        # the kernel's K/row requirements, so only B and C need N padding.
        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)
        A_local_buf = torch.empty(
            (m, K), device="cuda", dtype=torch.bfloat16
        )

        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :LOGICAL_N].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        dist.barrier()
        mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B_kernel, C_kernel)
        torch.cuda.synchronize()

        # Ignore the padded output columns and compare the original 6284-wide
        # result against the unpadded PyTorch reference.
        is_correct = check_close(
            f"ag-gemm-kda-mla M={m} N={LOGICAL_N} padded_n={padded_n}",
            C_kernel[:, :LOGICAL_N],
            C_ref,
        )
        all_correct = all_correct and is_correct

        if is_chief:
            status = "passed :)" if is_correct else "FAILED :("
            print(
                f"M={m} local_m={local_m} N={LOGICAL_N} "
                f"padded_n={padded_n}: {status}",
                flush=True,
            )

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, A_local_buf, B_kernel, C_kernel
        dist.barrier()

    if not all_correct:
        if is_chief:
            print("Correctness checks failed; skipping benchmarks.", flush=True)
        dist.destroy_process_group()
        return 1

    if is_chief:
        print(
            f"All correctness checks passed. Benchmarking with "
            f"warmup={args.warmup}, iters={args.iters}...",
            flush=True,
        )

    cutlass_ok, cutlass_why = cutlass_availability()
    cutlass_vote = torch.tensor([1 if cutlass_ok else 0], device="cuda")
    dist.all_reduce(cutlass_vote, op=dist.ReduceOp.MIN)
    cutlass_ok = bool(cutlass_vote.item())
    if not cutlass_ok and is_chief:
        print(
            f"[skip] CUTLASS all-gather GEMM: "
            f"{cutlass_why or 'unavailable on a peer'}",
            flush=True,
        )

    tk_ok, tk_why = tk_availability(world_size)
    tk_vote = torch.tensor([1 if tk_ok else 0], device="cuda")
    dist.all_reduce(tk_vote, op=dist.ReduceOp.MIN)
    tk_ok = bool(tk_vote.item())
    if not tk_ok and is_chief:
        print(
            f"[skip] ThunderKittens all-gather GEMM: "
            f"{tk_why or 'unavailable on a peer'}",
            flush=True,
        )

    # Allocate fresh tensors for the benchmark pass so no performance result
    # is emitted until the complete correctness suite has passed.
    bench_shapes = bench_shape_order(GLOBAL_M, args.shape_seed)
    if is_chief:
        order = " ".join(str(x) for x in bench_shapes)
        streams = getattr(mod, "A_COPY_STREAMS", None)
        print(f"\nbenchmark shape order (seed {args.shape_seed}): {order}\n"
              f"cooldown {args.cooldown:g}s between phases and shapes"
              + (f" | A copy streams: {streams}" if streams else ""),
              flush=True)
    for m in bench_shapes:
        local_m = m // world_size
        padded_n = padded_n_for_m(m, LOGICAL_N)

        cutlass_ms = None
        cutlass_tiler = None
        cutlass_tune_log = []
        if cutlass_ok:
            # CUTLASS owns its loop: upstream run() captures a graph and replays
            # it back to back, so it cannot join the rotation below. Bracket it
            # with the same cooldowns at least, and see kernel_b2b_ms for the
            # like-for-like number.
            cooldown(args.cooldown)
            try:
                cutlass_ms, cutlass_tiler, cutlass_tune_log = cutlass_benchmark(
                    m=m,
                    n=padded_n,
                    k=K,
                    warmup=args.warmup,
                    iterations=args.iters,
                )
            except Exception as exc:
                cutlass_why = f"{type(exc).__name__}: {exc}"

            shape_vote = torch.tensor(
                [1 if cutlass_ms is not None else 0], device="cuda"
            )
            dist.all_reduce(shape_vote, op=dist.ReduceOp.MIN)
            if not shape_vote.item():
                cutlass_ms = None
                if is_chief:
                    print(
                        f"  [skip] CUTLASS M={m} N={padded_n}: "
                        f"{cutlass_why or 'failed on a peer'}",
                        flush=True,
                    )
            elif is_chief:
                tune_note = (
                    "autotuned"
                    if _env_enabled(_CUTLASS_ENV_AUTOTUNE)
                    else "matched to mKernel"
                )
                print(
                    f"  CUTLASS config M={m}: mma_tiler_mn={cutlass_tiler} "
                    f"cluster_shape_mn={_CUTLASS_CLUSTER_SHAPE} ({tune_note})",
                    flush=True,
                )
                if _env_enabled(_CUTLASS_ENV_AUTOTUNE):
                    for tiler, tune_ms in sorted(
                        cutlass_tune_log, key=lambda item: item[1]
                    ):
                        mark = " <- best" if tiler == cutlass_tiler else ""
                        if tiler == matched_cutlass_tiler(m):
                            mark += " (matches mKernel tile)"
                        print(
                            f"    [autotune] mma_tiler_mn={tiler}: "
                            f"{tune_ms:8.3f} ms  "
                            f"{gemm_tflops(m, padded_n, K, tune_ms):8.2f} "
                            f"TFLOP/s{mark}",
                            flush=True,
                        )

        torch.manual_seed(42 + rank)
        torch.cuda.manual_seed(42 + rank)
        A_ref_local = torch.randn(
            (local_m, K), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        A_ref = torch.empty((m, K), device="cuda", dtype=torch.bfloat16)
        B_ref = torch.randn(
            (K, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        ) / (K**0.25)
        C_ref = torch.empty(
            (m, LOGICAL_N), device="cuda", dtype=torch.bfloat16
        )

        A_kernel = mod.DistBuffer(
            (local_m, K),
            dtype=torch.bfloat16,
            local_rank=local_rank,
            local_world_size=local_world_size,
            multicast=True,
        )
        A_kernel.data_.copy_(A_ref_local)
        A_local_buf = torch.empty(
            (m, K), device="cuda", dtype=torch.bfloat16
        )
        B_kernel = torch.zeros(
            (K, padded_n), device="cuda", dtype=torch.bfloat16
        )
        B_kernel[:, :LOGICAL_N].copy_(B_ref)
        C_kernel = torch.zeros(
            (m, padded_n), device="cuda", dtype=torch.bfloat16
        )

        def run_all_gather() -> None:
            dist.all_gather_into_tensor(A_ref, A_ref_local)

        def run_cublas_logical() -> None:
            torch.mm(A_ref, B_ref, out=C_ref)

        def run_cublas_padded() -> None:
            torch.mm(A_ref, B_kernel, out=C_kernel)

        def run_baseline_logical() -> None:
            run_all_gather()
            run_cublas_logical()

        def run_baseline_padded() -> None:
            run_all_gather()
            run_cublas_padded()

        def run_kernel() -> None:
            mod.ag_gemm_kda_mla(A_kernel, A_local_buf, B_kernel, C_kernel)

        tk_state = None
        tk_comm_sms = None
        tk_tune_log = []
        if tk_ok:
            try:
                tk_state = make_tk_state(
                    m=m,
                    k=K,
                    n=padded_n,
                    local_rank=local_rank,
                    world_size=world_size,
                    a_local=A_ref_local,
                    b=B_kernel,
                )
            except Exception as exc:
                tk_why = f"{type(exc).__name__}: {exc}"

            shape_vote = torch.tensor(
                [1 if tk_state is not None else 0], device="cuda"
            )
            dist.all_reduce(shape_vote, op=dist.ReduceOp.MIN)
            if not shape_vote.item():
                tk_state = None
                if is_chief:
                    print(
                        f"  [skip] ThunderKittens M={m} N={padded_n}: "
                        f"{tk_why or 'failed on a peer'}",
                        flush=True,
                    )
            else:
                # ParallelKittens requires all ranks to finish initializing its
                # multicast barrier before the first fused launch.
                dist.barrier()
                check_comm_sms = configured_tk_comm_sms()
                run_all_gather()
                run_cublas_logical()
                launch_tk(tk_state, check_comm_sms)
                torch.cuda.synchronize()
                if not check_close(
                    f"ThunderKittens AG-GEMM M={m}",
                    tk_state["c"][:, :LOGICAL_N],
                    C_ref,
                ):
                    raise RuntimeError(
                        f"ThunderKittens correctness failed at M={m} N={padded_n}"
                    )

                tk_comm_sms, tk_tune_log = tune_tk_comm_sms(
                    tk_state,
                    warmup=args.warmup,
                    iterations=args.iters,
                )
                if is_chief:
                    tune_note = (
                        "autotuned"
                        if _env_enabled(_TK_ENV_AUTOTUNE)
                        else f"set by {_TK_ENV_COMM_SMS}"
                    )
                    print(
                        f"  ThunderKittens config M={m}: "
                        f"num_comm_sms={tk_comm_sms} ({tune_note})",
                        flush=True,
                    )
                    if _env_enabled(_TK_ENV_AUTOTUNE):
                        for comm_sms, tune_ms in sorted(
                            tk_tune_log, key=lambda item: item[1]
                        ):
                            mark = " <- best" if comm_sms == tk_comm_sms else ""
                            print(
                                f"    [autotune] num_comm_sms={comm_sms}: "
                                f"{tune_ms:8.3f} ms  "
                                f"{gemm_tflops(m, padded_n, K, tune_ms):8.2f} "
                                f"TFLOP/s{mark}",
                                flush=True,
                            )

        # Interleaved measurement, as in bench/gemm_ar_blackwell_bench.py.
        # Measuring each condition to completion in turn confounds the
        # implementation with the thermal and clock state it happened to run in:
        # the first one measured gets a cold part, the last one a soaked one. A
        # balanced rotation gives every condition every position instead.
        launchers = {
            "all_gather": run_all_gather,
            "cublas_logical": run_cublas_logical,
            "cublas_padded": run_cublas_padded,
            "baseline_logical": run_baseline_logical,
            "baseline_padded": run_baseline_padded,
            "kernel": run_kernel,
        }
        if tk_state is not None:
            launchers["tk"] = lambda: launch_tk(tk_state, tk_comm_sms)
        conditions = tuple(launchers)

        ORDERS = williams_orders(conditions)
        # A whole number of rotations, so the balancing is exact rather than
        # approximate -- the last partial rotation would favour whatever sits
        # early in it.
        iterations = max(1, round(args.iters / len(ORDERS))) * len(ORDERS)

        cooldown(args.cooldown)
        # Warm on the rotation the timed loop uses, so no condition pays another
        # condition's cold start once measurement begins.
        for it in range(args.warmup):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                launchers[cond]()
        cooldown(args.cooldown)

        samples = {c: [] for c in conditions}
        for it in range(iterations):
            for cond in ORDERS[it % len(ORDERS)]:
                sync_ranks()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record()
                launchers[cond]()
                e_ev.record()
                samples[cond].append((s_ev, e_ev))

        # Events only read back once the stream has drained.
        torch.cuda.synchronize()
        dist.barrier()
        timings = {c: median_then_max(elapsed_ms(samples[c])) for c in conditions}

        all_gather_ms = timings["all_gather"]
        cublas_logical_ms = timings["cublas_logical"]
        cublas_padded_ms = timings["cublas_padded"]
        baseline_logical_ms = timings["baseline_logical"]
        baseline_padded_ms = timings["baseline_padded"]
        kernel_ms = timings["kernel"]
        tk_ms = timings.get("tk")

        # Same kernel, measured the way CUTLASS measures itself: warm up, then
        # time N launches back to back with no per-iteration sync. That hides
        # launch overhead behind the queue, so the difference from kernel_ms is
        # how much of this kernel's cost is exposed per launch rather than
        # absorbed by a pipelined loop -- and it is the number to compare
        # against CUTLASS's graph-replay timing.
        cooldown(args.cooldown)
        kernel_b2b_ms = benchmark_cuda(run_kernel, args.warmup, args.iters)
        relative_performance = baseline_padded_ms / kernel_ms
        logical_relative_performance = baseline_logical_ms / kernel_ms

        if is_chief:
            print(
                f"M={m} local_m={local_m} N={LOGICAL_N} "
                f"padded_n={padded_n}\n"
                f"  {'NCCL all-gather':<26} {all_gather_ms:8.3f} ms\n"
                f"  {f'cuBLAS N={LOGICAL_N}':<26} {cublas_logical_ms:8.3f} ms  "
                f"{gemm_tflops(m, LOGICAL_N, K, cublas_logical_ms):8.2f} "
                f"TFLOP/s\n"
                f"  {f'cuBLAS N={padded_n}':<26} {cublas_padded_ms:8.3f} ms  "
                f"{gemm_tflops(m, padded_n, K, cublas_padded_ms):8.2f} "
                f"TFLOP/s\n"
                f"  {f'cuBLAS + NCCL N={LOGICAL_N}':<26} "
                f"{baseline_logical_ms:8.3f} ms  "
                f"{gemm_tflops(m, LOGICAL_N, K, baseline_logical_ms):8.2f} "
                f"TFLOP/s\n"
                f"  {f'cuBLAS + NCCL N={padded_n}':<26} "
                f"{baseline_padded_ms:8.3f} ms  "
                f"{gemm_tflops(m, padded_n, K, baseline_padded_ms):8.2f} "
                f"TFLOP/s  (matched baseline)",
                flush=True,
            )
            if cutlass_ms is not None:
                print(
                    f"  {'CUTLASS AG-GEMM':<26} {cutlass_ms:8.3f} ms  "
                    f"{gemm_tflops(m, padded_n, K, cutlass_ms):8.2f} TFLOP/s  "
                    f"({baseline_padded_ms / cutlass_ms:6.3f}x vs matched)",
                    flush=True,
                )
            if tk_ms is not None:
                print(
                    f"  {'ThunderKittens AG-GEMM':<26} {tk_ms:8.3f} ms  "
                    f"{gemm_tflops(m, padded_n, K, tk_ms):8.2f} TFLOP/s  "
                    f"({baseline_padded_ms / tk_ms:6.3f}x vs matched)",
                    flush=True,
                )
            kernel_line = (
                f"  {'ag_gemm_kda_mla':<26} {kernel_ms:8.3f} ms  "
                f"{gemm_tflops(m, padded_n, K, kernel_ms):8.2f} TFLOP/s  "
                f"({relative_performance:6.3f}x vs matched, "
                f"{logical_relative_performance:6.3f}x vs logical)"
            )
            if cutlass_ms is not None:
                verdict = "BEATS" if kernel_ms < cutlass_ms else "behind"
                kernel_line += (
                    f"  {cutlass_ms / kernel_ms:6.3f}x vs CUTLASS "
                    f"({verdict})"
                )
            if tk_ms is not None:
                verdict = "BEATS" if kernel_ms < tk_ms else "behind"
                kernel_line += (
                    f"  {tk_ms / kernel_ms:6.3f}x vs ThunderKittens "
                    f"({verdict})"
                )
            print(kernel_line, flush=True)
            b2b_line = (
                f"  {'ag_gemm_kda_mla (b2b)':<26} {kernel_b2b_ms:8.3f} ms  "
                f"{gemm_tflops(m, padded_n, K, kernel_b2b_ms):8.2f} TFLOP/s  "
                f"(back-to-back, no per-iter sync"
            )
            if cutlass_ms is not None:
                verdict = "BEATS" if kernel_b2b_ms < cutlass_ms else "behind"
                b2b_line += (
                    f"; {cutlass_ms / kernel_b2b_ms:6.3f}x vs CUTLASS "
                    f"({verdict}), like-for-like"
                )
            print(b2b_line + ")", flush=True)

        del A_ref_local, A_ref, B_ref, C_ref
        del A_kernel, A_local_buf, B_kernel, C_kernel
        tk_state = None
        # Between shapes, not just between phases: the next shape should not
        # start on the heat this one left behind.
        cooldown(args.cooldown)

    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
