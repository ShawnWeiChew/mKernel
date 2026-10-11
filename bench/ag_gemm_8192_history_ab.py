"""Local historical AG-GEMM comparison; see ag_gemm_8192_history_ab.md.

Builds unmodified 47dd7ea and f21b220 snapshots, then runs only
M=8192, N=3648, K=7168 on eight local GPUs. No remote execution.
"""
from __future__ import annotations

import argparse
from datetime import timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]
REVISIONS = {
    "before": "55fda5ccd855b8478e3c1c0f68025459e8213f38",
    "after": "f21b22075e76c2036456de4425d2a2779de24de4",
}
M, N, K, WORLD = 8192, 3648, 7168, 8


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gpu_status():
    try:
        return subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name,uuid,clocks.sm,clocks.mem,"
             "temperature.gpu,power.draw,utilization.gpu,memory.used",
             "--format=csv"], text=True, timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def run_logged(command, log, *, cwd=None):
    log.with_suffix(".command.json").write_text(json.dumps(command, indent=2) + "\n")
    with log.open("w") as stream:
        result = subprocess.run(command, cwd=cwd, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        print(log.read_text()[-12000:], file=sys.stderr)
        raise RuntimeError(f"command failed ({result.returncode}); see {log}")


def build_snapshots(args, output):
    manifest = {"shape": {"m": M, "n": N, "k": K}, "world_size": WORLD,
                "arch": args.arch, "python": sys.executable,
                "harness_sha256": sha256(Path(__file__)), "variants": {}}
    for name, revision in REVISIONS.items():
        snapshot = output / name
        snapshot.mkdir()
        archive = subprocess.check_output(
            ["git", "archive", revision, "Makefile", "include",
             "src/ag_gemm_warp_specialized.cu"], cwd=ROOT,
        )
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(snapshot, filter="data")
        compute = args.arch.replace("sm_", "compute_")
        command = [
            "make", "GPU=blackwell", f"CUDA_HOME={args.cuda_home}",
            f"PYTHON={sys.executable}", "INTRA_NUM_DEVICES=8",
            f"ARCH=-gencode arch={compute},code={args.arch}",
            # Link-only CUDA driver stub search path also permits GPU-less builds.
            # Do not add it to runtime rpath / LD_LIBRARY_PATH.
            f"BACKEND_LIBS=-L{args.cuda_home}/lib64/stubs",
            "ag-gemm-warp-specialized",
        ]
        print(f"Building {name}: {revision[:7]} (log: {output / (name + '-build.log')})", flush=True)
        run_logged(command, output / f"{name}-build.log", cwd=snapshot)
        library = snapshot / "build/libag_gemm_warp_specialized.so"
        manifest["variants"][name] = {
            "revision": revision, "library": str(library), "sha256": sha256(library),
        }
    manifest["nvcc"] = subprocess.check_output(
        [str(args.cuda_home / "bin/nvcc"), "--version"], text=True,
    ).strip()
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def summarize(trials):
    medians = {
        name: statistics.median(t["median_ms"] for t in trials if t["variant"] == name)
        for name in REVISIONS
    }
    rounds = sorted({t["round"] for t in trials})
    ratios = []
    for round_idx in rounds:
        means = {name: statistics.mean(t["median_ms"] for t in trials
                                      if t["round"] == round_idx and t["variant"] == name)
                 for name in REVISIONS}
        ratios.append(means["before"] / means["after"])
    return {"median_trial_ms": medians,
            "before_over_after_ratio": medians["before"] / medians["after"],
            "paired_round_ratios": ratios}


def run_worker(args):
    # Import the exact library directly. Avoid the normal loader's selection of
    # newer torch-tagged binaries, and never load both revisions in one process.
    os.environ["MKERNEL_BIND_RETAINED_HANDLE"] = "1"
    import torch
    import torch.distributed as dist

    local_rank = int(os.environ["LOCAL_RANK"])
    if int(os.environ["WORLD_SIZE"]) != WORLD or int(os.environ["LOCAL_WORLD_SIZE"]) != WORLD:
        raise RuntimeError("this experiment requires eight ranks on one local node")
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl", timeout=timedelta(seconds=180),
                            device_id=torch.device(f"cuda:{local_rank}"))
    try:
        module_name = "mkernel_release_ag_gemm_warp_specialized"
        spec = importlib.util.spec_from_file_location(module_name, args.library)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = mod
        spec.loader.exec_module(mod)
        rank = dist.get_rank()

        def synchronize():
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.synchronize()

        def check(c, ref, label):
            # Same BF16 acceptance criteria as bench/common.py, applied equally
            # to both revisions; the full benchmark harness is otherwise fixed.
            observed, expected = c.view(M, N).float(), ref.float()
            error = (observed - expected).abs()
            max_abs = error.max().item()
            mean_abs = error.mean().item()
            ref_mean = expected.abs().mean().item()
            max_rel = (error / expected.abs().clamp_min(1e-6)).max().item()
            passed = (max_abs <= 0.35 or max_rel <= 0.08
                      or mean_abs <= 0.01 * max(ref_mean, 1e-6))
            vote = torch.tensor(int(passed), device="cuda")
            dist.all_reduce(vote, op=dist.ReduceOp.MIN)
            if not vote.item():
                raise RuntimeError(f"{label} correctness failed, rank={rank}, "
                                   f"max_abs={max_abs}, mean_abs={mean_abs}")
            if rank == 0:
                print(f"{label}: PASS", flush=True)

        a = mod.DistBuffer((WORLD, M // WORLD, K), dtype=torch.bfloat16,
                           local_rank=rank, local_world_size=WORLD, multicast=True)
        a.data_.zero_()
        b = torch.empty((N, K), device="cuda", dtype=torch.bfloat16)
        c = torch.empty((WORLD, M // WORLD, N), device="cuda", dtype=torch.bfloat16)
        ready = None
        if args.variant == "after":
            if mod.ag_gemm_warp_specialized_ready_words != WORLD + 1:
                raise RuntimeError("unexpected epoch allocation size")
            ready = mod.DistBuffer((mod.ag_gemm_warp_specialized_ready_words,),
                                  dtype=torch.int32, local_rank=rank,
                                  local_world_size=WORLD, multicast=True)
            ready.data_.zero_()
        synchronize()

        def fill_inputs(seed):
            # Baseline has no entry handshake: never rewrite source shards until
            # every rank finishes, and keep them fixed throughout timed batches.
            synchronize()
            torch.manual_seed(seed + rank)
            local_a = torch.randn((M // WORLD, K), device="cuda", dtype=torch.bfloat16) / K ** 0.25
            a.data_[rank].copy_(local_a)
            torch.manual_seed(seed + 10000)
            b.copy_(torch.randn_like(b) / K ** 0.25)
            full_a = torch.empty((M, K), device="cuda", dtype=torch.bfloat16)
            dist.all_gather_into_tensor(full_a, local_a)
            ref = full_a @ b.T
            synchronize()
            return ref

        def launch():
            if args.variant == "before":
                mod.ag_gemm_warp_specialized(a, b, c, M)
            else:
                mod.ag_gemm_warp_specialized_prepare(ready, M, N)
                mod.ag_gemm_warp_specialized_launch(a, ready, b, c)

        reference = fill_inputs(42)
        c.fill_(float("nan"))
        launch()  # Initialize copy-stream resources before graph capture.
        synchronize()
        check(c, reference, "eager")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            launch()  # Exactly one AG-GEMM invocation per replay, both versions.
        synchronize()
        reference = fill_inputs(123)
        c.fill_(float("nan"))
        for _ in range(16):
            graph.replay()
        synchronize()
        check(c, reference, "graph replay with changed inputs")

        samples = []
        telemetry_before = gpu_status() if rank == 0 else None
        for sample in range(args.samples):
            for _ in range(args.warmup):
                graph.replay()
            synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.iters):
                graph.replay()
            end.record()
            end.synchronize()
            local_ms = start.elapsed_time(end) / args.iters
            rank_ms = [None] * WORLD
            dist.all_gather_object(rank_ms, local_ms)
            samples.append({"ms": max(rank_ms), "rank_ms": rank_ms})
            if rank == 0:
                print(f"{args.variant} sample {sample + 1}: {max(rank_ms) * 1000:.3f} us", flush=True)
        synchronize()
        check(c, reference, "after timing")
        if rank == 0:
            times = [s["ms"] for s in samples]
            median = statistics.median(times)
            result = {
                "variant": args.variant, "revision": REVISIONS[args.variant],
                "shape": {"m": M, "n": N, "k": K}, "world_size": WORLD,
                "library": str(args.library), "library_sha256": sha256(args.library),
                "torch": torch.__version__, "cuda": torch.version.cuda,
                "device": torch.cuda.get_device_name(), "warmup": args.warmup,
                "iters": args.iters, "samples": samples, "median_ms": median,
                "mad_ms": statistics.median(abs(t - median) for t in times),
                "min_ms": min(times), "max_ms": max(times),
                "gpu_status_before": telemetry_before, "gpu_status_after": gpu_status(),
            }
            args.result.write_text(json.dumps(result, indent=2) + "\n")
        synchronize()  # All peer reads finish before source buffers are freed.
    finally:
        dist.destroy_process_group()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cuda-home", type=Path,
                   default=Path(os.environ.get("CUDA_HOME", "/usr/local/cuda-13.2")))
    p.add_argument("--arch", choices=("sm_103a", "sm_100a"), default="sm_103a")
    p.add_argument("--output-dir", type=Path, help="new directory for snapshots, logs and JSON")
    p.add_argument("--build-only", action="store_true", help="compile locally without GPU execution")
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--warmup", type=int, default=100)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--variant", choices=tuple(REVISIONS), help=argparse.SUPPRESS)
    p.add_argument("--library", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--result", type=Path, help=argparse.SUPPRESS)
    args = p.parse_args()
    if min(args.rounds, args.samples, args.warmup, args.iters) <= 0:
        p.error("rounds, samples, warmup and iters must be positive")
    if args.worker:
        if args.variant is None or args.library is None or args.result is None:
            p.error("worker requires variant, library and result")
        run_worker(args)
        return
    if args.output_dir:
        output = args.output_dir.resolve()
        output.mkdir(parents=True, exist_ok=False)
    else:
        (ROOT / "build").mkdir(exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="ag_gemm_8192_ab-", dir=ROOT / "build"))
    print(f"Experiment directory: {output}", flush=True)
    manifest = build_snapshots(args, output)
    if args.build_only:
        print("Both revisions built; no GPU execution requested.", flush=True)
        return
    results = {"manifest": manifest, "warmup": args.warmup, "iters": args.iters,
               "samples_per_trial": args.samples, "trials": []}
    for round_idx in range(args.rounds):
        order = ("before", "after", "after", "before")
        if round_idx % 2:
            order = ("after", "before", "before", "after")
        for position, variant in enumerate(order):
            tag = f"trial-{round_idx + 1}-{position + 1}-{variant}"
            result_path = output / f"{tag}.json"
            command = [sys.executable, "-m", "torch.distributed.run", "--standalone",
                       "--nproc-per-node=8", str(Path(__file__).resolve()), "--worker",
                       "--variant", variant, "--library", manifest["variants"][variant]["library"],
                       "--result", str(result_path), "--samples", str(args.samples),
                       "--warmup", str(args.warmup), "--iters", str(args.iters)]
            print(f"Running locally: {tag}; log: {output / (tag + '.log')}", flush=True)
            run_logged(command, output / f"{tag}.log", cwd=ROOT)
            trial = json.loads(result_path.read_text())
            trial.update({"round": round_idx, "position": position})
            results["trials"].append(trial)
            (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
            print(f"{tag}: {trial['median_ms'] * 1000:.3f} us", flush=True)
    results["summary"] = summarize(results["trials"])
    (output / "results.json").write_text(json.dumps(results, indent=2) + "\n")
    print(json.dumps(results["summary"], indent=2), flush=True)
    print(f"Before/after ratio >1 means current code is faster. Results: {output / 'results.json'}")


if __name__ == "__main__":
    main()
