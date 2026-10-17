#!/usr/bin/env python3
"""Run gemm_traversal_bench.py on one Modal Blackwell GPU.

    pip install --upgrade modal
    modal setup
    modal run bench/gemm_traversal_modal.py
    modal run bench/gemm_traversal_modal.py --gpu B200 \
        --bench-args "--shapes 2048 4096 --samples 3" \
        --output build/gemm_traversal_b200.json
"""

from pathlib import Path
import json
import shlex
import shutil
import subprocess
import sys
import tempfile

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/workspace/mKernel"
ARCHITECTURES = {"B200": "sm_100a", "B300": "sm_103a", "B200+": None}

image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12"
    )
    .entrypoint([])
    .apt_install("make", "g++", "cuda-nsight-compute-13-2")
    .pip_install("torch==2.10.0", index_url="https://download.pytorch.org/whl/cu130")
    .env({"PYTHONUNBUFFERED": "1", "CUDA_HOME": "/usr/local/cuda"})
    .add_local_dir(ROOT / "include", f"{REMOTE_ROOT}/include")
)
# Upload only the benchmark and its build inputs, including local source edits.
for relative in (
    "bench/gemm_traversal_bench.py",
    "bench/Makefile.gemm_traversal",
    "src/gemm_traversal_bench.cu",
    "src/ag_gemm_warp_specialized.cu",
):
    image = image.add_local_file(ROOT / relative, f"{REMOTE_ROOT}/{relative}")

app = modal.App("mkernel-gemm-traversal", image=image)


def resolve_arch(arch: str | None) -> str:
    if arch is None:
        import torch

        # B200+ can allocate either GPU; architecture-specific kernels must
        # match the actual device, including the benchmark's capability check.
        capability = torch.cuda.get_device_capability(0)
        arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(capability)
        if arch is None:
            raise RuntimeError(f"Unsupported GPU compute capability: {capability}")
        print(f"Detected {torch.cuda.get_device_name(0)}; using {arch}", flush=True)
    return arch


@app.function(gpu="B200+", cpu=4, memory=8192, timeout=3600)
def benchmark(arch: str | None, bench_args: list[str]) -> str | None:
    arch = resolve_arch(arch)
    # A fresh path prevents build-only/validation-only runs returning stale JSON.
    with tempfile.TemporaryDirectory(prefix="gemm-traversal-") as directory:
        result = Path(directory) / "results.json"
        command = [
            sys.executable, "-u", f"{REMOTE_ROOT}/bench/gemm_traversal_bench.py",
            *bench_args,
            "--build", "--arch", arch, "--cuda-home", "/usr/local/cuda",
            "--device", "0", "--json", str(result),
        ]
        print(f"Running: {shlex.join(command)}", flush=True)
        subprocess.run(command, cwd=REMOTE_ROOT, check=True)
        return result.read_text() if result.is_file() else None


@app.function(gpu="B200+", cpu=4, memory=16384, timeout=3600)
def profile(arch: str | None, bench_args: list[str], ncu_set: str) -> dict[str, bytes]:
    arch = resolve_arch(arch)
    ncu = shutil.which("ncu") or "/usr/local/cuda/bin/ncu"
    with tempfile.TemporaryDirectory(prefix="gemm-ncu-") as directory:
        folder = Path(directory)
        status = {"arch": arch, "ncu_set": ncu_set, "ok": False}

        def run(command, log_name):
            print(f"Running: {shlex.join(command)}", flush=True)
            # Log to disk so NCU diagnostics can be downloaded on failure too.
            with (folder / log_name).open("w") as log:
                completed = subprocess.run(command, cwd=REMOTE_ROOT, stdout=log,
                                           stderr=subprocess.STDOUT)
            print((folder / log_name).read_text(), flush=True)
            return completed.returncode

        # Fail fast if this Modal runtime cannot expose GPU performance counters.
        probe = [ncu, "--set", "basic", "--launch-count", "1", "--export",
                 str(folder / "probe"), sys.executable, "-c",
                 "import torch; x = torch.ones(1024, device='cuda'); "
                 "x.add_(1); torch.cuda.synchronize()"]
        code = run(probe, "probe.log")
        status.update(stage="probe", returncode=code)
        if code == 0 and (folder / "probe.ncu-rep").is_file():
            command = [sys.executable, "-u", f"{REMOTE_ROOT}/bench/gemm_traversal_bench.py",
                       *bench_args, "--arch", arch, "--cuda-home", "/usr/local/cuda",
                       "--device", "0"]
            code = run([*command, "--build-only"], "build.log")
            status.update(stage="build", returncode=code)
            if code == 0:
                for kernel in ("device_Mlocal_N", "Mglobal_N", "cuBLAS"):
                    report = folder / f"{kernel}.ncu-rep"
                    code = run([
                        ncu, "--set", ncu_set, "--profile-from-start", "off",
                        "--replay-mode", "application", "--cache-control", "none",
                        "--clock-control", "none", "--graph-profiling", "node",
                        "--export", str(report), *command, "--profile-kernel", kernel,
                        "--json", str(folder / f"{kernel}.json"),
                    ], f"{kernel}.log")
                    status.update(stage=kernel, returncode=code)
                    if code != 0 or not report.is_file():
                        break
                    code = run([ncu, "--import", str(report), "--csv", "--page", "raw"],
                               f"{kernel}.csv")
                    status.update(stage=f"{kernel} CSV", returncode=code)
                    if code != 0:
                        break
                else:
                    status["ok"] = True
        (folder / "status.json").write_text(json.dumps(status, indent=2) + "\n")
        return {path.name: path.read_bytes() for path in folder.iterdir() if path.is_file()}


@app.local_entrypoint()
def main(
    gpu: str = "B200+",
    bench_args: str = "",
    output: str = "build/gemm_traversal_modal.json",
    ncu: bool = False,
    ncu_output: str = "build/gemm_traversal_ncu",
    ncu_set: str = "detailed",
):
    """Forward benchmark options with --bench-args and save JSON locally."""
    gpu = gpu.upper()
    if gpu not in ARCHITECTURES:
        raise ValueError("--gpu must be B200, B300, or B200+")
    arguments = shlex.split(bench_args)
    if ncu:
        if ncu_set not in {"basic", "detailed", "full"}:
            raise ValueError("--ncu-set must be basic, detailed, or full")
        if any(arg.split("=", 1)[0] in {"--build-only", "--validate-only", "--profile-kernel"}
               for arg in arguments):
            raise ValueError("--ncu manages profiling; omit build-only/validate-only/profile-kernel")
        if not any(arg.split("=", 1)[0] == "--shapes" for arg in arguments):
            arguments = ["--shapes", "32768", *arguments]
    # These options are owned by the wrapper to match the remote environment.
    managed = {"--arch", "--cuda-home", "--device", "--json"}
    for argument in arguments:
        if argument.split("=", 1)[0] in managed:
            raise ValueError(
                f"{argument} is managed by the launcher; use --gpu and --output"
            )
    if ncu:
        artifacts = profile.with_options(gpu=gpu).remote(ARCHITECTURES[gpu], arguments, ncu_set)
        destination = Path(ncu_output).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        for name, data in artifacts.items():
            (destination / name).write_bytes(data)
        print(f"Saved NCU artifacts to {destination}")
        status = json.loads(artifacts["status.json"])
        if not status["ok"]:
            raise SystemExit(f"NCU stopped at {status['stage']}; inspect {destination}/*.log")
        return
    result = benchmark.with_options(gpu=gpu).remote(ARCHITECTURES[gpu], arguments)
    if result is not None:
        destination = Path(output).expanduser()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(result)
        print(f"Saved {destination}")
