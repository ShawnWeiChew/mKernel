"""Run the standalone multimem-vs-NCCL RS sweep on eight Modal Blackwell GPUs.

    pip install modal
    modal setup
    modal run bench/modal_gemm_rs_bench.py

    # Small correctness-only smoke test before the full sweep:
    modal run bench/modal_gemm_rs_bench.py --shapes 2048 --check-only

    # Subset of RS warps per CTA (default sweeps 1,2,4,8):
    modal run bench/modal_gemm_rs_bench.py --num-warps 1,8

    # Subset of RS tile rows (default 32,64,128,256; non-dividing tiles skipped):
    modal run bench/modal_gemm_rs_bench.py --tile-rows 32,64

    # Optional fixed GPU type instead of the default B200+ pool:
    MKERNEL_MODAL_GPU=B200 modal run bench/modal_gemm_rs_bench.py

Default: one B200+:8 container, all five GEMM+RS shapes, 32/64/128/256-row RS tiles,
1/2/4/8 RS warps per CTA, raw multimem and multimem+exit-barrier timings. After provisioning, detect the
GPU capability and compile sm_100a (B200) or sm_103a (B300). Dependencies are
cached in the image; compilation runs on the GPU container outside timing.
Only the needed source files are sent;
local build products are excluded. Results, logs, and run metadata are returned
to --output-dir on this machine. Failed benchmark runs save their logs too.

Modal GPU configuration: https://modal.com/docs/guide/gpu
CUDA image setup: https://modal.com/docs/guide/cuda
"""

from datetime import datetime, timezone
import json
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/mkernel"
GPU_TYPE = os.environ.get("MKERNEL_MODAL_GPU", "B200+")
if GPU_TYPE not in ("B200+", "B200", "B300"):
    raise ValueError("MKERNEL_MODAL_GPU must be B200+, B200 or B300")
CUDA_IMAGE = "nvidia/cuda:13.2.0-devel-ubuntu24.04"
TORCH_VERSION = "2.13.0"

image = (
    modal.Image.from_registry(CUDA_IMAGE, add_python="3.12")
    .entrypoint([])
    .apt_install("build-essential", "libibverbs-dev", "libnuma1")
    .pip_install(f"torch=={TORCH_VERSION}", index_url="https://download.pytorch.org/whl/cu132")
    .pip_install("numpy==2.2.6")
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "1"})
    .add_local_dir(ROOT / "include", f"{REMOTE_ROOT}/include", copy=True)
    .add_local_file(ROOT / "Makefile", f"{REMOTE_ROOT}/Makefile", copy=True)
    .add_local_file(ROOT / "src/gemm_rs_blackwell.cu", f"{REMOTE_ROOT}/src/gemm_rs_blackwell.cu", copy=True)
    .add_local_file(ROOT / "python/load_module.py", f"{REMOTE_ROOT}/python/load_module.py", copy=True)
    .add_local_file(ROOT / "bench/gemm_rs_blackwell_rs_bench.py", f"{REMOTE_ROOT}/bench/gemm_rs_blackwell_rs_bench.py", copy=True)
    .workdir(REMOTE_ROOT)
)
app = modal.App("mkernel-rs-benchmark")


def select_arch(capabilities):
    """Choose the architecture from all eight provisioned GPUs, not the request."""
    if len(capabilities) != 8 or len(set(capabilities)) != 1:
        raise RuntimeError(f"Expected eight GPUs with matching capabilities; got {capabilities}")
    targets = {(10, 0): ("100a", "B200"), (10, 3): ("103a", "B300")}
    if capabilities[0] not in targets:
        raise RuntimeError(f"Unsupported GPU capability: {capabilities[0]}")
    return targets[capabilities[0]]


def optional_diagnostic(command):
    """Keep GPU diagnostics in the log without making NVML a benchmark dependency."""
    import subprocess

    label = " ".join(command)
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"[diagnostic unavailable] {label}: {exc}\n"
    output = result.stdout + result.stderr
    if result.returncode:
        return f"[diagnostic unavailable] {label} exited {result.returncode}\n{output}"
    return output


@app.function(image=image, gpu=f"{GPU_TYPE}:8", cpu=16, memory=65536,
              timeout=2100, max_containers=1, retries=0)
def run_benchmark(shapes: str, n: int, warmup: int, iters: int,
                  graph_size: int, repeats: int, check_only: bool, num_warps: str,
                  tile_rows: str):
    import hashlib
    import signal
    import subprocess
    import sys
    import uuid

    import torch

    capabilities = [torch.cuda.get_device_capability(device) for device in range(torch.cuda.device_count())]
    arch, actual_gpu = select_arch(capabilities)
    topology = optional_diagnostic(["nvidia-smi", "topo", "-m"])
    gpu_info = optional_diagnostic(["nvidia-smi"])
    print(gpu_info, flush=True)
    print(topology, flush=True)
    metadata = {
        "gpu_request": f"{GPU_TYPE}:8", "gpu": f"{actual_gpu}:8", "arch": arch,
        "device_names": [torch.cuda.get_device_name(device) for device in range(8)],
        "capabilities": capabilities, "cuda_image": CUDA_IMAGE,
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
        "nccl": torch.cuda.nccl.version(),
    }
    build_command = [
        "make", "-B", "GPU=blackwell", "INTRA_NUM_DEVICES=8",
        "CUDA_HOME=/usr/local/cuda", f"PYTHON={sys.executable}",
        f"ARCH=-gencode arch=compute_{arch},code=sm_{arch}", "gemm-rs-blackwell",
    ]
    metadata["build_command"] = build_command
    print(f"Provisioned {actual_gpu}:8; compiling sm_{arch}", flush=True)
    def run_command(command, timeout):
        with subprocess.Popen(
            command, cwd=REMOTE_ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, start_new_session=True,
        ) as process:
            try:
                log, _ = process.communicate(timeout=timeout)
                return process.returncode, log
            except subprocess.TimeoutExpired:
                # Stop make/nvcc or torchrun and all their child processes.
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    log, _ = process.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    log, _ = process.communicate()
                return 124, log + f"\nCommand exceeded {timeout} seconds.\n"

    env = dict(os.environ, DIST_BROKER_KEY=f"rs_{uuid.uuid4().hex[:12]}", NCCL_DEBUG="WARN")
    build_code, build_log = run_command(build_command, 480)
    print(build_log, flush=True)
    if build_code:
        return {"returncode": build_code, "results": [], "metadata": metadata,
                "log": gpu_info + "\n" + topology + "\n" + build_log}
    result_path = Path("/tmp/rs-results.json")
    result_path.unlink(missing_ok=True)
    command = [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
        # End torchrun option parsing before benchmark flags such as --n.
        "--nproc-per-node=8", "--max-restarts=0", "--", "bench/gemm_rs_blackwell_rs_bench.py",
        "--shapes", shapes, "--n", str(n), "--warmup", str(warmup), "--iters", str(iters),
        "--graph-size", str(graph_size), "--repeats", str(repeats),
        "--num-warps", *num_warps.split(","),
        "--tile-rows", *tile_rows.split(","),
        "--save-json", str(result_path),
    ]
    if check_only:
        command.append("--check-only")
    print("Running:", " ".join(command), flush=True)
    returncode, log = run_command(command, 1500)
    print(log, flush=True)
    sources = ["src/gemm_rs_blackwell.cu", "include/operators/gemm_rs/gemm_rs_blackwell.cuh",
               "include/operators/gemm_rs/gemm_rs_blackwell_session.cuh", "bench/gemm_rs_blackwell_rs_bench.py"]
    return {
        "returncode": returncode,
        "log": gpu_info + "\n" + topology + "\n" + build_log + "\n" + log,
        "results": json.loads(result_path.read_text()) if result_path.exists() else [],
        "metadata": {
            **metadata, "command": command,
            "source_sha256": {
                path: hashlib.sha256((Path(REMOTE_ROOT) / path).read_bytes()).hexdigest()
                for path in sources
            },
            "nccl_env": {key: value for key, value in env.items() if key.startswith("NCCL_")},
        },
    }


@app.local_entrypoint()
def main(shapes: str = "2048,4096,8192,16384,32768", n: int = 7168,
         warmup: int = 5, iters: int = 20, graph_size: int = 20, repeats: int = 5,
         check_only: bool = False, num_warps: str = "1,2,4,8",
         tile_rows: str = "32,64,128,256",
         output_dir: str = "results/rs-modal"):
    # Validate locally before requesting eight GPUs.
    values = [int(value) for value in shapes.split(",")]
    tiles = tile_rows.split(",")
    if not tiles or any(value not in ("32", "64", "128", "256") for value in tiles):
        raise ValueError("--tile-rows must be a comma-separated subset of 32,64,128,256")
    alignment = 8 * min(int(value) for value in tiles)
    if any(m <= 0 or m % alignment for m in values):
        raise ValueError(f"M values must be positive multiples of {alignment} on 8 GPUs; "
                         "tiles that do not divide a shape are skipped for it")
    if n <= 0 or n % 256 or warmup < 0 or min(iters, graph_size, repeats) < 1:
        raise ValueError("N must be a positive multiple of 256; timing counts must be positive (warmup >= 0)")
    if not num_warps or any(value not in ("1", "2", "4", "8") for value in num_warps.split(",")):
        raise ValueError("--num-warps must be a comma-separated subset of 1,2,4,8")
    result = run_benchmark.remote(shapes, n, warmup, iters, graph_size, repeats, check_only,
                                  num_warps, tile_rows)
    destination = Path(output_dir).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    actual_gpu = result["metadata"]["gpu"].split(":")[0].lower()
    prefix = destination / f"rs-{actual_gpu}-8gpu-{stamp}"
    log_path = prefix.with_suffix(".log")
    json_path = prefix.with_suffix(".json")
    log_path.write_text(result["log"])
    json_path.write_text(json.dumps({key: value for key, value in result.items() if key != "log"}, indent=2) + "\n")
    print(f"Saved log: {log_path.resolve()}")
    print(f"Saved results/metadata: {json_path.resolve()}")
    if result["returncode"]:
        raise RuntimeError(f"Remote benchmark failed with code {result['returncode']}; see {log_path}")
