"""Run the mKernel GEMM+RS in-kernel timing profiler on eight Modal B200+ GPUs.

Run from the repo with the Modal CLI installed and authenticated:
    modal run bench/modal_gemm_rs_blackwell_profile.py
    modal run bench/modal_gemm_rs_blackwell_profile.py --shapes 8192,16384 --ranks all

B200+ may land on B200 (sm_100a) or B300 (sm_103a), so the image builds the
profile .so for both. The remote side records one launch per shape and returns
the .npz traces, which are written to --out-dir and rendered to PDF locally
(pass --no-render to skip). Re-render or zoom later without a GPU:
    python python/timings.py traces/gemm_rs_blackwell_M8192_rank0.npz --blocks 0-7
"""

from pathlib import Path
import subprocess
import sys

import modal


ROOT = Path(__file__).resolve().parents[1]
REMOTE_ROOT = "/opt/mkernel"
REMOTE_TRACES = "/tmp/traces"
# The Makefile's GPU=blackwell targets sm_103a only; that cubin will not load on B200.
FAT_ARCH = "-gencode arch=compute_100a,code=sm_100a -gencode arch=compute_103a,code=sm_103a"

app = modal.App("mkernel-gemm-rs-blackwell-profile")
image = (
    modal.Image.from_registry("nvidia/cuda:13.2.0-devel-ubuntu24.04", add_python="3.12")
    .apt_install("build-essential")
    .pip_install("torch==2.9.1", index_url="https://download.pytorch.org/whl/cu130")
    .pip_install("numpy")
    .env({"CUDA_HOME": "/usr/local/cuda", "PYTHONUNBUFFERED": "1"})
    .add_local_file(ROOT / "Makefile", f"{REMOTE_ROOT}/Makefile", copy=True)
    .add_local_dir(ROOT / "include", f"{REMOTE_ROOT}/include", copy=True)
    .add_local_file(
        ROOT / "src/gemm_rs_blackwell.cu",
        f"{REMOTE_ROOT}/src/gemm_rs_blackwell.cu", copy=True,
    )
    # The CPU build uses the CUDA driver stub only for linking. At runtime,
    # Modal supplies the real GPU driver; never put stubs in LD_LIBRARY_PATH.
    .run_commands(
        f"LIBRARY_PATH=/usr/local/cuda/lib64/stubs make -C {REMOTE_ROOT} "
        "GPU=blackwell CUDA_HOME=/usr/local/cuda INTRA_NUM_DEVICES=8 "
        f'ARCH="{FAT_ARCH}" '
        "BACKEND_LIBS= BACKEND_DEFINES= gemm-rs-blackwell-profile"
    )
)
for relative_path in (
    "bench/gemm_rs_blackwell_profile.py",
    "bench/gemm_rs_blackwell_bench.py",
    "bench/common.py",
    "python/load_module.py",
    "python/load_thirdparty.py",
    "python/timings.py",
):
    image = image.add_local_file(ROOT / relative_path, f"{REMOTE_ROOT}/{relative_path}")


@app.function(image=image, gpu="B200+:8", cpu=16, memory=65536, timeout=1200)
def profile(shapes: str, ranks: str, warmup: int) -> dict[str, bytes]:
    """Returns {file name: .npz bytes} for every trace the dumping ranks wrote."""
    subprocess.run(["nvidia-smi", "--query-gpu=index,name,compute_cap", "--format=csv"])
    subprocess.run(
        [
            sys.executable, "-m", "torch.distributed.run",
            "--standalone", "--nnodes=1", "--nproc-per-node=8", "--max-restarts=0",
            "bench/gemm_rs_blackwell_profile.py",
            "--shapes", shapes, "--ranks", ranks, "--warmup", str(warmup),
            "--out-dir", REMOTE_TRACES,
            # Render locally instead of holding eight GPUs while matplotlib runs.
            "--no-render",
        ],
        cwd=REMOTE_ROOT,
        check=True,
        timeout=1140,
    )
    return {path.name: path.read_bytes() for path in sorted(Path(REMOTE_TRACES).glob("*.npz"))}


@app.local_entrypoint()
def main(
    shapes: str = "8192",
    ranks: str = "0",
    warmup: int = 10,
    out_dir: str = str(ROOT / "traces"),
    render: bool = True,
):
    traces = profile.remote(shapes, ranks, warmup)
    if not traces:
        raise SystemExit("the remote run wrote no traces")

    sys.path.insert(0, str(ROOT / "python"))
    import timings

    dest_dir = Path(out_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, blob in traces.items():
        dest = dest_dir / name
        dest.write_bytes(blob)
        print(f"wrote {dest} ({len(blob) / 1e6:.1f} MB)", flush=True)
        if render:
            timings.main([str(dest)])
