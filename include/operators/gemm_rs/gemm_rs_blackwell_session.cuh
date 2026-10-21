#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "ATen/core/TensorBody.h"
#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    m.def(
        "gemm_blackwell_launch",
        [](at::Tensor A, at::Tensor B, at::Tensor C) {
            gemm_rs_blackwell::launch_gemm_only(A, B, C);
        },
        pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"),
        "Single-GPU GEMM benchmark: C = A @ B.T. Requires RS disabled in the "
        "kernel source. Uses ordinary BF16 tensors, no exit barrier or PDL overlap.");
    m.def(
        "gemm_rs_blackwell_prepare",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::prepare(A, B, C, ready, completion_epochs);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        "Initialize local counters once. Synchronize the stream and all ranks before launch.");
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
           bool with_exit_barrier) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A, B, C, ready, completion_epochs, with_exit_barrier);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("with_exit_barrier") = true,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "Set with_exit_barrier=False only for benchmarking with RS disabled in the "
        "kernel; this skips the exit barrier and disables PDL launch overlap.");
}
