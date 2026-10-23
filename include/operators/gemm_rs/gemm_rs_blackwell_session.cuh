#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
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
           int swizzle_size) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A, B, C, ready, completion_epochs, swizzle_size);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("swizzle_size") = gemm_rs_blackwell::DEFAULT_SWIZZLE_SIZE,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "swizzle_size (1, 2, 4, or 8) is the tile-column width of each rasterization "
        "supergroup.");
}
