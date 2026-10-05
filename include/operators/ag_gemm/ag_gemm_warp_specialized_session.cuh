#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def(
        "ag_gemm_warp_specialized",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C, int logical_global_m) {
            ag_gemm_warp_specialized::entrypoint<dist::ParallelBuffer, at::Tensor>(
                A, B, C, logical_global_m);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));

    // BEGIN temporary M=8192, N=3648 memcpy-slice benchmark bindings.
    // These are deliberately copy-pasted so the experiment can be removed as
    // one self-contained block after the tuning run.
    m.def(
        "ag_gemm_warp_specialized_memcpy_slices_1",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C, int logical_global_m) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                        "A.local_world_size must match the compiled INTRA_NUM_DEVICES");
            TORCH_CHECK(logical_global_m == 8192 && B.size(0) == 3648,
                        "memcpy-slice benchmark binding requires M=8192 and N=3648");

            using fg = ag_gemm_warp_specialized::fused_globals<128, 256, 2, 1, 1>;
            fg globals = ag_gemm_warp_specialized::ag_gemm_warp_specialized_make_globals<
                dist::ParallelBuffer,
                at::Tensor,
                128,
                256,
                2,
                1,
                1>(A,
                   B,
                   C,
                   dev_idx,
                   logical_global_m,
                   static_cast<int>(B.size(0)),
                   static_cast<int>(B.size(1)),
                   nullptr);
            ag_gemm_warp_specialized::launch_ag_gemm_warp_specialized<128, 256, 2, 10, 1, 1>(
                globals);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));

    m.def(
        "ag_gemm_warp_specialized_memcpy_slices_2",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C, int logical_global_m) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                        "A.local_world_size must match the compiled INTRA_NUM_DEVICES");
            TORCH_CHECK(logical_global_m == 8192 && B.size(0) == 3648,
                        "memcpy-slice benchmark binding requires M=8192 and N=3648");

            using fg = ag_gemm_warp_specialized::fused_globals<128, 256, 2, 1, 2>;
            fg globals = ag_gemm_warp_specialized::ag_gemm_warp_specialized_make_globals<
                dist::ParallelBuffer,
                at::Tensor,
                128,
                256,
                2,
                1,
                2>(A,
                   B,
                   C,
                   dev_idx,
                   logical_global_m,
                   static_cast<int>(B.size(0)),
                   static_cast<int>(B.size(1)),
                   nullptr);
            ag_gemm_warp_specialized::launch_ag_gemm_warp_specialized<128, 256, 2, 10, 1, 2>(
                globals);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));

    m.def(
        "ag_gemm_warp_specialized_memcpy_slices_4",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C, int logical_global_m) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                        "A.local_world_size must match the compiled INTRA_NUM_DEVICES");
            TORCH_CHECK(logical_global_m == 8192 && B.size(0) == 3648,
                        "memcpy-slice benchmark binding requires M=8192 and N=3648");

            using fg = ag_gemm_warp_specialized::fused_globals<128, 256, 2, 1, 4>;
            fg globals = ag_gemm_warp_specialized::ag_gemm_warp_specialized_make_globals<
                dist::ParallelBuffer,
                at::Tensor,
                128,
                256,
                2,
                1,
                4>(A,
                   B,
                   C,
                   dev_idx,
                   logical_global_m,
                   static_cast<int>(B.size(0)),
                   static_cast<int>(B.size(1)),
                   nullptr);
            ag_gemm_warp_specialized::launch_ag_gemm_warp_specialized<128, 256, 2, 10, 1, 4>(
                globals);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));
    // END temporary M=8192, N=3648 memcpy-slice benchmark bindings.
}
