#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

template <ag_gemm_warp_specialized::AgStrategy STRATEGY>
void bind_ag_gemm_launch(pybind11::module_& m, const char* name) {
    m.def(
        name,
        [](dist::ParallelBuffer& A,
           dist::ParallelBuffer& A_copy_ready,
           const at::Tensor& B,
           at::Tensor& C) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            const int M = C.size(0) * C.size(1);
            ag_gemm_warp_specialized::
                launch<dist::ParallelBuffer, dist::ParallelBuffer, at::Tensor, STRATEGY>(
                    A,
                    A_copy_ready,
                    B,
                    C,
                    M,
                    B.size(0),
                    B.size(1),
                    dev_idx,
                    at::cuda::getCurrentCUDAStream().stream());
        },
        pybind11::arg("A"),
        pybind11::arg("A_copy_ready"),
        pybind11::arg("B"),
        pybind11::arg("C"));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("ag_gemm_warp_specialized_ready_words") = ag_gemm_warp_specialized::READY_WORDS;
    m.def(
        "ag_gemm_warp_specialized_prepare",
        [](dist::ParallelBuffer& A_copy_ready, int M, int N) {
            const int dev_idx = A_copy_ready.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            ag_gemm_warp_specialized::prepare(
                A_copy_ready, M, N, dev_idx, at::cuda::getCurrentCUDAStream().stream());
        },
        pybind11::arg("A_copy_ready"),
        pybind11::arg("M"),
        pybind11::arg("N"));
    using ag_gemm_warp_specialized::AgStrategy;
    bind_ag_gemm_launch<AgStrategy::AUTO>(m, "ag_gemm_warp_specialized_launch");
    bind_ag_gemm_launch<AgStrategy::PULL>(m, "ag_gemm_warp_specialized_launch_pull");
    bind_ag_gemm_launch<AgStrategy::MULTICAST_PUSH>(
        m, "ag_gemm_warp_specialized_launch_multicast_push");
}
