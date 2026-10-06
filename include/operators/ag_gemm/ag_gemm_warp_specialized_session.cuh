#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def(
        "ag_gemm_warp_specialized",
        [](dist::ParallelBuffer& A,
           const at::Tensor& A_local_buf,
           const at::Tensor& B,
           at::Tensor& C,
           int logical_global_m) {
            ag_gemm_warp_specialized::entrypoint<1>(
                A, A_local_buf, B, C, logical_global_m);
        },
        pybind11::arg("A"),
        pybind11::arg("A_local_buf"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));

    m.def(
        "ag_gemm_warp_specialized_batch_2",
        [](dist::ParallelBuffer& A,
           const at::Tensor& A_local_buf,
           const at::Tensor& B,
           at::Tensor& C,
           int logical_global_m) {
            ag_gemm_warp_specialized::entrypoint<2>(
                A, A_local_buf, B, C, logical_global_m);
        },
        pybind11::arg("A"),
        pybind11::arg("A_local_buf"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));
}
