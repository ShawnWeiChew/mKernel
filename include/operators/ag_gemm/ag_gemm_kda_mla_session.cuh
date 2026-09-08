#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("ag_gemm_kda_mla",
          &ag_gemm_kda_mla::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("A_local_buf"),
          pybind11::arg("B"),
          pybind11::arg("C"));
    // Tuning surface for the bench: pick the schedule explicitly instead of
    // letting entrypoint's hand-picked table choose it.
    m.def("ag_gemm_kda_mla_tuned",
          &ag_gemm_kda_mla::entrypoint_tuned,
          pybind11::arg("A"),
          pybind11::arg("A_local_buf"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("col_block"),
          pybind11::arg("num_cta"),
          pybind11::arg("supergroup_width"));
    m.def("ag_gemm_kda_mla_tuning_configs", &ag_gemm_kda_mla::tuning_configs);
    m.def("ag_gemm_kda_mla_granularity",
          &ag_gemm_kda_mla::tuning_config_granularity,
          pybind11::arg("num_cta"),
          pybind11::arg("col_block"));
}
