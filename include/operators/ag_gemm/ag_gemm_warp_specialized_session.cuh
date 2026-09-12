#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("ag_gemm_warp_specialized",
          &ag_gemm_warp_specialized::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("A_local_buf"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("logical_global_m"),
          // -1 = tuned default; otherwise a supergroup width in {5,10,15,20,25}
          // or a consumer-warp count in {1,2}. Only honored at KDA/MLA M in
          // {8192, 16384, 32768} -- every other shape ignores both.
          pybind11::arg("supergroup_width") = -1,
          pybind11::arg("consumer_warps") = -1);
}
