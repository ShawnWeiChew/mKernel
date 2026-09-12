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
          // -1 = tuned default (2 at M in {16384, 32768}, both KDA and MLA);
          // 1 or 2 to force a path for A/B testing.
          pybind11::arg("consumer_warps") = -1);
}
