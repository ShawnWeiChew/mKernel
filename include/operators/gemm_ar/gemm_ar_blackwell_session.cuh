#pragma once

#include <torch/csrc/utils/pybind.h>

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("gemm_ar_intranode_blackwell",
          &gemm_ar_intranode_blackwell::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("C_final"));
    // Optional. The cache is keyed on (pointers, shape), so a stale entry can
    // never be matched by a buffer that would need a different descriptor;
    // this just reclaims the few KB per entry after buffers are retired.
    m.def("gemm_ar_intranode_blackwell_clear_cache",
          &gemm_ar_intranode_blackwell::clear_globals_cache);
}