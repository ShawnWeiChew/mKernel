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
    // NVLink/NVSwitch bandwidth probe — the AR loop on its own, swept over the
    // tile config. Driven by bench/mnvl_bw_sweep.py.
    m.def("test_nvswitch_bw",
          &gemm_ar_intranode_blackwell::mnvl_bw_test::bw_test_entrypoint,
          pybind11::arg("C"),
          pybind11::arg("C_final"),
          pybind11::arg("subtile_m"),
          pybind11::arg("subtile_n"),
          pybind11::arg("num_comm_sm"),
          pybind11::arg("ar_unroll"),
          pybind11::arg("supergroup_width") = 8,
          pybind11::arg("num_repeats") = 1);
}
