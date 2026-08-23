#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def("gemm_ar_intranode_blackwell",
          &gemm_ar_intranode_blackwell::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("C_final"),
          pybind11::arg("epoch"),
          pybind11::arg("gemm_to_ar_signal_strategy"),
          pybind11::arg("num_comp_sm") =
              gemm_ar_intranode_blackwell::DEFAULT_NUM_COMP_SM,
          pybind11::arg("ar_unroll") =
              gemm_ar_intranode_blackwell::AR_UNROLL_BY_SHAPE,
          pybind11::arg("signal_depth") =
              gemm_ar_intranode_blackwell::DEFAULT_SIGNAL_DEPTH);
    // Lets the bench sweep exactly the splits this module was built with,
    // rather than a hardcoded list that can drift from the build flags.
    m.def("compiled_comp_sm_splits",
          &gemm_ar_intranode_blackwell::compiled_comp_sm_splits);
    m.def("compiled_ar_unrolls", &gemm_ar_intranode_blackwell::compiled_ar_unrolls);
    m.def("compiled_strategies", &gemm_ar_intranode_blackwell::compiled_strategies);
    m.def("compiled_signal_depths",
          &gemm_ar_intranode_blackwell::compiled_signal_depths);
    m.def("num_blocks", &gemm_ar_intranode_blackwell::num_blocks);
}