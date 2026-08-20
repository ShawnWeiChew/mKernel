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
    m.def("gemm_ar_intranode_blackwell_profile",
          &gemm_ar_intranode_blackwell::entrypoint_profile,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("barrier"),
          pybind11::arg("C_final"),
          pybind11::arg("timings"));
    // Optional. The cache is keyed on (pointers, shape), so a stale entry can
    // never be matched by a buffer that would need a different descriptor;
    // this just reclaims the few KB per entry after buffers are retired.
    m.def("gemm_ar_intranode_blackwell_clear_cache",
          &gemm_ar_intranode_blackwell::clear_globals_cache);

#ifdef PROFILE_TIMINGS
    // Only a PROFILE=1 build carries these. A production loader that must not
    // pay the emit overhead can hard-fail on
    // hasattr(mod, "EVENTS_PER_BLOCK") -- the attribute exists nowhere else.
    //
    // TIMING_EVENTS is a name -> id dict. The dump path copies it into the
    // .npz, so a saved trace stays renderable after this enum has grown and
    // without importing this .so at all.
    {
        namespace ns = gemm_ar_intranode_blackwell;
        pybind11::dict ev;
        for (const auto& e : ns::TIMING_EVENT_TABLE) ev[e.name] = (int)e.id;
        m.attr("TIMING_EVENTS") = ev;
        m.attr("EVENTS_PER_BLOCK") = (int)ns::EVENTS_PER_BLOCK;
        m.attr("TIMING_RECORD_SIZE") = (int)sizeof(ns::TimingRecord);
        m.attr("TIMING_NUM_BLOCKS") = (int)ns::config::NUM_BLOCKS;
        m.attr("TIMING_NUM_WARPS") = (int)ns::config::NUM_WARPS;
    }
#endif
}
