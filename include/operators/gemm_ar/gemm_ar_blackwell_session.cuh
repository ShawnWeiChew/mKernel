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
          pybind11::arg("timings_ptr") = 0);

    // The comp/comm split the dispatch switch picks for this M. The profiler
    // needs it to label blocks as GEMM or all-reduce CTAs; -1 means the switch
    // does not handle that M.
    m.def("num_comp_sm_for_m",
          &gemm_ar_intranode_blackwell::gemm_ar_blackwell_num_comp_sm,
          pybind11::arg("M"));
    m.attr("NUM_BLOCKS") = (int)gemm_ar_intranode_blackwell::config::NUM_BLOCKS;
    // INTRA_NUM_DEVICES is baked in at compile time. Exporting it lets the
    // profile driver refuse a world size the .so was not built for, which
    // otherwise deadlocks in the multicast path rather than erroring.
    m.attr("NUM_DEVICES") = (int)gemm_ar_intranode_blackwell::config::NUM_DEVICES;

    // Which warp of a comp CTA plays which role. The renderer turns records
    // into rows with this, so exporting it keeps the plot honest if the warp
    // specialisation in config_t is ever retuned.
    {
        using cfg = gemm_ar_intranode_blackwell::config;
        pybind11::dict layout;
        layout["NUM_WARPS"] = (int)cfg::NUM_WARPS;
        layout["EPILOGUE_WARPS"] = (int)cfg::EPILOGUE_WARPS;
        layout["PRODUCER_WARP_ID"] = (int)cfg::PRODUCER_WARP_ID;
        layout["FIRST_CONSUMER_WARP_ID"] = (int)cfg::FIRST_CONSUMER_WARP_ID;
        layout["CONSUMER_WARPS"] = (int)cfg::CONSUMER_WARPS;
        layout["WARPGROUP_WARPS"] = (int)kittens::WARPGROUP_WARPS;
        layout["NUM_CLUSTERS"] = (int)cfg::NUM_CLUSTERS;
        m.attr("WARP_LAYOUT") = layout;
    }

#ifdef PROFILE_TIMINGS
    // Only a profile build carries these. A runtime that must refuse the emit
    // overhead can probe hasattr(mod, "EVENTS_PER_BLOCK") and hard-fail.
    {
        pybind11::dict ev;
#define GEMM_AR_BLACKWELL_EXPORT_EVENT(name) \
    ev[#name] = (int)gemm_ar_intranode_blackwell::EV_##name;
        GEMM_AR_BLACKWELL_TIMING_EVENTS(GEMM_AR_BLACKWELL_EXPORT_EVENT)
#undef GEMM_AR_BLACKWELL_EXPORT_EVENT
        m.attr("TIMING_EVENTS") = ev;
        m.attr("EVENTS_PER_BLOCK") = (int)mkernel_timings::EVENTS_PER_BLOCK;
        m.attr("TIMING_RECORD_SIZE") = (int)sizeof(mkernel_timings::TimingRecord);
        m.attr("TIMING_WARP_ID_SHIFT") = (int)mkernel_timings::WARP_ID_SHIFT;
    }
#endif
}
