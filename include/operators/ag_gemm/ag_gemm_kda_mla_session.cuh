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
    pybind11::arg("C"),
    pybind11::arg("timings_ptr") = 0);

    // The launch geometry the profiler needs. Both template instantiations share
    // NUM_BLOCKS, the warp split and NUM_DEVICES -- only the pipeline depths and
    // COL_BLOCK differ -- so reading them off the default one is safe.
    {
        using fg = ag_gemm_kda_mla::fused_globals<ag_gemm_kda_mla::DEFAULT_ROW_BLOCK,
                                                  ag_gemm_kda_mla::DEFAULT_COL_BLOCK>;
        m.attr("NUM_BLOCKS") = (int)fg::NUM_BLOCKS;
        // INTRA_NUM_DEVICES is baked in at compile time. Exporting it lets the
        // profile driver refuse a world size the .so was not built for, which
        // otherwise deadlocks in the multicast path rather than erroring.
        m.attr("NUM_DEVICES") = (int)fg::NUM_DEVICES;
        m.attr("K") = (int)fg::K;
        // Copy streams the staging all-gather is spread over. Exported so a
        // bench can report which setting produced a number.
        m.attr("A_COPY_STREAMS") = (int)ag_gemm_kda_mla::A_COPY_STREAMS;

        // Which warp plays which role. The renderer turns records into rows with
        // this, so exporting it keeps the plot honest if the warp specialisation
        // is ever retuned. There is no comp/comm SM split in this kernel: every
        // CTA computes, so the renderer is handed num_comp_sm == NUM_BLOCKS.
        pybind11::dict layout;
        layout["NUM_WARPS"] = (int)(fg::NUM_THREADS / kittens::WARP_THREADS);
        layout["EPILOGUE_WARPS"] = (int)fg::EPILOGUE_WARPS;
        layout["PRODUCER_WARP_ID"] = (int)fg::EPILOGUE_WARPS;
        layout["FIRST_CONSUMER_WARP_ID"] = (int)(fg::EPILOGUE_WARPS + fg::PRODUCER_WARPS);
        layout["CONSUMER_WARPS"] = (int)fg::CONSUMER_WARPS;
        layout["WARPGROUP_WARPS"] = (int)kittens::WARPGROUP_WARPS;
        layout["NUM_CLUSTERS"] = (int)fg::NUM_CLUSTERS;
        m.attr("WARP_LAYOUT") = layout;
    }

    m.def("col_block_for_m", &ag_gemm_kda_mla::ag_gemm_kda_mla_col_block, pybind11::arg("M"));

#ifdef PROFILE_TIMINGS
    // (peer, start_ms, end_ms) per staged shard, relative to a reference event
    // taken just before the first copy. Synchronize before calling.
    m.def("copy_times", &ag_gemm_kda_mla::ag_gemm_kda_mla_copy_times, pybind11::arg("dev_idx"));

    // Only a profile build carries these. A runtime that must refuse the emit
    // overhead can probe hasattr(mod, "EVENTS_PER_BLOCK") and hard-fail.
    {
        pybind11::dict ev;
#define AG_GEMM_KDA_MLA_EXPORT_EVENT(name) ev[#name] = (int)ag_gemm_kda_mla::EV_##name;
        AG_GEMM_KDA_MLA_TIMING_EVENTS(AG_GEMM_KDA_MLA_EXPORT_EVENT)
#undef AG_GEMM_KDA_MLA_EXPORT_EVENT
        m.attr("TIMING_EVENTS") = ev;
        m.attr("EVENTS_PER_BLOCK") = (int)mkernel_timings::EVENTS_PER_BLOCK;
        m.attr("TIMING_RECORD_SIZE") = (int)sizeof(mkernel_timings::TimingRecord);
        m.attr("TIMING_WARP_ID_SHIFT") = (int)mkernel_timings::WARP_ID_SHIFT;
    }
#endif
}
