#pragma once

#include <torch/csrc/utils/pybind.h>

#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
#ifdef PROFILE_TIMINGS
    m.def("ag_gemm_kda_mla",
          &ag_gemm_kda_mla::entrypoint,
          pybind11::arg("A"),
          pybind11::arg("B"),
          pybind11::arg("C"),
          pybind11::arg("timings_ptr") = 0);

    // Only profile builds carry these attributes. A production loader can
    // probe `hasattr(mod, "EVENTS_PER_BLOCK")` and hard-fail on a profile .so.
    {
        pybind11::dict ev;
        ev["CTA_BEGIN"] = (int)ag_gemm_kda_mla::EV_CTA_BEGIN;
        ev["CTA_END"] = (int)ag_gemm_kda_mla::EV_CTA_END;
        ev["PROD_TILE_BEGIN"] = (int)ag_gemm_kda_mla::EV_PROD_TILE_BEGIN;
        ev["PROD_TILE_DONE"] = (int)ag_gemm_kda_mla::EV_PROD_TILE_DONE;
        ev["PROD_K_BEGIN"] = (int)ag_gemm_kda_mla::EV_PROD_K_BEGIN;
        ev["PROD_K_STAGE_READY"] = (int)ag_gemm_kda_mla::EV_PROD_K_STAGE_READY;
        ev["PROD_K_ISSUED"] = (int)ag_gemm_kda_mla::EV_PROD_K_ISSUED;
        ev["MMA_TILE_BEGIN"] = (int)ag_gemm_kda_mla::EV_MMA_TILE_BEGIN;
        ev["MMA_TMEM_READY"] = (int)ag_gemm_kda_mla::EV_MMA_TMEM_READY;
        ev["MMA_TILE_DONE"] = (int)ag_gemm_kda_mla::EV_MMA_TILE_DONE;
        ev["MMA_K_BEGIN"] = (int)ag_gemm_kda_mla::EV_MMA_K_BEGIN;
        ev["MMA_K_INPUT_READY"] = (int)ag_gemm_kda_mla::EV_MMA_K_INPUT_READY;
        ev["MMA_K_ISSUED"] = (int)ag_gemm_kda_mla::EV_MMA_K_ISSUED;
        ev["EPI_TILE_BEGIN"] = (int)ag_gemm_kda_mla::EV_EPI_TILE_BEGIN;
        ev["EPI_MMA_READY"] = (int)ag_gemm_kda_mla::EV_EPI_MMA_READY;
        ev["EPI_TMEM_LOADED"] = (int)ag_gemm_kda_mla::EV_EPI_TMEM_LOADED;
        ev["EPI_TILE_DONE"] = (int)ag_gemm_kda_mla::EV_EPI_TILE_DONE;
        ev["EPI_DRAIN_BEGIN"] = (int)ag_gemm_kda_mla::EV_EPI_DRAIN_BEGIN;
        ev["EPI_DRAIN_DONE"] = (int)ag_gemm_kda_mla::EV_EPI_DRAIN_DONE;
        m.attr("TIMING_EVENTS") = ev;

        pybind11::dict roles;
        roles["CTA"] = (int)::timings::ROLE_CTA;
        roles["PRODUCER"] = (int)::timings::ROLE_PRODUCER;
        roles["MMA"] = (int)::timings::ROLE_MMA;
        roles["EPILOGUE"] = (int)::timings::ROLE_EPILOGUE;
        m.attr("TIMING_ROLES") = roles;

        m.attr("EVENTS_PER_BLOCK") = (int)::timings::EVENTS_PER_BLOCK;
        m.attr("TIMING_RECORD_SIZE") = (int)sizeof(::timings::TimingRecord);
        // Both COL_BLOCK specializations launch the same grid.
        m.attr("TIMING_NUM_BLOCKS") =
            (int)ag_gemm_kda_mla::fused_globals<ag_gemm_kda_mla::DEFAULT_ROW_BLOCK,
                                                ag_gemm_kda_mla::DEFAULT_COL_BLOCK>::NUM_BLOCKS;
#ifdef PROFILE_TIMINGS_FINE
        m.attr("TIMING_FINE") = true;
#else
        m.attr("TIMING_FINE") = false;
#endif
    }
#else
    // Shipping build: no timings_ptr in the signature, so a caller that tries
    // to hand this .so a ring buffer gets a TypeError instead of a silent no-op.
    m.def(
        "ag_gemm_kda_mla",
        [](dist::ParallelBuffer& A, const at::Tensor& B, at::Tensor& C) {
            ag_gemm_kda_mla::entrypoint(A, B, C);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"));
#endif  // PROFILE_TIMINGS
}
