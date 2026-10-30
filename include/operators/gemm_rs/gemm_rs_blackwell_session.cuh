#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#ifdef PROFILE_TIMINGS
#include <pybind11/stl.h>

#include <optional>
#endif

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    {
        // One ready counter per READY_ROWS x READY_COLS region, for every signal
        // granularity: size `ready` as (M / world_size / rows, N / cols).
        using fg = gemm_rs_blackwell::fused_globals<gemm_rs_blackwell::DEFAULT_NUM_EPILOGUE_WARPS,
                                                    gemm_rs_blackwell::DEFAULT_C_TILE_DIVISOR,
                                                    gemm_rs_blackwell::DEFAULT_SWIZZLE_SIZE,
                                                    gemm_rs_blackwell::DEFAULT_SIGNALS_PER_TILE,
                                                    gemm_rs_blackwell::DEFAULT_RS_STRIP_ROWS>;
        m.attr("gemm_rs_blackwell_ready_rows") = fg::READY_ROWS;
        m.attr("gemm_rs_blackwell_ready_cols") = fg::READY_COLS;
    }
    m.def(
        "gemm_rs_blackwell_prepare",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::prepare(A, B, C, ready, completion_epochs);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        "Initialize local counters once. Synchronize the stream and all ranks before launch.");
#ifdef PROFILE_TIMINGS
    using namespace gemm_rs_blackwell;
    constexpr int timing_num_blocks = fused_globals<DEFAULT_NUM_EPILOGUE_WARPS,
                                                    DEFAULT_C_TILE_DIVISOR,
                                                    DEFAULT_SWIZZLE_SIZE,
                                                    DEFAULT_SIGNALS_PER_TILE,
                                                    DEFAULT_RS_STRIP_ROWS>::NUM_BLOCKS;
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
           int swizzle_size,
           int signals_per_tile,
           int rs_strip_rows,
           std::optional<at::Tensor> timings) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            TimingRecord* timing_records = nullptr;
            if (timings.has_value()) {
                TORCH_CHECK(timings->is_cuda(), "timings must be a CUDA tensor");
                TORCH_CHECK(timings->scalar_type() == at::kLong,
                            "timings must have dtype torch.int64");
                TORCH_CHECK(timings->is_contiguous(), "timings must be contiguous");
                TORCH_CHECK(timings->device() == A.device(),
                            "timings and A must be on the same CUDA device");
                constexpr int64_t required_elements =
                    static_cast<int64_t>(timing_num_blocks) * EVENTS_PER_BLOCK * 2;
                TORCH_CHECK(timings->numel() >= required_elements,
                            "timings needs at least ",
                            required_elements,
                            " int64 elements, got ",
                            timings->numel());
                timing_records = reinterpret_cast<TimingRecord*>(timings->data_ptr<int64_t>());
            }
            gemm_rs_blackwell::launch(A,
                                      B,
                                      C,
                                      ready,
                                      completion_epochs,
                                      swizzle_size,
                                      signals_per_tile,
                                      rs_strip_rows,
                                      timing_records);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("swizzle_size") = DEFAULT_SWIZZLE_SIZE,
        pybind11::arg("signals_per_tile") = DEFAULT_SIGNALS_PER_TILE,
        pybind11::arg("rs_strip_rows") = DEFAULT_RS_STRIP_ROWS,
        pybind11::arg("timings") = pybind11::none(),
        "Profile build of gemm_rs_blackwell_launch. Pass a zeroed int64 tensor of "
        "TIMING_NUM_BLOCKS * EVENTS_PER_BLOCK * 2 elements to record one launch.");

    pybind11::dict events;
    for (const auto& event : TIMING_EVENT_NAMES) {
        events[event.name] = static_cast<int>(event.id);
    }
    m.attr("TIMING_EVENTS") = events;
    m.attr("EVENTS_PER_BLOCK") = EVENTS_PER_BLOCK;
    m.attr("TIMING_RECORD_SIZE") = static_cast<int>(sizeof(TimingRecord));
    m.attr("TIMING_NUM_BLOCKS") = timing_num_blocks;
    m.attr("TIMING_PAYLOAD_WARP_SHIFT") = static_cast<int>(timing_profile::PAYLOAD_WARP_SHIFT);
#else
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
           int swizzle_size,
           int signals_per_tile,
           int rs_strip_rows) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(
                A, B, C, ready, completion_epochs, swizzle_size, signals_per_tile, rs_strip_rows);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("swizzle_size") = gemm_rs_blackwell::DEFAULT_SWIZZLE_SIZE,
        pybind11::arg("signals_per_tile") = gemm_rs_blackwell::DEFAULT_SIGNALS_PER_TILE,
        pybind11::arg("rs_strip_rows") = gemm_rs_blackwell::DEFAULT_RS_STRIP_ROWS,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "swizzle_size (1, 2, 4, or 8) is the tile-column width of each rasterization "
        "supergroup. signals_per_tile (1, 2, or 4) is how many ready-counter "
        "publications each 256x256 tile is split into; rs_strip_rows (8, 16, or 32) is the "
        "height of each reduce-scatter strip.");
#endif
}
