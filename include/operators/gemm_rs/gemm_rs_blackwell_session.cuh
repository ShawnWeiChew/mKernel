#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#ifdef PROFILE_TIMINGS
#include <optional>

#include <pybind11/stl.h>
#endif

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
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
    constexpr int timing_num_blocks =
        fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>::NUM_BLOCKS;
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
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
            gemm_rs_blackwell::launch(A, B, C, ready, completion_epochs, timing_records);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
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
           dist::ParallelBuffer& completion_epochs) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A, B, C, ready, completion_epochs);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch.");
#endif
}
