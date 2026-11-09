#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"
#include "pybind11/stl.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    // Shape of the int32 `sync` DistBuffer (allocate with multicast=True; PULL reads it
    // through the multicast VA). It is fixed, so one buffer serves every M and N, but
    // don't switch sync variants on a buffer without re-running prepare.
    m.attr("gemm_rs_blackwell_sync_shape") =
        pybind11::make_tuple(gemm_rs_blackwell::SYNC_BUFFER_WORDS);
    m.def(
        "gemm_rs_blackwell_prepare",
        [](at::Tensor A, at::Tensor B, dist::ParallelBuffer& C, dist::ParallelBuffer& sync) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::prepare(A, B, C, sync);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("sync"),
        "Initialize local counters once. Synchronize the stream and all ranks before launch.");
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& sync,
           std::optional<std::string> signal_strategy,
           std::optional<bool> fused_exit_barrier) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            TORCH_CHECK(signal_strategy.has_value() == fused_exit_barrier.has_value(),
                        "pass both signal_strategy and fused_exit_barrier, or neither");
            std::optional<gemm_rs_blackwell::SyncVariant> variant;
            if (signal_strategy) {
                TORCH_CHECK(*signal_strategy == "push" || *signal_strategy == "pull",
                            "signal_strategy must be 'push' or 'pull', got ",
                            *signal_strategy);
                variant = gemm_rs_blackwell::SyncVariant{
                    *signal_strategy == "push" ? gemm_rs_blackwell::PUSH
                                               : gemm_rs_blackwell::PULL,
                    *fused_exit_barrier};
            }
            gemm_rs_blackwell::entrypoint(A, B, C, sync, nullptr, variant);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("sync"),
        pybind11::arg("signal_strategy") = std::nullopt,
        pybind11::arg("fused_exit_barrier") = std::nullopt,
        "Launch GEMM+RS and its exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "The tile config is chosen from a per-M tuned table (M <= 32768). "
        "signal_strategy ('push' | 'pull') and fused_exit_barrier override the tuned "
        "sync variant; give both or neither.");
}
