#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "ATen/core/TensorBody.h"
#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    using fg = gemm_rs_blackwell::fused_globals<
        gemm_rs_blackwell::DEFAULT_NUM_EPILOGUE_WARPS,
        gemm_rs_blackwell::DEFAULT_C_TILE_DIVISOR>;
    m.attr("gemm_blackwell_num_consumer_warps") = fg::CONSUMER_WARPS;
    m.attr("gemm_blackwell_cluster_rows") = fg::ROW_BLOCK * fg::NUM_CLUSTERS;
    m.attr("gemm_blackwell_consumer_variants") = pybind11::make_tuple(1, 2);
    m.attr("rs_blackwell_blocks") = fg::NUM_BLOCKS;
    m.attr("rs_blackwell_stores_per_batch") = fg::RS_STORES_PER_BATCH;
    m.def(
        "rs_blackwell_launch",
        [](dist::ParallelBuffer& input, at::Tensor output, dist::ParallelBuffer& epochs,
           int tile_rows, bool with_exit_barrier, int num_warps) {
            gemm_rs_blackwell::launch_rs_only(
                input, output, epochs, tile_rows, num_warps, with_exit_barrier);
        },
        pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("epochs"),
        pybind11::arg("tile_rows") = 256, pybind11::arg("with_exit_barrier") = true,
        pybind11::arg("num_warps") = 1,
        "Standalone multimem SUM reduce-scatter. Initialize epochs once to zero and "
        "synchronize all input writers before launch. Keep input immutable until all "
        "ranks finish. Output is a separate compact [M/world_size,N] BF16 tensor.");
    m.def(
        "gemm_blackwell_launch",
        [](at::Tensor A, at::Tensor B, at::Tensor C, int num_consumer_warps) {
            gemm_rs_blackwell::launch_gemm_only(A, B, C, num_consumer_warps);
        },
        pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"),
        pybind11::arg("num_consumer_warps") = gemm_rs_blackwell::DEFAULT_NUM_CONSUMER_WARPS,
        "Single-GPU GEMM benchmark: C = A @ B.T. Requires RS disabled in the "
        "kernel source. Uses ordinary BF16 tensors, no exit barrier or PDL overlap.");
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
    m.def(
        "gemm_rs_blackwell_launch",
        [](at::Tensor A,
           at::Tensor B,
           dist::ParallelBuffer& C,
           dist::ParallelBuffer& ready,
           dist::ParallelBuffer& completion_epochs,
           bool with_exit_barrier) {
            c10::cuda::CUDAGuard device_guard(C.local_rank_);
            gemm_rs_blackwell::launch(A, B, C, ready, completion_epochs, with_exit_barrier);
        },
        pybind11::arg("A"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("ready"),
        pybind11::arg("completion_epochs"),
        pybind11::arg("with_exit_barrier") = true,
        "Launch GEMM+RS and the exit barrier on the current stream. The result is "
        "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
        "alive and serialize launches; graph replays advance the device epoch. "
        "Set with_exit_barrier=False only for benchmarking with RS disabled in the "
        "kernel; this skips the exit barrier and disables PDL launch overlap.");
}
