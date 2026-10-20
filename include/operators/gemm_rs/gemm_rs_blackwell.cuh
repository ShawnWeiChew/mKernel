#pragma once

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/parallel_buffer.cuh"
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <type_traits>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"

namespace gemm_rs_blackwell {
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = 2;
static constexpr int DEFAULT_NUM_EPILOGUE_WARPS = 1;
static constexpr int DEFAULT_C_TILE_DIVISOR = 4;

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
struct fused_globals;

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
void launch_gemm_rs_blackwell(const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>& G);

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int CONSUMER_WARPS = 2;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = _NUM_EPILOGUE_WARPS;
    static constexpr int REDUCE_SCATTER_WARPS = 1;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;

    static constexpr int NUM_THREADS =
        (CONSUMER_WARPS + PRODUCER_WARPS + REDUCE_SCATTER_WARPS + EPILOGUE_WARPS) *
        kittens::WARP_THREADS;
    static constexpr int NUM_CLUSTERS = DEFAULT_NUM_CTA;

    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    static constexpr int TMEM_PIPELINE_STAGES =
        kittens::MAX_TENSOR_COLS / COL_BLOCK / CONSUMER_WARPS;
    static constexpr int NUM_TMEM_SLOTS = TMEM_PIPELINE_STAGES * CONSUMER_WARPS;

    // used for SMEM allocation
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = 4;
    static constexpr int EPILOGUE_PIPELINE_STAGES = 2;

    static constexpr int C_TILE_DIVISOR = _C_TILE_DIVISOR;

    using A_tile = kittens::st_bf<ROW_BLOCK / CONSUMER_WARPS, RED_BLOCK>;  // [M, K]
    using B_tile = kittens::st_bf<COL_BLOCK / NUM_CLUSTERS, RED_BLOCK>;    // [N, K]

    using C_tt_tile = kittens::tt<comm::bf16, ROW_BLOCK / CONSUMER_WARPS, COL_BLOCK>;  // [M, N]
    using C_tile = kittens::st_bf<ROW_BLOCK / CONSUMER_WARPS, COL_BLOCK / C_TILE_DIVISOR>;

    static constexpr int MAX_DYNAMIC_SHARED_MEMORY = 227 * 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (sizeof(A_tile) * CONSUMER_WARPS + sizeof(B_tile)) * PRODUCER_CONSUMER_PIPELINE_STAGES +
        sizeof(C_tile) * EPILOGUE_PIPELINE_STAGES + 1024;
    static constexpr bool SMEM_FITS = DYNAMIC_SHARED_MEMORY <= MAX_DYNAMIC_SHARED_MEMORY;

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, C_tile>;
    using C_dist_tensor = dist::distributed_tensor<C_local_tensor, NUM_DEVICES, true>;

    using C_ready_barrier = dist::barrier_distributed_tensor<NUM_DEVICES>;

    int dev_idx;
    int M;
    int N;
    int K;

    static constexpr int epoch = 1;

    A_local_tensor A;
    B_local_tensor B;
    C_dist_tensor C;

    C_ready_barrier C_barrier;

    cudaStream_t stream = nullptr;

    struct pipeline_inputs {
        A_tile A[CONSUMER_WARPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    // bit 0: TMA producer
    // bit 1: TMA consumer
    // bit 2-3: TMEM producer
    // bit 4-5: TMEM consumer
    // bit 6: TMEM allocate
    // bit 7: TMEM deallocate
    static constexpr int PHASE_BITS_INIT = 0b00001101;
};

template <typename>
inline constexpr bool always_false_v = false;

template <int _NUM_EPILOGUE_WARPS,
          int _C_TILE_DIVISOR,
          typename DistributedTensor,
          typename LocalTensor>
__host__ inline fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR> gemm_rs_blackwell_make_globals(
    LocalTensor& A,
    LocalTensor& B,
    DistributedTensor& C,
    DistributedTensor& C_barrier,
    int dev_idx,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>;
    if constexpr (std::is_same_v<LocalTensor, at::Tensor> &&
                  std::is_same_v<DistributedTensor, dist::ParallelBuffer>) {
        return {.A = dist::local_tensor_from_tensor(A),
                .B = dist::local_tensor_from_tensor(B),
                .C = dist::distributed_tensor_from_buffer(C),
                .C_barrier = dist::distributed_tensor_from_buffer(C_barrier),
                .dev_idx = dev_idx,
                .M = M,
                .N = N,
                .K = K,
                .stream = stream};
    } else {
        static_assert(always_false_v<LocalTensor>, "Unsupported LocalTensorType");
    }
}
};  // namespace gemm_rs_blackwell