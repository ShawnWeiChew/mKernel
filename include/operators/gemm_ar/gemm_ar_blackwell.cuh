#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "comm/comm.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"

namespace gemm_ar_intranode_blackwell {
struct fused_globals;
void launch_fused_gemm_ar_blackwell(const fused_globals& G);

struct config {
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    // static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    // NOTE: I can just use a single warpgroup for both the consumer, producer and the epilogue
    // Maybe I can also save some SMs just for all-reduce?
    // I need to have a regular epilogue, and then do the all reduce -- maybe I can save SMs just
    // for this
    static constexpr int CONSUMER_WARPS = 1;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPS = 4;
    static constexpr int NUM_CLUSTERS = 2;
    // TODO: get a number for this
    // static constexpr int INTRANODE_COMM_WARPS = ???;
    static constexpr int NUM_WARPS = CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

    static constexpr int PRODUCER_REGISTERS = 40;
    static constexpr int CONSUMER_REGISTERS = 232;

    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
};

struct fused_globals {
    // TODO: tune
    static constexpr int PIPELINE_STAGES = 5;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;
    static constexpr int MMA_K = 16;

    using A_tile = kittens::st_bf<ROW_BLOCK, RED_BLOCK>;

    // NOTE: I am storing it as BT
    static_assert(COL_BLOCK % config::NUM_CLUSTERS == 0, "COL_BLOCK should be divisible");
    using B_tile = kittens::st_bf<RED_BLOCK, COL_BLOCK / config::NUM_CLUSTERS>;
    // TODO: benchmark against writing to SMEM and then to GMEM,
    // compared to just writing to GMEM

    using C_tt_tile = kittens::tt<float, ROW_BLOCK, COL_BLOCK>;
    using C_tile = kittens::st_bf<ROW_BLOCK, COL_BLOCK>;

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;

    // I assume that this gives me a pointer to global memory, not sure
    // this part is so sketchy help
    using C_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, C_tile>;
    using C_distributed_tensor =
        dist::distributed_tensor<C_local_tensor, config::NUM_DEVICES, true>;
    using C_final_tensor = dist::distributed_tensor<C_local_tensor, config::NUM_DEVICES, true>;
    using barrier_distributed_tensor = dist::barrier_distributed_tensor<config::NUM_DEVICES>;

    A_local_tensor A;
    B_local_tensor B;

    C_final_tensor C_final;
    // write to the distributed tensor first, then ld into registers and then into C
    C_distributed_tensor C_dist;

    // barriers

    // TODO: scope it to a tile later, start with global barrier
    barrier_distributed_tensor comp_comm_barrier;

    int dev_idx;
    int M;
    int N;
    int K;
};

__host__ inline fused_globals gemm_ar_blackwell_make_globals(const at::Tensor& A,
                                                             const at::Tensor& B,
                                                             dist::ParallelBuffer& C,
                                                             dist::ParallelBuffer& barrier,
                                                             dist::ParallelBuffer& C_final,
                                                             int dev_idx,
                                                             int M,
                                                             int N,
                                                             int K) {
    return {
        .A = ::dist::local_tensor_from_tensor<fused_globals::A_local_tensor>(A),
        .B = ::dist::local_tensor_from_tensor<fused_globals::B_local_tensor>(B),
        .C_final =
            ::dist::distributed_tensor_from_buffer<fused_globals::C_distributed_tensor>(C_final),
        .C_dist = ::dist::distributed_tensor_from_buffer<fused_globals::C_distributed_tensor>(C),
        .comp_comm_barrier =
            ::dist::distributed_tensor_from_buffer<fused_globals::barrier_distributed_tensor>(
                barrier),
        .dev_idx = dev_idx,
        .M = M,
        .N = N,
        .K = K};
}

void entrypoint(const at::Tensor& A,
                const at::Tensor& B,
                dist::ParallelBuffer& C,
                dist::ParallelBuffer& barrier,
                dist::ParallelBuffer& C_final) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    fused_globals G = gemm_ar_blackwell_make_globals(A, B, C, barrier, C_final, dev_idx, M, N, K);

    launch_fused_gemm_ar_blackwell(G);
    MKERNEL_CUDACHECK(cudaGetLastError());
    MKERNEL_CUDACHECK(cudaDeviceSynchronize());
}

};  // namespace gemm_ar_intranode_blackwell
