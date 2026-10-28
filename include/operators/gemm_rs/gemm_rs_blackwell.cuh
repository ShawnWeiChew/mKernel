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
#ifndef MKERNEL_GEMM_RS_NUM_CONSUMER_WARPS
#define MKERNEL_GEMM_RS_NUM_CONSUMER_WARPS 2
#endif
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = MKERNEL_GEMM_RS_NUM_CONSUMER_WARPS;
static_assert(DEFAULT_NUM_CONSUMER_WARPS == 1 || DEFAULT_NUM_CONSUMER_WARPS == 2,
              "NUM_CONSUMER_WARPS must be 1 or 2");
static constexpr int DEFAULT_NUM_EPILOGUE_WARPS = 1;
static constexpr int DEFAULT_C_TILE_DIVISOR = 4;

// One persistent 32-bit word on each rank; no multicast allocation is needed.
// Only the owning rank writes its completed epoch. Peers acquire-load it.
using completion_epoch_local_tensor = dist::local_tensor<uint32_t, 1, 1, 1, 1>;
using completion_epoch_dist_tensor =
    dist::distributed_tensor<completion_epoch_local_tensor, INTRA_NUM_DEVICES, false>;

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
struct fused_globals;

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
void launch_gemm_rs_blackwell(const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>& G,
                              bool with_exit_barrier = true);

// Call once before capture/replay, with no work using these buffers in flight.
// Wait for this stream's initialization to finish locally, then synchronize all
// ranks before the first launch. Keep the same tile layout and serialize launches
// on G.stream; both allocations must remain alive across all graph replays.
template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
void initialize_gemm_rs_blackwell_state(
    const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>& G);

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
struct fused_globals {
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int CONSUMER_WARPS = _NUM_CONSUMER_WARPS;
    static_assert(CONSUMER_WARPS == 1 || CONSUMER_WARPS == 2);
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = _NUM_EPILOGUE_WARPS;
    static constexpr int REDUCE_SCATTER_WARPS = 4;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;

    // Warp layout: [epilogue WG(s)] [producer, MMA warp(s), idle] [RS WG]. setmaxnreg
    // is .sync.aligned over a whole warpgroup, so the producer/MMA tail is padded
    // out with idle warps and the RS warps get a warpgroup of their own.
    static constexpr int PRODUCER_WARP_ID = EPILOGUE_WARPS;
    static constexpr int FIRST_CONSUMER_WARP_ID = PRODUCER_WARP_ID + PRODUCER_WARPS;
    static constexpr int MAINLOOP_WARPGROUP = EPILOGUE_WARPGROUPS;
    static constexpr int RS_WARPGROUP = MAINLOOP_WARPGROUP + 1;
    static constexpr int FIRST_RS_WARP_ID = RS_WARPGROUP * kittens::WARPGROUP_WARPS;
    static_assert(FIRST_CONSUMER_WARP_ID + CONSUMER_WARPS <= FIRST_RS_WARP_ID,
                  "Producer/MMA warps must fit in one warpgroup");
    static_assert(REDUCE_SCATTER_WARPS == kittens::WARPGROUP_WARPS,
                  "The RS warps must form exactly one warpgroup");

    static constexpr int NUM_WARPS = FIRST_RS_WARP_ID + REDUCE_SCATTER_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

    static constexpr int MAINLOOP_REGISTERS = 40;
    static constexpr int RS_REGISTERS = 160;
    static constexpr int EPILOGUE_REGISTERS = 224;
    // guard against register overallocation
    static_assert((MAINLOOP_REGISTERS + RS_REGISTERS) * kittens::WARPGROUP_THREADS +
                          EPILOGUE_REGISTERS * EPILOGUE_WARPS * kittens::WARP_THREADS <=
                      65536,
                  "setmaxnreg budget exceeds the register file");
    // setmaxnreg.dec can only lower a warp's count from the launch allocation.
    static constexpr int REGISTER_CEILING = (65536 / NUM_THREADS / 8) * 8;
    static_assert(MAINLOOP_REGISTERS <= REGISTER_CEILING && RS_REGISTERS <= REGISTER_CEILING,
                  "Decreased register counts must not exceed the launch allocation");
    static constexpr int NUM_CLUSTERS = DEFAULT_NUM_CTA;

    // Preserve the 128-row MMA tile per consumer; fewer consumers shrink the CTA tile.
    static constexpr int ROW_BLOCK = 128 * CONSUMER_WARPS;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    // Keep 16 wide-store results live: 128 packed 32-bit registers per RS lane.
    // Each batch issues 32 x 16-byte reductions before 16 x 32-byte stores.
    static constexpr int RS_STORES_PER_BATCH = 16;

    // One semaphore pair per consumer protects one live accumulator tile.
    // Adding stages requires separate per-slot semaphore phases.
    static constexpr int TMEM_PIPELINE_STAGES = 1;
    static constexpr int NUM_TMEM_SLOTS = TMEM_PIPELINE_STAGES * CONSUMER_WARPS;
    static_assert(NUM_TMEM_SLOTS * COL_BLOCK <= kittens::MAX_TENSOR_COLS);

    // used for SMEM allocation
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = 4;
    static constexpr int EPILOGUE_PIPELINE_STAGES = 2;

    static constexpr int C_TILE_DIVISOR = _C_TILE_DIVISOR;

    using A_tile = kittens::st_bf<ROW_BLOCK / CONSUMER_WARPS, RED_BLOCK>;  // [M, K]
    using B_tile = kittens::st_bf<COL_BLOCK / NUM_CLUSTERS, RED_BLOCK>;    // [N, K]

    using C_tt_tile = kittens::tt<float, ROW_BLOCK / CONSUMER_WARPS, COL_BLOCK>;  // [M, N]
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

    A_local_tensor A;
    B_local_tensor B;
    C_dist_tensor C;

    C_ready_barrier C_barrier;
    completion_epoch_dist_tensor completion_epochs;

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
    DistributedTensor& completion_epochs,
    int dev_idx,
    int M,
    int N,
    int K,
    cudaStream_t stream) {
    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>;
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if constexpr (std::is_same_v<LocalTensor, at::Tensor> &&
                  std::is_same_v<DistributedTensor, dist::ParallelBuffer>) {
        TORCH_CHECK(completion_epochs.data_.numel() == 1 &&
                        completion_epochs.data_.element_size() == sizeof(uint32_t),
                    "completion_epochs must contain one 32-bit word per rank");
        return {
            .dev_idx = dev_idx,
            .M = M,
            .N = N,
            .K = K,
            .A = dist::local_tensor_from_tensor<typename fg::A_local_tensor>(A),
            .B = dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
            .C = dist::distributed_tensor_from_buffer<typename fg::C_dist_tensor>(C),
            .C_barrier =
                dist::distributed_tensor_from_buffer<typename fg::C_ready_barrier>(C_barrier),
            .completion_epochs = dist::distributed_tensor_from_buffer<completion_epoch_dist_tensor>(
                completion_epochs),
            .stream = stream};
    } else {
        static_assert(always_false_v<LocalTensor>, "Unsupported LocalTensorType");
    }
#else
    static_assert(always_false_v<LocalTensor>, "Construct fused_globals directly without Torch");
#endif
}
};  // namespace gemm_rs_blackwell
