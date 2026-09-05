#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "comm/comm.cuh"
#include "common/cuda_checks.cuh"
#include "common/timings.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"

namespace gemm_ar_intranode_blackwell {
struct fused_globals;

template <int SUPERGROUP_WIDTH, int AR_UNROLL, int GEMM_TO_AR_SIGNAL_STRATEGY, int _NUM_COMP_SM>
void launch_fused_gemm_ar_blackwell(const fused_globals& G);

template <int _NUM_COMP_SM>
struct config_t {
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int NUM_COMP_SM = _NUM_COMP_SM;
    static constexpr int NUM_COMM_SM = NUM_BLOCKS - NUM_COMP_SM;
    // NOTE: I can just use a single warpgroup for both the consumer, producer and the epilogue
    // Maybe I can also save some SMs just for all-reduce?
    // I need to have a regular epilogue, and then do the all reduce -- maybe I can save SMs just
    // for this
    static constexpr int CONSUMER_WARPS = 2;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPS = 4 * CONSUMER_WARPS;
    // +1 padding warp: setmaxnreg is .sync.aligned over a whole warpgroup, so
    // the producer/consumer tail has to be a complete one.
    static constexpr int NUM_WARPS = CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS + 1;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;
    static constexpr int NUM_CLUSTERS = 2;

    // TMEM is allocated/freed with tcgen05 .cta_group::2, so a cluster has to be
    // wholly comp or wholly comm. A cluster straddling the split would leave one
    // CTA waiting on a peer that never allocates.
    static_assert(NUM_BLOCKS % NUM_CLUSTERS == 0, "The grid must be a whole number of clusters");
    static_assert(NUM_COMP_SM % NUM_CLUSTERS == 0,
                  "The comp/comm split must fall on a cluster boundary");

    static constexpr int PRODUCER_WARP_ID = EPILOGUE_WARPS;
    static constexpr int FIRST_CONSUMER_WARP_ID = PRODUCER_WARP_ID + PRODUCER_WARPS;

    static constexpr int EPILOGUE_WARPGROUPS = EPILOGUE_WARPS / kittens::WARPGROUP_WARPS;
    static_assert(EPILOGUE_WARPS % kittens::WARPGROUP_WARPS == 0,
                  "The epilogue warps must form whole warpgroups");
    static_assert(FIRST_CONSUMER_WARP_ID / kittens::WARPGROUP_WARPS >= EPILOGUE_WARPGROUPS,
                  "Producer/consumer warps must not share a warpgroup with the epilogue");

    // guard against register overallocation
    static constexpr int ALLOC_WARPS =
        ((NUM_WARPS + kittens::WARPGROUP_WARPS - 1) / kittens::WARPGROUP_WARPS) *
        kittens::WARPGROUP_WARPS;
    static constexpr int REGISTER_CEILING = (65536 / (ALLOC_WARPS * kittens::WARP_THREADS) / 8) * 8;
    static_assert(ALLOC_WARPS * kittens::WARP_THREADS * REGISTER_CEILING <= 65536,
                  "Register request does not fit the per-SM register file");

    static constexpr int NUM_WARPGROUPS = NUM_WARPS / kittens::WARPGROUP_WARPS;
    static constexpr int EPILOGUE_REGISTERS = 224;
    static constexpr int MAINLOOP_REGISTERS = 56;
    static_assert(EPILOGUE_REGISTERS * EPILOGUE_WARPGROUPS +
                          MAINLOOP_REGISTERS * (NUM_WARPGROUPS - EPILOGUE_WARPGROUPS) <=
                      REGISTER_CEILING * NUM_WARPGROUPS,
                  "Register split over-subscribes the launch register pool");

    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;

    // Bit positions inside the packed phasebits word, one bit per barrier
    // array. mma_finish and tma_load are shared across consumers, so they get a
    // single bit each; the epilogue barriers are per consumer / per epilogue
    // warpgroup, so they get CONSUMER_WARPS contiguous bits each.
    static constexpr int PHASE_BIT_MMA_FINISH = 0;
    static constexpr int PHASE_BIT_TMA_LOAD = PHASE_BIT_MMA_FINISH + 1;
    static constexpr int PHASE_BIT_EPILOGUE_FINISHED = PHASE_BIT_TMA_LOAD + 1;
    static constexpr int PHASE_BIT_EPILOGUE_READY = PHASE_BIT_EPILOGUE_FINISHED + CONSUMER_WARPS;
    static_assert(EPILOGUE_WARPGROUPS == CONSUMER_WARPS,
                  "The epilogue_ready phase bits are indexed by warpgroup but sized by consumer");
    static_assert(PHASE_BIT_EPILOGUE_READY + CONSUMER_WARPS <= 32,
                  "The phase bits must fit in a single 32-bit word");

    // Barriers whose first wait expects a completed phase start at 1:
    // mma_finish (the pipeline slot is free before the first MMA) and
    // epilogue_finished (tmem is free before the first epilogue).
    static constexpr uint32_t PHASE_BITS_INIT = (1u << PHASE_BIT_MMA_FINISH) |
        (((1u << CONSUMER_WARPS) - 1) << PHASE_BIT_EPILOGUE_FINISHED);
};

constexpr int _DEFAULT_NUM_COMP_SM = 128;
using config = config_t<_DEFAULT_NUM_COMP_SM>;

enum GemmToArSignalStrategy {
    PUSH = 0,  // write to host buffer to signal completion
    PULL = 1,  // write to own buffer, host will poll with multimem
};

// Launch configuration per M, as a single list so the dispatch switch below and
// the NUM_COMP_SM the profiler reports back to Python cannot drift apart. The
// comp/comm split is what tells the renderer which blocks are GEMM CTAs and
// which are all-reduce CTAs, so a stale copy would mislabel every row.
//
// X(M, SUPERGROUP_WIDTH, AR_UNROLL, SIGNAL_STRATEGY, NUM_COMP_SM)
//
// AR_UNROLL 43 at M=32768 is the ceiling: (128*256/2) 4-byte register loads
// over 384 threads is 42.67 operations per thread, so a higher unroll is inert.
#define GEMM_AR_BLACKWELL_CONFIGS(X)                          \
    X(2048, 4, 16, GemmToArSignalStrategy::PULL, 108)         \
    X(4096, 4, 16, GemmToArSignalStrategy::PULL, 108)         \
    X(8192, 8, 16, GemmToArSignalStrategy::PULL, 100)         \
    X(16384, 8, 16, GemmToArSignalStrategy::PULL, 100)        \
    X(32768, 8, 43, GemmToArSignalStrategy::PUSH, 140)

// -1 for an M the dispatch switch does not handle.
__host__ inline int gemm_ar_blackwell_num_comp_sm(int M) {
    switch (M) {
#define GEMM_AR_BLACKWELL_CASE_COMP_SM(M_, SG_, UNROLL_, SIGNAL_, COMP_SM_) \
    case (M_):                                                             \
        return COMP_SM_;
        GEMM_AR_BLACKWELL_CONFIGS(GEMM_AR_BLACKWELL_CASE_COMP_SM)
#undef GEMM_AR_BLACKWELL_CASE_COMP_SM
        default:
            return -1;
    }
}

#ifdef PROFILE_TIMINGS
// Append-only: these values are the ABI of every saved .npz. Renumbering makes
// historical traces decode as the wrong phases.
enum TimingEvent : uint32_t {
    // Shared prologue (semaphore init, TMEM provisioning, cluster sync).
    EV_SETUP_BEGIN = 0,
    EV_SETUP_DONE = 1,

    // Producer warp: one pipeline step per K block.
    EV_LOAD_STEP_BEGIN = 2,   // top of the step, before waiting on the ring slot
    EV_LOAD_MMA_FREE = 3,     // mma_finish observed: the stage is free to refill
    EV_LOAD_TMA_ISSUED = 4,   // A/B cluster loads issued for the stage

    // Consumer (MMA) warps.
    EV_MMA_TILE_BEGIN = 5,    // top of an output tile
    EV_MMA_TMEM_FREE = 6,     // epilogue released the accumulator
    EV_MMA_STEP_BEGIN = 7,    // top of a K step
    EV_MMA_INPUTS_READY = 8,  // tma_load observed for the stage
    EV_MMA_ISSUED = 9,        // mm2/mma2 issued

    // Epilogue warpgroups.
    EV_EPI_TILE_BEGIN = 10,    // top of an output tile
    EV_EPI_MMA_DONE = 11,      // epilogue_ready observed: the mainloop is done
    EV_EPI_TMEM_READ = 12,     // accumulator drained TMEM -> registers
    EV_EPI_SMEM_WRITTEN = 13,  // all chunks staged to SMEM and their TMA stores issued
    EV_EPI_STORE_DONE = 14,    // TMA stores drained to the distributed buffer
    EV_EPI_SIGNALLED = 15,     // comp -> comm barrier written

    // Comm (all-reduce) CTAs.
    EV_AR_TILE_BEGIN = 16,   // top of a comm tile, before polling the barrier
    EV_AR_SIGNAL_SEEN = 17,  // the GEMM signal for this tile landed
    EV_AR_TILE_DONE = 18,    // multimem reduce + store for the tile finished
};

// Kept beside the enum so the pybind export and the enum cannot diverge.
#define GEMM_AR_BLACKWELL_TIMING_EVENTS(X)  \
    X(SETUP_BEGIN)                          \
    X(SETUP_DONE)                           \
    X(LOAD_STEP_BEGIN)                      \
    X(LOAD_MMA_FREE)                        \
    X(LOAD_TMA_ISSUED)                      \
    X(MMA_TILE_BEGIN)                       \
    X(MMA_TMEM_FREE)                        \
    X(MMA_STEP_BEGIN)                       \
    X(MMA_INPUTS_READY)                     \
    X(MMA_ISSUED)                           \
    X(EPI_TILE_BEGIN)                       \
    X(EPI_MMA_DONE)                         \
    X(EPI_TMEM_READ)                        \
    X(EPI_SMEM_WRITTEN)                     \
    X(EPI_STORE_DONE)                       \
    X(EPI_SIGNALLED)                        \
    X(AR_TILE_BEGIN)                        \
    X(AR_SIGNAL_SEEN)                       \
    X(AR_TILE_DONE)
#endif  // PROFILE_TIMINGS

struct fused_globals {
    static constexpr int PIPELINE_STAGES = 4;
    // NOTE: this would hide the smem -> gmem stores behind the rmem -> smem stores. It is likely
    // that NUM_C_TILES is larger at bigger tile sizes
    // the benefit of this is that we can save on SMEM budget to expand later
    // EPILOGUE STAGES states how many times we split the epilogue loads
    static constexpr int EPILOGUE_STAGES = 8;
    static constexpr int NUM_C_TILES = 2;
    static_assert(EPILOGUE_STAGES % NUM_C_TILES == 0,
                  "The column split must be a whole number of staging-buffer rings");
    static constexpr int ROW_BLOCK = 256;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = kittens::st_bf<ROW_BLOCK / config::CONSUMER_WARPS, RED_BLOCK>;

    static_assert(COL_BLOCK % config::NUM_CLUSTERS == 0, "COL_BLOCK should be divisible");
    using B_tile = kittens::st_bf<RED_BLOCK, COL_BLOCK / config::NUM_CLUSTERS>;

    using C_tt_tile = kittens::tt<float, ROW_BLOCK / config::CONSUMER_WARPS, COL_BLOCK>;
    static_assert(config::CONSUMER_WARPS * C_tt_tile::cols <= kittens::MAX_TENSOR_COLS,
                  "The TMEM accumulators for all consumers must fit in tensor memory");

    static_assert(COL_BLOCK % EPILOGUE_STAGES == 0, "COL_BLOCK should be divisible");
    using C_tile = kittens::st_bf<ROW_BLOCK / config::CONSUMER_WARPS, COL_BLOCK / EPILOGUE_STAGES>;

    static constexpr int DYNAMIC_SHARED_MEMORY =
        ((sizeof(A_tile) * config::CONSUMER_WARPS + sizeof(B_tile)) * PIPELINE_STAGES) +
        (sizeof(C_tile) * NUM_C_TILES * config::CONSUMER_WARPS) +
        1024;  // NOTE: must add 1024 so this can be aligned by TK
    static_assert(DYNAMIC_SHARED_MEMORY <= 227 * 1024, "SMEM allocation too large");

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;
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
    barrier_distributed_tensor comp_comm_barrier;

    int dev_idx;
    int M;
    int N;
    int K;

    // used so that launches can be chained together
    int epoch;

#ifdef PROFILE_TIMINGS
    // NUM_BLOCKS * EVENTS_PER_BLOCK records, partitioned by blockIdx.x so CTAs
    // never contend. Null on an unprofiled launch, which short-circuits the
    // store in emit_timing_impl.
    ::mkernel_timings::TimingRecord* timings;
#endif

    struct pipeline_inputs {
        A_tile A[config::CONSUMER_WARPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };
};

__host__ inline fused_globals gemm_ar_blackwell_make_globals(const at::Tensor& A,
                                                             const at::Tensor& B,
                                                             dist::ParallelBuffer& C,
                                                             dist::ParallelBuffer& barrier,
                                                             dist::ParallelBuffer& C_final,
                                                             int dev_idx,
                                                             int M,
                                                             int N,
                                                             int K,
                                                             int epoch,
                                                             uint64_t timings_ptr) {
    (void)timings_ptr;
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
        .K = K,
        .epoch = epoch,
#ifdef PROFILE_TIMINGS
        .timings = reinterpret_cast<::mkernel_timings::TimingRecord*>(timings_ptr),
#endif
    };
}

void entrypoint(const at::Tensor& A,
                const at::Tensor& B,
                dist::ParallelBuffer& C,
                dist::ParallelBuffer& barrier,
                dist::ParallelBuffer& C_final,
                const int epoch,
                // Device pointer to a NUM_BLOCKS * EVENTS_PER_BLOCK ring of
                // TimingRecords, or 0. Ignored unless built with
                // -DPROFILE_TIMINGS, so the signature is the same either way and
                // the bench does not need two call sites.
                const uint64_t timings_ptr = 0) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    fused_globals G = gemm_ar_blackwell_make_globals(
        A, B, C, barrier, C_final, dev_idx, M, N, K, epoch, timings_ptr);

    // int SUPERGROUP_WIDTH, int AR_UNROLL, int GEMM_TO_AR_SIGNAL_STRATEGY, int _NUM_COMP_SM
    switch (M) {
#define GEMM_AR_BLACKWELL_CASE_LAUNCH(M_, SG_, UNROLL_, SIGNAL_, COMP_SM_) \
    case (M_): {                                                          \
        launch_fused_gemm_ar_blackwell<SG_, UNROLL_, SIGNAL_, COMP_SM_>(G);\
        break;                                                            \
    }
        GEMM_AR_BLACKWELL_CONFIGS(GEMM_AR_BLACKWELL_CASE_LAUNCH)
#undef GEMM_AR_BLACKWELL_CASE_LAUNCH
    }
}
};  // namespace gemm_ar_intranode_blackwell
