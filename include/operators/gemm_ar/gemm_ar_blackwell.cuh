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

template <int SUPERGROUP_WIDTH, int AR_UNROLL, int GEMM_TO_AR_SIGNAL_STRATEGY, int COMP_SM>
void launch_fused_gemm_ar_blackwell(const fused_globals& G);

// Default comp/comm SM split. Only NUM_COMP_SM varies across the sweep;
// everything else in config_t is split-independent, so `config` (the default
// instantiation) stays valid everywhere that does not care about the split.
static constexpr int DEFAULT_NUM_COMP_SM = 128;

// The comp/comm split as a template parameter: comp SMs run the GEMM, the
// remaining NUM_BLOCKS - NUM_COMP_SM run the all-reduce. Sweeping this trades
// GEMM throughput against AR drain rate, and the optimum is shape dependent.
template <int NUM_COMP_SM_ = DEFAULT_NUM_COMP_SM>
struct config_t {
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int NUM_COMP_SM = NUM_COMP_SM_;
    static constexpr int NUM_COMM_SM = NUM_BLOCKS - NUM_COMP_SM;
    static_assert(NUM_COMP_SM > 0, "need at least one comp SM");
    static_assert(NUM_COMM_SM > 0,
                  "need at least one comm SM; the AR is never run by comp SMs");
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
    // Clusters are co-scheduled, so a cluster may not straddle the comp/comm
    // boundary: blockIdx.x < NUM_COMP_SM has to split whole clusters.
    static_assert(NUM_COMP_SM % NUM_CLUSTERS == 0,
                  "comp SMs must be a whole number of clusters");
    static_assert(NUM_BLOCKS % NUM_CLUSTERS == 0,
                  "grid must be a whole number of clusters");

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
};

using config = config_t<DEFAULT_NUM_COMP_SM>;

enum GemmToArSignalStrategy {
    PUSH = 0,  // write to host buffer to signal completion
    PULL = 1,  // write to own buffer, host will poll with multimem
};

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
                                                             int epoch) {
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
        .epoch = epoch};
}

// Comp/comm SM splits compiled into the module. Each entry costs a full set of
// kernel instantiations (x2 signal strategies x3 shape buckets), so the sweep is
// opt-in -- build with -DGEMM_AR_COMP_SM_SWEEP to get all of them. Every value
// must be even (whole clusters) and less than config_t<>::NUM_BLOCKS; config_t
// static_asserts both.
// Each axis list can be overridden from the build, e.g.
//   -D'GEMM_AR_FOR_EACH_UNROLL(F)=F(16)'
// which compiles just that value instead of the sweep default. Useful for
// pinning axes you have already settled and sweeping only the open ones.
#ifndef GEMM_AR_FOR_EACH_COMP_SM
#ifdef GEMM_AR_COMP_SM_SWEEP
// 144/140/136 extend toward the GEMM-bound end: at M=32768 the optimum sat at
// 132, the previous top of the range, and per-SM parity with cutlass says the
// remaining deficit is exactly the SMs not doing GEMM. These probe how few comm
// SMs the all-reduce can be squeezed into before it becomes the bottleneck.
#define GEMM_AR_FOR_EACH_COMP_SM(F) \
    F(144) F(140) F(136) F(132) F(128) F(126) F(124) F(122) F(118) F(116) \
    F(112) F(108) F(104) F(100)
#else
#define GEMM_AR_FOR_EACH_COMP_SM(F) F(128)
#endif
#endif

// AR_UNROLL = independent multimem load/store requests each AR thread keeps in
// flight. The non-sweep list keeps both values the shape heuristic picks (32
// for M <= 2048, 64 above), so a default build behaves exactly as before.
#ifndef GEMM_AR_FOR_EACH_UNROLL
#ifdef GEMM_AR_UNROLL_SWEEP
#define GEMM_AR_FOR_EACH_UNROLL(F) F(8) F(16) F(32) F(64)
#else
#define GEMM_AR_FOR_EACH_UNROLL(F) F(32) F(64)
#endif
#endif

// SIGNAL_DEPTH = how many tiles' TMA stores stay in flight when a tile is
// announced to the comm SMs. 0 drains fully and announces the tile just stored
// (the original behaviour). D > 0 leaves D tiles in flight and announces the
// tile D iterations back, trading a shorter epilogue stall for a later signal
// -- which gives the comm SMs their work later, so it can go either way.
#ifndef GEMM_AR_FOR_EACH_SIGNAL_DEPTH
#ifdef GEMM_AR_SIGNAL_DEPTH_SWEEP
#define GEMM_AR_FOR_EACH_SIGNAL_DEPTH(F) F(0) F(1) F(2)
#else
#define GEMM_AR_FOR_EACH_SIGNAL_DEPTH(F) F(0)
#endif
#endif

// Passed as ar_unroll to fall back on the shape heuristic instead of pinning a
// value. Must stay <= 0 so it can never collide with a real unroll factor.
static constexpr int AR_UNROLL_BY_SHAPE = 0;
static constexpr int DEFAULT_SIGNAL_DEPTH = 0;

inline int default_ar_unroll(int M) { return M <= 2048 ? 32 : 64; }

// SUPERGROUP_WIDTH stays derived from M rather than swept: it sets the tile
// walk, and comp and comm must agree on it or the barrier coordinates diverge.
template <int STRATEGY, int COMP_SM, int AR_UNROLL, int SIGNAL_DEPTH>
inline void gemm_ar_dispatch_shape(const fused_globals& G, int M) {
    if (M <= 4096) {
        launch_fused_gemm_ar_blackwell<4, AR_UNROLL, STRATEGY, COMP_SM, SIGNAL_DEPTH>(G);
    } else {
        launch_fused_gemm_ar_blackwell<8, AR_UNROLL, STRATEGY, COMP_SM, SIGNAL_DEPTH>(G);
    }
}

template <int STRATEGY, int COMP_SM, int AR_UNROLL>
inline bool gemm_ar_dispatch_depth(const fused_globals& G, int M, int signal_depth) {
    bool ok = false;
#define GEMM_AR_TRY_DEPTH(D)                                                    \
    if (!ok && signal_depth == (D)) {                                           \
        gemm_ar_dispatch_shape<STRATEGY, COMP_SM, AR_UNROLL, D>(G, M);           \
        ok = true;                                                              \
    }
    GEMM_AR_FOR_EACH_SIGNAL_DEPTH(GEMM_AR_TRY_DEPTH)
#undef GEMM_AR_TRY_DEPTH
    return ok;
}

// The comp_sm x unroll cross product is built from nested template functions
// rather than nested macros -- the macros stay one-dimensional and readable.
template <int STRATEGY, int COMP_SM>
inline bool gemm_ar_dispatch_unroll(const fused_globals& G,
                                    int M,
                                    int ar_unroll,
                                    int signal_depth) {
    bool matched = false, ok = false;
#define GEMM_AR_TRY_UNROLL(UN)                                                  \
    if (!matched && ar_unroll == (UN)) {                                        \
        matched = true;                                                         \
        ok = gemm_ar_dispatch_depth<STRATEGY, COMP_SM, UN>(G, M, signal_depth);  \
    }
    GEMM_AR_FOR_EACH_UNROLL(GEMM_AR_TRY_UNROLL)
#undef GEMM_AR_TRY_UNROLL
    // Distinguishing the two lets the caller name the axis that is missing.
    return matched && ok;
}

// Set to 0 to leave a strategy uninstantiated entirely -- halves the kernel
// count once you have settled on one. A call naming a disabled strategy fails
// the TORCH_CHECK in entrypoint rather than silently running the other one.
#ifndef GEMM_AR_ENABLE_PUSH
#define GEMM_AR_ENABLE_PUSH 1
#endif
#ifndef GEMM_AR_ENABLE_PULL
#define GEMM_AR_ENABLE_PULL 1
#endif
#if !GEMM_AR_ENABLE_PUSH && !GEMM_AR_ENABLE_PULL
#error "at least one of GEMM_AR_ENABLE_PUSH / GEMM_AR_ENABLE_PULL must be 1"
#endif

template <int COMP_SM>
inline bool gemm_ar_dispatch_strategy(const fused_globals& G,
                                      int M,
                                      int strategy,
                                      int ar_unroll,
                                      int signal_depth) {
    if (strategy == GemmToArSignalStrategy::PUSH) {
#if GEMM_AR_ENABLE_PUSH
        return gemm_ar_dispatch_unroll<GemmToArSignalStrategy::PUSH, COMP_SM>(
            G, M, ar_unroll, signal_depth);
#else
        return false;
#endif
    }
#if GEMM_AR_ENABLE_PULL
    return gemm_ar_dispatch_unroll<GemmToArSignalStrategy::PULL, COMP_SM>(
        G, M, ar_unroll, signal_depth);
#else
    return false;
#endif
}

// Lets the benchmark sweep exactly what was compiled instead of hardcoding a
// list that can drift from GEMM_AR_FOR_EACH_COMP_SM.
inline std::vector<int> compiled_comp_sm_splits() {
    std::vector<int> out;
#define GEMM_AR_COLLECT_COMP_SM(SM) out.push_back(SM);
    GEMM_AR_FOR_EACH_COMP_SM(GEMM_AR_COLLECT_COMP_SM)
#undef GEMM_AR_COLLECT_COMP_SM
    return out;
}

inline std::vector<int> compiled_ar_unrolls() {
    std::vector<int> out;
#define GEMM_AR_COLLECT_UNROLL(UN) out.push_back(UN);
    GEMM_AR_FOR_EACH_UNROLL(GEMM_AR_COLLECT_UNROLL)
#undef GEMM_AR_COLLECT_UNROLL
    return out;
}

inline std::vector<int> compiled_strategies() {
    std::vector<int> out;
#if GEMM_AR_ENABLE_PUSH
    out.push_back(GemmToArSignalStrategy::PUSH);
#endif
#if GEMM_AR_ENABLE_PULL
    out.push_back(GemmToArSignalStrategy::PULL);
#endif
    return out;
}

inline std::vector<int> compiled_signal_depths() {
    std::vector<int> out;
#define GEMM_AR_COLLECT_DEPTH(D) out.push_back(D);
    GEMM_AR_FOR_EACH_SIGNAL_DEPTH(GEMM_AR_COLLECT_DEPTH)
#undef GEMM_AR_COLLECT_DEPTH
    return out;
}

inline int num_blocks() { return config::NUM_BLOCKS; }

void entrypoint(const at::Tensor& A,
                const at::Tensor& B,
                dist::ParallelBuffer& C,
                dist::ParallelBuffer& barrier,
                dist::ParallelBuffer& C_final,
                const int epoch,
                int gemm_to_ar_signal_strategy,
                int num_comp_sm,
                int ar_unroll,
                int signal_depth) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    fused_globals G =
        gemm_ar_blackwell_make_globals(A, B, C, barrier, C_final, dev_idx, M, N, K, epoch);

    TORCH_CHECK((gemm_to_ar_signal_strategy == GemmToArSignalStrategy::PUSH &&
                 GEMM_AR_ENABLE_PUSH) ||
                    (gemm_to_ar_signal_strategy == GemmToArSignalStrategy::PULL &&
                     GEMM_AR_ENABLE_PULL),
                "Unknown or not-compiled gemm_to_ar_signal_strategy ",
                gemm_to_ar_signal_strategy,
                "; expected PUSH(0) or PULL(1)");

    const int unroll =
        (ar_unroll <= AR_UNROLL_BY_SHAPE) ? default_ar_unroll(M) : ar_unroll;

    bool matched_sm = false, launched = false;
#define GEMM_AR_TRY_COMP_SM(SM)                                                       \
    if (!launched && num_comp_sm == (SM)) {                                           \
        matched_sm = true;                                                            \
        launched = gemm_ar_dispatch_strategy<(SM)>(                                   \
            G, M, gemm_to_ar_signal_strategy, unroll, signal_depth);                  \
    }
    GEMM_AR_FOR_EACH_COMP_SM(GEMM_AR_TRY_COMP_SM)
#undef GEMM_AR_TRY_COMP_SM

    // Split the two failures so the message names the axis that is missing
    // rather than blaming whichever was checked first.
    TORCH_CHECK(matched_sm,
                "num_comp_sm=",
                num_comp_sm,
                " is not compiled into this module; rebuild with "
                "-DGEMM_AR_COMP_SM_SWEEP or call compiled_comp_sm_splits()");
    TORCH_CHECK(launched,
                "ar_unroll=",
                unroll,
                " / signal_depth=",
                signal_depth,
                " is not compiled into this module; rebuild with "
                "-DGEMM_AR_UNROLL_SWEEP / -DGEMM_AR_SIGNAL_DEPTH_SWEEP, or call "
                "compiled_ar_unrolls() / compiled_signal_depths()");
}
};  // namespace gemm_ar_intranode_blackwell
