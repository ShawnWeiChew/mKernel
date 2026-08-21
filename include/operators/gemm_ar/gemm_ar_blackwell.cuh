#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>
#include <unordered_map>
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

// Compile-time profiler toggle. The instrumentation costs %globaltimer reads
// and global stores on the critical path, so it stays out of the SASS unless
// this is on. Build with -DGEMM_AR_BLACKWELL_PROFILE=1 to turn it on without
// touching the source.
#ifndef GEMM_AR_BLACKWELL_PROFILE
#define GEMM_AR_BLACKWELL_PROFILE 0
#endif
inline constexpr bool PROFILE_ENABLED = GEMM_AR_BLACKWELL_PROFILE;

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
void launch_fused_gemm_ar_blackwell(const fused_globals& G);

struct config {
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int STATIC_SHARED_MEMORY = 1024;
    static constexpr int NUM_COMP_SM = 148;
    static constexpr int NUM_COMM_SM = NUM_BLOCKS - NUM_COMP_SM;
    // static constexpr int DYNAMIC_SHARED_MEMORY = MAX_SHARED_MEMORY - STATIC_SHARED_MEMORY;
    // NOTE: I can just use a single warpgroup for both the consumer, producer and the epilogue
    // Maybe I can also save some SMs just for all-reduce?
    // I need to have a regular epilogue, and then do the all reduce -- maybe I can save SMs just
    // for this
    static constexpr int CONSUMER_WARPS = 1;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPS = 4;
    static constexpr int NUM_CLUSTERS = 2;
    static constexpr int NUM_WARPS = CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS;
    static constexpr int NUM_THREADS = NUM_WARPS * kittens::WARP_THREADS;

    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
};

struct fused_globals {
    // TODO: tune
    static constexpr int PIPELINE_STAGES = 5;
    // Accumulator stages in TMEM. Each C_tt_tile is COL_BLOCK float columns
    // wide and TMEM only has MAX_TENSOR_COLS of them, so with COL_BLOCK == 256
    // exactly two accumulators fit. This is what epilogue_stage_id rings over.
    static constexpr int EPILOGUE_STAGES = 2;
    // NOTE: this would hide the smem -> gmem stores behind the rmem -> smem stores. It is likely
    // that NUM_C_TILES is larger at bigger tile sizes
    // the benefit of this is that we can save on SMEM budget to expand later
    static constexpr int NUM_C_TILES = 8;
    static constexpr int ROW_BLOCK = 128;
    static constexpr int COL_BLOCK = 256;
    static constexpr int RED_BLOCK = 64;

    using A_tile = kittens::st_bf<ROW_BLOCK, RED_BLOCK>;

    // NOTE: I am storing it as BT
    static_assert(COL_BLOCK % config::NUM_CLUSTERS == 0, "COL_BLOCK should be divisible");
    using B_tile = kittens::st_bf<RED_BLOCK, COL_BLOCK / config::NUM_CLUSTERS>;
    // TODO: benchmark against writing to SMEM and then to GMEM,
    // compared to just writing to GMEM

    using C_tt_tile = kittens::tt<float, ROW_BLOCK, COL_BLOCK>;
    static_assert(EPILOGUE_STAGES * C_tt_tile::cols <= kittens::MAX_TENSOR_COLS,
                  "The TMEM accumulators for all epilogue stages must fit in tensor memory");

    // The epilogue splits one COL_BLOCK-wide accumulator into NUM_C_TILES column
    // chunks and pushes them out through NUM_C_TILES shared staging buffers.
    static_assert(COL_BLOCK % NUM_C_TILES == 0, "COL_BLOCK should be divisible");
    using C_tile = kittens::st_bf<ROW_BLOCK, COL_BLOCK / NUM_C_TILES>;

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

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    // Profiling variables
    int num_entries;
    int64_t* data_ptr;
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

namespace detail {

// Key for the fused_globals cache: every input cuTensorMapEncodeTiled bakes
// into a descriptor. A descriptor is a pure function of (address, dims,
// strides, tile shape), so two calls agreeing on all of these below produce
// byte-identical descriptors. That makes a cache hit safe even if a buffer was
// freed and something else was allocated at the same address with the same
// shape -- the descriptor that would be rebuilt is the one already stored.
struct globals_key {
    static constexpr int ND = config::NUM_DEVICES;
    // A, B, then (multicast + ND locals) for each of C_dist/barrier/C_final,
    // then dev_idx, M, N, K.
    static constexpr int WORDS = 2 + 3 * (1 + ND) + 4;
    std::array<uint64_t, WORDS> w{};
    bool operator==(const globals_key& o) const { return w == o.w; }
};

struct globals_key_hash {
    size_t operator()(const globals_key& k) const {
        size_t h = 1469598103934665603ull;  // FNV-1a
        for (uint64_t v : k.w) {
            h ^= (size_t)v;
            h *= 1099511628211ull;
        }
        return h;
    }
};

inline globals_key make_globals_key(const at::Tensor& A,
                                    const at::Tensor& B,
                                    dist::ParallelBuffer& C,
                                    dist::ParallelBuffer& barrier,
                                    dist::ParallelBuffer& C_final,
                                    int dev_idx,
                                    int M,
                                    int N,
                                    int K) {
    globals_key k;
    int i = 0;
    k.w[i++] = reinterpret_cast<uint64_t>(A.data_ptr());
    k.w[i++] = reinterpret_cast<uint64_t>(B.data_ptr());
    for (dist::ParallelBuffer* pb : {&C, &barrier, &C_final}) {
        k.w[i++] = reinterpret_cast<uint64_t>(pb->multicast_ptr_);
        for (int d = 0; d < globals_key::ND; ++d) {
            void* raw = (d < (int)pb->raw_ptrs_.size()) ? pb->raw_ptrs_[d] : nullptr;
            k.w[i++] = reinterpret_cast<uint64_t>(raw);
        }
    }
    k.w[i++] = (uint64_t)dev_idx;
    k.w[i++] = (uint64_t)M;
    k.w[i++] = (uint64_t)N;
    k.w[i++] = (uint64_t)K;
    return k;
}

using globals_cache =
    std::unordered_map<globals_key, std::unique_ptr<fused_globals>, globals_key_hash>;

inline globals_cache& the_globals_cache() {
    static globals_cache c;
    return c;
}

inline std::mutex& the_globals_cache_mutex() {
    static std::mutex m;
    return m;
}

// gemm_ar_blackwell_make_globals runs cuTensorMapEncodeTiled once per
// local_tensor it builds: one for A, one for B, and one per device slot of
// both C_dist and C_final -- 2 + 2*NUM_DEVICES driver calls. Doing that on
// every launch put a flat ~13us of host time inside the caller's timing
// window, which at M=N=2048 was larger than the kernel itself, and cost the
// same in production. Build once per (pointers, shape) and reuse.
inline const fused_globals& cached_globals(const at::Tensor& A,
                                           const at::Tensor& B,
                                           dist::ParallelBuffer& C,
                                           dist::ParallelBuffer& barrier,
                                           dist::ParallelBuffer& C_final,
                                           int dev_idx,
                                           int M,
                                           int N,
                                           int K) {
    const globals_key key = make_globals_key(A, B, C, barrier, C_final, dev_idx, M, N, K);

    std::lock_guard<std::mutex> lk(the_globals_cache_mutex());
    globals_cache& cache = the_globals_cache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        it = cache
                 .emplace(key,
                          std::make_unique<fused_globals>(gemm_ar_blackwell_make_globals(
                              A, B, C, barrier, C_final, dev_idx, M, N, K)))
                 .first;
    }
    // unique_ptr, so the pointee survives a rehash by another thread.
    return *it->second;
}

}  // namespace detail

// Drop every cached fused_globals. Not required for correctness -- a stale
// entry can only be matched by a buffer at the same address with the same
// shape, which needs the same descriptor anyway -- but useful to reclaim the
// few KB per entry after a set of buffers is retired.
inline void clear_globals_cache() {
    std::lock_guard<std::mutex> lk(detail::the_globals_cache_mutex());
    detail::the_globals_cache().clear();
}

void entrypoint(const at::Tensor& A,
                const at::Tensor& B,
                dist::ParallelBuffer& C,
                dist::ParallelBuffer& barrier,
                dist::ParallelBuffer& C_final) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    const fused_globals& G = detail::cached_globals(A, B, C, barrier, C_final, dev_idx, M, N, K);

    if (M <= 4096) {
        launch_fused_gemm_ar_blackwell<4, PROFILE_ENABLED>(G);
    } else {
        launch_fused_gemm_ar_blackwell<8, PROFILE_ENABLED>(G);
    }
}

// Profiling variant of `entrypoint`. Always launches the DO_PROFILE=true
// instantiation, independent of the PROFILE_ENABLED build toggle, so a single
// .so carries both the fast kernel and the instrumented one.
//
// `profiler` is int64 [config::NUM_BLOCKS * config::NUM_WARPS, 1 + num_entries * 4],
// one row per warp. Row layout: [count, (sm_id, tag, start_ns, duration_ns) * count].
// Zero it before the run you intend to keep -- the kernel only writes the
// entries it records, and never clears stale ones.
void entrypoint_profile(const at::Tensor& A,
                        const at::Tensor& B,
                        dist::ParallelBuffer& C,
                        dist::ParallelBuffer& barrier,
                        dist::ParallelBuffer& C_final,
                        at::Tensor& profiler,
                        int64_t num_entries) {
    const int dev_idx = C.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    constexpr int64_t ROWS = (int64_t)config::NUM_BLOCKS * config::NUM_WARPS;
    TORCH_CHECK(profiler.scalar_type() == at::kLong, "profiler must be int64");
    TORCH_CHECK(profiler.is_cuda() && profiler.is_contiguous(),
                "profiler must be a contiguous CUDA tensor");
    TORCH_CHECK(profiler.dim() == 2 && profiler.size(0) >= ROWS,
                "profiler must be 2-D with at least ",
                ROWS,
                " rows (NUM_BLOCKS * NUM_WARPS), got ",
                profiler.sizes());
    // Exact, not >=: Profiler::init strides by (1 + num_entries*4), so a wider
    // row would put every warp's slot at the wrong offset.
    TORCH_CHECK(profiler.size(1) == 1 + num_entries * 4,
                "profiler row must hold exactly 1 + num_entries*4 = ",
                1 + num_entries * 4,
                " int64s, got ",
                profiler.size(1));

    const int M = A.size(0), K = A.size(1), N = B.size(1);

    // Copy: the cached globals are shared across launches and must not be
    // mutated, and these two fields are per-run.
    fused_globals G = detail::cached_globals(A, B, C, barrier, C_final, dev_idx, M, N, K);
    G.num_entries = (int)num_entries;
    G.data_ptr = profiler.data_ptr<int64_t>();

    if (M <= 4096) {
        launch_fused_gemm_ar_blackwell<4, true>(G);
    } else {
        launch_fused_gemm_ar_blackwell<8, true>(G);
    }
}

};  // namespace gemm_ar_intranode_blackwell
