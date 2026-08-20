#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor_tensor.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
// clang-format on
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/gemm_ar/gemm_ar_blackwell.cuh"
#include "operators/gemm_ar/timings.cuh"

using namespace kittens;

namespace gemm_ar_intranode_blackwell {

// use snake-like pattern, referenced from:
// https://github.com/HazyResearch/ThunderKittens/blob/0230013a72b51338a137b50f69538ec69d4d4675/include/common/util.cuh#L367
template <int SUPERGROUP_WIDTH = 8>
__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int num_rows,
                                                                   int num_cols,
                                                                   int tile_idx) {
    const int supergroup_numel = num_rows * SUPERGROUP_WIDTH;
    const int supergroup_idx = tile_idx / supergroup_numel;

    const int row_idx = (tile_idx % supergroup_numel) / SUPERGROUP_WIDTH;
    const int col_idx = supergroup_idx * SUPERGROUP_WIDTH + tile_idx % SUPERGROUP_WIDTH;

    return {(supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx};
};

// ============================================================================
// In-kernel timing events
// ============================================================================
//
// Append-only. These integers are the ABI of every .npz already on disk: the
// renderer looks phases up by name, but a trace saved before an id moved will
// decode to the wrong phase. Add at the end, never renumber.
//
// The kernel runs six warps per CTA in three roles, and the interesting
// question is always which of them is waiting on which, so each role gets a
// chain of milestones rather than isolated begin/end pairs: consecutive
// milestones of one loop iteration share a payload, and the phase table on the
// host pairs (m[i], m[i+1]) into a span. That is one emit per boundary instead
// of two.
enum TimingEvent : uint32_t {
    // every warp, once: kernel entry -> after the cluster-wide barrier
    EV_SETUP_BEGIN = 0,
    EV_SETUP_DONE = 1,

    // producer warp (warp 4), once per K step
    EV_LOAD_STEP_BEGIN = 2,   // about to wait for the stage's MMA to drain
    EV_LOAD_MMA_FREE = 3,     // stage is free; about to issue the TMA loads
    EV_LOAD_TMA_ISSUED = 4,   // both load_async issued

    // MMA warp (warp 5 of cta_rank 0), once per output tile
    EV_MMA_TILE_BEGIN = 5,    // about to wait for the epilogue to free tmem
    EV_MMA_TMEM_FREE = 6,     // tmem accumulator is ours
    // ...and once per K step
    EV_MMA_STEP_BEGIN = 7,    // about to wait for this stage's TMA arrival
    EV_MMA_INPUTS_READY = 8,  // A/B are in smem; about to issue the MMA
    EV_MMA_ISSUED = 9,        // mm2_AB / mma2_AB issued

    // epilogue warps (0-3), once per output tile
    EV_EPI_TILE_BEGIN = 10,    // about to wait for the mainloop's commit
    EV_EPI_MMA_DONE = 11,      // accumulator is complete
    EV_EPI_TMEM_READ = 12,     // tmem -> registers done (tmem released here)
    EV_EPI_SMEM_WRITTEN = 13,  // registers -> smem done, warpgroup synced
    EV_EPI_TMA_ISSUED = 14,    // TMA store to C issued
};

// Exported to Python at module init so a .npz is self-describing and can be
// re-rendered long after this enum has grown. Keep in sync with the enum.
inline constexpr struct {
    const char* name;
    uint32_t id;
} TIMING_EVENT_TABLE[] = {
    {"SETUP_BEGIN", EV_SETUP_BEGIN},
    {"SETUP_DONE", EV_SETUP_DONE},
    {"LOAD_STEP_BEGIN", EV_LOAD_STEP_BEGIN},
    {"LOAD_MMA_FREE", EV_LOAD_MMA_FREE},
    {"LOAD_TMA_ISSUED", EV_LOAD_TMA_ISSUED},
    {"MMA_TILE_BEGIN", EV_MMA_TILE_BEGIN},
    {"MMA_TMEM_FREE", EV_MMA_TMEM_FREE},
    {"MMA_STEP_BEGIN", EV_MMA_STEP_BEGIN},
    {"MMA_INPUTS_READY", EV_MMA_INPUTS_READY},
    {"MMA_ISSUED", EV_MMA_ISSUED},
    {"EPI_TILE_BEGIN", EV_EPI_TILE_BEGIN},
    {"EPI_MMA_DONE", EV_EPI_MMA_DONE},
    {"EPI_TMEM_READ", EV_EPI_TMEM_READ},
    {"EPI_SMEM_WRITTEN", EV_EPI_SMEM_WRITTEN},
    {"EPI_TMA_ISSUED", EV_EPI_TMA_ISSUED},
};

// Emit helpers for fused_comp_sm. Both pull `G`, `s_timing_head`, `warp_id`,
// `timing_seq` and `DO_PROFILE` out of the enclosing scope, so they are only
// usable inside that function (and its lambdas) and are #undef'd right after.
//
// TIMING_MARK assumes the caller is already down to a single lane -- the
// producer and MMA warps run their entire body inside one elect_warp_leader().
// TIMING_MARK_LEADER elects a lane itself and is for the epilogue warps, where
// all 32 lanes are converged. Never nest them: elect.sync with a full member
// mask, issued by a lane that is already the sole survivor of an earlier
// elect, is not a converged execution.
//
// TIMING_NEXT_SEQ closes an iteration's pairing key. Every mark of one loop
// iteration must be emitted before it, and every iteration must call it, or
// two iterations share a key and the pairer matches across them.
#ifdef PROFILE_TIMINGS
#define TIMING_MARK(eid)                                                       \
    do {                                                                       \
        if constexpr (DO_PROFILE) {                                            \
            EMIT(G.timings,                                                    \
                 &s_timing_head,                                               \
                 (eid),                                                        \
                 ::mkernel_timings::pack_payload(warp_id, timing_seq));        \
        }                                                                      \
    } while (0)
#define TIMING_MARK_LEADER(eid)                                                \
    do {                                                                       \
        if constexpr (DO_PROFILE) {                                            \
            if (elect_warp_leader()) TIMING_MARK(eid);                         \
        }                                                                      \
    } while (0)
#define TIMING_NEXT_SEQ()                                                      \
    do {                                                                       \
        if constexpr (DO_PROFILE) ++timing_seq;                                \
    } while (0)
#else
#define TIMING_MARK(eid) ((void)0)
#define TIMING_MARK_LEADER(eid) ((void)0)
#define TIMING_NEXT_SEQ() ((void)0)
#endif

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();

#ifdef PROFILE_TIMINGS
    // One head for the whole CTA, in shared memory. All six warps write into
    // this CTA's slice of the ring and the atomicAdd hands out dense,
    // non-aliasing indices across them; a head per warp would restart at 0 in
    // every warp and stack six timelines on top of each other.
    __shared__ uint32_t s_timing_head;
    // Pairing key for the current loop iteration; the warp id packed into its
    // top bits is what keeps warps sharing the head from colliding.
    uint32_t timing_seq = 0;
    if constexpr (DO_PROFILE) {
        if (threadIdx.x == 0) s_timing_head = 0;
        // Nothing has diverged yet at kernel entry, so this is safe here and
        // only here -- it must land before any warp's first emit.
        __syncthreads();
    }
#endif

    TIMING_MARK_LEADER(EV_SETUP_BEGIN);

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.prefetch_tma<fused_globals::A_tile>();
        G.B.prefetch_tma<fused_globals::B_tile>();
        G.C_dist[G.dev_idx].prefetch_tma<fused_globals::C_tile>();
    }

    // One CLUSTER computes one output block, not one CTA. mm2_AB is a
    // cta_group::2 MMA, so M = A::rows * ncta = 256 and N = B::cols * ncta =
    // 256: each CTA feeds its own 128 rows of A and its own 128 columns of B
    // into the shared instruction, and its accumulator keeps the 128 output
    // rows belonging to its own A rows. The tile walk therefore has to be
    // indexed by cluster; indexing it by blockIdx.x pairs two unrelated tiles
    // inside one MMA, which leaves exactly half of every stored tile wrong.
    const int num_row_tiles = G.M / (fused_globals::ROW_BLOCK * config::NUM_CLUSTERS);
    const int num_col_tiles = G.N / fused_globals::COL_BLOCK;
    const int num_tiles_total = num_row_tiles * num_col_tiles;
    const int cluster_idx = blockIdx.x / config::NUM_CLUSTERS;
    const int num_comp_clusters = config::NUM_COMP_SM / config::NUM_CLUSTERS;

    // allocate smem and tmem
    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    fused_globals::pipeline_inputs(&inputs_smem)[fused_globals::PIPELINE_STAGES] =
        smem_allocator.allocate<fused_globals::pipeline_inputs, fused_globals::PIPELINE_STAGES>();
    fused_globals::C_tile& C_smem = smem_allocator.allocate<fused_globals::C_tile>();

    __shared__ semaphore tma_load[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore mma_finish[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore epilogue_ready[fused_globals::EPILOGUE_STAGES];
    __shared__ semaphore epilogue_finished[fused_globals::EPILOGUE_STAGES];

    tensor_allocator<1, config::NUM_CLUSTERS> tm_alloc{};

    // combined phasebits, one bit per barrier array (each flips once per full
    // ring traversal, so the bit is toggled when the stage index wraps to 0):
    // bit 3: epilogue_ready    - starts at 0
    // bit 2: epilogue_finished - starts at 1
    // bit 1: tma_load          - starts at 0
    // bit 0: mma_finish        - starts at 1
    uint32_t phasebits = 0b0101;
    fused_globals::C_tt_tile tmem[fused_globals::EPILOGUE_STAGES] = {
        tm_alloc.allocate<fused_globals::C_tt_tile>(0),
        tm_alloc.allocate<fused_globals::C_tt_tile>(fused_globals::COL_BLOCK)};

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; i++) {
            init_semaphore(tma_load[i], 0, 2);
            init_semaphore(mma_finish[i], 0, 1);
        }

#pragma unroll
        for (int i = 0; i < fused_globals::EPILOGUE_STAGES; i++) {
            init_semaphore(epilogue_ready[i], 0, 1);
            init_semaphore(epilogue_finished[i], WARPGROUP_WARPS * config::NUM_CLUSTERS, 0);
        }
    }
    everyone::tma::cluster::sync();

    TIMING_MARK_LEADER(EV_SETUP_DONE);
    TIMING_NEXT_SEQ();

    // tile_row_idx is this CTA's A row tile (ROW_BLOCK units, already rank
    // adjusted); tile_col_idx is the cluster's C column tile (COL_BLOCK units).
    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            TIMING_MARK(EV_LOAD_STEP_BEGIN);
            wait(mma_finish[input_stage_id], (phasebits & 0b1));
            TIMING_MARK(EV_LOAD_MMA_FREE);

            tma::cluster::expect_bytes(
                tma_load[input_stage_id],
                sizeof(fused_globals::A_tile) + sizeof(fused_globals::B_tile),
                0);

            tma::cluster::load_async(A_smem,
                                     G.A,
                                     {tile_row_idx, i},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);
            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {i,
                                      (tile_col_idx * config::NUM_CLUSTERS +
                                       cta_rank)},  // this has to be here becuase the epilogue
                                                    // warp loads the tiles in col units of 256
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            TIMING_MARK(EV_LOAD_TMA_ISSUED);
            TIMING_NEXT_SEQ();

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 1;
            }
        }
    };

    // each only handles 16
    auto consume = [&](int& input_stage_id, int& epilogue_stage_id) {
        TIMING_MARK(EV_MMA_TILE_BEGIN);
        wait(epilogue_finished[epilogue_stage_id], (phasebits >> 2) & 0b1);
        TIMING_MARK(EV_MMA_TMEM_FREE);
        TIMING_NEXT_SEQ();

        {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            TIMING_MARK(EV_MMA_STEP_BEGIN);
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            TIMING_MARK(EV_MMA_INPUTS_READY);

            mm2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            TIMING_MARK(EV_MMA_ISSUED);
            TIMING_NEXT_SEQ();

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;

            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int i = 1; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            TIMING_MARK(EV_MMA_STEP_BEGIN);
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            TIMING_MARK(EV_MMA_INPUTS_READY);

            mma2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            TIMING_MARK(EV_MMA_ISSUED);
            TIMING_NEXT_SEQ();

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready[epilogue_stage_id]);

        epilogue_stage_id = (epilogue_stage_id + 1) % fused_globals::EPILOGUE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << 2);
        }
    };

    auto epilogue = [&](int tile_row_idx, int tile_col_idx, int& epilogue_stage_id) {
        TIMING_MARK_LEADER(EV_EPI_TILE_BEGIN);
        wait(epilogue_ready[epilogue_stage_id], (phasebits >> 3) & 0b1);
        TIMING_MARK_LEADER(EV_EPI_MMA_DONE);
        tensor_after_thread_sync();

        rt_bf<fused_globals::ROW_BLOCK / 4, fused_globals::COL_BLOCK> c_reg;
        warpgroup::load_async(c_reg, tmem[epilogue_stage_id]);
        tensor_load_wait();
        TIMING_MARK_LEADER(EV_EPI_TMEM_READ);

        // signal tmem empty
        if (elect_warp_leader()) {
            // TODO: move this into dist namespace
            tma::cluster::arrive(epilogue_finished[epilogue_stage_id], 0);
        }
        warpgroup::sync(1);
        // this already does the swizzle inside it
        warpgroup::store(C_smem, c_reg);
        warpgroup::sync(1);
        TIMING_MARK_LEADER(EV_EPI_SMEM_WRITTEN);

        if (warpgroup::laneid() == 0) {
            dist::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                G.C_dist[G.dev_idx], C_smem, {tile_row_idx, tile_col_idx});
            // This definitely is not needed for GEMM, since nothing depends on it
            // dist::tma::store_async_wait();

            // Closed inside the same predicate that issues the store, and so
            // only by warp 0. Marking it from all four epilogue warps would
            // give the other three a span covering nothing but the emit
            // itself, labelled as if they had issued a store. The cost is that
            // warps 1-3 end their iteration at EPI_SMEM_WRITTEN with no
            // matching end -- deliberate, so "epi: issue store" legitimately
            // counts a quarter of what "epi: reg->smem" does.
            TIMING_MARK(EV_EPI_TMA_ISSUED);
        }
        TIMING_NEXT_SEQ();

        epilogue_stage_id = (epilogue_stage_id + 1) % fused_globals::EPILOGUE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << 3);
        }
    };

    // producer
    if (warp_id == 4) {
        if (elect_warp_leader()) {
            int input_stage_id = 0;
            for (int tile_id = cluster_idx; tile_id < num_tiles_total;
                 tile_id += num_comp_clusters) {
                auto [tile_row_id, tile_col_id] =
                    calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
                // A is split by rows across the cluster, B by columns.
                load(tile_row_id * config::NUM_CLUSTERS + cta_rank, tile_col_id, input_stage_id);
            }
        }
    } else if (warp_id == 5) {
        if (cta_rank == 0 && elect_warp_leader()) {
            int input_stage_id = 0;
            int epilogue_stage_id = 0;
            for (int iter = cluster_idx; iter < num_tiles_total; iter += num_comp_clusters) {
                consume(input_stage_id, epilogue_stage_id);
            }
        }
    } else if (warp_id >= 0 && warp_id < 4) {
        int epilogue_stage_id = 0;
        for (int tile_id = cluster_idx; tile_id < num_tiles_total; tile_id += num_comp_clusters) {
            auto [tile_row_id, tile_col_id] =
                calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
            // This CTA holds the 128 output rows fed by its own half of A.
            epilogue(tile_row_id * config::NUM_CLUSTERS + cta_rank, tile_col_id, epilogue_stage_id);

            // wait for the entire warpgroup, so that everything will be in HBM
            // TODO: I dont think this will be very different from using an atomic counter, since
            // these warpgroups are not going to get in the way of instruction issue
            // warpgroup::sync(2);

            // // // currently, we assign in a round-robin fashion?

            // const int device_to_signal = tile_id % config::NUM_DEVICES;
            // if (warpgroup::laneid() == 0) {
            //     dist::signal(G.comp_comm_barrier, {tile_row_id, tile_col_id}, device_to_signal,
            //     1);
            // }
        }
    }

    // No flush: the ring is zero-initialized by the host and %globaltimer never
    // reads back as 0, so the host recovers each CTA's event count as the index
    // of the first zero timestamp in its slot. Nothing has to be published.
}

#undef TIMING_MARK
#undef TIMING_MARK_LEADER
#undef TIMING_NEXT_SEQ

// ============================================================================
// Pipelined intra-node all-reduce tile helper
// ============================================================================
//
// Ported from gemm_ar.cu's gemm_ar_pipelined_ar_tile (see the comment block
// there). The naive version issued one multimem.ld_reduce and one multimem.st
// per element, both carrying a "memory" ASM clobber, so every element cost two
// serialized NVSwitch round-trips (~600 ns each) with exactly one 4-byte
// request in flight per thread. That caps the AR at a few tens of GB/s
// regardless of how much NVLink bandwidth is available.
//
// Fix: issue AR_UNROLL independent ld_reduce into separate registers before
// any store, using the no-clobber variants, so the warp scheduler can keep
// AR_UNROLL NVSwitch round-trips in flight at once.
//
// Safety: the caller's __syncthreads() after the per-tile barrier wait is the
// acquire fence that makes every device's writes to C_dist visible, so the
// individual loads do not need their own "memory" clobber. The ops are still
// `asm volatile`, so they are neither reordered against each other nor
// eliminated.
constexpr int AR_UNROLL = 8;

__device__ __forceinline__ void pipelined_ar_tile(const fused_globals& G,
                                                  int row_base,
                                                  int col_base) {
    // bf16_2 units — one 4-byte multimem access each.
    constexpr int UNITS_PER_ROW = fused_globals::COL_BLOCK / 2;            // 128
    constexpr int TOTAL_UNITS = fused_globals::ROW_BLOCK * UNITS_PER_ROW;  // 16384
    constexpr int NT = config::NUM_THREADS;
    constexpr int BATCH = AR_UNROLL * NT;

    for (int base = threadIdx.x; base < TOTAL_UNITS; base += BATCH) {
        comm::bf16_2* ld_ptrs[AR_UNROLL];
        comm::bf16_2* st_ptrs[AR_UNROLL];
        uint32_t tmps[AR_UNROLL];

        // Consecutive threads take consecutive bf16_2 units, so each warp's
        // requests coalesce into contiguous 128B chunks.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * NT;
            if (j < TOTAL_UNITS) {
                const int r = row_base + j / UNITS_PER_ROW;
                const int c = col_base + (j % UNITS_PER_ROW) * 2;
                ld_ptrs[u] = reinterpret_cast<comm::bf16_2*>(G.C_dist.mc_ptr_at({r, c}));
                st_ptrs[u] = reinterpret_cast<comm::bf16_2*>(G.C_final.mc_ptr_at({r, c}));
            }
        }

        // All loads before any store — this is the whole point of the helper.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            if (base + u * NT < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(tmps[u],
                                                                                 ld_ptrs[u]);
            }
        }

#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            if (base + u * NT < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(st_ptrs[u], tmps[u]);
            }
        }
    }
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    // TODO: figure out how the per-device split should look like?
    const int num_tiles_total = G.M * G.N / (fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK);
    const int comm_block_idx = blockIdx.x - config::NUM_COMP_SM;

    const int tile_id_stride = config::NUM_DEVICES * config::NUM_COMM_SM;
    for (int tile_id = G.dev_idx + comm_block_idx * config::NUM_DEVICES; tile_id < num_tiles_total;
         tile_id += tile_id_stride) {
        // wait for local device signal
        auto [tile_row_idx, tile_col_idx] = calculate_tile_idx<SUPERGROUP_WIDTH>(
            G.M / fused_globals::ROW_BLOCK, G.N / fused_globals::COL_BLOCK, tile_id);

        // we can use a relaxed wait here, since every operation after this is multimem, which does
        // not go through the L1 cache + signal from before is a release add operation
        if (threadIdx.x == 0) {
            dist::wait(
                G.comp_comm_barrier, {tile_row_idx, tile_col_idx}, G.dev_idx, config::NUM_DEVICES);
        }
        __syncthreads();

        // mc_ptr_at indexes elements, not tiles, so the tile coords have to be
        // scaled up before the intra-tile offset is added. Getting this wrong
        // makes an odd tile_col_idx produce an odd element column, i.e. a
        // 2-byte-aligned pointer for a 4-byte bf16x2 multimem access.
        const int row_base = tile_row_idx * fused_globals::ROW_BLOCK;
        const int col_base = tile_col_idx * fused_globals::COL_BLOCK;

        // All NUM_THREADS participate — the old loop used only the first 128,
        // leaving a third of the CTA's in-flight capacity on the table.
        pipelined_ar_tile(G, row_base, col_base);
    }
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    fused_comp_sm<SUPERGROUP_WIDTH, DO_PROFILE>(G);
    // if (blockIdx.x < config::NUM_COMP_SM) {
    // } else {
    //     fused_intranode_sm(G);
    // }
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__global__ __cluster_dims__(config::NUM_CLUSTERS) __launch_bounds__(
    config::NUM_THREADS) void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel<SUPERGROUP_WIDTH, DO_PROFILE>(G);
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int smem_size =
        ((G.ROW_BLOCK * G.RED_BLOCK + G.COL_BLOCK / config::NUM_CLUSTERS * G.RED_BLOCK) *
         sizeof(comm::bf16) * fused_globals::PIPELINE_STAGES) +
        ((G.ROW_BLOCK * G.COL_BLOCK) * sizeof(comm::bf16)) +
        1024;  // NOTE: must add 1024 so this can be aligned by TK
    const int num_threads = config::NUM_THREADS;
    const int grid = config::NUM_BLOCKS;  // set aside 20 SMs for comm

    // smem_size is built from compile-time constants, so this only has to be
    // set once — doing it per launch puts a host API call inside the caller's
    // timing window.
    auto this_kernel = gemm_ar_fused_kernel_stub<SUPERGROUP_WIDTH, DO_PROFILE>;
    static const bool smem_configured = [&] {
        MKERNEL_CUDACHECK(cudaFuncSetAttribute(
            this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        return true;
    }();
    (void)smem_configured;

    this_kernel<<<grid, num_threads, smem_size, stream>>>(G);
}

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"