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
#include "operators/gemm_ar/profiler.h"

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

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();
    Profiler prof;

    if constexpr (DO_PROFILE) {
        if (elect_warp_leader()) {
            // One slot per WARP, not per block: the warps here play different
            // roles (see the config::*_WARP_ID layout) and each records its own
            // timeline. Keying on blockIdx.x alone would have them all
            // interleave writes into one row and clobber each other's counters.
            // The host buffer must have config::NUM_WARPS rows per block.
            prof.init(G.num_entries, G.data_ptr, blockIdx.x * config::NUM_WARPS + warp_id);
            prof.start(ProfilerTag::Setup);
        }
    }

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
    fused_globals::C_tile(&C_smem)[config::CONSUMER_WARPS][fused_globals::NUM_C_TILES] =
        smem_allocator
            .allocate<fused_globals::C_tile, config::CONSUMER_WARPS, fused_globals::NUM_C_TILES>();

    __shared__ semaphore tma_load[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore mma_finish[fused_globals::PIPELINE_STAGES];
    __shared__ semaphore epilogue_ready[config::CONSUMER_WARPS];
    __shared__ semaphore epilogue_finished[config::CONSUMER_WARPS];

    tensor_allocator<1, config::NUM_CLUSTERS> tm_alloc{};

    // combined phasebits, one bit per barrier array. A bit is toggled once per
    // full ring traversal of its array, i.e. when the stage index wraps to 0 --
    // the epilogue rings hold a single barrier per consumer, so their bits flip
    // on every iteration.
    // bit 4-5: epilogue_ready (consumer 0 & consumer 1)   - starts at 0
    // bit 2-3: epilogue_finished (consumer 0 & consumer 1) - starts at 1
    // these two stay the same since the mma and tma are
    // shared
    // bit 1: tma_load          - starts at 0
    // bit 0: mma_finish        - starts at 1
    uint32_t phasebits = 0b001101;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fused_globals::PIPELINE_STAGES; i++) {
            // TMA load needs to wait for both clusters to finish
            init_semaphore(tma_load[i], 0, config::NUM_CLUSTERS);
            // tma in each CTA needs to wait for both consumers to finish
            init_semaphore(mma_finish[i], 0, config::CONSUMER_WARPS);
        }

#pragma unroll
        for (int c = 0; c < config::CONSUMER_WARPS; c++) {
            init_semaphore(epilogue_ready[c], 0, 1);
            // broadcasted back to the mma thread to signal that tmem is ready
            init_semaphore(epilogue_finished[c], WARPGROUP_WARPS * config::NUM_CLUSTERS, 0);
        }
    }

    everyone::tma::cluster::sync();
    if constexpr (DO_PROFILE) {
        if (elect_warp_leader()) {
            prof.stop();
        }
    }

    // tile_row_idx is this CTA's FIRST A row tile, in A_tile units
    // (ROW_BLOCK / CONSUMER_WARPS rows, already rank adjusted) -- consumer c
    // takes the tile c further along, which is the same row tile its epilogue
    // warpgroup stores back. tile_col_idx is the cluster's C column tile
    // (COL_BLOCK units).
    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::WaitMMA);
            }
            wait(mma_finish[input_stage_id], (phasebits & 0b1));
            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::IssueTMA);
            }

            tma::cluster::expect_bytes(tma_load[input_stage_id],
                                       sizeof(fused_globals::A_tile) * config::CONSUMER_WARPS +
                                           sizeof(fused_globals::B_tile),
                                       0);

#pragma unroll
            for (int c = 0; c < config::CONSUMER_WARPS; c++) {
                tma::cluster::load_async(inputs_smem[input_stage_id].A[c],
                                         G.A,
                                         {tile_row_idx + c, i},
                                         tma_load[input_stage_id],
                                         (uint16_t)(1 << cta_rank),
                                         0);
            }

            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {i,
                                      (tile_col_idx * config::NUM_CLUSTERS +
                                       cta_rank)},  // this has to be here becuase the epilogue
                                                    // warp loads the tiles in col units of 256
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 1;
            }
        }
    };

    // each only handles 16
    auto consume = [&](int& input_stage_id, fused_globals::C_tt_tile* tmem, const int consumer_id) {
        if constexpr (DO_PROFILE) {
            prof.start(ProfilerTag::WaitEpilogue);
        }
        wait(epilogue_finished[consumer_id], (phasebits >> (2 + consumer_id)) & 0b1);
        if constexpr (DO_PROFILE) {
            prof.stop();
        }

        {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::WaitTMA);
            }
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::IssueMMA);
            }
            mm2_AB(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);
            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;

            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int i = 1; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::WaitTMA);
            }
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            if constexpr (DO_PROFILE) {
                prof.start(ProfilerTag::IssueMMA);
            }
            mma2_AB(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);
            if constexpr (DO_PROFILE) {
                prof.stop();
            }

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready[consumer_id]);

        // This consumer owns exactly one accumulator, so epilogue_finished
        // completes once per output tile and its phase flips every iteration.
        phasebits ^= (1 << (2 + consumer_id));
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        fused_globals::C_tt_tile* tmem,
                        const int warpgroup_id) {
        if constexpr (DO_PROFILE) {
            if (elect_warp_leader()) {
                prof.start(ProfilerTag::WaitMainloop);
            }
        }
        wait(epilogue_ready[warpgroup_id], (phasebits >> (4 + warpgroup_id)) & 0b1);
        if constexpr (DO_PROFILE) {
            if (elect_warp_leader()) {
                prof.stop();
            }
        }
        tensor_after_thread_sync();

        if constexpr (DO_PROFILE) {
            if (elect_warp_leader()) {
                prof.start(ProfilerTag::Epilogue);
            }
        }
        constexpr int C_CHUNK_COLS = fused_globals::COL_BLOCK / fused_globals::EPILOGUE_STAGES;
        rt_bf<fused_globals::ROW_BLOCK / (4 * config::CONSUMER_WARPS), C_CHUNK_COLS>
            c_reg[fused_globals::EPILOGUE_STAGES];

#pragma unroll
        for (int i = 0; i < fused_globals::EPILOGUE_STAGES; i++) {
            warpgroup::load_async(
                c_reg[i],
                tmem[0]
                    .template subtile<
                        tt<float, fused_globals::ROW_BLOCK / config::CONSUMER_WARPS, C_CHUNK_COLS>>(
                        i * C_CHUNK_COLS));
        }
        tensor_load_wait();

        // signal tmem empty
        if (elect_warp_leader()) {
            // TODO: move this into dist namespace
            tma::cluster::arrive(epilogue_finished[warpgroup_id], 0);
        }

#pragma unroll
        for (int i = 0; i < fused_globals::EPILOGUE_STAGES; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fused_globals::NUM_C_TILES - 1>();
            // + 1 because __syncthreads makes use of id = 0
            warpgroup::sync(warpgroup_id + 1);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[warpgroup_id][i % fused_globals::NUM_C_TILES], c_reg[i]);
            warpgroup::sync(warpgroup_id + 1);

            if (warpgroup::laneid() == 0) {
                // C_tile is only COL_BLOCK / EPILOGUE_STAGES wide, so the TMA
                // column coordinate counts chunks, not COL_BLOCK tiles.
                dist::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    G.C_dist[G.dev_idx],
                    C_smem[warpgroup_id][i % fused_globals::NUM_C_TILES],
                    {tile_row_idx, tile_col_idx * fused_globals::EPILOGUE_STAGES + i});
            }
        }

        if constexpr (DO_PROFILE) {
            if (elect_warp_leader()) {
                prof.stop();
            }
        }

        // Same single-accumulator ring as the consumer side: epilogue_ready
        // completes once per output tile, so this flips every iteration.
        phasebits ^= (1 << (4 + warpgroup_id));
    };

    // Row tiles are A_tile/C_tile sized (ROW_BLOCK / CONSUMER_WARPS rows), so a
    // cluster block spans NUM_CLUSTERS * CONSUMER_WARPS of them: this CTA owns
    // the CONSUMER_WARPS tiles starting here, one per consumer.
    const int cta_row_tile_base = cta_rank * config::CONSUMER_WARPS;

    // producer + consumers share the tail warpgroup(s)
    if (warpgroup_id >= config::EPILOGUE_WARPGROUPS) {
        // warpgroup::decrease_registers<152>();

        if (warp_id == config::PRODUCER_WARP_ID) {
            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int tile_id = cluster_idx; tile_id < num_tiles_total;
                     tile_id += num_comp_clusters) {
                    auto [tile_row_id, tile_col_id] =
                        calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
                    // This specifies the 256 * 256 chunk that has to be loaded
                    load(tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                             cta_row_tile_base,
                         tile_col_id,
                         input_stage_id);
                }
            }
        } else if (warp_id >= config::FIRST_CONSUMER_WARP_ID &&
                   warp_id < config::FIRST_CONSUMER_WARP_ID + config::CONSUMER_WARPS) {
            if (cta_rank == 0 && elect_warp_leader()) {
                // consumer_id pairs this warp with epilogue warpgroup
                // consumer_id: same A tile, same accumulator, same semaphores.
                const int consumer_id = warp_id - config::FIRST_CONSUMER_WARP_ID;

                // give each warp its own view of tmem
                fused_globals::C_tt_tile tmem[1];
                tmem[0] = tm_alloc.allocate<fused_globals::C_tt_tile>(consumer_id *
                                                                      fused_globals::COL_BLOCK);

                int input_stage_id = 0;
                for (int iter = cluster_idx; iter < num_tiles_total; iter += num_comp_clusters) {
                    consume(input_stage_id, tmem, consumer_id);
                }
            }
        }
    } else {
        // give each warpgroup its own view of tmem
        fused_globals::C_tt_tile tmem[1];
        tmem[0] =
            tm_alloc.allocate<fused_globals::C_tt_tile>(warpgroup_id * fused_globals::COL_BLOCK);

        for (int tile_id = cluster_idx; tile_id < num_tiles_total; tile_id += num_comp_clusters) {
            // this returns an index in the 512 * 256 tile
            auto [tile_row_id, tile_col_id] =
                calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
            // This specifies the 128 * 256 tile that should be epilogu-ed --
            // the same row tile the producer loaded into A[warpgroup_id].
            epilogue(tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                         cta_row_tile_base + warpgroup_id,
                     tile_col_id,
                     tmem,
                     warpgroup_id);
        }
    }

    // Publish the event count. Nothing reads a warp's slot until this lands, so
    // without it every row reports zero events no matter what was recorded.
    if constexpr (DO_PROFILE) {
        if (elect_warp_leader()) {
            prof.flush();
        }
    }
}

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
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
__global__ __cluster_dims__(config::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(config::NUM_THREADS, 1) void gemm_ar_fused_kernel_stub(
        const __grid_constant__ fused_globals G) {
    fused_kernel<SUPERGROUP_WIDTH, DO_PROFILE>(G);
}

template <int SUPERGROUP_WIDTH, bool DO_PROFILE>
void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    constexpr int smem_size = fused_globals::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = config::NUM_THREADS;
    constexpr int grid = config::NUM_BLOCKS;  // set aside 20 SMs for comm

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