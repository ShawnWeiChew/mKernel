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

template <int SUPERGROUP_WIDTH>
__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();

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

    // tile_row_idx is this CTA's A row tile (ROW_BLOCK units, already rank
    // adjusted); tile_col_idx is the cluster's C column tile (COL_BLOCK units).
    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(mma_finish[input_stage_id], (phasebits & 0b1));

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

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 1;
            }
        }
    };

    // each only handles 16
    auto consume = [&](int& input_stage_id, int& epilogue_stage_id) {
        wait(epilogue_finished[epilogue_stage_id], (phasebits >> 2) & 0b1);

        {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            mm2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;

            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int i = 1; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            mma2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

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
        wait(epilogue_ready[epilogue_stage_id], (phasebits >> 3) & 0b1);
        tensor_after_thread_sync();

        rt_bf<fused_globals::ROW_BLOCK / 4, fused_globals::COL_BLOCK> c_reg;
        warpgroup::load_async(c_reg, tmem[epilogue_stage_id]);
        tensor_load_wait();

        // signal tmem empty
        if (elect_warp_leader()) {
            // TODO: move this into dist namespace
            tma::cluster::arrive(epilogue_finished[epilogue_stage_id], 0);
        }
        warpgroup::sync(1);
        // this already does the swizzle inside it
        warpgroup::store(C_smem, c_reg);
        warpgroup::sync(1);

        if (warpgroup::laneid() == 0) {
            dist::tma::store_async(G.C_dist[G.dev_idx], C_smem, {tile_row_idx, tile_col_idx});
        }

        epilogue_stage_id = (epilogue_stage_id + 1) % fused_globals::EPILOGUE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << 3);
        }

        if (warpgroup::laneid() == 0) {
            dist::tma::store_async_wait();
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
            warpgroup::sync(2);

            // The barrier is keyed on CTA tiles (ROW_BLOCK rows), not cluster
            // tiles, so signal the row this CTA actually stored to above — the
            // cluster row would leave every odd row unsignalled and double-count
            // every even one.
            //
            // comm_tile_id is the comm side's linear tile id for that same tile.
            // Both sides have to agree on it, because the comm loop claims tiles
            // by `id % NUM_DEVICES == dev_idx`; fused_intranode_sm inverts this
            // to recover the coordinate. Note it does not depend on dev_idx, so
            // every device picks the same owner for a given tile.
            const int c_row = tile_row_id * config::NUM_CLUSTERS + cta_rank;
            const int comm_tile_id = tile_id * config::NUM_CLUSTERS + cta_rank;
            if (warpgroup::laneid() == 0) {
                dist::signal(G.comp_comm_barrier,
                             {c_row, tile_col_id},
                             comm_tile_id % config::NUM_DEVICES,
                             1);
            }
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

template <int SUPERGROUP_WIDTH>
__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    // TODO: figure out how the per-device split should look like?
    const int num_tiles_per_row = G.N / fused_globals::COL_BLOCK;
    const int num_tiles_total = G.M * G.N / (fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK);
    const int comm_block_idx = blockIdx.x - config::NUM_COMP_SM;
    // Cluster rows, matching fused_comp_sm's walk — see the decode below.
    const int num_cluster_row_tiles = G.M / (fused_globals::ROW_BLOCK * config::NUM_CLUSTERS);

    const int tile_id_stride = config::NUM_DEVICES * config::NUM_COMM_SM;
    for (int tile_id = G.dev_idx + comm_block_idx * config::NUM_DEVICES; tile_id < num_tiles_total;
         tile_id += tile_id_stride) {
        // Decode through the *cluster* tile walk rather than running a second,
        // independent snake over CTA rows. fused_comp_sm computes one cluster
        // tile per iteration and emits two CTA tiles from it (one per cta_rank),
        // so tile_id == cluster_tile_id * NUM_CLUSTERS + cta_rank is exactly the
        // id the epilogue signals with, and this is its inverse. Keeping the two
        // sides on one formula is the point: the previous version walked its own
        // snake over G.M / ROW_BLOCK rows, which put comp and comm in different
        // tile-id spaces and left half the barrier slots waiting on a signal
        // that never came.
        const int cluster_tile_id = tile_id / config::NUM_CLUSTERS;
        const int sub_row = tile_id % config::NUM_CLUSTERS;
        auto [cluster_row_idx, tile_col_idx] = calculate_tile_idx<SUPERGROUP_WIDTH>(
            num_cluster_row_tiles, num_tiles_per_row, cluster_tile_id);
        const int tile_row_idx = cluster_row_idx * config::NUM_CLUSTERS + sub_row;

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

template <int SUPERGROUP_WIDTH>
__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < config::NUM_COMP_SM) {
        fused_comp_sm<SUPERGROUP_WIDTH>(G);
    } else {
        fused_intranode_sm<SUPERGROUP_WIDTH>(G);
    }
}

template <int SUPERGROUP_WIDTH>
__global__ __cluster_dims__(config::NUM_CLUSTERS) __launch_bounds__(
    config::NUM_THREADS) void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel<SUPERGROUP_WIDTH>(G);
}

template <int SUPERGROUP_WIDTH>
void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    const int smem_size =
        ((G.ROW_BLOCK * G.RED_BLOCK + G.COL_BLOCK / config::NUM_CLUSTERS * G.RED_BLOCK) *
         sizeof(comm::bf16) * fused_globals::PIPELINE_STAGES) +
        ((G.ROW_BLOCK * G.COL_BLOCK) * sizeof(comm::bf16)) +
        1024;  // NOTE: must add 1024 so this can be aligned by TK
    const int num_threads = config::NUM_THREADS;
    const int grid = config::NUM_BLOCKS;  // set aside 20 SMs for comm

    auto this_kernel = gemm_ar_fused_kernel_stub<SUPERGROUP_WIDTH>;
    // smem_size is built from compile-time constants, so this only has to be
    // set once — doing it per launch puts a host API call inside the caller's
    // timing window.
    static const bool smem_configured = [&] {
        MKERNEL_CUDACHECK(cudaFuncSetAttribute(
            this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        return true;
    }();
    (void)smem_configured;

    this_kernel<<<grid, num_threads, smem_size, stream>>>(G);
}

// ============================================================================
// NVSwitch / NVLink bandwidth probe (mnvl_bw_test)
// ============================================================================
//
// Standalone sweep harness: no GEMM, no barrier, no compute SMs — just the
// multimem all-reduce loop from pipelined_ar_tile, parameterised over the tile
// shape, the comm-SM count and the unroll depth. The question it answers is
// "given an M x N output block, what is the smallest tile a comm CTA can be
// handed before the NVSwitch, and not the tile walk, is the bottleneck?".
//
// Sweep axes (driven by bench/mnvl_bw_sweep.py):
//   problem shape     4096, 8192, 16384, 32768
//   NUM_COMM_SM       12, 16, 20, 24, 28, 32
//   SUBTILE_M         128, 256   (the whole CLUSTER drives one output tile)
//   SUBTILE_N         16, 32, 64, 128, 256
//   AR_UNROLL         4, 8, 16, 32
//   SUPERGROUP_WIDTH  4, 8       (snake walk, same as the fused kernel)
//
// NUM_THREADS is pinned at 384 — the config the fused kernel is expected to
// adopt. Only the axes that *must* be compile time are template parameters:
// NUM_COMM_SM rides in on gridDim.x, which keeps the instantiation count at
// 2*2*5*4 = 80 kernels instead of 480.
//
// Work split: device d owns the tile ids congruent to d mod NUM_DEVICES, so
// the node as a whole covers every tile exactly once. That makes the output
// checkable — see bench/mnvl_bw_sweep.py's --check mode — which matters,
// because an index bug that silently skips tiles looks like a *faster* kernel.
namespace mnvl_bw_test {

static constexpr int NUM_THREADS = 384;

namespace ar_detail {

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N, int NT>
__device__ __forceinline__ void ar_unroll_no_cache(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base) {
    constexpr int UNITS_PER_ROW = SUBTILE_N / 2;
    constexpr int TOTAL_UNITS = SUBTILE_M * UNITS_PER_ROW;
    constexpr int BATCH = AR_UNROLL * NT;

    for (int base = threadIdx.x; base < TOTAL_UNITS; base += BATCH) {
        uint32_t tmps[AR_UNROLL];

        // Consecutive threads take consecutive bf16_2 units, so each warp's
        // requests coalesce into contiguous 128B chunks.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * NT;
            if (j < TOTAL_UNITS) {
                const int r = row_base + j / UNITS_PER_ROW;
                const int c = col_base + (j % UNITS_PER_ROW) * 2;
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(
                    tmps[u], reinterpret_cast<comm::bf16_2*>(C_dist.mc_ptr_at({r, c})));
            }
        }

#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * NT;
            if (j < TOTAL_UNITS) {
                const int r = row_base + j / UNITS_PER_ROW;
                const int c = col_base + (j % UNITS_PER_ROW) * 2;
                comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(
                    reinterpret_cast<comm::bf16_2*>(C_final.mc_ptr_at({r, c})), tmps[u]);
            }
        }
    }
}

// AR_UNROLL != 32: cache both load and store addresses up front.
template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N, int NT>
__device__ __forceinline__ void ar_unroll_cached_st(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base) {
    constexpr int UNITS_PER_ROW = SUBTILE_N / 2;
    constexpr int TOTAL_UNITS = SUBTILE_M * UNITS_PER_ROW;
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
                ld_ptrs[u] = reinterpret_cast<comm::bf16_2*>(C_dist.mc_ptr_at({r, c}));
                st_ptrs[u] = reinterpret_cast<comm::bf16_2*>(C_final.mc_ptr_at({r, c}));
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

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N, int NT = NUM_THREADS>
__device__ __forceinline__ void experimental_ar_unroll(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base) {
    // bf16_2 units — one 4-byte multimem access each.
    static_assert(SUBTILE_N % 2 == 0, "SUBTILE_N must be even (bf16_2 units)");
    constexpr int UNITS_PER_ROW = SUBTILE_N / 2;
    constexpr int TOTAL_UNITS = SUBTILE_M * UNITS_PER_ROW;
    constexpr int BATCH = AR_UNROLL * NT;

    for (int base = threadIdx.x; base < TOTAL_UNITS; base += BATCH) {
        comm::bf16_2* ld_ptrs[AR_UNROLL];
        uint32_t tmps[AR_UNROLL];

        // Consecutive threads take consecutive bf16_2 units, so each warp's
        // requests coalesce into contiguous 128B chunks.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * NT;
            if (j < TOTAL_UNITS) {
                const int r = row_base + j / UNITS_PER_ROW;
                const int c = col_base + (j % UNITS_PER_ROW) * 2;
                ld_ptrs[u] = reinterpret_cast<comm::bf16_2*>(C_dist.mc_ptr_at({r, c}));
            }
        }

        // All loads before any store — this is the whole point of the helper.
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            const int j = base + u * NT;
            if (j < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::ld_reduce_add_weak_bits_no_clobber(tmps[u],
                                                                                 ld_ptrs[u]);
            }
        }

        const ptrdiff_t st_delta = C_final.mc_ptr - C_dist.mc_ptr;  // outside the loop
#pragma unroll
        for (int u = 0; u < AR_UNROLL; u++) {
            if (base + u * NT < TOTAL_UNITS) {
                comm::multimem<comm::bf16_2>::st_weak_bits_no_clobber(
                    reinterpret_cast<comm::bf16_2*>(reinterpret_cast<comm::bf16*>(ld_ptrs[u]) +
                                                    st_delta),
                    tmps[u]);
            }
        }
    }
}

}  // namespace ar_detail

template <int AR_UNROLL, int SUBTILE_M, int SUBTILE_N, int NT = NUM_THREADS>
__device__ __forceinline__ void experimental_ar_unroll(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int row_base,
    int col_base) {
    // bf16_2 units — one 4-byte multimem access each.
    static_assert(SUBTILE_N % 2 == 0, "SUBTILE_N must be even (bf16_2 units)");

    if constexpr (AR_UNROLL >= 128) {
        ar_detail::ar_unroll_no_cache<AR_UNROLL, SUBTILE_M, SUBTILE_N, NT>(
            C_dist, C_final, row_base, col_base);
    } else if (AR_UNROLL >= 32) {
        ar_detail::experimental_ar_unroll<AR_UNROLL, SUBTILE_M, SUBTILE_N, NT>(
            C_dist, C_final, row_base, col_base);
    } else {
        ar_detail::ar_unroll_cached_st<AR_UNROLL, SUBTILE_M, SUBTILE_N, NT>(
            C_dist, C_final, row_base, col_base);
    }
}

// __grid_constant__ is only legal on a __global__ function's parameters, so
// the descriptors are taken by const reference here and the kernel stub below
// is what actually owns the grid-constant copies.
template <int SUPERGROUP_WIDTH,
          int SUBTILE_M,
          int SUBTILE_N,
          int AR_UNROLL,
          int NUM_DEVICES = config::NUM_DEVICES>
__device__ __forceinline__ void fused_intranode_sm(
    const fused_globals::C_distributed_tensor& C_dist,
    const fused_globals::C_final_tensor& C_final,
    int M,
    int N,
    int dev_idx,
    int num_repeats) {
    const int num_comm_sm = gridDim.x;
    const int comm_block_idx = blockIdx.x;
    const int num_rows = M / SUBTILE_M;
    const int num_cols = N / SUBTILE_N;
    const int num_tiles_total = num_rows * num_cols;
    const int tile_id_stride = NUM_DEVICES * num_comm_sm;

    // Nothing in the loop mutates C_dist and multimem.st is a plain store, so
    // replaying the walk is idempotent — it just amortises launch overhead at
    // the small shapes, where a single pass is only a few microseconds.
    for (int rep = 0; rep < num_repeats; rep++) {
        for (int tile_id = dev_idx + comm_block_idx * NUM_DEVICES; tile_id < num_tiles_total;
             tile_id += tile_id_stride) {
            auto [tile_row_idx, tile_col_idx] =
                calculate_tile_idx<SUPERGROUP_WIDTH>(num_rows, num_cols, tile_id);

            const int row_base = tile_row_idx * SUBTILE_M;
            const int col_base = tile_col_idx * SUBTILE_N;

            experimental_ar_unroll<AR_UNROLL, SUBTILE_M, SUBTILE_N>(
                C_dist, C_final, row_base, col_base);
        }
    }
}

template <int SUPERGROUP_WIDTH, int SUBTILE_M, int SUBTILE_N, int AR_UNROLL>
__global__ __launch_bounds__(NUM_THREADS) void bw_test_kernel_stub(
    // NOTE: by value, not by reference — a __grid_constant__ parameter is the
    // link between the host-side descriptor and the kernel, so it has to be a
    // copy living in the kernel's parameter space.
    const __grid_constant__ fused_globals::C_distributed_tensor C_dist,
    const __grid_constant__ fused_globals::C_final_tensor C_final,
    int M,
    int N,
    int dev_idx,
    int num_repeats) {
    fused_intranode_sm<SUPERGROUP_WIDTH, SUBTILE_M, SUBTILE_N, AR_UNROLL>(
        C_dist, C_final, M, N, dev_idx, num_repeats);
}

template <int SUPERGROUP_WIDTH, int SUBTILE_M, int SUBTILE_N, int AR_UNROLL>
void launch_bw_test(const fused_globals::C_distributed_tensor& C_dist,
                    const fused_globals::C_final_tensor& C_final,
                    int M,
                    int N,
                    int dev_idx,
                    int num_comm_sm,
                    int num_repeats) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bw_test_kernel_stub<SUPERGROUP_WIDTH, SUBTILE_M, SUBTILE_N, AR_UNROLL>
        <<<num_comm_sm, NUM_THREADS, 0, stream>>>(C_dist, C_final, M, N, dev_idx, num_repeats);
}

// Runtime -> compile-time dispatch. Nested so the axes stay readable; adding a
// value to a sweep axis is a one-line change in the matching macro.
#define MNVL_LAUNCH_ARGS C_dist, C_final, M, N, dev_idx, num_comm_sm, num_repeats

#define MNVL_DISPATCH_UNROLL(SG, SM, SN)                                           \
    switch (ar_unroll) {                                                           \
        case 4:                                                                    \
            return launch_bw_test<SG, SM, SN, 4>(MNVL_LAUNCH_ARGS);                \
        case 8:                                                                    \
            return launch_bw_test<SG, SM, SN, 8>(MNVL_LAUNCH_ARGS);                \
        case 16:                                                                   \
            return launch_bw_test<SG, SM, SN, 16>(MNVL_LAUNCH_ARGS);               \
        case 32:                                                                   \
            return launch_bw_test<SG, SM, SN, 32>(MNVL_LAUNCH_ARGS);               \
        case 64:                                                                   \
            return launch_bw_test<SG, SM, SN, 64>(MNVL_LAUNCH_ARGS);               \
        case 128:                                                                  \
            return launch_bw_test<SG, SM, SN, 128>(MNVL_LAUNCH_ARGS);              \
        default:                                                                   \
            TORCH_CHECK(false, "mnvl_bw_test: unsupported ar_unroll=", ar_unroll); \
    }

#define MNVL_DISPATCH_SUBTILE_N(SG, SM)                                            \
    switch (subtile_n) {                                                           \
        case 16:                                                                   \
            MNVL_DISPATCH_UNROLL(SG, SM, 16)                                       \
        case 32:                                                                   \
            MNVL_DISPATCH_UNROLL(SG, SM, 32)                                       \
        case 64:                                                                   \
            MNVL_DISPATCH_UNROLL(SG, SM, 64)                                       \
        case 128:                                                                  \
            MNVL_DISPATCH_UNROLL(SG, SM, 128)                                      \
        case 256:                                                                  \
            MNVL_DISPATCH_UNROLL(SG, SM, 256)                                      \
        default:                                                                   \
            TORCH_CHECK(false, "mnvl_bw_test: unsupported subtile_n=", subtile_n); \
    }

#define MNVL_DISPATCH_SUBTILE_M(SG)                                                \
    switch (subtile_m) {                                                           \
        case 128:                                                                  \
            MNVL_DISPATCH_SUBTILE_N(SG, 128)                                       \
        case 256:                                                                  \
            MNVL_DISPATCH_SUBTILE_N(SG, 256)                                       \
        default:                                                                   \
            TORCH_CHECK(false, "mnvl_bw_test: unsupported subtile_m=", subtile_m); \
    }

// C and C_final are the two multicast buffers; M/N come from C's shape. The
// tile shape, comm-SM count, unroll depth and supergroup width are the sweep
// knobs, and num_repeats replays the tile walk inside one launch.
void bw_test_entrypoint(dist::ParallelBuffer& C_buf,
                        dist::ParallelBuffer& C_final_buf,
                        int subtile_m,
                        int subtile_n,
                        int num_comm_sm,
                        int ar_unroll,
                        int supergroup_width,
                        int num_repeats) {
    const int dev_idx = C_buf.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    TORCH_CHECK(C_buf.data_.dim() == 2 && C_final_buf.data_.dim() == 2,
                "mnvl_bw_test: expected 2D (M, N) buffers");
    TORCH_CHECK(C_buf.data_.sizes() == C_final_buf.data_.sizes(),
                "mnvl_bw_test: C and C_final must have the same shape");
    TORCH_CHECK(C_buf.multicast_ && C_final_buf.multicast_,
                "mnvl_bw_test: both buffers must be multicast-backed");

    const int M = (int)C_buf.data_.size(0);
    const int N = (int)C_buf.data_.size(1);

    TORCH_CHECK(num_comm_sm > 0, "mnvl_bw_test: num_comm_sm must be positive");
    TORCH_CHECK(num_repeats > 0, "mnvl_bw_test: num_repeats must be positive");
    // The walk has no tail handling: a leftover partial tile is simply never
    // all-reduced, which would quietly turn a correctness bug into a speedup.
    TORCH_CHECK(subtile_m > 0 && M % subtile_m == 0,
                "mnvl_bw_test: M=",
                M,
                " not divisible by subtile_m=",
                subtile_m);
    TORCH_CHECK(subtile_n > 0 && N % subtile_n == 0,
                "mnvl_bw_test: N=",
                N,
                " not divisible by subtile_n=",
                subtile_n);
    // calculate_tile_idx is only a bijection when the column count fills whole
    // supergroups; otherwise the last supergroup emits col_idx >= num_cols.
    TORCH_CHECK((N / subtile_n) % supergroup_width == 0,
                "mnvl_bw_test: num_cols=",
                N / subtile_n,
                " not divisible by supergroup_width=",
                supergroup_width);

    const auto C_dist =
        ::dist::distributed_tensor_from_buffer<fused_globals::C_distributed_tensor>(C_buf);
    const auto C_final =
        ::dist::distributed_tensor_from_buffer<fused_globals::C_final_tensor>(C_final_buf);

    switch (supergroup_width) {
        case 4:
            MNVL_DISPATCH_SUBTILE_M(4)
        case 8:
            MNVL_DISPATCH_SUBTILE_M(8)
        default:
            TORCH_CHECK(false, "mnvl_bw_test: unsupported supergroup_width=", supergroup_width);
    }
}

#undef MNVL_DISPATCH_SUBTILE_M
#undef MNVL_DISPATCH_SUBTILE_N
#undef MNVL_DISPATCH_UNROLL
#undef MNVL_LAUNCH_ARGS

};  // namespace mnvl_bw_test

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
