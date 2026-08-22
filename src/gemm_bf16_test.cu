/******************************************************************************
 * Experimental copy of the Blackwell GEMM compute path. See
 * include/operators/gemm_ar/gemm_bf16_test.cuh for what this is isolating.
 *
 * Structurally identical to fused_comp_sm in src/gemm_ar_blackwell.cu, minus
 * the profiler and the all-reduce path, with exactly two differences:
 *   - B arrives N x K and the MMA is mm2_ABt / mma2_ABt
 *   - C is one plain global tensor instead of C_dist[dev_idx]
 *****************************************************************************/

#include <cuda.h>
#include <cuda_bf16.h>

#include <cstdint>
#include <tuple>

#include "comm/comm.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor_tensor.cuh"
#include "common/types.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/gemm_ar/gemm_bf16_test.cuh"

using namespace kittens;

namespace gemm_bf16_test {

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
__device__ __forceinline__ void test_comp_sm(const test_globals& G) {
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.prefetch_tma<test_globals::A_tile>();
        G.B.prefetch_tma<test_globals::B_tile>();
        G.C.prefetch_tma<test_globals::C_tile>();
    }

    const int num_row_tiles = G.M / (test_globals::ROW_BLOCK * config::NUM_CLUSTERS);
    const int num_col_tiles = G.N / test_globals::COL_BLOCK;
    const int num_tiles_total = num_row_tiles * num_col_tiles;
    const int cluster_idx = blockIdx.x / config::NUM_CLUSTERS;
    const int num_comp_clusters = config::NUM_COMP_SM / config::NUM_CLUSTERS;

    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    test_globals::pipeline_inputs(&inputs_smem)[test_globals::PIPELINE_STAGES] =
        smem_allocator.allocate<test_globals::pipeline_inputs, test_globals::PIPELINE_STAGES>();
    test_globals::C_tile(&C_smem)[config::CONSUMER_WARPS][test_globals::NUM_C_TILES] =
        smem_allocator
            .allocate<test_globals::C_tile, config::CONSUMER_WARPS, test_globals::NUM_C_TILES>();

    __shared__ semaphore tma_load[test_globals::PIPELINE_STAGES];
    __shared__ semaphore mma_finish[test_globals::PIPELINE_STAGES];
    __shared__ semaphore epilogue_ready[config::CONSUMER_WARPS];
    __shared__ semaphore epilogue_finished[config::CONSUMER_WARPS];

    tensor_allocator<1, config::NUM_CLUSTERS> tm_alloc{};

    // bits 4-5: epilogue_ready per consumer    - start at 0
    // bits 2-3: epilogue_finished per consumer - start at 1
    // bit 1:    tma_load                       - starts at 0
    // bit 0:    mma_finish                     - starts at 1
    uint32_t phasebits = 0b001101;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < test_globals::PIPELINE_STAGES; i++) {
            init_semaphore(tma_load[i], 0, config::NUM_CLUSTERS);
            init_semaphore(mma_finish[i], 0, config::CONSUMER_WARPS);
        }
#pragma unroll
        for (int c = 0; c < config::CONSUMER_WARPS; c++) {
            init_semaphore(epilogue_ready[c], 0, 1);
            init_semaphore(epilogue_finished[c], WARPGROUP_WARPS * config::NUM_CLUSTERS, 0);
        }
    }

    everyone::tma::cluster::arrive_aligned();

    // tile_row_idx is this CTA's FIRST A row tile in A_tile units; consumer c
    // takes the tile c further along. tile_col_idx is the cluster's C column
    // tile in COL_BLOCK units.
    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / test_globals::RED_BLOCK; i++) {
            test_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(mma_finish[input_stage_id], (phasebits & 0b1));

            tma::cluster::expect_bytes(tma_load[input_stage_id],
                                       sizeof(test_globals::A_tile) * config::CONSUMER_WARPS +
                                           sizeof(test_globals::B_tile),
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

            // B is N x K here, so the tile coordinate is {n_tile, k_tile} --
            // the transpose of the K x N form. n counts B_tile::rows
            // (= COL_BLOCK / NUM_CLUSTERS) blocks, k counts RED_BLOCK blocks.
            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {tile_col_idx * config::NUM_CLUSTERS + cta_rank, i},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            input_stage_id = (input_stage_id + 1) % test_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 1;
            }
        }
    };

    auto consume = [&](int& input_stage_id, test_globals::C_tt_tile* tmem, const int consumer_id) {
        wait(epilogue_finished[consumer_id], (phasebits >> (2 + consumer_id)) & 0b1);

        {
            test_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            test_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            // ABt: N == B::rows * ncta, K == B::cols
            mm2_ABt(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % test_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int i = 1; i < G.K / test_globals::RED_BLOCK; i++) {
            test_globals::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            test_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            mma2_ABt(tmem[0], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % test_globals::PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready[consumer_id]);
        phasebits ^= (1 << (2 + consumer_id));
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        test_globals::C_tt_tile* tmem,
                        const int warpgroup_id,
                        bool is_last_tile) {
        wait(epilogue_ready[warpgroup_id], (phasebits >> (4 + warpgroup_id)) & 0b1);

        constexpr int C_CHUNK_COLS = test_globals::COL_BLOCK / test_globals::EPILOGUE_STAGES;
        rt_bf<test_globals::ROW_BLOCK / (4 * config::CONSUMER_WARPS), C_CHUNK_COLS>
            c_reg[test_globals::EPILOGUE_STAGES];
        // + 1 because __syncthreads makes use of id = 0
        const int epilogue_barrier = warpgroup_id + 1;

#pragma unroll
        for (int i = 0; i < test_globals::EPILOGUE_STAGES; i++) {
            warpgroup::load_async(
                c_reg[i],
                tmem[0]
                    .template subtile<
                        tt<float, test_globals::ROW_BLOCK / config::CONSUMER_WARPS, C_CHUNK_COLS>>(
                        i * C_CHUNK_COLS));
        }
        tensor_load_wait();
        warpgroup::sync(epilogue_barrier);

        if (elect_warp_leader()) {
            if (is_last_tile && warp_id == 0) {
                pdl::arrive();
            }
            tma::cluster::arrive(epilogue_finished[warpgroup_id], 0);
        }

#pragma unroll
        for (int i = 0; i < test_globals::EPILOGUE_STAGES; i++) {
            tma::store_async_read_wait<test_globals::NUM_C_TILES - 1>();
            warpgroup::sync(epilogue_barrier);
            warpgroup::store(C_smem[warpgroup_id][i % test_globals::NUM_C_TILES], c_reg[i]);
            warpgroup::sync(epilogue_barrier);

            if (warpgroup::laneid() == 0) {
                // Plain global tensor -- no per-device descriptor lookup.
                tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    G.C,
                    C_smem[warpgroup_id][i % test_globals::NUM_C_TILES],
                    {tile_row_idx, tile_col_idx * test_globals::EPILOGUE_STAGES + i});
            }
        }

        phasebits ^= (1 << (4 + warpgroup_id));
    };

    const int cta_row_tile_base = cta_rank * config::CONSUMER_WARPS;

    if (warpgroup_id >= config::EPILOGUE_WARPGROUPS) {
        warpgroup::decrease_registers<config::MAINLOOP_REGISTERS>();

        if (warp_id == config::PRODUCER_WARP_ID) {
            if (elect_warp_leader()) {
                pdl::wait();
                everyone::tma::cluster::wait();
                int input_stage_id = 0;
                for (int tile_id = cluster_idx; tile_id < num_tiles_total;
                     tile_id += num_comp_clusters) {
                    auto [tile_row_id, tile_col_id] =
                        calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
                    load(tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                             cta_row_tile_base,
                         tile_col_id,
                         input_stage_id);
                }
            }
        } else if (warp_id >= config::FIRST_CONSUMER_WARP_ID &&
                   warp_id < config::FIRST_CONSUMER_WARP_ID + config::CONSUMER_WARPS) {
            if (cta_rank == 0 && elect_warp_leader()) {
                everyone::tma::cluster::wait();
                const int consumer_id = warp_id - config::FIRST_CONSUMER_WARP_ID;

                test_globals::C_tt_tile tmem[1];
                tmem[0] =
                    tm_alloc.allocate<test_globals::C_tt_tile>(consumer_id *
                                                               test_globals::COL_BLOCK);

                int input_stage_id = 0;
                for (int iter = cluster_idx; iter < num_tiles_total; iter += num_comp_clusters) {
                    consume(input_stage_id, tmem, consumer_id);
                }
            }
        }
    } else {
        warpgroup::increase_registers<config::EPILOGUE_REGISTERS>();
        everyone::tma::cluster::wait_aligned();

        test_globals::C_tt_tile tmem[1];
        tmem[0] = tm_alloc.allocate<test_globals::C_tt_tile>(warpgroup_id * test_globals::COL_BLOCK);

        for (int tile_id = cluster_idx; tile_id < num_tiles_total; tile_id += num_comp_clusters) {
            auto [tile_row_id, tile_col_id] =
                calculate_tile_idx<SUPERGROUP_WIDTH>(num_row_tiles, num_col_tiles, tile_id);
            epilogue(tile_row_id * config::NUM_CLUSTERS * config::CONSUMER_WARPS +
                         cta_row_tile_base + warpgroup_id,
                     tile_col_id,
                     tmem,
                     warpgroup_id,
                     tile_id + num_comp_clusters >= num_tiles_total);
        }
    }
}

template <int SUPERGROUP_WIDTH>
__global__ __cluster_dims__(config::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(config::NUM_THREADS, 1) void gemm_bf16_test_kernel(
        const __grid_constant__ test_globals G) {
    test_comp_sm<SUPERGROUP_WIDTH>(G);
}

template <int SUPERGROUP_WIDTH>
void launch_gemm_bf16_test(const test_globals& G, cudaStream_t stream) {
    constexpr int smem_size = test_globals::DYNAMIC_SHARED_MEMORY;
    auto this_kernel = gemm_bf16_test_kernel<SUPERGROUP_WIDTH>;

    static const bool smem_configured = [&] {
        MKERNEL_CUDACHECK(cudaFuncSetAttribute(
            this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        return true;
    }();
    (void)smem_configured;

    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[0].val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = config::NUM_BLOCKS;
    launch_config.blockDim = config::NUM_THREADS;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;
    launch_config.attrs = attrs;
    launch_config.numAttrs = 1;

    MKERNEL_CUDACHECK(cudaLaunchKernelEx(&launch_config, this_kernel, G));
}

template void launch_gemm_bf16_test<4>(const test_globals&, cudaStream_t);
template void launch_gemm_bf16_test<8>(const test_globals&, cudaStream_t);

}  // namespace gemm_bf16_test
