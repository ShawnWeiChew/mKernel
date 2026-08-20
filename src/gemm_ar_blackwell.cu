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

__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int tile_id,
                                                                   int num_tiles_per_row) {
    int tile_row_idx = tile_id / (num_tiles_per_row * config::NUM_CLUSTERS) * config::NUM_CLUSTERS +
        tile_id % config::NUM_CLUSTERS;

    int tile_col_idx = (tile_id / config::NUM_CLUSTERS) % num_tiles_per_row;

    return {tile_row_idx, tile_col_idx};
};

__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    // TODO: prefetch tensormap
    // if (elect_warp_leader()) {
    // dist::tma::prefetch_tensormap(const TensorMapT *desc)
    // }
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();

    const int num_tiles_per_row = G.N / fused_globals::COL_BLOCK;
    const int num_tiles_total = G.M * G.N / (fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK);
    const int block_idx = blockIdx.x;

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
    uint32_t inputs_phasebit = 0xFFFF0000;
    uint32_t epilogue_phasebit = 0xFFFF0000;
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

    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(mma_finish[input_stage_id], get_phasebit<1>(inputs_phasebit, input_stage_id));

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

            update_phasebit<1>(inputs_phasebit, input_stage_id);
            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
        }
    };

    // each only handles 16
    auto consume = [&](int& input_stage_id, int& epilogue_stage_id) {
        wait(epilogue_finished[epilogue_stage_id],
             get_phasebit<1>(epilogue_phasebit, epilogue_stage_id));

        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            fused_globals::A_tile& A_smem = inputs_smem[input_stage_id].A;
            fused_globals::B_tile& B_smem = inputs_smem[input_stage_id].B;

            wait(tma_load[input_stage_id], get_phasebit<0>(inputs_phasebit, input_stage_id));
            if (i == 0) {
                mm2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);
            } else {
                mma2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);
            }

            // TODO: can probably optimize this away later to be one phasebit per barrier
            update_phasebit<0>(inputs_phasebit, input_stage_id);
            input_stage_id = (input_stage_id + 1) % fused_globals::PIPELINE_STAGES;
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready[epilogue_stage_id]);
        update_phasebit<1>(epilogue_phasebit, epilogue_stage_id);
        epilogue_stage_id = (epilogue_stage_id + 1) % fused_globals::EPILOGUE_STAGES;
    };

    auto epilogue = [&](int tile_row_idx, int tile_col_idx, int& epilogue_stage_id) {
        wait(epilogue_ready[epilogue_stage_id],
             get_phasebit<0>(epilogue_phasebit, epilogue_stage_id));
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
        warpgroup::store(C_smem, c_reg);
        warpgroup::sync(1);

        if (warpgroup::laneid() == 0) {
            dist::tma::store_async(G.C_dist[G.dev_idx], C_smem, {tile_row_idx, tile_col_idx});
            dist::tma::store_async_wait();
        }

        update_phasebit<0>(epilogue_phasebit, epilogue_stage_id);
        epilogue_stage_id = (epilogue_stage_id + 1) % fused_globals::EPILOGUE_STAGES;
    };

    // producer
    if (warp_id == 4) {
        if (elect_warp_leader()) {
            int input_stage_id = 0;
            for (int tile_id = block_idx; tile_id < num_tiles_total;
                 tile_id += config::NUM_COMP_SM) {
                auto [tile_row_id, tile_col_id] = calculate_tile_idx(tile_id, num_tiles_per_row);
                load(tile_row_id, tile_col_id, input_stage_id);
            }
        }
    } else if (warp_id == 5) {
        if (cta_rank == 0 && elect_warp_leader()) {
            int input_stage_id = 0;
            int epilogue_stage_id = 0;
            for (int iter = block_idx; iter < num_tiles_total; iter += config::NUM_COMP_SM) {
                consume(input_stage_id, epilogue_stage_id);
            }
        }
    } else if (warp_id >= 0 && warp_id < 4) {
        int epilogue_stage_id = 0;
        for (int tile_id = block_idx; tile_id < num_tiles_total; tile_id += config::NUM_COMP_SM) {
            auto [tile_row_id, tile_col_id] = calculate_tile_idx(tile_id, num_tiles_per_row);
            epilogue(tile_row_id, tile_col_id, epilogue_stage_id);

            // wait for the entire warpgroup, so that everything will be in HBM
            // TODO: I dont think this will be very different from using an atomic counter, since
            // these warpgroups are not going to get in the way of instruction issue
            warpgroup::sync(2);

            // // currently, we assign in a round-robin fashion?
            const int device_to_signal = tile_id % config::NUM_DEVICES;
            if (warpgroup::laneid() == 0) {
                dist::signal(G.comp_comm_barrier, {tile_row_id, tile_col_id}, device_to_signal, 1);
            }
        }
    }
}

__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    // TODO: figure out how the per-device split should look like?
    const int num_tiles_per_row = G.N / fused_globals::COL_BLOCK;
    const int num_tiles_total = G.M * G.N / (fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK);
    const int comm_block_idx = blockIdx.x - config::NUM_COMP_SM;

    const int tile_id_stride = config::NUM_DEVICES * config::NUM_COMM_SM;
    for (int tile_id = G.dev_idx + comm_block_idx * config::NUM_DEVICES; tile_id < num_tiles_total;
         tile_id += tile_id_stride) {
        // wait for local device signal
        auto [tile_row_idx, tile_col_idx] = calculate_tile_idx(tile_id, num_tiles_per_row);

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

        // TODO: pipeline the multimem loads without clobbering
        // multimem load shared across threads
        if (threadIdx.x < 128) {
            for (int i = threadIdx.x; i < fused_globals::ROW_BLOCK * fused_globals::COL_BLOCK / 2;
             i += 128) {
                const int start_idx_within_tile = i * 2;
                const int subtile_row_idx = start_idx_within_tile / fused_globals::COL_BLOCK;
                const int subtile_col_idx = start_idx_within_tile % fused_globals::COL_BLOCK;

                comm::bf16_2* mc_ld = reinterpret_cast<comm::bf16_2*>(G.C_dist.mc_ptr_at(
                    {row_base + subtile_row_idx, col_base + subtile_col_idx}));

                comm::bf16_2 tmp;
                comm::multimem<comm::bf16_2>::ld_reduce<comm::reduce_op::ADD, comm::memory_model::WEAK>(
                    tmp, mc_ld);

                // multimem store
                comm::bf16_2* mc_st = reinterpret_cast<comm::bf16_2*>(G.C_final.mc_ptr_at(
                    {row_base + subtile_row_idx, col_base + subtile_col_idx}));
                comm::multimem<comm::bf16_2>::st(mc_st, tmp);
            }
        }
        
    }
}

__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < config::NUM_COMP_SM) {
        fused_comp_sm(G);
    } else {
        fused_intranode_sm(G);
    }
}

__global__ __cluster_dims__(config::NUM_CLUSTERS) __launch_bounds__(
    config::NUM_THREADS) void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

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
    static const bool smem_configured = [&] {
        MKERNEL_CUDACHECK(cudaFuncSetAttribute(
            gemm_ar_fused_kernel_stub, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
        return true;
    }();
    (void)smem_configured;

    gemm_ar_fused_kernel_stub<<<grid, num_threads, smem_size, stream>>>(G);
}

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
