#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
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

__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    // TODO: prefetch tensormap
    // if (elect_warp_leader()) {
    // dist::tma::prefetch_tensormap(const TensorMapT *desc)
    // }
    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();

    const int num_tiles_per_row = G.N / fused_globals::COL_BLOCK;
    const int row_tile_id =
        blockIdx.x / (num_tiles_per_row * config::NUM_CLUSTERS) * config::NUM_CLUSTERS +
        blockIdx.x % config::NUM_CLUSTERS;
    // NOTE: CTA tile was specified to be N = 128, rather than 256
    const int col_tile_id = (blockIdx.x / config::NUM_CLUSTERS) % num_tiles_per_row;

    // allocate smem and tmem
    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    fused_globals::A_tile& A_smem = smem_allocator.allocate<fused_globals::A_tile>();
    fused_globals::B_tile& B_smem = smem_allocator.allocate<fused_globals::B_tile>();
    fused_globals::C_tile& C_smem = smem_allocator.allocate<fused_globals::C_tile>();

    __shared__ semaphore tma_load;
    __shared__ semaphore mma_finish;
    __shared__ semaphore epilogue_ready;

    tensor_allocator<1, config::NUM_CLUSTERS> tm_alloc{};
    uint32_t phasebit = 0xFFFF0000;
    fused_globals::C_tt_tile tmem = tm_alloc.allocate<fused_globals::C_tt_tile>(0);

    if (warp_id == 0 && elect_warp_leader()) {
        init_semaphore(tma_load, 0, 2);
        init_semaphore(mma_finish, 0, 1);
        init_semaphore(epilogue_ready, 0, 1);
    }
    everyone::tma::cluster::sync();

    auto load = [&](int iter_k) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            wait(mma_finish, get_phasebit<1>(phasebit, 0));

            tma::cluster::expect_bytes(
                tma_load, sizeof(fused_globals::A_tile) + sizeof(fused_globals::B_tile), 0);
            tma::cluster::load_async(
                A_smem, G.A, {row_tile_id, i}, tma_load, (uint16_t)(1 << cta_rank), 0);
            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {i, col_tile_id * config::NUM_CLUSTERS + cta_rank},
                                     tma_load,
                                     (uint16_t)(1 << cta_rank),
                                     0);

            update_phasebit<1>(phasebit, 0);
        }
    };

    // each only handles 16
    auto consume = [&](int iter_k) {
        for (int i = 0; i < G.K / fused_globals::RED_BLOCK; i++) {
            wait(tma_load, get_phasebit<0>(phasebit, 0));
            if (i == 0) {
                mm2_AB(tmem, A_smem, B_smem, mma_finish);
            } else {
                mma2_AB(tmem, A_smem, B_smem, mma_finish);
            }
            update_phasebit<0>(phasebit, 0);
        }

        kittens::detail::tcgen05::commit<config::NUM_CLUSTERS>(epilogue_ready);
    };

    auto epilogue = [&]() {
        wait(epilogue_ready, get_phasebit<0>(phasebit, 1));

        rt_bf<fused_globals::ROW_BLOCK / 4, fused_globals::COL_BLOCK> c_reg;
        warpgroup::load_async(c_reg, tmem);
        tensor_load_wait();

        warpgroup::sync(1);
        warpgroup::store(C_smem, c_reg);
        warpgroup::sync(1);

        if (warpgroup::laneid() == 0) {
            dist::tma::store_async(G.C_dist[G.dev_idx], C_smem, {row_tile_id, col_tile_id});
            dist::tma::store_async_wait();
        }
    };

    // producer
    if (warp_id == 4) {
        if (elect_warp_leader()) {
            load(0);
        }
    } else if (warp_id == 5) {
        if (cta_rank == 0 && elect_warp_leader()) {
            consume(0);
        }
    } else if (warp_id >= 0 && warp_id < 4) {
        epilogue();
    }

    everyone::tma::cluster::arrive_aligned();
    if (threadIdx.x == 0) {
        // TODO: difference between signal() and this
        comm::atomic_u32::release_add_sys(&G.comp_comm_barrier[G.dev_idx][{0, 0}], 1);
    }
}

__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    const int m_idx = (blockIdx.x - 128);
    const int n_idx = threadIdx.x * 16;

    if (threadIdx.x == 0) {
        // we can have a lot more waiters than signallers, 1 in each warp
        int val;
        do {
            comm::multimem<int>::ld_reduce<comm::reduce_op::MIN, comm::memory_model::STRONG>(
                val, reinterpret_cast<const int*>(G.comp_comm_barrier.mc_ptr_at({0, 0})));
        } while (val < (int)128);
    }

    __syncthreads();

    // TODO: not sure if this is necessary to prevent reordering?
    __threadfence_system();
    // NOTE: not sure what the difference between weak and strong is here
    // but I dont think strong in this case would be too big of a difference, since
    // __syncthreads already prevents some sort of reordering

    // only use 128 threads, each of which will load 16 items into GMEM
    // 128 * 16 = 2048
    if (threadIdx.x < 128) {
        for (int i = 0; i < 8; i++) {
            comm::bf16_2 res;
            comm::multimem<comm::bf16_2>::ld_reduce<comm::reduce_op::ADD, comm::memory_model::WEAK>(
                res, reinterpret_cast<comm::bf16_2*>(G.C_dist.mc_ptr_at({m_idx, n_idx + i * 2})));

            // I think there is an optimization that the stores can be split among the devices?
            reinterpret_cast<comm::bf16_2*>(&G.C_final[G.dev_idx][{m_idx, n_idx + i * 2}])[0] = res;
        }
    }
}

__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < 128) {
        fused_comp_sm(G);
    } else {
        fused_intranode_sm(G);
    }
}

__global__ __cluster_dims__(config::NUM_CLUSTERS) void gemm_ar_fused_kernel_stub(
    const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    // NOTE: must add 1024 so this can be aligned by TK
    const int smem_size =
        (G.ROW_BLOCK * G.RED_BLOCK + G.COL_BLOCK / config::NUM_CLUSTERS * G.RED_BLOCK +
         G.ROW_BLOCK * G.COL_BLOCK) *
            sizeof(comm::bf16) +
        1024;
    const int num_threads = config::NUM_THREADS;
    const int grid = G.M * G.N / (G.ROW_BLOCK * G.COL_BLOCK) + 2048;  // the last 2048 are for comm

    MKERNEL_CUDACHECK(cudaFuncSetAttribute(
        gemm_ar_fused_kernel_stub, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    gemm_ar_fused_kernel_stub<<<grid, num_threads, smem_size, stream>>>(G);
}

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
