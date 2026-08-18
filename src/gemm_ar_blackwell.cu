#include "comm/comm.cuh"
#include "common/cuda_checks.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "operators/gemm_ar/gemm_ar_blackwell.cuh"

#include "dist/tma.cuh"

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace kittens;

constexpr int M = 2048;
constexpr int N = 2048;
constexpr int K = 128;

namespace gemm_ar_intranode_blackwell {

__device__ __forceinline__ void fused_comp_sm(const fused_globals& G) {
    // allocate smem and tmem
    extern __shared__ int __shm[];
    tma_swizzle_allocator allocator((int*)&__shm[0]);

    __shared__ comm::bf16 A_smem = allocator.allocate<G.A_tile>();
    __shared__ comm::bf16 B_smem = allocator.allocate<G.B_tile>();

    __shared__ semaphore tma_load;
    __shared__ semaphore mma_finish;

    if (kittens::elect_warp_leader()) {
        kittens::init_semaphore(&tma_load, 1, 0);
        kittens::init_semaphore(&mma_finish, 1, 0);
    }

    // TODO: change when we are a cluster
    __syncthreads();

    if (threadIdx.x == 0) {
        // TODO: difference between signal() and this
        comm::atomic_u32::release_add_sys(&G.comp_comm_barrier[G.dev_idx][{0, 0}], 1);
    }
}

__device__ __forceinline__ void fused_intranode_sm(const fused_globals& G) {
    const int m_idx = (blockIdx.x - 4096);
    const int n_idx = threadIdx.x * 2;

    if (threadIdx.x == 0) {
        // we can have a lot more waiters than signallers, 1 in each warp
        int val;
        do {
            comm::multimem<int>::ld_reduce<comm::reduce_op::MIN, comm::memory_model::STRONG>(
                val, reinterpret_cast<const int*>(G.comp_comm_barrier.mc_ptr_at({0, 0})));
        } while (val < (int)4096);
    }

    __syncthreads();
    // NOTE: not sure what the difference between weak and strong is here
    // but I dont think strong in this case would be too big of a difference, since __syncthreads
    // already prevents some sort of reordering
    comm::bf16_2 res;
    comm::multimem<comm::bf16_2>::ld_reduce<comm::reduce_op::ADD, comm::memory_model::WEAK>(
        res, reinterpret_cast<comm::bf16_2*>(G.C_dist.mc_ptr_at({m_idx, n_idx})));

    // I think there is an optimization that the stores can be split among the devices?
    reinterpret_cast<comm::bf16_2*>(&G.C_final[G.dev_idx][{m_idx, n_idx}])[0] = res;
}

__device__ __forceinline__ void fused_kernel(const fused_globals& G) {
    if (blockIdx.x < 4096) {
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

    const int smem_size = (G.ROW_BLOCK * G.RED_BLOCK + G.COL_BLOCK / G.CLUSTER_SIZE * G.RED_BLOCK) *
        sizeof(comm::bf16);
    const int num_threads = config::NUM_THREADS;
    const int grid = G.M * G.N / (G.ROW_BLOCK * G.COL_BLOCK) + 20;  // the last 20 are for comm

    gemm_ar_fused_kernel_stub<<<grid, num_threads, smem_size, stream>>>(G);
}

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
