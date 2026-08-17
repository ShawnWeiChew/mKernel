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
    const int m_idx = blockIdx.x / 2;
    const int n_idx = threadIdx.x % N + (blockIdx.x % 2) * 1024;

    float val = __bfloat162float(G.A[{m_idx, 0}]) * __bfloat162float(G.B[{0, n_idx}]);
    for (int k_iter = 1; k_iter < K; k_iter++) {
        val += __bfloat162float(G.A[{m_idx, k_iter}]) * __bfloat162float(G.B[{k_iter, n_idx}]);
    }

    G.C_dist[G.dev_idx][{m_idx, n_idx}] = __float2bfloat16_rn(val);

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

__global__ void gemm_ar_fused_kernel_stub(const __grid_constant__ fused_globals G) {
    fused_kernel(G);
}

void launch_fused_gemm_ar_blackwell(const fused_globals& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    gemm_ar_fused_kernel_stub<<<M * N / 1024 + M * N / 2048, 1024, 0, stream>>>(G);
}

};  // namespace gemm_ar_intranode_blackwell

#include "operators/gemm_ar/gemm_ar_blackwell_session.cuh"
