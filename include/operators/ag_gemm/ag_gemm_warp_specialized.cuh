#pragma once

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "dist/parallel_buffer.cuh"
#include "dist/tma.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_mma_tcgen05_bf16.cuh"

namespace ag_gemm_warp_specialized {

// CTAs per cluster, i.e. the tcgen05 MMA CTA group. 2 splits COL_BLOCK across
// the pair so each CTA stages half the B tile; 1 gives every CTA its own MMA.
static constexpr int DEFAULT_NUM_CTA = 2;
static constexpr int DEFAULT_NUM_CONSUMER_WARPS = 1;

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
struct fused_globals;

// Number of tile columns visited before the snake pattern steps to the next
// supergroup; wider supergroups trade B-tile reuse for A-tile reuse in L2.
static constexpr int DEFAULT_SUPERGROUP_WIDTH = 5;

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int SUPERGROUP_WIDTH = DEFAULT_SUPERGROUP_WIDTH,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS,
          bool USE_MULTICAST_MEMCPY = false>
void launch_ag_gemm_warp_specialized(
    const fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>& G);

// for M < 512, this should be 128
template <int _ROW_BLOCK, int _COL_BLOCK, int _NUM_CTA, int _NUM_CONSUMER_WARPS>
struct fused_globals {
    // config items
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;

    // not sure if I want to use a warp specialized or sm specialized strategy yet
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int CONSUMER_WARPS = _NUM_CONSUMER_WARPS;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = 1;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;
    // CTAs per cluster. 2-CTA MMA is preferred for shapes that divide cleanly;
    // NUM_CLUSTERS is the cluster dimension the kernel launches with.
    static constexpr int NUM_CTA = _NUM_CTA;
    static constexpr int NUM_CLUSTERS = NUM_CTA;
    static_assert(NUM_CTA == 1 || NUM_CTA == 2, "tcgen05 only has 1- and 2-CTA MMA groups");
    static_assert(NUM_BLOCKS % NUM_CTA == 0, "NUM_BLOCKS must be a whole number of clusters");
    static_assert(_COL_BLOCK % NUM_CTA == 0, "COL_BLOCK must split evenly across the cluster");
    static constexpr int NUM_THREADS = (CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS) * 32;

    // this is pipelining along the reduction dimension
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = []() {
        if constexpr (_NUM_CONSUMER_WARPS == 2) {
            return 4;
        } else if constexpr (_NUM_CTA == 1 || _COL_BLOCK == 256) {
            return 6;
        } else {
            return 7;
        }
    }();
    // this is pipelining among different MMAs
    static constexpr int TMEM_PIPELINE_STAGES =
        kittens::MAX_TENSOR_COLS / _COL_BLOCK / CONSUMER_WARPS;
    static constexpr int NUM_TMEM_SLOTS = TMEM_PIPELINE_STAGES * CONSUMER_WARPS;
    // this is the number of epilogue stages that can be in flight at any time
    static constexpr int EPILOGUE_PIPELINE_STAGES = _COL_BLOCK == 128 ? 3 : 2;
    // this is the number of partitions for the epilogue tile in SMEM
    static constexpr int C_TILE_DIVISOR = 4;

    static constexpr int ROW_BLOCK = _ROW_BLOCK;
    static constexpr int COL_BLOCK = _COL_BLOCK;
    static constexpr int RED_BLOCK = 64;

    using A_tile = kittens::st_bf<ROW_BLOCK, RED_BLOCK>;
    // B is stored [N, K] (not [K, N]) so the reduction dimension is contiguous
    // in HBM -- the tile shape here mirrors that: rows are the N-chunk, cols
    // are the K-chunk. The MMA call reads it back with transpose::T, and the
    // TMA load coordinate is (n_tile, k_tile) to match.
    using B_tile = kittens::st_bf<COL_BLOCK / NUM_CLUSTERS, RED_BLOCK>;

    using C_tt_tile = kittens::tt<float, ROW_BLOCK, COL_BLOCK>;
    // for smem staging -- keep at least
    using C_tile = kittens::st_bf<ROW_BLOCK, COL_BLOCK / C_TILE_DIVISOR>;

    static constexpr int MAX_DYNAMIC_SHARED_MEMORY = 227 * 1024;
    static constexpr int DYNAMIC_SHARED_MEMORY =
        (sizeof(A_tile) * CONSUMER_WARPS + sizeof(B_tile)) * PRODUCER_CONSUMER_PIPELINE_STAGES +
        sizeof(C_tile) * EPILOGUE_PIPELINE_STAGES + 1024;
    // Deliberately not a static_assert: the tuner instantiates fused_globals
    // for every candidate so it can ask which ones fit. The hard check lives
    // in launch_ag_gemm_warp_specialized, so nothing oversized can actually launch.
    static constexpr bool SMEM_FITS = DYNAMIC_SHARED_MEMORY <= MAX_DYNAMIC_SHARED_MEMORY;

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using A_distributed_tensor = dist::distributed_tensor<A_local_tensor, NUM_DEVICES, true>;

    // we declare a separate A tensor here that is indexed as ((NUM_DEVICES, local_m), K), so that
    // TMA loads that go out of bounds will naturally zero themselves out
    using A_replicated_tensor = dist::local_tensor<comm::bf16, 1, NUM_DEVICES, -1, -1, A_tile>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<comm::bf16, 1, NUM_DEVICES, -1, -1, C_tile>;

    A_distributed_tensor A;
    A_replicated_tensor A_local_buf;
    B_local_tensor B;
    C_local_tensor C;

    // Copy-engine completion is published into local HBM.
    uint32_t* A_copy_ready;
    static constexpr uint32_t A_copy_epoch = 1;

    // Used only by USE_MULTICAST_MEMCPY. The first pointer is this rank's
    // ordinary mapping and the second is the multicast VA for the same int32
    // counter.
    int* multicast_barrier;
    int* multicast_barrier_mc;

    int dev_idx;
    int M;
    int N;
    static constexpr int K = 7168;

    struct pipeline_inputs {
        A_tile A[_NUM_CONSUMER_WARPS];
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    /*
     * bit 0: TMA producer -- starts with 1 (PRODUCER WARP)
     * bit 1: TMA consumer -- starts with 0 (CONSUMER WARP)
     * bits [2, 2+CONSUMER_WARPS): TMEM producer, one per consumer warp --
     *   starts with 1 (CONSUMER WARP), since there is no prior epilogue
     *   readout for the first pipeline fill to wait on
     * bits [2+CONSUMER_WARPS, 2+2*CONSUMER_WARPS): TMEM consumer, one per
     *   consumer warp -- starts with 0 (EPILOGUE WARP)
     */
    static constexpr int TMA_PRODUCER_BIT = 0b1;
    static constexpr int TMA_CONSUMER_BITS = 0b000;
    static constexpr int TMEM_PRODUCER_BITS = ((1 << CONSUMER_WARPS) - 1) << 2;
    static constexpr int TMEM_CONSUMER_BITS = 0;
    static constexpr int PHASE_BITS_INIT =
        TMA_PRODUCER_BIT | TMA_CONSUMER_BITS | TMEM_PRODUCER_BITS | TMEM_CONSUMER_BITS;
};

template <int _ROW_BLOCK,
          int _COL_BLOCK,
          int _NUM_CTA = DEFAULT_NUM_CTA,
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
__host__ inline fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>
ag_gemm_warp_specialized_make_globals(dist::ParallelBuffer& A,
                                      const at::Tensor& A_local_buf,
                                      const at::Tensor& B,
                                      at::Tensor& C,
                                      int dev_idx,
                                      int M,
                                      int N) {
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK, _NUM_CTA, _NUM_CONSUMER_WARPS>;

    return {.A = ::dist::distributed_tensor_from_buffer<typename fg::A_distributed_tensor>(A),
            .A_local_buf =
                ::dist::local_tensor_from_tensor<typename fg::A_replicated_tensor>(A_local_buf),
            .B = ::dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
            .C = ::dist::local_tensor_from_tensor<typename fg::C_local_tensor>(C),
            .A_copy_ready = nullptr,
            .multicast_barrier = nullptr,
            .multicast_barrier_mc = nullptr,
            .dev_idx = dev_idx,
            .M = M,
            .N = N};
}

template <bool USE_MULTICAST_MEMCPY>
void entrypoint_impl(dist::ParallelBuffer& A,
                     const at::Tensor& A_local_buf,
                     const at::Tensor& B,
                     at::Tensor& C,
                     const int logical_global_m,
                     int* multicast_barrier = nullptr,
                     int* multicast_barrier_mc = nullptr) {
    const int dev_idx = A.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    // C is now [NUM_DEVICES, local_m, N];
    const int M = C.size(0) * C.size(1), N = B.size(0);

    TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                "A.local_world_size must match the compiled INTRA_NUM_DEVICES");

    auto launch = [&]<int ROW_BLOCK,
                      int COL_BLOCK,
                      int NUM_CTA,
                      int SUPERGROUP_WIDTH,
                      int NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>() {
        using fg = fused_globals<ROW_BLOCK, COL_BLOCK, NUM_CTA, NUM_CONSUMER_WARPS>;
        fg globals = ag_gemm_warp_specialized_make_globals<
            ROW_BLOCK, COL_BLOCK, NUM_CTA, NUM_CONSUMER_WARPS>(
            A, A_local_buf, B, C, dev_idx, M, N);
        globals.multicast_barrier = multicast_barrier;
        globals.multicast_barrier_mc = multicast_barrier_mc;
        launch_ag_gemm_warp_specialized<ROW_BLOCK,
                                        COL_BLOCK,
                                        NUM_CTA,
                                        SUPERGROUP_WIDTH,
                                        NUM_CONSUMER_WARPS,
                                        USE_MULTICAST_MEMCPY>(globals);
    };

    // TODO: this only works for TP == 8
    constexpr int MIN_LARGE_GEMM_N = 6288;

    // use size of N to check which projection is being done
    if (N >= MIN_LARGE_GEMM_N) {
        switch (logical_global_m) {
            case 2048: {
                launch.template operator()<128, 128, 2, 15>();
                break;
            }
            case 3072: {
                launch.template operator()<128, 256, 2, 15>();
                break;
            }
            case 3584: {
                launch.template operator()<128, 256, 2, 20>();
                break;
            }
            case 4096: {
                launch.template operator()<128, 256, 2, 5>();
                break;
            }
            case 8192: {
                launch.template operator()<128, 256, 2, 5, 2>();
                break;
            }
            case 16384: {
                launch.template operator()<128, 256, 2, 5, 2>();
                break;
            }
            case 32768: {
                launch.template operator()<128, 256, 2, 5, 2>();
                break;
            }
            default:
                TORCH_CHECK(false, "ag_gemm_warp_specialized: no tile config for M=", M, " N=", N);
        }
    } else {
        switch (logical_global_m) {
            case 2048: {
                launch.template operator()<128, 128, 2, 25>();
                break;
            }
            case 3072: {
                launch.template operator()<128, 128, 1, 20>();
                break;
            }
            case 3584: {
                launch.template operator()<128, 256, 2, 10>();
                break;
            }
            case 4096: {
                launch.template operator()<128, 256, 2, 10>();
                break;
            }
            case 8192: {
                launch.template operator()<128, 256, 2, 10>();
                break;
            }
            case 16384: {
                launch.template operator()<128, 256, 2, 15>();
                break;
            }
            case 32768: {
                launch.template operator()<128, 256, 2, 15>();
                break;
            }
            default:
                TORCH_CHECK(false, "ag_gemm_warp_specialized: no tile config for M=", M, " N=", N);
        }
    }
}

void entrypoint(dist::ParallelBuffer& A,
                const at::Tensor& A_local_buf,
                const at::Tensor& B,
                at::Tensor& C,
                const int logical_global_m) {
    entrypoint_impl<false>(A, A_local_buf, B, C, logical_global_m);
}

// Multicast-copy baseline. A is a multicast DistBuffer containing the full
// [global_M, K] gather destination, while A_local is this rank's
// [global_M / NUM_DEVICES, K] source shard. barrier is a zero-initialized,
// multicast int32 DistBuffer with at least one element.
void entrypoint_multicast(dist::ParallelBuffer& A,
                          const at::Tensor& A_local,
                          dist::ParallelBuffer& barrier,
                          const at::Tensor& B,
                          at::Tensor& C,
                          const int logical_global_m) {
    constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;
    constexpr int K = fused_globals<128, 128>::K;

    TORCH_CHECK(C.dim() == 3 && C.size(0) == NUM_DEVICES,
                "C must have shape [NUM_DEVICES, local_M, N]");
    TORCH_CHECK(B.dim() == 2 && B.size(1) == K,
                "B must have the pre-transposed shape [N, K]");
    TORCH_CHECK(C.size(2) == B.size(0), "C's N dimension must match B.size(0)");
    const int64_t M = C.size(0) * C.size(1);

    TORCH_CHECK(A.multicast_ && A.multicast_ptr_ != nullptr,
                "multicast mode requires A to be a multicast DistBuffer");
    TORCH_CHECK(A.local_rank_ == barrier.local_rank_ &&
                    A.local_world_size_ == barrier.local_world_size_,
                "A and barrier must use the same local rank and world size");
    TORCH_CHECK(A.dtype_ == at::kBFloat16 && A.data_.is_contiguous(),
                "multicast A must be contiguous bfloat16");
    TORCH_CHECK(A.data_.dim() == 2 && A.data_.size(0) == M && A.data_.size(1) == K,
                "multicast A must have shape [global_M, K]");
    TORCH_CHECK(M % NUM_DEVICES == 0, "global M must divide evenly across devices");
    TORCH_CHECK(A_local.scalar_type() == at::kBFloat16 && A_local.is_contiguous(),
                "A_local must be contiguous bfloat16");
    TORCH_CHECK(A_local.dim() == 2 && A_local.size(0) == M / NUM_DEVICES &&
                    A_local.size(1) == K,
                "A_local must have shape [global_M / NUM_DEVICES, K]");
    TORCH_CHECK(B.scalar_type() == at::kBFloat16 && B.is_contiguous(),
                "B must be contiguous bfloat16");
    TORCH_CHECK(C.scalar_type() == at::kBFloat16 && C.is_contiguous(),
                "C must be contiguous bfloat16");
    TORCH_CHECK(A_local.device() == A.data_.device() && B.device() == A.data_.device() &&
                    C.device() == A.data_.device(),
                "A, A_local, B, and C must be on the same CUDA device");
    TORCH_CHECK(barrier.multicast_ && barrier.multicast_ptr_ != nullptr,
                "multicast mode requires barrier to be a multicast DistBuffer");
    TORCH_CHECK(barrier.dtype_ == at::kInt && barrier.data_.is_contiguous(),
                "multicast barrier must be contiguous int32");
    TORCH_CHECK(barrier.data_.numel() >= 1,
                "multicast barrier needs at least one int32 element");
    TORCH_CHECK(barrier.data_.device() == A.data_.device(),
                "multicast barrier must be on the same CUDA device as A");

    entrypoint_impl<true>(A,
                          A_local,
                          B,
                          C,
                          logical_global_m,
                          barrier.data_.data_ptr<int>(),
                          static_cast<int*>(barrier.multicast_ptr_));
}
};  // namespace ag_gemm_warp_specialized
