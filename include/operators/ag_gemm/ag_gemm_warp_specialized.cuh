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
          int _NUM_CONSUMER_WARPS = DEFAULT_NUM_CONSUMER_WARPS>
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

    // Copy-engine completion is published into local HBM. There is one
    // monotonically increasing epoch per source device.
    uint32_t* A_copy_ready;
    uint32_t A_copy_epoch;

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
            .A_copy_epoch = 0,
            .dev_idx = dev_idx,
            .M = M,
            .N = N};
}

// Runtime-to-compile-time dispatch for the shapes under active autotuning:
// `sw`/`cw` are runtime ints (already defaulted from supergroup_width /
// consumer_warps), and this expands to a chain of `if`s picking the matching
// compile-time instantiation. Every (SW, CW) pair used across the tunable
// shapes below compiles to the same handful of unique kernel instantiations
// (the template args don't depend on M), so listing it in every tunable case
// doesn't multiply compile time by the number of shapes.
#define AG_GEMM_TUNE_CASE(SW, CW)                                                  \
    if (sw == (SW) && cw == (CW)) {                                                \
        using fg = fused_globals<128, 256, 2, (CW)>;                               \
        fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2, (CW)>(     \
            A, A_local_buf, B, C, dev_idx, M, N);                                  \
        launch_ag_gemm_warp_specialized<128, 256, 2, (SW), (CW)>(globals);         \
    } else

void entrypoint(dist::ParallelBuffer& A,
                const at::Tensor& A_local_buf,
                const at::Tensor& B,
                at::Tensor& C,
                const int logical_global_m,  // used to determine what the actual shape being
                                             // operated on is, since M might be padded up
                // Autotuning knobs for the shapes under active tuning (KDA/MLA
                // at M in {8192, 16384, 32768}); -1 (the default) uses that
                // shape's currently-tuned default. Every other shape ignores
                // both arguments and always uses its own tuned config.
                const int supergroup_width = -1,
                const int consumer_warps = -1) {
    const int dev_idx = A.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    // C is now [NUM_DEVICES, local_m, N];
    const int M = C.size(0) * C.size(1), N = B.size(0);
    constexpr int K = fused_globals<128, 128>::K;

    TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                "A.local_world_size must match the compiled INTRA_NUM_DEVICES");

    // TODO: this only works for TP == 8
    constexpr int KDA_N = 6288;

    // use size of N to check which projection is being done
    if (N == KDA_N) {
        switch (logical_global_m) {
            case 2048: {
                using fg = fused_globals<128, 128, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 128, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 128, 2, 15>(globals);
                break;
            }
            case 3072: {
                using fg = fused_globals<128, 256, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 256, 2, 15>(globals);
                break;
            }
            case 3584: {
                using fg = fused_globals<128, 256, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 256, 2, 20>(globals);
                break;
            }
            case 4096: {
                using fg = fused_globals<128, 256, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 256, 2, 5>(globals);
                break;
            }
            case 8192: {
                const int sw = supergroup_width < 0 ? 20 : supergroup_width;
                const int cw = consumer_warps < 0 ? 1 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            case 16384: {
                const int sw = supergroup_width < 0 ? 5 : supergroup_width;
                const int cw = consumer_warps < 0 ? 2 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            case 32768: {
                const int sw = supergroup_width < 0 ? 10 : supergroup_width;
                const int cw = consumer_warps < 0 ? 2 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            default:
                TORCH_CHECK(false, "ag_gemm_warp_specialized: no tile config for M=", M, " N=", N);
        }
    } else {
        switch (logical_global_m) {
            case 2048: {
                using fg = fused_globals<128, 128, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 128, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 128, 2, 25>(globals);
                break;
            }
            case 3072: {
                using fg = fused_globals<128, 128, 1>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 128, 1>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 128, 1, 20>(globals);
                break;
            }
            case 3584: {
                using fg = fused_globals<128, 256, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 256, 2, 10>(globals);
                break;
            }
            case 4096: {
                using fg = fused_globals<128, 256, 2>;
                fg globals = ag_gemm_warp_specialized_make_globals<128, 256, 2>(
                    A, A_local_buf, B, C, dev_idx, M, N);
                launch_ag_gemm_warp_specialized<128, 256, 2, 10>(globals);
                break;
            }
            case 8192: {
                const int sw = supergroup_width < 0 ? 10 : supergroup_width;
                const int cw = consumer_warps < 0 ? 1 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            case 16384: {
                const int sw = supergroup_width < 0 ? 15 : supergroup_width;
                const int cw = consumer_warps < 0 ? 1 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            case 32768: {
                const int sw = supergroup_width < 0 ? 20 : supergroup_width;
                const int cw = consumer_warps < 0 ? 2 : consumer_warps;
                AG_GEMM_TUNE_CASE(5, 1) AG_GEMM_TUNE_CASE(5, 2)
                AG_GEMM_TUNE_CASE(10, 1) AG_GEMM_TUNE_CASE(10, 2)
                AG_GEMM_TUNE_CASE(15, 1) AG_GEMM_TUNE_CASE(15, 2)
                AG_GEMM_TUNE_CASE(20, 1) AG_GEMM_TUNE_CASE(20, 2)
                AG_GEMM_TUNE_CASE(25, 1) AG_GEMM_TUNE_CASE(25, 2) {
                    TORCH_CHECK(false,
                                "ag_gemm_warp_specialized: no tuned config for "
                                "supergroup_width=",
                                sw,
                                " consumer_warps=",
                                cw);
                }
                break;
            }
            default:
                TORCH_CHECK(false, "ag_gemm_warp_specialized: no tile config for M=", M, " N=", N);
        }
    }
}
#undef AG_GEMM_TUNE_CASE
};  // namespace ag_gemm_warp_specialized
