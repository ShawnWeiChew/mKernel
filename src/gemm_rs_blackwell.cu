#include "memory/tk_ops_thread_mma_tcgen05.cuh"
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include <ATen/ATen.h>
#include <ATen/cuda/CUDABlas.h>
#include <c10/cuda/CUDAGuard.h>

#include "dist/dbuf_buffer_bridge.cuh"
#endif

#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "comm/atomic_u32.cuh"
#include "comm/comm.cuh"
#include "comm/multimem.cuh"
#include "common/cuda_checks.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/gemm_rs/gemm_rs_blackwell.cuh"

// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
#include "memory/tk_ops_group_util_util.cuh"
// clang-format on

// TODO: reset barrier value at every epoch

using namespace kittens;

namespace gemm_rs_blackwell {
namespace {
template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mm_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 0, NUM_CTA>(d, a, b, sem);
}

template <int NUM_CTA, typename D, typename A, typename B>
__device__ __forceinline__ void mma_ABt_ncta(D& d, const A& a, const B& b, semaphore& sem) {
    kittens::mma<transpose::N, transpose::T, D, A, B, 1, NUM_CTA>(d, a, b, sem);
}

// __device__ __forceinline__ void multimem_ld_reduce(bf16_2* dst, const bf16_2* src) {
//     asm volatile("multimem.ld_reduce.relaxed.sys.global.add.v4.bf16_2 {%0, %1, %2, %3}, [%4];",
//                  dst[0],
//                  dst[1],
//                  dst[2],
//                  dst[3],
//                  src);
// }

// __device__ __forceinline__ void store_float8(float* ptr, const float* v) {
//     asm volatile(
//         "st.global.v8.f32 [%0], "
//         "{%1, %2, %3, %4, %5, %6, %7, %8};"
//         :
//         : "l"(ptr),
//           "f"(v[0]),
//           "f"(v[1]),
//           "f"(v[2]),
//           "f"(v[3]),
//           "f"(v[4]),
//           "f"(v[5]),
//           "f"(v[6]),
//           "f"(v[7])
//         : "memory");
// }

template <int SUPERGROUP_WIDTH = 8>
__device__ __forceinline__ std::tuple<int, int> calculate_tile_idx(int num_rows,
                                                                   int num_cols,
                                                                   int tile_idx) {
    static_assert(SUPERGROUP_WIDTH > 0, "SUPERGROUP_SIZE must be greater than 0");
    const int supergroup_numel = num_rows * SUPERGROUP_WIDTH;
    const int supergroup_idx = tile_idx / supergroup_numel;
    const int supersection_cols = (num_cols / SUPERGROUP_WIDTH) * SUPERGROUP_WIDTH;
    const int supersection_numel = num_rows * supersection_cols;
    const int finalsection_cols = num_cols - supersection_cols;
    int row_idx, col_idx;
    if (tile_idx < supersection_numel) {
        row_idx = (tile_idx % supergroup_numel) / SUPERGROUP_WIDTH;
        col_idx = supergroup_idx * SUPERGROUP_WIDTH + tile_idx % SUPERGROUP_WIDTH;
    } else {
        const int remainder_task_id = tile_idx - supersection_numel;
        row_idx = remainder_task_id / finalsection_cols;
        col_idx = supersection_cols + remainder_task_id % finalsection_cols;
    }
    return {(supergroup_idx & 1) ? num_rows - row_idx - 1 : row_idx, col_idx};
}
}  // namespace

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
__device__ __forceinline__ void gemm_rs_blackwell(
    const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>& G) {
    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>;

    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.template prefetch_tma<typename fg::A_tile>();
        G.B.template prefetch_tma<typename fg::B_tile>();
        G.C.template prefetch_tma<typename fg::C_tile>();
    }

    const int cluster_idx = blockIdx.x / fg::NUM_CLUSTERS;
    constexpr int num_clusters = fg::NUM_BLOCKS / fg::NUM_CLUSTERS;

    const int num_tile_rows_total = G.M / (fg::A_tile::rows);
    const int num_tile_rows_per_cta = num_tile_rows_total / fg::CONSUMER_WARPS;
    const int num_tile_rows_per_cluster = num_tile_rows_per_cta / fg::NUM_CLUSTERS;

    const int num_tile_cols = G.N / G.COL_BLOCK;

    const int total_num_tiles = num_tile_rows_per_cluster * num_tile_cols;

    // RS calculations
    const int rows_per_device = G.M / fg::NUM_DEVICES;

    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    typename fg::pipeline_inputs(&inputs_smem)[fg::PRODUCER_CONSUMER_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::pipeline_inputs, fg::PRODUCER_CONSUMER_PIPELINE_STAGES>();
    typename fg::C_tile(&C_smem)[fg::EPILOGUE_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::C_tile, fg::EPILOGUE_PIPELINE_STAGES>();

    __shared__ semaphore mma_finish[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];
    __shared__ semaphore tma_load[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];

    __shared__ semaphore epilogue_ready[fg::NUM_TMEM_SLOTS];
    __shared__ semaphore epilogue_tmem_finished[fg::NUM_TMEM_SLOTS];

    __shared__ semaphore tmem_allocated;
    __shared__ semaphore tmem_finished;

    __shared__ uint32_t tmem_addr;
    tensor_allocator<1, fg::NUM_CLUSTERS> tm_alloc{};

    uint32_t phasebits = fg::PHASE_BITS_INIT;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fg::PRODUCER_CONSUMER_PIPELINE_STAGES; i++) {
            init_semaphore(mma_finish[i], fg::CONSUMER_WARPS);
            init_semaphore(tma_load[i], fg::NUM_CLUSTERS);
        }

#pragma unroll
        for (int i = 0; i < fg::NUM_TMEM_SLOTS; i++) {
            init_semaphore(epilogue_ready[i], 1);
            init_semaphore(epilogue_tmem_finished[i], WARPGROUP_WARPS * fg::NUM_CLUSTERS);
        }

        init_semaphore(tmem_allocated, 1);
        init_semaphore(tmem_finished, fg::NUM_CLUSTERS);
    }

    // flush to ensure the mbarriers are visible
    everyone::tma::cluster::arrive_aligned();

    auto load = [&](int tile_row_idx, int tile_col_idx, int& input_stage_id) {
        for (int iter_k = 0; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile(&A_smem)[fg::CONSUMER_WARPS] = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(mma_finish[input_stage_id], (phasebits & 0b1));

            tma::cluster::expect_bytes(tma_load[input_stage_id], ((phasebits >> 1) & 0b1));
            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {tile_col_idx * fg::NUM_CLUSTERS + cta_rank, iter_k},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);
#pragma unroll
            for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                tma::cluster::load_async(A_smem[c],
                                         G.A,
                                         {tile_row_idx + c, iter_k},
                                         tma_load[input_stage_id],
                                         (uint16_t)(1 << cta_rank),
                                         0);
            }

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 0b1;
            }
        }
    };

    auto consume = [&](typename fg::C_tt_tile* tmem,
                       int& input_stage_id,
                       int& epilogue_stage_id,
                       const int consumer_id) {
        const int consumer_semaphore_idx = consumer_id + 2;
        wait(epilogue_tmem_finished[consumer_id], (phasebits >> consumer_semaphore_idx) & 0b1);

        {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            mm_ABt_ncta<fg::NUM_CTA>(
                tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int iter_k = 1; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A[consumer_id];
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;
            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);
            mma_ABt_ncta<fg::NUM_CTA>(
                tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<fg::NUM_CLUSTERS>(epilogue_ready[consumer_id]);
        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;

        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << consumer_semaphore_idx);
        }
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        typename fg::C_tt_tile* tmem,
                        const int consumer_id,
                        int& epilogue_stage_id,
                        int& epilogue_smem_idx,
                        bool is_last_tile) {
        constexpr int C_CHUNK_COLS = fg::COL_BLOCK / fg::C_TILE_DIVISOR;
        rt_bf<fg::A_tile::rows / WARPGROUP_WARPS, C_CHUNK_COLS> c_reg[fg::C_TILE_DIVISOR];

        const int epilogue_semaphore_idx = consumer_id + 4;
        wait(epilogue_ready[consumer_id], (phasebits >> epilogue_semaphore_idx) & 0b1);

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            warpgroup::load_async(
                c_reg[i],
                tmem[epilogue_stage_id].template subtile<tt<float, fg::A_tile::rows, C_CHUNK_COLS>>(
                    i * C_CHUNK_COLS));
        }

        tensor_load_wait();
        if (elect_warp_leader()) {
            tma::cluster::arrive(epilogue_tmem_finished[consumer_id], 0);
        }

        if (is_last_tile && consumer_id == 0) {
            warpgroup::sync(1);
            pdl::arrive();
        }

#pragma unroll
        for (int i = 0; i < C_CHUNK_COLS; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fg::EPILOGUE_PIPELINE_STAGES - 1>();
            warpgroup::sync(1);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[epilogue_smem_idx], c_reg[i]);
            warpgroup::sync(1);

            // TODO: check if warpgroup_id == 0 and elect_sync() work for this too
            if (warpgroup::laneid() == 0) {
                dist::tma::store_async(G.C[G.dev_idx],
                                       C_smem[epilogue_smem_idx],
                                       {tile_row_idx, tile_col_idx * fg::C_TILE_DIVISOR + i});
            }

            epilogue_smem_idx = (epilogue_smem_idx + 1) % fg::EPILOGUE_PIPELINE_STAGES;
        }

        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (0b1 << epilogue_semaphore_idx);
        }
    };

    if (warpgroup_id >= fg::EPILOGUE_WARPGROUPS) {
        if (warp_id == 4) {
            pdl::wait();
            everyone::tma::cluster::wait();

            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int i = cluster_idx; i < total_num_tiles; i += num_clusters) {
                    auto [tile_row_idx, tile_col_idx] =
                        calculate_tile_idx(num_tile_rows_per_cluster, num_tile_cols, i);
                    load(tile_row_idx * fg::NUM_CLUSTERS * fg::CONSUMER_WARPS +
                             cta_rank * fg::CONSUMER_WARPS,
                         tile_col_idx,
                         input_stage_id);
                }
            }

            pdl::arrive();
        } else if (warp_id == 5 || warp_id == 6) {
            if (cta_rank == 0 && elect_warp_leader()) {
                int input_stage_id = 0;
                int epilogue_stage_id = 0;
                const int consumer_id = warp_id - 5;

                typename fg::C_tt_tile tmem[fg::TMEM_PIPELINE_STAGES];

                pdl::wait();
                everyone::tma::cluster::wait();

#pragma unroll
                for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
                    tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(
                        (i + consumer_id * fg::TMEM_PIPELINE_STAGES) * fg::COL_BLOCK);
                }

                for (int i = cluster_idx; i < total_num_tiles; i += num_clusters) {
                    // TODO: check consume
                    consume(tmem, input_stage_id, epilogue_stage_id, cta_rank);
                }
            }

            pdl::arrive();
        } else {
            const int tiles_per_device = (G.M / fg::ROW_BLOCK) * num_tile_cols;

            // 16 is the width of the LDMC load
            constexpr int tiles_per_thread =
                fg::ROW_BLOCK * fg::COL_BLOCK * sizeof(comm::bf16) / (WARP_THREADS * 16);
            constexpr int total_inner_iter = 8;

            pdl::wait();
            everyone::tma::cluster::wait();

            for (int tile_idx = laneid(); tile_idx < tiles_per_device; tile_idx += WARP_THREADS) {
                // TODO: potential IMA here

                // wait for signal
                const int row_idx = tile_idx / rows_per_device;
                const int col_idx = tile_idx % rows_per_device;
                if (elect_warp_leader()) {
                    int val;
                    do {
                        val = comm::atomic_u32::relaxed_load_s32_gpu(
                            &G.C_barrier[G.dev_idx][{row_idx, col_idx}]);
                    } while (val < fg::NUM_DEVICES);
                }
                __syncwarp();

                // do RS
                for (int iter_loop = 0; iter_loop < tiles_per_thread;
                     iter_loop += total_inner_iter) {
                    // TODO: could potentially raise inner_iter to 16?

                    comm::bf16_2 tmp[total_inner_iter][4];
#pragma unroll
                    for (int inner_iter = 0; inner_iter < total_inner_iter; inner_iter++) {
                        multimem_ld_reduce(
                            tmp[inner_iter],
                            G.C.mc_ptr_at({row_idx * fg::ROW_BLOCK,
                                           col_idx * fg::COL_BLOCK + iter_loop * total_inner_iter +
                                               inner_iter}));
                    }

#pragma unroll
                    for (int inner_iter = 0; inner_iter < total_inner_iter / 2; inner_iter++) {
                        store_float8(
                            &G.C[G.dev_idx][{row_idx * fg::ROW_BLOCK,
                                             col_idx * fg::COL_BLOCK +
                                                 iter_loop * total_inner_iter + inner_iter * 2}],
                            &tmp[inner_iter * 2]);
                    }
                }
            }
            pdl::arrive();
        }
    } else {
        // TODO: adjust indexing based on epilogue_stage_id input format
        int epilogue_stage_id[fg::CONSUMER_WARPS] = {};
        int epilogue_smem_idx = 0;
        typename fg::C_tt_tile tmem[fg::NUM_TMEM_SLOTS];

        everyone::tma::cluster::wait();
        wait(tmem_allocated, 0);

#pragma unroll
        for (int i = 0; i < fg::NUM_TMEM_SLOTS; i++) {
            tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(i * fg::COL_BLOCK);
        }

        for (int tile_id = cluster_idx; tile_id < total_num_tiles; tile_id += num_clusters) {
            auto [tile_row_idx, tile_col_idx] =
                calculate_tile_idx(num_tile_rows_per_cluster, num_tile_cols, tile_id);

            const int cluster_level_tile_id = tile_row_idx * fg::NUM_CLUSTERS * fg::CONSUMER_WARPS +
                cta_rank * fg::CONSUMER_WARPS;
#pragma unroll
            for (int c = 0; c < fg::CONSUMER_WARPS; c++) {
                const int warp_level_tile_id = cluster_level_tile_id + c;
                epilogue(warp_level_tile_id,
                         tile_col_idx,
                         tmem[c * fg::TMEM_PIPELINE_STAGES],
                         c,
                         epilogue_stage_id[c],
                         epilogue_smem_idx,
                         tile_id + num_clusters >= total_num_tiles);
            }

            // TODO: for now, I am going to signal with a 256 * 256 tile, tune later
            dist::tma::store_async_wait();

            const int row_start = cluster_level_tile_id * fg::A_tile::rows;
            const int device_to_signal = row_start / rows_per_device;
            const int local_tile_for_device = (row_start % rows_per_device) / fg::ROW_BLOCK;

            if (warpgroup::laneid() == 0) {
                if (device_to_signal == G.dev_idx) {
                    comm::atomic_u32::release_add_gpu(
                        &G.C_barrier[device_to_signal][{local_tile_for_device, tile_col_idx}], 1);
                } else {
                    comm::atomic_u32::release_add_sys(
                        &G.C_barrier[device_to_signal][{local_tile_for_device, tile_col_idx}], 1);
                }
            }
        }

        tensor_before_thread_sync();
        group<fg::EPILOGUE_WARPS>::sync(1);

        if (group<fg::EPILOGUE_WARPS>::warpid() == 0) {
            if (elect_warp_leader()) {
#pragma unroll
                for (int peer = 0; peer < fg::NUM_CTA; peer++) {
                    tma::cluster::arrive(tmem_finished, peer);
                }
            }
            // Only reach here if we finish with our tmem. Other party as well
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
__global__ __cluster_dims__(fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(
        fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>::NUM_THREADS,
        1) void fused_kernel_stub(const __grid_constant__
                                      fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR> G) {
    gemm_rs_blackwell<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>(G);
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR>
inline void launch_gemm_rs_blackwell(const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>& G) {
    cudaStream_t stream = G.stream;

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!G.stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
#endif

    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>;
    constexpr int smem_size = fg::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = fg::NUM_THREADS;
    constexpr int grid = fg::NUM_BLOCKS;

    auto this_kernel = fused_kernel_stub<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchAttribute pdl_attr = {};
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    pdl_attr.val.programmaticStreamSerializationAllowed = 1;

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = num_threads;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;
    launch_config.attrs = &pdl_attr;
    launch_config.numAttrs = 1;

    MKERNEL_CUDACHECK(cudaLaunchKernelEx(&launch_config, this_kernel, G));
    MKERNEL_CUDACHECK(cudaGetLastError());
}
};  // namespace gemm_rs_blackwell