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

__device__ __forceinline__ void reduce_load_16B(uint32_t* out, const comm::bf16* src) {
    asm volatile(
        "multimem.ld_reduce.relaxed.sys.global.add.acc::f32.v4.bf16x2 "
        "{%0, %1, %2, %3}, [%4];"
        : "=r"(out[0]), "=r"(out[1]), "=r"(out[2]), "=r"(out[3])
        : "l"(src)
        // TODO: maybe I can remove the clobber here?
        : "memory");
}

__device__ __forceinline__ void store_32B(comm::bf16* dst, const uint32_t* in) {
    // Store the packed BF16 bits without converting them to float.
    asm volatile("st.global.v8.b32 [%0], {%1, %2, %3, %4, %5, %6, %7, %8};"
                 :
                 : "l"(dst),
                   "r"(in[0]),
                   "r"(in[1]),
                   "r"(in[2]),
                   "r"(in[3]),
                   "r"(in[4]),
                   "r"(in[5]),
                   "r"(in[6]),
                   "r"(in[7])
                 : "memory");
}

template <int TILE_ROWS, int TILE_COLS, int STORES_PER_BATCH>
__device__ __forceinline__ void reduce_scatter_tile(const comm::bf16* multicast_tile,
                                                    comm::bf16* local_tile,
                                                    size_t row_stride,
                                                    int lane) {
    constexpr int ELEMENTS_PER_STORE = 16;
    constexpr int LANES_PER_ROW = TILE_COLS / ELEMENTS_PER_STORE;
    static_assert(TILE_COLS == 256, "RS maps each half warp to one 256-element row");
    // TODO: What is ROWS_PER_STORE?
    constexpr int ROWS_PER_STORE = 32 / LANES_PER_ROW;
    constexpr int ROWS_PER_BATCH = STORES_PER_BATCH * ROWS_PER_STORE;
    static_assert(STORES_PER_BATCH > 0 && STORES_PER_BATCH <= 16);
    static_assert(TILE_ROWS % ROWS_PER_BATCH == 0, "RS requires whole register batches");

    const int lane_col = (lane % LANES_PER_ROW) * ELEMENTS_PER_STORE;
    const int lane_row = lane / LANES_PER_ROW;

    for (int batch_row = 0; batch_row < TILE_ROWS; batch_row += ROWS_PER_BATCH) {
        // Scalarized by the unrolled loops: eight packed words per 32-byte store.
        uint32_t reduced[STORES_PER_BATCH][8];

#pragma unroll
        for (int store_idx = 0; store_idx < STORES_PER_BATCH; ++store_idx) {
            const int row = batch_row + store_idx * ROWS_PER_STORE + lane_row;
            const comm::bf16* src = multicast_tile + size_t(row) * row_stride + lane_col;
            reduce_load_16B(&reduced[store_idx][0], src);
            reduce_load_16B(&reduced[store_idx][4], src + 8);
        }

        // Recompute addresses instead of keeping a pointer array live with the payload.
#pragma unroll
        for (int store_idx = 0; store_idx < STORES_PER_BATCH; ++store_idx) {
            const int row = batch_row + store_idx * ROWS_PER_STORE + lane_row;
            comm::bf16* dst = local_tile + size_t(row) * row_stride + lane_col;
            store_32B(dst, reduced[store_idx]);
        }
    }
}

__device__ __forceinline__ void store_16B(comm::bf16* dst, const uint32_t* in) {
    asm volatile("st.global.v4.b32 [%0], {%1, %2, %3, %4};"
                 :
                 : "l"(dst), "r"(in[0]), "r"(in[1]), "r"(in[2]), "r"(in[3])
                 : "memory");
}

// Alternative RS layout: each lane owns one 16-byte chunk, so a warp covers a
// full 256-element row and every multimem load instruction is one contiguous
// 512-byte span. Pair with LOADS_PER_BATCH = 2 * STORES_PER_BATCH of
// reduce_scatter_tile to keep the same bytes in flight per lane.
template <int TILE_ROWS, int TILE_COLS, int LOADS_PER_BATCH>
__device__ __forceinline__ void reduce_scatter_tile_16B(const comm::bf16* multicast_tile,
                                                        comm::bf16* local_tile,
                                                        size_t row_stride,
                                                        int lane) {
    constexpr int ELEMENTS_PER_LOAD = 8;
    static_assert(TILE_COLS == 32 * ELEMENTS_PER_LOAD, "RS maps each warp to one 256-element row");
    static_assert(LOADS_PER_BATCH > 0 && LOADS_PER_BATCH <= 32);
    static_assert(TILE_ROWS % LOADS_PER_BATCH == 0, "RS requires whole register batches");

    const int lane_col = lane * ELEMENTS_PER_LOAD;

    for (int batch_row = 0; batch_row < TILE_ROWS; batch_row += LOADS_PER_BATCH) {
        uint32_t reduced[LOADS_PER_BATCH][4];

#pragma unroll
        for (int load_idx = 0; load_idx < LOADS_PER_BATCH; ++load_idx) {
            const int row = batch_row + load_idx;
            reduce_load_16B(reduced[load_idx], multicast_tile + size_t(row) * row_stride + lane_col);
        }

#pragma unroll
        for (int load_idx = 0; load_idx < LOADS_PER_BATCH; ++load_idx) {
            const int row = batch_row + load_idx;
            store_16B(local_tile + size_t(row) * row_stride + lane_col, reduced[load_idx]);
        }
    }
}

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

__host__ __device__ __forceinline__ bool epoch_reached(uint32_t observed, uint32_t expected) {
    return uint32_t(observed - expected) < 0x80000000u;
}

}  // namespace

// A normal same-stream launch after fused_kernel_stub establishes completion of
// every local CTA and its global writes before this rank publishes its epoch.
// Do not enable PDL or signal early launch completion in this barrier kernel.
__global__ void exit_epoch_barrier(completion_epoch_dist_tensor epochs, int dev_idx) {
    __shared__ uint32_t completed_epoch;
    if (threadIdx.x == 0) {
        auto* local_epoch = epochs[dev_idx].raw_ptr;
        completed_epoch = comm::atomic_u32::acquire_load_gpu(local_epoch) + 1u;
        comm::atomic_u32::release_store_sys(local_epoch, completed_epoch);
    }
    __syncthreads();

    for (int peer = threadIdx.x; peer < INTRA_NUM_DEVICES; peer += blockDim.x) {
        while (!epoch_reached(comm::atomic_u32::acquire_load_sys(epochs[peer].raw_ptr),
                              completed_epoch)) {
            __nanosleep(16);
        }
    }
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
__device__ __forceinline__ void gemm_rs_blackwell(
    const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>& G) {
    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>;

    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A.template prefetch_tma<typename fg::A_tile>();
        G.B.template prefetch_tma<typename fg::B_tile>();
        G.C[G.dev_idx].template prefetch_tma<typename fg::C_tile>();
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

            tma::cluster::expect_bytes(tma_load[input_stage_id],
                                       sizeof(fg::A_tile) * fg::CONSUMER_WARPS + sizeof(fg::B_tile),
                                       0);
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
            mm_ABt_ncta<fg::NUM_CLUSTERS>(
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
            mma_ABt_ncta<fg::NUM_CLUSTERS>(
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
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
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
        if (warp_id == fg::PRODUCER_WARP) {
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
        } else if (warp_id >= fg::FIRST_CONSUMER_WARP &&
                   warp_id < fg::FIRST_CONSUMER_WARP + fg::CONSUMER_WARPS) {
            pdl::wait();
            everyone::tma::cluster::wait();

            // The two-CTA allocation requires a complete warp in each CTA,
            // before restricting MMA issue to the leader in CTA 0.
            if (warp_id == fg::FIRST_CONSUMER_WARP) {
                tm_alloc.provision(tmem_addr);
                if (elect_warp_leader()) {
                    arrive(tmem_allocated);
                }
            } else {
                wait(tmem_allocated, 0);
            }
            tm_alloc.set_addr(tmem_addr);

            if (cta_rank == 0 && elect_warp_leader()) {
                int input_stage_id = 0;
                int epilogue_stage_id = 0;
                const int consumer_id = warp_id - fg::FIRST_CONSUMER_WARP;

                typename fg::C_tt_tile tmem[fg::TMEM_PIPELINE_STAGES];

#pragma unroll
                for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
                    tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(
                        (i + consumer_id * fg::TMEM_PIPELINE_STAGES) * fg::COL_BLOCK);
                }

                for (int i = cluster_idx; i < total_num_tiles; i += num_clusters) {
                    // TODO: check consume
                    consume(tmem, input_stage_id, epilogue_stage_id, consumer_id);
                }
            }

            pdl::arrive();
        } else {
            // const int lane = laneid();
            // const int tile_rows_per_device = rows_per_device / fg::ROW_BLOCK;
            // const int tiles_per_device = tile_rows_per_device * num_tile_cols;

            // const int device_row_start = G.dev_idx * rows_per_device;

            // pdl::wait();
            // everyone::tma::cluster::wait();

            // // Should only read after PDL wait since the previous iteration may have modified the
            // // epoch
            // const uint32_t epoch =
            //     comm::atomic_u32::acquire_load_gpu(G.completion_epochs[G.dev_idx].raw_ptr) + 1u;
            // const uint32_t ready_count = epoch * uint32_t(fg::NUM_DEVICES);

            // for (int tile_idx = blockIdx.x; tile_idx < tiles_per_device;
            //      tile_idx += fg::NUM_BLOCKS) {
            //     // TODO: review indexing to make sure that all devices have active CTAs
            //     const int local_tile_row = tile_idx / num_tile_cols;
            //     const int tile_col = tile_idx % num_tile_cols;

            //     // These match the coordinates used by the epilogue's signal.
            //     if (lane == 0) {
            //         // Peers publish tile arrivals with system-scope atomics.
            //         while (!epoch_reached(comm::atomic_u32::acquire_load_sys(
            //                                   &G.C_barrier[G.dev_idx][{local_tile_row,
            //                                   tile_col}]),
            //                               ready_count)) {
            //             __nanosleep(16);
            //         }
            //     }
            //     __syncwarp();

            //     const int tile_row_start = device_row_start + local_tile_row * fg::ROW_BLOCK;
            //     const int tile_col_start = tile_col * fg::COL_BLOCK;
            //     reduce_scatter_tile<fg::ROW_BLOCK, fg::COL_BLOCK, fg::RS_STORES_PER_BATCH>(
            //         G.C.mc_ptr_at({tile_row_start, tile_col_start}),
            //         &G.C[G.dev_idx][{tile_row_start, tile_col_start}],
            //         G.C[G.dev_idx].cols(),
            //         lane);
            // }
        }

        pdl::arrive();
    } else {
        int epilogue_stage_id[fg::CONSUMER_WARPS] = {};
        int epilogue_smem_idx = 0;
        typename fg::C_tt_tile tmem[fg::NUM_TMEM_SLOTS];

        everyone::tma::cluster::wait();
        wait(tmem_allocated, 0);
        // Tile views and the eventual deallocation use the published base.
        tm_alloc.set_addr(tmem_addr);

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
                         &tmem[c * fg::TMEM_PIPELINE_STAGES],
                         c,
                         epilogue_stage_id[c],
                         epilogue_smem_idx,
                         tile_id + num_clusters >= total_num_tiles);
            }

            // // TODO: for now, I am going to signal with a 256 * 256 tile, tune later
            // dist::tma::store_async_wait();

            // // TODO: maybe the only way to validate this is by plugging in actual numbers?
            // const int row_start = cluster_level_tile_id * fg::A_tile::rows;
            // const int device_to_signal = row_start / rows_per_device;
            // const int local_tile_for_device = (row_start % rows_per_device) / fg::ROW_BLOCK;

            // if (warpgroup::laneid() == 0) {
            //     if (device_to_signal == G.dev_idx) {
            //         comm::atomic_u32::release_add_gpu(
            //             &G.C_barrier[device_to_signal][{local_tile_for_device, tile_col_idx}],
            //             1);
            //     } else {
            //         comm::atomic_u32::release_add_sys(
            //             &G.C_barrier[device_to_signal][{local_tile_for_device, tile_col_idx}],
            //             1);
            //     }
            // }
        }

        tensor_before_thread_sync();
        group<fg::EPILOGUE_WARPS>::sync(1);

        if (group<fg::EPILOGUE_WARPS>::warpid() == 0) {
            if (elect_warp_leader()) {
#pragma unroll
                for (int peer = 0; peer < fg::NUM_CLUSTERS; peer++) {
                    tma::cluster::arrive(tmem_finished, peer);
                }
            }
            // Only reach here if we finish with our tmem. Other party as well
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

// Standalone version of the RS warp's tile loop: one warp per CTA and the
// fused kernel's ROW_BLOCK x COL_BLOCK tiles, but write a compact, separate
// output shard so repeated benchmark launches do not reduce previously reduced
// input. LANE_BYTES selects the lane layout: 32 = two strided 16B loads + one
// 32B store per lane (the fused helper); 16 = one contiguous 16B load/store per
// lane, with the same bytes in flight per lane.
template <int LANE_BYTES>
__global__ __launch_bounds__(32) void rs_only_kernel(
    const comm::bf16* multicast_input, comm::bf16* output, int rows_per_rank, int N, int rank) {
    using fg = fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>;
    const int tile_cols = N / fg::COL_BLOCK;
    const int tiles = (rows_per_rank / fg::ROW_BLOCK) * tile_cols;
    for (int tile = blockIdx.x; tile < tiles; tile += gridDim.x) {
        const int row = (tile / tile_cols) * fg::ROW_BLOCK;
        const int col = (tile % tile_cols) * fg::COL_BLOCK;
        const comm::bf16* src = multicast_input + (size_t(rank) * rows_per_rank + row) * N + col;
        comm::bf16* dst = output + size_t(row) * N + col;
        if constexpr (LANE_BYTES == 32) {
            reduce_scatter_tile<fg::ROW_BLOCK, fg::COL_BLOCK, fg::RS_STORES_PER_BATCH>(
                src, dst, N, threadIdx.x);
        } else {
            static_assert(LANE_BYTES == 16, "LANE_BYTES must be 16 or 32");
            reduce_scatter_tile_16B<fg::ROW_BLOCK, fg::COL_BLOCK, 2 * fg::RS_STORES_PER_BATCH>(
                src, dst, N, threadIdx.x);
        }
    }
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
__global__ __cluster_dims__(
    fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(
        fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>::NUM_THREADS,
        1) void fused_kernel_stub(const __grid_constant__ fused_globals<_NUM_EPILOGUE_WARPS,
                                                                        _C_TILE_DIVISOR,
                                                                        _NUM_CONSUMER_WARPS> G) {
    gemm_rs_blackwell<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>(G);
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
inline void initialize_gemm_rs_blackwell_state(
    const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>& G) {
    cudaStream_t stream = G.stream;
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
#endif
    const auto& ready = G.C_barrier[G.dev_idx];
    const size_t ready_bytes = size_t(ready.rows()) * ready.cols() * sizeof(uint32_t);
    MKERNEL_CUDACHECK(cudaMemsetAsync(ready.raw_ptr, 0, ready_bytes, stream));
    MKERNEL_CUDACHECK(
        cudaMemsetAsync(G.completion_epochs[G.dev_idx].raw_ptr, 0, sizeof(uint32_t), stream));
}

template <int _NUM_EPILOGUE_WARPS, int _C_TILE_DIVISOR, int _NUM_CONSUMER_WARPS>
inline void launch_gemm_rs_blackwell(
    const fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>& G,
    bool with_exit_barrier) {
    cudaStream_t stream = G.stream;

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
    if (!G.stream) {
        stream = at::cuda::getCurrentCUDAStream().stream();
    }
#endif

    using fg = fused_globals<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>;
    constexpr int smem_size = fg::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = fg::NUM_THREADS;
    constexpr int grid = fg::NUM_BLOCKS;

    auto this_kernel = fused_kernel_stub<_NUM_EPILOGUE_WARPS, _C_TILE_DIVISOR, _NUM_CONSUMER_WARPS>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    cudaLaunchAttribute pdl_attr = {};
    pdl_attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
    // Without the exit barrier, serialize complete GEMMs, including output stores.
    pdl_attr.val.programmaticStreamSerializationAllowed = with_exit_barrier ? 1 : 0;

    cudaLaunchConfig_t launch_config = {};
    launch_config.gridDim = grid;
    launch_config.blockDim = num_threads;
    launch_config.dynamicSmemBytes = smem_size;
    launch_config.stream = stream;
    launch_config.attrs = &pdl_attr;
    launch_config.numAttrs = 1;

    MKERNEL_CUDACHECK(cudaLaunchKernelEx(&launch_config, this_kernel, G));
    MKERNEL_CUDACHECK(cudaGetLastError());

    // Full completion dependency, including in captured graphs. The next use
    // of these buffers on this stream cannot pass until every rank is done.
    if (with_exit_barrier) {
        exit_epoch_barrier<<<1, 32, 0, stream>>>(G.completion_epochs, G.dev_idx);
        MKERNEL_CUDACHECK(cudaGetLastError());
    }
}
// Local storage in the existing argument layout. With RS disabled, only C[0]
// is accessed; multicast addresses and communication counters are unused.
template <int NUM_CONSUMER_WARPS>
inline auto make_gemm_only_globals(
    uint64_t a, uint64_t b, uint64_t c, int M, int N, int K, cudaStream_t stream) {
    using fg =
        fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR, NUM_CONSUMER_WARPS>;
    uint64_t outputs[INTRA_NUM_DEVICES];
    uint64_t unused[INTRA_NUM_DEVICES] = {};
    std::fill_n(outputs, INTRA_NUM_DEVICES, c);
    return fg{
        .dev_idx = 0,
        .M = M,
        .N = N,
        .K = K,
        .A = dist::make_local_tensor<typename fg::A_local_tensor>(a, 1, 1, M, K),
        .B = dist::make_local_tensor<typename fg::B_local_tensor>(b, 1, 1, N, K),
        .C = dist::make_distributed_tensor<typename fg::C_dist_tensor>(0, outputs, 1, 1, M, N),
        .C_barrier =
            dist::make_distributed_tensor<typename fg::C_ready_barrier>(0, unused, 1, 1, 1, 1),
        .completion_epochs =
            dist::make_distributed_tensor<completion_epoch_dist_tensor>(unused, 1, 1, 1, 1),
        .stream = stream};
}

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
inline void launch_rs_only(dist::ParallelBuffer& input,
                           at::Tensor& output,
                           dist::ParallelBuffer& epochs,
                           int lane_bytes,
                           bool with_exit_barrier) {
    using fg = fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>;
    TORCH_CHECK(input.local_world_size_ == INTRA_NUM_DEVICES &&
                    epochs.local_world_size_ == INTRA_NUM_DEVICES &&
                    input.local_rank_ == epochs.local_rank_,
                "Input and epochs must match the compiled world size and local rank");
    TORCH_CHECK(input.multicast_ && input.multicast_ptr_, "Input requires multicast storage");
    TORCH_CHECK(input.data_.dim() == 2 && input.data_.scalar_type() == at::kBFloat16 &&
                    input.data_.is_contiguous(),
                "Input must be a contiguous BF16 matrix");
    TORCH_CHECK(output.is_cuda() && output.device() == input.data_.device() &&
                    output.scalar_type() == at::kBFloat16 && output.dim() == 2 &&
                    output.is_contiguous(),
                "Output must be contiguous BF16 on the input device");
    TORCH_CHECK(lane_bytes == 16 || lane_bytes == 32, "lane_bytes must be 16 or 32");
    const int64_t M = input.data_.size(0), N = input.data_.size(1);
    TORCH_CHECK(
        M > 0 && M % (INTRA_NUM_DEVICES * fg::ROW_BLOCK) == 0 && N > 0 && N % fg::COL_BLOCK == 0,
        "M must be divisible by world_size * ", fg::ROW_BLOCK, "; N by ", fg::COL_BLOCK);
    TORCH_CHECK(output.size(0) == M / INTRA_NUM_DEVICES && output.size(1) == N,
                "Expected output shape [M/world_size, N]");
    TORCH_CHECK(epochs.data_.numel() == 1 && epochs.data_.scalar_type() == at::kInt,
                "Epochs must contain one int32 per rank, initialized to zero before use");
    const auto input_begin = reinterpret_cast<uintptr_t>(input.data_.data_ptr());
    const auto output_begin = reinterpret_cast<uintptr_t>(output.data_ptr());
    TORCH_CHECK(output_begin + output.nbytes() <= input_begin ||
                    input_begin + input.data_.nbytes() <= output_begin,
                "Standalone RS output must not overlap input");
    c10::cuda::CUDAGuard device_guard(input.local_rank_);
    const auto stream = at::cuda::getCurrentCUDAStream().stream();
    auto kernel = lane_bytes == 16 ? rs_only_kernel<16> : rs_only_kernel<32>;
    kernel<<<fg::NUM_BLOCKS, 32, 0, stream>>>(
        reinterpret_cast<const comm::bf16*>(input.multicast_ptr_),
        reinterpret_cast<comm::bf16*>(output.data_ptr()),
        M / INTRA_NUM_DEVICES,
        N,
        input.local_rank_);
    MKERNEL_CUDACHECK(cudaGetLastError());
    if (with_exit_barrier) {
        auto epoch_tensor =
            dist::distributed_tensor_from_buffer<completion_epoch_dist_tensor>(epochs);
        exit_epoch_barrier<<<1, 32, 0, stream>>>(epoch_tensor, input.local_rank_);
        MKERNEL_CUDACHECK(cudaGetLastError());
    }
}

template <int NUM_CONSUMER_WARPS>
inline void launch_gemm_only_variant(at::Tensor& A, at::Tensor& B, at::Tensor& C) {
    using fg =
        fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR, NUM_CONSUMER_WARPS>;
    TORCH_CHECK(A.is_cuda() && B.is_cuda() && C.is_cuda(), "A, B, C must be CUDA tensors");
    TORCH_CHECK(A.device() == B.device() && A.device() == C.device(),
                "A, B, C must be on the same device");
    TORCH_CHECK(A.scalar_type() == at::kBFloat16 && B.scalar_type() == at::kBFloat16 &&
                    C.scalar_type() == at::kBFloat16,
                "A, B, C must be BF16");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2 && C.dim() == 2 && A.is_contiguous() &&
                    B.is_contiguous() && C.is_contiguous(),
                "A, B, C must be contiguous matrices");
    TORCH_CHECK(A.size(1) == B.size(1) && C.size(0) == A.size(0) && C.size(1) == B.size(0),
                "Expected A[M,K], B[N,K], C[M,N]");
    TORCH_CHECK(A.size(0) > 0 && B.size(0) > 0 && A.size(1) > 0 &&
                    A.size(0) % (fg::ROW_BLOCK * fg::NUM_CLUSTERS) == 0 &&
                    B.size(0) % fg::COL_BLOCK == 0 && A.size(1) % fg::RED_BLOCK == 0,
                "M, N, K must be positive multiples of ",
                fg::ROW_BLOCK * fg::NUM_CLUSTERS,
                ", ",
                fg::COL_BLOCK,
                ", ",
                fg::RED_BLOCK);
    c10::cuda::CUDAGuard device_guard(A.device());
    auto globals =
        make_gemm_only_globals<NUM_CONSUMER_WARPS>(reinterpret_cast<uint64_t>(A.data_ptr()),
                                                   reinterpret_cast<uint64_t>(B.data_ptr()),
                                                   reinterpret_cast<uint64_t>(C.data_ptr()),
                                                   A.size(0),
                                                   B.size(0),
                                                   A.size(1),
                                                   at::cuda::getCurrentCUDAStream().stream());
    launch_gemm_rs_blackwell(globals, false);
}

inline void launch_gemm_only(at::Tensor& A, at::Tensor& B, at::Tensor& C, int num_consumer_warps) {
    TORCH_CHECK(num_consumer_warps == 1 || num_consumer_warps == 2,
                "num_consumer_warps must be 1 or 2");
    if (num_consumer_warps == 1) {
        launch_gemm_only_variant<1>(A, B, C);
    } else {
        launch_gemm_only_variant<2>(A, B, C);
    }
}

inline void prepare(at::Tensor& A,
                    at::Tensor& B,
                    dist::ParallelBuffer& C,
                    dist::ParallelBuffer& ready,
                    dist::ParallelBuffer& completion_epochs) {
    auto globals =
        gemm_rs_blackwell_make_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>(
            A,
            B,
            C,
            ready,
            completion_epochs,
            C.local_rank_,
            A.size(0),
            B.size(0),
            A.size(1),
            at::cuda::getCurrentCUDAStream().stream());
    initialize_gemm_rs_blackwell_state(globals);
}

inline void launch(at::Tensor& A,
                   at::Tensor& B,
                   dist::ParallelBuffer& C,
                   dist::ParallelBuffer& ready,
                   dist::ParallelBuffer& completion_epochs,
                   bool with_exit_barrier = true) {
    auto globals =
        gemm_rs_blackwell_make_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>(
            A,
            B,
            C,
            ready,
            completion_epochs,
            C.local_rank_,
            A.size(0),
            B.size(0),
            A.size(1),
            at::cuda::getCurrentCUDAStream().stream());
    launch_gemm_rs_blackwell(globals, with_exit_barrier);
}
#endif

// Instantiate the default API so the compile target checks the fused kernel,
// initialization, and the exit-barrier launch rather than just template syntax.
template void
initialize_gemm_rs_blackwell_state<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>(
    const fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>& G);
template void launch_gemm_rs_blackwell<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>(
    const fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>& G,
    bool with_exit_barrier);
};  // namespace gemm_rs_blackwell

#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#include "operators/gemm_rs/gemm_rs_blackwell_session.cuh"
#endif
