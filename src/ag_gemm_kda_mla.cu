/**
 * AG-GEMM but for KDA's proj_qkvgfab and MLA's qkvg proj. Putting it in a different file just in
 * case more operations have to be fused later, depending on how well communication is hidden
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
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
#include "common/timings.cuh"
#include "common/tk_common_base_types.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_register_rt.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/tk_types_tensor.cuh"
#include "common/types.cuh"
#include "dist/dbuf_buffer_bridge.cuh"
#include "dist/distributed_buffer.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "memory/tk_ops_thread_memory_tile_tma.cuh"
#include "memory/tk_ops_thread_util_sync.cuh"
#include "memory/tk_ops_thread_util_tma.cuh"
#include "memory/tk_ops_thread_util_util.cuh"
#include "operators/ag_gemm/ag_gemm_kda_mla.cuh"

// clang-format off
// this has to go under tk_ops_group_group
#include "dist/tma.cuh"
#include "memory/tk_ops_group_util_util.cuh"
// clang-format on

// MAJOR TODO: think about how to deal with bad shapes!

using namespace kittens;

namespace ag_gemm_kda_mla {

namespace {

// Process-lifetime resources, one set per local CUDA device. This extension
// uses one host process per GPU, and launches are serialized by that process.
struct ACopyPipelineState {
    cudaStream_t stream = nullptr;
    cudaEvent_t main_pre_event = nullptr;
    uint32_t* ready = nullptr;
    uint32_t epoch = 0;
    bool initialized = false;
};

ACopyPipelineState A_copy_states[INTRA_NUM_DEVICES];

inline ACopyPipelineState& get_A_copy_state(int dev_idx) {
    ACopyPipelineState& state = A_copy_states[dev_idx];
    if (!state.initialized) {
        MKERNEL_CUDACHECK(cudaStreamCreateWithFlags(&state.stream, cudaStreamNonBlocking));
        MKERNEL_CUDACHECK(cudaEventCreateWithFlags(&state.main_pre_event, cudaEventDisableTiming));
        MKERNEL_CUDACHECK(cudaMalloc(&state.ready, INTRA_NUM_DEVICES * sizeof(uint32_t)));
        MKERNEL_CUDACHECK(cudaMemset(state.ready, 0, INTRA_NUM_DEVICES * sizeof(uint32_t)));
        state.initialized = true;
    }
    return state;
}

}  // namespace
#ifdef PROFILE_TIMINGS
// Payload discriminators, matching the role branch at the bottom of the kernel:
// warps 0-3 are the epilogue warpgroup (its leader stamps), warp 4 loads, warp 5
// issues the MMA.
static constexpr int PROFILE_EPILOGUE_WARP = 0;
static constexpr int PROFILE_PRODUCER_WARP = 4;
static constexpr int PROFILE_CONSUMER_WARP = 5;
#endif

// traverse the grid in a snake like pattern to raise L2 cache reuse
// https://github.com/HazyResearch/ThunderKittens/blob/0230013a72b51338a137b50f69538ec69d4d4675/include/common/util.cuh#L367
template <int SUPERGROUP_WIDTH = 5>
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
};

template <int _ROW_BLOCK, int _COL_BLOCK>
__device__ __forceinline__ void ag_gemm_kda_mla(const fused_globals<_ROW_BLOCK, _COL_BLOCK>& G) {
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK>;

    const int cta_rank = cluster_ctarank();
    const int warp_id = warpid();
    const int warpgroup_id = warpgroupid();

    if (warp_id == 0 && elect_warp_leader()) {
        G.A[G.dev_idx].template prefetch_tma<typename fg::A_tile>();
        G.A_local_buf.template prefetch_tma<typename fg::A_tile>();
        G.B.template prefetch_tma<typename fg::B_tile>();
        G.C.template prefetch_tma<typename fg::C_tile>();
    }
    // Zero the CTA's ring head and make it visible before anyone emits. One
    // lane per warp stamps SETUP_BEGIN; the renderer drops the warps that are
    // not plotted rows, so this costs a handful of records per CTA.
    MKERNEL_TIMING_PROLOGUE();
    MKERNEL_EMIT_IF(elect_warp_leader(), G.timings, warp_id, EV_SETUP_BEGIN, 0);

    const int cluster_idx = blockIdx.x / fg::NUM_CLUSTERS;
    const int local_m = G.A.rows();
    const int row_tiles_per_device = local_m / fg::ROW_BLOCK;
    const int cluster_rows_per_device = row_tiles_per_device / fg::NUM_CLUSTERS;
    const int num_comp_clusters = fg::NUM_BLOCKS / fg::NUM_CLUSTERS;

    // round up to the nearest multiple of the COL_BLOCK
    const int num_col_tiles = (G.N + _COL_BLOCK - 1) / _COL_BLOCK;
    const int cluster_tiles_per_device = cluster_rows_per_device * num_col_tiles;

    extern __shared__ int __shm[];
    tma_swizzle_allocator smem_allocator((int*)&__shm[0]);

    typename fg::pipeline_inputs(&inputs_smem)[fg::PRODUCER_CONSUMER_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::pipeline_inputs, fg::PRODUCER_CONSUMER_PIPELINE_STAGES>();
    typename fg::C_tile(&C_smem)[fg::EPILOGUE_PIPELINE_STAGES] =
        smem_allocator.allocate<fg::C_tile, fg::EPILOGUE_PIPELINE_STAGES>();

    __shared__ semaphore tma_load[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];
    __shared__ semaphore mma_finish[fg::PRODUCER_CONSUMER_PIPELINE_STAGES];

    __shared__ semaphore epilogue_ready[fg::TMEM_PIPELINE_STAGES];
    __shared__ semaphore epilogue_tmem_finished[fg::TMEM_PIPELINE_STAGES];

    __shared__ semaphore tmem_finished;

    __shared__ uint32_t tmem_addr;
    tensor_allocator<1, fg::NUM_CLUSTERS> tm_alloc{};

    uint32_t phasebits = fg::PHASE_BITS_INIT;

    if (warp_id == 0 && elect_warp_leader()) {
#pragma unroll
        for (int i = 0; i < fg::PRODUCER_CONSUMER_PIPELINE_STAGES; i++) {
            // tma finish has to be broadcasted to mma warp
            init_semaphore(tma_load[i], 0, 2);
            // mma warp will broadcast finish
            init_semaphore(mma_finish[i], 0, 1);
        }

#pragma unroll
        for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
            init_semaphore(epilogue_ready[i], 0, 1);
            // tmem finish has to be broadcasted back
            init_semaphore(epilogue_tmem_finished[i], WARPGROUP_WARPS * fg::NUM_CLUSTERS);
        }

        init_semaphore(tmem_finished, 1);
    } else if (warp_id == 1) {
        tm_alloc.provision(tmem_addr);
    }

    tensor_before_thread_sync();
    __syncthreads();
    tensor_after_thread_sync();
    tm_alloc.set_addr(tmem_addr);

    // flush to ensure the mbarriers are visible
    everyone::tma::cluster::sync();

    MKERNEL_EMIT_IF(elect_warp_leader(), G.timings, warp_id, EV_SETUP_DONE, 0);

    // Sequence numbers for the payload. They only have to be unique per
    // (warp, event pair), so the producer's steps and the consumer's tiles can
    // use independent counters.
    MKERNEL_TIMING_ONLY(uint32_t prod_step_seq = 0;)
    MKERNEL_TIMING_ONLY(uint32_t mma_tile_seq = 0;)
    MKERNEL_TIMING_ONLY(uint32_t mma_step_seq = 0;)
    MKERNEL_TIMING_ONLY(uint32_t epi_tile_seq = 0;)
    // Which tile the consumer is on, published by its loop so consume() can tell
    // a local A tile from one pulled over NVLink. Profile-only: the shipping
    // build never computes it.
    MKERNEL_TIMING_ONLY(int mma_tile_id = 0;)

    auto load = [&](int tile_row_idx, int tile_col_idx, int target_device, int& input_stage_id) {
        const int actual_target_device = (target_device + G.dev_idx) % fg::NUM_DEVICES;
        const bool is_local = actual_target_device == G.dev_idx;

        // The stream memory write executes on this GPU after the peer-to-local
        // D2D copy. Both the payload and flag reside in local HBM, so GPU-scope
        // acquire is sufficient for the consumer.
        if (!is_local) {
            while (comm::atomic_u32::acquire_load_gpu(&G.A_copy_ready[actual_target_device]) <
                   G.A_copy_epoch) {
                __nanosleep(64);
            }
        }

        const typename fg::A_local_tensor& A_gmem =
            is_local ? G.A[actual_target_device] : G.A_local_buf;
        const int A_tile_row_idx =
            is_local ? tile_row_idx : actual_target_device * row_tiles_per_device + tile_row_idx;

        for (int iter_k = 0; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;

            MKERNEL_EMIT(G.timings, PROFILE_PRODUCER_WARP, EV_LOAD_STEP_BEGIN, prod_step_seq);

            wait(mma_finish[input_stage_id], (phasebits >> 0 & 0b1));

            // Gap to LOAD_STEP_BEGIN is the producer stalled on the ring: the
            // consumer has not drained this stage yet.
            MKERNEL_EMIT(G.timings, PROFILE_PRODUCER_WARP, EV_LOAD_MMA_FREE, prod_step_seq);

            tma::cluster::expect_bytes(
                tma_load[input_stage_id], sizeof(fg::A_tile) + sizeof(fg::B_tile), 0);

            tma::cluster::load_async(B_smem,
                                     G.B,
                                     {iter_k, tile_col_idx * fg::NUM_CLUSTERS + cta_rank},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            tma::cluster::load_async(A_smem,
                                     A_gmem,
                                     {A_tile_row_idx, iter_k},
                                     tma_load[input_stage_id],
                                     (uint16_t)(1 << cta_rank),
                                     0);

            MKERNEL_EMIT(G.timings, PROFILE_PRODUCER_WARP, EV_LOAD_TMA_ISSUED, prod_step_seq);
            MKERNEL_TIMING_ONLY(prod_step_seq++;)

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= 0b1;
            }
        }
    };
    auto consume = [&](typename fg::C_tt_tile* tmem, int& input_stage_id, int& epilogue_stage_id) {
        // target_device 0 means A[(0 + dev_idx) % N] -- this device's own slice.
        // Anything else is an NVLink read, and the wait below is where that
        // latency lands.
        MKERNEL_TIMING_ONLY(const uint32_t inputs_ev = (mma_tile_id / cluster_tiles_per_device) != 0
                                ? EV_MMA_INPUTS_REMOTE
                                : EV_MMA_INPUTS_LOCAL;)

        MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_TILE_BEGIN, mma_tile_seq);

        wait(epilogue_tmem_finished[epilogue_stage_id], (phasebits >> 2) & 0b1);

        // Gap to MMA_TILE_BEGIN is the mainloop blocked on the epilogue: this
        // TMEM stage has not been drained yet.
        MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_TMEM_FREE, mma_tile_seq);
        MKERNEL_TIMING_ONLY(mma_tile_seq++;)

        {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;

            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_STEP_BEGIN, mma_step_seq);

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, inputs_ev, mma_step_seq);

            mm2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_ISSUED, mma_step_seq);
            MKERNEL_TIMING_ONLY(mma_step_seq++;)

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        for (int iter_k = 1; iter_k < G.K / fg::RED_BLOCK; iter_k++) {
            typename fg::A_tile& A_smem = inputs_smem[input_stage_id].A;
            typename fg::B_tile& B_smem = inputs_smem[input_stage_id].B;

            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_STEP_BEGIN, mma_step_seq);

            wait(tma_load[input_stage_id], (phasebits >> 1) & 0b1);

            // Gap to MMA_STEP_BEGIN is the MMA starved of inputs. Split local vs
            // remote: a remote bar that is much longer is the all-gather failing
            // to hide behind the compute.
            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, inputs_ev, mma_step_seq);

            mma2_AB(tmem[epilogue_stage_id], A_smem, B_smem, mma_finish[input_stage_id]);

            MKERNEL_EMIT(G.timings, PROFILE_CONSUMER_WARP, EV_MMA_ISSUED, mma_step_seq);
            MKERNEL_TIMING_ONLY(mma_step_seq++;)

            input_stage_id = (input_stage_id + 1) % fg::PRODUCER_CONSUMER_PIPELINE_STAGES;
            if (input_stage_id == 0) {
                phasebits ^= (1 << 1);
            }
        }

        kittens::detail::tcgen05::commit<fg::NUM_CLUSTERS>(epilogue_ready[epilogue_stage_id]);
        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;

        if (epilogue_stage_id == 0) {
            phasebits ^= (1 << 2);
        }
    };

    auto epilogue = [&](int tile_row_idx,
                        int tile_col_idx,
                        typename fg::C_tt_tile* tmem,
                        int& epilogue_stage_id,
                        int& epilogue_transfer_stage_id) {
        const auto& C_out = G.C;
        constexpr int C_CHUNK_COLS = fg::COL_BLOCK / fg::C_TILE_DIVISOR;
        rt_bf<fg::ROW_BLOCK / WARPGROUP_WARPS, C_CHUNK_COLS> c_reg[fg::C_TILE_DIVISOR];

        // One row per epilogue warpgroup, stamped by its leader -- which is also
        // the lane that issues the TMA store, so every phase closes on the same
        // thread that opened it.
        MKERNEL_TIMING_ONLY(const bool epi_leader = (warpgroup::laneid() == 0);)

        MKERNEL_EMIT_IF(
            epi_leader, G.timings, PROFILE_EPILOGUE_WARP, EV_EPI_TILE_BEGIN, epi_tile_seq);

        wait(epilogue_ready[epilogue_stage_id], (phasebits >> 3) & 0b1);

        // Gap to EPI_TILE_BEGIN is the epilogue waiting on the mainloop.
        MKERNEL_EMIT_IF(
            epi_leader, G.timings, PROFILE_EPILOGUE_WARP, EV_EPI_MMA_DONE, epi_tile_seq);

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            warpgroup::load_async(
                c_reg[i],
                // TODO: review this indexing
                tmem[epilogue_stage_id].template subtile<tt<float, fg::ROW_BLOCK, C_CHUNK_COLS>>(
                    i * C_CHUNK_COLS));
        }

        tensor_load_wait();

        MKERNEL_EMIT_IF(
            epi_leader, G.timings, PROFILE_EPILOGUE_WARP, EV_EPI_TMEM_READ, epi_tile_seq);

        if (elect_warp_leader()) {
            tma::cluster::arrive(epilogue_tmem_finished[epilogue_stage_id], 0);
        }

#pragma unroll
        for (int i = 0; i < fg::C_TILE_DIVISOR; i++) {
            // need to know that there is at least 1 slot of smem in C tile that is free
            dist::tma::store_async_read_wait<fg::EPILOGUE_PIPELINE_STAGES - 1>();
            warpgroup::sync(1);
            // this already does the swizzle inside it
            warpgroup::store(C_smem[epilogue_transfer_stage_id], c_reg[i]);
            warpgroup::sync(1);

            if (warpgroup::laneid() == 0) {
                // C_tile is only COL_BLOCK / EPILOGUE_STAGES wide, so the TMA
                // column coordinate counts chunks, not COL_BLOCK tiles.
                dist::tma::store_async<dim::ROW, cache_policy::EVICT_FIRST>(
                    C_out,
                    C_smem[epilogue_transfer_stage_id],
                    {tile_row_idx, tile_col_idx * fg::C_TILE_DIVISOR + i});
            }

            epilogue_transfer_stage_id =
                (epilogue_transfer_stage_id + 1) % fg::EPILOGUE_PIPELINE_STAGES;
        }

        // Every chunk is staged to SMEM and its TMA store issued; the stores
        // themselves stay in flight until the drain after the tile loop.
        MKERNEL_EMIT_IF(
            epi_leader, G.timings, PROFILE_EPILOGUE_WARP, EV_EPI_SMEM_WRITTEN, epi_tile_seq);
        MKERNEL_TIMING_ONLY(epi_tile_seq++;)

        epilogue_stage_id = (epilogue_stage_id + 1) % fg::TMEM_PIPELINE_STAGES;
        if (epilogue_stage_id == 0) {
            phasebits ^= (0b1 << 3);
        }
    };

    if (warpgroup_id >= fg::EPILOGUE_WARPGROUPS) {
        if (warp_id == 4) {
            if (elect_warp_leader()) {
                int input_stage_id = 0;
                for (int tile_id = cluster_idx;
                     tile_id < cluster_tiles_per_device * fg::NUM_DEVICES;
                     tile_id += num_comp_clusters) {
                    // work should be partitioned based on the rank tile size. M = GLOBAL_M / TP
                    auto [local_row_id, tile_col_idx] = calculate_tile_idx(
                        cluster_rows_per_device, num_col_tiles, tile_id % cluster_tiles_per_device);

                    int target_device = tile_id / cluster_tiles_per_device;
                    load(local_row_id * fg::NUM_CLUSTERS + cta_rank,
                         tile_col_idx,
                         target_device,
                         input_stage_id);
                }
            }
        } else if (warp_id == 5) {
            if (cta_rank == 0 && elect_warp_leader()) {
                int input_stage_id = 0;
                int epilogue_stage_id = 0;

                typename fg::C_tt_tile tmem[fg::TMEM_PIPELINE_STAGES];
#pragma unroll
                for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
                    tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(i * fg::COL_BLOCK);
                }

                for (int tile_id = cluster_idx;
                     tile_id < cluster_tiles_per_device * fg::NUM_DEVICES;
                     tile_id += num_comp_clusters) {
                    MKERNEL_TIMING_ONLY(mma_tile_id = tile_id;)
                    consume(tmem, input_stage_id, epilogue_stage_id);
                }
            }
        }
    } else {
        int epilogue_stage_id = 0;
        int epilogue_transfer_stage_id = 0;
        typename fg::C_tt_tile tmem[fg::TMEM_PIPELINE_STAGES];

#pragma unroll
        for (int i = 0; i < fg::TMEM_PIPELINE_STAGES; i++) {
            tmem[i] = tm_alloc.template allocate<fg::C_tt_tile>(i * fg::COL_BLOCK);
        }

        for (int tile_id = cluster_idx; tile_id < cluster_tiles_per_device * fg::NUM_DEVICES;
             tile_id += num_comp_clusters) {
            // work should be partitioned based on the rank tile size. M = GLOBAL_M / TP
            auto [local_tile_row, tile_col_idx] = calculate_tile_idx(
                cluster_rows_per_device, num_col_tiles, tile_id % cluster_tiles_per_device);

            const int target_device =
                (tile_id / cluster_tiles_per_device + G.dev_idx) % fg::NUM_DEVICES;
            const int local_cta_row = local_tile_row * fg::NUM_CLUSTERS + cta_rank;
            epilogue(target_device * row_tiles_per_device + local_cta_row,
                     tile_col_idx,
                     tmem,
                     epilogue_stage_id,
                     epilogue_transfer_stage_id);
        }

        // wait for store to complete before deallocation of tmem
        if (warpgroup::laneid() == 0) {
            dist::tma::store_async_wait();
        }

        tensor_before_thread_sync();
        group<fg::EPILOGUE_WARPS>::sync(1);

        if (group<fg::EPILOGUE_WARPS>::warpid() == 0) {
            if (elect_warp_leader()) {
                tma::cluster::arrive(tmem_finished, 1 - cta_rank);
            }
            // Only reach here if we finish with our tmem. Other party as well
            wait(tmem_finished, 0);
            tm_alloc.deprovision();
        }
    }
}

template <int _ROW_BLOCK, int _COL_BLOCK>
__global__ __cluster_dims__(fused_globals<_ROW_BLOCK, _COL_BLOCK>::NUM_CLUSTERS, 1, 1)
    __launch_bounds__(fused_globals<_ROW_BLOCK, _COL_BLOCK>::NUM_THREADS, 1) void fused_kernel_stub(
        const __grid_constant__ fused_globals<_ROW_BLOCK, _COL_BLOCK> G) {
    ag_gemm_kda_mla<_ROW_BLOCK, _COL_BLOCK>(G);
}

template <int _ROW_BLOCK, int _COL_BLOCK>
inline void launch_ag_gemm_kda_mla(const fused_globals<_ROW_BLOCK, _COL_BLOCK>& G) {
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK>;
    ACopyPipelineState& copy_state = get_A_copy_state(G.dev_idx);

    copy_state.epoch++;
    if (copy_state.epoch == 0) {
        // Zero is reserved for the process-startup not-ready state.
        MKERNEL_CUDACHECK(cudaMemset(copy_state.ready, 0, fg::NUM_DEVICES * sizeof(uint32_t)));
        copy_state.epoch = 1;
    }

    fg launch_G = G;
    launch_G.A_copy_ready = copy_state.ready;
    launch_G.A_copy_epoch = copy_state.epoch;

    // Capture prior work on the caller's stream. On repeated invocations this
    // prevents the copy stream from overwriting A_local_buf until the previous
    // persistent kernel on the caller stream has finished consuming it.
    MKERNEL_CUDACHECK(cudaEventRecord(copy_state.main_pre_event, stream));
    MKERNEL_CUDACHECK(cudaStreamWaitEvent(copy_state.stream, copy_state.main_pre_event, 0));

    const size_t shard_elements = static_cast<size_t>(G.A.rows()) * G.K;
    const size_t shard_bytes = shard_elements * sizeof(typename fg::A_local_tensor::dtype);

    // Stage one complete shard per remote device in the same ring order used
    // by the persistent kernel. The local shard is read directly from G.A.
    for (int distance = 1; distance < fg::NUM_DEVICES; ++distance) {
        const int peer = (G.dev_idx + distance) % fg::NUM_DEVICES;
        auto* dst = G.A_local_buf.raw_ptr + static_cast<size_t>(peer) * shard_elements;
        const auto* src = G.A[peer].raw_ptr;

        MKERNEL_CUDACHECK(
            cudaMemcpyAsync(dst, src, shard_bytes, cudaMemcpyDeviceToDevice, copy_state.stream));

        // Keep the default pre-write barrier: it publishes the copied shard
        // before the completion epoch. The kernel-side load only needs GPU
        // scope because it reads a flag and payload resident on this device.
        MKERNEL_CUCHECK(cuStreamWriteValue32(reinterpret_cast<CUstream>(copy_state.stream),
                                             reinterpret_cast<CUdeviceptr>(copy_state.ready + peer),
                                             copy_state.epoch,
                                             CU_STREAM_WRITE_VALUE_DEFAULT));
    }

    constexpr int smem_size = fg::DYNAMIC_SHARED_MEMORY;
    constexpr int num_threads = fg::NUM_THREADS;
    constexpr int grid = fg::NUM_BLOCKS;

    auto this_kernel = fused_kernel_stub<_ROW_BLOCK, _COL_BLOCK>;

    MKERNEL_CUDACHECK(
        cudaFuncSetAttribute(this_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    this_kernel<<<grid, num_threads, smem_size, stream>>>(launch_G);
    MKERNEL_CUDACHECK(cudaGetLastError());
}
};  // namespace ag_gemm_kda_mla

#include "operators/ag_gemm/ag_gemm_kda_mla_session.cuh"
