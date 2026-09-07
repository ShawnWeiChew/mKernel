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
#include "common/timings.cuh"
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

namespace ag_gemm_kda_mla {

template <int _ROW_BLOCK, int _COL_BLOCK>
struct fused_globals;

template <int _ROW_BLOCK, int _COL_BLOCK>
void launch_ag_gemm_kda_mla(const fused_globals<_ROW_BLOCK, _COL_BLOCK>& G);

static constexpr int DEFAULT_ROW_BLOCK = 128;
static constexpr int DEFAULT_COL_BLOCK = 128;

#ifdef PROFILE_TIMINGS
// Append-only: these values are the ABI of every saved .npz. Renumbering makes
// historical traces decode as the wrong phases.
//
// Unlike gemm_ar_blackwell there is no comm/comp SM split here -- every CTA is
// a compute CTA. The all-gather is staged by the copy engine into A_local_buf,
// so MMA_INPUTS_LOCAL/REMOTE now distinguish which *shard* a tile came from
// (this device's own, or a peer's staged copy) rather than which device it was
// read from: both are local HBM reads by the time the MMA sees them. The names
// are kept as-is because they are the ABI of every trace saved so far.
enum TimingEvent : uint32_t {
    // Shared prologue (semaphore init, TMEM provisioning, cluster sync).
    EV_SETUP_BEGIN = 0,
    EV_SETUP_DONE = 1,

    // Producer warp: one pipeline step per K block.
    EV_LOAD_STEP_BEGIN = 2,  // top of the step, before waiting on the ring slot
    EV_LOAD_MMA_FREE = 3,    // mma_finish observed: the stage is free to refill
    EV_LOAD_TMA_ISSUED = 4,  // A (possibly remote) + B loads issued

    // Consumer (MMA) warp.
    EV_MMA_TILE_BEGIN = 5,      // top of an output tile
    EV_MMA_TMEM_FREE = 6,       // epilogue released this TMEM stage
    EV_MMA_STEP_BEGIN = 7,      // top of a K step
    EV_MMA_INPUTS_LOCAL = 8,    // tma_load observed, A is this device's own shard
    EV_MMA_INPUTS_REMOTE = 9,   // tma_load observed, A is a peer's staged shard
    EV_MMA_ISSUED = 10,         // mm2/mma2 issued

    // Epilogue warpgroup.
    EV_EPI_TILE_BEGIN = 11,     // top of an output tile
    EV_EPI_MMA_DONE = 12,       // epilogue_ready observed: the mainloop is done
    EV_EPI_TMEM_READ = 13,      // accumulator drained TMEM -> registers
    EV_EPI_SMEM_WRITTEN = 14,   // all chunks staged to SMEM and stores issued

    // Copy engine handoff. A cudaMemcpyAsync executes no instrumentable device
    // code, so it cannot stamp itself -- but the producer blocks on the flag the
    // copy stream writes, and that spin *is* on %globaltimer like everything
    // else. Emitted once per (CTA, source device): the first wait is the real
    // one, later tiles from the same peer find the flag already set.
    EV_ACOPY_WAIT_BEGIN = 15,   // about to spin on A_copy_ready[peer]
    EV_ACOPY_READY = 16,        // peer's shard is visible in A_local_buf
};

// Kept beside the enum so the pybind export and the enum cannot diverge.
#define AG_GEMM_KDA_MLA_TIMING_EVENTS(X) \
    X(SETUP_BEGIN)                       \
    X(SETUP_DONE)                        \
    X(LOAD_STEP_BEGIN)                   \
    X(LOAD_MMA_FREE)                     \
    X(LOAD_TMA_ISSUED)                   \
    X(MMA_TILE_BEGIN)                    \
    X(MMA_TMEM_FREE)                     \
    X(MMA_STEP_BEGIN)                    \
    X(MMA_INPUTS_LOCAL)                  \
    X(MMA_INPUTS_REMOTE)                 \
    X(MMA_ISSUED)                        \
    X(EPI_TILE_BEGIN)                    \
    X(EPI_MMA_DONE)                      \
    X(EPI_TMEM_READ)                     \
    X(EPI_SMEM_WRITTEN)                  \
    X(ACOPY_WAIT_BEGIN)                  \
    X(ACOPY_READY)
#endif  // PROFILE_TIMINGS

// for M < 512, this should be 128
template <int _ROW_BLOCK, int _COL_BLOCK>
struct fused_globals {
    // config items
    static constexpr int NUM_DEVICES = INTRA_NUM_DEVICES;

    // not sure if I want to use a warp specialized or sm specialized strategy yet
    static constexpr int NUM_BLOCKS = 148;
    static constexpr int CONSUMER_WARPS = 1;
    static constexpr int PRODUCER_WARPS = 1;
    static constexpr int EPILOGUE_WARPGROUPS = 1;
    static constexpr int EPILOGUE_WARPS = EPILOGUE_WARPGROUPS * kittens::WARPGROUP_WARPS;
    static constexpr int NUM_CLUSTERS =
        2;  // will try to use 2-CTA as much as possible for instructions that cleanly divide it
    static constexpr int NUM_THREADS = (CONSUMER_WARPS + PRODUCER_WARPS + EPILOGUE_WARPS) * 32;

    // this is pipelining along the reduction dimension
    static constexpr int PRODUCER_CONSUMER_PIPELINE_STAGES = _COL_BLOCK == 128 ? 7 : 5;
    // this is pipelining among different MMAs
    static constexpr int TMEM_PIPELINE_STAGES = kittens::MAX_TENSOR_COLS / _COL_BLOCK;
    // this is the number of epilogue stages that can be in flight at any time
    static constexpr int EPILOGUE_PIPELINE_STAGES = 3;
    // this is the number of partitions for the epilogue tile in SMEM
    static constexpr int C_TILE_DIVISOR = _COL_BLOCK == 128 ? 2 : 4;

    // NOTE: based on PK paper, To sustain over 80% bandwidth utilization, the transfer granularity
    // must be at least 256 MB when using the copy engine, whereas device-side methods (TMA) achieve
    // comparable utilization with only 2 KB. The vllm / cutlass one uses copy engine, but even the
    // largest tile size with TP = 8 is only 4096 * 64 * 2 = 500 KB

    // TODO: load, in a ring like fashion, the data necessary from the target GMEM into SMEM
    // for correctness first, we can just load from the same peer every time

    // NOTE: potential problem with this is that the load is never cached in local L2, which may be
    // why it is not used if that is teh case, then fine grained cudaMemcpyAsync, which will be
    // dispatched to run on a separate stream will be better
    static constexpr int ROW_BLOCK = _ROW_BLOCK;
    static constexpr int COL_BLOCK = _COL_BLOCK;
    static constexpr int RED_BLOCK = 64;

    using A_tile = kittens::st_bf<ROW_BLOCK, RED_BLOCK>;
    using B_tile = kittens::st_bf<RED_BLOCK, COL_BLOCK / NUM_CLUSTERS>;

    using C_tt_tile = kittens::tt<float, ROW_BLOCK, COL_BLOCK>;
    // for smem staging -- keep at least
    using C_tile = kittens::st_bf<ROW_BLOCK, COL_BLOCK / C_TILE_DIVISOR>;

    static constexpr int DYNAMIC_SHARED_MEMORY =
        (sizeof(A_tile) + sizeof(B_tile)) * PRODUCER_CONSUMER_PIPELINE_STAGES +
        sizeof(C_tile) * EPILOGUE_PIPELINE_STAGES + 1024;
    static_assert(DYNAMIC_SHARED_MEMORY <= 227 * 1024, "SMEM allocation too large");

    using A_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, A_tile>;
    using A_distributed_tensor = dist::distributed_tensor<A_local_tensor, NUM_DEVICES, true>;
    using B_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, B_tile>;
    using C_local_tensor = dist::local_tensor<comm::bf16, 1, 1, -1, -1, C_tile>;

    A_distributed_tensor A;
    A_local_tensor A_local_buf;
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

#ifdef PROFILE_TIMINGS
    // NUM_BLOCKS * EVENTS_PER_BLOCK records, partitioned by blockIdx.x so CTAs
    // never contend. Null on an unprofiled launch, which short-circuits the
    // store in emit_timing_impl.
    ::mkernel_timings::TimingRecord* timings;
#endif

    struct pipeline_inputs {
        A_tile A;
        B_tile B;
    };

    struct pipeline_outputs {
        C_tile C;
    };

    /*
     * bit 0: TMA producer -- starts with 1 (PRODUCER WARP)
     * bit 1: TMA consumer -- starts with 0 (CONSUMER WARP)
     * bit 2: TMEM producer -- starts with 1 (CONSUMER WARP)
     * bit 3: TMEM consumer -- starts with 0 (EPILOGUE WARP)
     */
    static constexpr int TMA_PRODUCER_BIT = 0b1;
    static constexpr int TMA_CONSUMER_BIT = 0b00;
    static constexpr int TMEM_PROUCER_BIT = 0b100;
    static constexpr int TMEM_CONSUMER_BIT = 0b0000;
    static constexpr int PHASE_BITS_INIT =
        TMA_PRODUCER_BIT | TMA_CONSUMER_BIT | TMEM_PROUCER_BIT | TMEM_CONSUMER_BIT;
};

template <int _ROW_BLOCK, int _COL_BLOCK>
__host__ inline fused_globals<_ROW_BLOCK, _COL_BLOCK> ag_gemm_kda_mla_make_globals(
    dist::ParallelBuffer& A,
    const at::Tensor& A_local_buf,
    const at::Tensor& B,
    at::Tensor& C,
    int dev_idx,
    int M,
    int N,
    uint64_t timings_ptr) {
    using fg = fused_globals<_ROW_BLOCK, _COL_BLOCK>;
    (void)timings_ptr;

    return {
        .A = ::dist::distributed_tensor_from_buffer<typename fg::A_distributed_tensor>(A),
        .A_local_buf = ::dist::local_tensor_from_tensor<typename fg::A_local_tensor>(A_local_buf),
        .B = ::dist::local_tensor_from_tensor<typename fg::B_local_tensor>(B),
        .C = ::dist::local_tensor_from_tensor<typename fg::C_local_tensor>(C),
        .A_copy_ready = nullptr,
        .A_copy_epoch = 0,
        .dev_idx = dev_idx,
        .M = M,
        .N = N,
#ifdef PROFILE_TIMINGS
        .timings = reinterpret_cast<::mkernel_timings::TimingRecord*>(timings_ptr),
#endif
    };
}

#ifdef PROFILE_TIMINGS
// (peer, start_ms, end_ms) for each staged shard of the most recent launch,
// measured on the copy stream and expressed relative to a reference event taken
// just before the first copy. Widths are exact; the caller places them on the
// %globaltimer axis by anchoring against the kernel's own ACOPY_READY records.
//
// Both events must have completed, so synchronize before calling.
std::vector<std::tuple<int, float, float>> ag_gemm_kda_mla_copy_times(int dev_idx);
#endif

// COL_BLOCK is picked here, and the profiler needs it to size its expectations;
// keep the threshold in one place so Python can ask rather than guess.
__host__ inline int ag_gemm_kda_mla_col_block(int M) { return M <= 512 ? 128 : 256; }

void entrypoint(dist::ParallelBuffer& A,
                const at::Tensor& A_local_buf,
                const at::Tensor& B,
                at::Tensor& C,
                // Device pointer to a ring of TimingRecords, or 0. Ignored
                // unless built with -DPROFILE_TIMINGS, so the signature is the
                // same either way.
                const uint64_t timings_ptr = 0) {
    const int dev_idx = A.local_rank_;
    c10::cuda::CUDAGuard device_guard(dev_idx);

    const int M = C.size(0), N = B.size(1);
    constexpr int K = fused_globals<128, 128>::K;

    TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES,
                "A.local_world_size must match the compiled INTRA_NUM_DEVICES");
    TORCH_CHECK(A.data_.dim() == 2, "A must be a 2D tensor");
    TORCH_CHECK(A.data_.scalar_type() == at::kBFloat16, "A must be bfloat16");
    TORCH_CHECK(A.data_.size(1) == K, "A's K dimension must be ", K);
    TORCH_CHECK(A_local_buf.is_cuda() && A_local_buf.is_contiguous(),
                "A_local_buf must be a contiguous CUDA tensor");
    TORCH_CHECK(A_local_buf.device().index() == dev_idx,
                "A_local_buf must be on A's local device");
    TORCH_CHECK(A_local_buf.scalar_type() == at::kBFloat16,
                "A_local_buf must be bfloat16");
    TORCH_CHECK(A_local_buf.dim() == 2 && A_local_buf.size(0) == M &&
                    A_local_buf.size(1) == A.data_.size(1),
                "A_local_buf must have shape [global_M, K] = [",
                M,
                ", ",
                A.data_.size(1),
                "]");
    TORCH_CHECK(M == A.data_.size(0) * A.local_world_size_,
                "C's M dimension must equal A.local_M * world_size");

    if (ag_gemm_kda_mla_col_block(M) == 128) {
        using fg = fused_globals<128, 128>;
        fg globals = ag_gemm_kda_mla_make_globals<128, 128>(
            A, A_local_buf, B, C, dev_idx, M, N, timings_ptr);
        launch_ag_gemm_kda_mla<128, 128>(globals);
    } else {
        using fg = fused_globals<128, 256>;
        fg globals = ag_gemm_kda_mla_make_globals<128, 256>(
            A, A_local_buf, B, C, dev_idx, M, N, timings_ptr);
        launch_ag_gemm_kda_mla<128, 256>(globals);
    }
}
};  // namespace ag_gemm_kda_mla
