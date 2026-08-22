#pragma once
/******************************************************************************
 * Experimental copy of the Blackwell GEMM compute path.
 *
 * This exists to isolate two variables against the production kernel in
 * gemm_ar_blackwell.{cuh,cu}, which it otherwise mirrors exactly (same warp
 * layout, same pipeline depths, same register split, same tile walk):
 *
 *   1. B layout. The production kernel consumes B as K x N and issues mm2_AB.
 *      This one consumes B as N x K and issues mm2_ABt, which is the operand
 *      layout TK's bf16_b200 kernel uses. Both compute the same D; the question
 *      is only what the TMA and the MMA descriptor prefer.
 *
 *   2. Output buffer. The production kernel stores through
 *      C_dist[dev_idx] -- a distributed_tensor, i.e. a runtime index into an
 *      array of per-device descriptors living in the grid-constant bank. This
 *      one stores to a single plain global tensor, the way a torch tensor would
 *      arrive.
 *
 * Everything else is deliberately identical, so a delta measured against the
 * production kernel is attributable to those two changes and nothing else.
 * There is no all-reduce, no barrier, and no profiler here -- this is a
 * single-GPU measurement harness, not a shippable kernel.
 *****************************************************************************/

#include <ATen/ATen.h>

#include "comm/comm.cuh"
#include "common/tk_common_util.cuh"
#include "common/tk_types_shared_st.cuh"
#include "common/types.cuh"
#include "dist/local_tensor.cuh"
#include "memory/tk_ops_group_group.cuh"
#include "operators/gemm_ar/gemm_ar_blackwell.cuh"

namespace gemm_bf16_test {

// Reuse the production schedule verbatim -- warp roles, register split and the
// warpgroup arithmetic all have to match or the comparison means nothing.
using config = gemm_ar_intranode_blackwell::config;

struct test_globals {
    using prod = gemm_ar_intranode_blackwell::fused_globals;

    static constexpr int PIPELINE_STAGES = prod::PIPELINE_STAGES;
    static constexpr int EPILOGUE_STAGES = prod::EPILOGUE_STAGES;
    static constexpr int NUM_C_TILES = prod::NUM_C_TILES;
    static constexpr int ROW_BLOCK = prod::ROW_BLOCK;
    static constexpr int COL_BLOCK = prod::COL_BLOCK;
    static constexpr int RED_BLOCK = prod::RED_BLOCK;

    using A_tile = kittens::st_bf<ROW_BLOCK / config::CONSUMER_WARPS, RED_BLOCK>;

    // THE CHANGE: B is N x K, so a tile is (N per CTA) x K rather than
    // K x (N per CTA). mma_ABt takes trans_b = 0, which asserts
    // N == B::rows * ncta and K == B::cols -- the mirror of the AB form.
    static_assert(COL_BLOCK % config::NUM_CLUSTERS == 0, "COL_BLOCK should be divisible");
    using B_tile = kittens::st_bf<COL_BLOCK / config::NUM_CLUSTERS, RED_BLOCK>;

    using C_tt_tile = kittens::tt<float, ROW_BLOCK / config::CONSUMER_WARPS, COL_BLOCK>;
    static_assert(config::CONSUMER_WARPS * C_tt_tile::cols <= kittens::MAX_TENSOR_COLS,
                  "The TMEM accumulators for all consumers must fit in tensor memory");

    static_assert(COL_BLOCK % EPILOGUE_STAGES == 0, "COL_BLOCK should be divisible");
    using C_tile = kittens::st_bf<ROW_BLOCK / config::CONSUMER_WARPS, COL_BLOCK / EPILOGUE_STAGES>;

    static constexpr int DYNAMIC_SHARED_MEMORY =
        ((sizeof(A_tile) * config::CONSUMER_WARPS + sizeof(B_tile)) * PIPELINE_STAGES) +
        (sizeof(C_tile) * NUM_C_TILES * config::CONSUMER_WARPS) +
        1024;  // NOTE: must add 1024 so this can be aligned by TK
    static_assert(DYNAMIC_SHARED_MEMORY <= 227 * 1024, "SMEM allocation too large");

    // THE OTHER CHANGE: all three operands are plain global tensors. C in
    // particular is a single descriptor rather than distributed_tensor indexed
    // by dev_idx at runtime.
    using A_gl = dist::gl<comm::bf16, 1, 1, -1, -1, A_tile>;  // M x K
    using B_gl = dist::gl<comm::bf16, 1, 1, -1, -1, B_tile>;  // N x K
    using C_gl = dist::gl<comm::bf16, 1, 1, -1, -1, C_tile>;  // M x N

    A_gl A;
    B_gl B;
    C_gl C;

    int M;
    int N;
    int K;

    struct pipeline_inputs {
        A_tile A[config::CONSUMER_WARPS];
        B_tile B;
    };
};

template <int SUPERGROUP_WIDTH>
void launch_gemm_bf16_test(const test_globals& G, cudaStream_t stream);

}  // namespace gemm_bf16_test
