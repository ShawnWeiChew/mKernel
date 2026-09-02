/**
 * AG-GEMM but for KDA's proj_qkvgfab and MLA's qkvg proj. Putting it in a different file just in
 * case more operations have to be fused later, depending on how well communication is hidden
 */

#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_bf16.h>

#include <algorithm>
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

using namespace kittens;

namespace ag_gemm_mla_kda {};

#include "operators/ag_gemm/ag_gemm_kda_mla_session.cuh"