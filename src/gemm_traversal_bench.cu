// ctypes entry points for bench/gemm_traversal_bench.py. No PyTorch build dependency.
#ifndef MKERNEL_COMPILE_WITHOUT_TORCH
#define MKERNEL_COMPILE_WITHOUT_TORCH
#endif
#include <cublas_v2.h>

// Instantiate one kernel body twice; only its tile-coordinate mapper differs.
#include "ag_gemm_warp_specialized.cu"

namespace {
namespace ag = ag_gemm_warp_specialized;
using AgGlobals = ag::fused_globals<128, 256, 2, 2>;
static_assert(AgGlobals::CONSUMER_WARPS == 2 && AgGlobals::NUM_THREADS == 224);

template <int WIDTH>
void launch(const AgGlobals& globals, int traversal) {
    if (traversal == 0)
        ag::launch_ag_gemm_warp_specialized<128, 256, 2, WIDTH, 2,
            ag::AgStrategy::GEMM_ONLY, ag::TileTraversal::DEVICE_M_LOCAL_N>(globals);
    else
        ag::launch_ag_gemm_warp_specialized<128, 256, 2, WIDTH, 2,
            ag::AgStrategy::GEMM_ONLY, ag::TileTraversal::M_GLOBAL_N>(globals);
}

struct LocalMapping {
    comm::bf16* mc = nullptr;
    comm::bf16* pointers[INTRA_NUM_DEVICES] = {};
    comm::bf16** uc_ptrs = pointers;

    explicit LocalMapping(void* data) {
        for (auto& pointer : pointers) pointer = static_cast<comm::bf16*>(data);
    }
};

bool valid_shape(int m, int n, int k, int width) {
    return m >= 512 && n >= 256 && k >= 64 && m % 512 == 0 &&
        m % INTRA_NUM_DEVICES == 0 && (m / INTRA_NUM_DEVICES) % 128 == 0 &&
        n % 256 == 0 && k % 64 == 0 && (width == 4 || width == 8) &&
        (n / 256) % width == 0;
}
}  // namespace

extern "C" int gemm_bench_logical_devices() { return INTRA_NUM_DEVICES; }

// A=[M,K], B=[N,K], C=[M,N], all contiguous BF16 on the current GPU.
// Every logical shard points into the same fully initialized local A allocation.
extern "C" int gemm_bench_launch(void* a, void* b, void* c, int m, int n, int k,
                                  int width, int rank, int traversal, void* stream) {
    if (!a || !b || !c || !valid_shape(m, n, k, width) ||
        rank < 0 || rank >= INTRA_NUM_DEVICES || traversal < 0 || traversal > 1)
        return cudaErrorInvalidValue;
    LocalMapping a_mapping(a), unused_ready(nullptr);
    auto* b_ptr = static_cast<comm::bf16*>(b);
    auto* c_ptr = static_cast<comm::bf16*>(c);
    auto globals = ag::ag_gemm_warp_specialized_make_globals<128, 256, 2, 2>(
        a_mapping, unused_ready, b_ptr, c_ptr, rank, m, n, k,
        static_cast<cudaStream_t>(stream));
    // Both variants receive exactly the same descriptors and launch parameters.
    if (width == 4)
        launch<4>(globals, traversal);
    else
        launch<8>(globals, traversal);
    return cudaGetLastError();
}

extern "C" int gemm_bench_cublas_create(void** handle) {
    auto status = cublasCreate(reinterpret_cast<cublasHandle_t*>(handle));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    status = cublasSetMathMode(static_cast<cublasHandle_t>(*handle),
                              CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    if (status != CUBLAS_STATUS_SUCCESS) {
        cublasDestroy(static_cast<cublasHandle_t>(*handle));
        *handle = nullptr;
    }
    return status;
}

extern "C" int gemm_bench_cublas_destroy(void* handle) {
    return cublasDestroy(static_cast<cublasHandle_t>(handle));
}

extern "C" int gemm_bench_cublas(void* handle, void* a, void* b, void* c,
                                  int m, int n, int k, void* stream) {
    auto h = static_cast<cublasHandle_t>(handle);
    auto status = cublasSetStream(h, static_cast<cudaStream_t>(stream));
    if (status != CUBLAS_STATUS_SUCCESS) return status;
    const float alpha = 1.0f, beta = 0.0f;
    // C^T = B A^T. Row-major B[N,K] looks column-major [K,N], so transpose it.
    return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N,
                        n, m, k, &alpha, b, CUDA_R_16BF, k,
                        a, CUDA_R_16BF, k, &beta, c, CUDA_R_16BF, n,
                        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
