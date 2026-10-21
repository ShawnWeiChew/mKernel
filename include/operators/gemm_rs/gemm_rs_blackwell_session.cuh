#pragma once

// Included after the kernel definitions in gemm_rs_blackwell.cu.
#include <limits>
#include <torch/csrc/utils/pybind.h>

namespace gemm_rs_blackwell {
using python_globals = fused_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>;

inline python_globals checked_python_globals(at::Tensor& A,
                                             at::Tensor& B,
                                             dist::ParallelBuffer& C,
                                             dist::ParallelBuffer& ready,
                                             dist::ParallelBuffer& epochs) {
    using fg = python_globals;
    const int rank = C.local_rank_;
    auto check_tensor = [rank](const at::Tensor& tensor, at::ScalarType dtype,
                               int ndim, const char* name) {
        TORCH_CHECK(tensor.is_cuda() && tensor.get_device() == rank,
                    name, " must be on the local CUDA device");
        TORCH_CHECK(tensor.scalar_type() == dtype && tensor.dim() == ndim &&
                        tensor.is_contiguous(),
                    name, " has the wrong dtype, dimensions, or strides");
    };
    for (auto* buffer : {&C, &ready, &epochs}) {
        TORCH_CHECK(buffer->local_world_size_ == fg::NUM_DEVICES &&
                        buffer->local_rank_ == rank,
                    "All buffers must use the compiled GPU count and the same local rank");
    }
    check_tensor(A, at::kBFloat16, 2, "A");
    check_tensor(B, at::kBFloat16, 2, "B");
    check_tensor(C.data_, at::kBFloat16, 2, "C");
    check_tensor(ready.data_, at::kInt, 2, "ready");
    check_tensor(epochs.data_, at::kInt, 1, "completion_epochs");
    const int64_t M = A.size(0), N = B.size(0), K = A.size(1);
    TORCH_CHECK(M > 0 && N > 0 && K > 0 &&
                    M <= std::numeric_limits<int>::max() &&
                    N <= std::numeric_limits<int>::max() &&
                    K <= std::numeric_limits<int>::max(), "Invalid GEMM dimensions");
    TORCH_CHECK(B.size(1) == K && C.data_.size(0) == M && C.data_.size(1) == N,
                "Expected A[M,K], B[N,K], and C[M,N]");
    TORCH_CHECK(M % (fg::NUM_DEVICES * fg::ROW_BLOCK) == 0 &&
                    M % (fg::NUM_CLUSTERS * fg::ROW_BLOCK) == 0 &&
                    N % fg::COL_BLOCK == 0 && K % fg::RED_BLOCK == 0,
                "M must contain whole cluster tiles and whole 256-row tiles per rank; "
                "N must be divisible by 256 and K by 64");
    TORCH_CHECK(ready.data_.size(0) == M / fg::NUM_DEVICES / fg::ROW_BLOCK &&
                    ready.data_.size(1) == N / fg::COL_BLOCK,
                "ready must have shape [M / world_size / 256, N / 256]");
    TORCH_CHECK(epochs.data_.numel() == 1, "completion_epochs must have shape [1]");
    TORCH_CHECK(C.multicast_ && ready.multicast_, "C and ready require multicast buffers");
    return gemm_rs_blackwell_make_globals<DEFAULT_NUM_EPILOGUE_WARPS, DEFAULT_C_TILE_DIVISOR>(
        A, B, C, ready, epochs, rank, M, N, K, at::cuda::getCurrentCUDAStream().stream());
}
}  // namespace gemm_rs_blackwell

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.attr("gemm_rs_blackwell_world_size") = INTRA_NUM_DEVICES;
    m.def("gemm_rs_blackwell_prepare",
          [](at::Tensor A, at::Tensor B, dist::ParallelBuffer& C,
             dist::ParallelBuffer& ready, dist::ParallelBuffer& completion_epochs) {
              c10::cuda::CUDAGuard guard(C.local_rank_);
              auto globals = gemm_rs_blackwell::checked_python_globals(
                  A, B, C, ready, completion_epochs);
              cudaStreamCaptureStatus status;
              MKERNEL_CUDACHECK(cudaStreamIsCapturing(globals.stream, &status));
              TORCH_CHECK(status == cudaStreamCaptureStatusNone,
                          "Prepare once before graph capture, with no launches in flight");
              gemm_rs_blackwell::initialize_gemm_rs_blackwell_state(globals);
          },
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"),
          pybind11::arg("ready"), pybind11::arg("completion_epochs"),
          "Initialize local counters once. Synchronize the stream and all ranks before launch.");
    m.def("gemm_rs_blackwell_launch",
          [](at::Tensor A, at::Tensor B, dist::ParallelBuffer& C,
             dist::ParallelBuffer& ready, dist::ParallelBuffer& completion_epochs) {
              c10::cuda::CUDAGuard guard(C.local_rank_);
              auto globals = gemm_rs_blackwell::checked_python_globals(
                  A, B, C, ready, completion_epochs);
              gemm_rs_blackwell::launch_gemm_rs_blackwell(globals);
          },
          pybind11::arg("A"), pybind11::arg("B"), pybind11::arg("C"),
          pybind11::arg("ready"), pybind11::arg("completion_epochs"),
          "Launch GEMM+RS and the exit barrier on the current stream. The result is "
          "C.data_[rank * M/world_size : (rank+1) * M/world_size]. Keep all buffers "
          "alive and serialize launches; graph replays advance the device epoch.");
}
