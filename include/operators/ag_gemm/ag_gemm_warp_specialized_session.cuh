#pragma once

#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>

#include "dist/parallel_buffer.cuh"
#include "pybind11/cast.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    BIND_DIST_PARALLEL_BUFFER(m);
    m.def(
        "ag_gemm_warp_specialized",
        [](dist::ParallelBuffer& A,
           dist::ParallelBuffer& A_copy_ready,
           const at::Tensor& B,
           at::Tensor& C,
           int logical_global_m) {
            const int dev_idx = A.local_rank_;
            c10::cuda::CUDAGuard device_guard(dev_idx);
            TORCH_CHECK(A.local_world_size_ == INTRA_NUM_DEVICES &&
                            A_copy_ready.local_world_size_ == INTRA_NUM_DEVICES &&
                            A_copy_ready.local_rank_ == dev_idx,
                        "A and A_copy_ready must match the compiled device group");
            TORCH_CHECK(A.data_.dim() == 3 && A.data_.size(0) == INTRA_NUM_DEVICES &&
                            A.data_.scalar_type() == at::kBFloat16 && A.data_.is_contiguous(),
                        "A must be contiguous bf16 [devices, local_rows, K]");
            TORCH_CHECK(A_copy_ready.data_.scalar_type() == at::kInt &&
                            A_copy_ready.data_.is_contiguous() &&
                            A_copy_ready.data_.numel() >= INTRA_NUM_DEVICES,
                        "A_copy_ready must contain at least one int32 flag per device");
            TORCH_CHECK(B.is_cuda() && C.is_cuda() && B.get_device() == dev_idx &&
                            C.get_device() == dev_idx && B.is_contiguous() && C.is_contiguous() &&
                            B.scalar_type() == at::kBFloat16 && C.scalar_type() == at::kBFloat16,
                        "B and C must be contiguous bf16 tensors on A's device");
            TORCH_CHECK(B.dim() == 2 && B.size(1) == A.data_.size(2) && B.size(1) > 0 &&
                            B.size(1) % 64 == 0 && B.size(0) > 0 && B.size(0) % 8 == 0,
                        "B must be [N, K], with K divisible by 64 and N divisible by 8");
            TORCH_CHECK(C.dim() == 3 && C.size(0) == INTRA_NUM_DEVICES &&
                            C.size(1) == A.data_.size(1) && C.size(2) == B.size(0),
                        "C must be [devices, local_rows, N]");
            const int M = C.size(0) * C.size(1);
            TORCH_CHECK(logical_global_m > 0 && logical_global_m <= M &&
                            logical_global_m % INTRA_NUM_DEVICES == 0,
                        "logical_global_m must fit A and divide evenly across devices");
            ag_gemm_warp_specialized::
                entrypoint<dist::ParallelBuffer, dist::ParallelBuffer, at::Tensor>(
                    A,
                    A_copy_ready,
                    B,
                    C,
                    M,
                    B.size(0),
                    B.size(1),
                    dev_idx,
                    at::cuda::getCurrentCUDAStream().stream(),
                    logical_global_m);
        },
        pybind11::arg("A"),
        pybind11::arg("A_copy_ready"),
        pybind11::arg("B"),
        pybind11::arg("C"),
        pybind11::arg("logical_global_m"));
}
