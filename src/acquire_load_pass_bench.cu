#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>

#include <cstdint>

#include "comm/atomic_u32.cuh"

namespace {

__device__ __forceinline__ uint64_t globaltimer_ns() {
    uint64_t value;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value) :: "memory");
    return value;
}

// Measure one already-ready pass through the same wait used by
// ag_gemm_kda_mla. One full warp loads the same flag, like the producer warp in
// the real kernel; lane 0 records the result. Column 0 is the cost of two adjacent timer reads;
// column 1 contains the same timer reads around the acquire-load wait.
__global__ void measure_ready_pass_kernel(const int32_t* ready,
                                          uint32_t epoch,
                                          int64_t* samples,
                                          int num_samples) {
    // Warm the flag into cache before measuring the already-ready case. The
    // real producer also polls the same address, so its successful load is hot.
    (void)comm::atomic_u32::acquire_load_gpu(ready);

    for (int i = 0; i < num_samples; ++i) {
        uint64_t begin = globaltimer_ns();
        uint64_t end = globaltimer_ns();
        if (threadIdx.x == 0) {
            samples[2 * i] = static_cast<int64_t>(end - begin);
        }

        begin = globaltimer_ns();
        while (comm::atomic_u32::acquire_load_gpu(ready) < epoch) {
            __nanosleep(64);
        }
        end = globaltimer_ns();
        if (threadIdx.x == 0) {
            samples[2 * i + 1] = static_cast<int64_t>(end - begin);
        }
    }
}

torch::Tensor measure_ready_pass(torch::Tensor ready, int64_t epoch, int64_t num_samples) {
    TORCH_CHECK(ready.is_cuda(), "ready must be a CUDA tensor");
    TORCH_CHECK(ready.scalar_type() == torch::kInt32, "ready must have dtype torch.int32");
    TORCH_CHECK(ready.numel() == 1, "ready must contain exactly one value");
    TORCH_CHECK(ready.is_contiguous(), "ready must be contiguous");
    TORCH_CHECK(epoch >= 0 && epoch <= UINT32_MAX, "epoch must fit in uint32_t");
    TORCH_CHECK(num_samples > 0 && num_samples <= INT32_MAX,
                "num_samples must be in [1, INT32_MAX]");

    c10::cuda::CUDAGuard device_guard(ready.device());
    auto output = torch::empty(
        {num_samples, 2}, ready.options().dtype(torch::kInt64));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    measure_ready_pass_kernel<<<1, 32, 0, stream>>>(
        ready.data_ptr<int32_t>(),
        static_cast<uint32_t>(epoch),
        output.data_ptr<int64_t>(),
        static_cast<int>(num_samples));
    const cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "measure_ready_pass_kernel launch failed: ", cudaGetErrorString(error));
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("measure_ready_pass", &measure_ready_pass,
               pybind11::arg("ready"), pybind11::arg("epoch"),
               pybind11::arg("num_samples"));
}
