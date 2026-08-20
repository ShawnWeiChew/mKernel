/******************************************************************************
 * ThunderKittens bf16_b200 GEMM, wrapped behind a plain-C boundary.
 *
 * Why a separate translation unit
 * -------------------------------
 * mKernel vendors ThunderKittens: include/common/tk_types_*.cuh are full copies
 * of TK's types into `namespace kittens`, with their own #pragma once at paths
 * that differ from TK's originals. Including the upstream kittens.cuh alongside
 * them redefines every kittens:: type. So TK's kernel cannot share a TU with
 * gemm_ar_blackwell.cu -- but it can share a process. Nothing but pointers and
 * ints crosses the boundary below, so neither side sees the other's kittens.
 *
 * Compiled without -rdc, each .cu is a self-contained cubin, so the two device
 * images never merge. Host-side symbols are hidden (-fvisibility=hidden in the
 * Makefile rule) so only the extern "C" entry points below are exported.
 *
 * B layout: TK's b_gl is built as {ptr, nullptr, nullptr, N, K}, i.e. B is N x K.
 * Feed tk_gemm_create the TRANSPOSED B. This is the whole reason the standalone
 * harness keeps a Bt buffer around.
 *****************************************************************************/

#include <cuda_runtime.h>

#include <cstdio>
#include <functional>
#include <vector>

// Pull TK's headers in at global scope BEFORE renaming main, so the rename only
// touches the kernel file itself.
#include "kittens.cuh"
#include "kernels/gemm/common.cuh"

// bf16_b200_gemm.cu ships its own main() and its own benchmark loop. We want
// only `kernel<C>` and `globals<C>`; the rest is dead weight that still has to
// compile. Renaming main is the least invasive way to keep it out of the way.
#define main tk_bf16_b200_unused_main
#include "kernels/gemm/bf16_b200/bf16_b200_gemm.cu"
#undef main

using namespace kittens;

namespace {

// One entry per (shape, buffer group). Each closure owns a fully-built
// globals<C>, so the TMA descriptors are encoded once at create time rather
// than per launch -- cuTensorMapEncodeTiled is a host call and would otherwise
// land inside the timing window.
std::vector<std::function<void(cudaStream_t)>> g_launchers;
std::vector<const char*> g_names;

template <typename C>
int make_launcher(int M, int N, int K, const void* A, const void* Bt, void* D,
                  const char* name) {
    using G = globals<C>;
    typename G::a_gl Ag{(bf16*)A,  nullptr, nullptr, M, K};
    typename G::b_gl Bg{(bf16*)Bt, nullptr, nullptr, N, K};  // N x K -- transposed
    typename G::d_gl Dg{(bf16*)D,  nullptr, nullptr, M, N};
    G g{Ag, Bg, Dg};

    const int smem = g.dynamic_shared_memory();
    const dim3 grid = g.grid(), block = g.block();

    // Once per instantiation of C, not once per launch.
    static bool attr_set = false;
    if (!attr_set) {
        cudaError_t e = cudaFuncSetAttribute(
            kernel<C>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
        if (e != cudaSuccess) {
            std::fprintf(stderr, "tk shim: cudaFuncSetAttribute: %s\n",
                         cudaGetErrorString(e));
            return -1;
        }
        attr_set = true;
    }

    g_launchers.push_back([g, grid, block, smem](cudaStream_t s) {
        // NOTE: the 4th LaunchConfig argument is assumed to be the stream --
        // upstream passes a literal 0 there (bf16_b200_gemm.cu:315). If this
        // does not compile, that assumption is what to fix.
        LaunchConfig<true, true> lc(grid, block, smem, s, C::CLUSTER_SIZE);
        cudaLaunchKernelEx(lc, kernel<C>, g);
    });
    g_names.push_back(name);
    return (int)g_launchers.size() - 1;
}

}  // namespace

#define TK_EXPORT extern "C" __attribute__((visibility("default")))

// Returns a handle >= 0, or -1 if there is no tuned config for this shape.
//
// The configs are upstream's, keyed on N (bf16_b200_gemm.cu:368-378). They were
// autotuned at K == N. Every shape this harness runs uses K = N/4, which
// shortens the mainloop 4x and shifts the balance toward the epilogue, so these
// are NOT the optimal parameters here -- the resulting number is a floor on
// what TK's kernel can do, not its ceiling. Re-sweep before quoting it.
TK_EXPORT int tk_gemm_create(int M, int N, int K, const void* A, const void* Bt, void* D) {
    // Template parameters: Mb, Nb, Kb, SUPERGROUP_SIZE, OVERLAP_MMA_EPI,
    //                      LOAD_PIPE_DEPTH, EPI_PIPE_DEPTH
    switch (N) {
        case 1024:
            return make_launcher<config<256, 64, 128, 4, true, 5, 2>>(
                M, N, K, A, Bt, D, "cfg<256,64,128,4,T,5,2>");
        case 2048:
            return make_launcher<config<256, 256, 64, 8, true, 5, 4>>(
                M, N, K, A, Bt, D, "cfg<256,256,64,8,T,5,4>");
        case 4096:
            return make_launcher<config<256, 256, 64, 4, false, 4, 8>>(
                M, N, K, A, Bt, D, "cfg<256,256,64,4,F,4,8>");
        case 8192:
            return make_launcher<config<256, 256, 64, 8, false, 4, 8>>(
                M, N, K, A, Bt, D, "cfg<256,256,64,8,F,4,8>");
        case 16384:
            return make_launcher<config<256, 256, 64, 8, false, 4, 8>>(
                M, N, K, A, Bt, D, "cfg<256,256,64,8,F,4,8>");
        case 32768:  // no upstream config; reusing 16384's
            return make_launcher<config<256, 256, 64, 8, false, 4, 8>>(
                M, N, K, A, Bt, D, "cfg<256,256,64,8,F,4,8> (untuned)");
        default:
            return -1;
    }
}

TK_EXPORT void tk_gemm_launch(int handle, cudaStream_t s) { g_launchers[handle](s); }

TK_EXPORT const char* tk_gemm_name(int handle) { return g_names[handle]; }

// Handles hold device pointers owned by the caller. Drop them before the
// caller frees the buffers.
TK_EXPORT void tk_gemm_reset() {
    g_launchers.clear();
    g_names.clear();
}
