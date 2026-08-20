/******************************************************************************
 * Standalone single-GPU GEMM comparison: mKernel vs cuBLAS.
 *
 * Why this exists
 * ---------------
 * bench/gemm_ar_blackwell_bench.py measures the fused kernel with a discipline
 * suited to a collective: host barrier + full sync before every iteration, one
 * event pair per iteration, median across iterations. ThunderKittens' GEMM
 * benches measure the opposite way: one event pair around N back-to-back
 * launches, divided by N, with buffer rotation sized to defeat L2. The two
 * numbers are not comparable, so "we are at 0.8x of TK" was never a claim the
 * python bench could support.
 *
 * On the check-just-gemm branch the fused kernel is effectively single-GPU:
 * config::NUM_COMP_SM == NUM_BLOCKS, so fused_intranode_sm never runs. That
 * makes it possible to measure it under TK's exact discipline, in one process,
 * against cuBLAS, on identical buffers. That is what this file does.
 *
 * It includes src/gemm_ar_blackwell.cu directly rather than copying the kernel,
 * so it cannot drift from the thing being benchmarked.
 *
 * The two cuBLAS variants are the point of the B-layout question
 * -------------------------------------------------------------
 *   cublas-NN : B stored K x N. Same layout the mKernel kernel consumes.
 *   cublas-TN : B stored N x K. The layout TK's bf16_b200 kernel consumes, and
 *               the one its cuBLAS baseline is fed.
 * Both compute the same D. The gap between them is the cost of the layout
 * choice alone, which is the number needed before comparing against any TK
 * figure -- a TK result is measured in TN, this kernel runs NN.
 *
 * Build:  make gemm_blackwell_standalone
 * Run:    ./build/gemm_blackwell_standalone
 * Env:    BENCH_WARMUP (default 30), BENCH_ITERS (default 30)
 *****************************************************************************/

#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// The real kernel. stub_include/ must precede include/ on the -I line.
#include "../../src/gemm_ar_blackwell.cu"

namespace gab = gemm_ar_intranode_blackwell;
using bf16 = __nv_bfloat16;

#ifdef WITH_TK
// Implemented in tk_gemm_shim.cu, which is a separate translation unit because
// mKernel vendors ThunderKittens into namespace kittens and the upstream
// kittens.cuh would redefine all of it. Only pointers and ints cross this line.
extern "C" {
int         tk_gemm_create(int M, int N, int K, const void* A, const void* Bt, void* D);
void        tk_gemm_launch(int handle, cudaStream_t s);
const char* tk_gemm_name(int handle);
void        tk_gemm_reset();
}
#endif

#define CUDA_OK(call)                                                                      \
    do {                                                                                   \
        cudaError_t e_ = (call);                                                           \
        if (e_ != cudaSuccess) {                                                           \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__,                   \
                         cudaGetErrorString(e_));                                          \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

#define CUBLAS_OK(call)                                                                    \
    do {                                                                                   \
        cublasStatus_t s_ = (call);                                                        \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                 \
            std::fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
            std::exit(1);                                                                  \
        }                                                                                  \
    } while (0)

static void sleep_ms(int ms) { std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Cheap hash-based uniform [-1, 1]. Matches TK's fill<RANDOM> in distribution,
// not in bit pattern -- the reference is computed from the same buffers, so
// only the distribution matters.
__global__ void fill_random(bf16* p, size_t n, uint64_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 30; x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27; x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    float u = (float)(uint32_t)(x >> 32) * (1.0f / 4294967296.0f);  // [0,1)
    p[i] = __float2bfloat16(u * 2.0f - 1.0f);
}

// Bt[n*K + k] = B[k*N + n]
__global__ void transpose_kn_to_nk(const bf16* __restrict__ B, bf16* __restrict__ Bt,
                                   int K, int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && k < K) Bt[(size_t)n * K + k] = B[(size_t)k * N + n];
}

__global__ void diff_stats(const bf16* obs, const bf16* ref, size_t n,
                           unsigned* max_abs_bits, double* sum_abs, double* sum_ref) {
    __shared__ float s_max[256];
    __shared__ double s_abs[256], s_ref[256];
    int t = threadIdx.x;
    float m = 0.0f; double a = 0.0, r = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + t; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float o = __bfloat162float(obs[i]);
        float e = __bfloat162float(ref[i]);
        float d = fabsf(o - e);
        m = fmaxf(m, d);
        a += d;
        r += fabsf(e);
    }
    s_max[t] = m; s_abs[t] = a; s_ref[t] = r;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (t < s) {
            s_max[t] = fmaxf(s_max[t], s_max[t + s]);
            s_abs[t] += s_abs[t + s];
            s_ref[t] += s_ref[t + s];
        }
        __syncthreads();
    }
    if (t == 0) {
        // Values are non-negative, so the IEEE bit pattern orders like the float.
        atomicMax(max_abs_bits, __float_as_uint(s_max[0]));
        atomicAdd(sum_abs, s_abs[0]);
        atomicAdd(sum_ref, s_ref[0]);
    }
}

struct CheckResult { float max_abs; double mean_abs; double ref_mean; bool ok; };

static CheckResult check(const bf16* obs, const bf16* ref, size_t n) {
    unsigned* d_max; double *d_abs, *d_ref;
    CUDA_OK(cudaMalloc(&d_max, sizeof(unsigned)));
    CUDA_OK(cudaMalloc(&d_abs, sizeof(double)));
    CUDA_OK(cudaMalloc(&d_ref, sizeof(double)));
    CUDA_OK(cudaMemset(d_max, 0, sizeof(unsigned)));
    CUDA_OK(cudaMemset(d_abs, 0, sizeof(double)));
    CUDA_OK(cudaMemset(d_ref, 0, sizeof(double)));

    diff_stats<<<1024, 256>>>(obs, ref, n, d_max, d_abs, d_ref);
    CUDA_OK(cudaDeviceSynchronize());

    unsigned mb; double sa, sr;
    CUDA_OK(cudaMemcpy(&mb, d_max, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&sa, d_abs, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&sr, d_ref, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_max)); CUDA_OK(cudaFree(d_abs)); CUDA_OK(cudaFree(d_ref));

    CheckResult c;
    std::memcpy(&c.max_abs, &mb, sizeof(float));  // __uint_as_float is device-only
    c.mean_abs = sa / (double)n;
    c.ref_mean = sr / (double)n;
    // bf16 accumulation order differs between cuBLAS and the kernel; judge on
    // bulk error the way bench/common.py's check_close does, not on outliers.
    c.ok = c.mean_abs <= 0.01 * c.ref_mean;
    return c;
}

// ---------------------------------------------------------------------------
// Building fused_globals without a DistBuffer
// ---------------------------------------------------------------------------
//
// Every distributed_tensor slot is pointed at the same single-GPU allocation.
// That is sound here only because fused_intranode_sm never runs: the comm path
// is the sole consumer of mc_ptr and of the peer entries in gls[]. The compute
// path touches G.A, G.B, and G.C_dist[G.dev_idx] only.
//
// The epilogue still issues a dist::signal per tile into comp_comm_barrier
// (gemm_ar_blackwell.cu:213-216). Nothing consumes it with zero comm SMs, and
// here it degenerates to a local atomic add, but it is not free -- see the note
// printed at the end of main().
static gab::fused_globals make_globals(bf16* dA, bf16* dB, bf16* dC, int* dBar,
                                       int M, int N, int K) {
    using FG = gab::fused_globals;
    constexpr int ND = gab::config::NUM_DEVICES;

    bf16* cptrs[ND];
    int*  bptrs[ND];
    for (int i = 0; i < ND; ++i) { cptrs[i] = dC; bptrs[i] = dBar; }

    const size_t br = (size_t)(M / FG::ROW_BLOCK);
    const size_t bc = (size_t)(N / FG::COL_BLOCK);

    return FG{
        .A       = FG::A_local_tensor(dA, nullptr, nullptr, (size_t)M, (size_t)K),
        .B       = FG::B_local_tensor(dB, nullptr, nullptr, (size_t)K, (size_t)N),
        .C_final = FG::C_final_tensor(dC, cptrs, nullptr, nullptr, (size_t)M, (size_t)N),
        .C_dist  = FG::C_distributed_tensor(dC, cptrs, nullptr, nullptr, (size_t)M, (size_t)N),
        .comp_comm_barrier =
            FG::barrier_distributed_tensor(dBar, bptrs, (size_t)1, (size_t)1, br, bc),
        .dev_idx = 0,
        .M = M, .N = N, .K = K,
    };
}

static int mkernel_smem_bytes() {
    using FG = gab::fused_globals;
    return ((FG::ROW_BLOCK * FG::RED_BLOCK +
             FG::COL_BLOCK / gab::config::NUM_CLUSTERS * FG::RED_BLOCK) *
            (int)sizeof(bf16) * FG::PIPELINE_STAGES) +
           ((FG::ROW_BLOCK * FG::COL_BLOCK) * (int)sizeof(bf16)) + 1024;
}

static void launch_mkernel(const gab::fused_globals& G, cudaStream_t s) {
    const int smem = mkernel_smem_bytes();
    gab::gemm_ar_fused_kernel_stub<<<gab::config::NUM_BLOCKS, gab::config::NUM_THREADS, smem, s>>>(G);
}

// ---------------------------------------------------------------------------
// Timing: TK's discipline. One event pair around the whole loop, divide by
// iters, rotate buffer groups so consecutive launches do not reuse a warm L2.
// ---------------------------------------------------------------------------
template <typename LaunchFn>
static double bench_ms(LaunchFn&& launch, int groups, int warmup, int iters, cudaStream_t s) {
    for (int i = 0; i < warmup; ++i) launch(i % groups);
    CUDA_OK(cudaStreamSynchronize(s));

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start, s));
    for (int i = 0; i < iters; ++i) launch(i % groups);
    CUDA_OK(cudaEventRecord(stop, s));
    CUDA_OK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    return (double)ms / iters;
}

static double tflops(int M, int N, int K, double ms) {
    return (2.0 * M * N * K) / (ms * 1e9);
}

// ---------------------------------------------------------------------------

static void run_shape(int M, int N, int K, int warmup, int iters) {
    using FG = gab::fused_globals;
    if (M % FG::ROW_BLOCK || N % FG::COL_BLOCK || K % FG::RED_BLOCK) {
        std::printf("skip M=%d N=%d K=%d (needs M%%%d, N%%%d, K%%%d == 0)\n",
                    M, N, K, FG::ROW_BLOCK, FG::COL_BLOCK, FG::RED_BLOCK);
        return;
    }

    sleep_ms(500);  // cooldown between configurations, as TK does
    std::printf("\n================ M=%d N=%d K=%d ================\n", M, N, K);

    // TK's L2-eviction rule, verbatim: enough independent buffer groups to
    // cover 3x L2, or 1 group when a single set already exceeds that.
    int l2 = 0;
    CUDA_OK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size = 2 * ((size_t)M * K + (size_t)K * N + (size_t)M * N);
    const size_t ideal    = (size_t)l2 * 3;
    const int groups      = (arg_size > ideal) ? 1 : (int)(ideal / arg_size) + 1;

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    const size_t n_bar = (size_t)(M / FG::ROW_BLOCK) * (N / FG::COL_BLOCK);

    std::vector<bf16*> A(groups), B(groups), Bt(groups), C(groups);
    std::vector<int*>  Bar(groups);
    std::vector<gab::fused_globals> G;
    G.reserve(groups);

    for (int i = 0; i < groups; ++i) {
        CUDA_OK(cudaMalloc(&A[i],  nA * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&B[i],  nB * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&Bt[i], nB * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&C[i],  nC * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&Bar[i], n_bar * sizeof(int)));
        CUDA_OK(cudaMemset(C[i], 0, nC * sizeof(bf16)));
        CUDA_OK(cudaMemset(Bar[i], 0, n_bar * sizeof(int)));

        fill_random<<<(nA + 255) / 256, 256>>>(A[i], nA, 2024 + i * 100);
        fill_random<<<(nB + 255) / 256, 256>>>(B[i], nB, 2024 + i * 100 + 1);
        dim3 tb(32, 8), tg((N + 31) / 32, (K + 7) / 8);
        transpose_kn_to_nk<<<tg, tb>>>(B[i], Bt[i], K, N);

        G.push_back(make_globals(A[i], B[i], C[i], Bar[i], M, N, K));
    }
#ifdef WITH_TK
    // TK consumes B as N x K, so it gets Bt. Handles are created once, outside
    // any timed region: building globals<C> encodes three TMA descriptors on
    // the host, which must not land inside the measurement.
    std::vector<int> tk(groups, -1);
    bool tk_ok = true;
    for (int i = 0; i < groups; ++i) {
        tk[i] = tk_gemm_create(M, N, K, A[i], Bt[i], C[i]);
        if (tk[i] < 0) { tk_ok = false; break; }
    }
#endif
    bf16* Cref = nullptr;
    CUDA_OK(cudaMalloc(&Cref, nC * sizeof(bf16)));
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("buffer groups: %d (L2 = %d MiB, one arg set = %.1f MiB)\n",
                groups, l2 >> 20, arg_size / 1048576.0);

    cudaStream_t s;
    CUDA_OK(cudaStreamCreate(&s));
    cublasHandle_t h;
    CUBLAS_OK(cublasCreate(&h));
    CUBLAS_OK(cublasSetStream(h, s));

    const float alpha = 1.0f, beta = 0.0f;

    // D(MxN, row-major) = A(MxK, row-major) * B(KxN, row-major).
    // Column-major cuBLAS sees D' = N x M, B' = N x K (ld N), A' = K x M (ld K).
    auto gemm_nn = [&](bf16* Bsrc, bf16* Dst) {
        CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                               Bsrc, CUDA_R_16BF, N,
                               A[0], CUDA_R_16BF, K, &beta,
                               Dst, CUDA_R_16BF, N,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    // Same D, but B stored N x K -- TK's layout, and what its cuBLAS baseline runs.
    auto gemm_tn_g = [&](int g) {
        CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K, &alpha,
                               Bt[g], CUDA_R_16BF, K,
                               A[g], CUDA_R_16BF, K, &beta,
                               C[g], CUDA_R_16BF, N,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto gemm_nn_g = [&](int g) {
        CUBLAS_OK(cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                               B[g], CUDA_R_16BF, N,
                               A[g], CUDA_R_16BF, K, &beta,
                               C[g], CUDA_R_16BF, N,
                               CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Reference: cuBLAS NN on group 0 into its own buffer. A naive scalar
    // reference at these sizes (M*N*K up to 8.8e12) would run for hours.
    gemm_nn(B[0], Cref);
    CUDA_OK(cudaStreamSynchronize(s));

    CUDA_OK(cudaFuncSetAttribute(gab::gemm_ar_fused_kernel_stub,
                                 cudaFuncAttributeMaxDynamicSharedMemorySize,
                                 mkernel_smem_bytes()));

    struct Row { const char* name; double ms; CheckResult chk; };
    std::vector<Row> rows;

    // --- cuBLAS NN -----------------------------------------------------------
    {
        double ms = bench_ms(gemm_nn_g, groups, warmup, iters, s);
        CUDA_OK(cudaMemsetAsync(C[0], 0, nC * sizeof(bf16), s));
        gemm_nn_g(0);
        CUDA_OK(cudaStreamSynchronize(s));
        rows.push_back({"cublas-NN  (B is KxN)", ms, check(C[0], Cref, nC)});
    }
    sleep_ms(200);

    // --- cuBLAS TN -----------------------------------------------------------
    {
        double ms = bench_ms(gemm_tn_g, groups, warmup, iters, s);
        CUDA_OK(cudaMemsetAsync(C[0], 0, nC * sizeof(bf16), s));
        gemm_tn_g(0);
        CUDA_OK(cudaStreamSynchronize(s));
        rows.push_back({"cublas-TN  (B is NxK)", ms, check(C[0], Cref, nC)});
    }
    sleep_ms(200);

#ifdef WITH_TK
    // --- ThunderKittens bf16_b200 -------------------------------------------
    if (tk_ok) {
        auto run = [&](int g) { tk_gemm_launch(tk[g], s); };
        double ms = bench_ms(run, groups, warmup, iters, s);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemsetAsync(C[0], 0, nC * sizeof(bf16), s));
        run(0);
        CUDA_OK(cudaStreamSynchronize(s));
        std::printf("tk config: %s\n", tk_gemm_name(tk[0]));
        rows.push_back({"tk-b200    (B is NxK)", ms, check(C[0], Cref, nC)});
    } else {
        std::printf("tk-b200: no config for N=%d, skipped\n", N);
    }
    sleep_ms(200);
#endif

    // --- mKernel -------------------------------------------------------------
    {
        auto run = [&](int g) { launch_mkernel(G[g], s); };
        double ms = bench_ms(run, groups, warmup, iters, s);
        CUDA_OK(cudaGetLastError());
        CUDA_OK(cudaMemsetAsync(C[0], 0, nC * sizeof(bf16), s));
        run(0);
        CUDA_OK(cudaStreamSynchronize(s));
        rows.push_back({"mkernel    (B is KxN)", ms, check(C[0], Cref, nC)});
    }

    const double base = rows[0].ms;
    std::printf("%-22s %10s %12s %9s   %s\n", "", "ms", "TFLOP/s", "vs NN", "correctness");
    for (const Row& r : rows) {
        std::printf("%-22s %10.4f %12.1f %8.3fx   %s (max_abs=%.4f mean_abs=%.5f ref_mean=%.4f)\n",
                    r.name, r.ms, tflops(M, N, K, r.ms), base / r.ms,
                    r.chk.ok ? "ok  " : "FAIL", r.chk.max_abs, r.chk.mean_abs, r.chk.ref_mean);
    }

#ifdef WITH_TK
    tk_gemm_reset();  // handles hold these device pointers; drop before free
#endif
    CUBLAS_OK(cublasDestroy(h));
    CUDA_OK(cudaStreamDestroy(s));
    for (int i = 0; i < groups; ++i) {
        CUDA_OK(cudaFree(A[i]));  CUDA_OK(cudaFree(B[i]));  CUDA_OK(cudaFree(Bt[i]));
        CUDA_OK(cudaFree(C[i]));  CUDA_OK(cudaFree(Bar[i]));
    }
    CUDA_OK(cudaFree(Cref));
}

int main(int argc, char** argv) {
    const int warmup = std::getenv("BENCH_WARMUP") ? std::atoi(std::getenv("BENCH_WARMUP")) : 30;
    const int iters  = std::getenv("BENCH_ITERS")  ? std::atoi(std::getenv("BENCH_ITERS"))  : 30;

    static_assert(gab::config::NUM_COMM_SM == 0,
                  "This harness is single-GPU. Rebuild with NUM_COMP_SM == NUM_BLOCKS, "
                  "or the comm SMs will spin forever waiting on peers that do not exist.");

    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    std::printf("device: %s, SMs=%d, L2=%d MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20);
    std::printf("mkernel: NUM_BLOCKS=%d NUM_COMP_SM=%d NUM_THREADS=%d smem=%d B\n",
                gab::config::NUM_BLOCKS, gab::config::NUM_COMP_SM,
                gab::config::NUM_THREADS, mkernel_smem_bytes());
    std::printf("timing: one event pair around %d back-to-back launches, warmup=%d\n",
                iters, warmup);

    // K = N/4 mirrors the 4-rank tensor-parallel slice the python bench uses.
    for (int n : {2048, 4096, 8192, 16384, 32768}) run_shape(n, n, n / 4, warmup, iters);

    // Square shapes, for direct comparison against published TK / cuBLAS numbers.
    if (argc > 1 && std::string(argv[1]) == "--square")
        for (int n : {2048, 4096, 8192, 16384}) run_shape(n, n, n, warmup, iters);

    std::printf(
        "\nnote: mkernel still issues one dist::signal per tile "
        "(src/gemm_ar_blackwell.cu:213-216).\n"
        "      Nothing consumes it with zero comm SMs. Guard it with "
        "`if constexpr (config::NUM_COMM_SM > 0)`\n"
        "      to remove it from the measurement.\n");
    return 0;
}
