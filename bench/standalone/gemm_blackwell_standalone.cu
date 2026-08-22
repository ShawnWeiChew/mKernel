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
#include <functional>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

// The real kernel. stub_include/ must precede include/ on the -I line.
#include "../../src/gemm_ar_blackwell.cu"

#ifndef INTRA_NUM_DEVICES
#define INTRA_NUM_DEVICES 8
#endif

namespace gab = gemm_ar_intranode_blackwell;
using bf16 = __nv_bfloat16;

#ifdef WITH_TK
// Implemented in tk_gemm_shim.cu, which is a separate translation unit because
// mKernel vendors ThunderKittens into namespace kittens and the upstream
// kittens.cuh would redefine all of it. Only pointers and ints cross this line.
extern "C" {
int tk_gemm_create(int M, int N, int K, const void* A, const void* Bt, void* D);
void tk_gemm_launch(int handle, cudaStream_t s);
const char* tk_gemm_name(int handle);
void tk_gemm_reset();
}
#endif

#define CUDA_OK(call)                                                                             \
    do {                                                                                          \
        cudaError_t e_ = (call);                                                                  \
        if (e_ != cudaSuccess) {                                                                  \
            std::fprintf(stderr, "CUDA %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); \
            std::exit(1);                                                                         \
        }                                                                                         \
    } while (0)

#define CUBLAS_OK(call)                                                                     \
    do {                                                                                    \
        cublasStatus_t s_ = (call);                                                         \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                  \
            std::fprintf(stderr, "cuBLAS %s:%d: status %d\n", __FILE__, __LINE__, (int)s_); \
            std::exit(1);                                                                   \
        }                                                                                   \
    } while (0)

static void sleep_ms(int ms) {
    std::this_thread::sleep_for(std::chrono::milliseconds(ms));
}

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------

// Cheap hash-based uniform [-1, 1]. Matches TK's fill<RANDOM> in distribution,
// not in bit pattern -- the reference is computed from the same buffers, so
// only the distribution matters.
__global__ void fill_random(bf16* p, size_t n, uint64_t seed) {
    size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n)
        return;
    uint64_t x = i * 0x9E3779B97F4A7C15ull + seed;
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    float u = (float)(uint32_t)(x >> 32) * (1.0f / 4294967296.0f);  // [0,1)
    p[i] = __float2bfloat16(u * 2.0f - 1.0f);
}

// Bt[n*K + k] = B[k*N + n]
__global__ void transpose_kn_to_nk(const bf16* __restrict__ B,
                                   bf16* __restrict__ Bt,
                                   int K,
                                   int N) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    int k = blockIdx.y * blockDim.y + threadIdx.y;
    if (n < N && k < K)
        Bt[(size_t)n * K + k] = B[(size_t)k * N + n];
}

__global__ void diff_stats(const bf16* obs,
                           const bf16* ref,
                           size_t n,
                           unsigned* max_abs_bits,
                           double* sum_abs,
                           double* sum_ref) {
    __shared__ float s_max[256];
    __shared__ double s_abs[256], s_ref[256];
    int t = threadIdx.x;
    float m = 0.0f;
    double a = 0.0, r = 0.0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + t; i < n;
         i += (size_t)gridDim.x * blockDim.x) {
        float o = __bfloat162float(obs[i]);
        float e = __bfloat162float(ref[i]);
        float d = fabsf(o - e);
        m = fmaxf(m, d);
        a += d;
        r += fabsf(e);
    }
    s_max[t] = m;
    s_abs[t] = a;
    s_ref[t] = r;
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

struct CheckResult {
    float max_abs;
    double mean_abs;
    double ref_mean;
    bool ok;
};

static CheckResult check(const bf16* obs, const bf16* ref, size_t n) {
    unsigned* d_max;
    double *d_abs, *d_ref;
    CUDA_OK(cudaMalloc(&d_max, sizeof(unsigned)));
    CUDA_OK(cudaMalloc(&d_abs, sizeof(double)));
    CUDA_OK(cudaMalloc(&d_ref, sizeof(double)));
    CUDA_OK(cudaMemset(d_max, 0, sizeof(unsigned)));
    CUDA_OK(cudaMemset(d_abs, 0, sizeof(double)));
    CUDA_OK(cudaMemset(d_ref, 0, sizeof(double)));

    diff_stats<<<1024, 256>>>(obs, ref, n, d_max, d_abs, d_ref);
    CUDA_OK(cudaDeviceSynchronize());

    unsigned mb;
    double sa, sr;
    CUDA_OK(cudaMemcpy(&mb, d_max, sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&sa, d_abs, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(&sr, d_ref, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_OK(cudaFree(d_max));
    CUDA_OK(cudaFree(d_abs));
    CUDA_OK(cudaFree(d_ref));

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
static gab::fused_globals make_globals(
    bf16* dA, bf16* dB, bf16* dC, int* dBar, int M, int N, int K) {
    using FG = gab::fused_globals;
    constexpr int ND = gab::config::NUM_DEVICES;

    bf16* cptrs[ND];
    int* bptrs[ND];
    for (int i = 0; i < ND; ++i) {
        cptrs[i] = dC;
        bptrs[i] = dBar;
    }

    const size_t br = (size_t)(M / FG::ROW_BLOCK);
    const size_t bc = (size_t)(N / FG::COL_BLOCK);

    return FG{
        .A = FG::A_local_tensor(dA, nullptr, nullptr, (size_t)M, (size_t)K),
        .B = FG::B_local_tensor(dB, nullptr, nullptr, (size_t)K, (size_t)N),
        .C_final = FG::C_final_tensor(dC, cptrs, nullptr, nullptr, (size_t)M, (size_t)N),
        .C_dist = FG::C_distributed_tensor(dC, cptrs, nullptr, nullptr, (size_t)M, (size_t)N),
        .comp_comm_barrier =
            FG::barrier_distributed_tensor(dBar, bptrs, (size_t)1, (size_t)1, br, bc),
        .dev_idx = 0,
        .M = M,
        .N = N,
        .K = K,
    };
}

static int mkernel_smem_bytes() {
    // Shared with launch_fused_gemm_ar_blackwell -- do not re-derive it here.
    return gab::fused_globals::DYNAMIC_SHARED_MEMORY;
}

static void launch_mkernel(const gab::fused_globals& G, cudaStream_t s) {
    const int smem = mkernel_smem_bytes();

    if (G.M == 2048) {
        gab::gemm_ar_fused_kernel_stub<4, false>
            <<<gab::config::NUM_BLOCKS, gab::config::NUM_THREADS, smem, s>>>(G);
    } else {
        gab::gemm_ar_fused_kernel_stub<8, false>
            <<<gab::config::NUM_BLOCKS, gab::config::NUM_THREADS, smem, s>>>(G);
    }
}

// ---------------------------------------------------------------------------
// Timing protocol
// ---------------------------------------------------------------------------
// Every arm goes through a byte-identical sequence, so no arm can be favoured
// by where it happens to sit in the schedule:
//
//   for each round:                       (BENCH_ROUNDS, default 5)
//     for each arm, starting at round%n:  (rotated, so nobody is always first)
//       cool down to the idle clock       (>= BENCH_COOLDOWN_S, then poll NVML)
//       warmup launches, rotated inputs
//       one event pair around BENCH_ITERS back-to-back launches, rotated inputs
//       round mean = elapsed / iters
//
// Reported figure is the MEDIAN of the round means, with the min..max spread
// printed alongside so a noisy run is visible rather than hidden.
//
// Input rotation: launch i uses buffer group i % groups, and `groups` is sized
// to cover 3x L2, so a launch never re-reads its own previous inputs out of
// cache. When one argument set already exceeds 3x L2 there is one group and the
// rotation is a no-op -- L2 is thrashed by the problem itself at that point.

#ifndef NO_NVML
#include <nvml.h>
static bool g_nvml_up = false;
static nvmlDevice_t g_nvml_dev;
static unsigned int g_idle_clock_mhz = 0;

static unsigned int sm_clock_mhz() {
    unsigned int c = 0;
    if (!g_nvml_up || nvmlDeviceGetClockInfo(g_nvml_dev, NVML_CLOCK_SM, &c) != NVML_SUCCESS)
        return 0;
    return c;
}

// Baseline = lowest SM clock seen over a quiet second. Everything after this is
// compared against it, so an already-hot GPU at startup does not poison the run.
static void clock_monitor_init() {
    if (nvmlInit_v2() != NVML_SUCCESS) return;
    if (nvmlDeviceGetHandleByIndex_v2(0, &g_nvml_dev) != NVML_SUCCESS) return;
    g_nvml_up = true;
    unsigned int lo = ~0u;
    for (int i = 0; i < 10; ++i) {
        unsigned int c = sm_clock_mhz();
        if (c && c < lo) lo = c;
        sleep_ms(100);
    }
    g_idle_clock_mhz = (lo == ~0u) ? 0 : lo;
    std::printf("clock monitor: idle SM clock = %u MHz\n", g_idle_clock_mhz);
}
static void clock_monitor_shutdown() {
    if (g_nvml_up) nvmlShutdown();
}
#else
static unsigned int sm_clock_mhz() { return 0; }
static void clock_monitor_init() { std::printf("clock monitor: disabled (NO_NVML)\n"); }
static void clock_monitor_shutdown() {}
static unsigned int g_idle_clock_mhz = 0;
#endif

// Sleep the floor, then keep polling until the SM clock has actually settled
// back to the idle baseline. Without the poll the floor is a guess; with it we
// know the next arm starts from the same thermal/clock state as the last one.
static void cooldown(double floor_s, double tolerance, double max_extra_s) {
    CUDA_OK(cudaDeviceSynchronize());
    sleep_ms((int)(floor_s * 1000.0));
    if (!g_idle_clock_mhz) return;  // no NVML: the floor is all we have

    const unsigned int target = (unsigned int)(g_idle_clock_mhz * (1.0 + tolerance));
    double waited = 0.0;
    while (waited < max_extra_s) {
        unsigned int c = sm_clock_mhz();
        if (!c || c <= target) return;
        sleep_ms(250);
        waited += 0.25;
    }
    std::printf("  [warn] SM clock still %u MHz (idle %u) after %.0fs extra cooldown\n",
                sm_clock_mhz(),
                g_idle_clock_mhz,
                max_extra_s);
}

struct BenchOpts {
    int warmup;
    int iters;
    int rounds;
    double cooldown_s;
    double cooldown_tol;
    double cooldown_max_extra_s;
};

// One timed round: warmup, then a single event pair around `iters` launches.
template <typename LaunchFn>
static double time_round(LaunchFn&& launch, int groups, const BenchOpts& o, cudaStream_t s) {
    for (int i = 0; i < o.warmup; ++i)
        launch(i % groups);
    CUDA_OK(cudaStreamSynchronize(s));

    cudaEvent_t start, stop;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start, s));
    for (int i = 0; i < o.iters; ++i)
        launch(i % groups);
    CUDA_OK(cudaEventRecord(stop, s));
    CUDA_OK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&ms, start, stop));
    CUDA_OK(cudaEventDestroy(start));
    CUDA_OK(cudaEventDestroy(stop));
    return (double)ms / o.iters;
}

struct Arm {
    const char* name;
    std::function<void(int)> launch;  // launch(buffer group)
    std::vector<double> round_ms;
};

struct Stat {
    double median, lo, hi;
};
static Stat summarize(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    double med = (n % 2) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
    return {med, v.front(), v.back()};
}

// Runs every arm through the identical cooldown/warmup/measure sequence,
// rotating the starting arm each round.
static void run_protocol(std::vector<Arm>& arms,
                         int groups,
                         const BenchOpts& o,
                         cudaStream_t s) {
    const int n = (int)arms.size();
    for (int round = 0; round < o.rounds; ++round) {
        for (int k = 0; k < n; ++k) {
            Arm& arm = arms[(round + k) % n];
            cooldown(o.cooldown_s, o.cooldown_tol, o.cooldown_max_extra_s);
            arm.round_ms.push_back(time_round(arm.launch, groups, o, s));
            CUDA_OK(cudaGetLastError());
        }
        std::printf("  round %d/%d done\n", round + 1, o.rounds);
        std::fflush(stdout);
    }
}

static double tflops(int M, int N, int K, double ms) {
    return (2.0 * M * N * K) / (ms * 1e9);
}

// ---------------------------------------------------------------------------

static void run_shape(int M, int N, int K, const BenchOpts& o) {
    using FG = gab::fused_globals;
    if (M % FG::ROW_BLOCK || N % FG::COL_BLOCK || K % FG::RED_BLOCK) {
        std::printf("skip M=%d N=%d K=%d (needs M%%%d, N%%%d, K%%%d == 0)\n",
                    M,
                    N,
                    K,
                    FG::ROW_BLOCK,
                    FG::COL_BLOCK,
                    FG::RED_BLOCK);
        return;
    }

    sleep_ms(500);  // cooldown between configurations, as TK does
    std::printf("\n================ M=%d N=%d K=%d ================\n", M, N, K);

    // TK's L2-eviction rule, verbatim: enough independent buffer groups to
    // cover 3x L2, or 1 group when a single set already exceeds that.
    int l2 = 0;
    CUDA_OK(cudaDeviceGetAttribute(&l2, cudaDevAttrL2CacheSize, 0));
    const size_t arg_size = 2 * ((size_t)M * K + (size_t)K * N + (size_t)M * N);
    const size_t ideal = (size_t)l2 * 3;
    const int groups = (arg_size > ideal) ? 1 : (int)(ideal / arg_size) + 1;

    const size_t nA = (size_t)M * K, nB = (size_t)K * N, nC = (size_t)M * N;
    const size_t n_bar = (size_t)(M / FG::ROW_BLOCK) * (N / FG::COL_BLOCK);

    std::vector<bf16*> A(groups), B(groups), Bt(groups), C(groups);
    std::vector<int*> Bar(groups);
    std::vector<gab::fused_globals> G;
    G.reserve(groups);

    for (int i = 0; i < groups; ++i) {
        CUDA_OK(cudaMalloc(&A[i], nA * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&B[i], nB * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&Bt[i], nB * sizeof(bf16)));
        CUDA_OK(cudaMalloc(&C[i], nC * sizeof(bf16)));
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
        if (tk[i] < 0) {
            tk_ok = false;
            break;
        }
    }
#endif
    bf16* Cref = nullptr;
    CUDA_OK(cudaMalloc(&Cref, nC * sizeof(bf16)));
    CUDA_OK(cudaDeviceSynchronize());
    std::printf("buffer groups: %d (L2 = %d MiB, one arg set = %.1f MiB)\n",
                groups,
                l2 >> 20,
                arg_size / 1048576.0);

    cudaStream_t s;
    CUDA_OK(cudaStreamCreate(&s));
    cublasHandle_t h;
    CUBLAS_OK(cublasCreate(&h));
    CUBLAS_OK(cublasSetStream(h, s));

    const float alpha = 1.0f, beta = 0.0f;

    // D(MxN, row-major) = A(MxK, row-major) * B(KxN, row-major).
    // Column-major cuBLAS sees D' = N x M, B' = N x K (ld N), A' = K x M (ld K).
    auto gemm_nn = [&](bf16* Bsrc, bf16* Dst) {
        CUBLAS_OK(cublasGemmEx(h,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               N,
                               M,
                               K,
                               &alpha,
                               Bsrc,
                               CUDA_R_16BF,
                               N,
                               A[0],
                               CUDA_R_16BF,
                               K,
                               &beta,
                               Dst,
                               CUDA_R_16BF,
                               N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    // Same D, but B stored N x K -- TK's layout, and what its cuBLAS baseline runs.
    auto gemm_tn_g = [&](int g) {
        CUBLAS_OK(cublasGemmEx(h,
                               CUBLAS_OP_T,
                               CUBLAS_OP_N,
                               N,
                               M,
                               K,
                               &alpha,
                               Bt[g],
                               CUDA_R_16BF,
                               K,
                               A[g],
                               CUDA_R_16BF,
                               K,
                               &beta,
                               C[g],
                               CUDA_R_16BF,
                               N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    auto gemm_nn_g = [&](int g) {
        CUBLAS_OK(cublasGemmEx(h,
                               CUBLAS_OP_N,
                               CUBLAS_OP_N,
                               N,
                               M,
                               K,
                               &alpha,
                               B[g],
                               CUDA_R_16BF,
                               N,
                               A[g],
                               CUDA_R_16BF,
                               K,
                               &beta,
                               C[g],
                               CUDA_R_16BF,
                               N,
                               CUBLAS_COMPUTE_32F,
                               CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };

    // Reference: cuBLAS NN on group 0 into its own buffer. A naive scalar
    // reference at these sizes (M*N*K up to 8.8e12) would run for hours.
    gemm_nn(B[0], Cref);
    CUDA_OK(cudaStreamSynchronize(s));

    if (M == 2048) {
        CUDA_OK(cudaFuncSetAttribute(gab::gemm_ar_fused_kernel_stub<4, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     mkernel_smem_bytes()));
    } else {
        CUDA_OK(cudaFuncSetAttribute(gab::gemm_ar_fused_kernel_stub<8, false>,
                                     cudaFuncAttributeMaxDynamicSharedMemorySize,
                                     mkernel_smem_bytes()));
    }

    // Build the arm list. Correctness is checked once per arm, before timing,
    // so the checks cannot perturb the measured sequence.
    std::vector<Arm> arms;
    arms.push_back({"cublas-NN  (B is KxN)", [&](int g) { gemm_nn_g(g); }, {}});
    arms.push_back({"cublas-TN  (B is NxK)", [&](int g) { gemm_tn_g(g); }, {}});
#ifdef WITH_TK
    if (tk_ok) {
        arms.push_back({"tk-b200    (B is NxK)", [&](int g) { tk_gemm_launch(tk[g], s); }, {}});
        std::printf("tk config: %s\n", tk_gemm_name(tk[0]));
    } else {
        std::printf("tk-b200: no config for N=%d, skipped\n", N);
    }
#endif
    arms.push_back({"mkernel    (B is KxN)", [&](int g) { launch_mkernel(G[g], s); }, {}});

    std::vector<CheckResult> checks;
    for (Arm& a : arms) {
        CUDA_OK(cudaMemsetAsync(C[0], 0, nC * sizeof(bf16), s));
        a.launch(0);
        CUDA_OK(cudaStreamSynchronize(s));
        CUDA_OK(cudaGetLastError());
        checks.push_back(check(C[0], Cref, nC));
    }

    std::printf("protocol: %d rounds x %d iters (warmup %d), rotated arm order, "
                ">=%.0fs cooldown per arm\n",
                o.rounds,
                o.iters,
                o.warmup,
                o.cooldown_s);
    std::fflush(stdout);
    run_protocol(arms, groups, o, s);

    const Stat base = summarize(arms[0].round_ms);
    std::printf("\n%-22s %10s %12s %9s %11s   %s\n",
                "",
                "ms",
                "TFLOP/s",
                "vs NN",
                "spread",
                "correctness");
    for (size_t i = 0; i < arms.size(); ++i) {
        const Stat st = summarize(arms[i].round_ms);
        const CheckResult& c = checks[i];
        std::printf("%-22s %10.4f %12.1f %8.3fx %10.1f%%   %s "
                    "(max_abs=%.4f mean_abs=%.5f ref_mean=%.4f)\n",
                    arms[i].name,
                    st.median,
                    tflops(M, N, K, st.median),
                    base.median / st.median,
                    100.0 * (st.hi - st.lo) / st.median,
                    c.ok ? "ok  " : "FAIL",
                    c.max_abs,
                    c.mean_abs,
                    c.ref_mean);
    }

#ifdef WITH_TK
    tk_gemm_reset();  // handles hold these device pointers; drop before free
#endif
    CUBLAS_OK(cublasDestroy(h));
    CUDA_OK(cudaStreamDestroy(s));
    for (int i = 0; i < groups; ++i) {
        CUDA_OK(cudaFree(A[i]));
        CUDA_OK(cudaFree(B[i]));
        CUDA_OK(cudaFree(Bt[i]));
        CUDA_OK(cudaFree(C[i]));
        CUDA_OK(cudaFree(Bar[i]));
    }
    CUDA_OK(cudaFree(Cref));
}

static double env_d(const char* k, double dflt) {
    const char* v = std::getenv(k);
    return v ? std::atof(v) : dflt;
}
static int env_i(const char* k, int dflt) {
    const char* v = std::getenv(k);
    return v ? std::atoi(v) : dflt;
}

int main(int argc, char** argv) {
    BenchOpts o;
    o.warmup = env_i("BENCH_WARMUP", 10);
    o.iters = env_i("BENCH_ITERS", 25);
    o.rounds = env_i("BENCH_ROUNDS", 5);
    o.cooldown_s = env_d("BENCH_COOLDOWN_S", 8.0);
    o.cooldown_tol = env_d("BENCH_COOLDOWN_TOL", 0.05);
    o.cooldown_max_extra_s = env_d("BENCH_COOLDOWN_MAX_S", 60.0);

    static_assert(gab::config::NUM_COMM_SM == 0,
                  "This harness is single-GPU. Rebuild with NUM_COMP_SM == NUM_BLOCKS, "
                  "or the comm SMs will spin forever waiting on peers that do not exist.");

    cudaDeviceProp p;
    CUDA_OK(cudaGetDeviceProperties(&p, 0));
    std::printf(
        "device: %s, SMs=%d, L2=%d MiB\n", p.name, p.multiProcessorCount, p.l2CacheSize >> 20);
    std::printf("mkernel: NUM_BLOCKS=%d NUM_COMP_SM=%d NUM_THREADS=%d smem=%d B\n",
                gab::config::NUM_BLOCKS,
                gab::config::NUM_COMP_SM,
                gab::config::NUM_THREADS,
                mkernel_smem_bytes());
    std::printf("timing: %d rounds x %d back-to-back launches (warmup %d), rotated arm order,\n"
                "        >=%.0fs cooldown per timed arm, polled to within %.0f%% of idle SM clock\n",
                o.rounds,
                o.iters,
                o.warmup,
                o.cooldown_s,
                100.0 * o.cooldown_tol);
    clock_monitor_init();

    // K = N/4 mirrors the 4-rank tensor-parallel slice the python bench uses.
    for (int n : {2048, 4096, 8192, 16384, 32768})
        run_shape(n, n, n / INTRA_NUM_DEVICES, o);

    // Square shapes, for direct comparison against published TK / cuBLAS numbers.
    if (argc > 1 && std::string(argv[1]) == "--square")
        for (int n : {2048, 4096, 8192, 16384})
            run_shape(n, n, n, o);
    clock_monitor_shutdown();
    return 0;
}
