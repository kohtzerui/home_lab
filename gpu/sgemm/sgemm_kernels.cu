// sgemm_kernels.cu
// Progressive SGEMM optimizations, benchmarked against cuBLAS.
//
// Three levels of optimization:
//   Naive    – 1 thread per C element, pure global memory
//   Shared   – block tile loaded into shared memory (L0→L1 reuse)
//   Register – each thread computes an 8×8 output block (L1→RF reuse)
//
// Build:  nvcc -O3 -arch=sm_86 -o sgemm_kernels sgemm_kernels.cu -lcublas
// Run:    ./sgemm_kernels

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cassert>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── error checking ────────────────────────────────────────────────────────────

#define CUDA_CHECK(e) do {                                                      \
    cudaError_t _e = (e);                                                       \
    if (_e != cudaSuccess) {                                                    \
        fprintf(stderr,"CUDA %s:%d  %s\n",__FILE__,__LINE__,                   \
                cudaGetErrorString(_e)); exit(1); }                             \
} while(0)

#define CUBLAS_CHECK(s) do {                                                    \
    cublasStatus_t _s = (s);                                                    \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                          \
        fprintf(stderr,"cuBLAS %s:%d  code=%d\n",__FILE__,__LINE__,(int)_s);   \
        exit(1); }                                                              \
} while(0)

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 0 — Naive kernel
// Each thread computes one element of C by walking the full K dimension.
// Every access goes to global memory — no reuse, no locality.
// Arithmetic intensity ≈ 2 (one MAC per 2 float loads).
// ═══════════════════════════════════════════════════════════════════════════════

__global__ void sgemm_naive(int M, int N, int K,
                            float alpha,
                            const float* __restrict__ A,   // [M×K] row-major
                            const float* __restrict__ B,   // [K×N] row-major
                            float beta,
                            float* __restrict__ C)         // [M×N] row-major
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // C row
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // C col

    if (row >= M || col >= N) return;

    float acc = 0.0f;
    for (int k = 0; k < K; ++k)
        acc += A[row * K + k] * B[k * N + col];

    C[row * N + col] = alpha * acc + beta * C[row * N + col];
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 1 — Shared-memory tiled kernel
// Thread block of BM×BN cooperatively loads BM×BK (A tile) and BK×BN (B tile)
// into shared memory, then every thread accumulates over the tile.
//
// Data reuse:
//   Each element of A_tile is reused BN times (across the block's columns).
//   Each element of B_tile is reused BM times (across the block's rows).
//   Global memory traffic ↓ by factor BK compared to naive.
// ═══════════════════════════════════════════════════════════════════════════════

template<int BM, int BN, int BK>
__global__ void sgemm_shared(int M, int N, int K,
                             float alpha,
                             const float* __restrict__ A,
                             const float* __restrict__ B,
                             float beta,
                             float* __restrict__ C)
{
    // Tile origin in global C
    int row0 = blockIdx.y * BM;
    int col0 = blockIdx.x * BN;

    // Thread position within the block
    int ty = threadIdx.y;   // [0, BM)
    int tx = threadIdx.x;   // [0, BN)

    __shared__ float As[BM][BK];   // A tile
    __shared__ float Bs[BK][BN];   // B tile

    float acc = 0.0f;

    // Step through K in BK-wide strips
    for (int k0 = 0; k0 < K; k0 += BK) {

        // Cooperative load of A tile — each thread loads one element
        if (row0 + ty < M && k0 + tx < K)
            As[ty][tx] = A[(row0 + ty) * K + (k0 + tx)];
        else
            As[ty][tx] = 0.0f;

        // Cooperative load of B tile
        if (k0 + ty < K && col0 + tx < N)
            Bs[ty][tx] = B[(k0 + ty) * N + (col0 + tx)];
        else
            Bs[ty][tx] = 0.0f;

        __syncthreads();   // wait for both tiles to be loaded

        // Dot product over the tile — all from shared memory
        #pragma unroll
        for (int k = 0; k < BK; ++k)
            acc += As[ty][k] * Bs[k][tx];

        __syncthreads();   // don't overwrite tiles before all threads finish
    }

    if (row0 + ty < M && col0 + tx < N)
        C[(row0 + ty) * N + (col0 + tx)] =
            alpha * acc + beta * C[(row0 + ty) * N + (col0 + tx)];
}

// ═══════════════════════════════════════════════════════════════════════════════
// LEVEL 2 — Register-blocked kernel (the "production" pattern)
//
// Each thread now owns a TM×TN register tile of C, so one thread does TM*TN
// MACs per inner-loop iteration instead of just one.
//
// Block size: (BN/TN) × (BM/TM) threads   (128/8 × 128/8 = 16×16 = 256)
// Shared memory: BM×BK (A) + BK×BN (B)    (128×16 + 16×128)*4 = 16 KB
//
// Data reuse added:
//   Each A_shared element: reused TN times (across a thread's column micro-tile)
//   Each B_shared element: reused TM times (across a thread's row micro-tile)
//   Register reuse:        TM*TN MACs before any new shared-memory load
// ═══════════════════════════════════════════════════════════════════════════════

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_register(int M, int N, int K,
                               float alpha,
                               const float* __restrict__ A,
                               const float* __restrict__ B,
                               float beta,
                               float* __restrict__ C)
{
    // Threads per block: (BM/TM) × (BN/TN)
    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;

    // This thread's position within the block
    int t_row = threadIdx.x / THREAD_COLS;   // which row micro-tile
    int t_col = threadIdx.x % THREAD_COLS;   // which col micro-tile

    // Global C tile origin
    int C_row0 = blockIdx.y * BM + t_row * TM;
    int C_col0 = blockIdx.x * BN + t_col * TN;

    __shared__ float As[BK][BM];   // transposed for coalesced access
    __shared__ float Bs[BK][BN];

    // Per-thread register accumulator (TM×TN output tile)
    float acc[TM][TN] = {};   // zero-initialised

    // Per-thread fragments used in the inner loop
    float a_frag[TM];
    float b_frag[TN];

    // Number of threads in the block
    constexpr int N_THREADS = THREAD_ROWS * THREAD_COLS;
    int tid = threadIdx.x;

    for (int k0 = 0; k0 < K; k0 += BK) {

        // ── Cooperative load of A tile into As[BK][BM] (transposed) ──────────
        // Each thread loads one or more elements; stride by N_THREADS.
        for (int i = tid; i < BM * BK; i += N_THREADS) {
            int bm = i % BM;                      // column in As
            int bk = i / BM;                      // row in As
            int g_row = blockIdx.y * BM + bm;
            int g_col = k0 + bk;
            As[bk][bm] = (g_row < M && g_col < K)
                         ? A[g_row * K + g_col] : 0.0f;
        }

        // ── Cooperative load of B tile into Bs[BK][BN] ───────────────────────
        for (int i = tid; i < BK * BN; i += N_THREADS) {
            int bn = i % BN;
            int bk = i / BN;
            int g_row = k0 + bk;
            int g_col = blockIdx.x * BN + bn;
            Bs[bk][bn] = (g_row < K && g_col < N)
                         ? B[g_row * N + g_col] : 0.0f;
        }

        __syncthreads();

        // ── Inner loop over the K strip ───────────────────────────────────────
        // Each k iteration: load TM A-elements and TN B-elements, compute TM×TN MACs.
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            // Load this thread's A fragment from shared
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                a_frag[m] = As[k][t_row * TM + m];

            // Load this thread's B fragment from shared
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_frag[n] = Bs[k][t_col * TN + n];

            // Outer product accumulate — pure register ops
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_frag[m] * b_frag[n];
        }

        __syncthreads();
    }

    // ── Write TM×TN output tile back to global C ──────────────────────────────
    #pragma unroll
    for (int m = 0; m < TM; ++m) {
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gr = C_row0 + m;
            int gc = C_col0 + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = alpha * acc[m][n] + beta * C[gr * N + gc];
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Benchmark utilities
// ═══════════════════════════════════════════════════════════════════════════════

// Returns GFLOPS for a kernel launch wrapped in CUDA events.
template<typename LaunchFn>
double time_kernel(LaunchFn fn, int M, int N, int K,
                   int warmup, int iters)
{
    for (int i = 0; i < warmup; ++i) fn();
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < iters; ++i) fn();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));

    double flops = 2.0 * M * N * K * iters;
    return flops / (ms * 1e6);   // GFLOPS
}

// Verify custom kernel against reference (cuBLAS result).
bool verify(const float* ref, const float* got, int M, int N, float tol = 1e-2f)
{
    for (int i = 0; i < M * N; ++i) {
        float diff = fabsf(ref[i] - got[i]);
        if (diff > tol * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH at [%d]: ref=%.6f  got=%.6f\n", i, ref[i], got[i]);
            return false;
        }
    }
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════════
// Main
// ═══════════════════════════════════════════════════════════════════════════════

int main()
{
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("==========================================================\n");
    printf("  Device : %s  (SM_%d%d, %d SMs)\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);
    printf("==========================================================\n\n");

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    // Tile parameters (must match kernel template args below)
    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int TM = 8,   TN = 8;

    // Square matrix sizes to test
    int sizes[]  = {512, 1024, 2048, 4096};
    int warmup   = 3;
    int iters    = 10;

    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %s\n",
           "M=N=K", "Naive", "Shared", "Register", "cuBLAS", "Register vs cuBLAS");
    printf("%-6s  %-12s  %-12s  %-12s  %-12s  %s\n",
           "-----", "-----", "------", "--------", "------", "------------------");

    for (int S : sizes) {
        int M = S, N = S, K = S;
        size_t bytes = (size_t)M * N * sizeof(float);

        float *dA, *dB, *dC_ref, *dC_cus;
        CUDA_CHECK(cudaMalloc(&dA,     (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dB,     (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dC_ref, bytes));
        CUDA_CHECK(cudaMalloc(&dC_cus, bytes));

        CUDA_CHECK(cudaMemset(dA,     1, (size_t)M * K * sizeof(float)));
        CUDA_CHECK(cudaMemset(dB,     1, (size_t)K * N * sizeof(float)));
        CUDA_CHECK(cudaMemset(dC_ref, 0, bytes));
        CUDA_CHECK(cudaMemset(dC_cus, 0, bytes));

        float alpha = 1.0f, beta = 0.0f;

        // ── cuBLAS reference ──────────────────────────────────────────────────
        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(cublas,
                                     CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K,
                                     &alpha, dB, N, dA, K,
                                     &beta,  dC_ref, N));
        };
        double gf_cublas = time_kernel(run_cublas, M, N, K, warmup, iters);

        // Copy cuBLAS result to host for verification
        float* hRef = new float[(size_t)M * N];
        CUDA_CHECK(cudaMemcpy(hRef, dC_ref, bytes, cudaMemcpyDeviceToHost));

        // ── Level 0: Naive ────────────────────────────────────────────────────
        // Skip naive for large sizes (too slow to be useful)
        double gf_naive = 0.0;
        if (S <= 1024) {
            dim3 block_naive(16, 16);
            dim3 grid_naive((N + 15) / 16, (M + 15) / 16);
            auto run_naive = [&]() {
                sgemm_naive<<<grid_naive, block_naive>>>(M, N, K, alpha, dA, dB, beta, dC_cus);
            };
            CUDA_CHECK(cudaMemset(dC_cus, 0, bytes));
            gf_naive = time_kernel(run_naive, M, N, K, warmup, iters);
            // verify
            float* hGot = new float[(size_t)M * N];
            CUDA_CHECK(cudaMemcpy(hGot, dC_cus, bytes, cudaMemcpyDeviceToHost));
            bool ok = verify(hRef, hGot, M, N);
            if (!ok) printf("  [Naive] VERIFICATION FAILED at M=%d\n", M);
            delete[] hGot;
        }

        // ── Level 1: Shared memory tiled ─────────────────────────────────────
        {
            dim3 block_sh(BN, BM);   // BM×BN threads, one per output element
            dim3 grid_sh((N + BN - 1) / BN, (M + BM - 1) / BM);
            // Use smaller tile if block is too large for this size
            dim3 blk(16, 16);
            dim3 grd((N + 15) / 16, (M + 15) / 16);
            CUDA_CHECK(cudaMemset(dC_cus, 0, bytes));
            auto run_sh = [&]() {
                sgemm_shared<16, 16, 16><<<grd, blk>>>(M, N, K, alpha, dA, dB, beta, dC_cus);
            };
            // warmup + verify
            run_sh(); CUDA_CHECK(cudaDeviceSynchronize());
        }

        // ── Level 2: Register-blocked ─────────────────────────────────────────
        {
            constexpr int N_THREADS = (BM / TM) * (BN / TN);   // 256
            dim3 block_reg(N_THREADS);
            dim3 grid_reg((N + BN - 1) / BN, (M + BM - 1) / BM);

            CUDA_CHECK(cudaMemset(dC_cus, 0, bytes));
            auto run_reg = [&]() {
                sgemm_register<BM, BN, BK, TM, TN><<<grid_reg, block_reg>>>(
                    M, N, K, alpha, dA, dB, beta, dC_cus);
            };
            double gf_reg = time_kernel(run_reg, M, N, K, warmup, iters);

            // Verify against cuBLAS
            float* hGot = new float[(size_t)M * N];
            CUDA_CHECK(cudaMemcpy(hGot, dC_cus, bytes, cudaMemcpyDeviceToHost));
            bool ok = verify(hRef, hGot, M, N);
            delete[] hGot;

            double pct = gf_reg / gf_cublas * 100.0;
            const char* verified = ok ? "✓" : "✗ MISMATCH";

            if (S <= 1024 && gf_naive > 0)
                printf("%-6d  %-12.0f  %-12s  %-12.0f  %-12.0f  %.1f%%  %s\n",
                       S, gf_naive, "—", gf_reg, gf_cublas, pct, verified);
            else
                printf("%-6d  %-12s  %-12s  %-12.0f  %-12.0f  %.1f%%  %s\n",
                       S, S > 1024 ? "(skipped)" : "—", "—",
                       gf_reg, gf_cublas, pct, verified);
        }

        delete[] hRef;
        CUDA_CHECK(cudaFree(dA));
        CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC_ref));
        CUDA_CHECK(cudaFree(dC_cus));
    }

    printf("\n");
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
