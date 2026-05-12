// sgemm_async.cu
// Double-buffered register-tiled SGEMM using cp.async PTX intrinsics.
//
// cp.async (Ampere sm_80+) is a hardware instruction that copies data from
// global memory directly into shared memory via a dedicated DMA engine,
// WITHOUT going through registers and WITHOUT stalling the issuing warp.
//
// This lets us hide the GDDR6→SMEM latency (100-200 cycles) behind the
// FP32 FMA instructions computing the previous tile — the same concept as
// double-buffered BRAM in HLS systolic arrays.
//
// Build:  nvcc -O3 -arch=sm_86 -o sgemm_async sgemm_async.cu -lcublas
// Run:    ./sgemm_async

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── Error checking ────────────────────────────────────────────────────────────

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
// PTX helpers for cp.async — no extra headers needed
//
// cp.async.ca.shared.global [smem], [gmem], 4
//   Copies 4 bytes from gmem → smem via the async copy engine.
//   The issuing warp continues immediately without waiting.
//
// cp.async.commit_group
//   Marks all outstanding async copies as one named "group".
//
// cp.async.wait_group N
//   Stalls until at most N groups are still in flight.
//   N=0 → wait for all,  N=1 → wait for all but the most recent.
// ═══════════════════════════════════════════════════════════════════════════════

__device__ __forceinline__
void cp_async4(void* smem, const void* gmem)
{
    uint32_t smem_addr = __cvta_generic_to_shared(smem);
    asm volatile(
        "cp.async.ca.shared.global [%0], [%1], 4;\n"
        : : "r"(smem_addr), "l"(gmem));
}

__device__ __forceinline__ void cp_async_commit()
{
    asm volatile("cp.async.commit_group;\n" ::);
}

// N must be a compile-time constant (PTX requires it)
template<int N>
__device__ __forceinline__ void cp_async_wait()
{
    asm volatile("cp.async.wait_group %0;\n" : : "n"(N));
}

// ═══════════════════════════════════════════════════════════════════════════════
// SYNCHRONOUS baseline — register-tiled, synchronous loads
// ═══════════════════════════════════════════════════════════════════════════════

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_sync(int M, int N, int K,
                           float alpha,
                           const float* __restrict__ A,
                           const float* __restrict__ B,
                           float beta,
                           float* __restrict__ C)
{
    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;
    constexpr int N_THREADS   = THREAD_ROWS * THREAD_COLS;

    int t_row = threadIdx.x / THREAD_COLS;
    int t_col = threadIdx.x % THREAD_COLS;
    int tid   = threadIdx.x;

    __shared__ float As[BK][BM];
    __shared__ float Bs[BK][BN];

    float acc[TM][TN] = {};
    float a_frag[TM], b_frag[TN];

    for (int k0 = 0; k0 < K; k0 += BK) {
        for (int i = tid; i < BM * BK; i += N_THREADS) {
            int bm = i % BM, bk = i / BM;
            int gr = blockIdx.y * BM + bm, gc = k0 + bk;
            As[bk][bm] = (gr < M && gc < K) ? A[gr * K + gc] : 0.0f;
        }
        for (int i = tid; i < BK * BN; i += N_THREADS) {
            int bn = i % BN, bk = i / BN;
            int gr = k0 + bk, gc = blockIdx.x * BN + bn;
            Bs[bk][bn] = (gr < K && gc < N) ? B[gr * N + gc] : 0.0f;
        }
        __syncthreads();   // ← stall here waiting for loads

        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m) a_frag[m] = As[k][t_row * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; ++n) b_frag[n] = Bs[k][t_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_frag[m] * b_frag[n];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gr = blockIdx.y * BM + t_row * TM + m;
            int gc = blockIdx.x * BN + t_col * TN + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = alpha * acc[m][n] + beta * C[gr * N + gc];
        }
}

// ═══════════════════════════════════════════════════════════════════════════════
// ASYNC kernel — double-buffered with cp.async PTX
//
// Ping-pong buffers: As[2][BK][BM], Bs[2][BK][BN]
//
// Timeline for strip i:
//
//   strip i-1: [compute i-1] ───────────────────────────────┐
//   strip i  :               [DMA load i] ──────────────────┤ overlap
//   strip i+1:                             [DMA load i+1] ──┘
//              ↕ cp_async_wait<1> waits for load i to complete
//                before compute i starts, but load i+1 is still running
//
// Assumes M, N, K are multiples of BM, BN, BK for clean cp.async paths.
// (All test sizes 512, 1024, 2048, 4096 satisfy this.)
// ═══════════════════════════════════════════════════════════════════════════════

template<int BM, int BN, int BK, int TM, int TN>
__global__ void sgemm_async(int M, int N, int K,
                            float alpha,
                            const float* __restrict__ A,
                            const float* __restrict__ B,
                            float beta,
                            float* __restrict__ C)
{
    constexpr int THREAD_ROWS = BM / TM;
    constexpr int THREAD_COLS = BN / TN;
    constexpr int N_THREADS   = THREAD_ROWS * THREAD_COLS;

    int t_row = threadIdx.x / THREAD_COLS;
    int t_col = threadIdx.x % THREAD_COLS;
    int tid   = threadIdx.x;

    // ── Double ping-pong shared memory buffers ────────────────────────────────
    // Total: 2*(BK*BM + BK*BN)*4 = 2*(16*128 + 16*128)*4 = 32 KB
    __shared__ float As[2][BK][BM];
    __shared__ float Bs[2][BK][BN];

    float acc[TM][TN] = {};
    float a_frag[TM], b_frag[TN];

    int n_strips = K / BK;   // assumes K % BK == 0

    // ── Helper: issue async loads for strip s into buffer buf ─────────────────
    // cp_async4 fires and forgets — the warp does NOT wait.
    auto issue_loads = [&](int s, int buf) {
        int k0 = s * BK;
        for (int i = tid; i < BM * BK; i += N_THREADS) {
            int bm = i % BM, bk = i / BM;
            cp_async4(&As[buf][bk][bm],
                      &A[(blockIdx.y * BM + bm) * K + (k0 + bk)]);
        }
        for (int i = tid; i < BK * BN; i += N_THREADS) {
            int bn = i % BN, bk = i / BN;
            cp_async4(&Bs[buf][bk][bn],
                      &B[(k0 + bk) * N + (blockIdx.x * BN + bn)]);
        }
    };

    // ── Prologue: kick off loads for strip 0 into buffer 0 ───────────────────
    issue_loads(0, 0);
    cp_async_commit();   // group 0 submitted

    // ── Main pipeline loop ────────────────────────────────────────────────────
    for (int strip = 0; strip < n_strips; ++strip) {
        int curr_buf = strip & 1;
        int next_buf = 1 - curr_buf;

        if (strip + 1 < n_strips) {
            // Fire DMA loads for the NEXT strip into the other buffer.
            // These will complete while we are computing the current strip.
            issue_loads(strip + 1, next_buf);
            cp_async_commit();   // group for strip+1 submitted

            // Wait for current strip's group to land (all groups except latest 1).
            // The latest 1 (for strip+1) keeps running in background.
            cp_async_wait<1>();
        } else {
            // Last strip — no new group issued, wait for everything.
            cp_async_wait<0>();
        }

        // Barrier: all threads in the block must see the shared mem update.
        __syncthreads();

        // ── Inner FMA loop — runs while next strip's DMA is in flight ─────────
        #pragma unroll
        for (int k = 0; k < BK; ++k) {
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                a_frag[m] = As[curr_buf][k][t_row * TM + m];
            #pragma unroll
            for (int n = 0; n < TN; ++n)
                b_frag[n] = Bs[curr_buf][k][t_col * TN + n];
            #pragma unroll
            for (int m = 0; m < TM; ++m)
                #pragma unroll
                for (int n = 0; n < TN; ++n)
                    acc[m][n] += a_frag[m] * b_frag[n];
        }

        // Ensure all threads finish reading curr_buf before a future strip
        // tries to overwrite it (ping-pong safety).
        __syncthreads();
    }

    // ── Writeback ─────────────────────────────────────────────────────────────
    #pragma unroll
    for (int m = 0; m < TM; ++m)
        #pragma unroll
        for (int n = 0; n < TN; ++n) {
            int gr = blockIdx.y * BM + t_row * TM + m;
            int gc = blockIdx.x * BN + t_col * TN + n;
            if (gr < M && gc < N)
                C[gr * N + gc] = alpha * acc[m][n] + beta * C[gr * N + gc];
        }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Timing + verification utilities
// ═══════════════════════════════════════════════════════════════════════════════

template<typename Fn>
double time_kernel(Fn fn, int M, int N, int K, int warmup, int iters)
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

    return 2.0 * M * N * K * iters / (ms * 1e6);
}

bool verify(const float* ref, const float* got, int sz, float tol = 1e-2f)
{
    for (int i = 0; i < sz; ++i) {
        float d = fabsf(ref[i] - got[i]);
        if (d > tol * (fabsf(ref[i]) + 1.0f)) {
            printf("  MISMATCH [%d]: ref=%.5f  got=%.5f\n", i, ref[i], got[i]);
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

    constexpr int BM = 128, BN = 128, BK = 16, TM = 8, TN = 8;
    constexpr int N_THREADS = (BM / TM) * (BN / TN);   // 256

    // All sizes must be multiples of BM=128, BN=128, BK=16 for clean async paths
    int sizes[] = {512, 1024, 2048, 4096};
    int warmup  = 3, iters = 10;

    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %s\n",
           "M=N=K", "Sync(GF)", "Async(GF)", "cuBLAS(GF)",
           "Async/Sync", "Async/cuBLAS");
    printf("%-6s  %-10s  %-10s  %-10s  %-10s  %s\n",
           "-----", "--------", "---------", "----------",
           "----------", "------------");

    for (int S : sizes) {
        int M = S, N = S, K = S;
        size_t bA = (size_t)M * K * 4;
        size_t bB = (size_t)K * N * 4;
        size_t bC = (size_t)M * N * 4;

        float *dA, *dB, *dC_ref, *dC_cus;
        CUDA_CHECK(cudaMalloc(&dA, bA));
        CUDA_CHECK(cudaMalloc(&dB, bB));
        CUDA_CHECK(cudaMalloc(&dC_ref, bC));
        CUDA_CHECK(cudaMalloc(&dC_cus, bC));
        CUDA_CHECK(cudaMemset(dA,     1, bA));
        CUDA_CHECK(cudaMemset(dB,     1, bB));
        CUDA_CHECK(cudaMemset(dC_ref, 0, bC));

        float alpha = 1.0f, beta = 0.0f;

        // cuBLAS reference
        auto run_cublas = [&]() {
            CUBLAS_CHECK(cublasSgemm(cublas, CUBLAS_OP_N, CUBLAS_OP_N,
                                     N, M, K, &alpha, dB, N, dA, K,
                                     &beta, dC_ref, N));
        };
        double gf_cublas = time_kernel(run_cublas, M, N, K, warmup, iters);
        float* hRef = new float[(size_t)M * N];
        CUDA_CHECK(cudaMemcpy(hRef, dC_ref, bC, cudaMemcpyDeviceToHost));

        dim3 block(N_THREADS);
        dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

        // Sync kernel
        CUDA_CHECK(cudaMemset(dC_cus, 0, bC));
        auto run_sync = [&]() {
            sgemm_sync<BM, BN, BK, TM, TN><<<grid, block>>>(
                M, N, K, alpha, dA, dB, beta, dC_cus);
        };
        double gf_sync = time_kernel(run_sync, M, N, K, warmup, iters);

        // Async kernel
        CUDA_CHECK(cudaMemset(dC_cus, 0, bC));
        auto run_async = [&]() {
            sgemm_async<BM, BN, BK, TM, TN><<<grid, block>>>(
                M, N, K, alpha, dA, dB, beta, dC_cus);
        };
        double gf_async = time_kernel(run_async, M, N, K, warmup, iters);

        float* hGot = new float[(size_t)M * N];
        CUDA_CHECK(cudaMemcpy(hGot, dC_cus, bC, cudaMemcpyDeviceToHost));
        bool ok = verify(hRef, hGot, M * N);

        printf("%-6d  %-10.0f  %-10.0f  %-10.0f  %-10.1f%%  %.1f%%  %s\n",
               S, gf_sync, gf_async, gf_cublas,
               gf_async / gf_sync * 100.0,
               gf_async / gf_cublas * 100.0,
               ok ? "✓" : "✗ MISMATCH");

        delete[] hRef; delete[] hGot;
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC_ref)); CUDA_CHECK(cudaFree(dC_cus));
    }

    printf("\n");
    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
