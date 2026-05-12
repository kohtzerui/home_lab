// sgemm_wmma.cu — Tensor Core SGEMM using WMMA intrinsics
// Inputs: float32. Tiles are converted to fp16 in shared memory.
// Accumulation: fp16 matmul → fp32 accumulator.
//
// Block tile : BM=128, BN=128, BK=16
// Warp layout: 2×2 warps per block (128 threads total)
// Each warp  : 4×4 = 16 WMMA 16×16 tiles (covers 64×64 of C)
//
// Build: nvcc -O3 -arch=sm_86 -o sgemm_wmma sgemm_wmma.cu -lcublas
// Run  : ./sgemm_wmma

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>

using namespace nvcuda::wmma;

#define CUDA_CHECK(e) do { cudaError_t _e=(e); if(_e!=cudaSuccess){          \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,                      \
            cudaGetErrorString(_e));exit(1);} } while(0)
#define CUBLAS_CHECK(s) do { cublasStatus_t _s=(s);                           \
    if(_s!=CUBLAS_STATUS_SUCCESS){fprintf(stderr,"cuBLAS err %d\n",(int)_s);  \
    exit(1);} } while(0)

// ── PTX cp.async helpers (same as sgemm_async.cu) ────────────────────────────
__device__ __forceinline__ void cp_async4(void* s, const void* g) {
    uint32_t a = __cvta_generic_to_shared(s);
    asm volatile("cp.async.ca.shared.global [%0],[%1],4;\n"::"r"(a),"l"(g));
}
__device__ __forceinline__ void cp_async_commit() {
    asm volatile("cp.async.commit_group;\n"::);
}
template<int N> __device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n"::"n"(N));
}

// ═══════════════════════════════════════════════════════════════════════════════
// WMMA kernel
// ─────────────────────────────────────────────────────────────────────────────
// Shared memory layout (FP16):
//   As[BM][BK]  — A tile, row-major (M rows, K cols)
//   Bs[BK][BN]  — B tile, row-major (K rows, N cols)
//
// WMMA loads:
//   matrix_a row_major: load_matrix_sync(a_frag, &As[m_start][k], BK)
//   matrix_b row_major: load_matrix_sync(b_frag, &Bs[k][n_start], BN)
// ═══════════════════════════════════════════════════════════════════════════════

template<int BM, int BN, int BK>
__global__ void sgemm_wmma_kernel(int M, int N, int K,
                                  float alpha,
                                  const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float beta,
                                  float* __restrict__ C)
{
    constexpr int WMMA_M = 16, WMMA_N = 16, WMMA_K = 16;
    // 2×2 warp grid per block → 4 warps = 128 threads
    constexpr int WARP_ROWS = 2, WARP_COLS = 2;
    // Each warp's output region
    constexpr int WM = BM / WARP_ROWS;   // 64
    constexpr int WN = BN / WARP_COLS;   // 64
    // WMMA tiles per warp
    constexpr int TM = WM / WMMA_M;      // 4
    constexpr int TN = WN / WMMA_N;      // 4
    constexpr int N_THREADS = WARP_ROWS * WARP_COLS * 32; // 128

    int tid     = threadIdx.x;
    int warp_id = tid / 32;
    int warp_r  = warp_id / WARP_COLS;   // 0 or 1
    int warp_c  = warp_id % WARP_COLS;   // 0 or 1

    int block_row = blockIdx.y * BM;
    int block_col = blockIdx.x * BN;
    int warp_row  = warp_r * WM;          // offset within block (0 or 64)
    int warp_col  = warp_c * WN;          // offset within block (0 or 64)

    // ── Shared memory (FP16) ──────────────────────────────────────────────────
    // As[BM][BK] = 128×16 halfs = 4 KB
    // Bs[BK][BN] = 16×128 halfs = 4 KB  →  total 8 KB
    __shared__ half As[BM][BK];
    __shared__ half Bs[BK][BN];

    // ── Per-warp accumulators (TM × TN WMMA fragments) ───────────────────────
    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[TM][TN];
    #pragma unroll
    for (int m = 0; m < TM; m++)
        #pragma unroll
        for (int n = 0; n < TN; n++)
            fill_fragment(acc[m][n], 0.0f);

    // ── Main K loop ───────────────────────────────────────────────────────────
    for (int k0 = 0; k0 < K; k0 += BK) {

        // Load A tile: global float → shared half  [BM × BK]
        for (int i = tid; i < BM * BK; i += N_THREADS) {
            int bm = i / BK, bk = i % BK;
            int gr = block_row + bm, gc = k0 + bk;
            As[bm][bk] = (gr < M && gc < K)
                         ? __float2half(A[gr * K + gc]) : __float2half(0.f);
        }
        // Load B tile: global float → shared half  [BK × BN]
        for (int i = tid; i < BK * BN; i += N_THREADS) {
            int bk = i / BN, bn = i % BN;
            int gr = k0 + bk, gc = block_col + bn;
            Bs[bk][bn] = (gr < K && gc < N)
                         ? __float2half(B[gr * N + gc]) : __float2half(0.f);
        }
        __syncthreads();

        // ── WMMA inner loop ───────────────────────────────────────────────────
        // For each K chunk within the strip, for each warp output tile:
        fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
        fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, row_major> b_frag;

        #pragma unroll
        for (int k = 0; k < BK; k += WMMA_K) {
            #pragma unroll
            for (int m = 0; m < TM; m++) {
                // As[warp_row + m*16 .. +16][k .. +16], row stride = BK
                load_matrix_sync(a_frag,
                    &As[warp_row + m * WMMA_M][k], BK);
                #pragma unroll
                for (int n = 0; n < TN; n++) {
                    // Bs[k .. +16][warp_col + n*16 .. +16], row stride = BN
                    load_matrix_sync(b_frag,
                        &Bs[k][warp_col + n * WMMA_N], BN);
                    mma_sync(acc[m][n], a_frag, b_frag, acc[m][n]);
                }
            }
        }
        __syncthreads();
    }

    // ── Writeback: scale by alpha, add beta*C, store ──────────────────────────
    #pragma unroll
    for (int m = 0; m < TM; m++) {
        #pragma unroll
        for (int n = 0; n < TN; n++) {
            int gr0 = block_row + warp_row + m * WMMA_M;
            int gc0 = block_col + warp_col + n * WMMA_N;
            if (gr0 >= M || gc0 >= N) continue;

            if (beta == 0.0f) {
                for (int e = 0; e < (int)acc[m][n].num_elements; e++)
                    acc[m][n].x[e] *= alpha;
                store_matrix_sync(&C[gr0 * N + gc0], acc[m][n], N, mem_row_major);
            } else {
                fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> c_frag;
                load_matrix_sync(c_frag, &C[gr0 * N + gc0], N, mem_row_major);
                for (int e = 0; e < (int)acc[m][n].num_elements; e++)
                    c_frag.x[e] = alpha * acc[m][n].x[e] + beta * c_frag.x[e];
                store_matrix_sync(&C[gr0 * N + gc0], c_frag, N, mem_row_major);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Utilities
// ═══════════════════════════════════════════════════════════════════════════════

template<typename Fn>
double time_kernel(Fn fn, int M, int N, int K, int wu, int it)
{
    for (int i = 0; i < wu; i++) fn();
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0)); CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < it; i++) fn();
    CUDA_CHECK(cudaEventRecord(t1)); CUDA_CHECK(cudaEventSynchronize(t1));
    float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0)); CUDA_CHECK(cudaEventDestroy(t1));
    return 2.0 * M * N * K * it / (ms * 1e6);
}

bool verify(const float* ref, const float* got, int sz, float tol = 0.05f)
{
    // FP16 conversion introduces more rounding, so tolerance is looser here
    for (int i = 0; i < sz; i++) {
        float d = fabsf(ref[i] - got[i]);
        if (d > tol * (fabsf(ref[i]) + 1.f)) {
            printf("  MISMATCH[%d]: ref=%.4f got=%.4f\n", i, ref[i], got[i]);
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
    printf("Device: %s  (SM_%d%d, %d SMs)\n\n",
           prop.name, prop.major, prop.minor, prop.multiProcessorCount);

    cublasHandle_t cublas;
    CUBLAS_CHECK(cublasCreate(&cublas));

    constexpr int BM = 128, BN = 128, BK = 16;
    constexpr int N_THREADS = 128;   // 4 warps

    int sizes[] = {512, 1024, 2048, 4096};
    int wu = 3, it = 10;

    printf("%-6s  %-10s  %-10s  %-12s  %s\n",
           "M=N=K","WMMA(GF)","cuBLAS(GF)","WMMA/cuBLAS","OK?");
    printf("%-6s  %-10s  %-10s  %-12s  %s\n",
           "-----","--------","----------","------------","---");

    for (int S : sizes) {
        int M=S, N=S, K=S;
        size_t bA=(size_t)M*K*4, bB=(size_t)K*N*4, bC=(size_t)M*N*4;

        float *dA,*dB,*dC_ref,*dC_cus;
        CUDA_CHECK(cudaMalloc(&dA,bA)); CUDA_CHECK(cudaMalloc(&dB,bB));
        CUDA_CHECK(cudaMalloc(&dC_ref,bC)); CUDA_CHECK(cudaMalloc(&dC_cus,bC));
        CUDA_CHECK(cudaMemset(dA,1,bA)); CUDA_CHECK(cudaMemset(dB,1,bB));
        CUDA_CHECK(cudaMemset(dC_ref,0,bC));

        float alpha=1.f, beta=0.f;

        // cuBLAS reference
        auto run_cublas = [&](){
            CUBLAS_CHECK(cublasSgemm(cublas,CUBLAS_OP_N,CUBLAS_OP_N,
                N,M,K,&alpha,dB,N,dA,K,&beta,dC_ref,N));
        };
        double gf_cublas = time_kernel(run_cublas,M,N,K,wu,it);
        float* hRef = new float[(size_t)M*N];
        CUDA_CHECK(cudaMemcpy(hRef,dC_ref,bC,cudaMemcpyDeviceToHost));

        // WMMA kernel
        dim3 block(N_THREADS);
        dim3 grid((N+BN-1)/BN, (M+BM-1)/BM);
        CUDA_CHECK(cudaMemset(dC_cus,0,bC));
        auto run_wmma = [&](){
            sgemm_wmma_kernel<BM,BN,BK><<<grid,block>>>(
                M,N,K,alpha,dA,dB,beta,dC_cus);
        };
        double gf_wmma = time_kernel(run_wmma,M,N,K,wu,it);
        float* hGot = new float[(size_t)M*N];
        CUDA_CHECK(cudaMemcpy(hGot,dC_cus,bC,cudaMemcpyDeviceToHost));
        bool ok = verify(hRef,hGot,M*N);

        printf("%-6d  %-10.0f  %-10.0f  %-12.1f%%  %s\n",
               S, gf_wmma, gf_cublas,
               gf_wmma/gf_cublas*100.0,
               ok?"✓":"✗ MISMATCH");

        delete[] hRef; delete[] hGot;
        CUDA_CHECK(cudaFree(dA)); CUDA_CHECK(cudaFree(dB));
        CUDA_CHECK(cudaFree(dC_ref)); CUDA_CHECK(cudaFree(dC_cus));
    }

    CUBLAS_CHECK(cublasDestroy(cublas));
    return 0;
}
