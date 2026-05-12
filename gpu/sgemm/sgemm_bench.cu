// sgemm_bench.cu
// Benchmarks cuBLAS SGEMM across several matrix sizes.
// Measures peak single-precision TFLOPS on the attached GPU.
//
// Build:  nvcc -O3 -o sgemm_bench sgemm_bench.cu -lcublas
// Run:    ./sgemm_bench

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cublas_v2.h>

// ── helpers ──────────────────────────────────────────────────────────────────

#define CUDA_CHECK(err)                                                         \
    do {                                                                        \
        cudaError_t _e = (err);                                                 \
        if (_e != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s:%d  %s\n",                          \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(err)                                                       \
    do {                                                                        \
        cublasStatus_t _s = (err);                                              \
        if (_s != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error %s:%d  code=%d\n",                   \
                    __FILE__, __LINE__, (int)_s);                               \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ── benchmark one size ────────────────────────────────────────────────────────

// Computes C = α·A·B + β·C  where A[M×K], B[K×N], C[M×N]
// Returns achieved GFLOPS.
double bench_sgemm(cublasHandle_t handle,
                   int M, int N, int K,
                   int warmup_iters, int bench_iters)
{
    float *dA, *dB, *dC;
    size_t bytesA = (size_t)M * K * sizeof(float);
    size_t bytesB = (size_t)K * N * sizeof(float);
    size_t bytesC = (size_t)M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc(&dA, bytesA));
    CUDA_CHECK(cudaMalloc(&dB, bytesB));
    CUDA_CHECK(cudaMalloc(&dC, bytesC));

    // initialise with random-ish values so the compiler can't optimise away
    CUDA_CHECK(cudaMemset(dA, 1, bytesA));
    CUDA_CHECK(cudaMemset(dB, 1, bytesB));
    CUDA_CHECK(cudaMemset(dC, 0, bytesC));

    const float alpha = 1.0f, beta = 0.0f;

    // cuBLAS uses column-major, so we pass B first to compute C = A·B
    // (equivalent to the standard row-major A·B)
    auto run = [&]() {
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,
                                 N, M, K,
                                 &alpha,
                                 dB, N,
                                 dA, K,
                                 &beta,
                                 dC, N));
    };

    // warm-up
    for (int i = 0; i < warmup_iters; ++i) run();
    CUDA_CHECK(cudaDeviceSynchronize());

    // timed run
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));

    CUDA_CHECK(cudaEventRecord(t0));
    for (int i = 0; i < bench_iters; ++i) run();
    CUDA_CHECK(cudaEventRecord(t1));
    CUDA_CHECK(cudaEventSynchronize(t1));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));

    // FLOP count: 2·M·N·K per GEMM (one multiply + one add per element)
    double flops_per_iter = 2.0 * (double)M * (double)N * (double)K;
    double total_flops    = flops_per_iter * bench_iters;
    double gflops         = total_flops / (ms * 1e6);   // ms→s: /1e3, G: /1e9

    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));

    return gflops;
}

// ── main ──────────────────────────────────────────────────────────────────────

int main()
{
    // Print device info
    int dev = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    printf("==========================================================\n");
    printf("  Device : %s\n", prop.name);
    printf("  SM     : %d x SM_%d%d\n",
           prop.multiProcessorCount,
           prop.major, prop.minor);
    printf("  Mem    : %.0f MB  |  Mem bus: %d-bit\n",
           prop.totalGlobalMem / 1e6,
           prop.memoryBusWidth);
    printf("==========================================================\n\n");

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // sizes to sweep (square matrices for simplicity)
    int sizes[] = {512, 1024, 2048, 4096, 8192};
    int warmup  = 5;
    int iters   = 20;

    printf("%-8s  %-10s  %-12s  %s\n",
           "M=N=K", "time/iter", "GFLOPS", "vs RTX3060 peak (12.7 TFLOPS)");
    printf("%-8s  %-10s  %-12s  %s\n",
           "------", "---------", "------", "-----------------------------");

    double peak_tflops = 12.74;   // RTX 3060 FP32 spec

    for (int s : sizes) {
        double gflops = bench_sgemm(handle, s, s, s, warmup, iters);
        double pct    = gflops / (peak_tflops * 1000.0) * 100.0;
        // estimate ms per iter from gflops
        double flops  = 2.0 * s * s * s;
        double ms_est = flops / (gflops * 1e6);
        printf("%-8d  %-10.3f ms  %-12.1f  %.1f%%\n",
               s, ms_est, gflops, pct);
    }

    printf("\n");
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
