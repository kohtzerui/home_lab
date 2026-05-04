# CUDA Learning Guide — From Zero to DGEMM

## Overview

**CUDA** is NVIDIA's programming model for writing code that runs on their GPUs. You write a special function (a "kernel") in C/C++ that executes on thousands of GPU threads simultaneously.

Your goal: implement a **DGEMM kernel** (double-precision matrix multiply) from scratch, optimize it step by step, benchmark it against cuBLAS (NVIDIA's optimized library), and write a blog post about the results. This is the single most important missing signal on your resume for both Google and NVIDIA roles.

---

## Hardware Strategy: Local Dev + Cloud Benchmarks

You have two distinct needs that are best served by different hardware:

| Need | Best Tool | Why |
|---|---|---|
| Daily CUDA learning & kernel iteration | Local GPU (RTX 3060 via eGPU / SFF desktop) | Fast compile → run loop, no per-minute cost, works offline |
| Final portfolio benchmark numbers | Cloud GPU (RunPod / Vast.ai A100) | Impressive hardware name, clean results, no PCIe bottleneck |

### Why Cloud for Portfolio Numbers?

A recruiter at Google or NVIDIA reading your blog post will notice the GPU name:

- *"1.2 TFLOPS on RTX 3060"* → looks like a gaming experiment
- *"18.7 TFLOPS on A100-SXM4-80GB, 94% of theoretical peak"* → looks like real HPC work

Beyond branding:
- **A100 has full FP64** (~9.7 TFLOPS) — your DGEMM numbers will actually be meaningful
- **Multi-GPU experiments** are possible on cloud (2×A100, 4×A100) for distributed CUDA demos
- **No PCIe bottleneck** — if you're using the M.2 eGPU adapter, your local results have an asterisk

### Cost

~$10–20 of RunPod credit is enough for a full HPL + DGEMM benchmark session on an A100 that you publish in your portfolio. You only need to do this **once**, after your kernel is fully debugged locally.

```
Workflow:
  1. Write + debug CUDA kernel locally on RTX 3060 (free, fast)
  2. Achieve target efficiency (e.g. 70% of cuBLAS locally)
  3. Spin up RunPod A100 (~$0.50/hr), run final benchmarks
  4. Record clean results, shut down the instance
  5. Blog post uses the A100 numbers
```

> [!TIP]
> Use **RunPod** or **Vast.ai** for GPU rental. Select an instance with an **A100 SXM4 80GB** or **A100 PCIe 40GB** for HPC-relevant results. Always verify `nvidia-smi` and `nvcc --version` before starting your timed benchmark run.

---

## Prerequisites

### Hardware
- NVIDIA GPU (your planned RTX 3060 12GB via ADT-Link R3G + external PSU)
- Any x86 host with an M.2 slot (your Beelink S12 Pro)

### Software
- **Linux** (Ubuntu 22.04 or 24.04 recommended — CUDA on Windows works but is worse for HPC dev)
- **NVIDIA Driver** — `nvidia-driver-535` or newer
- **CUDA Toolkit** — version 12.x from [NVIDIA's repo](https://developer.nvidia.com/cuda-downloads)
- **Nsight Compute** — comes with CUDA Toolkit (GPU profiler)

### Installation
```bash
# Add NVIDIA repo (Ubuntu)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update

# Install
sudo apt install cuda-toolkit-12-4 nvidia-driver-535

# Verify
nvcc --version
nvidia-smi
```

---

## Part 1: The GPU Hardware Model

### Why GPUs Exist for This

You already understand FPGA parallelism (spatial — you lay out circuits in 2D fabric). GPUs are a different flavor:

| | FPGA (your KV260) | GPU (RTX 3060) |
|---|---|---|
| **Parallelism** | Spatial — circuits in fabric | Temporal — thousands of threads share execution units |
| **Programmability** | Configure once, runs forever | Launch new kernels any time |
| **Memory** | BRAM/URAM (small, fast) + DDR | Registers → Shared Mem → L2 → VRAM |
| **FP64 throughput** | ~30–50 GFLOPS (DSP-limited) | ~200 GFLOPS (artificially limited on consumer) |
| **FP32 throughput** | ~100–200 GFLOPS | ~12,700 GFLOPS |

### GPU Architecture

```
RTX 3060 (GA106)
├── Streaming Multiprocessor (SM) #0
│   ├── 128 CUDA Cores (FP32 ALUs)
│   ├── 2 FP64 Units              ← why FP64 is 1/64th of FP32 on consumer GPUs
│   ├── 4 Tensor Cores             ← for AI (FP16/INT8 matrix ops)
│   ├── 128 KB Shared Memory / L1 Cache (configurable split)
│   ├── 65,536 Registers (32-bit each)
│   └── Warp Schedulers (issue 32 threads at once)
├── SM #1
├── SM #2
│   ...
├── SM #27  (28 SMs on RTX 3060)
├── L2 Cache (3 MB)
└── VRAM — 12 GB GDDR6 (Global Memory)
    └── Bandwidth: ~360 GB/s
```

**Key insight:** A GPU is not one fast processor. It's dozens of **Streaming Multiprocessors (SMs)**, each running hundreds of threads simultaneously. The trick is keeping all of them busy and fed with data.

### Memory Hierarchy (Critical to Understand)

```
Speed:     FASTEST ──────────────────────────────► SLOWEST
Capacity:  SMALLEST ─────────────────────────────► LARGEST

┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌─────────────┐
│ Registers │  │ Shared Memory│  │ L2 Cache │  │ Global Mem  │
│           │  │ (per block)  │  │ (shared) │  │ (VRAM)      │
│ ~20 TB/s  │  │ ~2-5 TB/s    │  │ ~1 TB/s  │  │ ~0.36 TB/s  │
│ 255 regs  │  │ 48-100 KB    │  │ 3 MB     │  │ 12 GB       │
│ per thread│  │ per SM       │  │          │  │             │
└──────────┘  └──────────────┘  └──────────┘  └─────────────┘
```

**The entire optimization challenge is moving data up this hierarchy.** Your naive kernel reads from global memory (slow). Your optimized kernel reads from shared memory (fast) and registers (fastest).

---

## Part 2: CUDA Programming Basics

### The Thread Hierarchy

```
Grid (all threads for one kernel launch)
├── Block (0,0)          ← runs on one SM
│   ├── Warp 0 (threads 0–31)    ← 32 threads execute same instruction (SIMT)
│   ├── Warp 1 (threads 32–63)
│   └── ...
├── Block (0,1)          ← runs on another SM
│   └── ...
├── Block (1,0)
│   └── ...
└── ...
```

- **Thread** — single execution context, has its own registers
- **Warp** — 32 threads executing the SAME instruction simultaneously (like SIMD)
- **Block** — group of threads that can share fast memory and synchronize
- **Grid** — all blocks for the whole kernel launch

### Your First CUDA Program

```cuda
// hello_cuda.cu
#include <stdio.h>

// This function runs on the GPU
__global__ void hello_kernel() {
    int threadId = blockIdx.x * blockDim.x + threadIdx.x;
    printf("Hello from GPU thread %d!\n", threadId);
}

int main() {
    // Launch 2 blocks of 4 threads each = 8 threads total
    hello_kernel<<<2, 4>>>();

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    return 0;
}
```

```bash
nvcc hello_cuda.cu -o hello_cuda
./hello_cuda
```

### Key CUDA Concepts

| Concept | What It Means | FPGA Analogy |
|---|---|---|
| `__global__` | Function runs on GPU, called from CPU | Your accelerator's top-level module |
| `<<<grid, block>>>` | Launch configuration (how many threads) | N/A — your hardware is fixed at synthesis time |
| `threadIdx.x` | Thread's index within its block | PE index in systolic array |
| `blockIdx.x` | Block's index within the grid | Which tile you're computing |
| `__shared__` | Fast on-chip memory shared by block | BRAM in your FPGA design |
| `__syncthreads()` | Barrier — wait for all threads in block | Pipeline stall / handshake |
| `cudaMemcpy()` | Transfer data CPU ↔ GPU | AXI DMA transfer PS ↔ PL |

---

## Part 3: Matrix Multiply — Naive to Optimized

### The Math

```
C[i][j] = Σ(k=0 to K-1) A[i][k] × B[k][j]
```

For N×N matrices: N³ multiply-adds = O(N³) operations.
For N=4096: ~137 billion FLOPs. A GPU can do this in milliseconds.

---

### Level 1: Naive Kernel (~2–5% of Peak)

Each thread computes ONE element of C:

```cuda
// Naive SGEMM — one thread per output element
// This is correct but terribly slow
__global__ void sgemm_naive(
    const float *A,    // M × K matrix (row-major in memory)
    const float *B,    // K × N matrix
    float *C,          // M × N matrix
    int M, int N, int K
) {
    // Which element of C am I computing?
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // Every iteration: 2 global memory reads (SLOW!)
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Launch
dim3 block(16, 16);                      // 256 threads per block
dim3 grid((N + 15) / 16, (M + 15) / 16); // enough blocks to cover C
sgemm_naive<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
```

#### Why This Is Slow

Every thread reads an entire row of A and column of B from global memory:
- Each thread: 2 × K reads from VRAM
- Total threads: M × N
- Total memory traffic: absurdly redundant — neighboring threads re-read the same data

**The kernel is memory-bound** — the GPU's compute units sit idle waiting for data.

---

### Level 2: Tiled with Shared Memory (~30–50% of Peak)

**The fix:** Threads cooperate to load small tiles into fast shared memory, then everyone reads from there.

```cuda
#define TILE_SIZE 32

__global__ void sgemm_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    // ---- Shared memory tiles (fast, on-chip) ----
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // ---- Slide tile window across K dimension ----
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {

        // Step 1: Each thread loads ONE element into shared memory
        // 32×32 = 1024 threads cooperate to load a 32×32 tile
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;

        // Step 2: Wait for ALL threads to finish loading
        __syncthreads();

        // Step 3: Compute partial sum from shared memory
        // These reads are ~300× faster than global memory!
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }

        // Step 4: Wait before overwriting tiles
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

#### Why It's Faster

Each float from A and B is loaded from global memory ONCE, then reused by 32 threads. Memory traffic drops ~32×.

```
Visually:

Matrix A               Matrix B               Matrix C
┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐     ┌───┬───┬───┬───┐
│   │   │   │   │     │░░░│   │   │   │     │   │   │   │   │
│░░░│░░░│   │   │←A   │░░░│   │   │   │     │   │   │   │   │
│░░░│░░░│   │   │tile │░░░│   │   │   │     │░░░│   │   │   │←output
├───┼───┼───┼───┤     ├───┼───┼───┼───┤     │░░░│   │   │   │ block
│   │   │   │   │     │   │   │   │   │     ├───┼───┼───┼───┤
└───┴───┴───┴───┘     └───┴───┴───┴───┘     └───┴───┴───┴───┘
                           ↑ B tile
```

---

### Level 3: Register Tiling (~60–80% of Peak)

Each thread computes a small **submatrix** (e.g., 8×8) of C, keeping intermediate values in registers (fastest memory on the GPU).

```cuda
#define BM 128      // Block tile height
#define BN 128      // Block tile width
#define BK 8        // Block tile depth
#define TM 8        // Thread tile height
#define TN 8        // Thread tile width

__global__ void sgemm_register_tiled(
    const float *A, const float *B, float *C,
    int M, int N, int K
) {
    __shared__ float smemA[BM][BK];  // 128 × 8
    __shared__ float smemB[BK][BN];  // 8 × 128

    // Each thread accumulates an 8×8 submatrix in registers
    float regC[TM][TN] = {0.0f};   // 64 registers per thread
    float regA[TM];
    float regB[TN];

    int tx = threadIdx.x;  // 0..15
    int ty = threadIdx.y;  // 0..15

    int c_row = blockIdx.y * BM + ty * TM;
    int c_col = blockIdx.x * BN + tx * TN;

    for (int bk = 0; bk < K; bk += BK) {
        // Cooperatively load tiles into shared memory
        // (each thread loads multiple elements to fill the larger tiles)
        // ... loading code ...

        __syncthreads();

        for (int k = 0; k < BK; k++) {
            // Load into registers
            for (int i = 0; i < TM; i++)
                regA[i] = smemA[ty * TM + i][k];
            for (int j = 0; j < TN; j++)
                regB[j] = smemB[k][tx * TN + j];

            // Outer product: 8×8 = 64 FMAs, ALL from registers
            for (int i = 0; i < TM; i++)
                for (int j = 0; j < TN; j++)
                    regC[i][j] += regA[i] * regB[j];
        }

        __syncthreads();
    }

    // Write 8×8 result to global memory
    for (int i = 0; i < TM; i++)
        for (int j = 0; j < TN; j++)
            if (c_row + i < M && c_col + j < N)
                C[(c_row + i) * N + (c_col + j)] = regC[i][j];
}
```

**Why registers matter:** The inner loop (64 FMAs) reads entirely from registers at ~20 TB/s — the compute units are finally fully fed.

---

### Level 4: DGEMM (Double-Precision)

Structurally identical to SGEMM, but with `double` instead of `float`:

```cuda
#define TILE_SIZE 16    // Smaller — double uses 2× memory

__global__ void dgemm_tiled(
    const double *A, const double *B, double *C,
    int M, int N, int K
) {
    __shared__ double tileA[TILE_SIZE][TILE_SIZE];  // 16×16×8B = 2 KB
    __shared__ double tileB[TILE_SIZE][TILE_SIZE];  // 16×16×8B = 2 KB

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    double sum = 0.0;

    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int a_col = t * TILE_SIZE + threadIdx.x;
        int b_row = t * TILE_SIZE + threadIdx.y;

        tileA[threadIdx.y][threadIdx.x] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0;
        tileB[threadIdx.y][threadIdx.x] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_SIZE; k++)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}
```

Key differences from SGEMM:
- **TILE_SIZE is smaller** (16 vs 32) — doubles use 2× shared memory
- **Throughput is much lower on consumer GPUs** — RTX 3060 has only 2 FP64 units per SM (vs 128 FP32)
- Everything else is structurally identical

> [!WARNING]
> **Consumer GPUs are intentionally crippled for FP64.** RTX 3060 does ~12,700 GFLOPS FP32 but only ~200 GFLOPS FP64 (1/64 ratio). For HPL you need FP64. This is fine for learning — the optimization techniques are identical. The resume line will say "implemented CUDA DGEMM achieving X% of cuBLAS" and that's credible regardless of whether it's on an RTX or an A100.

---

## Part 4: Profiling with Nsight Compute

Once you write your kernel, you measure it:

```bash
# Compile
nvcc -O3 -arch=sm_86 sgemm.cu -o sgemm   # sm_86 for RTX 3060

# Profile
ncu --set full ./sgemm

# Key metrics to look at:
#   SM Throughput (%)       → are your SMs busy?
#   Memory Throughput (%)   → are you bandwidth-limited?
#   Achieved Occupancy      → enough warps to hide latency?
#   FLOP Efficiency (%)     → your kernel vs theoretical peak
```

### What the Metrics Tell You

| Metric | Low Value Means | Fix |
|---|---|---|
| SM Throughput | Compute units idle | Increase occupancy (more threads) |
| Memory Throughput | Memory pipe unused | Prefetch, coalesce accesses |
| Occupancy | Not enough warps to hide latency | Reduce register/shared mem usage |
| FLOP Efficiency | Far from peak | Better tiling, register blocking |

**Goal:** Get as close to cuBLAS as possible. A well-written tiled kernel hits 60–80%. Above 90% requires double buffering, vectorized loads (`float4`), bank-conflict avoidance.

---

## Part 5: Connecting to Your FPGA Work

Since you already understand the FPGA systolic array, here's how concepts map:

| FPGA Concept | CUDA Equivalent |
|---|---|
| BRAM tile buffers | `__shared__` memory tiles |
| Systolic array PEs | Thread block doing FMAs |
| AXI DMA from DDR | `cudaMemcpy()` from host |
| DSP48E2 multiply-accumulate | CUDA core FMA instruction |
| Pipeline II=1 (HLS) | Warp executing one instruction per cycle |
| UNROLL factor=8 (HLS) | 8 threads each doing one multiply |
| DATAFLOW (HLS) | CUDA streams (overlapping compute + transfer) |

**The mental model transfers directly.** Your tile buffers in BRAM are analogous to shared memory tiles. Your systolic PEs are analogous to threads doing FMAs. Your DMA from DDR is analogous to global memory loads.

---

## Part 6: Advanced Topics (After You Have the Basics)

### CUDA Streams (Overlap Compute + Transfer)
```cuda
cudaStream_t stream1, stream2;
cudaStreamCreate(&stream1);
cudaStreamCreate(&stream2);

// While GPU computes tile 1, DMA transfers tile 2
cudaMemcpyAsync(d_A2, h_A2, size, cudaMemcpyHostToDevice, stream2);
kernel<<<grid, block, 0, stream1>>>(d_A1, d_B1, d_C1);
```
This is like your FPGA's DATAFLOW pragma — overlap data movement with computation.

### cuBLAS (The Reference to Beat)
```cuda
#include <cublas_v2.h>

cublasHandle_t handle;
cublasCreate(&handle);

double alpha = 1.0, beta = 0.0;
cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha, d_B, N, d_A, K, &beta, d_C, N);
```
Always benchmark your kernel against cuBLAS — it's what NVIDIA's own engineers optimized.

### Bank Conflicts in Shared Memory
Shared memory is divided into 32 banks. If multiple threads in a warp access the same bank, accesses serialize. Fix: pad shared memory arrays.
```cuda
__shared__ float tileA[TILE_SIZE][TILE_SIZE + 1];  // +1 padding avoids conflicts
```

### Vectorized Memory Access (float4 / double2)
Load multiple values per transaction to maximize memory bandwidth:
```cuda
float4 val = reinterpret_cast<float4*>(&A[row * K + col])[0];
// Loads 4 floats in one 128-bit transaction instead of 4 separate 32-bit loads
```

---

## Complete Learning Path (Ordered)

### Phase 1: Setup & Hello World (2–3 days)
- [ ] Install Ubuntu on Beelink (dual-boot or replace Windows)
- [ ] Connect RTX 3060 via ADT-Link R3G + external PSU
- [ ] Install NVIDIA driver + CUDA toolkit
- [ ] `nvidia-smi` should show the GPU
- [ ] Compile and run hello_cuda.cu
- [ ] Write a simple vector_add kernel — verify against CPU

### Phase 2: Naive SGEMM (2–3 days)
- [ ] Implement `sgemm_naive` — one thread per C element
- [ ] Write a CPU reference `sgemm_cpu` for correctness checking
- [ ] Time it: `cudaEventRecord` before and after
- [ ] Calculate GFLOPS: `2.0 * M * N * K / (time_ms * 1e6)`
- [ ] Compare against cuBLAS `cublasSgemm` — you should be at ~2–5%

### Phase 3: Tiled SGEMM (3–5 days)
- [ ] Implement `sgemm_tiled` with shared memory
- [ ] Start with TILE_SIZE=16, then try 32
- [ ] Verify correctness
- [ ] Profile with `ncu` — check memory throughput and SM utilization
- [ ] Target: 30–50% of cuBLAS

### Phase 4: Register-Tiled SGEMM (5–7 days)
- [ ] Implement register tiling — each thread computes TM×TN submatrix
- [ ] Experiment with BM, BN, BK, TM, TN
- [ ] Add shared memory padding to avoid bank conflicts
- [ ] Profile and iterate
- [ ] Target: 60–80% of cuBLAS

### Phase 5: DGEMM (3–5 days)
- [ ] Port your best SGEMM kernel to FP64
- [ ] Adjust tile sizes (halve them — doubles use 2× memory)
- [ ] Benchmark against `cublasDgemm`
- [ ] Note: on RTX 3060, FP64 will be slow — that's expected and fine

### Phase 6: Cloud Benchmark Run (~$10–20, 1–2 hrs)
- [ ] Spin up a RunPod / Vast.ai A100 instance
- [ ] Upload your final kernel (the one hitting 60–80% of cuBLAS locally)
- [ ] Run HPL + your DGEMM kernel + cuBLAS reference benchmark
- [ ] Record: GFLOPS achieved, % of cuBLAS, % of A100 FP64 peak
- [ ] Screenshot Nsight Compute session on the A100
- [ ] Shut down the instance — keep the results

### Phase 7: Analysis & Blog Post (3–5 days)
- [ ] Create performance charts: GFLOPS vs matrix size for each kernel version (use A100 numbers)
- [ ] Show % of cuBLAS at each optimization level
- [ ] Profile screenshots from Nsight Compute (both local dev shots and final A100 shots)
- [ ] Compare FPGA (KV260) vs GPU (A100) for DGEMM — this is a compelling side-by-side
- [ ] Write blog post: "FPGA vs GPU for Dense Linear Algebra — A Practical Comparison"
- [ ] Include a section on your two-machine workflow (local RTX 3060 dev → A100 benchmarks)

---

## Resources

| Resource | What It Is |
|---|---|
| [Simon Boehm's SGEMM blog](https://siboehm.com/articles/22/CUDA-MMM) | Best practical walkthrough on the internet — follow this |
| [NVIDIA CUTLASS](https://github.com/NVIDIA/cutlass) | Open-source reference GEMM implementations |
| [Programming Massively Parallel Processors (Hwu/Kirk)](https://www.elsevier.com/books/programming-massively-parallel-processors/hwu/978-0-323-91231-0) | The textbook for GPU programming |
| [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/) | Official reference (chapters 1–5 to start) |
| [Nsight Compute docs](https://docs.nvidia.com/nsight-compute/) | Profiling guide |
| [Lei Mao's CUDA blog](https://leimao.github.io/) | Great intermediate CUDA content |

> [!TIP]
> **Start with Simon Boehm's blog.** It walks through exactly this same progression (naive → tiled → register-tiled) with code you can compile and benchmark immediately. It's the single best resource for learning CUDA GEMM optimization.
