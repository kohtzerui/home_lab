# Beelink S12 Pro — RTX 3060 eGPU Setup & CUDA Benchmarking

> **Goal:** Connect the RTX 3060 via ADT-Link R3G, install CUDA 13.2 on Rocky Linux 9,
> verify end-to-end GPU functionality, and establish a cuBLAS DGEMM baseline.

---

## Hardware Setup

### Components
| Component | Model |
|-----------|-------|
| Host | Beelink S12 Pro (Intel N100, M.2 slot) |
| eGPU adapter | ADT-Link R3G (M.2 PCIe → PCIe x16) |
| GPU | NVIDIA GeForce RTX 3060 12GB (LHR) |
| External PSU | ATX PSU powering the RTX 3060 |

### Safe Connection Sequence

> [!CAUTION]
> The ADT-Link is **not hot-plug safe**. Always shut down fully before connecting or disconnecting.

```
1. sudo shutdown now          ← full power-off on Beelink
2. Wait for complete power-off
3. Connect ADT-Link to M.2 slot
4. Connect RTX 3060 PCIe power from external PSU
5. Power on PSU, then power on Beelink
```

### Display Routing

When the RTX 3060 is connected, the system routes display **through the GPU**, not the
Beelink's built-in HDMI. Plug your monitor into the **RTX 3060's HDMI or DisplayPort**.

---

## CUDA Installation (Rocky Linux 9)

### Step 1: Add NVIDIA CUDA Repo

```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo

sudo dnf clean all
```

### Step 2: Install CUDA (driver + full toolkit)

```bash
sudo dnf install -y cuda
```

> [!NOTE]
> This installs ~4.2 GB and unpacks to ~8.6 GB. Includes:
> - NVIDIA driver 595.71.05
> - CUDA Toolkit 13.2
> - cuBLAS, cuFFT, cuSolver, cuSparse, and other libraries
> - Nsight Compute + Nsight Systems profilers
> - DKMS kernel module (via `kmod-nvidia-open-dkms`)

### Step 3: Reboot

```bash
sudo reboot
```

The NVIDIA kernel module (`nvidia.ko`) is loaded at boot after install.

### Step 4: Set PATH

```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## Verification

### Driver + GPU detected

```bash
nvidia-smi
```

Expected output (RTX 3060):
```
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 595.71.05   Driver Version: 595.71.05      CUDA Version: 13.2              |
+-----------------------------------+------------------------+----------------------------+
|   0  NVIDIA GeForce RTX 3060  Off | 00000000:02:00.0  On |                        N/A |
|  0%   36C    P8    15W / 170W     |     9MiB / 12288MiB  |    0%          Default     |
+-----------------------------------+------------------------+----------------------------+
```

Key things to confirm:
- GPU appears at `02:00.0` (ADT-Link PCIe passthrough working)
- `Disp.A: On` — display is active through the GPU
- Temp ~36°C at idle (0dB fan mode — fans off until ~50°C)

### Compiler

```bash
nvcc --version
# Cuda compilation tools, release 13.2, V13.2.78
```

### PCIe detection

```bash
lspci | grep -i nvidia
# 02:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060 Lite Hash Rate] (rev a1)
# 02:00.1 Audio device: NVIDIA Corporation GA106 High Definition Audio Controller (rev a1)
```

### CUDA kernel execution (quick sanity check)

```bash
cat << 'EOF' > /tmp/cuda_test.cu
#include <stdio.h>

__global__ void hello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("=== Device: %s ===\n", prop.name);
    printf("VRAM:        %.0f MB\n", prop.totalGlobalMem / 1e6);
    printf("SM count:    %d\n", prop.multiProcessorCount);
    printf("Mem bus:     %d-bit\n", prop.memoryBusWidth);
    printf("Warp size:   %d\n", prop.warpSize);
    printf("Max threads/block: %d\n", prop.maxThreadsPerBlock);

    hello<<<1, 4>>>();
    cudaDeviceSynchronize();
    return 0;
}
EOF

nvcc -o /tmp/cuda_test /tmp/cuda_test.cu && /tmp/cuda_test
```

Expected output:
```
=== Device: NVIDIA GeForce RTX 3060 ===
VRAM:        12481 MB
SM count:    28
Mem bus:     192-bit
Warp size:   32
Max threads/block: 1024
Hello from GPU thread 0!
Hello from GPU thread 1!
Hello from GPU thread 2!
Hello from GPU thread 3!
```

---

## Baseline Benchmark — cuBLAS DGEMM

### What it measures

Double-precision (FP64) matrix multiply: **C = A × B**, where A, B, C are N×N matrices.
This is the core operation in HPL and scientific computing benchmarks.

### Benchmark code

```cuda
// cublas_dgemm_bench.cu
#include <stdio.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main() {
    int N = 4096;
    size_t bytes = (size_t)N * N * sizeof(double);
    double *A, *B, *C;
    cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);
    cudaMemset(A, 1, bytes); cudaMemset(B, 1, bytes); cudaMemset(C, 0, bytes);

    cublasHandle_t handle;
    cublasCreate(&handle);
    double alpha = 1.0, beta = 0.0;

    // Warmup
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    cudaDeviceSynchronize();

    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < 10; i++)
        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    double gflops = (2.0 * N * N * N * 10) / (ms / 1000.0) / 1e9;
    printf("N=%d | %.2f ms/iter | %.2f GFLOPS\n", N, ms / 10, gflops);

    cublasDestroy(handle);
    cudaFree(A); cudaFree(B); cudaFree(C);
}
```

```bash
nvcc -o cublas_dgemm cublas_dgemm_bench.cu -lcublas && ./cublas_dgemm
```

### Results (2026-05-10)

| Matrix Size | Time/iter | GFLOPS | % of FP64 Peak |
|-------------|-----------|--------|----------------|
| N=4096 | 744.27 ms | **184.66** | **~92.8%** |

### Context

| Metric | Value |
|--------|-------|
| RTX 3060 FP64 theoretical peak | ~199 GFLOPS |
| cuBLAS efficiency achieved | 92.8% |
| RTX 3060 FP32 theoretical peak | 12,740 GFLOPS |
| FP64:FP32 ratio (consumer GPU) | 1:64 (hardware-limited) |

> [!NOTE]
> The RTX 3060 is a consumer GPU intentionally crippled for FP64. 92.8% cuBLAS efficiency
> means the software is nearly optimal — the ceiling is the hardware. For serious FP64 HPC
> numbers, run the final benchmark on a cloud A100 (~9.7 TFLOPS FP64).

---

## SGEMM Optimization Journey (2026-05-12)

### Goal
Build a hand-written CUDA SGEMM kernel from scratch, progressively optimizing it
toward cuBLAS, to understand GPU memory hierarchy and tensor core usage.

### Environment
| | |
|---|---|
| GPU | RTX 3060 (SM_86 Ampere, 28 SMs, 12GB GDDR6) |
| Driver | 595.71.05 |
| CUDA | 13.2 |
| Host | Beelink S12 Pro via ADT-Link eGPU |

### Thermal validation (before benchmarking)
```
nvidia-smi dmon -s pcut -d 1 &
while true; do ./sgemm_bench; done
```
Observations under sustained load:
- **Power draw:** 137–142 W (81–84% of 170 W TDP) — no power throttle
- **Temperature:** peaks at **70°C** (13°C headroom before 83°C throttle)
- **SM clock:** 1807–1830 MHz — above the 1777 MHz boost spec
- **Result:** eGPU chain fully healthy, fans confirmed working

### Benchmark results — all sizes FP32 (GFLOPS)

| M=N=K | Naive | Register | +cp.async | +WMMA | cuBLAS |
|-------|------:|--------:|---------:|------:|-------:|
| 512 | 785 | 1,812 | 2,208 | 1,248 | ~5,500 |
| 1024 | 800 | 3,166 | 3,613 | **4,298** | ~7,400 |
| 2048 | — | 4,374 | 4,781 | 4,108 | ~8,065 |
| 4096 | — | 4,852 | 5,236 | **5,091** | ~8,100 |

> [!NOTE]
> RTX 3060 FP32 theoretical peak is ~12.74 TFLOPS. cuBLAS reaches ~64% of peak;
> our best hand-written kernel reaches ~65% of cuBLAS (41% of peak).

### Optimization techniques applied

#### Level 0 — Naive (785 GFLOPS @ 4K)
One thread per output element. Every MAC reads directly from global memory.
Arithmetic intensity ≈ 2 FLOPs / 2 floats loaded — fully memory-bound.

```cuda
// Thread (row, col) computes one C element
for (int k = 0; k < K; ++k)
    acc += A[row*K+k] * B[k*N+col];   // two global loads per iteration
```

#### Level 1 — Register-tiled SGEMM (4,852 GFLOPS @ 4K — **6.2× speedup**)
Thread block loads BM×BK (A) and BK×BN (B) tiles into shared memory.
Each thread then computes a TM×TN = 8×8 register output tile.

- Global memory traffic ↓ by factor BK (reuse across the tile)
- FP32 FMAs operate purely on registers → maximum throughput
- Block: 128×128 tile, 256 threads, BK=16

```cuda
// Each thread accumulates an 8×8 output block
for (int k = 0; k < BK; ++k) {
    for (int m = 0; m < TM; ++m) a_frag[m] = As[k][t_row*TM+m];
    for (int n = 0; n < TN; ++n) b_frag[n] = Bs[k][t_col*TN+n];
    for (int m = 0; m < TM; ++m)
        for (int n = 0; n < TN; ++n)
            acc[m][n] += a_frag[m] * b_frag[n];   // pure register ops
}
```

#### Level 2 — cp.async double buffering (+5–22% over register kernel)
Ampere's dedicated DMA engine copies GMEM→SMEM without stalling the warp.
Ping-pong shared memory buffers let tile[k+1] load while tile[k] computes.

```cuda
// PTX intrinsics (no extra headers needed)
void cp_async4(void* smem, const void* gmem);   // fire-and-forget copy
void cp_async_commit();                          // seal this batch as a group
void cp_async_wait<N>();                         // stall until ≤N groups in-flight

// Pattern: issue next load, then wait for current, then compute
issue_loads(strip+1, next_buf);   // DMA to other buffer
cp_async_commit();
cp_async_wait<1>();               // wait for strip, keep strip+1 flying
__syncthreads();
compute(curr_buf);                // FMAs run while DMA loads next strip
```

Gain is largest at small sizes (22% at 512) where load latency is proportionally
higher. At 4K the FMA pipeline already hides some latency on its own.

#### Level 3 — WMMA Tensor Cores (5,091 GFLOPS @ 4K)
Ampere Tensor Cores execute a 16×16×16 FP16→FP32 matrix multiply per warp
per clock — 8,192 FLOPs vs ~32 FLOPs for scalar FMAs. Inputs must be FP16;
our kernel converts float→half during the shared memory load phase.

```cuda
// Warp-level, all 32 threads participate collectively
fragment<matrix_a, 16,16,16, half, row_major> a_frag;
fragment<matrix_b, 16,16,16, half, row_major> b_frag;
fragment<accumulator, 16,16,16, float>         c_frag;

load_matrix_sync(a_frag, &As[warp_row+m*16][k], BK);
load_matrix_sync(b_frag, &Bs[k][warp_col+n*16], BN);
mma_sync(c_frag, a_frag, b_frag, c_frag);   // 8192 FLOPs per warp
```

WMMA anomalies observed:
- **512: slower than async** — tensor core warp sync overhead dominates small tiles
- **2048: slightly slower than async** — 16 fp32 accumulator fragments per warp
  (~128 registers) reduces occupancy, starving the pipeline

### Remaining gap to cuBLAS
cuBLAS combines ALL techniques simultaneously plus:
- Larger tiles (256×128 or larger)
- cp.async AND WMMA in the same kernel (our kernels use one or the other)
- Shared memory padding to eliminate bank conflicts
- FP16 inputs directly (no float→half conversion overhead per strip)
- Hand-written SASS for sm_86 with architecture-specific instruction scheduling

### Source files
| File | Description |
|------|-------------|
| `~/sgemm_bench.cu` | cuBLAS SGEMM sweep across sizes |
| `~/sgemm_kernels.cu` | Naive + register-tiled kernels vs cuBLAS |
| `~/sgemm_async.cu` | Sync vs cp.async double-buffered kernel |
| `~/sgemm_wmma.cu` | Tensor Core WMMA kernel vs cuBLAS |

---

## Recommendations & Next Steps

### 1. Fix ethernet autoconnect (already done)
```bash
sudo nmcli connection modify enp1s0 connection.autoconnect yes
```
After reboot, the Beelink may get a new DHCP IP — check via the physical screen or router.
Consider assigning a static IP in your router's DHCP reservation table for the Beelink's MAC address.

### 2. Console font (optional — for physical display readability)
```bash
sudo dnf install -y terminus-fonts-console
# Edit /etc/vconsole.conf: FONT=ter-v32b
sudo systemctl restart systemd-vconsole-setup
```

### 3. Write a custom CUDA SGEMM kernel
Follow the progression in `learning/theory/learning_cuda_gpu.md`:
- **Phase 1** ✅ Done — setup, nvcc, nvidia-smi, CUDA hello world
- **Phase 2** ✅ Done — Naive SGEMM (785 GFLOPS @ 4K)
- **Phase 3** ✅ Done — Register-tiled SGEMM (4,852 GFLOPS @ 4K — 6.2× naive)
- **Phase 4** ✅ Done — cp.async double buffering (5,236 GFLOPS) + WMMA tensor cores (5,091 GFLOPS)
- **Phase 5** — Combine WMMA + cp.async in one kernel (target: 80–90% of cuBLAS)
- **Phase 6** — Port to DGEMM (FP64) for HPL-equivalent benchmark

### 4. Cloud benchmark run (~$10–20)
Once your custom kernel hits 60–80% of cuBLAS locally, rent an A100 on RunPod/Vast.ai
for a final credible portfolio benchmark run. See theory doc for workflow.

### 5. Install pciutils (useful for future debugging)
```bash
sudo dnf install -y pciutils
```

---

## Known Issues & Gotchas

| Issue | Cause | Fix |
|-------|-------|-----|
| No display on Beelink HDMI | RTX 3060 takes over display routing | Use RTX 3060's HDMI/DP port |
| GRUB shell on boot | Keypresses during GRUB timeout (c = GRUB CLI) | Power cycle; don't touch keyboard during boot |
| SSH timeout after reboot | DHCP assigns new IP; `enp1s0` didn't autoconnect | `sudo nmcli device connect enp1s0`; check new IP |
| `clockRate`/`memoryClockRate` missing | Removed from CUDA 13.2 `cudaDeviceProp` | Use `nvidia-smi` for clock info instead |
| RTX 3060 fans not spinning | 0dB fan mode (idle below ~50°C) | Normal — fans spin under GPU load |
