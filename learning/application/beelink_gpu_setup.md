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

### 3. Write a custom CUDA DGEMM kernel
Follow the progression in `learning/theory/learning_cuda_gpu.md`:
- **Phase 1** ✅ Done — setup, nvcc, nvidia-smi, CUDA hello world
- **Phase 2** — Implement naive SGEMM (one thread per output element)
- **Phase 3** — Tiled SGEMM with shared memory (target: 30–50% of cuBLAS)
- **Phase 4** — Register-tiled SGEMM (target: 60–80% of cuBLAS)
- **Phase 5** — Port to DGEMM (FP64)

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
