# PCIe Lane Verification — ADT-Link + RTX 3060 on Beelink S12 Pro

Created: 2026-05-05

---

## Overview

The **ADT-Link R3G** connects your RTX 3060 to the Beelink S12 Pro via its M.2 slot.
The M.2 slot provides PCIe lanes — but how many, and at what speed, matters significantly for GPU workloads.

This guide walks through:
1. What lane config to expect
2. How to verify after first boot
3. What the bandwidth ceiling actually means for your work
4. How to detect and debug degraded links

---

## Expected Hardware Configuration

| Component | Spec |
|---|---|
| **Host** | Beelink S12 Pro (Intel N100) |
| **M.2 Slot** | PCIe 3.0 x1 (single lane — the N100's limitation) |
| **ADT-Link R3G** | M.2 → PCIe x16 riser (electrically x1 or x4 depending on host) |
| **GPU** | RTX 3060 (expects x16 slot, works in any width) |
| **Effective link** | PCIe 3.0 **x1** |

> [!WARNING]
> The Intel N100's M.2 slot exposes **only 1 PCIe lane (x1)**. Even though the ADT-Link physically provides an x16 connector to the GPU, the actual data path is PCIe 3.0 x1.
> This is a known ceiling — understand it before benchmarking.

### What PCIe 3.0 x1 Actually Means

```
PCIe 3.0 bandwidth per lane: ~985 MB/s (unidirectional)

x1  →  ~985 MB/s  (your setup)
x4  →  ~3.9 GB/s
x8  →  ~7.9 GB/s
x16 →  ~15.8 GB/s  (desktop GPU native)
```

For reference, GDDR6 memory bandwidth on RTX 3060 is **~360 GB/s** — the GPU is
orders of magnitude faster internally. The PCIe link is the **bottleneck for
host↔GPU transfers**, not for GPU compute.

**What this means in practice:**

| Workload | Impact of x1 PCIe |
|---|---|
| CUDA kernel compute (SGEMM, DGEMM) | **None** — runs entirely on VRAM |
| Model inference (weights already loaded) | **None** |
| Bulk data transfer (upload big matrix to GPU) | **Significant** — ~10× slower than desktop |
| CUDA-Aware MPI / multi-node transfers | **Significant** |
| nvcc compile + small test runs | **None** |

> [!TIP]
> For your CUDA DGEMM learning workflow, PCIe x1 is **completely fine**. You upload the matrix once, compute repeatedly on VRAM, then download the result once. The compute time dominates by orders of magnitude.

---

## Step 1: Verify the GPU Is Detected

After connecting the ADT-Link and booting Linux:

```bash
# List all PCI devices
lspci

# Filter for NVIDIA
lspci | grep -i nvidia
```

Expected output:
```
01:00.0 VGA compatible controller: NVIDIA Corporation GA106 [GeForce RTX 3060] (rev a1)
01:00.1 Audio device: NVIDIA Corporation GA106 High Definition Audio Controller (rev a1)
```

If **nothing appears**, the GPU is not being detected — see Troubleshooting section below.

---

## Step 2: Check Link Width and Speed

This is the critical step — it tells you exactly what PCIe negotiated:

```bash
# Get full PCIe link info for the GPU
# Replace '01:00.0' with your actual PCI address from lspci above
sudo lspci -vvv -s 01:00.0 | grep -E "LnkCap|LnkSta"
```

Expected output:
```
    LnkCap: Port #0, Speed 8GT/s, Width x16, ASPM L0s L1, Exit Latency L0s <512ns, L1 <4us
    LnkSta: Speed 8GT/s (ok), Width x1 (downgraded)
```

### How to Read This

| Field | Meaning |
|---|---|
| `LnkCap` | What the GPU **supports** (hardware capability) |
| `LnkSta` | What was actually **negotiated** (real link speed) |
| `Speed 8GT/s` | PCIe Gen 3 (8 GT/s = Gen 3, 16 GT/s = Gen 4) |
| `Width x1` | One lane active |
| `(downgraded)` | GPU wanted x16, got x1 — this is **expected** with M.2 |
| `(ok)` | Running at full capability — would appear on a proper x16 slot |

> [!NOTE]
> Seeing `Width x1 (downgraded)` is **not a problem** — it's expected and correct for your setup. The GPU trained at x1 bandwidth.

---

## Step 3: Verify with nvidia-smi

Once NVIDIA drivers are installed:

```bash
nvidia-smi
```

You should see something like:
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.x    |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|                               |                      |               MIG M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0 Off |                  N/A |
|  0%   35C    P8     9W / 170W |      0MiB / 12288MiB |      0%      Default |
+-----------------------------------------------------------------------------+
```

Key things to confirm:
- GPU name shows **RTX 3060**
- Memory shows **12288 MiB** (~12 GB)
- PCI Bus ID matches what `lspci` showed (`01:00.0`)
- Temperature is reasonable (35–55°C idle)

---

## Step 4: Measure Real PCIe Bandwidth

Confirm what bandwidth you're actually getting:

```bash
# Install cuda-samples (comes with CUDA toolkit)
# Navigate to bandwidth test
cd /usr/local/cuda/samples/1_Utilities/bandwidthTest
sudo make
./bandwidthTest
```

Or build a quick standalone test:

```bash
# One-liner bandwidth test via cuda-samples (if installed)
/usr/local/cuda/extras/demo_suite/bandwidthTest
```

Expected output for PCIe 3.0 x1:

```
Host to Device Bandwidth, 1 Device(s)
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     ~900-980         ← PCIe 3.0 x1 ceiling

Device to Host Bandwidth, 1 Device(s)
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     ~900-980

Device to Device Bandwidth, 1 Device(s)
   Transfer Size (Bytes)        Bandwidth(MB/s)
   33554432                     ~300000+         ← GDDR6 internal, not PCIe
```

> [!TIP]
> H2D/D2H at **~900–980 MB/s** confirms PCIe 3.0 x1. If you see significantly less (e.g. 200 MB/s), the link may have negotiated at Gen 1 or there's a cable/connector issue.

---

## Step 5: Run a Quick Compute Sanity Check

Verify compute is fully functional regardless of PCIe width:

```bash
# If you have cuda-samples
cd /usr/local/cuda/samples/0_Introduction/vectorAdd
sudo make
./vectorAdd
# Expected: "Test PASSED"

# GFLOPS check via deviceQuery
cd /usr/local/cuda/samples/1_Utilities/deviceQuery
sudo make
./deviceQuery
```

`deviceQuery` will print full hardware specs — look for:
```
CUDA Capability Major/Minor version number: 8.6       ← Ampere (RTX 3060)
Total amount of global memory: 12288 MBytes
Number of Multiprocessors: 28
```

---

## Quick Reference Checklist

Run this in order after first boot with GPU connected:

```bash
# 1. Is the GPU visible?
lspci | grep -i nvidia

# 2. What link width negotiated?
sudo lspci -vvv -s $(lspci | grep -i nvidia | head -1 | cut -d' ' -f1) | grep -E "LnkCap|LnkSta"

# 3. Are drivers loaded?
nvidia-smi

# 4. Is CUDA working?
nvcc --version
nvidia-smi | grep "CUDA Version"

# 5. What's the actual bandwidth?
/usr/local/cuda/extras/demo_suite/bandwidthTest

# 6. Full hardware dump
/usr/local/cuda/samples/1_Utilities/deviceQuery/deviceQuery
```

---

## Troubleshooting

### GPU Not Detected (`lspci` shows nothing)

```bash
# Check if kernel sees a PCIe device at all
lspci | grep -i pci
dmesg | grep -i pcie
dmesg | grep -i nvidia
```

Common causes:
- ADT-Link not fully seated in M.2 slot → reseat it
- GPU not powered (ATX breakout board not on, PSU switch off) → check PSU
- BIOS has PCIe slot disabled → check BIOS settings
- Cable fault → try reseating the ribbon cable on both ends

### Link Negotiated at Gen 1 (2.5 GT/s) Instead of Gen 3

Symptoms: `LnkSta: Speed 2.5GT/s` or bandwidthTest shows ~250 MB/s instead of ~980 MB/s

```bash
# Force PCIe Gen 3 (try this)
sudo nvidia-smi --gpu-reset
# Or set via BIOS: look for "PCIe Speed" and set to "Gen 3" or "Auto"
```

Common causes:
- BIOS defaulting to Gen 1 for compatibility
- Driver not loading correctly → reinstall nvidia driver
- ADT-Link cable quality issue (less common)

### `nvidia-smi` Works but CUDA Fails

```bash
# Check CUDA installation
ls /usr/local/cuda
nvcc --version
echo $PATH  # should include /usr/local/cuda/bin
echo $LD_LIBRARY_PATH  # should include /usr/local/cuda/lib64
```

Fix:
```bash
echo 'export PATH=/usr/local/cuda/bin:$PATH' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

---

## What This Means for Your Benchmarks

Since you'll publish CUDA DGEMM numbers, note this in your blog post:

```
Local development environment:
  - Host: Beelink S12 Pro (Intel N100)
  - GPU: RTX 3060 12GB via ADT-Link R3G (PCIe 3.0 x1)
  - PCIe bandwidth ceiling: ~985 MB/s H2D/D2H
  - GPU compute unaffected: 28 SMs, 12 GB GDDR6 @ ~360 GB/s

Note: PCIe x1 does not affect GEMM kernel benchmarks since
all computation occurs in VRAM. Reported GFLOPS reflect true
GPU compute throughput, not host transfer throughput.
```

This preempts any reviewer question about the non-standard PCIe setup.

---

## Related Docs

- [`learning_cuda_gpu.md`](./learning_cuda_gpu.md) — CUDA kernel development (SGEMM → DGEMM)
- [`qnap_switch_setup.md`](./qnap_switch_setup.md) — Network setup for the cluster
- [`low_latency_principles_for_hpc.md`](./low_latency_principles_for_hpc.md) — HPC performance theory
