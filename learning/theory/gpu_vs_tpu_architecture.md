# GPU vs TPU — Architectural Deep Dive

## Why This Document Exists

You've been building DGEMM kernels on both FPGA (systolic array on KV260) and GPU (CUDA on RTX 3060). You already *intuitively* understand the two paradigms — spatial dataflow vs temporal thread parallelism — but you've been approaching TPUs without a clear mental model of where they fit. This document fixes that.

**The one-sentence summary:** A TPU is essentially your FPGA systolic array, but fabricated as a purpose-built ASIC at datacenter scale, with Google's compiler (XLA) replacing your HLS pragmas.

---

## Part 1: The Fundamental Philosophical Split

Every accelerator answers one question differently: **how do you keep the math units fed with data?**

| | GPU | TPU |
|---|---|---|
| **What it is** | General-purpose parallel processor with thousands of small, flexible cores | Purpose-built ASIC with one or two massive, specialized matrix engines |
| **Design philosophy** | Temporal parallelism — launch thousands of threads, hide memory latency by switching between them | Spatial parallelism — hardwire a grid of ALUs, pump data through like a pipeline |
| **Analogy** | A city of workers constantly running to a warehouse (memory) to fetch parts, build something, run back | A factory assembly line where parts flow station-to-station and the product assembles itself |
| **Programmability** | Write custom kernels in CUDA — full control over every thread | Write model code in JAX/TensorFlow — the XLA compiler maps it to hardware |

### Where Your FPGA Fits

You already built a systolic array on the KV260. The TPU's Matrix Multiply Unit (MXU) is architecturally the *same concept* — but instead of mapping onto reconfigurable fabric with DSP48E2 slices, it's etched permanently into silicon at a massive scale:

| | Your KV260 FPGA | Google TPU |
|---|---|---|
| **Compute structure** | Systolic array in PL fabric | Systolic array (MXU) in custom ASIC |
| **Array size** | ~8×8 to 32×32 (resource-limited) | 128×128 (v2–v5p) or 256×256 (v6+) |
| **Configuration** | Reconfigurable at synthesis time (HLS/RTL) | Fixed at fabrication — one architecture forever |
| **Compiler** | Vitis HLS → bitstream | XLA → TPU executable |
| **Clock speed** | ~200–300 MHz | ~1.0–1.7 GHz |
| **Memory** | DDR4 via AXI (limited BW) | HBM (terabytes/sec bandwidth) |

> [!TIP]
> When you read about TPU systolic arrays, mentally map it to the systolic array you already built. The data movement patterns (weight-stationary, input-stationary, output-stationary) are identical concepts — the TPU just does it with 16,384–65,536 MACs instead of your 64–1024.

---

## Part 2: GPU Architecture — The Bottleneck Model

You already know this from your CUDA learning guide, but let's frame it specifically for comparison with TPUs.

### How a GPU Does Matrix Multiply

```
For each element C[i][j]:
  1. Thread wakes up
  2. Fetch A[i][k] from VRAM → L2 → Shared Memory → Register     ← DATA MOVEMENT
  3. Fetch B[k][j] from VRAM → L2 → Shared Memory → Register     ← DATA MOVEMENT
  4. Multiply and accumulate in register                           ← ACTUAL MATH
  5. Repeat steps 2-4 for all k
  6. Write C[i][j] back to VRAM                                    ← DATA MOVEMENT
```

**The fundamental problem:** Steps 2, 3, and 6 are data movement. Step 4 is actual computation. On a naive kernel, the GPU spends >90% of its time and energy just *moving data*, not doing math.

Your optimization journey (naive → tiled → register-tiled) was entirely about reducing this data movement:

```
Optimization Level        Data Movement Cost       Compute Utilization
─────────────────────────────────────────────────────────────────────
Naive (1 thread/element)  Every FMA reads VRAM     ~2-5% of peak
Tiled (shared memory)     Tile loaded once, reused  ~30-50% of peak
Register-tiled            Inner loop all registers  ~60-80% of peak
cuBLAS (NVIDIA's best)    + vectorized loads, etc.  ~90-95% of peak
```

Even at 90%+ utilization, the GPU's architecture *inherently* requires this fetch-compute-store cycle. The memory hierarchy exists because thousands of independent threads need to access a shared pool of data.

### GPU Memory Hierarchy (The Tax You Always Pay)

```
┌─────────────────────────────────────────────────────────┐
│                    GPU Die                               │
│                                                         │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐              │
│  │   SM #0  │  │   SM #1  │  │  SM #27  │   ...        │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│              │
│  ││Register││  ││Register││  ││Register││  ~20 TB/s    │
│  │└───┬────┘│  │└───┬────┘│  │└───┬────┘│              │
│  │    ↕     │  │    ↕     │  │    ↕     │              │
│  │┌────────┐│  │┌────────┐│  │┌────────┐│              │
│  ││Shared  ││  ││Shared  ││  ││Shared  ││  ~2-5 TB/s  │
│  ││Memory  ││  ││Memory  ││  ││Memory  ││              │
│  │└───┬────┘│  │└───┬────┘│  │└───┬────┘│              │
│  └────┼─────┘  └────┼─────┘  └────┼─────┘              │
│       └──────────────┼──────────────┘                   │
│                      ↕                                  │
│              ┌──────────────┐                           │
│              │   L2 Cache   │           ~1 TB/s         │
│              └──────┬───────┘                           │
│                     ↕                                   │
│              ┌──────────────┐                           │
│              │  VRAM (HBM)  │           ~0.4-3.4 TB/s   │
│              └──────────────┘                           │
└─────────────────────────────────────────────────────────┘

Every arrow (↕) is a data movement cost you pay in latency and energy.
```

### NVIDIA's Response: Tensor Cores

NVIDIA recognized the inefficiency of doing matrix math on general-purpose CUDA cores. Starting with the Volta architecture (2017), they added **Tensor Cores** — small, hardwired matrix units embedded within each SM.

```
Tensor Core Operation (single instruction):
  D[4×4] = A[4×4] × B[4×4] + C[4×4]

  - Inputs A, B: FP16 / BF16 / INT8 (lower precision = higher throughput)
  - Accumulator C, D: FP32 (higher precision = numerical stability)
  - One instruction replaces 128 individual FMA operations
```

**Tensor Cores are conceptually a tiny systolic array inside the GPU.** They perform a fixed matrix-multiply-accumulate on small tiles. But they still live within the GPU's memory hierarchy — data must still be loaded from shared memory into the Tensor Core, and results written back.

| | CUDA Cores | Tensor Cores |
|---|---|---|
| **Operation** | Scalar FMA (1 multiply-add) | Matrix MMA (4×4 or larger tile) |
| **Precision** | FP32, FP64 | FP16/BF16 inputs, FP32 accumulate |
| **Throughput** | ~12,700 GFLOPS FP32 (RTX 3060) | ~100+ TFLOPS FP16 (A100) |
| **Flexibility** | Any computation | Matrix multiply only |
| **Data source** | Registers | Still fetched from shared memory → registers |

> [!IMPORTANT]
> Tensor Cores are NVIDIA admitting that the TPU's approach was right for matrix math. But GPUs keep the general-purpose CUDA cores alongside them for everything else (activation functions, normalization, custom ops). This is the GPU's superpower and its overhead.

---

## Part 3: TPU Architecture — The Wave Model

### The Core Insight: Eliminate Data Movement

The TPU asks: *what if we didn't have a memory hierarchy at all for the math?*

Instead of thousands of threads fetching from shared pools, the TPU hardwires a massive grid of ALUs and **flows data through them**. The math happens *as data moves* — there is no separate "fetch" and "compute" phase.

### The Matrix Multiply Unit (MXU) — A Systolic Array

This is the heart of every TPU. It's architecturally identical to what you built on the KV260, but at enormous scale.

```
TPU v5p MXU: 128 × 128 = 16,384 MACs
TPU v6e MXU: 256 × 256 = 65,536 MACs

Your KV260:  16 × 16  = 256 MACs (typical resource-constrained design)
```

#### Weight-Stationary Dataflow

The TPU's MXU uses a **weight-stationary** dataflow — the same concept you implement in your FPGA systolic array:

```
Step 1: Pre-load weights into every PE
═══════════════════════════════════════

        W[0,0]  W[0,1]  W[0,2]  W[0,3]
          ↓       ↓       ↓       ↓
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   ──→  │ MAC │→│ MAC │→│ MAC │→│ MAC │  ← Activations will flow in from left
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           ↓       ↓       ↓       ↓      ← Partial sums flow downward
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   ──→  │ MAC │→│ MAC │→│ MAC │→│ MAC │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           ↓       ↓       ↓       ↓
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   ──→  │ MAC │→│ MAC │→│ MAC │→│ MAC │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           ↓       ↓       ↓       ↓
        ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐
   ──→  │ MAC │→│ MAC │→│ MAC │→│ MAC │
        └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘
           ↓       ↓       ↓       ↓
        [result] [result] [result] [result]


Step 2: Stream activations and accumulate
═════════════════════════════════════════

Cycle 1:  a[0] enters row 0
Cycle 2:  a[0] propagates right, a[1] enters row 1  (staggered!)
Cycle 3:  a[0] continues, a[1] propagates, a[2] enters row 2
  ...

Each PE does:  partial_sum += activation × stored_weight
Then passes:   activation → right neighbor
               partial_sum → downward neighbor

Results emerge from the bottom after N + N - 1 cycles (pipeline fill + drain).
```

#### Why This Is Revolutionary for Matrix Math

Compare the data movement for computing one output element:

```
GPU (even optimized):
  1. Load tile of A from VRAM → Shared Memory      ← memory transaction
  2. Load tile of B from VRAM → Shared Memory      ← memory transaction
  3. __syncthreads()                                ← synchronization cost
  4. Each thread reads from shared mem → register   ← memory transaction
  5. FMA in register                                ← ACTUAL MATH
  6. __syncthreads()                                ← synchronization cost
  7. Repeat for next tile
  8. Write result to VRAM                           ← memory transaction

TPU systolic array:
  1. Pre-load weights (once, amortized)             ← one-time cost
  2. Stream activation into left edge               ← one read
  3. Data flows through grid, math happens          ← ACTUAL MATH (automatic!)
  4. Result emerges from bottom edge                ← one write
  No explicit synchronization. No cache hierarchy. No thread scheduling.
```

**The GPU does math between memory accesses. The TPU does math *during* data movement.** This is the fundamental difference.

---

## Part 4: The Full TPU Chip Architecture

The MXU is the heart, but a TPU chip has more:

```
Google TPU Chip (e.g., v5p)
├── TensorCore #0
│   ├── MXU #0 (128×128 systolic array)
│   ├── MXU #1 (128×128 systolic array)
│   ├── Vector Processing Unit (VPU)      ← activation functions, softmax, etc.
│   ├── Scalar Unit                       ← control flow, addressing
│   └── On-chip SRAM (scratchpad)         ← like BRAM on your FPGA
├── TensorCore #1
│   ├── MXU #2
│   ├── MXU #3
│   ├── VPU
│   ├── Scalar Unit
│   └── On-chip SRAM
├── HBM (High Bandwidth Memory)
│   └── 95 GB @ 2,765 GB/s               ← vs RTX 3060's 12 GB @ 360 GB/s
└── ICI (Inter-Chip Interconnect)
    └── 4,800 Gbps to neighboring TPUs    ← hardwired, no PCIe
```

### What Each Part Does

| Component | Purpose | GPU Equivalent |
|---|---|---|
| **MXU** | Matrix multiply (the heavy lifting) | Tensor Cores (but much larger) |
| **VPU** | Element-wise ops (ReLU, exp, add) | CUDA cores doing activation functions |
| **Scalar Unit** | Control flow, loop counters | CPU-side kernel launch logic |
| **On-chip SRAM** | Fast scratchpad for tiles | Shared memory (`__shared__`) |
| **HBM** | Bulk data storage | VRAM (Global Memory) |
| **ICI** | Chip-to-chip communication | NVLink |

> [!NOTE]
> The VPU is important — not everything in a neural network is matrix multiply. Softmax, layer normalization, and activation functions are element-wise operations. The TPU has a separate vector unit for these, but it's far less powerful than the MXU. This is why TPUs are less efficient on models with heavy non-matmul computation.

---

## Part 5: Software Stack — CUDA vs XLA

This is where the experience of *using* GPUs vs TPUs diverges most dramatically.

### GPU: You Write the Kernel

```
Developer → writes CUDA kernel → nvcc compiles → GPU executes threads

You control:
  - Thread layout (grid, block dimensions)
  - Memory placement (shared, registers, global)
  - Synchronization (__syncthreads)
  - Data access patterns (coalescing, bank conflicts)
  - Tile sizes, unroll factors, everything
```

This is powerful but demanding. Your DGEMM journey (naive → tiled → register-tiled) is a perfect example — you had to manually discover and implement every optimization.

### TPU: The Compiler Writes the Kernel

```
Developer → writes JAX/TensorFlow model → XLA compiler → optimized TPU executable

XLA controls:
  - How to partition matrices across MXUs
  - Which operations to fuse (e.g., MatMul + Add + ReLU → one pass)
  - Data layout and tiling for the systolic array
  - Memory management (what lives in SRAM vs HBM)
  - Inter-chip communication patterns
```

**You never write low-level code for the systolic array.** XLA is the equivalent of Vitis HLS for TPUs — it takes a high-level description and maps it to hardware. The tradeoff: you get less control, but Google's compiler team has spent years optimizing XLA for their hardware.

### The Static vs Dynamic Tradeoff

| | GPU (CUDA) | TPU (XLA) |
|---|---|---|
| **Shape handling** | Dynamic — any tensor shape, any time | Static — shapes must be known at compile time |
| **Recompilation** | No recompilation needed for different sizes | Shape change → full XLA recompilation (slow) |
| **Branching** | Full control flow (if/else in kernels) | Limited — branches hurt systolic utilization |
| **Custom ops** | Write any CUDA kernel you want | Must fit within XLA's supported operations |
| **Debugging** | printf in kernels, Nsight tools | Limited visibility into systolic execution |

> [!WARNING]
> **This is the TPU's Achilles' heel.** If your model has dynamic shapes (variable-length sequences, sparse attention, Mixture-of-Experts routing), the TPU's rigid systolic pipeline becomes inefficient. The MXU needs regular, predictable data — exactly like your FPGA systolic array chokes on irregular access patterns.

---

## Part 6: Scale — Pods, Interconnects, and Supercomputers

### GPU Scaling: NVLink + NVSwitch

NVIDIA scales GPUs using a hierarchical interconnect:

```
GPU Cluster (e.g., DGX H100)
├── Node (8× H100 GPUs)
│   ├── GPU 0 ←──NVLink──→ GPU 1    (900 GB/s bidirectional)
│   ├── GPU 0 ←──NVLink──→ GPU 2
│   │   ...  (all-to-all via NVSwitch)
│   └── NVSwitch fabric (non-blocking any-to-any)
├── Node ←──InfiniBand──→ Node      (400 Gb/s per link)
├── Node ←──InfiniBand──→ Node
└── ...

Topology: Switched fabric (Clos network)
Philosophy: Any GPU can talk to any GPU at full bandwidth
Strength: Flexible — handles irregular communication patterns
```

### TPU Scaling: ICI + Optical Circuit Switching

Google scales TPUs using a fundamentally different approach:

```
TPU Pod (e.g., v5p — up to 8,960 chips)
├── Cube (64 TPU chips)
│   ├── Arranged in 3D torus (4 × 4 × 4)
│   ├── Each chip connects to 6 neighbors (±X, ±Y, ±Z)
│   └── ICI links: direct copper, ~4,800 Gbps per chip
├── Cube ←──Optical Circuit Switch──→ Cube
│   ├── MEMS mirrors redirect light beams through fiber
│   ├── Reconfigurable topology (can reroute around failures)
│   └── Near-zero latency optical links
├── Cube ←──OCS──→ Cube
└── ...

Topology: 3D Torus (neighbor-focused, wraps around edges)
Philosophy: The entire pod is ONE accelerator
Strength: Massive scale, efficient collective operations (all-reduce)
```

### Comparison

| | GPU (NVLink/NVSwitch) | TPU (ICI/OCS) |
|---|---|---|
| **Intra-node topology** | Switched fabric (any-to-any) | 3D torus (neighbor-to-neighbor) |
| **Inter-node link** | InfiniBand / RoCE (standard networking) | Optical Circuit Switching (custom) |
| **Max cluster size** | ~thousands of GPUs (Frontier: 37,888) | ~9,216 TPU chips per superpod |
| **Communication pattern** | Flexible — any GPU talks to any GPU | Optimized for nearest-neighbor + all-reduce |
| **Fault tolerance** | Software-level checkpointing | Hardware OCS rerouting around dead chips |
| **Best for** | Irregular communication (MoE, sparse) | Synchronous bulk training (dense models) |

> [!NOTE]
> The 3D torus topology means a TPU pod is essentially a single, massive systolic array of systolic arrays. It's spatial parallelism all the way down — from the MACs inside the MXU, to the chips inside the cube, to the cubes inside the pod.

---

## Part 7: TPU Generational Evolution

| Gen | Year | MXU Size | HBM | Key Innovation |
|---|---|---|---|---|
| **v1** | 2015 | 256×256 (INT8) | — | Inference only. Deployed for Search, Translate. |
| **v2** | 2017 | 128×128 | HBM | First training-capable TPU. Introduced bfloat16. |
| **v3** | 2018 | 128×128 | HBM | Liquid cooling. Higher clocks. |
| **v4** | 2021 | 128×128 | HBM | 3D torus + Optical Circuit Switching. SparseCore for embeddings. |
| **v5e** | 2023 | 128×128 | HBM | Cost-optimized. 2D torus. |
| **v5p** | 2023 | 128×128 | 95 GB | Performance-optimized. 8,960-chip pods. |
| **v6 (Trillium)** | 2024 | **256×256** | HBM | 4× MXU throughput. Doubled HBM + ICI bandwidth. |
| **v7 (Ironwood)** | 2025 | 256×256 | HBM | Inference-focused. Native FP8. 9,216-chip superpods. |

**Key trend:** The MXU stayed at 128×128 for seven years before quadrupling to 256×256. This mirrors your FPGA work — scaling the systolic array is the single highest-leverage optimization, but it has enormous area and routing implications.

---

## Part 8: When to Use What

### GPU Wins When:

- **Custom kernels required** — you need operations not in standard ML frameworks
- **Dynamic shapes** — variable-length sequences, sparse data, complex control flow
- **Mixed workloads** — graphics + compute + AI on the same hardware
- **Prototyping** — rapid iteration, eager execution, printf debugging
- **Ecosystem** — PyTorch-first workflows, CUDA libraries, community code
- **On-premise** — you own the hardware and need flexibility

### TPU Wins When:

- **Pure matrix math at scale** — large dense models (LLMs, vision transformers)
- **Power efficiency matters** — 2-3× better FLOPS/watt for target workloads
- **Massive parallelism** — models that need thousands of chips working synchronously
- **Fixed model architecture** — stable shapes, no dynamic branching
- **Google Cloud** — TPUs are only available on GCP (no on-premise option)
- **JAX/TensorFlow** — frameworks with native XLA support

### The Honest Truth

For most AI practitioners:
- **Training a large model on GCP?** → TPU is likely more cost-effective
- **Research with custom ops and PyTorch?** → GPU is the pragmatic choice
- **Understanding computer architecture?** → Study both. The concepts transfer.

---

## Part 9: Connecting Everything — Your Three Accelerators

You now have experience with three fundamentally different accelerator architectures. Here's how they relate:

```
                    Flexibility
                        ↑
                        │
                  GPU   │   CPU
              (CUDA)    │  (x86)
                 ●      │    ●
                        │
    ────────────────────┼────────────────→ Generality
                        │
            TPU  ●      │
           (MXU)        │    ● FPGA
                        │  (your KV260)
                        │
                    Efficiency
                   (for target workload)
```

| Concept | FPGA (KV260) | GPU (RTX 3060) | TPU |
|---|---|---|---|
| **Compute unit** | DSP48E2 in systolic array | CUDA Core / Tensor Core | MAC in MXU systolic array |
| **Fast memory** | BRAM/URAM | Shared Memory (`__shared__`) | On-chip SRAM (scratchpad) |
| **Bulk memory** | DDR4 via AXI | VRAM (GDDR6/HBM) | HBM |
| **Compiler** | Vitis HLS / Vivado | nvcc (CUDA compiler) | XLA |
| **Tiling** | `#pragma HLS ARRAY_PARTITION` | Manual tile in shared memory | XLA auto-tiles |
| **Pipelining** | `#pragma HLS PIPELINE II=1` | Warp scheduling hides latency | Systolic pipeline inherent |
| **Parallelism** | `#pragma HLS UNROLL` | Thread-level (thousands of threads) | Spatial (hardwired MACs) |
| **Interconnect** | AXI bus | NVLink / PCIe | ICI (3D torus + OCS) |
| **Reconfigurable?** | Yes (reprogram at runtime) | No (fixed silicon, but flexible software) | No (fixed silicon, fixed function) |

> [!TIP]
> **Your portfolio story writes itself:** "I implemented the same algorithm — dense matrix multiply — on three fundamentally different architectures: spatial dataflow on FPGA, temporal parallelism on GPU, and studied how Google's TPU scales the systolic approach to datacenter level." This demonstrates genuine architectural understanding, not just framework usage.

---

## Part 10: The Bigger Picture — Why Both Exist

The GPU and TPU represent two valid solutions to the same problem, each making different tradeoffs:

```
The Spectrum of Compute Architecture:

GENERAL PURPOSE ◄──────────────────────────────► SPECIALIZED
     CPU              GPU              TPU           FPGA/ASIC
     
  - Runs anything   - Runs parallel   - Runs matrix  - Runs ONE thing
  - Slow at math     workloads well    math fast      perfectly
  - Very flexible  - Fairly flexible - Rigid         - Reconfigurable
  - High overhead  - Medium overhead - Low overhead  - Zero overhead
```

**GPUs exist because flexibility has enormous value.** Not every AI workload is dense matrix multiply. Mixture-of-Experts, sparse attention, reinforcement learning, generative models with complex sampling — all of these need the GPU's ability to run arbitrary code efficiently.

**TPUs exist because specialization has enormous efficiency.** If 90% of your compute is matrix multiply (true for most large transformer training), then 90% of the GPU's flexibility is wasted transistors. The TPU throws away everything except what matters and devotes the entire die to MAC units and their data supply.

**Neither is universally better. Understanding both makes you a better engineer.**

---

## Resources

| Resource | What It Is |
|---|---|
| [Google's TPU Research Paper (2017)](https://arxiv.org/abs/1704.04760) | The original paper: "In-Datacenter Performance Analysis of a Tensor Processing Unit" |
| [Google Cloud TPU Docs](https://cloud.google.com/tpu/docs) | Official documentation for programming TPUs |
| [ByteByteGo: GPU vs TPU](https://bytebytego.com) | Excellent visual comparison of architectures |
| [XLA Documentation](https://www.tensorflow.org/xla) | How the XLA compiler maps ops to TPU hardware |
| [JAX Quickstart](https://jax.readthedocs.io/) | The preferred framework for TPU development |
| [Your CUDA Guide](./learning_cuda_gpu.md) | Your existing GPU learning document |
| [Your FPGA Guide](./learning_kv260_fpga.md) | Your existing FPGA/systolic array document |

> [!TIP]
> **Read the 2017 TPU paper.** It's one of the most influential computer architecture papers of the decade, and with your FPGA systolic array experience, you'll understand it far more deeply than most readers. It directly compares the TPU v1 against contemporary GPUs and CPUs on real workloads.
