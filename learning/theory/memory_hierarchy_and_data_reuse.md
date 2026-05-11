# Memory, Bandwidth, and Data Reuse — How the Whole System Works Together

## What This Document Is For

This is a follow-on to `gpu_vs_tpu_architecture.md`. That document explained *what* GPUs and TPUs are. This document answers the deeper question that came up in practice:

**Why does the same matrix multiply run at 2% efficiency on a naive kernel but 80% on an optimized one — on the exact same hardware?**

The answer is entirely about how you manage data movement. This document explains:
1. What bandwidth actually is and why it's the fundamental bottleneck
2. The three-layer strategy for reducing how much data you move
3. Every abbreviation and short form, defined in plain English
4. How your FPGA experience and CUDA learning stack on top of each other

---

## First: A Glossary of Terms Used in This Document

Before anything else — definitions for every piece of jargon, so you never have to stop and guess.

### Memory Types (from fastest to slowest)

**Register** — The fastest possible storage. Lives directly inside the compute unit (CUDA core, DSP, ALU). Each thread has its own private set. Accessing a register takes zero extra time — the value is already there. Think of it as the compute unit's own hands.

**SRAM (Static Random Access Memory)** — Fast on-chip memory built from transistors that hold their state as long as power is on. Used for caches, shared memory, and scratchpads. Expensive in silicon area, so there's never much of it. On a GPU this is Shared Memory. On your FPGA this is BRAM (Block RAM). On a TPU this is the on-chip scratchpad.

**DRAM (Dynamic Random Access Memory)** — Bulk storage memory. Each bit is a tiny capacitor that slowly leaks charge and needs constant refreshing — that's why it's "dynamic." Cheap and dense, but slow. VRAM, system RAM, and HBM are all DRAM.

**VRAM (Video RAM)** — The large memory bank on a GPU card. "Video" is historical — it's just the GPU's main DRAM pool. Your RTX 3060 has 12 GB of it.

**GDDR6 (Graphics Double Data Rate 6)** — The specific DRAM standard used on consumer GPU cards like your RTX 3060. The chips sit on the PCB near the GPU die. Fast for what it is (~360 GB/s) but physically far from the compute.

**HBM (High Bandwidth Memory)** — DRAM stacked in 3D directly on top of or beside the compute die using tiny copper pillars called "through-silicon vias." Because the chips are millimetres away instead of centimetres, you can have thousands of parallel wires between them → massively more bandwidth. Used on H100, A100, and all TPU generations.

### Compute Terms

**ALU (Arithmetic Logic Unit)** — The part of any processor that does math: add, subtract, multiply, compare. CUDA cores are ALUs. They're general-purpose — you can tell them to do any math operation.

**MAC (Multiply-Accumulate)** — Hardware specifically hardwired to do `accumulator += A × B`. It's not general-purpose — it only does this one thing. But because it only does one thing, it does it very efficiently with minimal circuitry. Your FPGA's DSP48E2 is a MAC. Every PE in a TPU's MXU is a MAC.

**FMA (Fused Multiply-Add)** — Virtually identical to MAC, but the term used for *programmable* cores (like CUDA cores or CPU cores). "Fused" means the multiply and add happen in a single instruction with a single rounding step, instead of two separate instructions. The operation is: `result = (A × B) + C`. All of GEMM (matrix multiply) is just billions of these chained together.

**FLOP (Floating Point Operation)** — One arithmetic operation on a decimal number. **GFLOPS** = billions of FLOPs per second. **TFLOPS** = trillions per second. Your RTX 3060 does ~12,700 GFLOPS of FP32 math.

**FP64 / FP32 / FP16 / BF16 / INT8** — The *precision* (size) of the numbers being operated on:
```
FP64  = 64-bit float = "double" in C++ — most precise, most memory, slowest
FP32  = 32-bit float = "float"  in C++ — standard GPU compute
FP16  = 16-bit float               — half the memory, used in AI training
BF16  = "Brain Float 16" (Google)  — same range as FP32, less precise, TPU-native
INT8  = 8-bit integer              — inference only, smallest and fastest
```
Smaller number = less memory per value = less bandwidth needed = faster, but less accurate.

**PE (Processing Element)** — Generic term for one compute unit in an array. In your KV260 systolic array, one PE = one DSP48E2. In a TPU MXU, one PE = one MAC in the grid.

**MXU (Matrix Multiply Unit)** — Google's name for the systolic array inside a TPU chip. It's architecturally identical to what you built on the FPGA, just fabricated in silicon at 128×128 or 256×256 scale.

### GPU-Specific Terms

**SM (Streaming Multiprocessor)** — The repeating building block of an NVIDIA GPU. Your RTX 3060 has 28 of them. Each SM is like a mini-GPU with its own CUDA cores, registers, and shared memory. When you launch a kernel, CUDA assigns blocks to SMs.

**Warp** — A group of exactly 32 threads that execute the same instruction simultaneously (like SIMD). The SM's warp scheduler picks which warp to run each cycle. Having many warps lets the GPU hide memory latency by switching to a different warp while one is waiting for data.

**SIMT (Single Instruction, Multiple Threads)** — The GPU execution model. One instruction is broadcast to 32 threads (one warp) at once. All 32 execute it on their own data simultaneously. If threads in a warp take different branches (if/else), the GPU has to run them serially — called *warp divergence*, and it kills performance.

**`__shared__`** — The CUDA keyword that places a variable in Shared Memory (on-chip SRAM). All threads in the same block can read and write it. ~2–5 TB/s access speed vs ~360 GB/s for VRAM.

**`__syncthreads()`** — A barrier instruction. All threads in the block must reach this point before any of them continue. Required after cooperative loads into shared memory so no thread reads a value before another thread has finished writing it.

### Naming Conventions in GEMM Kernels

In optimized matrix multiply (GEMM) code, these variable names appear constantly:

```
M = number of rows in the output matrix C
N = number of columns in the output matrix C
K = the shared inner dimension (columns of A = rows of B)

B prefix = Block tile  — the chunk one entire thread BLOCK handles
T prefix = Thread tile — the chunk one individual THREAD handles

BM = Block tile rows    (e.g. 128) — one block handles 128 rows of C
BN = Block tile columns (e.g. 128) — one block handles 128 columns of C
BK = Block tile depth   (e.g. 8)   — each step processes 8 elements of K
TM = Thread tile rows   (e.g. 8)   — one thread handles 8 rows of C
TN = Thread tile columns(e.g. 8)   — one thread handles 8 columns of C

smemA = "shared memory A" — tile of matrix A loaded into on-chip SRAM
smemB = "shared memory B" — tile of matrix B loaded into on-chip SRAM
regA  = "register A"      — one thread's private slice loaded into registers
regB  = "register B"      — one thread's private slice loaded into registers
regC  = "register C"      — one thread's private output accumulator (in registers)
```

---

## What Is Bandwidth?

**Bandwidth** is how much data can physically move between two places per second. It is measured in GB/s (gigabytes per second) or TB/s (terabytes per second).

A simple analogy: a water pipe. Bandwidth is the *diameter* of the pipe — it determines the maximum flow rate. No matter how fast you pump, you cannot push more water through than the pipe diameter allows.

```
Memory Type          Bandwidth        Analogy pipe size
─────────────────────────────────────────────────────────
Registers            ~20 TB/s         Fire hose (millimetres from compute)
Shared Memory (SRAM) ~2–5 TB/s        Large pipe
L2 Cache             ~1 TB/s          Medium pipe
VRAM/GDDR6           ~360 GB/s        Garden hose
HBM (H100)           ~3,350 GB/s      Wide industrial pipe
```

**The fundamental problem:** Compute is fast. Memory is slow. Your RTX 3060 can do 12,700 GFLOPS — which means it needs 2 × 12,700 GB/s of data to keep every FMA fed (two operands per FMA). But VRAM only delivers 360 GB/s. The math units are 35× faster than the pipe feeding them.

**This is called the Memory Wall** — and every optimization technique in high-performance computing is ultimately a different way to fight it.

---

## What Can Actually Solve the Bandwidth Problem?

There are three levers. The industry pulls all three simultaneously.

### Lever 1: Wider Pipe — More Physical Wires

More wires between chip and memory = more data per second. This is purely a hardware/packaging engineering problem:

```
DDR4 (your KV260 FPGA):   64 pins    →  ~25 GB/s
GDDR6 (RTX 3060):        192 pins    →  ~360 GB/s
HBM3 (H100 GPU):       5,120 pins    →  ~3,350 GB/s
```

HBM achieves this by physically stacking DRAM dies directly on or beside the compute die using tiny copper pillars. Shorter distance between chips = you can fit thousands of wires = dramatically more bandwidth. This is why HBM is in every serious AI accelerator despite being expensive.

### Lever 2: Faster Signaling — Push Bits Faster Per Wire

Each wire can toggle faster. But this hits physical limits quickly — faster toggling means more power, more heat, more electrical noise between adjacent wires. Diminishing returns beyond a certain point.

### Lever 3: Need Less of It — Data Reuse

This is the *software and architecture* lever. Instead of increasing bandwidth supply, **reduce bandwidth demand** by reusing data that's already on-chip:

| Technique | How it reduces bandwidth demand |
|---|---|
| **Systolic arrays** (TPU, your FPGA) | One memory read serves N MACs as data flows through |
| **Tiling to shared memory** (CUDA Layer 1) | One VRAM read serves all 256 threads in a block |
| **Register blocking** (CUDA Layer 2) | 16 shared memory reads fuel 64 FMAs per thread |
| **Quantization** (FP32 → INT8) | 4× less data per number to move |
| **Sparsity / pruning** | Skip zero values entirely, don't fetch them |
| **Compute-in-memory** (emerging) | Do the math *inside* the memory chip itself |

**Lever 3 is where most innovation happens**, because levers 1 and 2 are constrained by physics and cost. The entire optimization journey you're doing with DGEMM (naive → tiled → register-tiled) is all Lever 3.

---

## The Three-Layer Data Reuse Strategy

The Level 3 SGEMM/DGEMM kernel stacks all three data reuse techniques as nested layers. Each layer solves the bandwidth problem at a different level of the memory hierarchy.

### The Big Picture

```
VRAM (12 GB, ~360 GB/s — slow)
  │
  │  LAYER 1: TILING
  │  "Don't let each thread fetch its own data from VRAM.
  │   Instead, 256 threads cooperate to bulk-load one shared tile."
  │  Result: ~32× less VRAM traffic
  ▼
Shared Memory (128 KB per SM, ~2–5 TB/s — fast)
  │
  │  LAYER 2: REGISTER BLOCKING
  │  "Don't re-read shared memory for every multiply.
  │   Each thread grabs its private slice into registers first."
  │  Result: ~4× less shared memory traffic
  ▼
Registers (per thread, ~20 TB/s — fastest)
  │
  │  LAYER 3: OUTER PRODUCT
  │  "64 multiply-adds with everything already in registers.
  │   No memory access of any kind. Pure math."
  │  Result: zero additional memory cost
  ▼
Write final result to VRAM (once, at the very end)
```

### Layer 1 — Tiling (VRAM → Shared Memory)

**The problem it solves:** In the naive kernel, every thread independently reads its own row from A and column from B out of VRAM. Neighbouring threads re-read the same data. Massive redundancy.

**The fix:** All 256 threads in a block cooperate to load one tile of A and one tile of B into shared memory together. Everyone contributes one element. Then everyone reads from the fast shared copy.

```
Block tile size: BM=128 rows, BN=128 cols, BK=8 depth

smemA[128][8] — a 128×8 tile of matrix A loaded into on-chip SRAM
smemB[8][128] — an 8×128 tile of matrix B loaded into on-chip SRAM

Total VRAM reads:    2 × 128 × 8 = 2,048 floats
Serves:              256 threads × their entire computation on this tile
Without tiling:      each thread would read separately = ~32× more VRAM traffic
```

This is exactly what your FPGA does with AXI DMA — bulk-transfer a tile from DDR into BRAM before processing it.

### Layer 2 — Register Blocking (Shared Memory → Registers)

**The problem it solves:** Even reading from shared memory every FMA is wasteful. Shared memory is shared by all 256 threads and has limited bandwidth. And re-reading the same values repeatedly is redundant.

**The fix:** Each thread grabs its personal slice from shared memory into private registers *once*, then does all its math from those registers.

```
Each thread's tile: TM=8 rows, TN=8 cols

regA[8] — 8 floats from smemA (this thread's row slice)   → loaded once
regB[8] — 8 floats from smemB (this thread's column slice) → loaded once

Cost: 16 shared memory reads per thread per K-step
Produces: 64 FMAs (8×8 outer product)

Without register blocking: 64 FMAs × 2 reads each = 128 shared memory reads
With register blocking:    16 reads → 64 FMAs = 4× less shared memory traffic
```

On your FPGA, this maps to each PE keeping its own local accumulator register rather than writing partial sums back to BRAM every cycle. Same concept — keeping the working data as close to the compute as possible.

### Layer 3 — Outer Product (Pure Registers)

**The problem it solves:** N/A — there is no memory access problem here. This layer is the *payoff* of layers 1 and 2.

**The mechanism:** With `regA` and `regB` loaded, each thread computes a full 8×8 submatrix of C using only register-to-register operations:

```c
// regC[8][8] lives entirely in this thread's register file
// regA[8] and regB[8] are also in registers
// Nothing touches memory during this loop

for (int i = 0; i < 8; i++)
    for (int j = 0; j < 8; j++)
        regC[i][j] += regA[i] * regB[j];  // 64 FMAs, all register operands
```

At the very end (after all K-steps are done), `regC` is written to VRAM exactly once per element.

This is identical to your FPGA DSP pipeline running at II=1 — no stalls, no memory access, just the accumulator updating every cycle.

### How the Three Layers Compound

The savings multiply together, not add:

```
                    VRAM reads per FMA    Efficiency vs peak
────────────────────────────────────────────────────────────
Naive kernel              2.0             ~2–5%
+ Layer 1 (tiling)        0.0625          ~30–50%
+ Layer 2 (reg blocking)  0.015           ~60–80%
+ cuBLAS (+ vectorized    ~0.010          ~90–95%
  loads, double buffering)

For a 4096×4096 matrix multiply, the difference between naive and
register-tiled is approximately: 4096³ × 2 / 0.36TB/s ≈ 383 seconds (naive)
                              vs: effectively compute-bound at ~0.3 seconds
```

---

## How Your Knowledge Stacks: FPGA → CUDA → TPU

This is the key insight from this conversation. You're not learning three separate things. You're learning the same idea — *keep data close to compute* — at three different abstraction levels.

```
Layer   | FPGA (KV260)              | CUDA (GPU)              | TPU
────────┼───────────────────────────┼─────────────────────────┼────────────────────────
Layer 1 | AXI DMA: DDR → BRAM      | Tiling: VRAM → smem     | HBM prefetch → SRAM
        | (bulk load before compute)| (cooperative block load) | (XLA compiler does it)
────────┼───────────────────────────┼─────────────────────────┼────────────────────────
Layer 2 | PE reads from BRAM into   | Register blocking:      | Weight-stationary:
        | its local pipeline regs   | smem → regA, regB       | weights pre-loaded into
        | (each PE owns its slice)  | (each thread owns slice)| each PE before compute
────────┼───────────────────────────┼─────────────────────────┼────────────────────────
Layer 3 | DSP48 pipeline: II=1,     | Outer product:          | Systolic flow:
        | acc += A × B every cycle  | regC[i][j] += regA[i]   | data moves PE→PE,
        | (no memory mid-compute)   |            * regB[j]     | math happens in transit
```

### What FPGA Gives You for Free

- **Why Layer 3 matters**: You've already seen that if a PE had to write its partial sum back to BRAM every cycle, your II blows up from 1 to 10+. The outer product runs at full throughput *only because* `regC` never leaves the register file. You know this viscerally.
- **Why tiling exists**: You wrote AXI DMA transfers and double-buffered BRAM (ping-pong buffers with `#pragma HLS DATAFLOW`). CUDA tiling is the same idea — load the next tile while computing the current one.
- **The memory hierarchy intuition**: Fast-close vs slow-far is something you already reason about from BRAM vs DDR on the FPGA.

### What CUDA Teaches That FPGA Doesn't

- **Layer 2 — Cooperative shared memory**: On the FPGA, you hardwired which PE reads from which BRAM address at synthesis time. In CUDA, 256 threads share one pool of memory at runtime. You have to coordinate *who* loads *which* element using `threadIdx` arithmetic, synchronize with `__syncthreads()`, and avoid bank conflicts (multiple threads hitting the same SRAM bank = serialized access).
- **Thread indexing replaces hardwired addresses**: On your FPGA, the systolic array routing is fixed in the bitstream. In CUDA, each thread computes its `c_row` and `c_col` at runtime from its thread/block index — the software does what your RTL wiring did.
- **The register pressure problem**: If you use a pointer or dynamic index to access `regC`, the compiler may not keep it in actual registers — it'll "spill" to local memory (slow VRAM). You have to write the loop in a way the compiler recognizes as register-safe. There's no equivalent problem on the FPGA because you explicitly controlled register usage with pragmas.

### What TPU Adds on Top

- **The compiler owns all three layers**: You don't write tiling code, you don't manually manage the scratchpad, you don't write the systolic loop. XLA (Google's compiler) analyzes your high-level model and generates all of it. This is the equivalent of writing `C = A * B` in HLS and having the tool figure out the pipeline — except Google's tool is far more sophisticated.
- **The tradeoff**: Because XLA owns the optimization, you get less visibility and control. If your model has unusual shapes or operations that don't fit the systolic pattern, performance drops and you can't easily fix it.

### The Full Transfer Map

```
You learned FPGA first:
  ✅ Data must live close to compute to avoid pipeline stalls
  ✅ Bulk-load a tile into fast memory (BRAM), process it, load next
  ✅ Keep partial sums in local registers (accumulator), not shared storage
  ✅ Pipelining: II=1 means one result every clock cycle
  ✅ Systolic arrays: data flows through PEs, math happens in transit

You're now learning CUDA:
  ✅ All FPGA intuition applies directly
  🆕 Shared memory is cooperative — threads coordinate at runtime
  🆕 __syncthreads() = software version of your pipeline handshake
  🆕 Thread indices replace your hardwired PE routing
  🆕 Bank conflicts — a CUDA-specific hazard, no FPGA equivalent
  🆕 Register pressure — compiler decides what actually goes in registers

Then when you look at TPU / XLA:
  ✅ Layer 3 (systolic compute) = your FPGA PE grid, just much bigger
  ✅ Layer 2 (SRAM scratchpad) = your BRAM tiles
  ✅ Layer 1 (HBM prefetch) = your AXI DMA
  🆕 The compiler owns all three — you write high-level model code only
  🆕 Static shapes required — XLA compiles once per shape, unlike CUDA
```

---

## ASIC vs Core — A Quick Clarifier

This came up when discussing what a TPU fundamentally is.

**A "core"** (CPU core, CUDA core, etc.) has a **fetch-decode-execute** cycle:
```
1. Fetch instruction from memory
2. Decode what it means
3. Execute it on some data
4. Move to next instruction
```
It has an instruction pointer and can run *any* sequence of instructions. It's programmable. That's what makes it a "core."

**An ASIC (Application-Specific Integrated Circuit)** has no instruction decoder. There is no program. The circuit *is* the algorithm — transistors are permanently connected to implement one specific computation. Data flows in, result flows out. You cannot reprogram it without fabricating a new chip.

```
Pure ASIC                                              Pure Core
(no instructions)                                    (fully programmable)
    │                                                      │
    ▼                                                      ▼
 Bitcoin     TPU's       GPU Tensor    CUDA        CPU
 miner      MXU         Core          Core         Core
    │         │             │            │            │
 Hardwired   Hardwired    Hardwired    Programmable  Fully
 SHA-256     systolic     small MMA    FMA unit      general
 only        array        only         + scheduler   purpose
```

The TPU chip is a hybrid: the MXU (systolic array) is pure ASIC — no instructions, just data flowing through hardwired MACs. But it also has scalar and vector units that *are* programmable cores, handling control flow and non-matmul operations. The ASIC does the heavy math; the cores handle everything around it.

Your FPGA sits in an interesting position: it's not an ASIC (you can reprogram it) but it's not a core either (there's no instruction fetch). It's *reconfigurable silicon* — you're deciding at synthesis time what circuit to build, and that circuit then behaves like an ASIC until you reprogram it.

---

## Summary: The One Mental Model to Keep

Every optimization in this space is the same idea expressed at different scales:

> **Compute is cheap. Moving data is expensive. The closer your data is to the compute, the less you pay.**

```
Register    → closest,  fastest,  smallest  (per-thread, per-PE)
SRAM        → close,    fast,     small     (shared memory, BRAM, TPU scratchpad)
DRAM/HBM    → far,      slow,     large     (VRAM, DDR, bulk storage)
```

Every technique — tiling, register blocking, systolic arrays, HBM, quantization — is just a different way of exploiting this hierarchy. Your FPGA work taught you this from the hardware up. CUDA teaches you to express it in software. TPU/XLA automates it through compilation. Same fight, different tools.
