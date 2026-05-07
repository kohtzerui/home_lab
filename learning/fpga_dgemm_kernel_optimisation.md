# FPGA DGEMM Kernel Optimisation — From Naive to Production

> Companion to `learning_kv260_fpga.md`. This file focuses purely on the HLS kernel code,
> what each optimisation does in hardware, and why it matters.

---

## Level 0: Naive — Sequential Triple Loop

```cpp
void dgemm_naive(double A[N][N], double B[N][N], double C[N][N]) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < N; k++)
                sum += A[i][k] * B[k][j];
            C[i][j] = sum;
        }
}
```

### What's wrong

- Every `A[i][k]` and `B[k][j]` read goes to **DDR** (off-chip, ~100 ns latency).
- No pipelining — each multiply waits for the previous one to finish.
- Total cycles ≈ N³ × (DDR latency) — catastrophically slow.

### Lesson

> **Rule 1: Never compute directly from DDR.** Always copy tiles into on-chip BRAM first.

---

## Level 1: Local Buffers — Move Data On-Chip

```cpp
#define TILE 8

void dgemm_local(double A[TILE][TILE], double B[TILE][TILE], double C[TILE][TILE]) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return

    // On-chip BRAM buffers
    double localA[TILE][TILE];
    double localB[TILE][TILE];
    double localC[TILE][TILE];

    // Burst-read from DDR → BRAM
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            localA[i][j] = A[i][j];
            localB[i][j] = B[i][j];
        }

    // Compute from BRAM (1-cycle access instead of ~100 ns)
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            double sum = 0.0;
            for (int k = 0; k < TILE; k++)
                sum += localA[i][k] * localB[k][j];
            localC[i][j] = sum;
        }

    // Write back BRAM → DDR
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++)
            C[i][j] = localC[i][j];
}
```

### What improved

- Reads/writes hit **BRAM** (1 cycle) instead of DDR (~100 cycles).
- Burst transfers amortise DDR latency across entire tiles.

### What's still wrong

- BRAM has only **2 read ports**. The inner loop reads `localA[i][k]` and `localB[k][j]`
  every cycle — that works for 1 MAC, but you can't parallelise without more ports.
- No pipelining pragma — HLS generates sequential logic.

### Lesson

> **Rule 2: BRAM ports are the first bottleneck.** You need ARRAY_PARTITION to get more ports.

---

## Level 2: PIPELINE + ARRAY_PARTITION — Real Parallelism

```cpp
#define TILE 8

void dgemm_pipelined(double A[TILE][TILE], double B[TILE][TILE], double C[TILE][TILE]) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return

    double localA[TILE][TILE];
    double localB[TILE][TILE];
    double localC[TILE][TILE];

    // Split arrays across multiple BRAM banks
    #pragma HLS ARRAY_PARTITION variable=localA complete dim=2  // 8 banks along columns
    #pragma HLS ARRAY_PARTITION variable=localB complete dim=1  // 8 banks along rows

    // Load A
    LOAD_A: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            #pragma HLS PIPELINE II=1
            localA[i][j] = A[i][j];
        }

    // Load B
    LOAD_B: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            #pragma HLS PIPELINE II=1
            localB[i][j] = B[i][j];
        }

    // Compute — the key optimisation
    COMPUTE: for (int i = 0; i < TILE; i++) {
        for (int k = 0; k < TILE; k++) {
            #pragma HLS PIPELINE II=1
            double a_val = localA[i][k];
            for (int j = 0; j < TILE; j++) {
                #pragma HLS UNROLL
                localC[i][j] += a_val * localB[k][j];
            }
        }
    }

    // Store C
    STORE_C: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = localC[i][j];
        }
}
```

### What each pragma does in hardware

| Pragma | Software analogy | Hardware reality |
|---|---|---|
| `PIPELINE II=1` | "Start a new loop iteration every cycle" | Creates a pipelined datapath — multiple iterations in-flight simultaneously |
| `UNROLL` (on j) | "Execute all j iterations at once" | **Physically duplicates** 8 MAC units — 8 multipliers + 8 adders in parallel |
| `ARRAY_PARTITION complete dim=2` on A | "Give each column its own memory" | Splits 1 BRAM into 8 independent banks — 8 simultaneous reads |
| `ARRAY_PARTITION complete dim=1` on B | "Give each row its own memory" | Same — 8 banks for B, enabling 8 parallel `B[k][j]` reads |

### Performance estimate

```
Without pragmas:  TILE³ = 512 cycles (sequential)
With pragmas:     TILE² = 64 cycles  (8 MACs per cycle × 64 k-iterations / 8 parallel = 64)
Speedup:          ~8× from UNROLL alone
```

### What's still wrong

- **Broadcast architecture** — all 8 MACs read from the same BRAM banks.
  Works fine for 8 PEs. Fails at 64+ PEs (too many readers per bank).
- **No overlap** between load and compute — compute waits until all data is loaded.

### Lesson

> **Rule 3: UNROLL creates parallel hardware. ARRAY_PARTITION provides the memory bandwidth to feed it.**
> They must be used together — UNROLL without PARTITION causes port contention and II > 1.

---

## Level 3: DATAFLOW + Double Buffering — Hide Memory Latency

```cpp
#define TILE 16

void load_tile(double DDR[TILE][TILE], double local[TILE][TILE]) {
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            #pragma HLS PIPELINE II=1
            local[i][j] = DDR[i][j];
        }
}

void compute_tile(double A[TILE][TILE], double B[TILE][TILE], double C[TILE][TILE]) {
    #pragma HLS ARRAY_PARTITION variable=A complete dim=2
    #pragma HLS ARRAY_PARTITION variable=B complete dim=1

    for (int i = 0; i < TILE; i++)
        for (int k = 0; k < TILE; k++) {
            #pragma HLS PIPELINE II=1
            double a_val = A[i][k];
            for (int j = 0; j < TILE; j++) {
                #pragma HLS UNROLL
                C[i][j] += a_val * B[k][j];
            }
        }
}

void store_tile(double local[TILE][TILE], double DDR[TILE][TILE]) {
    for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++) {
            #pragma HLS PIPELINE II=1
            DDR[i][j] = local[i][j];
        }
}

void dgemm_dataflow(double A[TILE][TILE], double B[TILE][TILE], double C[TILE][TILE]) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return

    // DATAFLOW makes load, compute, store run concurrently
    #pragma HLS DATAFLOW

    double bufA[TILE][TILE], bufB[TILE][TILE], bufC[TILE][TILE];

    load_tile(A, bufA);
    load_tile(B, bufB);
    compute_tile(bufA, bufB, bufC);
    store_tile(bufC, C);
}
```

### What DATAFLOW does

```
Without DATAFLOW (sequential):
  |--- Load A ---|--- Load B ---|--- Compute ---|--- Store C ---|
  Total = T_load_A + T_load_B + T_compute + T_store

With DATAFLOW (overlapped):
  |--- Load A ---|
       |--- Load B ---|
            |--- Compute ---|
                 |--- Store C ---|
  Total ≈ max(T_load, T_compute, T_store)  ← much faster
```

HLS automatically inserts **ping-pong buffers** (double buffers) between stages.
While compute processes tile N, the load stage is already fetching tile N+1.

### Lesson

> **Rule 4: Separate load/compute/store into functions and use DATAFLOW.**
> This hides DDR latency behind computation — the single biggest performance win after PIPELINE.

---

## Level 4: Systolic Array — The Architecture That Scales

At Level 2–3, all PEs broadcast-read from shared BRAM. This hits a wall at ~16 PEs.
A systolic array eliminates shared memory entirely — each PE only talks to its neighbours.

```cpp
#include <hls_stream.h>

#define SA 4      // 4×4 array
#define TK 16     // K-dimension tile

// Each PE: multiply-accumulate, pass data to neighbours
void pe(hls::stream<double> &a_in, hls::stream<double> &a_out,
        hls::stream<double> &b_in, hls::stream<double> &b_out,
        double &c_out) {
    #pragma HLS INLINE off
    double acc = 0.0;
    for (int k = 0; k < TK + SA - 1; k++) {
        #pragma HLS PIPELINE II=1
        double a = a_in.read();
        double b = b_in.read();
        acc += a * b;
        a_out.write(a);  // pass right →
        b_out.write(b);  // pass down  ↓
    }
    c_out = acc;
}

// Input skew: row i delayed by i cycles (zeros padded)
void skew_a(double A[SA][TK], hls::stream<double> a_feed[SA]) {
    for (int t = 0; t < TK + SA - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < SA; i++) {
            int k = t - i;
            a_feed[i].write((k >= 0 && k < TK) ? A[i][k] : 0.0);
        }
    }
}

// Same for B columns
void skew_b(double B[TK][SA], hls::stream<double> b_feed[SA]) {
    for (int t = 0; t < TK + SA - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < SA; j++) {
            int k = t - j;
            b_feed[j].write((k >= 0 && k < TK) ? B[k][j] : 0.0);
        }
    }
}
```

### Why systolic beats broadcast

| | Broadcast (Level 2) | Systolic (Level 4) |
|---|---|---|
| Data source | Shared BRAM, fan-out to all PEs | PE-to-PE streams, nearest-neighbour |
| Max PEs (KV260) | ~8–16 (port-limited) | ~64–128 (DSP-limited) |
| Wire length | Long (BRAM → distant PE) | Short (PE → adjacent PE) |
| Clock at scale | Degrades | Stays high |

### Lesson

> **Rule 5: Systolic arrays trade shared memory for local forwarding.**
> This is how TPUs and Tensor Cores work. The PE is trivial — the topology is the innovation.

---

## Level 5: Production Techniques (from SPCL gemm_hls)

The [spcl/gemm_hls](https://github.com/spcl/gemm_hls) project achieved **132 GFLOPS FP64** on
a VCU1525. These are the techniques beyond what's shown above:

### 1. Multi-level tiling

```
DDR (4 GB) → URAM tile (fits ~KB) → BRAM tile (fits ~bytes) → PE registers
```

Each level reduces the working set. The outermost tile fits in URAM, the innermost in registers.
This maximises the **compute-to-communication ratio** — more MACs per DDR byte transferred.

### 2. Wide memory ports (512-bit)

```cpp
#pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0 \
    max_read_burst_length=64 num_read_outstanding=16
```

A single DDR read fetches 512 bits = 8 doubles simultaneously.
Combined with burst transfers, this saturates the DDR bandwidth (~17 GB/s on KV260).

### 3. Kernel-level double buffering

While the systolic array processes tile (i, j, k), the feeder prefetches tile (i, j, k+1).
Implemented via `DATAFLOW` between feeder and compute functions.

### 4. Communication-avoiding tiling

Choose tile dimensions so that the **compute time ≥ load time**:

```
Compute time = Tm × Tn × Tk / (PEs × freq)
Load time    = (Tm × Tk + Tk × Tn) × 8 bytes / bandwidth

Set compute_time ≥ load_time → solve for Tk
```

If compute finishes before the next tile is loaded, you're **memory-bound**.
If the next tile arrives before compute finishes, you're **compute-bound** (ideal).

### 5. FP32 with iterative refinement

For maximum GFLOPS on limited DSPs, compute in FP32 (1 DSP per MAC vs 3-4 for FP64),
then use iterative refinement to recover FP64 accuracy:

```
x₀ = solve(A, b) in FP32
r  = b - A·x₀           in FP64 (on ARM CPU)
d  = solve(A, r) in FP32
x₁ = x₀ + d             in FP64
Repeat until ||r|| < ε
```

This is exactly what HPL-MxP (HPL-AI) does. It's a legitimate technique used in TOP500.

---

## Level 6: Faster — `ap_fixed` + Template Parameters + Power-of-2 Everything

Applying the low-latency principles from `low_latency_principles_for_hpc.md` directly to HLS:

### Why this is faster than Level 4

| Change | Principle | Speedup Mechanism |
|---|---|---|
| `ap_fixed` instead of `double` | Mechanical sympathy — match hardware | 1 DSP per MAC (not 3–4) → 4× more PEs for same resource |
| `constexpr` tile sizes | Pay at compile time, not runtime | HLS unrolls loops fully, eliminates all bounds checks |
| Power-of-2 tile sizes | Bitwise ops are free, modulo is not | `& (TILE-1)` replaces `% TILE` — saves cycles in address calc |
| Accumulator as local variable | Keep hot data in registers | Stays in flip-flop, not BRAM — 0-cycle access |
| No sync in hot path | Don't synchronise inside timing region | DATAFLOW between stages, never stall the compute pipeline |

```cpp
#include "ap_fixed.h"
#include "hls_stream.h"

// Fixed-point type: 32 bits total, 16 integer bits
// 1 DSP per MAC (vs 3-4 for FP64) = 4× more PEs available
typedef ap_fixed<32, 16> fixed_t;

// Template parameters → compiler resolves all sizes at elaboration time
// HLS can fully unroll every loop — no runtime bounds checks
template<int SA_SIZE, int TILE_K>
void pe_fast(
    hls::stream<fixed_t> &a_in, hls::stream<fixed_t> &a_out,
    hls::stream<fixed_t> &b_in, hls::stream<fixed_t> &b_out,
    fixed_t &c_out
) {
    #pragma HLS INLINE off
    fixed_t acc = 0;

    // TILE_K + SA_SIZE - 1 is known at compile time → fully unrolled
    for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
        #pragma HLS PIPELINE II=1
        fixed_t a = a_in.read();
        fixed_t b = b_in.read();
        acc += a * b;    // ap_fixed: 1 DSP48E2, not 3–4
        a_out.write(a);
        b_out.write(b);
    }
    c_out = acc;
}

// Skew feeder — power-of-2 TILE_K allows bitwise masking
template<int SA_SIZE, int TILE_K>
void skew_a_fast(fixed_t A[SA_SIZE][TILE_K],
                 hls::stream<fixed_t> a_feed[SA_SIZE]) {
    static_assert((TILE_K & (TILE_K - 1)) == 0,
                  "TILE_K must be power of 2 for bitwise masking");

    for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < SA_SIZE; i++) {
            #pragma HLS UNROLL
            int k = t - i;
            // Bitwise AND instead of bounds check — 1 cycle, not a branch
            bool valid = (k >= 0) & (k < TILE_K);
            a_feed[i].write(valid ? A[i][k] : fixed_t(0));
        }
    }
}
```

### What `ap_fixed<32,16>` means

```
ap_fixed<W, I>
         │   └─ integer bits (range: -2^(I-1) to 2^(I-1)-1)
         └───── total width in bits

ap_fixed<32, 16>:
  - 16 integer bits  → range ±32767
  - 16 fractional bits → precision ~0.000015
  - 1 DSP48E2 per MAC  (vs 3–4 for FP64)
  - Use iterative refinement (see Level 5) to recover accuracy if needed
```

### Resource comparison: FP64 vs ap_fixed

| Precision | DSPs per MAC | Max PEs on KV260 (1248 DSPs) | Est. GFLOPS @200 MHz |
|---|---|---|---|
| FP64 (double) | 3–4 | ~64 (conservative) | ~25.6 |
| FP32 (float) | 1–2 | ~128 | ~51.2 |
| ap_fixed<32,16> | 1 | ~200 | ~80 |
| ap_fixed<16,8> | 0.5 (2 per DSP) | ~400 | ~160 |

### Lesson

> **Rule 6: Match your arithmetic type to your DSP budget.**
> FP64 is honest but expensive. ap_fixed gives more PEs for the same silicon.
> Use iterative refinement (Level 5) to recover accuracy after computing in fixed-point.

---

## Low-Latency Principles Applied to HLS

*From `low_latency_principles_for_hpc.md` — translated from CUDA to Vitis HLS.*

### 1. Accumulator in register, not BRAM

```cpp
// WRONG — forces accumulator into BRAM (array = BRAM in HLS)
double localC[TILE][TILE];           // HLS maps this to BRAM: ~1-2 cycle access
for (int k ...) localC[i][j] += ...; // every iteration hits BRAM

// CORRECT — scalar accumulator stays in a flip-flop register
double acc = 0.0;                    // HLS keeps this in a register: 0-cycle
for (int k ...) acc += A[i][k] * B[k][j];
localC[i][j] = acc;                  // ONE write to BRAM at the end
```

This is the HLS equivalent of the CUDA principle: *"accumulator in register, not shared memory."*

### 2. Never use modulo on the critical path — use power-of-2 tile sizes

```cpp
// SLOW — integer division inside pipelined loop
int bank = addr % TILE;    // synthesises to a divider circuit: ~20+ cycles

// FAST — bitwise AND, only valid when TILE is power of 2
int bank = addr & (TILE - 1);   // 1 LUT, 0 extra latency
```

Consequence: **always set TILE = 8, 16, 32, 64.** Never 12, 20, etc.
The HLS tool cannot optimise integer division away — it will literally instantiate a divider.

### 3. Don't stall the pipeline — keep II=1

In HLS, `II` (Initiation Interval) is the FPGA equivalent of `cudaDeviceSynchronize()` inside a loop.
If the inner loop achieves II=2 instead of II=1, your throughput halves.

```cpp
// Common II > 1 cause: read-after-write dependency on an array
double localC[8][8];
for (int k = 0; k < 8; k++) {
    #pragma HLS PIPELINE II=1
    localC[i][j] += val;   // READ localC[i][j] → compute → WRITE localC[i][j]
                           // HLS sees: write latency > 1 cycle → II=2 or more
}

// Fix: use scalar accumulator (breaks the BRAM read-write dependency)
double acc = 0.0;
for (int k = 0; k < 8; k++) {
    #pragma HLS PIPELINE II=1
    acc += val;            // register → register: no memory dependency → II=1
}
localC[i][j] = acc;
```

### 4. Profile before optimising — read the synthesis report

After C-synthesis in Vitis HLS, check the report for:

```
Timing:
  Estimated clock period: X ns  ← must be < your target (e.g., 5 ns for 200 MHz)

Latency:
  Loop 'COMPUTE': II=? Latency=?  ← II=1 is your goal

Utilisation:
  DSP:   X / 1248  ← headroom for more PEs?
  BRAM:  X / 144   ← are tiles fitting on-chip?
  FF:    X / 234K
  LUT:   X / 117K
```

If II > 1 on the compute loop, the report shows the **dependency** causing it.
Fix that dependency before adding more PEs — more PEs on a slow pipeline just wastes DSPs.

### 5. Shift the bottleneck — compute-bound, not memory-bound

Same roofline principle as CUDA:

```
Memory-bound (bad):  compute finishes, waiting for next DDR tile
Compute-bound (good): next DDR tile arrives, waiting for compute to finish

Target: compute_time ≥ load_time

Compute time = (Tm × Tn × Tk) / (num_PEs × freq)
Load time    = (Tm × Tk + Tk × Tn) × bytes_per_element / DDR_bandwidth

If compute_time < load_time → add more PEs (or increase Tk)
If compute_time > load_time → widen the DDR bus (or use DATAFLOW to overlap)
```

For KV260 DDR bandwidth ~17 GB/s and 64 FP64 PEs at 200 MHz:
- Load time for 16×16 tile = (256 + 256) × 8 / 17e9 ≈ 240 ns
- Compute time = 256 × 256 MACs / (64 × 200M) ≈ 5 µs → **compute-bound** ✅

---

## Optimisation Summary Table

| Level | Technique | Key Pragma | Speedup vs Previous | Bottleneck Addressed |
|---|---|---|---|---|
| 0 | Naive | (none) | baseline | — |
| 1 | Local buffers | `m_axi` burst | ~100× | DDR latency |
| 2 | Pipeline + Partition | `PIPELINE`, `ARRAY_PARTITION`, `UNROLL` | ~8× | Sequential execution |
| 3 | Dataflow | `DATAFLOW` | ~2–3× | Load/compute serialisation |
| 4 | Systolic array | `hls::stream`, PE grid | ~4–8× | BRAM port contention |
| 5 | Production tricks | Wide ports, multi-tile, mixed precision | ~2× | Memory bandwidth |
| 6 | `ap_fixed` + templates + power-of-2 | `ap_fixed<W,I>`, `template<int>` | ~2–4× | DSP efficiency, II stalls, modulo overhead |

### Cumulative: Level 0 → Level 6 ≈ **20,000–60,000×** improvement

---

## KV260 Realistic Performance Targets

| Configuration | Est. GFLOPS | Notes |
|---|---|---|
| 4×4 systolic, FP64, 200 MHz | ~6.4 | 16 PEs × 2 FLOP × 200M |
| 8×8 systolic, FP64, 200 MHz | ~25.6 | 64 PEs × 2 FLOP × 200M |
| 8×8 systolic, FP32, 300 MHz | ~38.4 | More PEs available in FP32 |
| 16×16 systolic, FP32, 250 MHz | ~128 | Tight fit, may need tuning |

For comparison: ARM Cortex-A53 on KV260 ≈ **0.5 GFLOPS FP64**. Even a 4×4 array is a 12× speedup.

---

## What To Read Next

| Resource | Why |
|---|---|
| `learning_kv260_fpga.md` Layer 2.5 | Full systolic array theory with cycle-by-cycle trace |
| [spcl/gemm_hls](https://github.com/spcl/gemm_hls) | Production HLS systolic GEMM — study the source |
| [Vitis_Accel_Examples/systolic_array](https://github.com/Xilinx/Vitis_Accel_Examples) | Simplest working systolic example |
| UG1399 (Vitis HLS User Guide) | The pragma bible — look up any pragma here |
| SPCL FPGA'20 paper | "Flexible Communication Avoiding Matrix Multiplication on FPGA" |
