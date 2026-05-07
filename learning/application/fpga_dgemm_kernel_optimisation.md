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
#include "ap_int.h"

// Fixed-point: 32 bits total, 16 integer bits
// 1 DSP48E2 per MAC — 4× more PEs vs FP64
typedef ap_fixed<32, 16> fixed_t;

// ─────────────────────────────────────────────
// Processing Element
// ─────────────────────────────────────────────
template<int SA_SIZE, int TILE_K>
void pe_fast(
    hls::stream<fixed_t> &a_in,  hls::stream<fixed_t> &a_out,
    hls::stream<fixed_t> &b_in,  hls::stream<fixed_t> &b_out,
    hls::stream<fixed_t> &c_out  // stream output for DATAFLOW compatibility
) {
    #pragma HLS INLINE off
    fixed_t acc = 0;

    PE_LOOP: for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
        #pragma HLS PIPELINE II=1
        // Tell HLS: no loop-carried dependency on acc across iterations
        // (the accumulation IS the dependency, but it's correctly pipelined)
        #pragma HLS DEPENDENCE variable=acc inter false
        fixed_t a = a_in.read();
        fixed_t b = b_in.read();
        acc += a * b;
        a_out.write(a);   // pass right →
        b_out.write(b);   // pass down  ↓
    }
    c_out.write(acc);     // emit result once — stream keeps DATAFLOW happy
}

// ─────────────────────────────────────────────
// Skew feeder for A rows
// FIX 1: no branch inside pipeline — use ap_uint arithmetic mask
// FIX 2: ap_uint<1> mask costs 0 DSPs (pure LUT logic)
// ─────────────────────────────────────────────
template<int SA_SIZE, int TILE_K>
void skew_a_fast(fixed_t A[SA_SIZE][TILE_K],
                 hls::stream<fixed_t> a_feed[SA_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=A complete dim=2  // all cols readable at once
    static_assert((TILE_K & (TILE_K - 1)) == 0,
                  "TILE_K must be power of 2");

    SKEW_A: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int i = 0; i < SA_SIZE; i++) {
            #pragma HLS UNROLL
            int k = t - i;
            // Branchless: ap_uint<1> mask — 0 or 1, no if/else synthesised
            ap_uint<1> valid = (ap_uint<8>(k) < ap_uint<8>(TILE_K)) &
                               (t >= i ? ap_uint<1>(1) : ap_uint<1>(0));
            // Arithmetic select: no conditional, synthesises to LUT MUX
            fixed_t val = valid ? A[i][k & (TILE_K - 1)] : fixed_t(0);
            a_feed[i].write(val);
        }
    }
}

// ─────────────────────────────────────────────
// Skew feeder for B columns  (was MISSING in Level 5/6)
// ─────────────────────────────────────────────
template<int SA_SIZE, int TILE_K>
void skew_b_fast(fixed_t B[TILE_K][SA_SIZE],
                 hls::stream<fixed_t> b_feed[SA_SIZE]) {
    #pragma HLS ARRAY_PARTITION variable=B complete dim=2  // all cols readable at once
    static_assert((TILE_K & (TILE_K - 1)) == 0,
                  "TILE_K must be power of 2");

    SKEW_B: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            int k = t - j;
            ap_uint<1> valid = (ap_uint<8>(k) < ap_uint<8>(TILE_K)) &
                               (t >= j ? ap_uint<1>(1) : ap_uint<1>(0));
            fixed_t val = valid ? B[k & (TILE_K - 1)][j] : fixed_t(0);
            b_feed[j].write(val);
        }
    }
}

// ─────────────────────────────────────────────
// Top-level: SA_SIZE × SA_SIZE systolic array
// DATAFLOW overlaps: skew_a | skew_b | all PEs in parallel
// ─────────────────────────────────────────────
template<int SA_SIZE, int TILE_K>
void systolic_dgemm(
    fixed_t A[SA_SIZE][TILE_K],   // tile of A (on-chip)
    fixed_t B[TILE_K][SA_SIZE],   // tile of B (on-chip)
    fixed_t C[SA_SIZE][SA_SIZE]   // output tile
) {
    #pragma HLS DATAFLOW  // skew feeders + all PEs run concurrently

    // Inter-PE streams — A flows right, B flows down
    hls::stream<fixed_t> a_pipe[SA_SIZE][SA_SIZE + 1];
    hls::stream<fixed_t> b_pipe[SA_SIZE + 1][SA_SIZE];
    hls::stream<fixed_t> c_pipe[SA_SIZE][SA_SIZE];    // PE → result
    #pragma HLS STREAM variable=a_pipe depth=2
    #pragma HLS STREAM variable=b_pipe depth=2
    #pragma HLS STREAM variable=c_pipe depth=1

    // Input feeds (skewed)
    hls::stream<fixed_t> a_feed[SA_SIZE];
    hls::stream<fixed_t> b_feed[SA_SIZE];
    #pragma HLS STREAM variable=a_feed depth=TILE_K+SA_SIZE
    #pragma HLS STREAM variable=b_feed depth=TILE_K+SA_SIZE

    skew_a_fast<SA_SIZE, TILE_K>(A, a_feed);
    skew_b_fast<SA_SIZE, TILE_K>(B, b_feed);

    // Wire feeds into left/top edges
    FEED_A: for (int i = 0; i < SA_SIZE; i++)
        for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
            #pragma HLS PIPELINE II=1
            a_pipe[i][0].write(a_feed[i].read()); 
        }
    FEED_B: for (int j = 0; j < SA_SIZE; j++)
        for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
            #pragma HLS PIPELINE II=1
            b_pipe[0][j].write(b_feed[j].read());
        }

    // Instantiate PE grid — UNROLL creates SA_SIZE×SA_SIZE physical PEs
    PE_ROWS: for (int i = 0; i < SA_SIZE; i++) {
        PE_COLS: for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            pe_fast<SA_SIZE, TILE_K>(
                a_pipe[i][j],    a_pipe[i][j+1],
                b_pipe[i][j],    b_pipe[i+1][j],
                c_pipe[i][j]
            );
        }
    }

    // Collect results
    COLLECT: for (int i = 0; i < SA_SIZE; i++)
        for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS PIPELINE II=1
            C[i][j] = c_pipe[i][j].read();
        }
}
```

### ⚠️ Critical Correction: `ap_fixed<32,16>` does NOT map to 1 DSP48E2

This was a **factual error** in the Level 6 design, caught by deep research.

```
DSP48E2 physical ports (UltraScale+, used on KV260):
  Multiplier A-port: 27 bits maximum
  Multiplier B-port: 18 bits maximum
  Accumulator P-reg: 48 bits

ap_fixed<32, 16> multiplication:
  32 bits > 27-bit A-port  ← EXCEEDS HARDWARE LIMIT

What HLS actually does:
  Option A: Cascade 2 DSPs (splits 32-bit into partial products) → 2× resource cost
  Option B: Use 1 DSP + LUT fabric for upper bits → routing delay, timing failure at high freq

Correct types for exactly 1 DSP48E2 per MAC:
  Symmetric:   ap_fixed<18, 9>  × ap_fixed<18, 9>   (fits both ports)
  Asymmetric:  ap_fixed<27, 13> × ap_fixed<18, 9>   (max precision, 1 DSP)
  High density: ap_fixed<16, 8> × ap_fixed<16, 8>   (fits easily, allows 2 MACs per DSP)

Note: AP_SAT (saturation) ALSO breaks DSP inference — use AP_WRAP only.
Saturation requires fabric comparators that break the internal DSP feedback loop.
```

### Corrected resource table: DSP48E2 (KV260)

| Precision | Fits DSP48E2 ports? | DSPs per MAC | Max PEs (1248 DSPs) | Est. GFLOPS @200 MHz |
|---|---|---|---|---|
| FP64 (double) | ❌ (uses FPGA FP cores) | 3–4 | ~64 | ~25.6 |
| FP32 (float) | ❌ | 1–2 | ~128 | ~51.2 |
| `ap_fixed<32,16>` | ❌ **exceeds 27-bit A-port** | **2 (cascaded)** | ~100 | ~40 |
| `ap_fixed<27,13>×ap_fixed<18,9>` | ✅ asymmetric | **1** | ~200 | ~80 |
| `ap_fixed<16,8>` | ✅ fits easily | **1** (or 2/DSP) | ~200–400 | ~80–160 |

### DSP architecture limits by Xilinx family

| Family | DSP Type | A-port | B-port | Accumulator |
|---|---|---|---|---|
| 7 Series (Zynq-7000) | DSP48E1 | 25 bits | 18 bits | 48 bits |
| UltraScale+ (KV260) | DSP48E2 | **27 bits** | **18 bits** | 48 bits |
| Versal | DSP58 | 34 bits | 24 bits | 58 bits |

### Lesson (corrected)

> **Rule 6: Match your arithmetic type to the PHYSICAL DSP PORT LIMITS.**
> `ap_fixed<32,16>` on a KV260 cascades two DSPs — halving your PE count.
> Use `ap_fixed<27,13>` × `ap_fixed<18,9>` for maximum density at 1 DSP/MAC.
> Never use `AP_SAT` on the accumulator — it breaks the DSP's internal P-register loop.

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

## Level 7: Production-Grade — DSP Intrinsics + `hls::task` + SRL Skewing

*Based on AMD Vitis HLS UG1399, SPCL research, and deep microarchitectural analysis.*

Level 6 fixed the fundamentals. Level 7 changes the **execution model** to match how FPGA hardware actually works.

---

### 7.1 Correct DSP Mapping: Asymmetric Types + `BIND_OP`

Fix the precision first. Then optionally use DSP intrinsics for guaranteed mapping.

```cpp
// Correct asymmetric types for exactly 1 DSP48E2 per MAC
// A-port: 27 bits max, B-port: 18 bits max
typedef ap_fixed<27, 13> weight_t;    // A-port: weights (higher precision)
typedef ap_fixed<18,  9> act_t;       // B-port: activations
typedef ap_fixed<48, 22> acc_t;       // maps to DSP's 48-bit P accumulator

// --- Option A: Use BIND_OP (simpler, compiler-assisted) ---
template<int SA_SIZE, int TILE_K>
void pe_correct(
    hls::stream<weight_t> &a_in, hls::stream<weight_t> &a_out,
    hls::stream<act_t>    &b_in, hls::stream<act_t>    &b_out,
    hls::stream<acc_t>    &c_out
) {
    #pragma HLS INLINE off
    acc_t acc = 0;
    // Bind the accumulation to DSP with explicit latency
    // Must match so PIPELINE II=1 can be met
    #pragma HLS BIND_OP variable=acc op=add impl=dsp

    PE_LOOP: for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
        #pragma HLS PIPELINE II=1
        weight_t a = a_in.read();
        act_t    b = b_in.read();
        // 27-bit × 18-bit → fits DSP48E2 exactly → 1 DSP, P-register accumulates
        acc += (acc_t)(a * b);
        a_out.write(a);
        b_out.write(b);
    }
    c_out.write(acc);
}
```

```cpp
// --- Option B: DSP intrinsic (guaranteed mapping, bypasses compiler heuristics) ---
#include "hls_dsp_builtins.h"
using namespace hls::dsp48e2;

typedef ap_int<27> A_t;   // raw int for intrinsic (no ap_fixed wrapping)
typedef ap_int<18> B_t;
typedef ap_int<48> P_t;   // P-register output

void pe_intrinsic(
    hls::stream<A_t> &a_in, hls::stream<A_t> &a_out,
    hls::stream<B_t> &b_in, hls::stream<B_t> &b_out,
    hls::stream<P_t> &c_out
) {
    #pragma HLS INLINE off
    P_t acc = 0;

    PE_LOOP: for (int k = 0; k < TILE_K + SA_SIZE - 1; k++) {
        #pragma HLS PIPELINE II=1
        A_t a = a_in.read();
        B_t b = b_in.read();
        // Structural instantiation of DSP48E2 — no compiler guessing
        // P = A * B + P  (uses internal DSP feedback, not fabric routing)
        acc = mul_add<REG_A1 | REG_P>(a, b, acc);
        a_out.write(a);
        b_out.write(b);
    }
    c_out.write(acc);
}
```

**Why `REG_A1 | REG_P` matters:** These flags enable the DSP's internal pipeline registers,
allowing the feedback path (`P → P + A×B`) to close timing at 300+ MHz without leaving the DSP slice.

---

### 7.2 Replace Arithmetic Skewing with Shift Register Logic (SRL)

Arithmetic masking (Level 6) uses LUTs to *compute* which elements are valid each cycle.
SRL uses LUTs as *memory* to *delay* elements — no compute, no routing congestion.

```cpp
// Xilinx SRLs: LUT configured as a shift register
// SRL16E: 16-deep shift register using 1 LUT
// SRL32E: 32-deep shift register using 1 LUT
// Cost: 1 LUT per bit per delay stage — far cheaper than arithmetic masking at scale

template<int DELAY>
void srl_delay(
    hls::stream<act_t> &in,
    hls::stream<act_t> &out
) {
    #pragma HLS INLINE off
    act_t shift_reg[DELAY];
    // Bind to SRL primitive — not flip-flops, not BRAM
    #pragma HLS BIND_STORAGE variable=shift_reg impl=srl type=fifo

    SRL_LOOP: for (;;) {
        #pragma HLS PIPELINE II=1
        // Shift everything one position
        act_t in_val = in.read();
        for (int i = DELAY - 1; i > 0; i--) {
            #pragma HLS UNROLL
            shift_reg[i] = shift_reg[i-1];
        }
        shift_reg[0] = in_val;
        out.write(shift_reg[DELAY - 1]);
    }
}

// SRL-based skew feeder — row i gets i cycles of delay, zero arithmetic
template<int SA_SIZE>
void skew_a_srl(
    hls::stream<act_t> a_raw[SA_SIZE],    // undelayed rows
    hls::stream<act_t> a_skewed[SA_SIZE]  // delayed rows
) {
    #pragma HLS DATAFLOW
    for (int i = 0; i < SA_SIZE; i++) {
        #pragma HLS UNROLL
        // Row 0: 0 delay, Row 1: 1 delay, Row i: i delays
        // Uses i LUTs per bit — pure spatial delay, no computation
        srl_delay<i>(a_raw[i], a_skewed[i]);
    }
}
```

**Comparison:**

| Skew Method | Resources | Timing risk | Scales to 256×256? |
|---|---|---|---|
| Arithmetic mask (Level 6) | LUTs for comparators + MUX | Critical path grows with SA_SIZE | ❌ becomes bottleneck |
| SRL delay (Level 7) | 1 LUT per bit per delay | Zero logic depth | ✅ scales freely |

---

### 7.3 Replace `DATAFLOW` Sequential Loops with `hls::task` (DTLP)

The Level 6 `systolic_dgemm` uses `DATAFLOW` but puts sequential loops (FEED_A, FEED_B)
inside the dataflow region. This forces the compiler to infer oversized FIFOs to synchronise.

`hls::task` (Data-Driven Task-Level Parallelism) defines tasks that run **continuously and independently**,
triggering when input data is available — exactly like real hardware.

```cpp
#include "hls_task.h"

// Each task runs forever — no start/stop handshake
// Streams are the only interface — perfectly decoupled

void a_skew_task(
    hls::stream<act_t> &raw_in,
    hls::stream<act_t> &skewed_out,
    int delay
) {
    // Infinite loop — task never returns, always consuming input
    act_t shift_reg[16];
    #pragma HLS BIND_STORAGE variable=shift_reg impl=srl

    while (true) {
        #pragma HLS PIPELINE II=1
        act_t val = raw_in.read();
        for (int i = 15; i > 0; i--) shift_reg[i] = shift_reg[i-1];
        shift_reg[0] = val;
        skewed_out.write(shift_reg[delay - 1]);
    }
}

// Top-level using hls::task — hardware-native model
template<int SA_SIZE, int TILE_K>
void systolic_dgemm_task(
    hls::stream<act_t> a_input[SA_SIZE],  // DMA feeds these
    hls::stream<act_t> b_input[SA_SIZE],
    hls::stream<acc_t> c_output[SA_SIZE][SA_SIZE]
) {
    // Thread-local streams: compiler infers lightweight, localised FIFOs
    hls_thread_local hls::stream<act_t> a_skewed[SA_SIZE];
    hls_thread_local hls::stream<act_t> b_skewed[SA_SIZE];
    hls_thread_local hls::stream<act_t> a_pipe[SA_SIZE][SA_SIZE+1];
    hls_thread_local hls::stream<act_t> b_pipe[SA_SIZE+1][SA_SIZE];
    #pragma HLS STREAM variable=a_pipe depth=2
    #pragma HLS STREAM variable=b_pipe depth=2

    // Skew tasks — one per row/column, all run concurrently, always
    hls_thread_local hls::task a_skew_tasks[SA_SIZE];
    hls_thread_local hls::task b_skew_tasks[SA_SIZE];
    for (int i = 0; i < SA_SIZE; i++) {
        #pragma HLS UNROLL
        a_skew_tasks[i](a_skew_task, a_input[i], a_skewed[i], i);
        b_skew_tasks[i](a_skew_task, b_input[i], b_skewed[i], i);
    }

    // PE grid tasks — SA_SIZE×SA_SIZE autonomous MAC units
    hls_thread_local hls::task pe_tasks[SA_SIZE][SA_SIZE];
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL
            pe_tasks[i][j](pe_correct<SA_SIZE, TILE_K>,
                           a_pipe[i][j],   a_pipe[i][j+1],
                           b_pipe[i][j],   b_pipe[i+1][j],
                           c_output[i][j]);
        }
    }

    // Wire skewed feeds to pipe edges (also task-based)
    // In real design: edge wiring also becomes hls::task
}
```

**What changes with `hls::task` vs `DATAFLOW`:**

| | `DATAFLOW` (Level 6) | `hls::task` (Level 7) |
|---|---|---|
| Execution model | Run-to-completion function calls | Continuously running hardware FSMs |
| FIFO sizing | Compiler guesses (can be huge) | Minimal — just pipeline depth |
| Sequential loops inside | Violates dataflow semantics | Not possible — tasks are infinite |
| Latency between tiles | Start/stop overhead per tile | Zero — stream is always flowing |
| Matches hardware reality | Partially | Exactly |

---

### 7.4 Square Tiling for Minimum Off-Chip Traffic

The research confirms: tile dimensions should be as **square as possible** to maximise data reuse.

```
Given on-chip memory capacity S (elements):
  Optimal: Tm ≈ Tn ≈ √S

Why square:
  A tile of Tm×Tk reads Tm×Tk elements of A and Tk×Tn elements of B
  to produce Tm×Tn elements of C (which stays on-chip).

  Reuse ratio = (Tm × Tn × Tk MACs) / (Tm×Tk + Tk×Tn) loads
              = Tm×Tn×Tk / (Tk×(Tm+Tn))
              = Tm×Tn / (Tm+Tn)

  This is maximised when Tm = Tn (square tiles)

For KV260 with ~144 BRAMs × 36 Kb = ~648 KB on-chip:
  Reserve ~400 KB for tiles → S ≈ 400K / 2 bytes = 200K elements
  Optimal tile: √(200K/2) ≈ 316 → use 256×256 (next power-of-2)
  SA_SIZE = 16, TILE_K = 16 is a reasonable starting point
```

**Peak throughput formula:**

```
Performance = 2 × SA_SIZE × SA_SIZE × Frequency

For 16×16 array at 300 MHz:
  = 2 × 256 × 300M = 153.6 GOPS (int8/fixed)
  = 2 × 128 × 200M = 51.2 GFLOPS (FP32, 2 DSPs/MAC)
```

---

### 7.5 `hls::stream_of_blocks` for Array Interfaces in DTLP

Standard scalar streams can't efficiently carry 2D matrix tiles. `stream_of_blocks` provides
a synchronised multi-dimensional Ping-Pong buffer between memory movers and compute tasks.

```cpp
#include "hls_streamofblocks.h"

// DMA mover reads a tile from DDR and writes into a block
void dma_read_task(
    ap_uint<512>* ddr_ptr,
    hls::stream_of_blocks<act_t[TILE_M][TILE_K]> &a_tiles
) {
    while (true) {
        hls::write_lock<act_t[TILE_M][TILE_K]> lock(a_tiles);
        // Fill lock.data[i][j] from DDR burst
        for (int i = 0; i < TILE_M; i++)
            for (int j = 0; j < TILE_K; j++) {
                #pragma HLS PIPELINE II=1
                lock.data[i][j] = /* burst read from ddr_ptr */;
            }
        // lock releases here → block becomes readable by compute task
    }
}

// Compute task acquires tile via read_lock (stable view, no pointer aliasing)
void compute_task(
    hls::stream_of_blocks<act_t[TILE_M][TILE_K]> &a_tiles,
    hls::stream<acc_t> &results
) {
    while (true) {
        hls::read_lock<act_t[TILE_M][TILE_K]> lock(a_tiles);
        // lock.data is stable for the duration of this block
        // Feed to systolic array...
    }
}
```

**Benefit:** Compiler allocates only 2× tile memory (ping-pong), not SA_SIZE×TILE_K scalar FIFOs.

---

### Level 7 Lesson

> **Rule 7: Design for the hardware execution model, not the software execution model.**
> `hls::task` produces FSMs that react to data — that is how FPGA logic actually works.
> `DATAFLOW` with sequential loops is a software approximation of hardware; `hls::task` is hardware.

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
| 6 | `ap_fixed` + templates + power-of-2 *(with correction)* | `ap_fixed<27,13>×ap_fixed<18,9>`, `template<int>` | ~2–4× | DSP port limits, II stalls, modulo overhead |
| 7 | DSP intrinsics + `hls::task` + SRL skewing | `hls::task`, `BIND_OP impl=dsp`, `BIND_STORAGE impl=srl` | ~2× clock freq improvement | Timing closure, FIFO bloat, skew routing congestion |

### Cumulative: Level 0 → Level 7 ≈ **50,000–100,000×** improvement

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
| UG1399 — *Tasks and Channels* | `hls::task` and `hls_thread_local` stream model |
| UG1399 — *Using DSP Intrinsics* | `hls::dsp48e2::mul_add`, guaranteed DSP mapping |
| UG1399 — *DSP Multi-Operation Matching* | When BIND_OP works and when it fails |
| UG1399 — *Accumulation* | How the P-register accumulator maps to HLS |
| UG1399 — *Specifying Arrays as Stream-of-Blocks* | `hls::stream_of_blocks` for tile buffers |
| UG579 — *UltraScale DSP48E2 User Guide* | Physical port geometry: A=27 bits, B=18 bits |
| SPCL FPGA’20 paper | "Flexible Communication Avoiding Matrix Multiplication on FPGA" |
| arxiv: Stream-HLS | Automatic dataflow acceleration using MLIR/polyhedral compilers |
