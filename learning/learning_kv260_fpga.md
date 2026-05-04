# KV260 FPGA Learning Guide — What You Need to Know
https://nus-ee4218.github.io/labs/General/Installing_Vitis/

## Overview

The KV260 is a **Xilinx Zynq UltraScale+ MPSoC** board. It has two distinct halves:
- **PS (Processing System)** — 4× ARM Cortex-A53 cores running Linux
- **PL (Programmable Logic)** — the FPGA fabric where you build custom hardware

Your goal: use the PL to build a **DGEMM accelerator** (matrix multiply in hardware) that the PS calls from HPL, replacing the slow CPU BLAS with a custom hardware engine.

---

## The Learning Stack (Bottom Up)

```
Layer 5:  HPL Integration (link your accelerator into HPL's BLAS calls)
Layer 4:  Linux Driver    (PS-side C code that talks to your hardware via DMA)
Layer 3:  System Design   (Vivado block design — wire PS ↔ DMA ↔ your IP ↔ DDR)
Layer 2:  Kernel Design   (the actual matrix multiply hardware — HLS or Verilog)
Layer 1:  FPGA & Zynq     (how the hardware works — DSPs, BRAM, AXI buses)
Layer 0:  Digital Design   (you already have this from EE2026)
```

You need to learn Layers 1–5. Here's each one in detail.

---

## Layer 1: Understanding the KV260 Hardware

### What You're Working With

| Resource | KV260 (XCK26 / ZU5EV) |
|---|---|
| Logic Cells | 256K |
| **DSP48E2 Slices** | **1,248** |
| Block RAM (36Kb each) | 144 blocks (~5.1 Mb) |
| UltraRAM (288Kb each) | 64 blocks (~17.5 Mb) |
| DDR4 | 4 GB, 64-bit, non-ECC |
| PS Cores | 4× Cortex-A53 @ 1.5 GHz |
| PL Clock (typical) | 200–300 MHz |

### Why DSP Count Matters for DGEMM

HPL's core kernel is **DGEMM** (double-precision general matrix multiply). A single FP64 fused multiply-add (FMA) requires **~3–4 DSP48E2 slices** on UltraScale+ because each DSP48E2 does 27×18-bit integer multiply — you need to decompose the 53-bit FP64 mantissa multiplication across multiple slices.

**Rough estimate:**
```
1,248 DSPs ÷ 4 per FMA ≈ ~312 FP64 FMA units (theoretical max)
Realistic after routing/control: ~64–128 PEs in a systolic array
At 200 MHz: 128 PEs × 2 FLOP/PE/cycle × 200 MHz = ~51.2 GFLOPS (FP64 peak)
```

For comparison, an NVIDIA A100 does ~19,500 GFLOPS FP64. But for a $250 board running at ~5W, the GFLOPS/watt ratio is actually interesting.

### What You Need to Learn
- **DSP48E2 architecture** — how multiply-accumulate is mapped at the silicon level
- **BRAM vs. URAM** — when to use each (URAM is bigger but has restrictions)
- **AXI interconnect** — the bus protocol that connects everything on Zynq
- **Clock domains** — PS runs at fixed clocks, PL clock is configurable

### Resources
- **UG1085** — [Zynq UltraScale+ Technical Reference Manual](https://docs.amd.com/r/en-US/ug1085-zynq-ultrascale-trm) — the bible for your chip
- **DS891** — [ZU5EV Data Sheet](https://docs.amd.com/v/u/en-US/ds891-zynq-ultrascale-plus-overview) — resource counts and specs
- **UG579** — [UltraScale+ DSP48E2 User Guide](https://docs.amd.com/v/u/en-US/ug579-ultrascale-dsp48e2) — how the DSP slices work

---

## Layer 2: Kernel Design (The Actual Accelerator)

You have **two paths** here. Start with HLS, drop to Verilog later if needed.

### Path A: Vitis HLS (Recommended Starting Point)

**What it is:** Write your DGEMM kernel in **C++** with special pragma annotations. The tool synthesizes it into hardware (Verilog/VHDL) automatically.

**Why start here:**
- 10× faster iteration than hand-written Verilog
- You can verify correctness by just compiling and running the C++ code
- Achieves 80–95% of hand-tuned Verilog performance
- The pragmas teach you hardware thinking without wiring individual signals

**Key concepts to learn:**

| Pragma | What It Does | Hardware Effect |
|---|---|---|
| `#pragma HLS PIPELINE II=1` | Process new input every clock cycle | Creates pipelined datapath |
| `#pragma HLS UNROLL factor=8` | Duplicate hardware N times | Creates N parallel compute units |
| `#pragma HLS ARRAY_PARTITION` | Split array into separate memories | Enables parallel reads/writes |
| `#pragma HLS DATAFLOW` | Run functions concurrently | Creates task-level pipeline |
| `#pragma HLS INTERFACE` | Define physical port types | Maps to AXI-Stream, AXI-MM, etc. |

**Example — Tiled DGEMM kernel in HLS (~60 lines):**

```cpp
#include "hls_stream.h"

#define TILE 8

void matmul_tile(
    double A[TILE][TILE],
    double B[TILE][TILE],
    double C[TILE][TILE]
) {
    #pragma HLS INTERFACE m_axi port=A offset=slave bundle=gmem0
    #pragma HLS INTERFACE m_axi port=B offset=slave bundle=gmem1
    #pragma HLS INTERFACE m_axi port=C offset=slave bundle=gmem2
    #pragma HLS INTERFACE s_axilite port=return

    // Local tile buffers (mapped to BRAM)
    double local_A[TILE][TILE];
    double local_B[TILE][TILE];
    double local_C[TILE][TILE];

    #pragma HLS ARRAY_PARTITION variable=local_A complete dim=2
    #pragma HLS ARRAY_PARTITION variable=local_B complete dim=1

    // Load tiles from DDR → BRAM
    LOAD_A: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++)
            #pragma HLS PIPELINE II=1
            local_A[i][j] = A[i][j];

    LOAD_B: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++)
            #pragma HLS PIPELINE II=1
            local_B[i][j] = B[i][j];

    // Initialize C
    INIT_C: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++)
            #pragma HLS PIPELINE II=1
            local_C[i][j] = 0.0;

    // Compute C += A × B
    COMPUTE: for (int i = 0; i < TILE; i++) {
        for (int k = 0; k < TILE; k++) {
            #pragma HLS PIPELINE II=1
            double a_val = local_A[i][k];
            for (int j = 0; j < TILE; j++) {
                #pragma HLS UNROLL  // 8 parallel MACs
                local_C[i][j] += a_val * local_B[k][j];
            }
        }
    }

    // Store result BRAM → DDR
    STORE_C: for (int i = 0; i < TILE; i++)
        for (int j = 0; j < TILE; j++)
            #pragma HLS PIPELINE II=1
            C[i][j] = local_C[i][j];
}
```

**What the tool generates:** The `UNROLL` on j creates 8 parallel FP64 MAC units. The `PIPELINE II=1` on k means a new k iteration starts every cycle. Combined, this produces hardware that computes an entire row of C in 8 cycles.

### Path B: Hand-Written Verilog (Advanced)

**When to use:** Only after you have a working HLS design and want to squeeze the last 5–20% of performance from critical PEs.

**Key concepts:**
- **Systolic array** — see the full deep-dive in Layer 2.5 below
- **FP64 IP cores** — you instantiate Xilinx's `floating_point` IP from Vivado's IP Catalog for multiply/add
- **Pipeline depth** — FP64 multiply is ~8 cycles, FP64 add is ~12 cycles on UltraScale+
- **Skew registers** — each row/column of the systolic array gets delayed by one cycle to create the wave effect

**Resources for Layer 2:**
- **UG1399** — [Vitis HLS User Guide](https://docs.amd.com/r/en-US/ug1399-vitis-hls) — the HLS bible
- **GitHub** — [Xilinx/Vitis-Tutorials](https://github.com/Xilinx/Vitis-Tutorials) — hands-on examples
- **GitHub** — [Xilinx/Vitis_Accel_Examples](https://github.com/Xilinx/Vitis_Accel_Examples)

---

## Layer 2.5: Systolic Arrays — The Architecture That Powers TPUs

This is the single most important architectural concept in your FPGA project. Google's TPU, NVIDIA's Tensor Cores, and every modern AI/HPC accelerator ASIC uses some form of systolic dataflow. If you can build one on your KV260, you'll understand *why* these architectures dominate — not just *that* they do.

### Why This Section Exists

Your Path A kernel (the tiled HLS DGEMM above) uses `UNROLL` + `PIPELINE` pragmas. That produces correct, fast hardware — but it's a **broadcast architecture**: data is loaded from BRAM into a set of parallel MAC units that all read from the same memory ports simultaneously. This works, but it's fundamentally different from a systolic array.

```
Your UNROLL approach:                     A systolic array:

  BRAM ──broadcast──► MAC0                A ──► [PE] ──► [PE] ──► [PE]
       ──broadcast──► MAC1                       ↓        ↓        ↓
       ──broadcast──► MAC2                B ──► [PE] ──► [PE] ──► [PE]
       ──broadcast──► MAC3                       ↓        ↓        ↓
       ──broadcast──► MAC4                     [PE] ──► [PE] ──► [PE]
       ──broadcast──► MAC5                       ↓        ↓        ↓
       ──broadcast──► MAC6                     C out    C out    C out
       ──broadcast──► MAC7
                                          No broadcast. Each PE only talks
  All MACs read same BRAM port.           to its immediate neighbours.
  Works for 8 PEs. Doesn't scale         Scales to hundreds of PEs because
  to 128 PEs (port contention).           wires are short and local.
```

**The broadcast approach fails at scale** because BRAM ports become a bottleneck — you can't have 128 MACs all reading the same memory simultaneously. Systolic arrays solve this by eliminating the need for any shared memory access during computation.

---

### The History: Where Systolic Arrays Come From

H.T. Kung (Carnegie Mellon) and Charles Leiserson (MIT) published the seminal paper *"Systolic Arrays (for VLSI)"* in 1978. The key insight was about VLSI wire cost:

> *"The I/O bottleneck of a VLSI chip is that the number of I/O pins grows much slower than the number of transistors inside the chip. A systolic architecture avoids this bottleneck by having data flow rhythmically through a network of simple processors, being used multiple times before it exits."*

The name "systolic" comes from the medical term — blood pumped rhythmically through the body by the heart. In a systolic array, data pulses through the PE grid like blood through arteries.

**Why this matters in 2026:** The same wire-cost argument that motivated Kung and Leiserson is *more* relevant today than in 1978. Moving data across a chip costs 100–1000× more energy than a floating-point multiply. Systolic arrays minimize data movement by making every value pass through multiple PEs, getting reused at each one.

---

### The Core Architecture

A systolic array for matrix multiply (C = A × B) is a 2D grid of identical Processing Elements:

```
Input skew                          PE Array                        Output
(delay regs)
                    ┌─────────┐  ┌─────────┐  ┌─────────┐
  a[0][0..K] ─────►│ PE(0,0) ├─►│ PE(0,1) ├─►│ PE(0,2) │──► a drains out
                    │ acc_00  │  │ acc_01  │  │ acc_02  │
                    └────┬────┘  └────┬────┘  └────┬────┘
                         │b           │b           │b
  a[1][0..K] ─── D ────►│       ┌────▼────┐  ┌────▼────┐
                    ┌────▼────┐  │ PE(1,1) ├─►│ PE(1,2) │
                    │ PE(1,0) ├─►│ acc_11  │  │ acc_12  │
                    │ acc_10  │  └────┬────┘  └────┬────┘
                    └────┬────┘       │b           │b
                         │b     ┌────▼────┐  ┌────▼────┐
  a[2][0..K] ── D ─ D ─►│      │ PE(2,1) ├─►│ PE(2,2) │
                    ┌────▼────┐ │ acc_21  │  │ acc_22  │
                    │ PE(2,0) ├►│         │  │         │
                    │ acc_20  │ └────┬────┘  └────┬────┘
                    └────┬────┘      │            │
                         ▼           ▼            ▼
                       b drains    b drains     b drains

  D = one clock cycle delay register
```

**Each PE does exactly three things per clock cycle:**
1. Multiply its `a_in` × `b_in` and add to its local accumulator
2. Pass `a_in` to its right neighbour (→)
3. Pass `b_in` to its bottom neighbour (↓)

That's the entire architecture. No control logic. No address generation. No memory ports. Just multiply-accumulate-and-pass.

---

### The Processing Element (PE)

```
         b_in
          │
          ▼
   ┌──────────────┐
   │              │
a_in ──► │  acc += a × b  │ ──► a_out (= a_in, delayed 1 cycle)
   │              │
   │   acc (local │
   │   register)  │
   └──────┬───────┘
          │
          ▼
        b_out (= b_in, delayed 1 cycle)
```

**In HLS C++:**

```cpp
// A single Processing Element
void pe(
    double a_in, double b_in,    // inputs from neighbours
    double &a_out, double &b_out, // outputs to neighbours
    double &acc                   // local accumulator (stays in PE)
) {
    #pragma HLS INLINE off
    #pragma HLS PIPELINE II=1

    acc += a_in * b_in;   // the only computation
    a_out = a_in;         // pass A rightward
    b_out = b_in;         // pass B downward
}
```

That's it. The entire architectural intelligence is in how you connect these PEs and how you skew the input data — not in the PE itself.

---

### Input Skewing: Why the Delays Exist

If you fed all rows of A and all columns of B simultaneously, the wrong elements would meet at each PE. The skew delays ensure that `A[i][k]` and `B[k][j]` arrive at `PE(i,j)` at exactly the right cycle.

**For a 3×3 multiply (C = A × B):**

```
Without skew (WRONG):              With skew (CORRECT):

Cycle 0: all of row 0 enters   →  Cycle 0: only a[0][0] enters
         all of col 0 enters   →           only b[0][0] enters
         PE(0,0) gets a00,b00  ✓  PE(0,0) gets a00,b00  ✓
         PE(0,1) gets a01,b01  ✗  PE(0,1) gets nothing  (waiting)
         PE(1,0) gets a10,b10  ✗  PE(1,0) gets nothing  (waiting)

                                   Cycle 1: a[0][1] enters row 0
                                            a[1][0] enters row 1 (1 cycle late)
                                            b[1][0] enters col 0
                                            b[0][1] enters col 1 (1 cycle late)
```

**Skew rule:**
- Row `i` of A is delayed by `i` cycles before entering the left edge
- Column `j` of B is delayed by `j` cycles before entering the top edge

This creates a diagonal wavefront that sweeps across the array:

```
Cycle:  0       1       2       3       4

       [*]     [*][*]  [*][*]  [ ][*]  [ ][ ]
       [ ][ ]  [*][ ]  [*][*]  [*][*]  [ ][*]
       [ ][ ]  [ ][ ]  [*][ ]  [*][*]  [*][*]

       * = PE is actively computing
       Active PEs move as a diagonal wave
```

---

### Cycle-by-Cycle Execution Trace (3×3 Example)

Let's multiply:

```
A = | 1  2  3 |     B = | 7   8  |     C = A × B = | 58   64  |
    | 4  5  6 |         | 9  10  |                  | 139  154 |
                        | 11  12 |
```

Using a **2×2 output-stationary systolic array** (accumulator stays in PE):

For simplicity, feeding the K=3 inner dimension through a 2×2 array:

```
PE(0,0) computes C[0][0] = 1×7 + 2×9 + 3×11 = 7 + 18 + 33 = 58
PE(0,1) computes C[0][1] = 1×8 + 2×10 + 3×12 = 8 + 20 + 36 = 64
PE(1,0) computes C[1][0] = 4×7 + 5×9 + 6×11 = 28 + 45 + 66 = 139
PE(1,1) computes C[1][1] = 4×8 + 5×10 + 6×12 = 32 + 50 + 72 = 154
```

**The execution unfolds over cycles 0–4:**

| Cycle | PE(0,0) a×b | PE(0,0) acc | PE(0,1) a×b | PE(0,1) acc | PE(1,0) a×b | PE(1,0) acc | PE(1,1) a×b | PE(1,1) acc |
|---|---|---|---|---|---|---|---|---|
| 0 | 1×7 | **7** | — | 0 | — | 0 | — | 0 |
| 1 | 2×9 | **25** | 1×8 | **8** | 4×7 | **28** | — | 0 |
| 2 | 3×11 | **58** ✓ | 2×10 | **28** | 5×9 | **73** | 4×8 | **32** |
| 3 | — | 58 | 3×12 | **64** ✓ | 6×11 | **139** ✓ | 5×10 | **82** |
| 4 | — | 58 | — | 64 | — | 139 | 6×12 | **154** ✓ |

**Key observation:** Each PE finishes at a different cycle (the wavefront). Total compute time = K + (N-1) + (M-1) = 3 + 1 + 1 = 5 cycles. After the initial fill, one new result completes every cycle.

> [!IMPORTANT]
> **This is why systolic arrays are efficient:** Once the pipeline is full, the array produces one complete output per cycle, regardless of array size. A 256×256 array (like the TPU) does 65,536 MACs per cycle.

---

### Three Dataflow Variants

Which data stays "stationary" in the PE defines the dataflow type. All three implement the same math, but have different energy and bandwidth tradeoffs:

#### 1. Output Stationary (OS) — Best for DGEMM on KV260

```
What stays in PE:  The partial sum (accumulator for C[i][j])
What flows:        A flows right (→), B flows down (↓)
When C is read:    Only after ALL K iterations complete (one write per output)
```

**Why it's best for your HPL/DGEMM project:**
- Minimises writes to external memory — each C element is written once after full accumulation
- Accumulator stays in a flip-flop register — fastest possible storage
- The PE code above implements this variant

**Who uses it:** Google TPU v1 (partially), Berkeley Gemmini accelerator

#### 2. Weight Stationary (WS) — Best for Neural Network Inference

```
What stays in PE:  The weight (a value from the weight matrix)
What flows:        Activations flow through, partial sums flow down
When weights move: Only when switching to a new layer
```

**Why used for inference:** In neural network inference, the same weights are applied to millions of input vectors. Loading weights once and streaming inputs through is optimal.

**Who uses it:** Google TPU v1 (for inference), many edge AI chips

#### 3. Row Stationary (RS) — Best for Diverse Workloads

```
What stays in PE:  Rows of all three matrices (A, B, C) — composite approach
What flows:        Controlled by a more complex scheduler
Advantage:         Maximises reuse of ALL data types
```

**Who uses it:** MIT Eyeriss (the academic gold standard for efficient inference)

**For your KV260 DGEMM project:** Use **output stationary**. It's the simplest to implement, maps directly to the DGEMM computation, and minimises DDR write traffic (your #1 bottleneck).

---

### Comparison: Your UNROLL Design vs Systolic Array

| Property | HLS UNROLL (Path A) | Systolic Array |
|---|---|---|
| **Data source** | BRAM (shared) | PE-to-PE forwarding (local) |
| **Communication** | Broadcast from BRAM ports | Nearest-neighbour only |
| **Wiring** | Fan-out from BRAM to all MACs | Short, regular, local |
| **Scalability limit** | BRAM port count (~2-4 ports) | DSP count (hundreds) |
| **Max PEs (KV260)** | ~8–16 (port-limited) | ~64–128 (DSP-limited) |
| **Routing pressure** | High (long wires from BRAM) | Low (short wires between adjacent PEs) |
| **Clock frequency** | Lower at scale (long paths) | Higher at scale (short paths) |
| **Implementation effort** | Low (~60 lines of HLS) | Medium (~150 lines of HLS) |
| **What it teaches** | HLS pragmas, memory tiling | Spatial computing, dataflow architecture |

**Bottom line:** Start with UNROLL (Path A) to get something working. Then build a systolic array to understand the architecture that actually scales. Both produce valid DGEMM results — the difference is what you learn and what you can talk about in interviews.

---

### Building a Systolic Array in HLS (Full Implementation)

Here's how to implement an output-stationary systolic array for DGEMM in Vitis HLS. This is **Path C** — a middle ground between the simple UNROLL (Path A) and hand-written Verilog (Path B).

```cpp
#include <hls_stream.h>
#include <ap_fixed.h>

// ============================================================
// Configuration
// ============================================================
#define SA_SIZE  4     // 4×4 systolic array (start small!)
#define TILE_K   16    // inner dimension tile size

// ============================================================
// Single Processing Element
// ============================================================
void pe_compute(
    hls::stream<double> &a_in,  hls::stream<double> &a_out,
    hls::stream<double> &b_in,  hls::stream<double> &b_out,
    double &c_result
) {
    double acc = 0.0;

    PE_LOOP: for (int k = 0; k < TILE_K; k++) {
        #pragma HLS PIPELINE II=1

        double a_val = a_in.read();
        double b_val = b_in.read();

        acc += a_val * b_val;

        a_out.write(a_val);   // pass A to right neighbour
        b_out.write(b_val);   // pass B to bottom neighbour
    }

    c_result = acc;
}

// ============================================================
// Systolic Array: SA_SIZE × SA_SIZE grid of PEs
// ============================================================
void systolic_array(
    hls::stream<double> a_feed[SA_SIZE],    // SA_SIZE rows of A
    hls::stream<double> b_feed[SA_SIZE],    // SA_SIZE cols of B
    double C_out[SA_SIZE][SA_SIZE]           // output tile
) {
    #pragma HLS DATAFLOW

    // Inter-PE communication channels
    // Horizontal (A flows right): SA_SIZE rows × (SA_SIZE+1) columns
    hls::stream<double> a_pipe[SA_SIZE][SA_SIZE + 1];
    #pragma HLS STREAM variable=a_pipe depth=2

    // Vertical (B flows down): (SA_SIZE+1) rows × SA_SIZE columns
    hls::stream<double> b_pipe[SA_SIZE + 1][SA_SIZE];
    #pragma HLS STREAM variable=b_pipe depth=2

    // Connect input feeds to left edge and top edge
    FEED_A: for (int i = 0; i < SA_SIZE; i++) {
        FEED_A_K: for (int k = 0; k < TILE_K; k++) {
            #pragma HLS PIPELINE II=1
            a_pipe[i][0].write(a_feed[i].read());
        }
    }

    FEED_B: for (int j = 0; j < SA_SIZE; j++) {
        FEED_B_K: for (int k = 0; k < TILE_K; k++) {
            #pragma HLS PIPELINE II=1
            b_pipe[0][j].write(b_feed[j].read());
        }
    }

    // Instantiate the PE grid
    PE_ROWS: for (int i = 0; i < SA_SIZE; i++) {
        PE_COLS: for (int j = 0; j < SA_SIZE; j++) {
            #pragma HLS UNROLL  // physically instantiate all PEs
            pe_compute(
                a_pipe[i][j],     a_pipe[i][j+1],    // A: left → right
                b_pipe[i][j],     b_pipe[i+1][j],    // B: top → bottom
                C_out[i][j]                           // accumulator
            );
        }
    }

    // Drain unused outputs (right edge of A, bottom edge of B)
    // In practice: just let the streams drain or use dummy consumers
}
```

> [!WARNING]
> **This simplified code omits input skewing for clarity.** In a real implementation, you need delay registers before the PE grid to stagger the arrival of different rows/columns. See the "Input Skewing" section above and the SPCL `gemm_hls` reference for production-quality skew logic.

#### Adding Input Skew (The Missing Piece)

```cpp
// Skew buffer: delays row i by i cycles, col j by j cycles
void skew_a_input(
    double A_tile[SA_SIZE][TILE_K],
    hls::stream<double> a_feed[SA_SIZE]
) {
    // Total cycles needed = TILE_K + SA_SIZE - 1 (to drain the pipeline)
    SKEW_CYCLE: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        SKEW_ROW: for (int i = 0; i < SA_SIZE; i++) {
            int k = t - i;  // row i is delayed by i cycles
            if (k >= 0 && k < TILE_K) {
                a_feed[i].write(A_tile[i][k]);
            } else {
                a_feed[i].write(0.0);  // padding zeros during fill/drain
            }
        }
    }
}

// Same pattern for B columns:
void skew_b_input(
    double B_tile[TILE_K][SA_SIZE],
    hls::stream<double> b_feed[SA_SIZE]
) {
    SKEW_CYCLE: for (int t = 0; t < TILE_K + SA_SIZE - 1; t++) {
        #pragma HLS PIPELINE II=1
        SKEW_COL: for (int j = 0; j < SA_SIZE; j++) {
            int k = t - j;
            if (k >= 0 && k < TILE_K) {
                b_feed[j].write(B_tile[k][j]);
            } else {
                b_feed[j].write(0.0);
            }
        }
    }
}
```

**What this produces in hardware:**
```
Cycle  0:  a_feed[0]=A[0][0]  a_feed[1]=0        a_feed[2]=0        a_feed[3]=0
Cycle  1:  a_feed[0]=A[0][1]  a_feed[1]=A[1][0]  a_feed[2]=0        a_feed[3]=0
Cycle  2:  a_feed[0]=A[0][2]  a_feed[1]=A[1][1]  a_feed[2]=A[2][0]  a_feed[3]=0
Cycle  3:  a_feed[0]=A[0][3]  a_feed[1]=A[1][2]  a_feed[2]=A[2][1]  a_feed[3]=A[3][0]
  ...        the diagonal wavefront
```

---

### How This Connects to Google TPU and NVIDIA Tensor Cores

Understanding systolic arrays on your KV260 directly prepares you to discuss the architectures that Google and NVIDIA actually build:

#### Google TPU (v1 through v5)

```
TPU v1 Matrix Multiply Unit (MXU):
┌──────────────────────────────────────┐
│  256 × 256 = 65,536 MAC units       │
│  Weight-stationary dataflow          │
│  INT8/BF16 precision                 │
│  65,536 MACs × 700 MHz = ~92 TOPS   │
│  Connected to unified buffer (24 MB) │
└──────────────────────────────────────┘
```

**Exactly the same architecture as your 4×4 array, scaled up:**
- Your PE does `acc += a * b; pass a right; pass b down` → TPU PE does the same
- Your skew registers create the wavefront → TPU has identical skew logic
- Your output is drained after K iterations → TPU drains to unified buffer
- The *only* differences: (1) 256×256 vs 4×4, (2) INT8 vs FP64, (3) ASIC vs FPGA

**Interview talking point:** *"I implemented an output-stationary systolic array on an FPGA. The TPU's MXU is architecturally identical — it's the same PE structure, same dataflow pattern, same wavefront execution model, just scaled to 256×256 in silicon."*

#### NVIDIA Tensor Cores

Tensor Cores are **small systolic-like units** embedded inside each SM:

```
Ampere SM:
├── 128 CUDA Cores (scalar FP32)
├── 4 Tensor Cores ← "tiny systolic arrays"
│   └── Each: 4×4×4 matrix multiply per cycle
│       └── D = A(4×4) × B(4×4) + C(4×4)
│       └── Supports FP16, BF16, TF32, INT8, FP64
└── Shared Memory / L1 Cache
```

**Key difference from TPU:** Tensor Cores are integrated into the GPU's programmable SM, giving flexibility. The TPU MXU is standalone, giving efficiency. Your KV260 systolic array is closer to the TPU model — a dedicated spatial computing engine connected via DMA.

| | Your KV260 | Google TPU v1 | NVIDIA Tensor Core |
|---|---|---|---|
| Array size | 4×4 to 8×8 | 256×256 | 4×4×4 |
| Dataflow | Output stationary | Weight stationary | Instruction-driven |
| Precision | FP64 (or FP32) | INT8 / BF16 | FP16 / BF16 / TF32 / FP64 |
| Context | Standalone accelerator | Standalone ASIC | Embedded in GPU SM |
| Programmability | Fixed function | Fixed function | Warp-level matrix API (WMMA) |
| Peak ops/cycle | 16–64 MACs | 65,536 MACs | 64 MACs |

---

### KV260 Resource Budget for a Systolic Array

Planning your array size requires knowing what fits:

```
                              Per PE (FP64)    Per PE (FP32)
DSP48E2 slices:               3–4              1
Registers (flip-flops):       ~200             ~100
LUTs:                         ~300             ~150
BRAM:                         0 (uses regs)    0 (uses regs)

KV260 Available:
  DSP48E2:    1,248
  LUTs:       117,120
  Registers:  234,240
  BRAM 36Kb:  144

FP64 systolic array sizes that fit:
  4×4   = 16 PEs  →  ~64 DSPs  →  easily fits (5% of DSPs)
  8×8   = 64 PEs  →  ~256 DSPs →  fits well (20% of DSPs)
  16×16 = 256 PEs →  ~1024 DSPs → tight (82% of DSPs) — may fail timing

FP32 systolic array sizes:
  8×8   = 64 PEs  →  ~64 DSPs  →  easily fits
  16×16 = 256 PEs →  ~256 DSPs →  fits well
  32×32 = 1024 PEs → ~1024 DSPs → tight but achievable
```

**Recommendation:** Start with **4×4 FP64**, verify correctness, then scale to **8×8 FP64**. If targeting maximum GFLOPS for the blog post, consider **16×16 FP32** and using iterative refinement for FP64 accuracy.

---

### The Learning Progression (Updated)

Your original Phase 2 (HLS kernel) should now be split:

#### Phase 2A: UNROLL-Based Tiled DGEMM (1–2 weeks)
- [ ] Implement the Path A kernel (already in this doc)
- [ ] C-simulate, C-synthesize, read the utilization report
- [ ] Understand: broadcast from BRAM, port contention limits
- [ ] **Learning**: HLS pragmas, memory tiling, tool flow

#### Phase 2B: Systolic Array DGEMM (2–3 weeks)
- [ ] Implement a single PE function (5 lines of HLS)
- [ ] Build a 4×4 systolic array with `hls::stream` connections
- [ ] Add input skew logic for rows and columns
- [ ] C-simulate: verify against CPU reference for 4×4, 8×8, 16×16 matrices
- [ ] C-synthesize: compare DSP usage vs UNROLL design
- [ ] Scale to 8×8 → read synthesis report → does it still meet timing at 200 MHz?
- [ ] **Learning**: Spatial computing, dataflow architecture, PE interconnection

#### Phase 2C: Compare and Analyse (3–5 days)
- [ ] Same matrix size, same FPGA → compare UNROLL vs systolic array
- [ ] Metrics: DSP count, BRAM count, max clock frequency, achieved GFLOPS
- [ ] Write up: "Why the systolic array uses fewer BRAM ports but more routing"
- [ ] **Learning**: Architecture tradeoffs, the broadcast vs systolic choice
- [ ] This comparison IS the blog post centrepiece

---

### Open-Source References (Study These)

| Project | What It Is | Why Study It |
|---|---|---|
| [spcl/gemm_hls](https://github.com/spcl/gemm_hls) | Scalable systolic GEMM in Vitis HLS | Production-quality HLS systolic array; achieved 132 GFLOPS FP64 on VCU1525; FPGA'20 paper |
| [Xilinx/Vitis_Accel_Examples/systolic_array](https://github.com/Xilinx/Vitis_Accel_Examples/tree/master/cpp_kernels/systolic_array) | Official Xilinx systolic array tutorial | Simplest starting point; learn the coding patterns |
| [AutoSA](https://github.com/autosa-compiler/autosa) | Automatic systolic array compiler from C code | Generates HLS systolic arrays from loop nests; shows how tiling/mapping works |
| [Xilinx/Vitis_Libraries/blas](https://github.com/Xilinx/Vitis_Libraries/tree/master/blas) | FPGA-optimized BLAS library | Production GEMM with systolic dataflow; L1/L2/L3 implementations |
| [Berkeley Gemmini](https://github.com/ucb-bar/gemmini) | Open-source output-stationary systolic array | RISC-V based; excellent documentation of OS dataflow |

> [!TIP]
> **Start with the Xilinx systolic_array example**, then study the SPCL `gemm_hls` source. The SPCL code achieves 132 GFLOPS FP64 on a VCU1525 — that's a datacenter FPGA, but the architecture is identical to what you'd build on KV260 (just fewer PEs). Their [FPGA'20 paper](https://spcl.inf.ethz.ch/Publications/.pdf/gemm-fpga.pdf) explains every design decision. Read it.

### Key Papers (Read These When Ready)

| Paper | Year | Why Read It |
|---|---|---|
| Kung & Leiserson, *"Systolic Arrays for VLSI"* | 1978 | The original — defines the entire field |
| Jouppi et al., *"In-Datacenter Performance Analysis of a Tensor Processing Unit"* | 2017 | Google's TPU v1 paper — describes the 256×256 systolic MXU |
| de Fine Licht et al., *"Flexible Communication Avoiding Matrix Multiplication on FPGA with HLS"* | 2020 | The SPCL gemm_hls paper — directly relevant to your KV260 work |
| Chen et al., *"Eyeriss: An Energy-Efficient Reconfigurable Accelerator"* | 2017 | Row-stationary dataflow — the most energy-efficient variant |

---

## Layer 3: System Design (Vivado Block Design)

This is where you wire your accelerator into the Zynq system.

### The Architecture

```
┌──────────────────────────────────────────────────────┐
│                  Processing System (PS)               │
│  ┌──────────────┐    ┌──────────────┐                │
│  │ Cortex-A53   │    │   DDR4       │                │
│  │ (runs HPL    │◄──►│  Controller  │                │
│  │  + MPI)      │    │  (4 GB)      │                │
│  └──────┬───────┘    └──────┬───────┘                │
│         │ AXI-Lite          │ AXI HP                  │
│         │ (control)         │ (data)                  │
├─────────┼───────────────────┼────────────────────────┤
│         ▼                   ▼     Programmable Logic  │
│  ┌──────────────┐    ┌──────────────┐                │
│  │  Control     │    │   AXI DMA    │                │
│  │  Registers   │    │  (MM2S/S2MM) │                │
│  └──────────────┘    └──────┬───────┘                │
│                             │ AXI-Stream              │
│                      ┌──────▼───────┐                │
│                      │  Your DGEMM  │                │
│                      │  Accelerator │                │
│                      └──────────────┘                │
└──────────────────────────────────────────────────────┘
```

### What You Build in Vivado Block Design

1. **Zynq PS block** — auto-configured for KV260
2. **AXI DMA** — moves data between DDR and your accelerator via streaming
3. **AXI Interconnect** — routes control signals (AXI-Lite) and data (AXI-Stream/AXI-MM)
4. **Your IP** — the DGEMM accelerator exported from Vitis HLS
5. **Clocking** — PL clock wizard for your target frequency (200–300 MHz)

### Key Concepts
- **AXI-Lite** — slow control bus (read/write registers, start/stop accelerator)
- **AXI-Stream** — high-throughput data bus (streaming tiles in/out of accelerator)
- **AXI Memory-Mapped (AXI-MM)** — direct memory access to DDR from PL
- **AXI DMA** — hardware engine that converts between memory-mapped DDR and streaming
- **Address mapping** — your accelerator's control registers appear at specific memory addresses visible from the ARM cores

### The Vivado Workflow

```
1. Create Vivado Project  →  target: xck26-sfvc784-2LV-c (KV260 part)
2. Import HLS IP          →  from Vitis HLS export_design step
3. Create Block Design    →  drag & drop PS, DMA, your IP, connect signals
4. Validate Design        →  Vivado checks all AXI connections are valid
5. Generate HDL Wrapper   →  turns block design into synthesizable Verilog
6. Run Synthesis          →  maps logic to FPGA resources (~10–30 min)
7. Run Implementation     →  places and routes on actual FPGA fabric (~30–60 min)
8. Generate Bitstream     →  the .bit file that programs the FPGA
9. Export Hardware (.xsa)  →  includes bitstream + memory map for software
```

### Resources
- **UG1393** — [Vitis Unified IDE User Guide](https://docs.amd.com/r/en-US/ug1393-vitis-application-acceleration)
- **UG994** — [Vivado IP Integrator Guide](https://docs.amd.com/r/en-US/ug994-vivado-ip-subsystems)

---

## Layer 4: Linux Driver (PS-Side Software)

Once your hardware is built, you need software on the ARM cores to control it.

### Option A: Bare-Metal (XilDMA library)

Simpler but no Linux. Good for initial testing.

```c
#include "xaxidma.h"
#include "xparameters.h"

#define TILE_SIZE 8

static XAxiDma dma;

void dgemm_fpga(int M, int N, int K,
                double *A, int lda,
                double *B, int ldb,
                double *C, int ldc)
{
    double tile_a[TILE_SIZE * TILE_SIZE];
    double tile_b[TILE_SIZE * TILE_SIZE];
    double tile_c[TILE_SIZE * TILE_SIZE];

    for (int ii = 0; ii < M; ii += TILE_SIZE) {
        for (int jj = 0; jj < N; jj += TILE_SIZE) {
            memset(tile_c, 0, sizeof(tile_c));
            for (int kk = 0; kk < K; kk += TILE_SIZE) {
                // Pack tiles from full matrices
                for (int i = 0; i < TILE_SIZE; i++)
                    for (int j = 0; j < TILE_SIZE; j++)
                        tile_a[i*TILE_SIZE+j] = A[(ii+i)*lda + (kk+j)];
                for (int i = 0; i < TILE_SIZE; i++)
                    for (int j = 0; j < TILE_SIZE; j++)
                        tile_b[i*TILE_SIZE+j] = B[(kk+i)*ldb + (jj+j)];

                // Flush cache (ARM cache ≠ DDR — DMA reads from DDR)
                Xil_DCacheFlushRange((UINTPTR)tile_a, sizeof(tile_a));
                Xil_DCacheFlushRange((UINTPTR)tile_b, sizeof(tile_b));

                // Start accelerator
                Xil_Out32(ACCEL_BASE + 0x00, 1);

                // DMA: send tile_a → PL
                XAxiDma_SimpleTransfer(&dma, (UINTPTR)tile_a,
                    sizeof(tile_a), XAXIDMA_DMA_TO_DEVICE);
                while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE));

                // DMA: send tile_b → PL
                XAxiDma_SimpleTransfer(&dma, (UINTPTR)tile_b,
                    sizeof(tile_b), XAXIDMA_DMA_TO_DEVICE);
                while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE));

                // DMA: receive tile_c ← PL
                XAxiDma_SimpleTransfer(&dma, (UINTPTR)tile_c,
                    sizeof(tile_c), XAXIDMA_DEVICE_TO_DMA);
                while (XAxiDma_Busy(&dma, XAXIDMA_DEVICE_TO_DMA));

                Xil_DCacheInvalidateRange((UINTPTR)tile_c, sizeof(tile_c));
            }
            // Unpack result
            for (int i = 0; i < TILE_SIZE; i++)
                for (int j = 0; j < TILE_SIZE; j++)
                    C[(ii+i)*ldc + (jj+j)] += tile_c[i*TILE_SIZE+j];
        }
    }
}
```

### Option B: PetaLinux + UIO/VFIO (Production)

For HPL integration you need Linux (MPI requires it). Use PetaLinux to build a Linux image that includes your hardware platform.

**Key concepts:**
- **Device Tree** — tells Linux what hardware exists and where (addresses, interrupts)
- **UIO (Userspace I/O)** — maps hardware registers directly to userspace (simplest)
- **DMA-BUF** — Linux kernel framework for DMA-capable memory allocation
- **Cache coherency** — ARM caches and DMA don't automatically agree; you need explicit flush/invalidate

### Resources
- **UG1144** — [PetaLinux Reference Guide](https://docs.amd.com/r/en-US/ug1144-petalinux-tools-reference-guide)
- **UG1393** — Vitis application acceleration flow (explains the XRT runtime)

---

## Layer 5: HPL Integration

The final step: making HPL call your FPGA accelerator instead of the CPU BLAS library.

### How HPL Uses BLAS

HPL calls standard BLAS routines. The critical one is:
```c
cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
            M, N, K, alpha, A, lda, B, ldb, beta, C, ldc);
```

### Integration Strategy

1. **Build a custom BLAS library** that implements `cblas_dgemm` using your FPGA driver
2. **Link HPL against your library** instead of OpenBLAS/ATLAS
3. **Fallback logic**: For small matrices, use CPU. For large tiles, offload to FPGA.

The overhead of DMA transfers means the FPGA only wins for tiles above a certain size — you'll need to benchmark to find the crossover point.

---

## Complete Learning Path (Ordered)

### Phase 1: Foundations (1–2 weeks)
- [ ] Install Vivado + Vitis HLS (2024.1 or later)
- [ ] Download KV260 BSP and platform files
- [ ] Run the Xilinx "Hello World" accelerator tutorial on KV260
- [ ] Read UG1399 chapters 1–4 (HLS concepts)
- [ ] Complete 2–3 Vitis-Tutorials examples (vector add, matrix multiply)

### Phase 2: DGEMM Kernel (2–3 weeks)
- [ ] Write DGEMM tile kernel in Vitis HLS (start with TILE=4, then 8)
- [ ] C-simulate for correctness against a CPU reference
- [ ] C-synthesize — read the resource utilization report (DSPs, BRAM, clock)
- [ ] Iterate on pragmas: tune PIPELINE, UNROLL, ARRAY_PARTITION
- [ ] Co-simulate to verify RTL matches C behavior
- [ ] Export IP for Vivado

### Phase 3: System Integration (1–2 weeks)
- [ ] Create Vivado block design: Zynq PS + DMA + your IP
- [ ] Generate bitstream
- [ ] Write bare-metal test application — send a tile, verify output
- [ ] Measure latency and throughput

### Phase 4: Linux & HPL (2–3 weeks)
- [ ] Build PetaLinux image with your hardware platform
- [ ] Write userspace driver (UIO or XRT)
- [ ] Build custom BLAS shim that calls FPGA for large tiles
- [ ] Link HPL against your BLAS
- [ ] Run HPL, measure GFLOPS, compare against CPU-only baseline
- [ ] Blog post: "FPGA-Accelerated HPL on KV260"

---

## Key Gotchas

> [!WARNING]
> ### Things that will trip you up

1. **Memory bandwidth is your real bottleneck** — 4 GB DDR4 @ 64-bit gives ~12.8 GB/s. An 8×8 systolic array at 200 MHz consuming 64-bit values needs ~25.6 GB/s for A+B feeds. You MUST maximize tile reuse in BRAM/URAM.

2. **Cache coherency** — The ARM cores have L1/L2 caches. DMA reads from DDR, not cache. If you forget to flush the cache before DMA and invalidate after, your data will be stale/wrong. This is the #1 cause of "it works in simulation but not on hardware."

3. **Build times are long** — Vivado synthesis: 10–30 min. Implementation: 30–60 min. Bitstream: 10 min. Budget for this when iterating. HLS C-simulation is seconds — do your debugging there.

4. **FP64 is expensive on FPGA** — Consider starting with FP32 (1 DSP per MAC instead of 4) to get the architecture working, then switch to FP64. Mixed-precision HPL (iterative refinement) is a well-studied technique.

5. **KV260 is an edge/vision board, not an HPC card** — Your achievable GFLOPS will be modest. The value is in the **learning** and the **architecture patterns** that transfer to larger Alveo/Versal/datacenter FPGAs.
