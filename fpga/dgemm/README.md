# FPGA DGEMM — KV260 Systolic Array Accelerator

Vitis HLS kernel implementing an output-stationary systolic array for General Matrix-Matrix Multiplication (GEMM) on the Xilinx KV260 (UltraScale+).

## Directory Layout

```
fpga/dgemm/
├── src/
│   ├── systolic_dgemm_v7.cpp   ← Optimised kernel  (Level 7)
│   └── dgemm_naive.cpp         ← Naive baseline    (Level 0, double, no pragmas)
└── tb/
    └── tb_dgemm.cpp            ← Testbench: correctness + benchmark comparison
```

## Architecture

| Parameter | Value | Rationale |
|---|---|---|
| Array size | 8×8 = 64 PEs | Fits KV260 DSP budget with headroom |
| Tile K | 16 | Power-of-2, bitwise addressing |
| Weight type | `ap_fixed<27,13>` | Fits DSP48E2 A-port (27-bit max) |
| Activation type | `ap_fixed<18,9>` | Fits DSP48E2 B-port (18-bit max) |
| Accumulator | `ap_fixed<48,22>` | Maps to DSP48E2 48-bit P-register |
| Skewing | SRL (shift register LUT) | Zero arithmetic, scales freely |
| Dataflow | Functions only, no loops inside | Avoids FIFO bloat |

## Key Design Rules (learned the hard way)

1. **`ap_fixed<32,x>` does NOT map to 1 DSP48E2** — 32 bits exceeds the 27-bit A-port, causing DSP cascading. Use `ap_fixed<27,13>` × `ap_fixed<18,9>`.
2. **Never use `AP_SAT` on the accumulator** — saturation logic breaks the DSP's internal P-register feedback loop and spills to LUTs.
3. **No raw loops inside `#pragma HLS DATAFLOW`** — use functions only, or HLS infers oversized FIFOs.
4. **SRL over arithmetic masking** — `BIND_STORAGE impl=srl` is one LUT per bit of delay, not comparators + MUX chains.

## Expected Performance (post-synthesis)

| Metric | Target |
|---|---|
| Clock | 250 MHz |
| Throughput | ~80 GOPS |
| DSP usage | ~64 / 1248 |
| BRAM usage | ~6 / 144 |

## How to Benchmark

### Step 1 — C-Simulation (verify correctness of both kernels)

```tcl
open_project dgemm_hls
add_files src/systolic_dgemm_v7.cpp
add_files src/dgemm_naive.cpp
add_files -tb tb/tb_dgemm.cpp
open_solution sol_v7
set_part xczu5ev-sfvc784-2-e
create_clock -period 4 -name default
csim_design
```

Expected output:
```
PASS: dgemm_naive (err=~0 < tol=1e-9)
PASS: systolic_v7  (err=<0.01 < tol=0.01)
```

### Step 2 — C-Synthesis: Optimised kernel

```tcl
set_top systolic_dgemm
csynth_design
# Check: solution/syn/report/systolic_dgemm_csynth.rpt
#   PE_LOOP II = 1   (target)
#   DSP    ~64       (target)
```

### Step 3 — C-Synthesis: Naive baseline

```tcl
set_top dgemm_naive
csynth_design
# Check: solution/syn/report/dgemm_naive_csynth.rpt
#   INNER_K II = 5+   (expected — no pipeline pragma)
#   DSP     3-4       (FP64 multiplier)
```

### Expected benchmark comparison

| Metric | `dgemm_naive` | `systolic_dgemm_v7` | Ratio |
|---|---|---|---|
| Loop II | 5–20 | 1 | ~10× |
| Latency (64×64) | ~1.3M cycles | ~5,900 cycles | ~220× |
| DSP | 3–4 (sequential) | 64 (parallel) | 16× |
| Est. GFLOPS | ~0.05 | ~80 | ~1,600× |

## Build for IP Export (optimised kernel only)

```tcl
open_project dgemm_hls
set_top systolic_dgemm
add_files src/systolic_dgemm_v7.cpp
open_solution sol1
set_part xczu5ev-sfvc784-2-e
create_clock -period 4 -name default
csynth_design
export_design -format ip_catalog
```

## References

- `learning/application/fpga_dgemm_kernel_optimisation.md` — full optimisation walkthrough (Levels 0–7)
- AMD UG1399 — Vitis HLS User Guide (BIND_OP, BIND_STORAGE, DATAFLOW semantics)
- AMD UG579 — UltraScale DSP48E2 User Guide (port geometry)
- [spcl/gemm_hls](https://github.com/spcl/gemm_hls) — production reference implementation
