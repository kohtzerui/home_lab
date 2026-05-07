# Performance Profiling for HPL — `perf`, Flamegraphs & MLPerf

## Why This Matters

Running HPL and getting a GFLOPS number is **not enough** — anybody can run `./xhpl`.
What makes your project memorable to Google/NVIDIA recruiters is **understanding *why* your system performs the way it does**, being able to attribute bottlenecks precisely, and showing iterative improvement. That's the engineering.

**The story you want to tell:**

> *"I ran HPL on my 3-node cluster, profiled the bottleneck with perf + flamegraphs, discovered it was [MPI lat / NUMA effect / cache pressure / memory bandwidth], fixed it, and recovered X% performance."*

This section integrates profiling tools into your existing DGEMM + HPL project so that every benchmark you run produces a **learning artifact**, not just a number.

---

## Mental Model: Three Profiling Layers

```
Layer 3: MLPerf / HPL              "Am I competitive vs the world?"
                  ↑
Layer 2: perf stat + flamegraphs   "Where does time go on the CPU?"
                  ↑
Layer 1: nsys / ncu                "What's happening inside my GPU kernels?"
```

You should understand all three layers and be able to navigate between them:
- MLPerf/HPL tells you **what** the final number is
- `perf` tells you **where** the CPU bottleneck is
- `ncu` tells you **why** the GPU kernel is slow

---

## Part 1: `perf` — Linux CPU Profiling

### What `perf` Measures

`perf` reads **hardware performance counters** built into every modern CPU. These counters tick in real time as your program runs. They measure physical events — cache hits, branch mispredictions, stalled cycles — directly from silicon.

```
Intel/AMD CPU
├── Core 0
│   ├── PMU (Performance Monitoring Unit)
│   │   ├── Counter 0: "instructions retired"      ← perf can read this
│   │   ├── Counter 1: "LLC cache misses"          ← perf can read this
│   │   ├── Counter 2: "branch mispredictions"     ← perf can read this
│   │   └── Counter 3: "stalled cycles"            ← perf can read this
│   └── Core logic
└── Core 1
    └── ...
```

### `perf stat` — The Starting Point

Run this on every HPL experiment:

```bash
# Basic hardware counter summary
perf stat -e \
  instructions,cycles,\
  cache-misses,cache-references,\
  branch-misses,branches,\
  stalled-cycles-frontend,stalled-cycles-backend \
  mpirun -np 4 ./xhpl

# IPC shorthand (summary only)
perf stat ./xhpl
```

**What to look for:**

| Metric | Formula | Healthy Range | Red Flag |
|---|---|---|---|
| IPC | `instructions / cycles` | 1.5–3.5 for HPL | < 1.0 = stalling |
| Cache miss rate | `cache-misses / cache-references` | < 1% | > 5% = memory-bound |
| Branch miss rate | `branch-misses / branches` | < 1% | > 5% = bad prediction |
| Frontend stall % | `stalled-cycles-frontend / cycles` | < 10% | > 30% = decode bottleneck |
| Backend stall % | `stalled-cycles-backend / cycles` | < 20% | > 50% = execution/memory bottleneck |

**For HPL specifically** — you expect to be **backend stalled** on memory bandwidth.
DGEMM is compute-intensive once tiled, but HPL also does triangular solves (DTRSM) and panel factorizations (DGETRF) which can be cache-bound.

```bash
# NUMA-aware: see per-socket memory events on a multi-socket machine
perf stat -e node-loads,node-load-misses,node-prefetch-misses ./xhpl
```

### `perf record` + `perf report` — Call-level Profiling

```bash
# Record with call graph (stack traces)
perf record -F 999 -g --call-graph dwarf mpirun -np 4 ./xhpl

# Interactive TUI to explore
perf report

# Or dump to text
perf report --stdio | head -100
```

`perf report` shows you a ranked list of **functions by % CPU time**. For HPL you expect to see:
- `dgemm_` (BLAS DGEMM) — should dominate (~80%+)
- `dtrsm_` (triangular solve)
- `dlaswp_` (row swaps / pivoting)
- `MPI_Allreduce`, `MPI_Send` etc. (communication)

If `MPI_Allreduce` or `MPI_Send` appears at > 15%, you have a **communication bottleneck** — tune your HPL.dat `P × Q` ratio or NB block size.

---

## Part 2: Flamegraphs

### What They Are

A flamegraph is a visual rendering of `perf record` stack traces. Width = time. Height = call depth.

```
                  █ dgemm_ (BLAS)      ← 78% of time here (GOOD)
        █████████████████████
     ██ dtrsm_   █ dlaswp_
 ████████████████████████████████
 HPL main loop
─────────────────────────────────── time →
```

**Wide flat bars** = hot spots. **Tall towers** = deep call stacks (often recursive).

### Setup (One-time)

```bash
# Clone Brendan Gregg's tools
git clone https://github.com/brendangregg/FlameGraph.git
cd FlameGraph

# Verify perf is installed
perf --version
```

### Generate a Flamegraph for HPL

```bash
# 1. Record HPL (99 Hz sampling, 60 seconds)
perf record -F 99 -g --call-graph dwarf -o perf.data \
    mpirun -np 4 ./xhpl

# 2. Export stack traces
perf script -i perf.data > out.perf

# 3. Collapse stacks
./FlameGraph/stackcollapse-perf.pl out.perf > out.folded

# 4. Render SVG (interactive)
./FlameGraph/flamegraph.pl out.folded > hpl_flamegraph.svg

# Open in browser — click to zoom into any frame
```

### What to Look For in an HPL Flamegraph

**Good pattern** (compute-bound):
```
████████████████████████████████████████  dgemm_   (BLAS)       ~ 80%
████  dtrsm_  ████  dlaswp_  ██ MPI     ← secondary operations  ~ 20%
```

**Bad pattern 1** (DGEMM not using all your CPU time):
```
████████████  dgemm_   ███████████████ MPI_Allreduce             ← latency-bound
```
Fix: Adjust HPL.dat P×Q grid; increase NB block size; check network latency.

**Bad pattern 2** (BLAS calling many small functions):
```
█ sgemm_ █ memset █ malloc █ memcpy █ sgemm_ █ memset ...       ← call overhead
```
Fix: HPL NB is too small; increase it (try 128, 192, 256).

**Bad pattern 3** (NUMA effects):
```
████████ numa_alloc ████ migrate_pages ████ dgemm_               ← memory migration
```
Fix: Pin processes with `numactl --membind=0 --cpunodebind=0`.

---

## Part 3: Integrating Into Your HPL Project

### HPL.dat Knobs and What They Control

```
HPL.dat parameters that affect profiling outcomes:

Ns     = problem size        → bigger = more compute-bound, fewer edge effects
NBs    = block size          → too small = overhead; too large = cache miss; sweet spot ~192-256
Ps, Qs = process grid shape  → P×Q = total MPI ranks; P≈Q is ideal (minimize surface area)
PFACTs = panel factorization → affects DGETRF hot path
```

### Experiment Protocol (Max Learning)

For each experiment, run the same 3 commands:

```bash
# 1. Run HPL and capture stdout (overall GFLOPS)
mpirun -np 4 ./xhpl 2>&1 | tee results/exp_NB192_4proc.log

# 2. perf stat (hardware counters summary)
perf stat -e instructions,cycles,cache-misses,stalled-cycles-backend \
    mpirun -np 4 ./xhpl 2>&1 | tee results/exp_NB192_4proc_perfstat.log

# 3. Flamegraph (visual profile)
perf record -F 99 -g --call-graph dwarf -o results/exp_NB192_4proc_perf.data \
    mpirun -np 4 ./xhpl
perf script -i results/exp_NB192_4proc_perf.data | \
    ./FlameGraph/stackcollapse-perf.pl | \
    ./FlameGraph/flamegraph.pl > results/exp_NB192_4proc_flamegraph.svg
```

This gives you: **one number (GFLOPS) + one table (counters) + one image (flamegraph)** per experiment.

### Variable This Experiment Matrix

Run this grid — each row teaches you something different:

| Experiment | Variable Changed | What You Learn |
|---|---|---|
| `baseline_NB64`  | NB = 64  | Small block: overhead-bound |
| `baseline_NB128` | NB = 128 | Medium block |
| `baseline_NB192` | NB = 192 | Near-optimal usually |
| `baseline_NB256` | NB = 256 | Large block: cache pressure |
| `grid_1x4`       | P=1, Q=4 | Tall grid: column comm heavy |
| `grid_2x2`       | P=2, Q=2 | Square grid: balanced |
| `grid_4x1`       | P=4, Q=1 | Wide grid: row comm heavy |
| `numa_unbound`   | No numactl | NUMA effects visible |
| `numa_bound`     | `numactl --membind` | NUMA fixed |
| `blas_openblas`  | OpenBLAS | vs |
| `blas_mkl`       | Intel MKL | BLAS library comparison |

**For each row**: record GFLOPS, IPC, cache miss rate, and save the flamegraph SVG.  
Then **tell the story** in your blog post: "When I changed X, the flamegraph showed Y changed, and the hardware counter Z moved from A to B."

---

## Part 4: GPU Side — nsys + ncu (Your CUDA DGEMM)

These complement `perf` by profiling *inside* the GPU kernel.

```bash
# Nsight Systems — timeline view (zoom out)
nsys profile --stats=true -o results/cuda_dgemm_timeline \
    ./cuda_dgemm 4096 4096 4096

# Nsight Compute — kernel-level metrics (zoom in)
ncu --set full --page raw \
    -o results/cuda_dgemm_kernel \
    ./cuda_dgemm 4096 4096 4096

# Key metrics in ncu output:
#   sm__throughput.avg.pct_of_peak_sustained_elapsed   → SM busy %
#   l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum       → global mem reads
#   smsp__sass_thread_inst_executed_op_dfma_pred_on.sum → actual FP64 FMAs executed
```

### ncu Roofline Analysis

```bash
# Generate roofline model (shows if you're compute-bound or memory-bound)
ncu --set roofline -o roofline_report ./cuda_dgemm
ncu-ui roofline_report.ncu-rep   # Open GUI
```

The roofline plot shows your kernel as a point:
- **Left of the ridge point** = memory-bandwidth bound → you need better tiling
- **Right of the ridge point** = compute-bound → you're doing well, squeeze more FLOPs

---

## Part 5: MLPerf Context

You won't submit results to MLPerf (they require certified hardware + runs), but you use MLPerf as a **reference baseline** for your own analysis.

### How to Use MLPerf Results Without Submitting

```bash
# Download MLPerf HPL reference results
# https://mlcommons.org/benchmarks/hpc/

# For comparison: what does an A100 cluster achieve per node?
# MLPerf HPC v3.0 results: ~10-20 TFLOPS per A100 per node (HPL)
# Your RTX 3060: ~200 GFLOPS FP64 theoretical peak
# Goal: achieve > 60% of peak → ~120+ GFLOPS on your DGEMM
```

### Framing in Your Blog / Resume

> *"Benchmarked DGEMM throughput against MLPerf HPC v3.0 reference targets, achieving X% efficiency at FP64 on an RTX 3060. Profiled with Nsight Compute to identify [specific bottleneck] as the limiting factor."*

That single sentence tells a recruiter you know: (1) industry standards, (2) profiling tools, (3) what efficiency means in context.

---

## Part 6: Full Learning Progression

### Phase A: CPU HPL Baseline (Before You Buy the GPU)

**Goal:** Understand what "bottleneck analysis" means on a single machine.

```
Week 1
 ├── Run HPL on your Beelink S12 Pro (CPU only)
 ├── perf stat → record IPC, cache miss rate
 ├── Generate first flamegraph
 ├── Try 3 different NB values → plot GFLOPS vs NB
 └── Write notes: "my IPC was X, cache miss rate was Y, DGEMM took Z% of time"
```

**Learning outcome:** You understand HPL's hot path and what the hardware counters mean. You have a flamegraph you can put in your blog.

### Phase B: MPI Multi-Process HPL

```
Week 2
 ├── Run HPL with 1, 2, 4 MPI ranks
 ├── perf stat per-rank → is MPI overhead growing?
 ├── flamegraph → does MPI_Allreduce widen?
 ├── Try P×Q grids (1×4, 2×2, 4×1)
 └── Plot: GFLOPS vs #ranks, #ranks vs MPI% of flamegraph
```

**Learning outcome:** You can show scaling efficiency and attribute overhead to communication vs compute. Classic HPC systems analysis.

### Phase C: CUDA DGEMM + nsys/ncu

```
Week 3–5 (aligns with your existing CUDA learning plan)
 ├── Implement naive → tiled → register-tiled SGEMM
 ├── Profile each with ncu: check roofline position
 ├── Document: "naive was memory-bound (left of roofline), tiled crossed the ridge"
 ├── Port to DGEMM
 └── Compare RTX 3060 DGEMM vs Beelink CPU DGEMM (perf perspective)
```

**Learning outcome:** You can navigate both CPU and GPU profiling tools. The FPGA vs CPU vs GPU comparison is your blog post centrepiece.

### Phase D: KV260 FPGA + Cross-Platform Analysis

```
Week 6–8 (aligns with your FPGA plan)
 ├── Run DGEMM on KV260 via Vitis HLS
 ├── Measure: GFLOPS, GFLOPS/Watt, latency
 ├── Profile CPU host side with perf (DMA overhead, PCIe latency)
 └── Three-way comparison: FPGA vs CPU vs GPU on a single chart
```

**The grand comparison table for your blog:**

| Platform | GFLOPS (FP64) | Power (W) | GFLOPS/Watt | Bottleneck (profiler says) |
|---|---|---|---|---|
| Beelink S12 Pro (CPU) | ~X | ~15W | X/15 | LLC cache miss (perf stat) |
| RTX 3060 (GPU) | ~Y | ~170W | Y/170 | FP64 unit scarcity (ncu roofline) |
| KV260 FPGA | ~Z | ~10W | Z/10 | DSP48 count limit (Vivado report) |

This table + flamegraphs + ncu roofline screenshots = a blog post that any HPC engineer will respect.

---

## Cheat Sheet: Commands to Run Every Session

```bash
# ==== BEFORE benchmarking: set CPU to performance mode ====
sudo cpupower frequency-set -g performance
sudo sh -c "echo 1 > /proc/sys/kernel/perf_event_paranoid"  # allow perf

# ==== Run HPL + profile ====
perf stat -e instructions,cycles,cache-misses,stalled-cycles-backend \
    mpirun -np $(nproc) ./xhpl

# ==== Full flamegraph ====
perf record -F 99 -g --call-graph dwarf mpirun -np $(nproc) ./xhpl && \
perf script | ~/FlameGraph/stackcollapse-perf.pl | \
~/FlameGraph/flamegraph.pl > flamegraph_$(date +%Y%m%d_%H%M).svg

# ==== CUDA kernel profiling ====
ncu --set full -o profile_$(date +%Y%m%d) ./cuda_dgemm 4096 4096 4096
nsys profile --stats=true -o timeline_$(date +%Y%m%d) ./cuda_dgemm 4096 4096 4096

# ==== Quick GFLOPS estimate (from HPL output) ====
grep "WR" HPL.out | awk '{print $7, "GFLOPS"}'
```

---

## Resources

| Resource | What It Covers |
|---|---|
| [Brendan Gregg — Flamegraphs](https://www.brendangregg.com/flamegraphs.html) | Original author's guide — the canonical reference |
| [Brendan Gregg — perf Examples](https://www.brendangregg.com/perf.html) | Massive collection of perf command examples |
| [Intel VTune Profiler](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html) | Alternative to perf if on Intel (more GUI) |
| [MLCommons HPC Benchmarks](https://mlcommons.org/benchmarks/hpc/) | Download reference results for comparison |
| [HPL Tuning Guide](https://netlib.org/benchmark/hpl/faqs.html) | Official HPL FAQ — NB, P, Q tuning |
| [NVIDIA Nsight Compute Docs](https://docs.nvidia.com/nsight-compute/) | ncu metrics reference |
| [Roofline Model Paper](https://dl.acm.org/doi/10.1145/1498765.1498785) | Original Williams et al. paper (worth reading) |

> [!TIP]
> **Blog post outline that emerges naturally from this work:**
> 1. "What HPL actually does" (intro)
> 2. "My baseline flamegraph — where time goes" (perf + flamegraph)
> 3. "Tuning NB: a hardware counter story" (perf stat table)
> 4. "Moving to the GPU: Nsight Compute roofline" (ncu)
> 5. "FPGA vs GPU vs CPU: a fair comparison" (grand table)
> 
> Each section has a visual artifact (flamegraph SVG, ncu screenshot, chart). That's a post that takes someone from zero to understanding heterogeneous HPC benchmarking.
