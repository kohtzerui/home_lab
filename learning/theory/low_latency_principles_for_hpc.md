# Low-Latency Systems Principles — Applied to CUDA & HPC

> Notes derived from David's CppCon talk: *"Low Latency Trading Systems in C++"*
> Mapped to your specific work: **CUDA DGEMM kernels + HPL benchmarking on NSCC A100**

The talk is about trading systems but every single principle in it translates
directly to GPU kernel engineering. The underlying problem is identical:
**squeeze maximum useful work out of hardware, limited by memory bandwidth and
cache hierarchy.**

---

## Principle 1 — Mechanical Sympathy: Write Algorithms That Match Your Hardware

### What the talk said
Linear search on a reversed vector beat `std::map`, branchless binary search,
and every "clever" data structure — because it maps perfectly to how a CPU
prefetcher works. Sequential memory access = free hardware prefetching.

### What this means for your CUDA kernels

The GPU equivalent of "cache locality" is **coalesced global memory access**.
When threads in the same warp (32 threads) access consecutive memory addresses,
the GPU coalesces them into a single 128-byte transaction. When they don't,
you get 32 separate transactions — a 32× bandwidth penalty.

```cuda
// ❌ UNCOALESCED — thread i reads column i of a row-major matrix
// Each thread jumps by N elements (stride = N words apart)
float val = A[row * N + threadIdx.x * N];   // strided access

// ✓ COALESCED — thread i reads element i in a row
// Consecutive threads → consecutive memory addresses
float val = A[row * N + threadIdx.x];       // sequential access
```

**Rule of thumb:** your innermost loop index should move with `threadIdx.x`.
If it doesn't, your kernel is memory-bound for the wrong reason.

**For your naive DGEMM kernel:** make sure the K-loop (the dot product loop)
is the inner loop and that thread-to-column mapping is sequential.

---

## Principle 2 — Understand Your Memory Hierarchy Before Optimising Anything

### What the talk said
He showed a graph where L1 → L2 → L3 → RAM caused visible performance cliffs.
His core message: **you must know which level of the hierarchy you are in
before you start "optimising."**

### The CUDA Memory Hierarchy (your actual working environment)

```
Level           Size (A100)   Latency       Notes
──────────────────────────────────────────────────────────────────
Registers       ~65K / SM     ~1 cycle      Fastest. Each thread has its own.
Shared Memory   48–164 KB/SM  ~5–10 cycles  Explicitly managed. Your #1 tool.
L1 Cache        ~192 KB/SM    ~20–30 cy     Auto-managed. Also backs shared mem.
L2 Cache        40 MB total   ~200 cycles   Chip-wide, shared across all SMs.
HBM (Global)    80 GB total   ~600 cycles   Main GPU memory. Very high bandwidth
                                             (~2 TB/s on A100) but high latency.
PCIe / NVLink   —             ~µs           Host ↔ Device transfer. Minimize this.
```

**The key question for every kernel you write:**
> *"Am I fitting my working set into shared memory / registers, or am I
> repeatedly hitting HBM?"*

For a DGEMM kernel with tile size T×T:
- Working set per tile = `2 × T × T × 8 bytes` (one A tile + one B tile, FP64)
- Shared memory limit ≈ 48 KB default → `T = sqrt(48000 / 16) ≈ 54` → use T=32 or T=64

---

## Principle 3 — Profile First, Optimise Second. Never Guess.

### What the talk said
He used Intel's **Top-Down Microarchitecture Analysis (TMA)** before measuring
specific counters. The 4 categories are exhaustive and non-overlapping:
- **Retiring** — useful work
- **Bad Speculation** — wasted work from branch mispredictions
- **Front-End Bound** — instruction fetch/decode bottleneck
- **Back-End Bound** — memory stall

He found 25% Bad Speculation, traced it to binary search branches with `perf record`.

### The CUDA equivalent: Nsight Compute's Roofline + SOL

Nsight Compute gives you the same 4-level breakdown, GPU-translated:

| CPU TMA Category   | GPU Nsight Equivalent             | What to look for                     |
|--------------------|-----------------------------------|--------------------------------------|
| Retiring           | **SM Throughput %**               | Are SMs actually busy?               |
| Bad Speculation    | **Warp Stalls (Execution Dep.)**  | Register dependency chains           |
| Front-End Bound    | **Warp Stalls (Instruction)**     | Instruction cache misses             |
| Back-End Bound     | **Memory Throughput %**           | HBM bandwidth utilisation            |

**The Roofline tells you which ceiling you're hitting:**
```
If your kernel is in the "Memory-Bound" region of the roofline:
    → Fix: increase arithmetic intensity (tile size, register blocking)
    → Fix: improve coalescing (reorder matrix layout)

If your kernel is in the "Compute-Bound" region:
    → Fix: reduce instruction count (fused multiply-add, FMA)
    → Fix: increase occupancy (tune block size, shared mem usage)
```

**Practical workflow (identical to what David described):**
```bash
# Step 1: Quick overview — where is the bottleneck category?
ncu --set full ./dgemm_kernel > profile_run1.txt

# Step 2: Find the specific metric within that category
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
              l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
              smsp__sass_thread_inst_executed_op_ffma_pred_on.sum \
    ./dgemm_kernel

# Step 3: Fix that one thing. Repeat.
```

> **Never micro-optimise (e.g., "let me try `__ldg` on this load") before
> you know what category of bottleneck you have.** This is the most common
> mistake. David said exactly this: "Engineers get excited and start measuring
> everything and nothing."

---

## Principle 4 — Eliminate Branches Inside Your Hot Loop

### What the talk said
30% of CPU cycles were spent on two conditional jumps inside `std::lower_bound`
(binary search). The fix: branchless binary search, which cut branch misses in
half.

### What this means for CUDA: Warp Divergence

In CUDA, the equivalent is **warp divergence**. A warp is 32 threads that
execute in lockstep. If threads in the same warp take different branches of an
`if/else`, both paths execute serially — you lose up to 50% throughput.

```cuda
// ❌ DIVERGENT — threads in the same warp will take different paths
// if threadIdx.x varies across the warp
if (threadIdx.x < some_threshold) {
    // path A — some threads
} else {
    // path B — other threads
}

// ✓ NON-DIVERGENT — all threads in a warp take the same path
// (boundary conditions handled outside the hot loop)
// Pad your matrix dimensions to multiples of the tile size.
// Eliminate the "if (row < N && col < M)" guard inside the K-loop.
```

**For your DGEMM:** the most common source of divergence is the boundary
check `if (tile_row < N && tile_col < M)`. Fix: pad A and B to multiples of
tile size with zeros before launching the kernel.

---

## Principle 5 — Simplicity and Performance Are Not Opposites

### What the talk said
The fastest order book implementation (linear search on a reversed vector)
was also the simplest — ~10 lines of code. He had measured 30 different
"clever" implementations. Simple won.

### What this means for your DGEMM progression

This is the justification for the **step-by-step kernel progression** in your
learning plan:

```
Naive kernel → Tiled (shared mem) → Register-blocked → + Vectorisation
```

Each step adds exactly one complexity. Each step has a measurable roofline
improvement. **Stop when the complexity of the next step exceeds its gain.**

For a portfolio blog post, showing a graph like:

```
Naive:              2.1 TFLOPS   (21% of A100 FP64 peak)
+ Tiled shmem:      6.8 TFLOPS   (70% of peak)
+ Register block:   8.4 TFLOPS   (86% of peak)
cuBLAS reference:   9.0 TFLOPS   (93% of peak)
```

...tells a better story than showing only the final number. The *journey*
through the hierarchy is the portfolio artifact.

---

## Principle 6 — "You Are Not Alone": Think at the System Level

### What the talk said
Running 6 independent processes caused near-complete L3 cache contention —
each process's throughput dropped to ~1/6. The point: **your code's performance
depends on what else is sharing the hardware.**

### What this means for your HPL runs at NSCC

This is exactly why HPL uses a **P×Q process grid** and why MPI rank placement
matters:

```
# Bad: all ranks on same node, competing for L3 + HBM bandwidth
mpirun -np 8 ./xhpl    # <- all 8 ranks might land on same NUMA node

# Good: pin ranks to specific cores, aware of NUMA topology
mpirun -np 8 --map-by socket --bind-to core ./xhpl
```

On an A100 node at NSCC:
- The CPU and GPU are connected by **PCIe** (or NVLink).
- MPI ranks that are on the "wrong" NUMA domain will have PCIe traffic
  crossing a NUMA boundary → extra latency.
- Use `numactl` or OpenMPI's `--map-by numa` to ensure CPU memory is local
  to the MPI rank's CPU socket.

The CUDA equivalent: if you have multi-GPU, make sure each GPU's data is
allocated on the host memory bank closest to that GPU.

---

## The 8-Principle Cheat-Sheet (Translated to GPU)

| David's Principle | GPU/HPC Translation |
|---|---|
| 1. No node containers (no pointer-chasing) | No indirection in hot kernels. Use flat arrays, not linked structures. |
| 2. Understand your problem domain | Know your arithmetic intensity before writing a single line. |
| 3. Leverage domain-specific properties | DGEMM is compute-bound above a certain tile size → use that. |
| 4. Simple + fast = done right | Stop adding complexity when gain < noise. |
| 5. Mechanical sympathy | Coalesce memory. Avoid divergence. Fill registers. |
| 6. Bypass what you don't need | Don't use unified memory by default. Use explicit `cudaMemcpy`. |
| 7. Right tool for right task | cuBLAS for production. Hand-rolled kernel for learning/profiling. |
| 8. Staying fast is harder than getting fast | Add Nsight Compute metrics to your CI, not just to one-off runs. |

---

## Profiling Tools Comparison (CPU talk → GPU equivalent)

| David's Tool | What it does | Your GPU Equivalent |
|---|---|---|
| `perf stat` | Aggregate hardware counters | `ncu --set basic` |
| `perf record` | Sampling profiler, finds hot functions | `nsys profile` (Nsight Systems) |
| Hardware PMCs (programmatic) | Precisely count cycles around hot code | `ncu --metrics` on specific kernel |
| Clang X-Ray | Zero-cost, patchable instrumentation | CUDA `nvtx` range markers (can enable/disable at runtime) |
| Distribution plot (not just median) | Shows tail latency, not just average | Histogram mode in Nsight Systems timeline |

---

## Actionable Checklist for Your Next Kernel

Before writing a new DGEMM optimisation:

```
[ ] Run ncu --set full on current version. Save result.
[ ] Check: is SM throughput < 70%? -> compute underutilised, fix occupancy
[ ] Check: is memory throughput > 80%? -> memory-bound, increase tile size
[ ] Check: are there warp stalls? -> look at dependency type
[ ] Check: is L2 hit rate < 50%? -> data reuse is poor, increase blocking
[ ] Make ONE change.
[ ] Re-run ncu. Compare specific metrics.
[ ] Record (kernel, change, metric before, metric after) in a table.
```

This table becomes the backbone of your blog post. It proves systematic
engineering, not random tweaking — which is exactly what a Google or NVIDIA
recruiter is looking for.

---

---

# C++ Optimization Case Study — "papy" Load Tester

> Notes derived from: *"I optimized my C++ app from 100 to 20,000 req/s"*
> Source project: **papy** — a pseudo-random JSON payload load tester
> Mapped to your work: **CUDA kernel engineering + DGEMM benchmarking**

This video is a practical optimization journey that rediscovers the same
principles as David's talk, but from a beginner's perspective. The value is
seeing exactly *which specific mistakes* cause performance to tank, and how
each fix maps to a measurable number.

---

## The Performance Journey (with numbers)

```
Starting point:   100  req/s   (baseline)
+ gzip:           170  req/s   (+70%)    ← reduce data transfer size
+ move init:      360  req/s   (+112%)   ← stop re-creating objects in hot loop
+ global devices: 2400 req/s   (+567%)   ← one device for all calls
+ constexpr/static: 3200 req/s (+33%)   ← bake constants at compile time
+ pass by ref:    8000 req/s   (+150%)   ← eliminate hidden copies
+ thread_local:   7600 req/s   (-5%)     ← race condition fix, stable
+ unlock I/O:    20000 req/s   (+163%)   ← the REAL bottleneck was stdout
```

**Key meta-lesson: the biggest win (I/O unlock, 2.6×) was something that had
nothing to do with the algorithm.** The "obvious" code was the actual bottleneck.
This exact pattern appears in GPU programming constantly.

---

## Optimization 1 — Stop Initializing Expensive Objects in the Hot Loop

### What happened in the video
40% of CPU time was spent inside a randomization function. The bug:
a new `std::random_device` was being constructed *every single call*.
Moving it to class scope (constructed once): 100 → 360 req/s.

### Direct CUDA equivalent: Don't Allocate GPU Memory Inside Kernels

```cuda
// ❌ BAD — allocating inside kernel launch loop
for (int i = 0; i < num_iterations; i++) {
    float* d_tmp;
    cudaMalloc(&d_tmp, size);          // VERY expensive, inside loop
    launch_kernel<<<grid, block>>>(d_tmp);
    cudaFree(d_tmp);
}

// ✓ GOOD — allocate once, reuse across iterations
float* d_tmp;
cudaMalloc(&d_tmp, size);              // once, outside loop
for (int i = 0; i < num_iterations; i++) {
    launch_kernel<<<grid, block>>>(d_tmp);   // reuse same buffer
}
cudaFree(d_tmp);
```

The same applies to CUDA streams, cuBLAS handles, and cuBLAS workspaces.
Create them once, reuse them. `cublasCreate()` inside a benchmark loop is
a classic beginner mistake that ruins throughput measurements.

---

## Optimization 2 — Batch Your Operations

### What happened in the video
Calling a function 70 times to get 70 values was slower than calling it once
for 70 values. The function-call overhead and setup cost dominated.

### Direct CUDA equivalent: Kernel Launch Overhead

```
cudaLaunchKernel has ~5–10 µs overhead on CPU side.

// ❌ BAD — launching one kernel per matrix row
for (int row = 0; row < N; row++) {
    compute_row<<<1, COLS>>>(A, B, C, row);
}
// N kernel launches = N × 10µs = potentially milliseconds of overhead

// ✓ GOOD — one kernel, all rows
compute_all_rows<<<grid, block>>>(A, B, C, N);
// 1 kernel launch = 10µs total
```

For your DGEMM: **never launch a kernel per tile.** The entire matrix multiply
is one kernel launch. Tiles are handled inside the kernel by thread blocks.

---

## Optimization 3 — Pop from Back, Not Front (O(1) vs O(n))

### What happened in the video
Every `vector.erase(begin())` shifts all remaining elements — O(n).
Switching to `vector.back()` + `vector.pop_back()` is O(1).
No functional difference when order doesn't matter.

### Direct CUDA equivalent: Memory Access Patterns in Shared Memory

```cuda
// ❌ BAD — "shift left" pattern in shared memory
// Thread 0 updates shmem[0], everyone else shifts
__shared__ float buf[TILE];
buf[0] = new_val;                  // forces all threads to re-read

// ✓ GOOD — use a circular index (no shifting, no data movement)
__shared__ float buf[TILE];
buf[write_idx % TILE] = new_val;   // O(1), no movement
```

More broadly: **never design a GPU algorithm that requires shifting elements
in an array.** The GPU has no "bulk move" instruction — shifting N elements
launches N threads and wastes memory bandwidth.

---

## Optimization 4 — Reserve Memory Upfront, Avoid Reallocation

### What happened in the video
`std::vector` growth (doubling + realloc + copy) was wasting time.
Calling `reserve(n)` before filling prevented all reallocations.

### Direct CUDA equivalent: Pre-allocate All GPU Buffers

```bash
# ❌ BAD pattern for HPL benchmark runs
# Letting HPL dynamically grow buffers during the run

# ✓ GOOD: calculate N exactly so GPU HBM is ~80% full
# N = floor(sqrt(0.80 × VRAM_bytes / 8))
# For A100 80GB: N = 89,000 (pre-calculated, not adjusted mid-run)
```

In CUDA kernels, if you need a scratch buffer, size it for the worst case
and allocate once before the timing region starts. **Never call `cudaMalloc`
between `cudaEventRecord` calls that you're using for benchmarking.**

---

## Optimization 5 — Eliminate Hidden Copies (Pass by Reference)

### What happened in the video
A vector was being assigned by value: the function returned a new vector,
which was copied into the caller's variable. The caller's `reserve()` was
then overwritten. Fix: pass the output vector by reference. 3200 → 8000 req/s.

### Direct CUDA equivalent: cuBLAS Workspace & Result Buffers

```cuda
// ❌ BAD — result buffer declared inside timing loop
// creates a new device allocation and implicit copy on every iteration
for (int i = 0; i < BENCH_ITERS; i++) {
    float* d_C;
    cudaMalloc(&d_C, M * N * sizeof(float));   // hidden allocation
    cublasSgemm(..., d_C, ...);
    cudaFree(d_C);
}

// ✓ GOOD — pass result buffer by "reference" (pre-allocated pointer)
float* d_C;
cudaMalloc(&d_C, M * N * sizeof(float));
for (int i = 0; i < BENCH_ITERS; i++) {
    cublasSgemm(..., d_C, ...);               // reuse same output buffer
}
```

For your DGEMM benchmarking: allocate `d_A`, `d_B`, `d_C` *once* before the
timing loop. Reset `d_C` to zero between iterations with `cudaMemset`, not
re-allocation.

---

## Optimization 6 — `constexpr` and `static`: Pay at Compile Time, Not Runtime

### What happened in the video
String literals and lookup tables that never changed were being constructed
at runtime, every call. `constexpr` moves them to compile time. `static`
makes them persist (one constructor total). 2400 → 3200 req/s.

### Direct CUDA equivalent: `__constant__` Memory

```cuda
// ❌ BAD — passing a lookup table as a kernel argument (copied every launch)
__global__ void kernel(float* lut, int lut_size, float* data) { ... }

// ✓ GOOD — store in constant memory (cached on-chip, broadcast to all threads)
__constant__ float LUT[LUT_SIZE];    // set once with cudaMemcpyToSymbol
__global__ void kernel(float* data) {
    float v = LUT[threadIdx.x];      // all threads in warp get same value
    ...                              // hardware broadcasts — no bandwidth cost
}
```

For your DGEMM: matrix dimensions (M, N, K), tile sizes, and stride values
should all be `constexpr` template parameters, not runtime arguments. This
lets the compiler unroll loops and eliminate all branch overhead.

---

## Optimization 7 — Race Conditions Only Appear Above a Throughput Threshold

### What happened in the video
A global randomization device was corrupted by concurrent threads — but only
at ~8000 req/s. At 360 req/s, threads were too slow to collide. Fix: make
the device `thread_local` — one per thread, no sharing.

### Direct CUDA equivalent: Shared Memory Bank Conflicts

```cuda
// ❌ BAD — multiple threads in same warp access same bank
// Bank conflict: serialised, 32× slower
__shared__ float shmem[32];
float v = shmem[threadIdx.x * 4];   // threads 0,8,16,24 hit same bank

// ✓ GOOD — pad to avoid conflict
__shared__ float shmem[32 + 1];     // +1 offset eliminates bank collision
float v = shmem[threadIdx.x * 4];
```

The point: **race conditions (bank conflicts, warp divergence) are throughput-
dependent.** The bug doesn't exist at low occupancy. At 80%+ GPU occupancy,
bank conflicts that were "fine" in testing become a real bottleneck.

---

## Optimization 8 — I/O Was the Real Bottleneck (The Great Realization)

### What happened in the video
CPU was only at 60% utilisation even with 16 threads saturated.
The diagnosis: `stdout` writes inside the hot loop were blocking all threads.
**Every `printf` is a system call that blocks the calling thread.**
Fix: rate-limit the metrics output to once per 250ms. Performance doubled.

> *"I can send an IP packet to Europe faster than I can send a pixel to
> my screen."* — John Carmack

### Direct CUDA equivalent: `cudaDeviceSynchronize()` in the Hot Path

```cuda
// ❌ CATASTROPHIC for benchmarking — synchronize inside timing loop
for (int i = 0; i < BENCH_ITERS; i++) {
    launch_kernel<<<grid, block>>>(d_A, d_B, d_C);
    cudaDeviceSynchronize();    // CPU blocks, GPU idle, CPU wakes, repeat
    // This measures: kernel time + sync overhead + OS scheduling jitter
}

// ✓ CORRECT — sync only for timing, not per iteration
cudaEventRecord(start);
for (int i = 0; i < BENCH_ITERS; i++) {
    launch_kernel<<<grid, block>>>(d_A, d_B, d_C);   // queue, don't wait
}
cudaEventRecord(stop);
cudaEventSynchronize(stop);    // ONE sync at the end
cudaEventElapsedTime(&ms, start, stop);
```

**This is the single most common benchmarking mistake in CUDA.**
Always use `cudaEvent` pairs, never `cudaDeviceSynchronize()` between
iterations. Your TFLOPS number will be 2–5× lower if you get this wrong.

---

## Summary Table: papy → CUDA

| papy Bug | Root Cause | CUDA Equivalent |
|---|---|---|
| Re-creating `random_device` each call | Expensive object in hot loop | `cudaMalloc` / `cublasCreate` inside timing loop |
| 70 individual calls instead of 1 batch | Function call overhead | 70 kernel launches instead of 1 |
| `erase(begin())` on vector | O(n) data movement | Shifting elements in shared mem array |
| No `vector.reserve()` | Realloc + copy mid-execution | Not pre-allocating device buffers before benchmark |
| Return-by-value (hidden copy) | Unintended data duplication | Allocating `d_C` inside timing loop |
| Runtime string/table construction | Paying at runtime what's known at compile time | Runtime `M,N,K` args instead of `constexpr` template params |
| Global `random_device` race condition | Concurrent shared mutable state | Shared memory bank conflicts at high occupancy |
| `stdout` writes blocking threads | I/O blocking compute threads | `cudaDeviceSynchronize()` inside benchmark loop |

---

---

# C++ Optimization Case Study � 'papy' Part 2 (Community Follow-Up)

> Notes derived from the follow-up papy video applying community-suggested optimisations.
> Mapped to your work: **CUDA kernel compilation, arithmetic choice, memory layout, benchmarking**

---

## Lesson 1 — Compiler Flags Matter as Much as Your Algorithm (`-O3`)

### What happened in the video
The single most impactful change: compiling with `-O3` (release mode).
Starting point was ~17,000 req/s. After `-O3`: effectively doubled or more.
Enables: function inlining, loop unrolling, auto-vectorisation, dead code
elimination, removal of debug checks. He didn't know about it before — it
was the single most-spammed suggestion in the comments.

### Direct CUDA equivalent: `nvcc` compilation flags

```bash
# WRONG — debug build (-G disables ALL GPU optimisations, can be >10x slower)
nvcc -G -g dgemm.cu -o dgemm_debug

# CORRECT — release build for any numbers you publish
nvcc -O3 -arch=sm_80 --use_fast_math -lineinfo dgemm.cu -o dgemm_release
#          ^A100      ^FMA + fast math  ^keeps line info for profiling
```

`--use_fast_math` specifically enables **Fused Multiply-Add (FMA)**:
`a*b + c` becomes a single instruction, not two. For DGEMM, the inner
loop is nothing but `C += A[k] * B[k]` — FMA halves the instruction count.

```bash
# Verify FMA instructions are actually in your binary
cuobjdump -sass ./dgemm_release | grep DFMA
# If you only see DMUL + DADD, your flags are wrong
```

This is the CUDA equivalent of forgetting `-O3`. Always verify build flags
before recording any benchmark number.

---

## Lesson 2 — Bitwise Ops Are Free; Modulo/Divide Are Not

### What happened in the video
MT19937 uses XOR + shifts (1 cycle each). `minstd_rand` uses large prime
multiplication + modulo (~40 cycles). They ended up comparable on modern
CPUs, but the principle is clear: bitwise operations are the cheapest
arithmetic a processor can execute.

### Direct CUDA equivalent: Avoid `%` in Hot Kernels

Integer division/modulo costs ~40 cycles on a GPU. Bitwise AND costs 1 cycle.
When the divisor is a power of 2, always use the AND form:

```cuda
// SLOW — modulo in inner kernel loop
int local_idx = global_idx % TILE_SIZE;    // ~40 cycles

// FAST — bitwise AND, only valid when TILE_SIZE is a power of 2
int local_idx = global_idx & (TILE_SIZE - 1);   // 1 cycle

// BEST — compiler eliminates it entirely with constexpr template param
template<int TILE_SIZE>
__global__ void dgemm_kernel(...) {
    int local_row = threadIdx.x & (TILE_SIZE - 1);
}
```

**Consequence:** always use powers of 2 for tile sizes (16, 32, 64).
The same applies to HPL's NB block size — powers of 2 eliminate modulo
in BLAS inner loops.

---

## Lesson 3 — Stack vs Heap: Keep Hot Data Off the Heap (SSO and RVO)

### What happened in the video
Two things he was unknowingly benefiting from:

- **SSO (Small String Optimization):** strings up to ~15 chars live directly
  inside the `std::string` object on the stack. No heap allocation, no pointer
  indirection. Already happening for his 4-12 char strings.

- **RVO (Return Value Optimization):** compiler constructs a returned object
  directly in the caller's memory, skipping copy/move entirely.

The lesson: **understand what the compiler does for you, so you don't write
code that accidentally defeats these optimisations.**

### Direct CUDA equivalent: Registers vs Shared Memory for Accumulators

The same hierarchy applies inside a CUDA kernel:

```cuda
// WRONG — forces accumulator through shared memory (~5-10 cycle latency)
__global__ void bad_dgemm() {
    __shared__ double accum;
    accum = 0.0;
    for (int k = 0; k < K; k++)
        accum += A[k] * B[k];   // every iteration hits shared mem
}

// CORRECT — accumulator stays in a register (1 cycle latency)
__global__ void good_dgemm() {
    double accum = 0.0;          // compiler allocates to register
    for (int k = 0; k < K; k++)
        accum += A[k] * B[k];   // inner loop never touches memory
    C[idx] = accum;              // one global write at the very end
}
```

```bash
# Check register count — target >= 32 for DGEMM
nvcc --ptxas-options=-v dgemm.cu
# "registers=N" — too few means register spilling to local (= global) memory
```

This is exactly what "register-blocked DGEMM" achieves: the C sub-tile lives
in registers across the entire K-loop, never written back until the end.

---

## Lesson 4 — Know When You've Shifted the Bottleneck

### What happened in the video
His explicit goal: flame graph should show network dominating, not compute.
Once that happened, the CPU side was sufficiently optimised — bottleneck had
moved to the right place. Also: unnecessary JSON fields add wire cost at
scale. For pure throughput, Protobuf (binary, no parsing) beats JSON.

### Direct CUDA equivalent: Shifting the Roofline Bottleneck

Each DGEMM optimisation should move the roofline dot upward and rightward:

```
Naive kernel       -> memory-bound  (below memory BW ceiling)
After tiling       -> compute-bound (approaching FP64 ceiling)
After reg-blocking -> instruction-throughput-bound (FMA pipeline)
                      -> you're done (or apply vectorisation)
```

Nsight confirms you've shifted the bottleneck when:
- Memory throughput drops (fewer HBM accesses per FLOP)
- SM Active Cycles increases (more time doing real compute)
- The roofline dot moves toward the compute ceiling

If the dot doesn't move after a change, the change didn't address the actual
bottleneck — equivalent to changing JSON field names when the real problem
is parsing overhead.

**The Protobuf vs JSON analogy in HPL:** MPI panel layout must be contiguous
(column-major, Fortran order) to allow direct DMA without `MPI_Pack` overhead.
This is why HPL.dat sets `L1 in (0=transposed) form` — transposed = column-major
= contiguous in memory = maximum MPI transfer efficiency.

---

## Combined Master Summary (All Three Videos)

| C++ Concept | CUDA / HPC Equivalent |
|---|---|
| `-O3` release flag | `nvcc -O3 --use_fast_math -arch=sm_80`; verify `DFMA` with `cuobjdump` |
| Bitwise ops over modulo | `& (TILE-1)` not `% TILE`; tile sizes must be powers of 2 |
| SSO — small data on stack | Accumulator in register, not `__shared__`, in inner K-loop |
| RVO — compiler builds in-place | Let compiler register-allocate; don't force through shared mem |
| Shift bottleneck to network | Nsight roofline: move from memory-bound to compute-bound per step |
| JSON vs Protobuf | Row-major vs column-major MPI panels; `L1 transposed` in HPL.dat |
| `random_device` in hot loop | `cudaMalloc` / `cublasCreate` inside timing loop |
| 70 calls instead of 1 batch | 70 kernel launches instead of 1 (each launch = ~10us overhead) |
| `erase(begin())` O(n) shift | No array-shifting algorithms on GPU; use circular indexing |
| No `vector.reserve()` | Pre-allocate all device buffers before `cudaEventRecord` |
| Return-by-value hidden copy | Allocating `d_C` inside benchmark timing loop |
| Global state race condition | Shared memory bank conflicts, visible only at high occupancy |
| `stdout` blocking threads | `cudaDeviceSynchronize()` inside benchmark iteration loop |
