# Branchless Programming, Cache Locality & Compile-Time Computation

> Source videos:
> - Video 1: *Sorting, Cache Locality, and Branch Prediction* (C++ performance benchmarks)
> - Video 2: *`constexpr` — a 1000x Speedup* (Dave's Garage)
> Mapped to your work: **CUDA kernels, Vitis HLS, and DGEMM optimisation**

---

## Part 1 — Cache Locality

### What a cache is (and why misses are catastrophic)

When the CPU reads memory, it doesn't fetch just the byte you asked for —
it fetches an entire **cache line** (64 bytes on x86). Everything nearby comes along for free.

```
Memory hierarchy (typical desktop/server):
  Registers     ~1 cycle      handful of values
  L1 cache      ~4 cycles     ~32–64 KB per core
  L2 cache      ~12 cycles    ~256 KB – 1 MB per core
  L3 cache      ~40 cycles    8–32 MB shared
  RAM (DDR5)    ~100+ cycles  16–128 GB
  SSD           ~100,000 cy   TBs
```

**Cache hit** = data already in cache → fast.
**Cache miss** = fetch from the next level down → expensive.

The key insight: **spatial locality**. Access memory sequentially and the
prefetcher loads the next cache line before you need it — effectively free.

---

### Benchmark: sequential vs random vector access

From Video 1 — summing elements of a vector:

```cpp
// Sequential — hardware prefetcher predicts perfectly
for (int i = 0; i < n; i++) total += vec[i];

// Random — prefetcher has no clue what's next
for (int i : random_indices) total += vec[i];
```

**Result:** Performance was identical until the vector exceeded L3 cache size (~8 MB).
Above that, random access collapsed — every access became a RAM fetch.

> **Key lesson:** For small data (fits in cache), access pattern doesn't matter.
> For large data, random access is 10–100× slower than sequential.

---

### Matrix multiplication: the loop order matters 2×+

All six orderings of `i, j, k` produce the same result — but not the same performance.

```cpp
// ❌ SLOW — naive ijk order (most people's first instinct)
for (int i = 0; i < N; i++)
    for (int j = 0; j < N; j++)
        for (int k = 0; k < N; k++)
            C[i][j] += A[i][k] * B[k][j];
//                               ^^^^^^^^
// B[k][j]: as k increases, jumps N elements in memory — cache miss every iteration
```

```cpp
// ✅ FAST — ikj order
for (int i = 0; i < N; i++)
    for (int k = 0; k < N; k++)
        for (int j = 0; j < N; j++)
            C[i][j] += A[i][k] * B[k][j];
//           ^^^^^^              ^^^^^^^^
// C[i][j]: sequential ✓
// A[i][k]: constant in inner loop → stays in register ✓
// B[k][j]: sequential ✓
```

**Speedup from loop reorder alone: >2× on 64×64 matrices.**

### Why ikj wins: memory access analysis

```
Row-major layout (C/C++):
A = [A00 A01 A02 | A10 A11 A12 | A20 A21 A22]
     ←── row 0 ──→ ←── row 1 ──→ ←── row 2 ──→

ijk inner loop (k varies, j fixed):
  A[i][k]  → sequential ✓ (moves along row i)
  B[k][j]  → stride N   ✗ (jumps to next row each step)

ikj inner loop (j varies, k fixed):
  A[i][k]  → constant   ✓ (held in a register)
  B[k][j]  → sequential ✓ (moves along row k)
  C[i][j]  → sequential ✓ (moves along row i)
```

### All 6 orderings ranked (64×64, row-major C++)

| Order | Inner access pattern | Cache-friendly? | Relative speed |
|---|---|---|---|
| **ikj** | B sequential, C sequential | ✅✅ | **fastest** |
| **kij** | B sequential, C sequential | ✅✅ | fast |
| ijk | B strided | ❌ | slow |
| jik | A strided | ❌ | slow |
| jki | A strided, C strided | ❌❌ | slower |
| kji | A strided, C strided | ❌❌ | **slowest** |

### Connection to your HLS/CUDA DGEMM

This is **exactly** why:
- HLS kernels load tiles into local BRAM before computing (avoids strided DDR access)
- CUDA kernels transpose B into shared memory before the K-loop
- `ARRAY_PARTITION variable=localB complete dim=1` in HLS gives sequential access along rows

> **Rule:** In any DGEMM, the B matrix access pattern is the enemy.
> Either transpose B, tile it, or restructure the loop to access it sequentially.

---

## Part 2 — Branch Prediction

### What branch prediction is

Modern CPUs don't wait to evaluate an `if` condition before fetching the next instruction.
They **guess** which way the branch goes, and execute speculatively ahead.

- **Correct guess** → free (work was already done)
- **Wrong guess** → pipeline flush — throw away ~15–20 cycles of speculative work

```
Pipeline stages (simplified):
  Fetch → Decode → Execute → Writeback

At a branch point:
  CPU guesses "true" → speculatively fetches instructions for the "true" path
  If wrong: must flush the pipeline and restart with the "false" path
  Cost: 15–20 wasted cycles per misprediction
```

### Benchmark: how predictability affects performance

From Video 1 — computing a product, multiplying by 2× or 1.5× based on a threshold:

```cpp
double get_product(const std::vector<double>& vec, double threshold) {
    double product = 1.0;
    for (double x : vec) {
        if (x > threshold)
            product *= 2.0 * x;   // taken   branch
        else
            product *= 1.5 * x;   // not-taken branch
    }
    return product;
}
```

| % above threshold | Branch predictable? | Relative performance |
|---|---|---|
| 0% or 100% | ✅ Always same path | **fastest** (~2×) |
| 10% or 90% | ✅ Mostly predictable | fast |
| 50% | ❌ Coin flip | **slowest** |

The branch predictor can learn patterns like:
- "always true" → predict true every time
- "alternating T/F/T/F" → predict alternating
- "50/50 random" → cannot predict → maximum penalty

### Branchless programming: remove the if entirely

Instead of branching, compute **both paths** and select the result arithmetically:

```cpp
// ❌ Branchy — creates mispredictions at 50% threshold
if (x > threshold)
    product *= 2.0 * x;
else
    product *= 1.5 * x;

// ✅ Branchless — arithmetic select, no branch
double multiplier = (x > threshold) ? 2.0 : 1.5;  // compiler → CMOV, no branch
product *= multiplier * x;

// Even more explicit — manual arithmetic version
int cond = (x > threshold);                        // 1 or 0
double multiplier = 1.5 + 0.5 * cond;              // 1.5 or 2.0
product *= multiplier * x;
```

### How CPUs implement branchless: CMOV

`CMOV` (Conditional Move) is a single x86 instruction:
```asm
; Evaluate condition, then:
; result = (condition) ? a : b
; No branch, no pipeline flush
CMOVG rax, rbx    ; rax = rbx if (rflags.greater), else unchanged
```

The compiler often generates `CMOV` automatically from ternary operators.
Use `godbolt.org` to verify your code compiles to `CMOV`, not `JNE`/`JE`.

### Branchless techniques

```cpp
// 1. Ternary → often compiles to CMOV
int result = (x > 0) ? a : b;

// 2. Arithmetic select — avoids any conditional
int mask = -(x > 0);           // 0xFFFFFFFF if true, 0x00000000 if false
int result = (mask & a) | (~mask & b);

// 3. Boolean arithmetic — for simple 0/1 cases
int clamped = std::min(std::max(x, 0), 255);  // branchless via SIMD min/max

// 4. Bit tricks — check sign without branch
int abs_x = (x ^ (x >> 31)) - (x >> 31);   // branchless abs()
```

> **Warning from Video 1:** Compilers are very good.
> If you write a simple `if/else`, the compiler may already emit `CMOV` for you.
> Check the assembly with `godbolt.org` before manually rewriting — you may do nothing.

---

## Part 3 — Branchless in CUDA: Warp Divergence

### The GPU version of branch misprediction

In CUDA, 32 threads execute as a **warp** in lockstep.
If threads in the same warp take different `if/else` paths, the GPU must:
1. Execute the "true" path with half the threads active (the rest are masked off)
2. Then execute the "false" path with the other half

This is **warp divergence** — effectively halving throughput.

```cuda
// ❌ DIVERGENT — threads in same warp go different ways
if (threadIdx.x < 16)
    result = a * 2.0;      // first 16 threads
else
    result = b * 1.5;      // last 16 threads
// GPU executes BOTH paths sequentially → 50% throughput

// ✅ NON-DIVERGENT — all threads take same path
// (condition uniform across warp, or arithmetic select used)
double multiplier = 1.5 + 0.5 * (double)(threadIdx.x < 16);
result = base * multiplier;   // one path, full throughput
```

### How CUDA handles branchless: predication

The `nvcc` compiler uses **predicated instructions** — the GPU equivalent of CMOV:
- All threads execute the instruction
- A predicate register controls whether the result is committed
- No pipeline flush, no serialisation

```bash
# Verify your kernel uses predication, not branches
cuobjdump --dump-sass ./kernel | grep -E "SEL|ISETP|@P"
# SEL = select (CMOV equivalent)
# @P  = predicated instruction
# If you see BRA (branch), you have divergence
```

### Practical CUDA branchless patterns for DGEMM

```cuda
// Boundary guard — the most common source of divergence
// ❌ Branchy (every thread tests if it's in bounds)
if (row < M && col < N)
    C[row * N + col] = acc;

// ✅ Pad matrices to tile-size multiples before launch
// → ALL threads are always in bounds → zero divergence
// Padding with zeros: C = (A_padded × B_padded) gives correct result
// for the unpadded region

// ✅ Or: branchless masked write
bool in_bounds = (row < M) & (col < N);   // & not && — avoids short circuit
C[row * N + col] = in_bounds ? acc : C[row * N + col]; // compiler → SEL
```

---

## Part 4 — Branchless in HLS: Pipeline Stalls from Conditionals

In Vitis HLS, an `if` inside a pipelined loop can increase II if:
- Both branches write to the same array (write-after-write hazard)
- The condition itself has a long combinational path

```cpp
// ❌ Conditional write can increase II
for (int k = 0; k < N; k++) {
    #pragma HLS PIPELINE II=1
    if (A[k] > 0)
        acc += A[k] * B[k];   // conditional accumulate — HLS may see II=2
}

// ✅ Predicated accumulate — always computes, selects 0 or result
for (int k = 0; k < N; k++) {
    #pragma HLS PIPELINE II=1
    double val = A[k] * B[k];
    double mask = (A[k] > 0) ? 1.0 : 0.0;  // branchless select
    acc += val * mask;                        // always accumulates, no if
}

// ✅ Even better — use ap_uint<1> as mask (1 LUT, no DSP)
for (int k = 0; k < N; k++) {
    #pragma HLS PIPELINE II=1
    ap_uint<1> valid = (A[k] > threshold) ? 1 : 0;
    acc += valid * A[k] * B[k];
}
```

---

## Part 5 — `constexpr`: Move Computation to Compile Time

### The key idea (from Video 2)

`constexpr` tells the compiler: *"you can evaluate this entire function at compile time."*

The cost moves from **runtime** to **compile time** — at runtime, the result is just a hardcoded constant.

```cpp
// Runtime version — called every time the program runs
int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

// constexpr version — compiler evaluates at compile time
constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);
}

constexpr int result = fibonacci(35);  // compiler does all the work
// At runtime: result is just the integer 9,227,465 — no function call
```

**Measured speedup from Video 2: >1000× for fibonacci(35)**
(From 40 ms → 40 µs — and even that was mostly clock overhead)

### Verify at the assembly level

```bash
g++ -O2 -S -o output.asm program.cpp
grep -A2 "result" output.asm
# Should see: movl $9227465, result(%rip)
# A single MOV — the entire recursive tree collapsed to one constant
```

### The constexpr rules

```cpp
// constexpr requirements:
// 1. Arguments must be compile-time constants (literal types: int, char, etc.)
// 2. Must have a return statement (no void)
// 3. No dynamic memory allocation (no new/delete)
// 4. No runtime-only calls (e.g., std::sqrt is NOT constexpr, but you can write your own)
// 5. Recursion is fine if depth is bounded and known at compile time

// constexpr sqrt (since std::sqrt is not constexpr)
constexpr double sqrt_ce(double x, double curr, double prev) {
    return (curr == prev) ? curr
           : sqrt_ce(x, 0.5 * (curr + x / curr), curr);
}
constexpr double sqrt_ce(double x) {
    return sqrt_ce(x, x, 0.0);
}
```

### Constexpr in HLS: template parameters

In Vitis HLS, `constexpr` + templates is the mechanism for fully unrolled loops:

```cpp
// Runtime tile size — HLS cannot fully unroll
void dgemm(int TILE, double A[], double B[], double C[]) {
    for (int i = 0; i < TILE; i++)  // TILE unknown at elaboration → partial unroll only
        ...
}

// Compile-time tile size — HLS fully unrolls all loops at elaboration time
template<int TILE>
void dgemm(double A[], double B[], double C[]) {
    for (int i = 0; i < TILE; i++)  // TILE known → fully unrolled → parallel hardware
        #pragma HLS UNROLL
        ...
}

// Call site — instantiate for specific tile size
dgemm<16>(A, B, C);   // compiler generates hardware for exactly TILE=16
```

**Why this matters for FPGA:** HLS synthesises hardware from loops.
If the loop bounds are compile-time constants, HLS can:
- Fully unroll the loop into parallel hardware units
- Eliminate all multiplexers and control logic for the loop counter
- Prove there are no inter-iteration dependencies → achieve II=1

### Sieve of Eratosthenes as constexpr (from Video 2)

The video showed a prime sieve up to 10 million implemented as a `constexpr` class.
At compile time, the compiler runs the entire sieve. At runtime: instant results.

```cpp
// Conceptual structure (simplified)
template<int N>
struct PrimeSieve {
    bool sieve[N];

    constexpr PrimeSieve() : sieve{} {
        // Fill sieve at compile time
        for (int i = 2; i < N; i++) sieve[i] = true;
        for (int i = 2; i * i < N; i++)
            if (sieve[i])
                for (int j = i*i; j < N; j += i)
                    sieve[j] = false;
    }

    constexpr bool is_prime(int n) const { return sieve[n]; }
};

constexpr PrimeSieve<1000000> primes;  // compiler runs the sieve
// At runtime: primes.is_prime(997) is just a memory lookup — 0 compute cost
```

---

## Summary: The Three Levers

| Lever | What it fixes | Tool | FPGA equivalent |
|---|---|---|---|
| **Cache locality** | Data not in cache when needed | Tiling, sequential access, ikj loop order | BRAM tiles, `memcpy` burst loads |
| **Branch prediction** | Pipeline flushes from mispredicted branches | Ternary, CMOV, arithmetic select | Remove `if` inside pipelined loops |
| **Compile-time compute** | Runtime work that's known statically | `constexpr`, templates | `template<int TILE>`, `#pragma HLS UNROLL` |

### Combined principle

> Big-O complexity does not determine real-world performance for fixed or small sizes.
> **Cache locality, branch predictability, and compile-time resolution** often matter more.
> An O(n²) algorithm with perfect cache behaviour can beat an O(n log n) algorithm with poor cache behaviour — as Video 1 showed with insertion sort beating quicksort on 16 elements.

---

## Actionable Checklist

```
Cache locality:
[ ] Inner loop index moves sequentially in memory
[ ] For matrix multiply: use ikj order, not ijk
[ ] Tile large matrices so working set fits in L2/L3 cache

Branch prediction:
[ ] Identify branches inside hot loops (profile first with perf/ncu)
[ ] Replace if/else with ternary or arithmetic select where possible
[ ] Verify with godbolt.org that compiler emits CMOV, not JNE
[ ] In CUDA: pad matrix dims to tile-size multiples to eliminate boundary guards
[ ] In HLS: scalar accumulator + predicated multiply to avoid II > 1

Compile-time compute:
[ ] Mark pure functions with constexpr if arguments are compile-time known
[ ] Use template<int TILE> in HLS instead of runtime tile size argument
[ ] Check assembly: constexpr result should appear as a single MOV instruction
[ ] For HLS: verify that HLS synthesis report shows loop fully unrolled (not partially)
```

---

## References

| Source | Key Concept |
|---|---|
| Video 1 — *Sorting, Cache, Branch Prediction* | ikj loop order, cache size measurement, branch predictor benchmark |
| Video 2 — *Dave's Garage: constexpr* | Fibonacci 1000× speedup, sieve as constexpr, compile-time vs runtime |
| [godbolt.org](https://godbolt.org) | Inspect assembly output — check for CMOV vs JNE, verify constexpr collapses |
| `cuobjdump --dump-sass` | Verify CUDA uses predication (SEL) not branching (BRA) |
| `low_latency_principles_for_hpc.md` | Branchless in HPL context, warp divergence, `__constant__` memory |
| `fpga_dgemm_kernel_optimisation.md` | HLS pragma pipeline, UNROLL, accumulator in register |
