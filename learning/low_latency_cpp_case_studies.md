# Low-Latency C++ — Case Studies & Profiling Workflow

> Source: David's CppCon talk — *"Low Latency Trading Systems in C++"*
> Speaker: 10-year market maker engineer, formerly defense systems
> See also: `low_latency_principles_for_hpc.md` for the 8 principles mapped to CUDA/GPU

---

## Context: Why Low Latency Matters in Trading

Two distinct reasons (both apply to HPC too):

1. **React fast to uncertain events** — a news event hits, stale prices cause bad trades
2. **Accuracy requires freshness** — even your models are wrong if fed stale information

> "Market making is a loser's game — you need to be consistently good at everything.
> There is no silver bullet."

The same is true of HPC kernel engineering.

---

## Case Study 1: The Order Book Data Structure Journey

### What an order book is

```
Bids (want to buy):          Asks (want to sell):
  $92 — 50 shares  ← best   $95 — 30 shares  ← best
  $90 — 100 shares           $97 — 80 shares
  $85 — 200 shares           $99 — 50 shares

API operations:
  add(price, volume, id)
  modify(id, new_volume)
  delete(id)
```

**Constraint:** ~1,000 price levels per side for a typical stock.
**Key observation:** updates are exponentially distributed toward the top of the book.

---

### Attempt 1: `std::map` — The Obvious Choice

```cpp
std::map<uint64_t, Level, std::greater<>> bids;  // ordered, high→low
std::map<uint64_t, Level>                 asks;  // ordered, low→high
```

**Why it seems perfect:** O(log n) insert/delete, iterators stay valid, hashmap can store iterators.

**Why it fails in production:**

`std::map` is a **node container** — each element is a separate heap allocation.
In production, heap is fragmented with other allocations → cache locality is terrible.

```
std::map memory layout:
  [node*] → [heap: key+val+left*+right*+parent*]   // 5 pointers per node
             [heap: ...]                             // completely random addresses
             [heap: ...]
  Every tree traversal = cache miss at each node
```

**Benchmark:** With artificial heap fragmentation (simulating production), the "true"
latency distribution has a long, fat tail — exactly what you do not want.

> **Principle 1: Avoid node containers.**
> `std::map`, `std::set`, `std::list`, `std::unordered_map` — all have poor cache locality.
> Sean Parent at Adobe: 90% of their Photoshop code uses `std::vector` for performance.
> Use flat hash maps (arrays/vectors) instead.

---

### Attempt 2: `std::vector` + `std::lower_bound` — Better Cache, New Problem

```cpp
std::vector<Level> bids;  // sorted descending (best bid at index 0)
std::vector<Level> asks;  // sorted ascending  (best ask at index 0)

// Insert
auto it = std::lower_bound(bids.begin(), bids.end(), price, cmp);
if (it->price == price) it->volume += delta;  // update existing
else bids.insert(it, Level{price, volume});   // insert new level → O(n) shift
```

**Cache locality:** ✅ Much better — all data contiguous.
**Complexity:** ❌ Insert is O(n) (vector shift).

**But the real problem — look at the data:**

```
Distribution of updated price levels (Nvidia stock, 1 week):
  Level 0 (best bid/ask):   ████████████████████  (most updates)
  Level 1:                  ████████
  Level 2:                  ████
  Level 3:                  ██
  Level 10+:                █
```

Updates are exponentially concentrated at the top. With best price at index 0,
**every update shifts the entire vector** — O(n) moves at the hottest level.

**Result:** Nice average, but a massive fat tail in the latency distribution.

---

### Attempt 3: Reversed Vector — One Line Fix

```cpp
// Best price at the END of the vector, not the beginning
// Updates at the top of the book → pop_back, not erase(begin())
std::vector<Level> bids;  // sorted ascending (best bid at BACK)
```

When the best price changes, we append/remove from the back — O(1), no shifting.

**Result:** Fat tail completely eliminated. Clean, narrow latency distribution.

> **Principle 2: Understand your problem domain.**
> The fix required understanding the *data distribution*, not just the data structure.
> A problem well-stated is half solved.

> **Principle 3: Leverage domain-specific properties.**
> The exponential update distribution at the top of the book is the key property.
> Without knowing it, you can't find this fix.

---

### Attempt 4: Profiling with `perf` — Finding the Branch Problem

After the vector fix, David used `perf stat` to get the TMA (Top-Down Microarchitecture Analysis):

```
Retiring:         ~60%   (useful work)
Bad Speculation:  ~25%   ← VERY HIGH
Front-End Bound:   ~8%
Back-End Bound:    ~7%
```

25% bad speculation is a red flag. `perf record` found the culprit:

```asm
; std::lower_bound assembly (binary search)
30% of CPU time on these two lines:
  cmp  rax, rdx
  jg   .L_branch_true    ← conditional jump #1
  ...
  cmp  rax, rdx
  jle  .L_branch_false   ← conditional jump #2
```

The binary search branches are unpredictable — market data is random,
so the branch predictor can't do better than ~50%.

**Hardware counter validation:**

```
                Before branchless  After branchless
Branch misses:     ~8M/s              ~4M/s   (halved ✓)
IPC:               1.4                1.6     (improved ✓)
Instructions:      lower              higher  (no early exit penalty)
```

---

### Attempt 5: Branchless Binary Search

```cpp
// Standard lower_bound — has early exit, hard to predict
auto it = std::lower_bound(bids.begin(), bids.end(), price, cmp);

// Branchless binary search — no early exit, always touches all log(n) elements
// Compiler generates CMOV instead of JNE
template<typename It, typename T, typename Cmp>
It branchless_lower_bound(It first, It last, const T& val, Cmp cmp) {
    auto len = std::distance(first, last);
    while (len > 1) {
        auto half = len / 2;
        // No if/else — ternary → CMOV
        first = cmp(*std::next(first, half - 1), val) ? std::next(first, half) : first;
        len -= half;
    }
    return first;
}
```

**Key tradeoff:**
- ✅ Branch misses halved → IPC improved
- ⚠️ No early exit → touches more memory than standard binary search
- Result: two peaks in distribution (warm cache vs cold cache accesses)

---

### Final Winner: Linear Search (The Surprising Result)

David benchmarked ~30 different implementations. The fastest was:

```cpp
// Just iterate from the back (best price) forward
Level* find_level(std::vector<Level>& book, uint64_t price) {
    for (auto it = book.rbegin(); it != book.rend(); ++it)
        if (it->price == price) return &(*it);
    return nullptr;
}
```

**Why this wins:**
- ✅ Perfect cache locality — sequential memory access, prefetcher predicts every access
- ✅ No branches in the search path (comparison is simple, predictable)
- ✅ For ~1,000 levels, the entire book fits in L2/L3 cache
- ✅ Narrowest latency distribution of all 30 implementations

> **Principle 4: Simplicity and performance are not opposites.**
> When you've done your job well as an engineer, the solution is both fast AND simple.
> Linear search over a sorted vector beat every clever data structure tried.

> **Principle 5: Mechanical sympathy.**
> Linear search is in harmony with the hardware — sequential access, predictable
> branches, perfect prefetching. The algorithm matches how the CPU actually works.

---

## Case Study 2: Profiling Workflow for Event-Driven Systems

### The problem with sampling profilers on hot paths

```cpp
// Typical event-driven loop
while (true) {
    auto packet = poll_network_card();    // 99.9% of time spent here (idle)
    if (is_interesting(packet)) {         // ← your hot function
        send_order(compute_price(packet));
    }
}
```

- `perf stat`: gives aggregate counters — misses the hot function entirely
- `perf record` (sampling): samples 1,000×/sec — almost never lands in the hot function

### Solution: Intrusive profiling with TSC

```cpp
struct ScopedTimer {
    uint64_t start;
    const char* name;

    ScopedTimer(const char* n) : name(n), start(__rdtsc()) {}

    ~ScopedTimer() {
        uint64_t elapsed = __rdtsc() - start;
        write_to_queue(name, elapsed);  // lock-free queue, never blocks
    }
};

void is_interesting(const Packet& p) {
    ScopedTimer t("is_interesting");    // TSC at entry and exit
    // ... your logic
}
```

**Problem:** Adding `ScopedTimer` everywhere pollutes the code, requires recompilation,
and the TSC reads themselves consume meaningful cycles at high frequency.

---

### Better Solution: Clang X-Ray

```bash
# Compile with instrumentation inserted at every function entry/exit
clang++ -O3 -fxray-instrument -fxray-instruction-threshold=1 \
        trading_system.cpp -o trading_system

# Ship this binary to production — overhead is near-zero (NOP sleds)
```

**How it works:**
1. Compiler inserts NOP sleds at every function entry/exit
2. In normal operation: NOPs execute in ~0 cycles — no overhead
3. When you want to profile: patch the NOP sleds at runtime with logging calls
4. No recompilation needed — same binary, toggle on/off dynamically

```bash
# Enable tracing at runtime (no recompile)
XRAY_OPTIONS="patch_premain=true xray_mode=xray-fdr" ./trading_system

# Analyze the trace
llvm-xray stack  xray-log.*.xray   # call stack analysis
llvm-xray graph  xray-log.*.xray   # call graph (DOT format)
```

**Selective instrumentation:**

```cpp
[[clang::xray_always_instrument]]  void critical_path() { ... }
[[clang::xray_never_instrument]]   void cold_path()     { ... }
```

| Method | Overhead (off) | Overhead (on) | Recompile needed? |
|---|---|---|---|
| Intrusive TSC | ~2–5 ns per call | ~2–5 ns per call | Yes |
| `perf record` | 0 | ~1–5% | No |
| Clang X-Ray | ~0 (NOPs) | ~10–30 ns per call | **No** |

> Clang X-Ray gives you The Best of Both Worlds: near-zero production overhead,
> full function-level tracing when needed, without recompilation.

---

## Case Study 3: Shared Memory Queue Design

### Why shared memory, not sockets

```
Multiple processes on one server:
  [Market Data Process] → shared memory → [Strategy 1]
                                        → [Strategy 2]
                                        → [Strategy 3...50]

Why not sockets?
  - Sockets go through kernel → context switches → jitter
  - Shared memory: as fast as a cache line read (~5 ns)
  - No kernel involvement after initial mmap()
```

### The queue design

```
Queue layout in shared memory:
  [Header: write_counter | read_counter | magic | version]
  [Ring buffer: data data data data ...]

Two atomic uint64 counters (monotonically increasing byte offsets):
  write_counter: advanced by producer BEFORE copying data
  read_counter:  advanced by producer AFTER copying data

Consumer logic:
  if (local_counter < read_counter)  → data available
  read size prefix → check write_counter for overflow guard → memcpy → advance local
```

### The 3 key optimisations

**Optimisation 1 — Reserve in bulk, touch atomic rarely**

```cpp
// ❌ SLOW — touch write_counter on EVERY message
write_counter += sizeof(size) + msg_size;  // atomic store per message

// ✅ FAST — reserve 100 KB at once, advance counter once per 1,000 messages
// Readers have slightly less visibility (100 KB lag) but 1000× fewer atomic ops
uint64_t reserved = write_counter.fetch_add(100 * 1024);
// Fill the 100 KB window with messages, then advance read_counter
```

Why this matters: consumers read `write_counter` on every iteration.
Fewer writes to that cache line = less cache coherency traffic across all CPUs.

**Optimisation 2 — Don't over-align**

```cpp
// ❌ COMMON MISTAKE — align everything to cache line (64 bytes)
alignas(64) Message msg;

// ✅ CORRECT for this queue — align to 8 bytes only
// Over-aligning reduces data density → more cache lines needed → worse locality
alignas(8) Message msg;
```

Cache-line alignment is correct for the header atomics (to prevent false sharing).
It is **wrong** for the message data (reduces locality).

**Optimisation 3 — Cache the read counter locally**

```cpp
// ❌ Reads write_counter even when we know data is available
uint64_t available = write_counter - local_counter;

// ✅ Only read write_counter when we've consumed all known data
class Consumer {
    uint64_t local_counter = 0;
    uint64_t cached_write  = 0;

    bool read(void* buf, size_t len) {
        if (local_counter + len > cached_write) {
            cached_write = write_counter.load();  // only re-read when needed
            if (local_counter + len > cached_write) return false;
        }
        memcpy(buf, ring + (local_counter % capacity), len);
        local_counter += len;
        return true;
    }
};
```

**Result:** Simple (~150 lines) queue that matches or outperforms LMAX Disruptor
and IPC libraries in the 1–8 consumer range.

> **Principle 7: Right tool for right task.**
> Don't use 100,000-line frameworks when 150 lines solves your specific problem correctly.

---

## Case Study 4: "You Are Not Alone" — L3 Cache Contention

### The experiment

Random-walk memory access benchmark, measuring throughput vs working set size:

```
Result with 1 worker (single process):
  Working set < L1 (32 KB):   high throughput
  Working set < L2 (256 KB):  medium throughput
  Working set < L3 (8 MB):    lower throughput
  Working set > L3:           very low (RAM latency)

Result with 6 workers on 6 different CPUs, same server:
  Same as 1 worker... EXCEPT in the L3 region
  → Scaling factor drops to ~1 (6 workers get same total throughput as 1)
  → Each worker fights the others for L3 bandwidth
```

### What this means

For most trading systems (and HPC workloads):
- L1: too small for most strategies
- L2: fits some hot data paths
- **L3: where most of your code actually runs**

When 6 processes share an L3 cache, they each get 1/6 of it effectively.

> **Principle 8: "You are not alone."**
> Your code's performance depends on everything else running on the same server.
> Optimising your process in isolation is necessary but not sufficient.
> Think about the entire server — which processes share L3, which NUMA domains are in use.

**Practical implications:**

```bash
# NUMA-aware process placement
numactl --cpunodebind=0 --membind=0 ./strategy_1  # pin to NUMA node 0
numactl --cpunodebind=1 --membind=1 ./strategy_2  # pin to NUMA node 1

# CPU isolation — reserve cores for latency-critical processes
# In /etc/default/grub:
# GRUB_CMDLINE_LINUX="isolcpus=2,3,4,5 nohz_full=2,3,4,5"
# → kernel scheduler won't use these cores → zero OS jitter on hot processes

# Verify L3 topology
lstopo             # visual NUMA/cache topology
numactl --hardware # NUMA node info
```

---

## Advanced C++ Details from the Talk

### `[[likely]]` / `[[unlikely]]` — Instruction Cache Hints

```cpp
// Without hint — compiler may place the add instruction far from the hot path
void add_order(OrderBook& book, Order& order) {
    if (level_exists(order.price))
        book.update_volume(order);  // COMMON case — should be hot
    else
        book.insert_level(order);  // RARE case
}

// With hint — compiler places rare branch code far away in binary
void add_order(OrderBook& book, Order& order) {
    if (level_exists(order.price)) [[likely]]
        book.update_volume(order);  // stays in hot code section
    else [[unlikely]]
        book.insert_level(order);   // moved to cold section of binary
}
```

Effect: the hot instructions are packed together → better instruction cache utilisation.
Measured improvement: modest but real, especially at scale.

---

### Lambda vs `std::function` — Never Use `std::function` on a Hot Path

```cpp
// ✅ Lambda (template parameter) — compiler knows the type
// Full inlining, zero overhead
template<typename Cmp>
Level* find_level(std::vector<Level>& book, uint64_t price, Cmp cmp) {
    // cmp is inlined completely — compiler sees its body
    for (auto& lvl : book)
        if (cmp(lvl.price, price)) return &lvl;
    return nullptr;
}

// ❌ std::function — type erasure kills performance
Level* find_level(std::vector<Level>& book, uint64_t price,
                  std::function<bool(uint64_t, uint64_t)> cmp) {
    // cmp is an indirect call — cannot be inlined
    // Virtual dispatch + potential heap allocation for captured state
    for (auto& lvl : book)
        if (cmp(lvl.price, price)) return &lvl;
    return nullptr;
}
```

**Why `std::function` is slow:**
1. Type erasure → indirect call (like a virtual function call) → prevents inlining
2. Indirect calls → branch predictor must predict jump target → more misses
3. If captured state > SBO (Small Buffer Optimisation, ~16 bytes) → heap allocation

**Rules:**
- Hot path: use `template<typename Func>` — always inlined
- Cold path / storage: `std::function` is fine
- C++26: `std::function_ref` for non-owning, lightweight callbacks

---

## The 8 Principles — Quick Reference

| # | Principle | Order Book Example | HPC/CUDA Equivalent |
|---|---|---|---|
| 1 | No node containers | `std::map` → `std::vector` | No dynamic alloc in kernel loop |
| 2 | Understand your domain | Reversed vector (updates at top of book) | Know your arithmetic intensity before coding |
| 3 | Leverage domain properties | Exponential update distribution | DGEMM is compute-bound above tile threshold |
| 4 | Simple + fast = done right | Linear search over 30 implementations | Naive tiled kernel beats over-engineered systolic for small N |
| 5 | Mechanical sympathy | Linear search = sequential access = prefetcher-friendly | Coalesced global loads, no warp divergence |
| 6 | Bypass what you don't need | Kernel-bypass NIC (SolarFlare efvi) | Use `cudaMemcpy` not unified memory |
| 7 | Right tool for right task | 150-line queue vs 100K-line framework | cuBLAS for production, hand-rolled for learning |
| 8 | Staying fast is harder | Alerts + audits on latency metrics | Add `ncu` metrics to CI, not just one-off runs |

---

## Profiling Workflow Summary

```
Step 1: perf stat
  → Get TMA categories: Retiring / Bad Speculation / Front-End / Back-End Bound
  → Tells you WHICH category to investigate (don't guess)

Step 2: perf record (sampling profiler)
  → Identify WHICH function/loop has the problem
  → Look for conditional jumps (JNE/JE) eating >10% of time

Step 3: Hardware PMCs (programmatic)
  perf_event_open() around the hot code
  → Measure branch misses, IPC, cache misses PRECISELY

Step 4: Clang X-Ray (for event-driven / production systems)
  → Zero-overhead when off
  → Toggle on in production without recompile
  → Full function-level latency, not sampling

Key discipline: Fix ONE thing per iteration. Measure before AND after.
"Engineers get excited and start measuring everything and nothing."
```

---

## References

| Resource | Why |
|---|---|
| `low_latency_principles_for_hpc.md` | Same 8 principles mapped to CUDA/GPU context |
| `branchless_and_cache_optimization.md` | Branchless techniques and cache locality details |
| [LLVM X-Ray docs](https://llvm.org/docs/XRay.html) | Full Clang X-Ray reference |
| Mike Acton — *Data-Oriented Design* (CppCon) | Struct-of-arrays, cache-first design |
| Fedor Pikus — *The Speed of Concurrency* (CppCon) | Lock-free queues and atomics |
| `perf` tutorial (Brendan Gregg) | CPU performance analysis workflow |
