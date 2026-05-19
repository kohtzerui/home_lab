# Advanced Inference Optimizations — Qwen3-VL-235B on ASPIRE2A

**Date:** 2026-05-19  
**Context:** APAC HPC-AI 2026 competition — Qwen3-VL-235B-A22B (MoE, 22B active) on NSCC ASPIRE2A  
**Prerequisite reading:** `qwen_inference_optimization.md` (RTX 3060 benchmarks with Qwen2.5-7B)

---

## 1. Speculative Decoding (Draft Model)

> ⚠️ **MLPerf Compliance Warning:** Speculative decoding is only explicitly allowed for DeepSeek-r1-Interactive in the MLCommons Closed division rules (Appendix C). Qwen3-VL-235B-A22B is **NOT listed**. The APAC competition may have its own rules — confirm with organizers before investing time here.

### What it is

Use a small model to *guess* multiple tokens, then verify them all in a single forward pass of the big model. The big model's output quality is **mathematically preserved** — rejected tokens fall back to the target distribution via rejection sampling.

### Why it works

Decode is memory-bandwidth-bound. Each forward pass of the 235B model loads all active weights from HBM regardless of whether it produces 1 token or verifies 5. Speculative decoding amortizes that cost:

```
Without speculation:
  5 tokens = 5 forward passes of 235B model = 5 × full weight load

With speculation (k=5, acceptance rate ~80%):
  Draft: 5 forward passes of 3B model (cheap, ~10× faster per pass)
  Verify: 1 forward pass of 235B model (verifies all 5 at once)
  Result: ~4 accepted tokens per 235B forward pass
  Effective speedup: ~3-4× decode throughput
```

### Draft model candidates for Qwen3-VL-235B

| Draft Model | Active Params | Speed vs 235B | Expected Acceptance Rate | Memory Overhead |
|-------------|--------------|---------------|-------------------------|-----------------|
| **Qwen3-VL-2B** | 2B | ~10× faster | Lower (~60-70%) | ~4 GB |
| **Qwen3-VL-8B** | 8B | ~3× faster | Medium (~75-80%) | ~16 GB |
| **Qwen3-VL-30B-A3B** | 3B active (MoE) | ~7× faster | **Highest (~80-85%)** | ~60 GB total weights |

**Recommended: Qwen3-VL-30B-A3B** — trained at 30B scale so its token distribution closely matches the 235B target, but only activates 3B parameters per token, making it nearly as fast as a dense 3B model. Same tokenizer, same vision encoder.

### What ngram speculation does differently (and why it's limited)

Ngram speculation is **not** speculative decoding. It uses no draft model — it pattern-matches repeated n-grams from the input prompt to predict output tokens. This only works when output text mirrors input text (summarization, code completion). For creative/reasoning tasks or vision-language tasks where the output is structured JSON, ngram acceptance rate approaches zero.

**Use ngram only as a fallback when no compatible draft model is available.** For Qwen3-VL, compatible draft models exist, so use them.

### vLLM configuration

```bash
# Draft model speculative decoding (the real thing)
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B \
    --speculative-config '{"model": "Qwen/Qwen3-VL-30B-A3B", "num_speculative_tokens": 5}' \
    --tensor-parallel-size 8
```

### Key considerations

- **Memory budget:** Both models must fit in GPU memory simultaneously. On 8× A100-80GB (640 GB total), the 235B target takes ~470 GB (INT4), leaving ~170 GB for draft model + KV caches.
- **Acceptance rate drops with temperature:** Higher sampling temperature → more randomness → draft predictions diverge from target → lower acceptance. At temperature=0 (greedy), acceptance is maximized.
- **Vision tokens:** The ViT encoder output is shared between draft and target (compute once, feed to both). This is a Qwen3-VL-specific advantage.

---

## 2. Chunked Prefill

### The problem

A single long-context prefill (e.g., an image-heavy prompt generating 2000+ tokens from the ViT encoder) monopolizes the GPU for hundreds of milliseconds. During this time, **all decode requests in the batch are stalled** — their inter-token latency spikes.

```
Without chunked prefill:
  Time ──→
  GPU: [========= PREFILL (300ms) =========][decode][decode][decode]
                                             ↑
                                   All decode requests waited 300ms

With chunked prefill (chunk_size=512):
  GPU: [prefill_chunk1][decode batch][prefill_chunk2][decode batch][prefill_chunk3][decode batch]
                        ↑                             ↑                             ↑
              Decode runs every ~10ms — latency stays low
```

### Why it matters for Qwen3-VL specifically

Qwen3-VL processes images through a Vision Transformer (ViT) that converts each image into **hundreds of visual tokens**. A single image can produce 256-1024 tokens. A request with multiple images easily generates 2000+ prefill tokens — enough to block decode for 200-500ms at 235B scale.

### vLLM configuration

```bash
# Enable chunked prefill with 512-token chunks
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B \
    --enable-chunked-prefill \
    --max-num-batched-tokens 512
```

### Trade-off

Chunked prefill slightly increases **total prefill time** (more scheduling overhead per chunk) but dramatically reduces **tail latency for decode requests**. For SLO-bound workloads where you need P99 latency < X ms, this is non-negotiable.

---

## 3. KV Cache Compression

### What it is

Model quantization (AWQ, GPTQ) shrinks the **model weights** from FP16 to INT4. KV cache compression is a separate technique that shrinks the **intermediate KV cache tensors** generated during inference. These are different memory pools:

```
GPU Memory Layout:
┌─────────────────────────────────┐
│  Model Weights (quantized)      │  ← AWQ/GPTQ handles this
│  ~470 GB for 235B INT4          │
├─────────────────────────────────┤
│  KV Cache (full FP16 by default)│  ← KV compression handles this
│  Grows with: num_sequences ×    │
│    context_length × num_layers  │
│    × head_dim × 2 (K and V)    │
├─────────────────────────────────┤
│  Activations (temporary)        │
└─────────────────────────────────┘
```

For Qwen3-VL-235B with 256K context support, the KV cache can easily dominate memory usage.

### Techniques

| Technique | Compression | Quality Impact | Use Case |
|-----------|-------------|---------------|----------|
| **FP8 KV cache** | 2× smaller | Negligible | Default choice — almost free |
| **INT4 KV cache** | 4× smaller | Small, measurable | When you need maximum concurrent sequences |
| **GQA (built into Qwen3)** | 4-8× smaller | Zero (architectural) | Already active — Qwen3 uses fewer K/V heads than Q heads |
| **Attention sinks (StreamingLLM)** | Unbounded savings | Lossy for middle context | Infinite-length streaming, not for accuracy-sensitive tasks |

### Why GQA matters (Qwen3 already does this)

Traditional multi-head attention: 32 Q heads, 32 K heads, 32 V heads → KV cache is 1:1 with Q.  
Grouped Query Attention: 32 Q heads, **8 K heads, 8 V heads** → KV cache is **4× smaller** for free.

Qwen3 uses GQA natively. This is why it can handle 256K context at all — without GQA, the KV cache would be 4× larger and wouldn't fit in memory.

### vLLM configuration

```bash
# FP8 KV cache quantization
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B \
    --kv-cache-dtype fp8
```

---

## 4. Kernel Fusion

### The general principle

Every GPU kernel launch involves:
1. CPU prepares arguments → sends to GPU driver → driver dispatches to SMs
2. Kernel loads data from HBM → computes → writes results back to HBM
3. Next kernel loads those results from HBM again

**Fusing** two kernels eliminates step 2's write and step 3's re-read — intermediate results stay in fast on-chip SRAM (shared memory / registers).

### Specific fusion opportunities

| Operation | Unfused | Fused | Memory Saved |
|-----------|---------|-------|-------------|
| **Dequant + GEMM** | Load INT4 → dequant to FP16 → write → load → GEMM | Load INT4 → dequant in registers → GEMM | 1 full tensor write+read eliminated |
| **QKV Projection** | 3 separate GEMMs (Q, K, V) | 1 GEMM with 3× output width | 2 full weight loads eliminated |
| **RMSNorm + Linear** | Normalize → write → load → project | Normalize in registers → project | 1 activation tensor write+read |
| **Attention (FlashAttention)** | Compute QK^T → write N×N matrix → softmax → write → multiply by V | Tile-based: never materialize full N×N | O(N²) → O(N) memory |
| **MoE Dispatch** | Route tokens → scatter → per-expert GEMM → gather | Fused routing + batched expert GEMM | Scatter/gather eliminated |

### Marlin (you already benchmarked this)

The 10.5× decode speedup you measured (AWQ → AWQ Marlin) was entirely from the first row: fusing dequantization into the GEMM kernel. This eliminated a full round-trip to HBM per weight matrix per token.

### FlashAttention / FlashInfer

FlashAttention computes attention **without ever materializing the N×N attention matrix**. For a 4096-token context:
- Unfused: allocates 4096 × 4096 × FP16 = 32 MB per head per layer — **just for an intermediate result**
- FlashAttention: processes attention in tiles using only ~256 KB of shared memory

FlashInfer goes further: optimized specifically for the **decode phase** where the query is a single token against a long KV cache (1×N attention instead of N×N).

```bash
# Use FlashInfer backend for decode-optimized attention
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-VL-235B-A22B \
    --attention-backend flashinfer
```

---

## 5. CUDA Graph Capture

### The problem

Each decode step runs a forward pass through all transformer layers. Each layer invokes multiple GPU kernels (attention, feedforward, normalization). For a 235B model, that's hundreds of kernel launches per token.

Each kernel launch has CPU-side overhead:
```
Python → PyTorch dispatcher → CUDA driver → GPU execution
         ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
                    ~10-50 μs per kernel launch
                    × 500 kernels per forward pass
                    = 5-25 ms of CPU overhead per token
```

When the GPU kernels themselves take <1ms each (memory-bandwidth-bound decode), this CPU overhead can be **50%+ of total latency**.

### The fix

**CUDA Graphs** record the entire forward pass once, then replay it as a single monolithic GPU operation:

```
First forward pass: record all kernel launches into a graph
Subsequent passes:  replay(graph) — one CPU call, entire forward pass executes

CPU overhead: ~10 μs per token (instead of 5-25 ms)
```

### Constraints

- Requires **fixed tensor shapes** — the batch size and sequence length must be constant between replays
- vLLM handles this automatically by maintaining a pool of pre-captured graphs for common batch sizes
- When batch size changes (request arrives/completes), vLLM falls back to eager execution for that step, then re-captures

### Why your batch=8 anomaly may have been this

At batch=8, if vLLM didn't have a pre-captured graph for that batch size, it fell back to eager mode. batch=4 and batch=16 may have hit cached graphs, explaining their much better performance.

---

## 6. Paged KV Cache (PagedAttention)

### What you already use (via vLLM) — but the mechanics matter

Traditional KV cache: pre-allocate a **contiguous** buffer per sequence for the maximum possible context length.

```
Traditional (max_len=4096, 100 sequences):
  Seq 0: [████░░░░░░░░░░░░]  ← 200 tokens used, 3896 wasted
  Seq 1: [██░░░░░░░░░░░░░░]  ← 50 tokens used, 4046 wasted
  ...
  Total: 100 × 4096 × (per-token KV size) allocated
  Utilization: ~5-10% typical
```

PagedAttention: allocate KV cache in **4KB pages** on demand, like virtual memory.

```
Paged (page_size=16 tokens):
  Seq 0: [page_0][page_1]...[page_12] → 200 tokens, 13 pages, ~3 tokens wasted
  Seq 1: [page_0]...[page_3]          → 50 tokens, 4 pages, ~14 tokens wasted
  ...
  Total: only pages actually needed are allocated
  Utilization: ~95%+
```

### Why this matters at 235B scale

With a 235B MoE model, KV cache per token per layer is large. The difference between 5% and 95% memory utilization directly translates to **how many concurrent sequences you can serve** — which determines throughput under batched workloads.

The `--gpu-memory-utilization 0.90` flag you tested controls how much total VRAM is reserved for the page pool. Higher = more pages = more concurrent sequences = higher throughput, but less headroom for activation memory spikes.

---

## 7. Continuous Batching with Preemption

### Beyond static batching

Your RTX 3060 benchmarks used static batching — fire N requests simultaneously, wait for all to finish. Real serving has requests arriving continuously at unpredictable times.

### How continuous batching works

```
Time ──→

Static batching:
  [Req 0 ████████████████████]
  [Req 1 ████████████████████]  ← must wait for longest request
  [Req 2 ████████████████████]
  [                           Req 3 waits in queue until batch completes]

Continuous batching:
  [Req 0 ████████████]         ← finishes early, slot freed
  [Req 1 ████████████████████]
  [Req 2 ████████]             ← finishes, slot freed
  [         Req 3 ████████████████]  ← inserted mid-batch
  [                  Req 4 ████████] ← inserted mid-batch
```

Each decode iteration, the scheduler:
1. Checks if any sequence has finished (hit EOS or max_tokens) → free its KV pages
2. Checks if new requests are waiting → prefill them into the freed slots
3. Runs one decode step for all active sequences simultaneously

### Preemption (SLO-aware scheduling)

When GPU memory is full but a high-priority request arrives:
1. **Swap:** evict a low-priority sequence's KV cache to CPU RAM
2. Run the high-priority request
3. Later, swap the evicted sequence back and resume it (no recomputation)

This is how frameworks meet **latency SLOs** — sacrifice throughput on low-priority work to guarantee P99 latency on high-priority work.

---

## 8. MoE-Specific: Expert Parallelism and Load Balancing

### Why this matters for Qwen3-VL-235B specifically

Qwen3-VL-235B is a Mixture-of-Experts model: 235B total parameters, but only **22B active** per token. Each token is routed to a subset of "expert" sub-networks by a learned gating function.

### The routing bottleneck

```
Standard Tensor Parallelism (TP=8):
  Each GPU holds 1/8 of every expert
  Token arrives → all 8 GPUs compute 1/8 of selected experts → all-reduce

Expert Parallelism (EP=8):
  Each GPU holds ALL of some experts (e.g., GPU 0 has experts 0-7, GPU 1 has experts 8-15)
  Token arrives → routed to the GPU holding the selected expert → point-to-point transfer

Hybrid TP+EP:
  TP within a node (NVLink, fast), EP across nodes (InfiniBand, slower)
```

### Load imbalance problem

Some experts are "hot" (selected by many tokens) and others are "cold" (rarely selected). If hot experts land on the same GPU, that GPU becomes the bottleneck.

**Mitigation strategies:**
- **Expert replication:** duplicate hot experts across multiple GPUs
- **Token dropping:** if an expert's queue is full, route overflow tokens to the next-best expert
- **Auxiliary load balancing loss:** Qwen3 is trained with a loss term that encourages uniform expert usage

---

## 9. Within-Boundary Sample Sorting (Scheduling Optimization)

### What it is

MLPerf forbids sorting samples **across** dataset boundaries (i.e., globally reordering the entire run to group similar-length queries). But sorting **within** a single dataset pass is allowed.

This means you can use a small reorder buffer — similar to what a production server's scheduler would do — to group queries by sequence length within a batch:

```
Random arrival order:   [2048 tok] [128 tok] [1900 tok] [200 tok] [50 tok] [1800 tok]
                         ↓ reorder within buffer window
Sorted within window:   [50 tok] [128 tok] [200 tok] [1800 tok] [1900 tok] [2048 tok]
                         ↓ batch short ones together
Batch 1: [50, 128, 200]     → minimal padding waste, fast
Batch 2: [1800, 1900, 2048] → uniform lengths, efficient
```

### Why it helps

Without sorting, a batch of `[50, 2048, 128]` pads the 50-token and 128-token queries to 2048 — wasting compute on padding tokens. Sorting by length within the scheduling window minimizes this waste.

### Qwen3-VL context

Qwen3-VL inputs contain both text and images. Image-heavy prompts produce many more tokens from the ViT encoder than text-only prompts. Sorting by total token count (text + visual tokens) within a scheduling window could reduce padding overhead significantly.

### What to investigate

- vLLM's scheduler already does some length-aware batching — measure how much additional gain manual sorting provides
- Determine the maximum reorder buffer size that stays within MLPerf "within boundary" rules
- Profile the padding waste on the Shopify Product Catalogue dataset specifically

---

## 10. xgrammar — Constrained JSON Decoding

### Why it matters for Qwen3-VL specifically

Qwen3-VL-235B outputs a **structured JSON object** per query:
```json
{"category": "Electronics > Computers > Laptops", "brand": "Apple", "is_secondhand": false}
```

Without constrained decoding, the model samples freely from the vocabulary — it can produce invalid JSON (unclosed brackets, wrong field names) which fails the `Category Hierarchical F1 Score >= 0.7824` quality check and wastes tokens on retries.

### How xgrammar works

At each decode step, xgrammar computes which tokens are **valid continuations** of the current JSON state (e.g., after `{"category": "`, only string characters are valid — not `}` or `,`). It masks out all invalid tokens before sampling. This:
- **Eliminates malformed outputs** — every response is valid JSON
- **Speeds up decoding** — the model doesn't waste steps on invalid token paths
- **Improves effective F1** — structured output maps more cleanly to the category hierarchy

### SGLang configuration
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-235B-A22B \
    --grammar-backend xgrammar
```

### vLLM equivalent
```bash
# vLLM uses "guided decoding" with similar effect
--guided-decoding-backend xgrammar
```

### Caveat
Constrained decoding adds a small CPU overhead per decode step (mask computation). Profile whether the quality gain outweighs the latency cost at your target QPS.

---

## 11. Radix Attention (Prefix Caching)

### What it is

SGLang's **RadixAttention** stores KV cache entries in a radix tree indexed by token sequences. When a new request shares a prefix with a previously computed request, the prefix's KV cache is reused — no recomputation.

```
Request 1: [system_prompt][image_tokens][user_query_A] → full prefill
Request 2: [system_prompt][image_tokens][user_query_B]
                ↑ same prefix ↑
           KV cache hit for system_prompt + image_tokens → only user_query_B needs prefill
```

### Why it matters for the Shopify benchmark

If all 48,289 queries share the same system prompt (e.g., *"You are a product classifier. Output JSON with fields: category, brand, is_secondhand."*), the first request computes and caches that prefix. Every subsequent request gets a full cache hit on the system prompt — potentially eliminating 10-30% of prefill work across the run.

### SGLang configuration
Enabled by default. Disable only when using DP attention (they conflict):
```bash
# Default (radix cache ON):
python3 -m sglang.launch_server --model-path Qwen/Qwen3-VL-235B-A22B

# Disable for DP attention configs:
python3 -m sglang.launch_server --model-path Qwen/Qwen3-VL-235B-A22B \
    --disable-radix-cache --enable-dp-attention
```

### vLLM equivalent
vLLM calls this **prefix caching** (`--enable-prefix-caching`). Less sophisticated than RadixAttention (no tree structure, simpler hash-based lookup).

---

## 12. Data Parallelism for Attention (DP Attention)

### The problem with pure Tensor Parallelism at high batch sizes

With TP=16 across 2 nodes, every attention operation requires an all-reduce across all 16 GPUs — expensive InfiniBand communication at every layer. At high batch sizes where the bottleneck is KV cache capacity rather than compute, this communication overhead dominates.

### How DP Attention works

Instead of splitting each attention head across all GPUs, divide GPUs into groups that each handle a subset of the batch independently:

```
TP=16 (standard):
  All 16 GPUs process every token together → 15 all-reduces per layer

TP=8, DP=2:
  GPU group 0 (8 GPUs): handles batch slice 0 → 7 all-reduces per layer
  GPU group 1 (8 GPUs): handles batch slice 1 → 7 all-reduces per layer
  → Half the communication, each group fully contained within one node (NVLink, fast)
```

### Configuration for 2 nodes × 8 H100s

```bash
# Option A: TP=16 across nodes (high communication)
--tp 16 --nnodes 2

# Option B: TP=8 per node + DP=2 (lower inter-node communication) ← likely better
--tp 16 --dp 2 --enable-dp-attention --nnodes 2

# Option C: Pure TP=8 per node, DP=2 for attention only
--tp 8 --dp 2 --enable-dp-attention
```

### When to use DP attention

- **Use DP attention** when: high batch sizes, KV cache is the bottleneck, nodes connected by InfiniBand (slow inter-node)
- **Don't use DP attention** when: small batch sizes, low-latency requirements, NVLink available across all GPUs

> ⚠️ DP attention is **incompatible with Radix Attention** — you must choose one or the other. Profile both on your target workload.

---

## 13. P/D Multiplexing (--enable-pdmux)

### Beyond static P/D disaggregation

Basic P/D disaggregation statically assigns GPUs as either "prefill workers" or "decode workers". If the request mix shifts (e.g., a burst of long prompts arrives), prefill workers get overloaded while decode workers sit idle.

**P/D Multiplexing** allows GPUs to dynamically switch roles:
- During a prefill burst → more GPUs act as prefill workers
- During a decode-heavy period → more GPUs act as decode workers
- Fine-grained scheduling at the request level, not the node level

### SGLang configuration
```bash
python3 -m sglang.launch_server \
    --model-path Qwen/Qwen3-VL-235B-A22B \
    --disaggregation-mode "prefill" \   # or "decode" for the other pool
    --enable-pdmux
```

### Trade-off
pdmux adds scheduling overhead and requires more complex coordination between workers. For a benchmark with predictable load (offline scenario), static P/D disaggregation may actually outperform pdmux.

---

## 14. SGLang vs vLLM — Framework Choice

### Why this matters

All earlier configs assume vLLM. The previous SBCC DeepSeek team explicitly chose **SGLang** for MoE inference and found it better. Key differences:

| Feature | vLLM | SGLang |
|---------|------|--------|
| Radix Attention | Basic prefix caching | Full RadixAttention tree |
| MoE support | Good | Optimized (DeepSeek native) |
| DP Attention | Limited | Built-in `--enable-dp-attention` |
| xgrammar | Yes (`--guided-decoding-backend`) | Yes (`--grammar-backend`) |
| P/D disaggregation | Yes (mooncake) | Yes (`--disaggregation-mode`) |
| Qwen3-VL support | ✅ Good | ✅ Being added |
| Community/docs | Larger | Smaller but growing |

### Recommendation
Benchmark **both** frameworks on ASPIRE2A with the Shopify dataset. SGLang is likely better for MoE throughput; vLLM may be easier to configure for Qwen3-VL's vision components.

---

## 15. Stream Interval Tuning

### What it controls

`--stream-interval N` controls how many tokens are buffered before being sent to the client:
- `N=1` (default): send every token immediately → lowest latency, highest overhead
- `N=10`: send every 10 tokens → higher throughput, slightly higher TTFT

### For offline/throughput benchmarks

In the **offline scenario** (all queries sent at once, measure total throughput), latency per token doesn't matter — only total tokens generated per second. A higher stream interval reduces token-by-token synchronization overhead.

```bash
--stream-interval 10   # start here, benchmark vs default
```

---

## 16. FlashAttention 3 (H100-specific)

### What's new in FA3

FlashAttention 3 (FA3) targets Hopper architecture (H100) specifically, using:
- **Warp-specialization**: separate warps for memory loads vs compute, fully overlapping
- **FP8 support**: native FP8 attention on H100 tensor cores
- Benchmarks show **~2× speedup** over FA2 on H100 for long contexts

### Why it wasn't stable last year

The SBCC team noted FA3 + Triton as "working on" — it was still experimental. As of mid-2026, FA3 is more stable in both SGLang and vLLM.

### Configuration
```bash
# SGLang
--attention-backend fa3

# vLLM
--attention-backend flash_attn  # check if fa3 is available in your vLLM version
```

> Only relevant if ASPIRE2A has H100s. Verify GPU type first — FA3 has no benefit on A100.

---

## Summary: Optimization Priority for APAC Competition

```
Priority   Optimization                Expected Impact    Effort     MLPerf Status
──────────────────────────────────────────────────────────────────────────────────────
   1       P/D Disaggregation          1.5-2× throughput  Medium     ✅ Allowed
   2       xgrammar JSON decoding      Quality + speed    Low        ✅ Allowed
   3       Quantization (AWQ/FP8)      2× memory savings  Low        ✅ Allowed
   4       KV Cache FP8               2× more sequences   Low        ✅ Allowed
   5       Radix Attention            10-30% prefill      Low        ✅ Allowed
   6       DP Attention (tp8+dp2)     Throughput at scale Low        ✅ Allowed
   7       Chunked Prefill            P99 latency fix     Low        ✅ Allowed
   8       FlashAttention 3           ~2× attn on H100   Low        ✅ Allowed (H100 only)
   9       FlashInfer attention       10-30% decode       Low        ✅ Allowed
  10       CUDA Graph tuning          5-15% decode        Medium     ✅ Allowed
  11       Within-boundary sorting    5-15% less padding  Low        ✅ Allowed
  12       Stream interval tuning     5-10% throughput    Low        ✅ Allowed
  13       SGLang (framework swap)    Variable            Medium     ✅ Allowed
  14       P/D Multiplexing (pdmux)   Dynamic scaling     Medium     ✅ Allowed
  15       Expert parallelism         10-20% at scale     High       ✅ Allowed
  16       Speculative Decoding       2-4× decode speed   Low        ⚠️ Ask organizers
  17       Custom kernel fusion       5-15%               Very High  ✅ Allowed
  18       NCCL / system tuning       5-10%               Medium     ✅ Allowed
```

Items 1-12 are **configuration-level flags** — change and measure. Items 13-18 require deeper engineering or are subject to rule clarification.
