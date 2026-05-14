# Qwen2.5-7B Inference Optimization — RTX 3060 Study

**Date:** 2026-05-14  
**Hardware:** Beelink S12 Pro (x86 host) → RTX 3060 12GB eGPU (Ampere SM_86, CUDA 13.2)  
**Framework:** vLLM 0.19.1  
**Model:** Qwen/Qwen2.5-7B-Instruct-AWQ  
**OS:** Rocky Linux 9 (Python 3.11)

---

## Setup

### Environment

```bash
sudo dnf install python3.11 python3.11-pip python3.11-devel -y
python3.11 -m venv ~/vllm-env
source ~/vllm-env/bin/activate
pip install vllm
```

### Gotchas Encountered

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| `pip: command not found` | pip not installed by default | `sudo dnf install python3-pip` |
| `unsupported operand type \| NoneType` | vLLM 0.11 requires Python 3.10+ for union type syntax | Reinstall with Python 3.11 |
| `CalledProcessError: gcc libcuda.so.1` | Triton couldn't find `libcuda.so.1` — searched `/lib64` not `/usr/lib64` | `sudo ln -s /usr/lib64/libcuda.so.1 /lib64/libcuda.so.1` (already existed); real fix was `sudo dnf install python3.11-devel` — only `pyconfig-64.h` existed, full headers were missing |
| `--speculative-model unrecognized` | Flag renamed in vLLM 0.19.x | Use `--speculative-config '{"model": "...", "num_speculative_tokens": 5}'` |
| Vocab size mismatch (7B=152064, 0.5B=151936) | Different tokenizer versions between Qwen2.5 model sizes | Use ngram method or match tokenizer versions |

### Server Start Command (best config)

```bash
python3 -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \
    --quantization awq_marlin \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.85
```

---

## Benchmark Methodology

Two synthetic benchmarks to isolate each inference phase:

### Benchmark 1 — Prefill-heavy (compute-bound)
- **Prompt:** `"Explain quantum computing. " × 100` = 402 tokens
- **Output:** `max_tokens=10` (negligible decode)
- **Measures:** time to process a long context (matrix multiply over full sequence)

### Benchmark 2 — Decode-heavy (memory-bandwidth-bound)
- **Prompt:** `"Tell me a very long story about a robot."` = 10 tokens
- **Output:** `max_tokens=500` (almost entirely decode)
- **Measures:** time to generate tokens sequentially (one token per forward pass)

```bash
time curl -s http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen2.5-7B-Instruct-AWQ", "prompt": "<PROMPT>", "max_tokens": <N>}' \
  > /dev/null
```

---

## Results

### Full Optimization Comparison

| Config | Prefill Time | Prefill tok/s | Decode Time | Decode tok/s |
|--------|-------------|---------------|------------|--------------|
| **AWQ (baseline)** | 1.603s | ~251 tok/s | 78.793s | ~6.4 tok/s |
| **AWQ Marlin** | 0.477s | ~843 tok/s | 7.444s | ~67 tok/s |
| **AWQ Marlin + ngram speculation** | 2.274s | ~176 tok/s | 8.483s | ~59 tok/s |

### Key Numbers

- **Marlin vs AWQ — Prefill:** 3.4× faster
- **Marlin vs AWQ — Decode:** 10.5× faster
- **Prefill/Decode gap (AWQ):** 39× (251 vs 6.4 tok/s)
- **Prefill/Decode gap (Marlin):** 12.5× (843 vs 67 tok/s)
- **VRAM at runtime:** 10,805 MiB / 12,288 MiB (88%)
- **Power draw under load:** 149W / 170W (87% TDP)

---

## Analysis

### Why Prefill and Decode Behave Differently

| Phase | Bottleneck | Why |
|-------|-----------|-----|
| **Prefill** | Compute-bound | All 402 tokens processed simultaneously as one large GEMM — GPU is doing real matrix multiplications |
| **Decode** | Memory-bandwidth-bound | One token generated per forward pass — each step loads all 7B model weights from VRAM, saturating GDDR6 (360 GB/s) |

`nvidia-smi` shows 100% GPU util for **both** phases — this is misleading. The metric shows SM occupancy, not whether the SM is doing compute vs waiting for memory. Use Nsight Compute (ncu) for true roofline analysis.

### Why Marlin Is So Much Better for Decode

AWQ stores weights in 4-bit quantized form. To multiply, the GPU must:
1. Load 4-bit weights from VRAM
2. Dequantize to FP16
3. Multiply

**AWQ kernel** does this sequentially — dequantize, then multiply.  
**Marlin kernel** fuses dequantization + GEMM into a single optimized kernel, dramatically reducing memory transactions per token. This is why the decode speedup (10.5×) is so much larger than the prefill speedup (3.4×) — decode is the phase that is most bottlenecked by memory transactions.

### Why the Prefill/Decode Gap Narrowed (39× → 12.5×)

Marlin's dequantization efficiency primarily benefits **decode** (memory-bound). Prefill was already relatively fast (compute-bound). So Marlin lifted decode throughput proportionally more, closing the gap between the two phases.

### Why ngram Speculation Regressed Performance

Ngram speculative decoding works by:
1. Finding n-gram patterns from the **input** that might appear in the **output**
2. Speculatively generating multiple tokens using those patterns
3. Verifying in batch with the main model

**It only wins when output text mirrors input text.** For `"Tell me a long story about a robot"` — a creative generative task with a 10-token input — there are essentially zero overlapping n-grams between input and output. The result: speculation overhead with zero benefit.

**Ngram is useful for:**
- Code completion (output continues input code patterns)
- Document summarisation (key phrases from long doc appear in summary)
- RAG / question answering (retrieved context phrases appear in answer)

**Ngram is not useful for:**
- Open-ended generation
- Creative tasks
- Short prompts → long outputs

---

## Connection to APAC HPC-AI Competition

The 39× prefill/decode throughput gap (even before Marlin) is the fundamental motivation for **disaggregated prefill-decode inference**:

- **Prefill nodes** can be compute-optimized (high FLOPS, smaller memory bandwidth)
- **Decode nodes** can be memory-bandwidth-optimized (large HBM, lower FLOPS)
- Separating them allows each to be scaled independently based on request mix

On NSCC ASPIRE2A with Qwen3-VL-235B, this gap will be even more pronounced due to the model size. The KV cache transfer between prefill and decode nodes (Mooncake, P/D-Serve, FlowKV) is the key engineering challenge.

---

## LinkedIn Data Points

> Profiled Qwen2.5-7B-Instruct (AWQ 4-bit) inference on RTX 3060 using vLLM:
> - **~251 tok/s prefill** vs **~6.4 tok/s decode** baseline — 39× throughput gap demonstrating compute-bound vs memory-bandwidth-bound inference phases
> - AWQ-Marlin kernel: **3.4× prefill** and **10.5× decode** speedup (67 tok/s) over standard AWQ
> - Ngram speculative decoding: regression on creative tasks, confirming workload-dependent optimization tradeoffs
> - Preparation for disaggregated Qwen3-VL-235B inference at APAC HPC-AI 2026 on NSCC ASPIRE2A

---

## Next Steps (Pending)

- [ ] **Step 3** — Batched concurrent requests (`bench_batch.py`) to measure real serving throughput vs single-request latency
- [ ] **Step 5** — KV cache tuning (`--max-model-len 8192`, `--gpu-memory-utilization 0.90`)
- [ ] **Step 6** — FlashInfer attention backend (`--attention-backend flashinfer`)
- [ ] **Step 7** — Speculative decoding with matched-tokenizer draft model (Qwen2.5-1.5B-Instruct, verify vocab match first)
- [ ] **ASPIRE2A** — Port optimized config to multi-GPU disaggregated setup for Qwen3-VL-235B
