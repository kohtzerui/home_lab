# SoC Compute Cluster — Qwen3-VL vLLM Setup Guide

**Last verified:** 2026-05-19  
**Node used:** xgpi1 (H100 NVL 96GB)  
**Working stack:** vLLM 0.15.1 + torch 2.9.1 (CUDA 12.6) + transformers 4.52.x

---

## 1. SSH Access

```bash
ssh tzerui@xlogin1.comp.nus.edu.sg
# Username: tzerui (SoC account, NOT NUS-ID)
# Note: From Jul 2025, DocHub login requires SoC account credentials only
```

---

## 2. GPU Node Reference

From `sinfo -o "%N %G"`:

| Node | GRES name (for --gres flag) | VRAM | Notes |
|------|---------------------------|------|-------|
| xgpk0 | `gpu:h200-141:4` | 4× 141GB | Usually 100% full |
| xgpi[0-12] | `gpu:h100-96:2` | 2× 96GB | Best for competition prep |
| xgpi[13-24] | `gpu:h100-47:4` | 4× 47GB MIG | More available |
| xgph[0-9], xgpj0 | `gpu:a100-80:1` | 80GB | Good for 32B testing |
| xgpg[0-9] | `gpu:a100-40:1` | 40GB | Smaller models only |

Check availability: `sinfo -o "%N %G %T" | grep -i "idle\|mix"`

---

## 3. Request an Interactive GPU Node

```bash
# H100-96 (preferred — 96GB, good for Qwen3-VL-8B/32B)
srun --partition=gpu --gres=gpu:h100-96:1 --mem=100G --cpus-per-task=16 --time=02:00:00 --pty bash

# A100-80 (fallback — easier to get, good for Qwen3-VL-32B)
srun --partition=gpu --gres=gpu:a100-80:1 --mem=100G --cpus-per-task=8 --time=02:00:00 --pty bash

# H100-47 MIG (most available)
srun --partition=gpu --gres=gpu:h100-47:1 --mem=50G --cpus-per-task=16 --time=02:00:00 --pty bash
```

---

## 4. Storage Layout

| Path | Size | Use |
|------|------|-----|
| `~` (home) | 185T pool, 65T free | Code, small files |
| `/mnt/scratch` | 161T pool, 67T free | Large model files (need to check permissions) |

**Note:** `/mnt/scratch/tzerui` was permission-denied on 2026-05-19. Use home dir (`~`) for now.  
Models stored at: `~/models/`

---

## 5. Python Environment Setup (first time only)

```bash
mkdir -p ~/models ~/envs ~/huggingface_cache
export HF_HOME=~/huggingface_cache

python3 -m venv ~/envs/vllm_env
source ~/envs/vllm_env/bin/activate
pip install --upgrade pip

# Install torch for CUDA 12.4 (compatible with driver 12.9)
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu124

# Install vLLM 0.15.1 (has native Qwen3-VL support, uses CUDA 12.x torch)
pip install vllm==0.15.1

# Fix transformers version (4.52.x supports qwen3_vl, stays on 4.x API)
pip install "transformers>=4.52,<5.0"

# Patch aimv2 conflict (vLLM tries to re-register a model type transformers now owns)
sed -i 's/AutoConfig.register("aimv2", AIMv2Config)/AutoConfig.register("aimv2", AIMv2Config, exist_ok=True)/' \
    ~/envs/vllm_env/lib/python3.12/site-packages/vllm/transformers_utils/configs/ovis.py
```

---

## 6. Download Models

```bash
# Qwen3-VL-8B (~17.5GB, ~85 seconds on cluster network)
hf download Qwen/Qwen3-VL-8B-Instruct --local-dir ~/models/Qwen3-VL-8B

# Qwen3-VL-32B (~64GB) — fits on H100-96
# hf download Qwen/Qwen3-VL-32B-Instruct --local-dir ~/models/Qwen3-VL-32B
```

---

## 7. Activate Environment (every session)

```bash
source ~/envs/vllm_env/bin/activate
export FLASHINFER_DISABLE_VERSION_CHECK=1
export HF_HOME=~/huggingface_cache
```

---

## 8. Start vLLM Server

```bash
# Session 1 (on compute node):
python3 -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen3-VL-8B \
    --dtype bfloat16 \
    --max-model-len 4096 \
    --port 8000 &

# Wait ~40 seconds for model load + CUDA graph capture
# Look for: "Application startup complete."
```

Enabled by default at 0.15.1:
- `enable_prefix_caching=True` (Radix Attention)
- `enable_chunked_prefill=True`
- `FLASH_ATTN` backend
- CUDA graph capture (51 batch sizes)

---

## 9. Test the API

```bash
# Session 2 (from login node or any SSH session):
curl http://xgpi1:8000/health

curl http://xgpi1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "/home/t/tzerui/models/Qwen3-VL-8B",
    "messages": [{"role": "user", "content": "What is the capital of France?"}],
    "max_tokens": 50
  }'
```

---

## 10. Known Issues & Fixes

| Issue | Fix |
|-------|-----|
| `torch._C._cuda_init() RuntimeError: driver too old` | vLLM 0.21.0 needs CUDA 13.x. Use `pip install vllm==0.15.1` |
| `ValueError: 'aimv2' is already used by Transformers` | Patch ovis.py: add `exist_ok=True` (see step 5) |
| `Qwen3VLConfig has no attribute 'vocab_size'` | transformers too old. Run `pip install "transformers>=4.52,<5.0"` |
| `AttributeError: Qwen2Tokenizer has no attribute all_special_tokens_extended` | transformers 5.x breaks vLLM 0.9.1. Stay on 4.x |
| `flashinfer-cubin version mismatch` | `export FLASHINFER_DISABLE_VERSION_CHECK=1` |
| `Unable to allocate resources: Requested node configuration is not available` | Wrong GRES name. Use lowercase: `h100-96`, `a100-80`, `h200-141` |
| `/mnt/scratch/tzerui: Permission denied` | Use `~/` home dir instead |

---

## 11. Verified Performance (2026-05-19, xgpi1, H100 NVL 96GB)

- Model load time: ~3 seconds (weights), ~30 seconds total (warmup + CUDA graphs)
- KV cache: 60.55 GiB available, 440,928 token capacity
- Max concurrency: 107× at 4096 tokens per request
- Prefix cache hit rate: 42.2% with shared 50-token system prompt
- Prompt throughput with prefix cache: 25.1 tok/s (vs 1.5 tok/s cold)

---

## 12. Next Steps

- [ ] Test with actual image inputs (VL pipeline)
- [ ] Run vllm benchmark: `python3 -m vllm.benchmarks.benchmark_throughput`
- [ ] Test FP8 KV cache: add `--kv-cache-dtype fp8` to server args
- [ ] Test with Qwen3-VL-32B on full 2× H100 (request 2 GPUs)
- [ ] Request `/mnt/scratch` write access for large model storage


# 1. Get a node
srun --partition=gpu --gres=gpu:h100-96:1 --mem=100G --cpus-per-task=16 --time=02:00:00 --pty bash

# 2. Activate
source ~/envs/vllm_env/bin/activate

# 3. Set env vars
export FLASHINFER_DISABLE_VERSION_CHECK=1

# 4. Start server
python3 -m vllm.entrypoints.openai.api_server \
    --model ~/models/Qwen3-VL-8B --dtype bfloat16 --max-model-len 4096 --port 8000 &
