# vLLM DGX Spark — DiffusionGemma 26B A4B-it

**Source:** [hermes-speedboat/vllm.diffusiongemma-26b_spark](https://github.com/hermes-speedboat/vllm.diffusiongemma-26b_spark)

Optimized vLLM inference setup for **DGX Spark (GB10, Grace Blackwell ARM64)** with 128 GB unified memory, running Google's experimental **DiffusionGemma 26B A4B** model.

## Quickstart

```bash
cd /srv/vllm2
git clone https://github.com/hermes-speedboat/vllm.diffusiongemma-26b_spark.git .
bash setup_vllm.sh        # uv + venv + vllm from main (diffusion gemma support)
bash download_model.sh     # ~48 GB BF16 model via snapshot_download()
bash vllm-server.sh        # start server on port 8000
```

## Performance Expectations (DGX Spark GB10)

| Metric | Value |
|--------|-------|
| Model size | 25.2B params (3.8B active), 48 GB BF16 |
| Quantized | ~24 GB via FP8 (`--quantization fp8`) |
| Speed (H100 ref) | 1000+ tok/s |
| Speed (RTX 5090 ref) | 700+ tok/s |
| Context | Configurable up to 256K tokens |

**Note:** Speed on DGX Spark (GB10) will differ from H100/RTX 5090 benchmarks. The model uses block-autoregressive diffusion (256-token canvas) which shifts the decode bottleneck from memory-bandwidth to compute.

## Architecture

- **Model:** DiffusionGemma 26B A4B-it (multimodal — text + image → text)
- **Quantization:** FP8 via vLLM `--quantization fp8` (~24 GB, fits in 30 GB budget)
- **Format:** BF16 safetensors (via `snapshot_download` from `huggingface_hub`)
- **Inference:** vLLM main branch with native DiffusionGemma support
- **Hardware:** DGX Spark (GB10), 128 GB unified memory
- **Attention backend:** TRITON_ATTN (required — shares Gemma 4's 512-dim attention)
- **Canvas:** 256 tokens per forward pass (parallel denoising)
- **Context:** Up to 262144 tokens
- **Diffusion steps:** 48 max denoising steps (Entropy-Bounded Sampler)
- **Reasoning:** Structured via `<|think|>` tokens (`--reasoning-parser gemma4`)
- **Function Calling:** Native structured tool use (`--tool-call-parser gemma4`)

## Why DiffusionGemma?

- **4× faster inference** than autoregressive models by generating 256-token blocks in parallel
- **3.8B active params** out of 25.2B total (MoE) — low compute per token
- **18 GB VRAM** when fully quantized — fits consumer GPUs
- **Apache 2.0** license
- **Bi-directional attention** — ideal for code infill, editing, non-linear generation
- **Experimental status** — quality is lower than autoregressive Gemma 4, but unmatched for speed-critical workflows

## Files

| File | Purpose |
|------|---------|
| `setup_vllm.sh` | Install uv, venv, vllm from main branch (native DiffusionGemma support) |
| `download_model.sh` | Download model via `snapshot_download()` |
| `vllm-server.sh` | Start vLLM with all optimized flags |
| `vllm.service` | Systemd user service file |
| `test_vllm.sh` | API functional test |
| `test_big.py` | Long-context stress test |
| `test_tools.py` | Tool-calling test |

## Configuration (Environment Variables)

All settings in `vllm-server.sh` are overridable via env vars:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_MODEL_LEN` | `262144` | Max context length (max context supported by this model family) |
| `GPU_MEM_UTIL` | `0.55` | Current tuned Spark default (2 concurrent app case). Adjust per node and workload. |
| `ATTN_BACKEND` | `TRITON_ATTN` | Attention backend (required) |
| `VLLM_QUANTIZATION` | (unset) | Disabled/avoids fp4 path, BF16 path forced (`VLLM_QUANTIZATION=fp4` blocked) |
| `VLLM_DTYPE` | `bfloat16` | BF16 model math |
| `TOOL_CALL_PARSER` | <disabled> | Tool calling parser intentionally disabled |
| `REASONING_PARSER` | <disabled> | Reasoning parser intentionally disabled |
| `MAX_DENOISING_STEPS` | `48` | Diffusion denoising steps |
| `CANVAS_LENGTH` | `256` | Tokens per diffusion block |
| `MODEL_REPO` | `google/diffusiongemma-26B-A4B-it` | HF model repo ID |

## Memory Budget

The model is 48 GB in BF16. On DGX Spark (128 GB unified):

- **BF16 unquantized runtime** is used (`VLLM_DTYPE=bfloat16`, no `--quantization` flag passed)
- `GPU_MEM_UTIL=0.55` is the current low-footprint baseline for this node
- **FP4 is explicitly blocked** (`VLLM_QUANTIZATION=fp4` is disabled in runtime wrapper)

## Current Verified Runtime

- Port: **8000** (`vllm.service`)
- Model dtype: **bfloat16**
- Memory cap: **`GPU_MEM_UTIL=0.55`**
- Max denoising steps: **48**
- Canvas length: **256**
- Quantization: **not in use** (BF16-first)
- FP4 guard: runtime strips `fp4`/`nvfp4*` if set
- Service: `systemctl --user` unit enabled and running on `spark.lb.bitbull.ch`
- Validation: both `/v1/models` and `/v1/chat/completions` return `200` on port 8000

**Prerequisites (headless/SSH-only):**
```bash
sudo apt install dbus dbus-user-session
sudo loginctl enable-linger $USER
echo 'export DBUS_SESSION_BUS_ADDRESS="unix:path=/run/user/$(id -u)/bus"' >> ~/.bashrc
source ~/.bashrc
```

**Install:**
```bash
mkdir -p ~/.config/systemd/user/
cp vllm.service ~/.config/systemd/user/
chmod +x /srv/vllm2/vllm-server.sh
[ ! -d /srv/vllm2/.venv ] || sudo chown -R $USER:$USER /srv/vllm2/.venv
systemctl --user daemon-reload
systemctl --user enable --now vllm
```

## Hermes Configuration

```bash
hermes config set model.base_url http://spark.lb.bitbull.ch:8000/v1
hermes config set model.provider custom
hermes config set model.default google/diffusiongemma-26B-A4B-it
```

## Notes

- **vLLM main branch required** — DiffusionGemma support was added after v0.23.0
- **TRITON_ATTN required** — shares Gemma 4's 512-dim full-attention; FlashInfer/FlashAttn won't work
- **Experimental quality** — for production quality, use autoregressive Gemma 4
- **Diffusion vs autoregressive:** The model generates 256 tokens per forward pass via iterative denoising, not left-to-right token prediction
- **Multimodal:** Supports image+text input by default (vision encoder is always loaded)