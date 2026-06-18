# vLLM DGX Spark — DiffusionGemma 26B A4B-it

**Source:** [hermes-speedboat/vllm.diffusiongemma-26b_spark](https://github.com/hermes-speedboat/vllm.diffusiongemma-26b_spark)

Optimized vLLM inference setup for **DGX Spark (GB10, Grace Blackwell ARM64)** with 128 GB unified memory, running Google's experimental **DiffusionGemma 26B A4B** model.

## Quickstart

```bash
cd /srv/vllm2
git clone https://github.com/hermes-speedboat/vllm.diffusiongemma-26b_spark.git .
bash setup_vllm.sh        # uv + venv + vllm from main (diffusion gemma support)
bash download_model.sh     # ~48 GB BF16 model via snapshot_download()
bash vllm-server.sh        # start server on port 8001
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
| `PORT` | `8001` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `MAX_MODEL_LEN` | `8192` | Max context length (increase carefully) |
| `GPU_MEM_UTIL` | `0.24` | GPU memory utilization (30 GB cap) |
| `ATTN_BACKEND` | `TRITON_ATTN` | Attention backend (required) |
| `QUANTIZATION` | `fp8` | Weight quantization (FP8 halves memory) |
| `TOOL_CALL_PARSER` | `gemma4` | Tool call parser |
| `REASONING_PARSER` | `gemma4` | Reasoning parser for thinking mode |
| `MAX_DENOISING_STEPS` | `48` | Diffusion denoising steps |
| `CANVAS_LENGTH` | `256` | Tokens per diffusion block |
| `MODEL_REPO` | `google/diffusiongemma-26B-A4B-it` | HF model repo ID |

## Memory Budget

The model is 48 GB in BF16. On DGX Spark (128 GB unified):

- **FP8 quantization** reduces weights to ~24 GB
- `GPU_MEM_UTIL=0.24` caps total GPU allocation at ~30 GB
- Remaining ~6 GB for KV cache and overhead at default context
- Increase `GPU_MEM_UTIL` or disable `--quantization` **only** if you have headroom

## Systemd Autostart

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
hermes config set model.base_url http://spark.lb.bitbull.ch:8001/v1
hermes config set model.provider custom
hermes config set model.default google/diffusiongemma-26B-A4B-it
```

## Notes

- **vLLM main branch required** — DiffusionGemma support was added after v0.23.0
- **TRITON_ATTN required** — shares Gemma 4's 512-dim full-attention; FlashInfer/FlashAttn won't work
- **Experimental quality** — for production quality, use autoregressive Gemma 4
- **Diffusion vs autoregressive:** The model generates 256 tokens per forward pass via iterative denoising, not left-to-right token prediction
- **Multimodal:** Supports image+text input by default (vision encoder is always loaded)