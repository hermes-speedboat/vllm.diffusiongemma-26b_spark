#!/usr/bin/env bash
# vllm-server.sh — vLLM Server for DiffusionGemma 26B A4B-it
# DGX Spark (GB10, 128 GB Unified Memory, Grace Blackwell ARM64)
#
# Requires vLLM from main branch (not a release) — DiffusionGemma support
# was added to vLLM main on 2026-06-12, after v0.23.0.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_REPO="${MODEL_REPO:-google/diffusiongemma-26B-A4B-it}"
MODEL_NAME="${MODEL_REPO##*/}"
MODEL_PATH="${MODEL_DIR}/${MODEL_NAME}"

# ----------------------------------------------------------------------
# Configuration (all overridable via env vars)
# ----------------------------------------------------------------------
PORT="${PORT:-8001}"
HOST="${HOST:-0.0.0.0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"

# Memory cap: 30 GB max. With 128 GB unified memory, GPU_MEM_UTIL=0.24
# gives ~30.7 GB. FP8 quantization halves model size from 48 GB to ~24 GB,
# leaving ~6 GB for KV cache and overhead.
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.24}"

# Weight quantization: FP8 halves memory bandwidth and model size.
# Critical for fitting the 48 GB BF16 model in a 30 GB budget.
QUANTIZATION="${QUANTIZATION:-fp8}"

# Attention backend: TRITON_ATTN is required for DiffusionGemma's 512-dim
# full-attention layers (shared with Gemma 4 architecture).
# FlashInfer and FlashAttn reject this config.
ATTN_BACKEND="${ATTN_BACKEND:-TRITON_ATTN}"

# Tool-call & reasoning (same format as Gemma 4)
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-gemma4}"
REASONING_PARSER="${REASONING_PARSER:-gemma4}"

# Diffusion-specific settings
MAX_DENOISING_STEPS="${MAX_DENOISING_STEPS:-48}"
CANVAS_LENGTH="${CANVAS_LENGTH:-256}"

# ----------------------------------------------------------------------
# Check if model exists
# ----------------------------------------------------------------------
if [ ! -d "${MODEL_PATH}" ]; then
  echo "Model not found: ${MODEL_PATH}"
  echo "Run download_model.sh first:"
  echo "  bash ${SCRIPT_DIR}/download_model.sh"
  exit 1
fi

# ----------------------------------------------------------------------
# GPU info
# ----------------------------------------------------------------------
echo "=== GPU Info ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "(no nvidia-smi)"
echo ""

# ----------------------------------------------------------------------
# Start server
# ----------------------------------------------------------------------
echo "=== Starting vLLM Server — DiffusionGemma 26B A4B ==="
echo " Model: ${MODEL_PATH}"
echo " Port: ${HOST}:${PORT}"
echo " Context: ${MAX_MODEL_LEN}"
echo " GPU Mem: ${GPU_MEM_UTIL} (~$((GPU_MEM_UTIL * 128)) GB on 128 GB total)"
echo " Quantization: ${QUANTIZATION}"
echo " Attention: ${ATTN_BACKEND}"
echo " Tool calls: ${TOOL_CALL_PARSER}"
echo " Reasoning: ${REASONING_PARSER}"
echo " Denoising steps: ${MAX_DENOISING_STEPS}"
echo " Canvas length: ${CANVAS_LENGTH}"
echo ""

ARGS=(
  "${MODEL_PATH}"
  "--port" "${PORT}"
  "--host" "${HOST}"
  "--max-model-len" "${MAX_MODEL_LEN}"
  "--gpu-memory-utilization" "${GPU_MEM_UTIL}"
  "--trust-remote-code"
  "--enforce-eager"
  "--attention-config" "{\"backend\": \"${ATTN_BACKEND}\"}"
  "--quantization" "${QUANTIZATION}"
  "--enable-auto-tool-choice"
  "--tool-call-parser" "${TOOL_CALL_PARSER}"
  "--reasoning-parser" "${REASONING_PARSER}"
  "--served-model-name" "google/diffusiongemma-26B-A4B-it"
  "--max-denoising-steps" "${MAX_DENOISING_STEPS}"
  "--canvas-length" "${CANVAS_LENGTH}"
)

echo "> vllm serve ${ARGS[*]}"
vllm serve "${ARGS[@]}"

echo ""
echo "Server stopped."