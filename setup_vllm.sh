#!/usr/bin/env bash
# setup_vllm.sh — vLLM Setup for DiffusionGemma 26B A4B on DGX Spark
# Run as normal user (e.g. vllm2). No root required.
# Installs vLLM from main branch — DiffusionGemma support requires code after v0.23.0
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/.venv"
MODEL_DIR="${SCRIPT_DIR}/models"

echo "=== vLLM Setup for DGX Spark — DiffusionGemma 26B A4B ==="
echo "Target: ${SCRIPT_DIR}"

# ---------------------------------------------------------------------------
# 1) Install uv (if not present)
# ---------------------------------------------------------------------------
if ! command -v uv &>/dev/null; then
  echo "[1/6] Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
  # shellcheck disable=SC1091
  source "$HOME/.bashrc" 2>/dev/null || true
else
  echo "[1/6] uv already installed: $(uv --version)"
fi

# ---------------------------------------------------------------------------
# 2) Create Python venv with uv
# ---------------------------------------------------------------------------
echo "[2/6] Creating Python venv with uv ..."
uv venv "${VENV_DIR}" --python 3.12

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# ---------------------------------------------------------------------------
# 3) Install build dependencies
# ---------------------------------------------------------------------------
echo "[3/6] Installing build dependencies ..."
uv pip install --upgrade pip
uv pip install cmake ninja packaging setuptools wheel

# ---------------------------------------------------------------------------
# 4) Install huggingface-hub
# ---------------------------------------------------------------------------
echo "[4/6] Installing huggingface-hub ..."
uv pip install huggingface-hub

# ---------------------------------------------------------------------------
# 5) Install vLLM from main branch (native DiffusionGemma support)
#    DGX Spark (GB10, Blackwell) uses CUDA 13.0
#    Installing from source ensures DiffusionGemmaForBlockDiffusion support.
# ---------------------------------------------------------------------------
echo "[5/6] Building and installing vLLM from main branch ..."
echo "  This may take 15-30 minutes on first install."

if [ -d "${SCRIPT_DIR}/vllm_src" ]; then
  echo "  Updating existing vllm_src ..."
  cd "${SCRIPT_DIR}/vllm_src"
  git pull origin main
  pip install -e .
  cd "${SCRIPT_DIR}"
else
  echo "  Cloning vllm main branch ..."
  git clone https://github.com/vllm-project/vllm.git "${SCRIPT_DIR}/vllm_src"
  cd "${SCRIPT_DIR}/vllm_src"
  pip install -e .
  cd "${SCRIPT_DIR}"
fi

# ---------------------------------------------------------------------------
# 6) Create model directory
# ---------------------------------------------------------------------------
echo "[6/6] Creating model directory: ${MODEL_DIR}"
mkdir -p "${MODEL_DIR}"

# ---------------------------------------------------------------------------
# Done
# ---------------------------------------------------------------------------
echo ""
echo "=== Setup complete ==="
echo ""
echo " Venv: source ${VENV_DIR}/bin/activate"
echo " Models: ${MODEL_DIR}/"
echo ""
echo "Next: download the model:"
echo " bash ${SCRIPT_DIR}/download_model.sh"
echo ""
echo "Then start the server:"
echo " bash ${SCRIPT_DIR}/vllm-server.sh"