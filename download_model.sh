#!/usr/bin/env bash
# download_model.sh — Download DiffusionGemma 26B A4B-it via huggingface_hub
# Run from the same directory as setup_vllm.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/.venv/bin/activate"

MODEL_DIR="${SCRIPT_DIR}/models"
MODEL_REPO="${MODEL_REPO:-google/diffusiongemma-26B-A4B-it}"

mkdir -p "${MODEL_DIR}"

echo "=== Model Download ==="
echo " Repo: ${MODEL_REPO}"
echo " Format: BF16 safetensors (25.2B params, ~48 GB)"
echo " Target: ${MODEL_DIR}/"
echo " Size: ~48 GB"

MODEL_NAME="${MODEL_REPO##*/}"
if [ -d "${MODEL_DIR}/${MODEL_NAME}" ]; then
  echo "Model already exists: ${MODEL_DIR}/${MODEL_NAME}"
  du -sh "${MODEL_DIR}/${MODEL_NAME}"
  echo "To re-download, remove and retry:"
  echo " rm -rf ${MODEL_DIR:?}/${MODEL_NAME}"
  echo " bash $0"
else
  echo "Downloading ${MODEL_REPO} (~48 GB) ..."
  python3 -c "
from huggingface_hub import snapshot_download
import os

model_path = snapshot_download(
    repo_id='${MODEL_REPO}',
    local_dir=os.path.join('${MODEL_DIR}', '${MODEL_NAME}'),
    ignore_patterns=['*.md', '*.h5', '*.ot', '*.msgpack'],
)
print(f'Downloaded to: {model_path}')
"
  echo ""
  echo "Download complete:"
  du -sh "${MODEL_DIR}/${MODEL_NAME}"
fi

echo ""
echo "=== Contents of ${MODEL_DIR}/${MODEL_NAME} ==="
ls -lh "${MODEL_DIR}/${MODEL_NAME}" | head -20

echo ""
echo "Done. Start the server with:"
echo " bash ${SCRIPT_DIR}/vllm-server.sh"