#!/usr/bin/env bash
# test_vllm.sh — API functional test for DiffusionGemma 26B A4B-it
set -euo pipefail

PORT="${PORT:-8001}"
HOST="${HOST:-127.0.0.1}"
BASE_URL="http://${HOST}:${PORT}/v1"

echo "=== Testing DiffusionGemma API at ${BASE_URL} ==="
echo ""

# 1) Check models endpoint
echo "--- Models ---"
curl -s "${BASE_URL}/models" | python3 -m json.tool | head -20
echo ""

# 2) Simple chat completion
echo "--- Chat Completion ---"
curl -s "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "google/diffusiongemma-26B-A4B-it",
    "messages": [
      {"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}
    ],
    "max_tokens": 50,
    "temperature": 0.7
  }' | python3 -m json.tool
echo ""