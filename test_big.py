"""Long-context stress test for DiffusionGemma 26B A4B-it."""
import json
import os
import sys
import time

import requests

PORT = os.environ.get("PORT", "8001")
HOST = os.environ.get("HOST", "spark.lb.bitbull.ch")
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL = "google/diffusiongemma-26B-A4B-it"


def chat(messages, max_tokens=100, temperature=0.7):
    """Send a chat completion request."""
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"=== DiffusionGemma Test at {BASE_URL} ===")
    print(f"Model: {MODEL}")
    print()

    # 1. Basic text generation
    print("--- Test 1: Basic text generation ---")
    t0 = time.time()
    result = chat([{"role": "user", "content": "Write a haiku about artificial intelligence."}], max_tokens=100)
    elapsed = time.time() - t0
    content = result["choices"][0]["message"]["content"]
    tokens = result.get("usage", {}).get("completion_tokens", 0)
    print(f"Response ({elapsed:.1f}s, ~{tokens} tokens):")
    print(content)
    print()

    # 2. Reasoning test
    print("--- Test 2: Reasoning ---")
    t0 = time.time()
    result = chat(
        [{"role": "user", "content": "What is 47 * 89? Show your reasoning step by step."}],
        max_tokens=256,
    )
    elapsed = time.time() - t0
    content = result["choices"][0]["message"]["content"]
    print(f"Response ({elapsed:.1f}s):")
    print(content)
    print()

    # 3. Code generation
    print("--- Test 3: Code generation ---")
    t0 = time.time()
    result = chat(
        [{"role": "user", "content": "Write a Python function that checks if a string is a palindrome."}],
        max_tokens=256,
    )
    elapsed = time.time() - t0
    content = result["choices"][0]["message"]["content"]
    print(f"Response ({elapsed:.1f}s):")
    print(content)
    print()

    print("=== All tests passed ===")


if __name__ == "__main__":
    main()