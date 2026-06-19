#!/usr/bin/env python3
"""Speed benchmark for DiffusionGemma 26B A4B-it on DGX Spark."""

import json
import os
import sys
import time
from statistics import mean, median

import requests

HOST = os.environ.get("HOST", "spark.lb.bitbull.ch")
PORT = os.environ.get("PORT", "8000")
BASE_URL = f"http://{HOST}:{PORT}/v1"
MODEL = "google/diffusiongemma-26B-A4B-it"


def chat(messages, max_tokens=128, temperature=0.0):
    """Send a chat completion request."""
    t0 = time.time()
    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=300,
    )
    elapsed = time.time() - t0
    resp.raise_for_status()
    data = resp.json()
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", 0)
    prompt_tokens = usage.get("prompt_tokens", 0)
    content = data["choices"][0]["message"]["content"]
    return {
        "elapsed": elapsed,
        "completion_tokens": completion_tokens,
        "prompt_tokens": prompt_tokens,
        "tokens_per_sec": completion_tokens / elapsed if elapsed > 0 else 0,
        "content": content.strip(),
        "usage": usage,
    }


def run_benchmark(name, messages, max_tokens=128, runs=3):
    """Run a benchmark test multiple times."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    results = []
    for i in range(runs):
        try:
            r = chat(messages, max_tokens=max_tokens)
            results.append(r)
            print(f"  Run {i+1}: {r['completion_tokens']} tokens in {r['elapsed']:.2f}s "
                  f"= {r['tokens_per_sec']:.2f} tok/s (prompt: {r['prompt_tokens']} tok)")
        except Exception as e:
            print(f"  Run {i+1}: FAILED - {e}")

    if results:
        tps = [r["tokens_per_sec"] for r in results]
        print(f"  --- {name} Summary ---")
        print(f"  Mean:   {mean(tps):.2f} tok/s")
        print(f"  Median: {median(tps):.2f} tok/s")
        print(f"  Min:    {min(tps):.2f} tok/s")
        print(f"  Max:    {max(tps):.2f} tok/s")

    return results


def main():
    print(f"=== DiffusionGemma 26B A4B-it Speed Benchmark ===")
    print(f"Server: {BASE_URL}")
    print(f"Model:  {MODEL}")
    print(f"Time:   {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"{'='*60}")

    all_results = {}

    # Test 1: Short generation (32 tokens)
    all_results["short"] = run_benchmark(
        "Short generation (max_tokens=32)",
        [{"role": "user", "content": "What is the capital of Switzerland? Answer in one word."}],
        max_tokens=32,
    )

    # Test 2: Medium generation (128 tokens)
    all_results["medium"] = run_benchmark(
        "Medium generation (max_tokens=128)",
        [{"role": "user", "content": "Write a short paragraph about artificial intelligence."}],
        max_tokens=128,
    )

    # Test 3: Longer generation (256 tokens)
    all_results["long"] = run_benchmark(
        "Longer generation (max_tokens=256)",
        [{"role": "user", "content": "Explain the concept of diffusion models in machine learning in detail."}],
        max_tokens=256,
    )

    # Test 4: With reasoning
    all_results["reasoning"] = run_benchmark(
        "Reasoning (max_tokens=256)",
        [{"role": "user", "content": "What is 47 * 89? Show your reasoning step by step."}],
        max_tokens=256,
    )

    # Summary
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY")
    print(f"{'='*60}")

    all_tps = []
    for name, results in all_results.items():
        if results:
            tps = [r["tokens_per_sec"] for r in results]
            mean_tps = mean(tps)
            all_tps.extend(tps)
            print(f"  {name:20s}: {mean_tps:8.2f} tok/s (avg of {len(results)} runs)")

    if all_tps:
        print(f"  {'─'*40}")
        print(f"  {'Grand average':20s}: {mean(all_tps):8.2f} tok/s")
        print(f"  {'Grand median':20s}: {median(all_tps):8.2f} tok/s")

    # Output JSON for README insertion
    summary = {}
    for name, results in all_results.items():
        if results:
            tps = [r["tokens_per_sec"] for r in results]
            summary[name] = {
                "mean_tok_s": round(mean(tps), 2),
                "median_tok_s": round(median(tps), 2),
                "min_tok_s": round(min(tps), 2),
                "max_tok_s": round(max(tps), 2),
                "runs": len(results),
            }
    if all_tps:
        summary["overall"] = {
            "mean_tok_s": round(mean(all_tps), 2),
            "median_tok_s": round(median(all_tps), 2),
            "runs": len(all_tps),
        }

    print(f"\nJSON summary for README:")
    print(json.dumps(summary, indent=2))

    return summary


if __name__ == "__main__":
    summary = main()