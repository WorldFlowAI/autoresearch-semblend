#!/usr/bin/env python3
"""EOS-collapse rate benchmark.

Measures the fraction of SemBlend cache-hit responses that generate
only 1 token before EOS — the "EOS-collapse" failure mode where the
model emits EOS immediately after KV injection.

Run this BEFORE and AFTER the min_fresh_tokens=256 fix to measure the
improvement.

Methodology:
  1. Register a set of seed donors (exact-match scenario, XSum 8K).
  2. Send each seed as a query (triggering a SemBlend cache hit).
  3. Count responses with n_tokens == 1 (EOS-collapse).
  4. Report collapse_rate = collapses / total_hits.

Expected results:
  Before fix (max_usable = prompt_len - 1):  ~25% collapse rate
  After fix  (max_usable = prompt_len - 256): ~0% collapse rate

Usage:
    python -m benchmarks.e2e.eos_collapse_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --n-seeds 20 \\
        --min-tokens 0
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import requests

# Dataset helpers
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
try:
    from benchmarks.e2e.dataset_loader import load_dataset_texts
except ImportError:
    load_dataset_texts = None

XSUM_SEED = 42
TARGET_TOKENS = 8192  # approximate 8K prompt via char count (~4 chars/token)
TARGET_CHARS = TARGET_TOKENS * 4


def build_prompt(text: str, instruction: str = "Summarize the following article:\n\n") -> str:
    truncated = text[:TARGET_CHARS]
    return f"{instruction}{truncated}\n\nSummary:"


def register_donor(endpoint: str, model: str, prompt: str) -> float:
    """Register a donor and return latency ms."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 50,
              "temperature": 0.0, "stream": False},
        timeout=300,
    )
    resp.raise_for_status()
    return (time.monotonic() - t0) * 1000


def query_semblend(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 200,
    min_tokens: int = 0,
) -> dict:
    """Query SemBlend and return response info."""
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens

    t0 = time.monotonic()
    resp = requests.post(f"{endpoint}/v1/completions", json=payload, timeout=300)
    latency_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()

    text = data["choices"][0].get("text", "")
    # Count non-whitespace tokens as a rough measure
    n_words = len(text.split())
    finish_reason = data["choices"][0].get("finish_reason", "?")
    usage = data.get("usage", {})
    completion_tokens = usage.get("completion_tokens", len(text.split()))

    return {
        "text": text[:200],
        "n_completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "latency_ms": latency_ms,
        "collapsed": completion_tokens <= 2,  # ≤2 tokens = EOS-collapse
    }


def run_benchmark(
    endpoint: str,
    model: str,
    n_seeds: int,
    min_tokens: int,
) -> dict:
    # Warm up / health check
    resp = requests.get(f"{endpoint}/health", timeout=10)
    resp.raise_for_status()

    # Load XSum articles
    if load_dataset_texts:
        try:
            texts = load_dataset_texts("xsum", split="validation", n=n_seeds * 2)
        except Exception:
            texts = None
    else:
        texts = None

    if not texts:
        # Fallback: synthetic long texts
        texts = [
            f"Article {i}: " + ("The quick brown fox jumps over the lazy dog. " * 200)
            for i in range(n_seeds * 2)
        ]

    prompts = [build_prompt(t) for t in texts[:n_seeds * 2]]

    print(f"\nEOS-Collapse Benchmark")
    print(f"  endpoint={endpoint}, model={model}")
    print(f"  n_seeds={n_seeds}, min_tokens={min_tokens}")
    print(f"  prompt_length: ~{len(prompts[0])} chars (~{len(prompts[0])//4} tokens)")
    print()

    # Phase 1: Register donors (restart vLLM state implicitly by using fresh seeds)
    print(f"Phase 1: Registering {n_seeds} seed donors...")
    donor_prompts = prompts[:n_seeds]
    for i, p in enumerate(donor_prompts):
        latency = register_donor(endpoint, model, p)
        print(f"  [{i+1}/{n_seeds}] donor registered ({latency:.0f}ms)")

    # Phase 2: Query with same prompts (should trigger SemBlend hits)
    print(f"\nPhase 2: Querying {n_seeds} prompts (expect cache hits)...")
    results = []
    for i, p in enumerate(donor_prompts):
        r = query_semblend(endpoint, model, p, max_tokens=200, min_tokens=min_tokens)
        results.append(r)
        status = "COLLAPSE" if r["collapsed"] else "OK"
        print(
            f"  [{i+1}/{n_seeds}] {status} | "
            f"tokens={r['n_completion_tokens']} | "
            f"finish={r['finish_reason']} | "
            f"{r['latency_ms']:.0f}ms | "
            f"text={repr(r['text'][:60])}"
        )

    # Compute stats
    n_collapsed = sum(1 for r in results if r["collapsed"])
    collapse_rate = n_collapsed / len(results) if results else 0.0
    mean_tokens = sum(r["n_completion_tokens"] for r in results) / max(len(results), 1)

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"  Total queries:   {len(results)}")
    print(f"  EOS-collapses:   {n_collapsed} ({collapse_rate*100:.1f}%)")
    print(f"  Mean tokens out: {mean_tokens:.1f}")
    print(f"{'='*60}")

    return {
        "n_total": len(results),
        "n_collapsed": n_collapsed,
        "collapse_rate": collapse_rate,
        "mean_completion_tokens": mean_tokens,
        "min_tokens_param": min_tokens,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--min-tokens", type=int, default=0,
                        help="min_tokens param sent to vLLM (0=disabled)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result = run_benchmark(args.endpoint, args.model, args.n_seeds, args.min_tokens)

    if args.output:
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
