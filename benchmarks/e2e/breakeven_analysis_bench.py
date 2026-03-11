#!/usr/bin/env python3
"""Break-even hit-rate analysis: at what hit rate does SemBlend become net-positive?

For each context length, measures three conditions:
  C = cold TTFT    (no SemBlend pipeline, baseline prefill)
  H = hit  TTFT    (SemBlend found a matching donor)
  M = miss TTFT    (SemBlend searched but found nothing)

From these, the break-even hit probability is:
  P_h* = overhead_miss / (overhead_miss + savings_hit)
       = (M - C) / ((M - C) + (C - H))

At any deployment hit rate P_h > P_h*, SemBlend delivers net latency savings.

Usage:
    python -m benchmarks.e2e.breakeven_analysis_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-samples 16 \
        --token-lengths 2048,4096,8192,16384
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Repeatable filler sentence to pad prompts to target token count.
# ~4 chars per token, each sentence ~120 chars => ~30 tokens.
_FILLER_SENTENCE = (
    "The research committee reviewed the quarterly findings and recommended "
    "further investigation into the observed anomalies. "
)

_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Document ID: {uid}\n\n"
    "{context}\n\n"
    "Summarize the key points above.<|im_end|>\n"
    "<|im_start|>assistant\n"
)


def _build_synthetic_context(target_chars: int) -> str:
    """Return repeating filler text of approximately *target_chars* characters."""
    repeats = max(1, target_chars // len(_FILLER_SENTENCE) + 1)
    return (_FILLER_SENTENCE * repeats)[:target_chars]


def _build_prompt(uid: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(uid=uid, context=context)


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Send a non-streaming completions request. Returns (latency_ms, ok)."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=300,
        )
        ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        return ms, True
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"    request failed ({elapsed:.0f}ms): {exc}")
        return 0.0, False


def _percentile(values: list[float], pct: int) -> float:
    """Return the *pct*-th percentile from a sorted copy of *values*."""
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
    return s[idx]


def run_condition(
    label: str,
    endpoint: str,
    model: str,
    prompts: list[str],
    max_tokens: int = 5,
) -> list[float]:
    """Run a batch of requests and return the list of successful latencies."""
    latencies: list[float] = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        t, ok = ttft_request(endpoint, model, prompt, max_tokens=max_tokens)
        tag = f"{t:.0f}ms" if ok else "FAIL"
        print(f"    [{i + 1}/{total}] {label} {tag}    ", end="\r")
        if ok:
            latencies.append(t)
    print()
    return latencies


def analyze_breakeven(
    cold: list[float], hit: list[float], miss: list[float]
) -> dict:
    """Compute break-even statistics from the three measurement vectors."""
    c_mean = statistics.mean(cold) if cold else 0.0
    h_mean = statistics.mean(hit) if hit else 0.0
    m_mean = statistics.mean(miss) if miss else 0.0

    c_p50 = _percentile(cold, 50) if cold else 0.0
    h_p50 = _percentile(hit, 50) if hit else 0.0
    m_p50 = _percentile(miss, 50) if miss else 0.0

    overhead_miss = max(0.0, m_mean - c_mean)
    savings_hit = max(0.0, c_mean - h_mean)

    denominator = overhead_miss + savings_hit
    breakeven_ph = overhead_miss / denominator if denominator > 0 else 1.0

    # Expected speedup at various hit rates
    speedup_at_ph: dict[str, float] = {}
    for ph in [0.25, 0.50, 0.75, 0.90, 1.00]:
        expected_ttft = ph * h_mean + (1.0 - ph) * m_mean
        spd = c_mean / expected_ttft if expected_ttft > 0 else 0.0
        speedup_at_ph[f"{ph:.2f}"] = round(spd, 3)

    return {
        "cold_mean_ms": round(c_mean, 1),
        "cold_p50_ms": round(c_p50, 1),
        "hit_mean_ms": round(h_mean, 1),
        "hit_p50_ms": round(h_p50, 1),
        "miss_mean_ms": round(m_mean, 1),
        "miss_p50_ms": round(m_p50, 1),
        "overhead_miss_ms": round(overhead_miss, 1),
        "savings_hit_ms": round(savings_hit, 1),
        "breakeven_hit_rate": round(breakeven_ph, 4),
        "speedup_at_hit_rate": speedup_at_ph,
        "n_cold": len(cold),
        "n_hit": len(hit),
        "n_miss": len(miss),
    }


_MISS_FILLER_SENTENCE = (
    "Quantum computing breakthroughs enable molecular simulation at "
    "unprecedented scales, transforming pharmaceutical drug discovery. "
)


def _build_miss_context(target_chars: int, seed: int) -> str:
    """Return unique miss-condition text that is topically different from donors."""
    import random
    rng = random.Random(seed)
    topics = [
        "Deep ocean exploration reveals bioluminescent organisms thriving "
        "near hydrothermal vents at extreme pressures and temperatures. ",
        "Ancient astronomical observatories demonstrate sophisticated "
        "mathematical understanding of celestial mechanics and seasonal cycles. ",
        "Sustainable urban agriculture integrates vertical farming technology "
        "with renewable energy systems to maximize food production density. ",
        "Advances in materials science yield self-healing polymers capable "
        "of restoring structural integrity after mechanical damage events. ",
        _MISS_FILLER_SENTENCE,
    ]
    parts = []
    while len("".join(parts)) < target_chars:
        parts.append(rng.choice(topics))
    return "".join(parts)[:target_chars]


def run_length(
    token_length: int,
    endpoint: str,
    model: str,
    n_samples: int,
) -> dict:
    """Run the full three-condition measurement for one context length."""
    target_chars = token_length * 4  # ~4 chars per token
    context_body = _build_synthetic_context(target_chars)

    # --- Generate unique prompts ---
    # Group A: used for cold measurement + donor registration + hit measurement
    uids_a = [f"brk-a-{uuid.uuid4().hex[:12]}" for _ in range(n_samples)]
    prompts_a = [_build_prompt(uid, context_body) for uid in uids_a]

    # Group B: TOPICALLY DIFFERENT content — SemBlend should NOT find a donor
    # because the document body is entirely different (not just a UUID change)
    prompts_b = []
    for i in range(n_samples):
        uid = f"brk-b-{uuid.uuid4().hex[:12]}"
        miss_body = _build_miss_context(target_chars, seed=i)
        prompts_b.append(_build_prompt(uid, miss_body))

    print(f"  Prompt length: ~{len(prompts_a[0])} chars "
          f"(~{len(prompts_a[0]) // 4} tokens)")

    # Phase 1: Cold TTFT — first time seeing these prompts
    print(f"  Phase 1: Cold TTFT (n={n_samples})")
    cold_latencies = run_condition("cold", endpoint, model, prompts_a)

    # Phase 2: Register donors — re-send group A prompts with longer generation
    # to ensure the KV cache is populated in LMCache
    print(f"  Phase 2: Register donors (n={n_samples})")
    run_condition("donor", endpoint, model, prompts_a, max_tokens=50)

    # Phase 3: Hit TTFT — re-send group A prompts (should match donors)
    print(f"  Phase 3: Hit TTFT (n={n_samples})")
    hit_latencies = run_condition("hit", endpoint, model, prompts_a)

    # Phase 4: Miss TTFT — send group B prompts (no matching donor)
    print(f"  Phase 4: Miss TTFT (n={n_samples})")
    miss_latencies = run_condition("miss", endpoint, model, prompts_b)

    result = analyze_breakeven(cold_latencies, hit_latencies, miss_latencies)
    result["token_length"] = token_length
    result["_raw_cold"] = cold_latencies
    result["_raw_hit"] = hit_latencies
    result["_raw_miss"] = miss_latencies
    return result


def print_table(results: list[dict]) -> None:
    """Print a formatted summary table."""
    header = (
        f"{'Tokens':>8} {'Cold':>10} {'Hit':>10} {'Miss':>10} "
        f"{'Overhead':>10} {'Savings':>10} {'BrkEven':>8} "
        f"{'Spd@50%':>8} {'Spd@75%':>8} {'Spd@90%':>8}"
    )
    print()
    print("=" * len(header))
    print("Break-Even Hit-Rate Analysis")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        sp = r.get("speedup_at_hit_rate", {})
        print(
            f"{r['token_length']:>8} "
            f"{r['cold_mean_ms']:>9.0f}ms "
            f"{r['hit_mean_ms']:>9.0f}ms "
            f"{r['miss_mean_ms']:>9.0f}ms "
            f"{r['overhead_miss_ms']:>9.0f}ms "
            f"{r['savings_hit_ms']:>9.0f}ms "
            f"{r['breakeven_hit_rate']:>7.1%} "
            f"{sp.get('0.50', 0):>7.2f}x "
            f"{sp.get('0.75', 0):>7.2f}x "
            f"{sp.get('0.90', 0):>7.2f}x"
        )

    print("-" * len(header))
    print()
    print("Interpretation:")
    print("  BrkEven = minimum hit rate for SemBlend to be net-positive vs cold prefill")
    print("  Overhead = extra latency on misses (M - C)")
    print("  Savings  = latency saved on hits (C - H)")
    print("  P_h* = Overhead / (Overhead + Savings)")
    print("  Spd@X%  = expected speedup at X% hit rate")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break-even hit-rate analysis for SemBlend"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of prompts per condition per length")
    parser.add_argument("--token-lengths", default="2048,4096,8192,16384",
                        help="Comma-separated context lengths in tokens")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    token_lengths = [int(x.strip()) for x in args.token_lengths.split(",")]

    print()
    print("Break-Even Hit-Rate Analysis")
    print(f"  endpoint={args.endpoint}")
    print(f"  model={args.model}")
    print(f"  n_samples={args.n_samples}")
    print(f"  token_lengths={token_lengths}")
    print()

    # Health check
    try:
        requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Endpoint health check failed: {exc}")
        sys.exit(1)

    results: list[dict] = []

    for tl in token_lengths:
        print(f"\n--- Context length: {tl} tokens ---")
        result = run_length(tl, args.endpoint, args.model, args.n_samples)
        results.append(result)

        # Print intermediate result
        be = result["breakeven_hit_rate"]
        print(f"  => Cold={result['cold_mean_ms']:.0f}ms, "
              f"Hit={result['hit_mean_ms']:.0f}ms, "
              f"Miss={result['miss_mean_ms']:.0f}ms")
        print(f"  => Break-even P_h* = {be:.1%}")

    print_table(results)

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import bootstrap_mean, bootstrap_speedup

    print("Bootstrap 95% Confidence Intervals")
    print("=" * 60)
    for r in results:
        tl = r["token_length"]
        cold_arr = np.array(r["_raw_cold"])
        hit_arr = np.array(r["_raw_hit"])
        miss_arr = np.array(r["_raw_miss"])
        print(f"\n  Context length: {tl} tokens")
        print(f"    Cold TTFT mean:  {bootstrap_mean(cold_arr)}")
        print(f"    Hit TTFT mean:   {bootstrap_mean(hit_arr)}")
        print(f"    Miss TTFT mean:  {bootstrap_mean(miss_arr)}")
        if len(cold_arr) > 0 and len(hit_arr) > 0:
            print(f"    Hit speedup:     {bootstrap_speedup(hit_arr, cold_arr)}")
        print(f"    Break-even P_h*: {r['breakeven_hit_rate']:.4f}")
    print()

    if args.output:
        # Strip internal raw arrays before serializing
        serializable_results = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ]
        output_data = {
            "endpoint": args.endpoint,
            "model": args.model,
            "n_samples": args.n_samples,
            "results": serializable_results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
