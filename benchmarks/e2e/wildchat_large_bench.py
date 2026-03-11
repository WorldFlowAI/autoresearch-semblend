#!/usr/bin/env python3
"""WildChat-1M large-scale SemBlend benchmark.

Runs SemBlend E2E on a large sample of real user conversations from
WildChat-1M, stratified by similarity bucket. Previous WildChat validation
used only n=30 pairs; this benchmark uses n>=500 for statistical rigor.

Methodology:
  1. Load WildChat similarity analysis results (pre-computed by
     wildchat_similarity_analysis.py).
  2. Sample pairs stratified by similarity bucket:
     [0.50-0.60), [0.60-0.70), [0.70-0.80), [0.80-0.90), [0.90-1.00]
  3. For each pair:
     a. Send prompt A (donor) → register.
     b. Send prompt B (target) → measure TTFT.
     c. Measure cold baseline for prompt B.
  4. Report per-bucket: hit rate, TTFT speedup, sample size.

Usage:
    python -m benchmarks.e2e.wildchat_large_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --similarity-file results/wildchat_similarity.json \
        --n-per-bucket 100 \
        --min-chars 6000 \
        --output results/wildchat_large.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

SIMILARITY_BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01),
]

PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{content}<|im_end|>"
)


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Measure request time. Returns (ms, ok)."""
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
    except Exception:
        return 0.0, False


def load_wildchat_pairs(
    similarity_file: str, min_chars: int = 6000
) -> list[dict]:
    """Load pre-computed WildChat similarity pairs."""
    path = Path(similarity_file)
    if not path.exists():
        print(f"ERROR: {similarity_file} not found.")
        print("Run wildchat_similarity_analysis.py first.")
        sys.exit(1)

    data = json.loads(path.read_text())

    # Extract pairs with sufficient length
    pairs = []
    for pair in data.get("pairs", []):
        sim = pair.get("similarity", 0)
        text_a = pair.get("text_a", "")
        text_b = pair.get("text_b", "")
        if len(text_a) >= min_chars and len(text_b) >= min_chars:
            pairs.append({
                "similarity": sim,
                "text_a": text_a,
                "text_b": text_b,
                "user_id": pair.get("user_id", ""),
            })

    return pairs


def generate_synthetic_pairs(
    n: int, min_chars: int = 6000
) -> list[dict]:
    """Generate synthetic similar pairs when WildChat data unavailable."""
    pairs = []
    rng = random.Random(42)
    base_texts = [
        "Artificial intelligence and machine learning are transforming "
        "industries across the globe. From healthcare to finance, "
        "autonomous systems are making decisions that previously required "
        "human expertise. The rapid advancement of large language models "
        "has enabled new applications in natural language processing, "
        "code generation, and creative writing. ",
        "Climate change represents one of the most significant challenges "
        "facing humanity in the 21st century. Rising global temperatures "
        "are causing widespread environmental disruption, including more "
        "frequent extreme weather events, rising sea levels, and shifts "
        "in ecosystems. International cooperation through agreements like "
        "the Paris Accord aims to limit warming. ",
        "The global economy is undergoing significant transformation driven "
        "by technological innovation and shifting geopolitical dynamics. "
        "Supply chain resilience, digital currencies, and the transition "
        "to renewable energy are reshaping markets. Trade relationships "
        "between major economies continue to evolve. ",
    ]

    for i in range(n):
        base = base_texts[i % len(base_texts)]
        # Pad to min_chars
        text_a = (base * (min_chars // len(base) + 2))[:min_chars]
        # Create variant: swap some sentences, change a few words
        words = text_a.split()
        n_swap = max(1, len(words) // 20)  # swap ~5% of words
        for _ in range(n_swap):
            idx = rng.randint(0, len(words) - 1)
            synonyms = ["significant", "important", "notable", "major",
                        "substantial", "considerable", "remarkable"]
            words[idx] = rng.choice(synonyms)
        text_b = " ".join(words)

        sim = 0.50 + rng.random() * 0.50  # random sim in [0.5, 1.0)
        pairs.append({
            "similarity": sim,
            "text_a": text_a,
            "text_b": text_b,
            "user_id": f"synthetic_{i}",
        })

    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--similarity-file",
        default="results/wildchat_similarity.json",
    )
    parser.add_argument("--n-per-bucket", type=int, default=200)
    parser.add_argument("--min-chars", type=int, default=6000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()

    total_n = args.n_per_bucket * len(SIMILARITY_BUCKETS)
    print(f"\nWildChat Large-Scale Benchmark")
    print(f"  endpoint={args.endpoint}, model={args.model}")
    print(f"  n_per_bucket={args.n_per_bucket}, min_chars={args.min_chars}")
    print(f"  total target: {total_n} pairs")
    print()

    # Load or generate pairs
    sim_path = Path(args.similarity_file)
    if sim_path.exists():
        all_pairs = load_wildchat_pairs(args.similarity_file, args.min_chars)
        print(f"  Loaded {len(all_pairs)} WildChat pairs (>={args.min_chars} chars)")
    else:
        print(f"  {args.similarity_file} not found, using synthetic pairs")
        all_pairs = generate_synthetic_pairs(total_n * 2, args.min_chars)

    # Stratify by bucket
    buckets: dict[str, list[dict]] = {}
    for lo, hi in SIMILARITY_BUCKETS:
        key = f"{lo:.2f}-{hi:.2f}"
        bucket_pairs = [
            p for p in all_pairs if lo <= p["similarity"] < hi
        ]
        random.Random(42).shuffle(bucket_pairs)
        buckets[key] = bucket_pairs[: args.n_per_bucket]
        print(f"  Bucket [{key}): {len(buckets[key])} pairs")

    # Phase 1: Cold baselines (sample from first bucket)
    print(f"\nPhase 1: Cold baselines")
    cold_sample = []
    for bucket_pairs in buckets.values():
        cold_sample.extend(bucket_pairs[:4])
    cold_ttfts: list[float] = []
    for i, pair in enumerate(cold_sample[:16]):
        prompt = PROMPT_TEMPLATE.format(content=pair["text_b"][:args.min_chars])
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"  [{i+1}/16] cold {t:.0f}ms", end="\r")

    if not cold_ttfts:
        print("\nNo cold measurements — aborting")
        return

    cold_mean = sum(cold_ttfts) / len(cold_ttfts)
    cold_p50 = sorted(cold_ttfts)[len(cold_ttfts) // 2]
    print(f"\n  Cold: n={len(cold_ttfts)}, mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Phase 2: Per-bucket benchmark
    print(f"\nPhase 2: Per-bucket SemBlend benchmark")
    print(f"  {'Bucket':<14} {'N':>4} {'Hits':>5} {'Hit%':>6} "
          f"{'P50':>8} {'Speedup':>8} {'Hit Spd':>8}")
    print("-" * 65)

    results: dict[str, dict] = {}

    for bucket_key, bucket_pairs in buckets.items():
        if not bucket_pairs:
            continue

        bucket_ttfts: list[float] = []
        bucket_hits = 0

        for pair in bucket_pairs:
            # Register donor (prompt A)
            donor_prompt = PROMPT_TEMPLATE.format(
                content=pair["text_a"][:args.min_chars]
            )
            ttft_request(args.endpoint, args.model, donor_prompt, max_tokens=50)

            # Query target (prompt B)
            target_prompt = PROMPT_TEMPLATE.format(
                content=pair["text_b"][:args.min_chars]
            )
            t, ok = ttft_request(
                args.endpoint, args.model, target_prompt, max_tokens=5
            )
            if ok:
                bucket_ttfts.append(t)
                if t < 0.70 * cold_mean:
                    bucket_hits += 1

        if not bucket_ttfts:
            continue

        n = len(bucket_ttfts)
        hit_pct = bucket_hits / n * 100
        p50 = sorted(bucket_ttfts)[n // 2]
        mean_t = sum(bucket_ttfts) / n
        speedup = cold_mean / mean_t if mean_t > 0 else 0.0

        hit_only = [t for t in bucket_ttfts if t < 0.70 * cold_mean]
        hit_speedup = (
            cold_mean / (sum(hit_only) / len(hit_only))
            if hit_only
            else 0.0
        )

        print(
            f"  {bucket_key:<14} {n:>4} {bucket_hits:>5} "
            f"{hit_pct:>5.0f}% {p50:>7.0f}ms {speedup:>7.2f}x "
            f"{hit_speedup:>7.2f}x"
        )

        results[bucket_key] = {
            "n": n,
            "hits": bucket_hits,
            "hit_pct": hit_pct,
            "p50_ms": p50,
            "mean_ms": mean_t,
            "speedup": speedup,
            "hit_only_speedup": hit_speedup,
            "_raw_ttfts": bucket_ttfts,
        }

    # Summary
    total_pairs = sum(r["n"] for r in results.values())
    total_hits = sum(r["hits"] for r in results.values())
    overall_hit = total_hits / total_pairs * 100 if total_pairs > 0 else 0

    print()
    print(f"Overall: {total_hits}/{total_pairs} hits ({overall_hit:.1f}%)")
    print(f"Cold baseline: mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import (
        bootstrap_mean,
        bootstrap_proportion,
        bootstrap_speedup,
    )

    print()
    print("=" * 65)
    print("Bootstrap 95% Confidence Intervals")
    print("=" * 65)

    cold_arr = np.array(cold_ttfts)
    print(f"  Cold TTFT mean: {bootstrap_mean(cold_arr)}")
    print(f"  Overall hit rate: {bootstrap_proportion(total_hits, total_pairs)}")

    for bucket_key, br in results.items():
        b_arr = np.array(br["_raw_ttfts"])
        print(f"\n  Bucket [{bucket_key}):")
        print(f"    TTFT mean:  {bootstrap_mean(b_arr)}")
        print(f"    Hit rate:   {bootstrap_proportion(br['hits'], br['n'])}")
        if len(b_arr) > 0:
            print(f"    Speedup:    {bootstrap_speedup(b_arr, cold_arr)}")
    print()

    if args.output:
        serializable_buckets = {
            k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
            for k, v in results.items()
        }
        out = {
            "cold_mean_ms": cold_mean,
            "cold_p50_ms": cold_p50,
            "total_pairs": total_pairs,
            "total_hits": total_hits,
            "overall_hit_pct": overall_hit,
            "buckets": serializable_buckets,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
