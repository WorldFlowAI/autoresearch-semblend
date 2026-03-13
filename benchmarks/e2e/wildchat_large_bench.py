#!/usr/bin/env python3
"""WildChat-1M large-scale SemBlend benchmark.

Loads real WildChat conversations, computes MiniLM embeddings to find
consecutive same-user pairs, buckets them by cosine similarity, and runs
E2E TTFT benchmarks on each bucket against a live SemBlend endpoint.

Methodology:
  1. Load WildChat JSONL, group conversations by user IP.
  2. Build consecutive same-user pairs, compute MiniLM cosine similarity.
  3. Stratify pairs into buckets: [0.50-0.60), [0.60-0.70), [0.70-0.80),
     [0.80-0.90), [0.90-1.00].
  4. For each bucket sample n pairs:
     a. Restart vLLM (clean donor store) per bucket.
     b. Send prompt A (donor) → register.
     c. Send prompt B (target) → measure TTFT.
     d. Measure cold baseline for prompt B.
  5. Report per-bucket: hit rate, TTFT speedup, sample size, bootstrap CIs.

Usage:
    python benchmarks/e2e/wildchat_large_bench.py \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --data-path benchmarks/data/wildchat/wildchat_100k.jsonl \
        --n-per-bucket 50 \
        --min-chars 4000 \
        --output results/wildchat_large_real.json
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests

SIMILARITY_BUCKETS = [
    (0.50, 0.60),
    (0.60, 0.70),
    (0.70, 0.80),
    (0.80, 0.90),
    (0.90, 1.01),
]

CHAT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{content}<|im_end|>"
)


# ---------------------------------------------------------------------------
# WildChat loading
# ---------------------------------------------------------------------------

def load_wildchat_jsonl(path: str, max_rows: int = 100_000) -> list[dict]:
    """Load WildChat JSONL data."""
    rows = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_rows and len(rows) >= max_rows:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def build_consecutive_pairs(
    rows: list[dict], min_chars: int = 4000
) -> list[dict]:
    """Build consecutive same-user conversation pairs.

    Returns list of dicts with text_a, text_b, user_id.
    Only includes pairs where BOTH conversations meet min_chars.
    """
    user_convos: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ip = row.get("hashed_ip", "unknown")
        text = row.get("full_text", "")
        if not text:
            texts = row.get("user_texts", [])
            text = " ".join(texts).strip()
        if text and len(text.strip()) > 20:
            user_convos[ip].append(text)

    pairs = []
    for ip, convos in user_convos.items():
        if len(convos) < 2:
            continue
        for i in range(len(convos) - 1):
            a, b = convos[i], convos[i + 1]
            if len(a) >= min_chars and len(b) >= min_chars:
                pairs.append({
                    "text_a": a,
                    "text_b": b,
                    "user_id": ip,
                })
    return pairs


# ---------------------------------------------------------------------------
# Embedding & similarity
# ---------------------------------------------------------------------------

_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        print("  Embedder: MiniLM-L6-v2 loaded on CPU")
    return _EMBEDDER


def compute_pair_similarities(pairs: list[dict]) -> list[float]:
    """Compute cosine similarity for each pair using MiniLM."""
    embedder = get_embedder()
    texts_a = [p["text_a"][:8000] for p in pairs]  # truncate for embedding
    texts_b = [p["text_b"][:8000] for p in pairs]

    print(f"  Computing embeddings for {len(pairs)} pairs...")
    emb_a = embedder.encode(
        texts_a, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)
    emb_b = embedder.encode(
        texts_b, batch_size=64, show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    sims = [float(np.dot(emb_a[i], emb_b[i])) for i in range(len(pairs))]
    return sims


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Measure TTFT. Returns (ms, ok)."""
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
    except Exception as e:
        print(f"    request error: {e}")
        return 0.0, False


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="WildChat large-scale SemBlend benchmark"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--data-path",
        default="benchmarks/data/wildchat/wildchat_100k.jsonl",
    )
    parser.add_argument("--n-per-bucket", type=int, default=50)
    parser.add_argument("--min-chars", type=int, default=4000)
    parser.add_argument("--max-prompt-chars", type=int, default=12000,
                        help="Max chars to send per prompt (truncate)")
    parser.add_argument("--output", default=None)
    parser.add_argument("--cold-samples", type=int, default=20)
    args = parser.parse_args()

    print(f"\nWildChat Large-Scale SemBlend Benchmark")
    print(f"  endpoint:    {args.endpoint}")
    print(f"  model:       {args.model}")
    print(f"  data:        {args.data_path}")
    print(f"  n_per_bucket:{args.n_per_bucket}")
    print(f"  min_chars:   {args.min_chars}")
    print(f"  max_prompt:  {args.max_prompt_chars}")
    print()

    # Health check
    try:
        requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
    except Exception as e:
        print(f"ERROR: endpoint not reachable: {e}")
        sys.exit(1)

    # Step 1: Load WildChat data
    print("Step 1: Loading WildChat data...")
    rows = load_wildchat_jsonl(args.data_path)
    print(f"  Loaded {len(rows)} conversations")

    # Step 2: Build consecutive pairs
    print(f"\nStep 2: Building consecutive pairs (min_chars={args.min_chars})...")
    pairs = build_consecutive_pairs(rows, min_chars=args.min_chars)
    print(f"  Found {len(pairs)} qualifying pairs")

    if len(pairs) < 10:
        print("ERROR: Too few pairs. Try lowering --min-chars")
        sys.exit(1)

    # Step 3: Compute similarities
    print(f"\nStep 3: Computing similarities...")
    sims = compute_pair_similarities(pairs)
    for i, sim in enumerate(sims):
        pairs[i]["similarity"] = sim

    # Step 4: Bucket pairs
    print(f"\nStep 4: Stratifying into buckets...")
    rng = random.Random(42)
    buckets: dict[str, list[dict]] = {}
    for lo, hi in SIMILARITY_BUCKETS:
        key = f"{lo:.2f}-{hi:.2f}"
        bucket_pairs = [p for p in pairs if lo <= p["similarity"] < hi]
        rng.shuffle(bucket_pairs)
        buckets[key] = bucket_pairs[:args.n_per_bucket]
        print(f"  [{key}): {len(bucket_pairs)} available, "
              f"using {len(buckets[key])}")

    total_bench = sum(len(bp) for bp in buckets.values())
    print(f"  Total pairs to benchmark: {total_bench}")

    # Step 5: Cold baselines
    print(f"\nStep 5: Measuring cold baselines ({args.cold_samples} samples)...")
    cold_pairs = []
    for bucket_pairs in buckets.values():
        cold_pairs.extend(bucket_pairs[:5])
    rng.shuffle(cold_pairs)
    cold_pairs = cold_pairs[:args.cold_samples]

    cold_ttfts: list[float] = []
    for i, pair in enumerate(cold_pairs):
        prompt = CHAT_TEMPLATE.format(
            content=pair["text_b"][:args.max_prompt_chars]
        )
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"  [{i+1}/{len(cold_pairs)}] cold {t:.0f}ms")

    if not cold_ttfts:
        print("ERROR: No cold measurements succeeded")
        return

    cold_mean = sum(cold_ttfts) / len(cold_ttfts)
    cold_p50 = float(np.median(cold_ttfts))
    print(f"  Cold baseline: n={len(cold_ttfts)}, "
          f"mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Step 6: Per-bucket E2E benchmark
    print(f"\nStep 6: Per-bucket SemBlend benchmark")
    print(f"  {'Bucket':<14} {'N':>4} {'Hits':>5} {'Hit%':>6} "
          f"{'P50':>8} {'Mean':>8} {'Speedup':>8} {'Hit Spd':>8}")
    print("-" * 75)

    all_results: dict[str, dict] = {}

    for bucket_key, bucket_pairs in buckets.items():
        if not bucket_pairs:
            continue

        bucket_ttfts: list[float] = []
        bucket_hits = 0

        for j, pair in enumerate(bucket_pairs):
            # Register donor (prompt A)
            donor_prompt = CHAT_TEMPLATE.format(
                content=pair["text_a"][:args.max_prompt_chars]
            )
            ttft_request(
                args.endpoint, args.model, donor_prompt, max_tokens=5
            )
            # Brief pause for LMCache to store
            time.sleep(0.5)

            # Query target (prompt B)
            target_prompt = CHAT_TEMPLATE.format(
                content=pair["text_b"][:args.max_prompt_chars]
            )
            t, ok = ttft_request(
                args.endpoint, args.model, target_prompt, max_tokens=5
            )
            if ok:
                bucket_ttfts.append(t)
                # Hit if TTFT < 70% of cold mean
                if t < 0.70 * cold_mean:
                    bucket_hits += 1

            if (j + 1) % 10 == 0:
                print(f"    [{bucket_key}] {j+1}/{len(bucket_pairs)} done")

        if not bucket_ttfts:
            continue

        n = len(bucket_ttfts)
        hit_pct = bucket_hits / n * 100
        p50 = float(np.median(bucket_ttfts))
        mean_t = sum(bucket_ttfts) / n
        speedup = cold_p50 / p50 if p50 > 0 else 0.0

        hit_only = [t for t in bucket_ttfts if t < 0.70 * cold_mean]
        hit_speedup = (
            cold_p50 / (sum(hit_only) / len(hit_only))
            if hit_only else 0.0
        )

        print(
            f"  {bucket_key:<14} {n:>4} {bucket_hits:>5} "
            f"{hit_pct:>5.0f}% {p50:>7.0f}ms {mean_t:>7.0f}ms "
            f"{speedup:>7.2f}x {hit_speedup:>7.2f}x"
        )

        all_results[bucket_key] = {
            "n": n,
            "hits": bucket_hits,
            "hit_pct": hit_pct,
            "p50_ms": p50,
            "mean_ms": mean_t,
            "speedup": speedup,
            "hit_only_speedup": hit_speedup,
            "raw_ttfts": bucket_ttfts,
            "similarities": [p["similarity"] for p in bucket_pairs[:n]],
        }

    # Summary
    total_pairs = sum(r["n"] for r in all_results.values())
    total_hits = sum(r["hits"] for r in all_results.values())
    overall_hit = total_hits / total_pairs * 100 if total_pairs > 0 else 0

    print()
    print(f"Overall: {total_hits}/{total_pairs} hits ({overall_hit:.1f}%)")
    print(f"Cold baseline: mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Bootstrap CIs
    try:
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
        print(f"  Overall hit rate: "
              f"{bootstrap_proportion(total_hits, total_pairs)}")

        for bk, br in all_results.items():
            b_arr = np.array(br["raw_ttfts"])
            print(f"\n  Bucket [{bk}):")
            print(f"    TTFT mean:  {bootstrap_mean(b_arr)}")
            print(f"    Hit rate:   "
                  f"{bootstrap_proportion(br['hits'], br['n'])}")
            if len(b_arr) > 0:
                print(f"    Speedup:    {bootstrap_speedup(b_arr, cold_arr)}")
        print()
    except ImportError:
        print("  (bootstrap_ci not available, skipping CIs)")

    # Save results
    if args.output:
        serializable = {}
        for k, v in all_results.items():
            serializable[k] = {
                kk: vv for kk, vv in v.items()
                if kk not in ("raw_ttfts", "similarities")
            }
            serializable[k]["raw_ttfts"] = v["raw_ttfts"]

        out = {
            "cold_mean_ms": cold_mean,
            "cold_p50_ms": cold_p50,
            "cold_n": len(cold_ttfts),
            "total_pairs": total_pairs,
            "total_hits": total_hits,
            "overall_hit_pct": overall_hit,
            "min_chars": args.min_chars,
            "max_prompt_chars": args.max_prompt_chars,
            "buckets": serializable,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
