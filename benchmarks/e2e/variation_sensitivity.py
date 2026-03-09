#!/usr/bin/env python3
"""Per-variation-type PPL sensitivity study.

Measures PPL ratio per variation type to create a quality vs semantic overlap curve
matching SemShareKV Figure 9. Uses 8K CNN/DM cluster data.

Methodology per cluster:
  1. SEED: Run inference on seed text → registers as donor (SemBlend adds automatically)
  2. VARIATION: Run inference on each variation type → SemBlend tries to match seed
  3. COLD: Run same variation with unique suffix to disable donor reuse → cold baseline
  4. PPL ratio = SEMBLEND_PPL / COLD_PPL

The variation types form a similarity gradient:
  exact (overlap=1.0) → reorder (0.90) → partial_80 (0.80) → paraphrase (0.77)
  → partial_60 (0.65) → partial_40 (0.47) → diverse (0.20)

Usage:
    python -m benchmarks.e2e.variation_sensitivity \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --clusters-file benchmarks/data/cnn_dailymail_clusters.json \
        --target-length 8192 \
        --n-clusters 25 \
        --max-tokens 64 \
        --output results/variation-sensitivity.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


OVERLAP_TYPE_COSINE = {
    "exact": 1.00,
    "reorder": 0.90,
    "partial_80": 0.80,
    "paraphrase": 0.77,
    "partial_60": 0.65,
    "partial_40": 0.47,
    "partial_20": 0.25,
    "diverse": 0.20,
}


@dataclass
class VariationResult:
    cluster_id: str
    overlap_type: str
    cosine_est: float
    expected_overlap: float
    cold_ppl: float
    semblend_ppl: float
    ppl_ratio: float
    semblend_ttft_ms: float
    cold_ttft_ms: float
    got_hit: bool


def call_vllm(endpoint: str, model: str, prompt: str, max_tokens: int,
              seed: int = 42, timeout: int = 120) -> dict:
    """Call vLLM /v1/completions with logprobs."""
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "logprobs": 1,
            "seed": seed,
            "temperature": 0.0,
        },
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def compute_ppl(logprobs_list: list) -> float:
    """Compute perplexity from vLLM logprobs list."""
    total_log_prob = 0.0
    n = 0
    for tok_logprobs in logprobs_list:
        if tok_logprobs is None:
            continue
        for token_id, lp in tok_logprobs.items():
            total_log_prob += lp
            n += 1
    if n == 0:
        return 1.0
    return math.exp(-total_log_prob / n)


def run_inference(endpoint: str, model: str, prompt: str, max_tokens: int,
                  seed: int = 42) -> tuple[float, float]:
    """Returns (ppl, ttft_ms)."""
    t0 = time.perf_counter()
    result = call_vllm(endpoint, model, prompt, max_tokens, seed=seed)
    ttft_ms = (time.perf_counter() - t0) * 1000
    choice = result["choices"][0]
    logprobs = choice.get("logprobs", {})
    lp_list = logprobs.get("token_logprobs", []) if logprobs else []
    ppl = compute_ppl([{i: lp} for i, lp in enumerate(lp_list) if lp is not None])
    return ppl, ttft_ms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", default="benchmarks/data/cnn_dailymail_clusters.json")
    parser.add_argument("--target-length", type=int, default=8192)
    parser.add_argument("--n-clusters", type=int, default=25)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--output", default="results/variation-sensitivity.json")
    args = parser.parse_args()

    # Load clusters
    clusters_path = Path(args.clusters_file)
    if not clusters_path.is_absolute():
        clusters_path = Path(__file__).parent.parent.parent / clusters_path
    with open(clusters_path) as f:
        all_clusters = json.load(f)

    # Filter to target length
    clusters = [c for c in all_clusters if c.get("target_token_length") == args.target_length]
    if not clusters:
        # Try without exact match
        lengths = sorted(set(c.get("target_token_length", 0) for c in all_clusters))
        closest = min(lengths, key=lambda x: abs(x - args.target_length))
        clusters = [c for c in all_clusters if c.get("target_token_length") == closest]
        print(f"Note: using length {closest} (closest to {args.target_length})")

    clusters = clusters[:args.n_clusters]
    print(f"Testing {len(clusters)} clusters at length ~{args.target_length} tokens")
    print(f"Max tokens for generation: {args.max_tokens}")

    # Test endpoint
    try:
        r = requests.get(f"{args.endpoint}/health", timeout=5)
        print(f"Endpoint health: {r.status_code}")
    except Exception as e:
        print(f"ERROR: Cannot reach {args.endpoint}: {e}")
        sys.exit(1)

    results: list[VariationResult] = []
    variation_types = ["exact", "reorder", "partial_80", "paraphrase", "partial_60", "partial_40", "diverse"]

    for ci, cluster in enumerate(clusters):
        cluster_id = cluster["cluster_id"]
        print(f"\n[{ci+1}/{len(clusters)}] Cluster {cluster_id[:12]}")

        # Get seed and register it as donor
        seed_text = cluster["seed_text"]
        print(f"  Registering seed as donor...")
        try:
            seed_ppl, seed_ttft = run_inference(args.endpoint, args.model, seed_text, args.max_tokens, seed=42)
            print(f"  Seed registered (TTFT: {seed_ttft:.0f}ms, PPL: {seed_ppl:.3f})")
        except Exception as e:
            print(f"  ERROR registering seed: {e}")
            continue

        # Index variations by type
        var_by_type = {}
        for v in cluster.get("variations", []):
            vtype = v.get("overlap_type", "unknown")
            if vtype not in var_by_type:
                var_by_type[vtype] = v

        for vtype in variation_types:
            if vtype not in var_by_type:
                continue
            variation = var_by_type[vtype]
            var_text = variation["text"]
            cos_est = OVERLAP_TYPE_COSINE.get(vtype, 0.0)
            exp_overlap = variation.get("expected_token_overlap", cos_est)

            # SemBlend run (donor registered)
            try:
                sb_ppl, sb_ttft = run_inference(args.endpoint, args.model, var_text, args.max_tokens, seed=100+ci)
            except Exception as e:
                print(f"  ERROR semblend {vtype}: {e}")
                continue

            # Cold baseline: append unique suffix to defeat SemBlend matching
            import hashlib
            unique_suffix = f"\n\n[UNIQUE_ID_{hashlib.md5(f'{cluster_id}_{vtype}'.encode()).hexdigest()[:8]}]"
            cold_text = var_text + unique_suffix
            try:
                cold_ppl, cold_ttft = run_inference(args.endpoint, args.model, cold_text, args.max_tokens, seed=200+ci)
            except Exception as e:
                print(f"  ERROR cold {vtype}: {e}")
                continue

            ppl_ratio = sb_ppl / cold_ppl if cold_ppl > 0 else 1.0
            got_hit = sb_ttft < cold_ttft * 0.75

            result = VariationResult(
                cluster_id=cluster_id,
                overlap_type=vtype,
                cosine_est=cos_est,
                expected_overlap=exp_overlap,
                cold_ppl=cold_ppl,
                semblend_ppl=sb_ppl,
                ppl_ratio=ppl_ratio,
                semblend_ttft_ms=sb_ttft,
                cold_ttft_ms=cold_ttft,
                got_hit=got_hit,
            )
            results.append(result)
            hit_str = "HIT" if got_hit else "miss"
            print(f"  {vtype:<12} cos={cos_est:.2f}  PPL={ppl_ratio:.3f}  [{hit_str}]  {sb_ttft:.0f}/{cold_ttft:.0f}ms")

    # Summarize by variation type
    print("\n" + "="*70)
    print("Per-variation-type summary:")
    print(f"{'Type':<14} {'Est Cos':<10} {'Mean PPL':<12} {'Median PPL':<12} {'Hit Rate':<10} n")
    by_type = defaultdict(list)
    for r in results:
        by_type[r.overlap_type].append(r)

    summary = {}
    for vtype in variation_types:
        if vtype not in by_type:
            continue
        items = by_type[vtype]
        ppls = [r.ppl_ratio for r in items]
        hit_rate = sum(1 for r in items if r.got_hit) / len(items)
        cos = items[0].cosine_est
        mean_ppl = statistics.mean(ppls)
        med_ppl = statistics.median(ppls)
        print(f"{vtype:<14} {cos:<10.2f} {mean_ppl:<12.3f} {med_ppl:<12.3f} {hit_rate*100:<10.0f}% {len(items)}")
        summary[vtype] = {
            "cosine_est": cos,
            "mean_ppl": mean_ppl,
            "median_ppl": med_ppl,
            "hit_rate": hit_rate,
            "n": len(items),
            "ppls": ppls,
        }

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "target_length": args.target_length,
        "n_clusters": len(clusters),
        "max_tokens": args.max_tokens,
        "summary": summary,
        "raw": [asdict(r) for r in results],
    }
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
