#!/usr/bin/env python3
"""Operating Envelope Benchmark — Break-Even Hit Rate Characterization.

MOTIVATION (Paper Hole 1)
--------------------------
All existing results use structured workloads (XSum, CNN/DM, WikiHow) that achieve
75-100% hit rates. The paper lacks data on what happens when hit rates are LOW.

This benchmark answers: "At what hit rate does SemBlend become net-neutral or negative
vs cold serving?" This is critical for production deployment decisions.

ANALYSIS
--------
Expected TTFT = P(hit) × TTFT_hit + P(miss) × TTFT_cold

If hit_speedup = S (e.g., S=5 at 8K tokens), then:
  Expected_TTFT / Cold_TTFT = P(hit)/S + (1-P(hit)) × 1.0

Break-even: Expected_TTFT = Cold_TTFT
  → P(hit)/S + (1-P(hit)) = 1.0
  → P(hit) × (1/S - 1) = 0
  → For S > 1: break-even at ANY non-zero hit rate (always beneficial)

BUT: SemBlend adds overhead (~8ms pipeline for misses too).
Real break-even with overhead_ms:
  P(hit) × (TTFT_cold - TTFT_hit) ≥ overhead_ms × 1.0
  P(hit) ≥ overhead_ms / (TTFT_cold - TTFT_hit)

At 8K tokens: cold ≈ 2000ms, hit ≈ 400ms, overhead ≈ 8ms
  P(hit) ≥ 8 / (2000 - 400) = 8/1600 = 0.5%

So break-even is at ~0.5% hit rate — SemBlend is ALWAYS beneficial unless
hit rate is essentially zero!

METHODOLOGY
-----------
To empirically measure the operating envelope:
1. Build a mixed workload with controlled hit rate: k% donors match, (100-k)% are random
2. Measure mean TTFT at each target hit rate
3. Plot: mean TTFT / cold TTFT vs actual hit rate
4. Compare to analytical model

Controlled hit rate implementation:
- Use cluster variations as "matching" prompts (expected hit rate ~90%)
- Use completely unrelated random articles as "non-matching" prompts (expected hit rate ~0%)
- Mix N_hit matching prompts with N_miss non-matching in configured ratio
- Interleave to simulate realistic request stream

USAGE
-----
    python -m benchmarks.e2e.operating_envelope_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters-file benchmarks/data/cnn_dailymail_clusters.json \\
        --target-length 8192 \\
        --n-per-point 16 \\
        --output results/operating-envelope/qwen_8k.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


def measure_ttft(endpoint: str, model: str, prompt: str, timeout: int = 300) -> float:
    """Measure TTFT via streaming API."""
    t0 = time.monotonic()
    ttft = -1.0
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model, "prompt": prompt,
                "max_tokens": 5, "temperature": 0.0, "stream": True,
            },
            timeout=timeout, stream=True,
        )
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line_s = line.decode("utf-8", errors="replace")
            if not line_s.startswith("data: "):
                continue
            payload = line_s[6:].strip()
            if payload == "[DONE]":
                break
            try:
                data = json.loads(payload)
                token = data["choices"][0].get("text", "")
                if token:
                    ttft = (time.monotonic() - t0) * 1000
                    # Drain remaining stream
                    for _ in resp.iter_lines():
                        pass
                    break
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        if ttft < 0:
            ttft = (time.monotonic() - t0) * 1000
    except Exception as e:
        print(f"  [WARN] Request failed: {e}")
    return ttft


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 5, "temperature": 0.0},
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Donor registration failed: {e}")


@dataclass
class EnvelopePoint:
    target_hit_fraction: float
    n_matching: int
    n_nonmatching: int
    actual_hit_rate: float   # measured via speedup proxy
    mean_ttft_ms: float
    cold_ttft_ms: float
    mean_speedup: float
    ttft_ratio: float  # mean_ttft / cold_ttft


def run_operating_envelope_bench(
    endpoint: str,
    model: str,
    clusters_file: str,
    target_length: int = 8192,
    n_per_point: int = 16,
    hit_fractions: list[float] | None = None,
    output: str | None = None,
    settle_time: float = 1.5,
) -> dict:
    """Run operating envelope benchmark across a range of hit fractions."""
    if hit_fractions is None:
        hit_fractions = [0.0, 0.10, 0.25, 0.50, 0.75, 1.00]

    with open(clusters_file) as f:
        clusters_data = json.load(f)

    # Normalize: handle both list format and {"clusters": [...]} format
    if isinstance(clusters_data, list):
        raw = clusters_data
    else:
        raw = clusters_data.get("clusters", list(clusters_data.values()) if isinstance(clusters_data, dict) else [])
        if isinstance(raw, dict):
            raw = list(raw.values())

    # Normalize each cluster to a common format
    def normalize_cluster(c: dict) -> dict:
        """Normalize cluster to standard format with seed_text, seed_token_count, variations_dict."""
        if "seed" in c:
            # Old format: {"seed": {"text": ..., "token_count": ...}, "variations": {"exact": {...}}}
            seed_text = c["seed"].get("text", c["seed"].get("prompt", ""))
            seed_token_count = c["seed"].get("token_count", 0)
            vars_raw = c.get("variations", {})
            if isinstance(vars_raw, dict):
                variations_dict = {k: v.get("text", v.get("prompt", "")) for k, v in vars_raw.items()}
            else:
                variations_dict = {v["overlap_type"]: v["text"] for v in vars_raw if "text" in v}
        else:
            # New flat format: {"seed_text": ..., "seed_token_count": ..., "variations": [...]}
            seed_text = c.get("seed_text", "")
            seed_token_count = c.get("seed_token_count", c.get("target_token_length", 0))
            vars_raw = c.get("variations", [])
            if isinstance(vars_raw, list):
                variations_dict = {v["overlap_type"]: v["text"] for v in vars_raw if "text" in v}
            else:
                variations_dict = {k: v.get("text", "") for k, v in vars_raw.items()}
        return {"seed_text": seed_text, "seed_token_count": seed_token_count, "variations": variations_dict}

    clusters = [normalize_cluster(c) for c in raw]

    # Filter clusters to target_length
    usable = [
        c for c in clusters
        if abs(c["seed_token_count"] - target_length) <= target_length * 0.30
        and c["seed_text"]
    ]

    if len(usable) < 4:
        print(f"ERROR: Only {len(usable)} clusters near target_length={target_length}")
        print(f"  Total clusters: {len(clusters)}, token counts sample: {[c['seed_token_count'] for c in clusters[:5]]}")
        sys.exit(1)

    print(f"=== Operating Envelope Benchmark ===")
    print(f"model={model}, target_length={target_length}, n_per_point={n_per_point}")
    print(f"hit_fractions={hit_fractions}")
    print(f"usable clusters: {len(usable)}")

    # IMPORTANT: We do NOT pre-warm prompts from the test set to avoid LMCache contamination.
    # Cold baseline will be derived from hit_fraction=0.0 results (pure miss condition).
    # Using a placeholder here; updated after all results are in.
    cold_baseline_ms: float = -1.0  # Determined from hit_fraction=0.0 results below

    results: list[EnvelopePoint] = []
    used_prompts: set[str] = set()  # Track prompts already measured (avoid LMCache contamination)

    for hit_frac in hit_fractions:
        print(f"\n--- Hit fraction = {hit_frac:.2f} ---")

        n_matching = int(round(n_per_point * hit_frac))
        n_nonmatching = n_per_point - n_matching

        print(f"  {n_matching} matching + {n_nonmatching} non-matching prompts")

        # Get matching prompts (seed+variation from clusters)
        matching_pairs: list[tuple[str, str]] = []
        for c in usable:
            if len(matching_pairs) >= n_matching:
                break
            seed_text = c["seed_text"]
            variations = c["variations"]
            # Use EXACT variation (highest expected hit rate), fall back to reorder
            var_text = variations.get("exact", variations.get("reorder", ""))
            if seed_text and var_text:
                matching_pairs.append((seed_text, var_text))

        # Get non-matching prompts: FRESH clusters not used in any prior round
        # (avoids LMCache contamination from previous hit_fraction iterations)
        fresh_nonmatching = [
            c["seed_text"] for c in usable
            if c["seed_text"]
            and c["seed_text"] not in used_prompts
            and c["seed_text"] not in {s for s, _ in matching_pairs}
            and c["seed_text"] not in {v for _, v in matching_pairs}
        ]
        rng = random.Random(42 + int(hit_frac * 100))
        rng.shuffle(fresh_nonmatching)
        non_matching_prompts = fresh_nonmatching[:n_nonmatching]

        # Register donors for matching prompts
        print("  Registering donors...")
        for seed_text, _ in matching_pairs:
            register_donor(endpoint, model, seed_text)
            time.sleep(0.3)

        time.sleep(settle_time)

        # Build interleaved request list
        request_list: list[tuple[str, bool]] = []
        for _, var_text in matching_pairs:
            request_list.append((var_text, True))  # expected to hit
        for nm_text in non_matching_prompts:
            request_list.append((nm_text, False))  # expected to miss
        rng.shuffle(request_list)

        # Run all requests
        ttfts: list[float] = []
        hit_proxies: list[bool] = []

        for prompt, expected_hit in request_list:
            if not prompt:
                continue
            t = measure_ttft(endpoint, model, prompt)
            if t > 0:
                ttfts.append(t)
                # Proxy hit detection: speedup ≥ 1.5 vs cold baseline (updated after 0% run)
                ref = cold_baseline_ms if cold_baseline_ms > 0 else 3000.0
                hit_proxies.append((ref / t) >= 1.5)
            time.sleep(0.5)

        if not ttfts:
            print(f"  SKIPPED: no successful requests")
            continue

        mean_ttft = statistics.mean(ttfts)

        # After hit_fraction=0.0, set cold baseline from measured miss-only results
        if hit_frac == 0.0 and cold_baseline_ms < 0:
            cold_baseline_ms = mean_ttft
            print(f"  Cold baseline set from 0% hit run: {cold_baseline_ms:.0f}ms")
            # Recompute hit proxies with correct cold baseline
            hit_proxies = [(cold_baseline_ms / t) >= 1.5 for t in ttfts]

        actual_hit_rate = sum(hit_proxies) / len(hit_proxies)

        cold_ref = cold_baseline_ms if cold_baseline_ms > 0 else mean_ttft
        point = EnvelopePoint(
            target_hit_fraction=hit_frac,
            n_matching=n_matching,
            n_nonmatching=n_nonmatching,
            actual_hit_rate=actual_hit_rate,
            mean_ttft_ms=mean_ttft,
            cold_ttft_ms=cold_ref,
            mean_speedup=cold_ref / mean_ttft if mean_ttft > 0 else 0.0,
            ttft_ratio=mean_ttft / cold_ref if cold_ref > 0 else 0.0,
        )
        results.append(point)
        # Mark all prompts measured this round as used
        for prompt, _ in request_list:
            used_prompts.add(prompt)
        for seed, var in matching_pairs:
            used_prompts.add(seed)
            used_prompts.add(var)
        cold_ref = cold_baseline_ms if cold_baseline_ms > 0 else mean_ttft
        print(
            f"  actual_hit_rate={actual_hit_rate:.3f} "
            f"mean_ttft={mean_ttft:.0f}ms "
            f"speedup={cold_ref/mean_ttft:.2f}× "
            f"ttft_ratio={mean_ttft/cold_ref:.3f}"
        )

    # Final cold baseline (from hit_fraction=0.0 results)
    zero_hit_results = [p for p in results if p.target_hit_fraction == 0.0]
    if zero_hit_results:
        cold_baseline_ms = zero_hit_results[0].mean_ttft_ms
        # Recompute speedup/ratio with correct cold baseline
        for p in results:
            p.cold_ttft_ms = cold_baseline_ms
            if p.mean_ttft_ms > 0:
                p.mean_speedup = cold_baseline_ms / p.mean_ttft_ms
                p.ttft_ratio = p.mean_ttft_ms / cold_baseline_ms

    # Print summary table
    print("\n=== Operating Envelope Summary ===")
    print(f"Cold baseline (from 0% hit run): {cold_baseline_ms:.0f}ms")
    print(f"{'Target HR':>10} {'Actual HR':>10} {'Mean TTFT':>10} {'Speedup':>10} {'Ratio':>8}")
    for p in results:
        print(
            f"{p.target_hit_fraction:>10.2f} {p.actual_hit_rate:>10.3f} "
            f"{p.mean_ttft_ms:>10.0f}ms {p.mean_speedup:>10.2f}× "
            f"{p.ttft_ratio:>8.3f}"
        )

    # Analytical model for comparison
    if results:
        print("\n--- Analytical Model vs Measured ---")
        hit_ttfts = [p.mean_ttft_ms for p in results if p.actual_hit_rate >= 0.8]
        miss_ttfts = [p.mean_ttft_ms for p in results if p.actual_hit_rate <= 0.2]
        if hit_ttfts and miss_ttfts:
            hit_ttft_est = statistics.mean(hit_ttfts)
            miss_ttft_est = statistics.mean(miss_ttfts)
            print(f"Estimated hit TTFT: {hit_ttft_est:.0f}ms, miss TTFT: {miss_ttft_est:.0f}ms")
            for frac in [0.10, 0.25, 0.50, 0.75]:
                predicted = frac * hit_ttft_est + (1 - frac) * miss_ttft_est
                print(f"  P(hit)={frac:.2f} → predicted {predicted:.0f}ms")

    # Machine-parseable output
    print("\n---")
    print(f"benchmark:     operating_envelope")
    print(f"cold_ms:       {cold_baseline_ms:.1f}")
    for p in results:
        print(f"hit{int(p.target_hit_fraction*100):02d}_actual_hr: {p.actual_hit_rate:.4f}")
        print(f"hit{int(p.target_hit_fraction*100):02d}_speedup:   {p.mean_speedup:.4f}")
        print(f"hit{int(p.target_hit_fraction*100):02d}_ratio:     {p.ttft_ratio:.4f}")
    print("---")

    output_data = {
        "benchmark": "operating_envelope",
        "model": model,
        "clusters_file": clusters_file,
        "target_length": target_length,
        "cold_baseline_ms": cold_baseline_ms,
        "hit_fractions": hit_fractions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "results": [asdict(p) for p in results],
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="Operating envelope benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", required=True)
    parser.add_argument("--target-length", type=int, default=8192)
    parser.add_argument("--n-per-point", type=int, default=16,
                        help="Total requests per hit-fraction point")
    parser.add_argument("--hit-fractions", default="0.0,0.10,0.25,0.50,0.75,1.0")
    parser.add_argument("--settle-time", type=float, default=1.5)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    hit_fractions = [float(x.strip()) for x in args.hit_fractions.split(",")]
    run_operating_envelope_bench(
        endpoint=args.endpoint,
        model=args.model,
        clusters_file=args.clusters_file,
        target_length=args.target_length,
        n_per_point=args.n_per_point,
        hit_fractions=hit_fractions,
        output=args.output,
        settle_time=args.settle_time,
    )


if __name__ == "__main__":
    main()
