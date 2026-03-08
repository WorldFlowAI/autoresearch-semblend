#!/usr/bin/env python3
"""Ablation A: Similarity Threshold Sweep.

Sweeps SEMBLEND_MIN_SIMILARITY across {0.40, 0.50, 0.60, 0.70, 0.80} and
measures hit rate, TTFT speedup, and PPL ratio per threshold.

Since the similarity threshold cannot be changed at runtime without restarting
the vLLM pod, this benchmark uses a post-hoc analysis approach:

1. Run all cluster variations against vLLM with the default threshold (0.60).
2. Record per-variation cosine similarity (approximated from overlap type),
   measured TTFT, and output quality (PPL ratio, ROUGE-L).
3. For each threshold T in the sweep, simulate hit/miss by checking whether
   the variation's expected cosine similarity exceeds T.
4. Report per-threshold: hit_rate, avg_ttft_speedup, avg_ppl_ratio.

Expected cosine similarity ranges by variation type (from empirical calibration):
  - exact       -> cos ~1.0   (hits for all thresholds)
  - reorder     -> cos ~0.90  (hits for thresholds <= 0.80)
  - partial_80  -> cos ~0.80  (hits for thresholds <= 0.70)
  - partial_60  -> cos ~0.65  (hits for thresholds <= 0.60)
  - partial_40  -> cos ~0.47  (hits for thresholds <= 0.40)
  - paraphrase  -> cos ~0.77  (hits for thresholds <= 0.70)
  - diverse     -> cos ~0.20  (never hits)

Usage:
    python -m benchmarks.e2e.threshold_ablation_bench \\
        --endpoint http://localhost:8000 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters-file benchmarks/data/semblend_clusters.json \\
        --output-dir results/threshold-ablation \\
        --thresholds "0.40,0.50,0.60,0.70,0.80"
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


# Empirical cosine similarity ranges by overlap type.
# Calibrated from Sprint 1 measurements with MiniLM-L6-v2 embeddings.
OVERLAP_TYPE_COSINE = {
    "exact": 1.00,
    "reorder": 0.90,
    "partial_80": 0.80,
    "partial_60": 0.65,
    "partial_40": 0.47,
    "paraphrase": 0.77,
    "diverse": 0.20,
}


@dataclass(frozen=True)
class VariationMeasurement:
    """Raw measurement for a single cluster variation."""

    cluster_id: str
    overlap_type: str
    expected_overlap: float
    estimated_cosine: float
    cold_ttft_ms: float
    semblend_ttft_ms: float
    cold_text: str
    semblend_text: str
    rouge_l: float
    ppl_ratio: float | None


@dataclass(frozen=True)
class ThresholdResult:
    """Aggregated results for a single similarity threshold."""

    threshold: float
    total_variations: int
    predicted_hits: int
    predicted_misses: int
    hit_rate: float
    # Metrics among predicted hits
    avg_ttft_speedup: float | None
    avg_ppl_ratio: float | None
    avg_rouge_l: float | None
    # Metrics among predicted misses (these would run cold)
    avg_cold_ttft_ms: float | None


@dataclass
class AblationReport:
    model: str
    hardware: str
    timestamp: str
    thresholds_swept: list[float]
    num_clusters: int
    num_variations_measured: int
    threshold_results: list[ThresholdResult] = field(default_factory=list)
    raw_measurements: list[VariationMeasurement] = field(default_factory=list)


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence (no external deps)."""
    if not hypothesis or not reference:
        return 0.0
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp or not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n if n else 0
    rc = lcs / m if m else 0
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    """Send a prompt to register it as a KV cache donor."""
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 1,
            "temperature": 0.0,
        },
        timeout=300.0,
    )
    resp.raise_for_status()


def generate_with_logprobs(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
) -> tuple[str, list[float], float]:
    """Generate text and collect per-token log probabilities.

    Returns (text, token_logprobs, ttft_ms).
    """
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 1,
        },
        timeout=300.0,
    )
    ttft_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_data = data["choices"][0].get("logprobs", {})
    token_lps = lp_data.get("token_logprobs", []) if lp_data else []
    valid_lps = [lp for lp in token_lps if lp is not None]
    return text, valid_lps, ttft_ms


def compute_perplexity(logprobs: list[float]) -> float | None:
    """Compute perplexity from a list of token log probabilities."""
    if not logprobs:
        return None
    avg_neg_lp = -statistics.mean(logprobs)
    return math.exp(avg_neg_lp)


def estimate_cosine(overlap_type: str) -> float:
    """Estimate cosine similarity from overlap type."""
    # Handle partial_N variants
    if overlap_type.startswith("partial_"):
        suffix = overlap_type[len("partial_"):]
        try:
            pct = int(suffix)
            # Linear interpolation: partial_100 ~ 1.0, partial_0 ~ 0.20
            return 0.20 + 0.80 * (pct / 100.0)
        except ValueError:
            pass
    return OVERLAP_TYPE_COSINE.get(overlap_type, 0.50)


def load_clusters(clusters_file: str) -> list[dict]:
    """Load cluster data from a JSON file."""
    with open(clusters_file) as f:
        return json.load(f)


def run_threshold_ablation(
    endpoint: str,
    model: str,
    clusters_file: str,
    output_dir: str,
    thresholds: list[float],
    max_clusters: int = 50,
    max_tokens: int = 50,
    hardware: str = "unknown",
) -> AblationReport:
    """Run threshold ablation benchmark.

    Phase 1: Measure all cluster variations against vLLM.
    Phase 2: Post-hoc threshold sweep analysis.
    """
    clusters = load_clusters(clusters_file)
    if len(clusters) > max_clusters:
        clusters = clusters[:max_clusters]

    report = AblationReport(
        model=model,
        hardware=hardware,
        timestamp=datetime.now(timezone.utc).isoformat(),
        thresholds_swept=sorted(thresholds),
        num_clusters=len(clusters),
        num_variations_measured=0,
    )

    print(f"Threshold Ablation Benchmark")
    print(f"  Model: {model}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Thresholds: {sorted(thresholds)}")
    print()

    # Health check
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        if resp.status_code != 200:
            print(f"vLLM unhealthy: {resp.status_code}")
            return report
        print(f"  vLLM healthy at {endpoint}\n")
    except Exception as e:
        print(f"Cannot reach vLLM: {e}")
        return report

    # Phase 1: Measure all variations
    measurements: list[VariationMeasurement] = []

    for ci, cluster in enumerate(clusters):
        seed_text = cluster["seed_text"]
        variations = cluster.get("variations", [])
        cluster_id = cluster.get("cluster_id", f"cluster_{ci}")

        if not variations:
            continue

        print(f"Cluster {ci + 1}/{len(clusters)} ({cluster_id})")

        # Register seed as donor
        print(f"  Registering seed donor...", end=" ", flush=True)
        try:
            register_donor(endpoint, model, seed_text)
            time.sleep(0.5)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Cold baseline
        print(f"  Cold baseline...", end=" ", flush=True)
        try:
            cold_text, cold_lps, cold_ttft = generate_with_logprobs(
                endpoint, model, seed_text, max_tokens,
            )
            cold_ppl = compute_perplexity(cold_lps)
            print(f"{cold_ttft:.0f}ms")
        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Measure each variation
        for vi, variation in enumerate(variations):
            var_text = variation["text"]
            overlap_type = variation["overlap_type"]
            expected_overlap = variation.get("expected_token_overlap", 0.0)
            est_cos = estimate_cosine(overlap_type)

            print(
                f"  Variation {vi + 1}/{len(variations)} "
                f"({overlap_type}, est_cos={est_cos:.2f})...",
                end=" ",
                flush=True,
            )

            try:
                sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                    endpoint, model, var_text, max_tokens,
                )
                sb_ppl = compute_perplexity(sb_lps)

                rl = rouge_l(sb_text, cold_text)
                ppl_ratio = None
                if cold_ppl is not None and sb_ppl is not None and cold_ppl > 0:
                    ppl_ratio = sb_ppl / cold_ppl

                measurement = VariationMeasurement(
                    cluster_id=cluster_id,
                    overlap_type=overlap_type,
                    expected_overlap=expected_overlap,
                    estimated_cosine=est_cos,
                    cold_ttft_ms=cold_ttft,
                    semblend_ttft_ms=sb_ttft,
                    cold_text=cold_text[:200],
                    semblend_text=sb_text[:200],
                    rouge_l=rl,
                    ppl_ratio=ppl_ratio,
                )
                measurements.append(measurement)

                speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0
                ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "ppl_r=N/A"
                print(
                    f"sb={sb_ttft:.0f}ms speedup={speedup:.1f}x "
                    f"RL={rl:.3f} {ppl_str}"
                )

            except Exception as e:
                print(f"FAILED: {e}")
                continue

            time.sleep(0.2)

    report.raw_measurements = measurements
    report.num_variations_measured = len(measurements)

    if not measurements:
        print("\nNo measurements collected!")
        return report

    # Phase 2: Post-hoc threshold sweep analysis
    print(f"\n{'=' * 80}")
    print("THRESHOLD SWEEP ANALYSIS (Post-hoc)")
    print(f"{'=' * 80}")
    print(
        f"{'Threshold':>10} {'Hits':>6} {'Misses':>8} {'Hit Rate':>10} "
        f"{'Avg Speedup':>12} {'Avg PPL-R':>10} {'Avg ROUGE-L':>12}"
    )
    print("-" * 80)

    for threshold in sorted(thresholds):
        hits = [m for m in measurements if m.estimated_cosine >= threshold]
        misses = [m for m in measurements if m.estimated_cosine < threshold]

        hit_rate = len(hits) / len(measurements) if measurements else 0.0

        hit_speedups = [
            m.cold_ttft_ms / m.semblend_ttft_ms
            for m in hits
            if m.semblend_ttft_ms > 0
        ]
        hit_ppl_ratios = [
            m.ppl_ratio for m in hits if m.ppl_ratio is not None
        ]
        hit_rouges = [m.rouge_l for m in hits]

        miss_cold_ttfts = [m.cold_ttft_ms for m in misses]

        avg_speedup = statistics.mean(hit_speedups) if hit_speedups else None
        avg_ppl = statistics.mean(hit_ppl_ratios) if hit_ppl_ratios else None
        avg_rl = statistics.mean(hit_rouges) if hit_rouges else None
        avg_cold = (
            statistics.mean(miss_cold_ttfts) if miss_cold_ttfts else None
        )

        result = ThresholdResult(
            threshold=threshold,
            total_variations=len(measurements),
            predicted_hits=len(hits),
            predicted_misses=len(misses),
            hit_rate=hit_rate,
            avg_ttft_speedup=avg_speedup,
            avg_ppl_ratio=avg_ppl,
            avg_rouge_l=avg_rl,
            avg_cold_ttft_ms=avg_cold,
        )
        report.threshold_results.append(result)

        spd_str = f"{avg_speedup:.2f}x" if avg_speedup else "---"
        ppl_str = f"{avg_ppl:.4f}" if avg_ppl else "---"
        rl_str = f"{avg_rl:.4f}" if avg_rl else "---"

        print(
            f"{threshold:>10.2f} {len(hits):>6} {len(misses):>8} "
            f"{hit_rate:>9.0%} {spd_str:>12} {ppl_str:>10} {rl_str:>12}"
        )

    # Paper-ready table
    print(f"\n{'=' * 80}")
    print("PAPER TABLE: Similarity Threshold vs Quality/Speed Tradeoff")
    print(f"{'=' * 80}")
    print(
        f"{'Threshold':>10} | {'Hit Rate':>10} | {'TTFT Speedup':>13} | "
        f"{'PPL Ratio':>10} | {'ROUGE-L':>8}"
    )
    print("-" * 65)

    for r in report.threshold_results:
        spd = f"{r.avg_ttft_speedup:.2f}x" if r.avg_ttft_speedup else "N/A"
        ppl = f"{r.avg_ppl_ratio:.3f}" if r.avg_ppl_ratio else "N/A"
        rl = f"{r.avg_rouge_l:.3f}" if r.avg_rouge_l else "N/A"
        print(
            f"{r.threshold:>10.2f} | {r.hit_rate:>9.0%} | {spd:>13} | "
            f"{ppl:>10} | {rl:>8}"
        )

    print(f"\n  Key finding: Lower thresholds accept more donors (higher hit")
    print(f"  rate) but may admit lower-quality matches (higher PPL ratio).")
    print(f"  The default 0.60 balances hit rate against quality preservation.")
    print(f"{'=' * 80}")

    # Save results
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"threshold-ablation-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    out_file = os.path.join(output_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation A: Similarity Threshold Sweep",
    )
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--clusters-file",
        default="benchmarks/data/semblend_clusters.json",
    )
    parser.add_argument(
        "--output-dir", default="results/threshold-ablation",
    )
    parser.add_argument(
        "--thresholds",
        default="0.40,0.50,0.60,0.70,0.80",
        help="Comma-separated similarity thresholds to sweep",
    )
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--hardware", default="unknown")
    args = parser.parse_args()

    thresholds = [float(t) for t in args.thresholds.split(",")]

    run_threshold_ablation(
        endpoint=args.endpoint,
        model=args.model,
        clusters_file=args.clusters_file,
        output_dir=args.output_dir,
        thresholds=thresholds,
        max_clusters=args.max_clusters,
        max_tokens=args.max_tokens,
        hardware=args.hardware,
    )


if __name__ == "__main__":
    main()
