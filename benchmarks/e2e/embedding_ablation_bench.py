#!/usr/bin/env python3
"""Ablation B: Embedding Model Comparison.

Compares MiniLM-L6-v2 embedding (384-dim) vs Jaccard token-set similarity
for donor matching quality. Shows that learned embeddings capture semantic
similarity (especially paraphrase) that token-level Jaccard misses.

For each cluster variation:
  1. Tokenize seed and variation.
  2. Compute Jaccard similarity locally: |A intersection B| / |A union B|.
  3. Send to vLLM (which uses MiniLM internally) and infer whether SemBlend
     found a donor by comparing TTFT to cold baseline.
  4. Report per variation type: Jaccard similarity vs MiniLM hit/miss.

Key expected finding: MiniLM catches paraphrase matches (Jaccard ~0.3-0.5,
cosine ~0.7-0.85) that Jaccard-only matching would miss entirely.

Usage:
    python -m benchmarks.e2e.embedding_ablation_bench \\
        --endpoint http://localhost:8000 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters-file benchmarks/data/semblend_clusters.json \\
        --output-dir results/embedding-ablation
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

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


# TTFT speedup threshold to classify a request as a SemBlend "hit".
# If TTFT_semblend / TTFT_cold < this ratio, we consider it a hit.
HIT_SPEEDUP_THRESHOLD = 1.3


@dataclass(frozen=True)
class VariationComparison:
    """Per-variation comparison of Jaccard vs MiniLM."""

    cluster_id: str
    overlap_type: str
    expected_token_overlap: float
    jaccard_similarity: float
    cold_ttft_ms: float
    semblend_ttft_ms: float
    ttft_speedup: float
    minilm_hit: bool
    rouge_l: float
    ppl_ratio: float | None


@dataclass(frozen=True)
class OverlapTypeStats:
    """Aggregated stats for a single overlap type."""

    overlap_type: str
    count: int
    avg_jaccard: float
    jaccard_stdev: float | None
    minilm_hit_rate: float
    avg_speedup_on_hit: float | None
    avg_rouge_l: float
    avg_ppl_ratio: float | None
    # Would Jaccard catch this at threshold 0.60?
    jaccard_would_hit_at_060: bool


@dataclass
class EmbeddingAblationReport:
    model: str
    hardware: str
    timestamp: str
    num_clusters: int
    num_variations: int
    overlap_type_stats: list[OverlapTypeStats] = field(default_factory=list)
    raw_comparisons: list[VariationComparison] = field(default_factory=list)


def rouge_l(hypothesis: str, reference: str) -> float:
    """ROUGE-L F1 via longest common subsequence."""
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


def compute_jaccard_from_text(text_a: str, text_b: str) -> float:
    """Compute Jaccard similarity on whitespace-tokenized word sets.

    This is a simpler proxy than token-ID Jaccard (which requires a
    tokenizer) but captures the same signal for ablation purposes.
    """
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = len(words_a & words_b)
    union = len(words_a | words_b)
    return intersection / union if union > 0 else 0.0


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
    """Compute perplexity from token log probabilities."""
    if not logprobs:
        return None
    avg_neg_lp = -statistics.mean(logprobs)
    return math.exp(avg_neg_lp)


def run_embedding_ablation(
    endpoint: str,
    model: str,
    clusters_file: str,
    output_dir: str,
    max_clusters: int = 50,
    max_tokens: int = 50,
    hardware: str = "unknown",
) -> EmbeddingAblationReport:
    """Run embedding model comparison ablation."""
    with open(clusters_file) as f:
        clusters = json.load(f)

    if len(clusters) > max_clusters:
        clusters = clusters[:max_clusters]

    report = EmbeddingAblationReport(
        model=model,
        hardware=hardware,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_clusters=len(clusters),
        num_variations=0,
    )

    print("Embedding Ablation Benchmark: MiniLM vs Jaccard")
    print(f"  Model: {model}")
    print(f"  Clusters: {len(clusters)}")
    print(f"  Hit speedup threshold: {HIT_SPEEDUP_THRESHOLD:.1f}x")
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

    comparisons: list[VariationComparison] = []

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

        for vi, variation in enumerate(variations):
            var_text = variation["text"]
            overlap_type = variation["overlap_type"]
            expected_overlap = variation.get("expected_token_overlap", 0.0)

            # Compute Jaccard similarity locally
            jaccard = compute_jaccard_from_text(seed_text, var_text)

            print(
                f"  {overlap_type:>12} Jaccard={jaccard:.3f}...",
                end=" ",
                flush=True,
            )

            try:
                sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                    endpoint, model, var_text, max_tokens,
                )
                sb_ppl = compute_perplexity(sb_lps)

                speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0
                minilm_hit = speedup >= HIT_SPEEDUP_THRESHOLD

                rl = rouge_l(sb_text, cold_text)
                ppl_ratio = None
                if cold_ppl is not None and sb_ppl is not None and cold_ppl > 0:
                    ppl_ratio = sb_ppl / cold_ppl

                comp = VariationComparison(
                    cluster_id=cluster_id,
                    overlap_type=overlap_type,
                    expected_token_overlap=expected_overlap,
                    jaccard_similarity=jaccard,
                    cold_ttft_ms=cold_ttft,
                    semblend_ttft_ms=sb_ttft,
                    ttft_speedup=speedup,
                    minilm_hit=minilm_hit,
                    rouge_l=rl,
                    ppl_ratio=ppl_ratio,
                )
                comparisons.append(comp)

                hit_tag = "HIT" if minilm_hit else "miss"
                ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "ppl_r=N/A"
                print(
                    f"speedup={speedup:.1f}x [{hit_tag}] "
                    f"RL={rl:.3f} {ppl_str}"
                )

            except Exception as e:
                print(f"FAILED: {e}")
                continue

            time.sleep(0.2)

    report.raw_comparisons = comparisons
    report.num_variations = len(comparisons)

    if not comparisons:
        print("\nNo comparisons collected!")
        return report

    # Aggregate by overlap type
    by_type: dict[str, list[VariationComparison]] = {}
    for c in comparisons:
        by_type.setdefault(c.overlap_type, []).append(c)

    print(f"\n{'=' * 95}")
    print("EMBEDDING ABLATION: MiniLM-L6-v2 vs Jaccard Token-Set Similarity")
    print(f"{'=' * 95}")
    print(
        f"{'Type':>12} | {'N':>4} | {'Jaccard':>9} | {'MiniLM Hit%':>12} | "
        f"{'Avg Speedup':>12} | {'ROUGE-L':>8} | {'PPL Ratio':>10} | "
        f"{'Jaccard@0.60':>12}"
    )
    print("-" * 95)

    for overlap_type in sorted(by_type.keys()):
        items = by_type[overlap_type]
        n = len(items)

        jaccards = [c.jaccard_similarity for c in items]
        avg_jac = statistics.mean(jaccards)
        jac_std = statistics.stdev(jaccards) if n > 1 else None

        hit_count = sum(1 for c in items if c.minilm_hit)
        hit_rate = hit_count / n

        hit_speedups = [c.ttft_speedup for c in items if c.minilm_hit]
        avg_spd = statistics.mean(hit_speedups) if hit_speedups else None

        rouges = [c.rouge_l for c in items]
        avg_rl = statistics.mean(rouges)

        ppl_ratios = [c.ppl_ratio for c in items if c.ppl_ratio is not None]
        avg_ppl = statistics.mean(ppl_ratios) if ppl_ratios else None

        # Would Jaccard >= 0.60 for this type on average?
        jaccard_hits = avg_jac >= 0.60

        stat = OverlapTypeStats(
            overlap_type=overlap_type,
            count=n,
            avg_jaccard=avg_jac,
            jaccard_stdev=jac_std,
            minilm_hit_rate=hit_rate,
            avg_speedup_on_hit=avg_spd,
            avg_rouge_l=avg_rl,
            avg_ppl_ratio=avg_ppl,
            jaccard_would_hit_at_060=jaccard_hits,
        )
        report.overlap_type_stats.append(stat)

        spd_str = f"{avg_spd:.2f}x" if avg_spd else "---"
        ppl_str = f"{avg_ppl:.3f}" if avg_ppl else "---"
        jac_hit_str = "YES" if jaccard_hits else "NO"

        print(
            f"{overlap_type:>12} | {n:>4} | {avg_jac:>8.3f} | "
            f"{hit_rate:>11.0%} | {spd_str:>12} | {avg_rl:>8.3f} | "
            f"{ppl_str:>10} | {jac_hit_str:>12}"
        )

    # Paper-ready summary
    print(f"\n{'=' * 80}")
    print("PAPER TABLE: Embedding-Based vs Token-Set Donor Matching")
    print(f"{'=' * 80}")
    print(
        f"{'Variation Type':>15} | {'Jaccard Sim':>12} | "
        f"{'MiniLM Hit Rate':>16} | {'Jaccard Catches':>16}"
    )
    print("-" * 70)

    for stat in report.overlap_type_stats:
        jac_catches = "Yes" if stat.jaccard_would_hit_at_060 else "No"
        print(
            f"{stat.overlap_type:>15} | {stat.avg_jaccard:>11.3f} | "
            f"{stat.minilm_hit_rate:>15.0%} | {jac_catches:>16}"
        )

    # Highlight the key finding
    paraphrase_stats = [
        s for s in report.overlap_type_stats if s.overlap_type == "paraphrase"
    ]
    if paraphrase_stats:
        ps = paraphrase_stats[0]
        print(f"\n  KEY FINDING: Paraphrase variations have Jaccard={ps.avg_jaccard:.3f}")
        print(f"  but MiniLM achieves {ps.minilm_hit_rate:.0%} hit rate.")
        if not ps.jaccard_would_hit_at_060:
            print(
                f"  Jaccard at threshold 0.60 would MISS these entirely, "
                f"losing {ps.count} potential donor matches."
            )
    print(f"{'=' * 80}")

    # Save
    os.makedirs(output_dir, exist_ok=True)
    run_id = f"embedding-ablation-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    out_file = os.path.join(output_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation B: Embedding Model Comparison (MiniLM vs Jaccard)",
    )
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--clusters-file",
        default="benchmarks/data/semblend_clusters.json",
    )
    parser.add_argument(
        "--output-dir", default="results/embedding-ablation",
    )
    parser.add_argument("--max-clusters", type=int, default=50)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--hardware", default="unknown")
    args = parser.parse_args()

    run_embedding_ablation(
        endpoint=args.endpoint,
        model=args.model,
        clusters_file=args.clusters_file,
        output_dir=args.output_dir,
        max_clusters=args.max_clusters,
        max_tokens=args.max_tokens,
        hardware=args.hardware,
    )


if __name__ == "__main__":
    main()
