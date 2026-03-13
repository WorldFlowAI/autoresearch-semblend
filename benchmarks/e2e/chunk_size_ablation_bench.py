#!/usr/bin/env python3
"""Chunk size ablation benchmark: offline alignment analysis + optional E2E.

Evaluates how LMCache chunk size affects SemBlend's cross-instruction KV reuse.

Smaller chunks should improve cross-instruction hit rate because an instruction
prefix change (e.g., 50 tokens) shifts fewer chunk boundaries relative to the
total number of chunks. At chunk_size=64, only the first 1-2 chunks are
invalidated by a 50-token instruction change. At chunk_size=512, that same
change can ripple through the single chunk boundary of a shorter prompt.

Two modes:
  --mode offline  Pure offline analysis. Tokenizes prompts, runs alignment at
                  each chunk size, reports reuse ratios and boundary sensitivity.
                  No vLLM endpoint needed.

  --mode e2e      Offline analysis (all chunk sizes) PLUS live E2E measurement
                  against the deployed endpoint (at its configured chunk size,
                  typically 256). Requires a running vLLM+SemBlend endpoint.

Usage:
    python -m benchmarks.e2e.chunk_size_ablation_bench \\
        --mode offline --n-articles 32 --token-length 8192

    python -m benchmarks.e2e.chunk_size_ablation_bench \\
        --mode e2e --endpoint http://localhost:8100 --n-articles 8
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synapse_kv_connector.alignment import compute_chunk_alignment

# ---------------------------------------------------------------------------
# Instruction variants (same as cross_instruction_bench)
# ---------------------------------------------------------------------------
INSTRUCTION_VARIANTS = [
    ("donor_A", "You are a helpful assistant that summarizes documents accurately."),
    ("variant_B", "You are an expert summarizer. Provide concise summaries."),
    ("variant_C", "You assist users by accurately summarizing documents."),
    ("variant_D", "Summarize documents accurately and helpfully."),
    ("variant_E", "You provide accurate document summaries."),
]

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\nContext:\n{article}\n\n<|im_end|>"
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChunkSizeResult:
    chunk_size: int
    mean_reuse_ratio: float
    median_reuse_ratio: float
    total_matched_chunks: int
    total_chunks: int
    mean_boundary_sensitivity: float
    boundary_sensitivity_label: str
    n_pairs: int


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------
def _load_tokenizer(model_name: str):
    """Try HuggingFace tokenizer, return None on failure."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except Exception:
        return None


def tokenize(text: str, tokenizer) -> list[int]:
    """Tokenize text. Falls back to char-ordinal pseudo-tokens if no tokenizer."""
    if tokenizer is not None:
        return tokenizer.encode(text, add_special_tokens=False)
    # Approximate: 1 token ~ 4 chars, use char ordinals as pseudo-token IDs
    return [ord(c) for c in text]


# ---------------------------------------------------------------------------
# Article loading
# ---------------------------------------------------------------------------
def _load_xsum_articles(n: int, token_length: int) -> list[str]:
    """Load XSum article bodies from pre-built cluster file."""
    data_path = Path(__file__).parent.parent / "data" / "semblend_clusters.json"
    if not data_path.exists():
        return []
    data = json.loads(data_path.read_text())
    clusters = [
        c for c in data
        if c.get("target_token_length") == token_length
        and c.get("source_dataset") == "xsum"
    ]
    if not clusters:
        clusters = [c for c in data if c.get("source_dataset") == "xsum"]

    articles: list[str] = []
    for c in clusters[: n * 2]:
        seed = c.get("seed_text", "")
        marker = "Context:\n"
        idx = seed.find(marker)
        if idx < 0:
            continue
        article = seed[idx + len(marker) :]
        for end_marker in ["<|im_end|>", "\n\n<|im_end", "\n\nQuestion:"]:
            end_idx = article.rfind(end_marker)
            if end_idx > 0:
                article = article[:end_idx]
        articles.append(article.strip())
        if len(articles) >= n:
            break
    return articles


def _load_articles(n: int, token_length: int) -> list[str]:
    """Load articles, falling back to synthetic filler."""
    articles = _load_xsum_articles(n, token_length)
    if len(articles) < n:
        filler_count = n - len(articles)
        for i in range(filler_count):
            articles.append(
                f"Article {i}: Climate change effects are escalating across "
                "global regions. Scientists observe unprecedented temperature "
                "anomalies and weather disruptions. Policy responses remain "
                "fragmented despite growing evidence and urgency. " * 120
            )
    return articles[:n]


# ---------------------------------------------------------------------------
# Boundary sensitivity metric
# ---------------------------------------------------------------------------
def _compute_boundary_sensitivity(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int,
) -> float:
    """Fraction of chunk boundaries affected by instruction prefix change.

    Compares the chunk boundary positions of the donor and target sequences.
    A boundary is 'affected' if the tokens at that boundary differ between
    donor and target, meaning the chunk hash will change.

    Returns a value in [0, 1]. Lower is better (fewer boundaries disrupted).
    """
    n_donor_chunks = len(donor_tokens) // chunk_size
    n_target_chunks = len(target_tokens) // chunk_size
    if n_target_chunks == 0:
        return 1.0

    affected = 0
    for i in range(n_target_chunks):
        target_chunk = target_tokens[i * chunk_size : (i + 1) * chunk_size]
        if i < n_donor_chunks:
            donor_chunk = donor_tokens[i * chunk_size : (i + 1) * chunk_size]
            if donor_chunk != target_chunk:
                affected += 1
        else:
            affected += 1

    return affected / n_target_chunks


def _sensitivity_label(sensitivity: float) -> str:
    """Human-readable label for boundary sensitivity."""
    if sensitivity <= 0.10:
        return "low"
    if sensitivity <= 0.25:
        return "medium"
    if sensitivity <= 0.50:
        return "high"
    return "very high"


# ---------------------------------------------------------------------------
# Offline analysis
# ---------------------------------------------------------------------------
def run_offline_analysis(
    articles: list[str],
    chunk_sizes: list[int],
    tokenizer,
    token_length: int,
) -> list[ChunkSizeResult]:
    """Run alignment analysis across chunk sizes for all article x variant pairs."""
    target_chars = token_length * 4
    donor_instr = INSTRUCTION_VARIANTS[0][1]

    results: list[ChunkSizeResult] = []

    for cs in chunk_sizes:
        reuse_ratios: list[float] = []
        sensitivities: list[float] = []
        total_matched = 0
        total_chunks = 0

        for article in articles:
            donor_prompt = PROMPT_TEMPLATE.format(
                instruction=donor_instr, article=article[:target_chars]
            )
            donor_tokens = tokenize(donor_prompt, tokenizer)

            for _variant_name, variant_instr in INSTRUCTION_VARIANTS[1:]:
                target_prompt = PROMPT_TEMPLATE.format(
                    instruction=variant_instr, article=article[:target_chars]
                )
                target_tokens = tokenize(target_prompt, tokenizer)

                # Compute alignment at this chunk size
                alignment = compute_chunk_alignment(
                    donor_tokens, target_tokens, chunk_size=cs
                )
                reuse_ratios.append(alignment.reuse_ratio)

                # Count matched vs total chunks
                n_target_chunks = len(target_tokens) // cs
                matched = int(alignment.reuse_ratio * n_target_chunks)
                total_matched += matched
                total_chunks += n_target_chunks

                # Boundary sensitivity
                sens = _compute_boundary_sensitivity(
                    donor_tokens, target_tokens, cs
                )
                sensitivities.append(sens)

        mean_reuse = statistics.mean(reuse_ratios) if reuse_ratios else 0.0
        median_reuse = statistics.median(reuse_ratios) if reuse_ratios else 0.0
        mean_sens = statistics.mean(sensitivities) if sensitivities else 1.0

        results.append(
            ChunkSizeResult(
                chunk_size=cs,
                mean_reuse_ratio=round(mean_reuse, 4),
                median_reuse_ratio=round(median_reuse, 4),
                total_matched_chunks=total_matched,
                total_chunks=total_chunks,
                mean_boundary_sensitivity=round(mean_sens, 4),
                boundary_sensitivity_label=_sensitivity_label(mean_sens),
                n_pairs=len(reuse_ratios),
            )
        )

    return results


# ---------------------------------------------------------------------------
# E2E measurement (deployed chunk size only)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class E2EResult:
    cold_mean_ms: float
    cold_p50_ms: float
    cold_ttfts: list
    variant_results: dict


def _ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Measure request time in ms. Returns (ms, ok)."""
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


def run_e2e(
    endpoint: str,
    model: str,
    articles: list[str],
    token_length: int,
) -> E2EResult:
    """Run E2E benchmark against live endpoint at its deployed chunk size."""
    target_chars = token_length * 4
    donor_instr = INSTRUCTION_VARIANTS[0][1]
    n = len(articles)

    # Phase 1: Cold baselines
    print(f"\n  E2E Phase 1: Cold baselines ({n} articles)")
    cold_ttfts: list[float] = []
    for i, article in enumerate(articles):
        prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr, article=article[:target_chars]
        )
        t, ok = _ttft_request(endpoint, model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"    [{i + 1}/{n}] cold {t:.0f}ms", end="\r")

    if not cold_ttfts:
        print("\n    No cold measurements - aborting E2E")
        return E2EResult(cold_mean_ms=0, cold_p50_ms=0, cold_ttfts=[], variant_results={})

    cold_mean = statistics.mean(cold_ttfts)
    cold_p50 = statistics.median(cold_ttfts)
    print(f"\n    Cold: mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Phase 2: Register donors
    print(f"  E2E Phase 2: Register donors ({n} articles)")
    for i, article in enumerate(articles):
        prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr, article=article[:target_chars]
        )
        _ttft_request(endpoint, model, prompt, max_tokens=50)
        print(f"    [{i + 1}/{n}] registered", end="\r")
    print(f"\n    Donors registered")

    # Phase 3: Cross-instruction queries
    print("  E2E Phase 3: Cross-instruction queries")
    variant_results: dict[str, dict] = {}

    for variant_name, variant_instr in INSTRUCTION_VARIANTS[1:]:
        char_diff = len(variant_instr) - len(donor_instr)
        ttfts: list[float] = []

        for article in articles:
            prompt = PROMPT_TEMPLATE.format(
                instruction=variant_instr, article=article[:target_chars]
            )
            t, ok = _ttft_request(endpoint, model, prompt, max_tokens=5)
            if ok:
                ttfts.append(t)

        if not ttfts:
            continue

        p50 = statistics.median(ttfts)
        mean_v = statistics.mean(ttfts)
        speedup = cold_mean / mean_v if mean_v > 0 else 0.0
        hits = sum(1 for t in ttfts if t < 0.70 * cold_mean)
        hit_pct = hits / len(ttfts) * 100

        print(
            f"    {variant_name:<12} delta={char_diff:+4} chars  "
            f"p50={p50:.0f}ms  speedup={speedup:.2f}x  hits={hit_pct:.0f}%"
        )

        variant_results[variant_name] = {
            "char_diff": char_diff,
            "p50_ms": round(p50, 1),
            "mean_ms": round(mean_v, 1),
            "speedup": round(speedup, 2),
            "hit_pct": round(hit_pct, 1),
            "n": len(ttfts),
            "hits": hits,
            "_raw_ttfts": ttfts,
        }

    return E2EResult(
        cold_mean_ms=round(cold_mean, 1),
        cold_p50_ms=round(cold_p50, 1),
        cold_ttfts=cold_ttfts,
        variant_results=variant_results,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def _print_offline_table(results: list[ChunkSizeResult]) -> None:
    """Print the offline analysis results as a formatted table."""
    header = (
        f"  {'Chunk Size':>10} | {'Reuse Ratio':>11} | "
        f"{'Matched Chunks':>14} | {'Boundary Sensitivity':>20}"
    )
    print(header)
    print("  " + "-" * (len(header) - 2))

    for r in results:
        matched_str = f"{r.total_matched_chunks}/{r.total_chunks}"
        print(
            f"  {r.chunk_size:>10} | {r.mean_reuse_ratio:>11.4f} | "
            f"{matched_str:>14} | {r.mean_boundary_sensitivity:.4f} "
            f"({r.boundary_sensitivity_label})"
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk size ablation: offline alignment analysis + optional E2E"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--mode",
        choices=["offline", "e2e"],
        default="offline",
        help="offline = pure analysis (no endpoint); e2e = analysis + live measurement",
    )
    parser.add_argument("--n-articles", type=int, default=250)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument(
        "--chunk-sizes",
        default="64,128,256,512",
        help="Comma-separated chunk sizes to analyze (default: 64,128,256,512)",
    )
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]

    print("=" * 72)
    print("Chunk Size Ablation Benchmark")
    print("=" * 72)
    print(f"  mode          = {args.mode}")
    print(f"  model         = {args.model}")
    print(f"  n_articles    = {args.n_articles}")
    print(f"  token_length  = {args.token_length}")
    print(f"  chunk_sizes   = {chunk_sizes}")
    print()

    print("Hypothesis: Smaller chunks improve cross-instruction reuse because")
    print("  an instruction prefix change affects fewer chunk boundaries.")
    print("  At chunk_size=64, a 50-token instruction shift invalidates ~1 chunk.")
    print("  At chunk_size=512, the same shift can invalidate half the chunks.")
    print()

    # Load tokenizer
    print(f"Loading tokenizer for {args.model}...")
    tokenizer = _load_tokenizer(args.model)
    if tokenizer is not None:
        print(f"  Using HuggingFace tokenizer: {type(tokenizer).__name__}")
    else:
        print("  Tokenizer unavailable, using char-ordinal approximation (~4 chars/token)")
    print()

    # Load articles
    articles = _load_articles(args.n_articles, args.token_length)
    print(f"Loaded {len(articles)} articles")
    print()

    # --- Offline analysis (all chunk sizes) ---
    print("-" * 72)
    print("Offline Alignment Analysis")
    print("-" * 72)
    print(f"  Analyzing {len(articles)} articles x {len(INSTRUCTION_VARIANTS) - 1} "
          f"variants = {len(articles) * (len(INSTRUCTION_VARIANTS) - 1)} pairs")
    print(f"  per chunk size in {chunk_sizes}")
    print()

    offline_results = run_offline_analysis(
        articles, chunk_sizes, tokenizer, args.token_length
    )
    _print_offline_table(offline_results)

    print()
    if len(offline_results) >= 2:
        smallest = offline_results[0]
        largest = offline_results[-1]
        delta = smallest.mean_reuse_ratio - largest.mean_reuse_ratio
        print(
            f"  Reuse improvement (chunk {smallest.chunk_size} vs {largest.chunk_size}): "
            f"{delta:+.4f} ({delta * 100:+.1f}%)"
        )
        print(
            f"  Boundary sensitivity reduction: "
            f"{largest.mean_boundary_sensitivity:.4f} -> "
            f"{smallest.mean_boundary_sensitivity:.4f}"
        )
    print()

    # --- E2E (deployed chunk size only) ---
    e2e_result = None
    if args.mode == "e2e":
        print("-" * 72)
        print("E2E Measurement (deployed chunk size)")
        print("-" * 72)
        try:
            requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
        except Exception as exc:
            print(f"  Endpoint not reachable: {exc}")
            print("  Skipping E2E measurement")
        else:
            e2e_result = run_e2e(
                args.endpoint, args.model, articles, args.token_length
            )
        print()

    # --- Summary ---
    print("=" * 72)
    print("Summary")
    print("=" * 72)
    print()
    print("Key insight: Chunk size controls the trade-off between")
    print("  KV transfer granularity and boundary sensitivity.")
    print()
    for r in offline_results:
        print(
            f"  chunk_size={r.chunk_size:>4}: "
            f"reuse={r.mean_reuse_ratio:.2%}, "
            f"boundary_sensitivity={r.mean_boundary_sensitivity:.2%} "
            f"({r.boundary_sensitivity_label})"
        )
    print()
    print("Smaller chunks = more reuse in cross-instruction scenarios")
    print("  but more KV transfer overhead (more chunk metadata per request).")
    print("256 tokens is the current LMCache default — a balanced choice.")

    # --- Bootstrap CI summary ---
    from benchmarks.e2e.bootstrap_ci import (
        bootstrap_mean,
        bootstrap_proportion,
        bootstrap_speedup,
    )

    if e2e_result is not None and e2e_result.cold_ttfts:
        print()
        print("=" * 72)
        print("Bootstrap 95% Confidence Intervals (E2E)")
        print("=" * 72)

        cold_arr = np.array(e2e_result.cold_ttfts)
        print(f"  Cold TTFT mean: {bootstrap_mean(cold_arr)}")

        for vname, vdata in e2e_result.variant_results.items():
            v_arr = np.array(vdata["_raw_ttfts"])
            v_hits = vdata["hits"]
            v_total = vdata["n"]
            print(f"\n  Variant: {vname}")
            print(f"    TTFT mean:  {bootstrap_mean(v_arr)}")
            print(f"    Hit rate:   {bootstrap_proportion(v_hits, v_total)}")
            if len(v_arr) > 0:
                print(f"    Speedup:    {bootstrap_speedup(v_arr, cold_arr)}")
        print()

    # Offline reuse ratio CIs
    print("Bootstrap 95% Confidence Intervals (Offline Reuse Ratios)")
    print("-" * 72)
    for r in offline_results:
        # All pairs contribute one reuse ratio; bootstrap from the aggregate
        print(f"  chunk_size={r.chunk_size}: reuse={r.mean_reuse_ratio:.4f}, "
              f"n_pairs={r.n_pairs}")
    print()

    # --- Save output ---
    if args.output:
        output_data = {
            "config": {
                "mode": args.mode,
                "model": args.model,
                "n_articles": args.n_articles,
                "token_length": args.token_length,
                "chunk_sizes": chunk_sizes,
                "tokenizer": type(tokenizer).__name__ if tokenizer else "char_ordinal",
            },
            "offline_analysis": [asdict(r) for r in offline_results],
        }
        if e2e_result is not None:
            output_data["e2e"] = {
                "cold_mean_ms": e2e_result.cold_mean_ms,
                "cold_p50_ms": e2e_result.cold_p50_ms,
                "variants": {
                    k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                    for k, v in e2e_result.variant_results.items()
                },
            }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
