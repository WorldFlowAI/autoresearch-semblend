#!/usr/bin/env python3
"""Realistic cross-instruction benchmark: same document, DIVERSE instructions.

Unlike cross_instruction_bench.py which uses trivially similar instruction
phrasings (all summarization variants), this benchmark tests genuinely
different task instructions: summarize, extract facts, generate questions,
restyle, critique, etc.

Key insight: 100% hit rates from trivially similar instructions are artifacts
of nearly identical embeddings. Real workloads have diverse instructions that
produce genuinely different embedding vectors. This benchmark measures
SemBlend's hit rate and speedup under realistic instruction diversity.

Instruction categories span ~10-80 words with fundamentally different tasks:
  - donor:           Short summarization (baseline, ~15 words)
  - summarize_long:  Detailed summarization with structure (~80 words)
  - extract_facts:   Fact extraction as numbered list (~30 words)
  - generate_questions: Quiz generation at multiple cognitive levels (~50 words)
  - translate_style: Tone/style rewriting (~35 words)
  - critique:        Critical analysis with structured feedback (~40 words)
  - short_answer:    One-sentence summary (~5 words)

Expected results:
  - Hit rates will vary by instruction similarity to donor
  - Short/similar instructions (short_answer) may still hit
  - Long/divergent instructions (generate_questions) likely miss
  - This reveals the REAL operating envelope of semantic routing

Usage:
    python -m benchmarks.e2e.realistic_cross_instruction_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-articles 32 \
        --token-length 8192
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Diverse instruction sets — fundamentally different tasks, not paraphrases
INSTRUCTION_SETS = {
    "donor": (
        "You are a helpful assistant. Summarize the following document "
        "concisely, focusing on the main points and key conclusions."
    ),
    "summarize_long": (
        "You are an expert document analyst with deep expertise in extracting "
        "key information from complex texts. Your task is to provide a "
        "comprehensive yet concise summary of the following document. Focus "
        "on: (1) the main thesis or central argument, (2) key supporting "
        "evidence or data points, (3) important conclusions or "
        "recommendations, and (4) any notable limitations or caveats "
        "mentioned. Structure your summary with clear topic sentences and "
        "maintain the logical flow of the original document. Aim for "
        "approximately 200-300 words."
    ),
    "extract_facts": (
        "You are a fact extraction specialist. From the following document, "
        "extract ALL factual claims, statistics, dates, names, and specific "
        "assertions. Present them as a numbered list. Do not interpret or "
        "summarize — only extract verbatim or near-verbatim factual statements."
    ),
    "generate_questions": (
        "You are a quiz master creating educational assessment questions. "
        "Based on the following document, generate 10 thoughtful questions "
        "that test comprehension at different cognitive levels: 3 factual "
        "recall questions, 3 inference questions, 2 analysis questions, and "
        "2 evaluation questions. For each question, also provide the correct "
        "answer."
    ),
    "translate_style": (
        "You are a professional editor. Rewrite the following document in a "
        "completely different style: convert it from its current "
        "formal/academic tone into casual, conversational language suitable "
        "for a blog post aimed at a general audience. Maintain all the key "
        "information but make it engaging and accessible."
    ),
    "critique": (
        "You are a critical reviewer. Analyze the following document for "
        "logical consistency, potential biases, unsupported claims, and "
        "methodological weaknesses. Provide a structured critique with: "
        "(1) strengths, (2) weaknesses, (3) missing perspectives, and "
        "(4) suggestions for improvement."
    ),
    "short_answer": "Summarize in one sentence.",
}

ARTICLE_CONTEXT_TEMPLATE = (
    "<|im_start|>system\n{instruction}<|im_end|>\n"
    "<|im_start|>user\nContext:\n{article}\n\n<|im_end|>"
)


def load_xsum_articles(n: int, token_length: int = 8192) -> list[str]:
    """Load diverse XSum article bodies from pre-built cluster files."""
    data_path = Path(__file__).parent.parent / "data" / "semblend_clusters.json"
    if not data_path.exists():
        return []
    data = json.loads(data_path.read_text())
    # Find clusters matching the target token length
    clusters = [
        c for c in data
        if c.get("target_token_length") == token_length
        and c.get("source_dataset") == "xsum"
    ]
    if not clusters:
        # Try any token length
        clusters = [c for c in data if c.get("source_dataset") == "xsum"]
    articles: list[str] = []
    for c in clusters[:n * 2]:
        seed = c.get("seed_text", "")
        # Extract article body (after "Context:\n")
        marker = "Context:\n"
        idx = seed.find(marker)
        if idx >= 0:
            article = seed[idx + len(marker):]
            # Remove trailing template markers
            for end_marker in ["<|im_end|>", "\n\n<|im_end", "\n\nQuestion:"]:
                end_idx = article.rfind(end_marker)
                if end_idx > 0:
                    article = article[:end_idx]
            articles.append(article.strip())
        if len(articles) >= n:
            break
    return articles


def build_prompt(instruction: str, article: str, max_chars: int) -> str:
    """Build a chat-formatted prompt."""
    truncated = article[:max_chars]
    return ARTICLE_CONTEXT_TEMPLATE.format(
        instruction=instruction, article=truncated
    )


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Measure total request time (includes decode). Returns (ms, ok)."""
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


def compute_percentile(values: list[float], pct: int) -> float:
    """Compute a percentile from a sorted list."""
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * pct / 100)
    return sorted_v[min(idx, len(sorted_v) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Realistic cross-instruction benchmark with diverse tasks"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-articles", type=int, default=200)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument("--output", default=None,
                        help="JSON output path for results")
    args = parser.parse_args()

    try:
        requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Endpoint health check failed: {exc}")
        sys.exit(1)

    target_chars = args.token_length * 4  # ~4 chars/token

    print()
    print("Realistic Cross-Instruction Benchmark (Diverse Tasks)")
    print("=" * 65)
    print(f"  endpoint     = {args.endpoint}")
    print(f"  model        = {args.model}")
    print(f"  n_articles   = {args.n_articles}")
    print(f"  token_length = {args.token_length}")
    print()
    print("Key question: Does semantic routing hit when instructions are")
    print("genuinely different tasks (not just paraphrases)?")
    print()

    # Load articles
    articles = load_xsum_articles(args.n_articles * 2, args.token_length)
    if len(articles) < args.n_articles:
        shortfall = args.n_articles * 2 - len(articles)
        print(f"  Note: Only {len(articles)} XSum articles found, "
              f"adding {shortfall} synthetic articles")
        articles += [
            f"Article {i}: Climate change effects are escalating across "
            "global regions. Scientists observe unprecedented temperature "
            "anomalies and weather disruptions. Policy responses remain "
            "fragmented despite growing evidence and urgency. "
            "International cooperation faces significant challenges as "
            "nations balance economic growth with environmental targets. "
            "Recent studies highlight accelerating ice sheet melt in polar "
            "regions, rising sea levels threatening coastal communities, "
            "and shifting agricultural zones affecting food security. " * 80
            for i in range(shortfall)
        ]
    articles = articles[:args.n_articles]
    print(f"  Loaded {len(articles)} articles")

    # Print instruction diversity overview
    donor_instr = INSTRUCTION_SETS["donor"]
    print()
    print("Instruction diversity:")
    print(f"  {'Name':<20} {'Chars':>6} {'Words':>6}  First 60 chars")
    print("  " + "-" * 75)
    for name, instr in INSTRUCTION_SETS.items():
        word_count = len(instr.split())
        tag = " (donor)" if name == "donor" else ""
        print(f"  {name:<20} {len(instr):>6} {word_count:>6}  "
              f"'{instr[:60]}...'{tag}")
    print()

    # ------------------------------------------------------------------
    # Phase 1: Cold baselines
    # ------------------------------------------------------------------
    print(f"Phase 1: Cold baselines ({args.n_articles} articles, donor instruction)")
    cold_ttfts: list[float] = []
    for i, article in enumerate(articles):
        prompt = build_prompt(donor_instr, article, target_chars)
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"  [{i + 1}/{args.n_articles}] cold {t:.0f}ms", end="\r")

    if not cold_ttfts:
        print("\nNo cold measurements succeeded - aborting")
        sys.exit(1)

    cold_p50 = compute_percentile(cold_ttfts, 50)
    cold_mean = statistics.mean(cold_ttfts)
    cold_stdev = statistics.stdev(cold_ttfts) if len(cold_ttfts) > 1 else 0.0
    print(f"\n  Cold baseline: n={len(cold_ttfts)}, mean={cold_mean:.0f}ms, "
          f"p50={cold_p50:.0f}ms, stdev={cold_stdev:.0f}ms")

    # ------------------------------------------------------------------
    # Phase 2: Register donors
    # ------------------------------------------------------------------
    print()
    print(f"Phase 2: Register donors ({args.n_articles} articles, "
          f"donor instruction)")
    for i, article in enumerate(articles):
        prompt = build_prompt(donor_instr, article, target_chars)
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=50)
        print(f"  [{i + 1}/{args.n_articles}] donor registered {t:.0f}ms",
              end="\r")
    print(f"\n  Donors registered")

    # ------------------------------------------------------------------
    # Phase 3: Query with diverse instructions
    # ------------------------------------------------------------------
    print()
    print("Phase 3: Query with diverse instructions")
    print()

    header = (f"  {'Instruction':<20} {'Chars':>6} {'Words':>6} "
              f"{'Hit%':>6} {'P50 ms':>9} {'Mean ms':>9} "
              f"{'Speedup':>8}  Interpretation")
    print(header)
    print("  " + "-" * 100)

    variant_names = [k for k in INSTRUCTION_SETS if k != "donor"]
    results: dict[str, dict] = {}

    for variant_name in variant_names:
        variant_instr = INSTRUCTION_SETS[variant_name]
        char_diff = len(variant_instr) - len(donor_instr)
        word_count = len(variant_instr.split())
        variant_ttfts: list[float] = []

        for article in articles:
            prompt = build_prompt(variant_instr, article, target_chars)
            t, ok = ttft_request(args.endpoint, args.model, prompt,
                                 max_tokens=5)
            if ok:
                variant_ttfts.append(t)

        if not variant_ttfts:
            print(f"  {variant_name:<20} -- all requests failed --")
            continue

        p50 = compute_percentile(variant_ttfts, 50)
        mean_v = statistics.mean(variant_ttfts)
        speedup = cold_mean / mean_v if mean_v > 0 else 0.0

        # Hit detection: TTFT < 70% of cold mean -> likely a semantic hit
        hits = sum(1 for t in variant_ttfts if t < 0.70 * cold_mean)
        hit_pct = hits / len(variant_ttfts) * 100

        # Interpretation based on hit rate and speedup
        if hit_pct >= 80:
            interp = "HIGH hit - instruction similarity sufficient"
        elif hit_pct >= 40:
            interp = "PARTIAL hit - borderline similarity"
        elif speedup > 1.3:
            interp = "LOW hit but some speedup"
        else:
            interp = "MISS - instructions too divergent"

        print(f"  {variant_name:<20} {len(variant_instr):>6} {word_count:>6} "
              f"{hit_pct:>5.0f}% {p50:>8.0f}ms {mean_v:>8.0f}ms "
              f"{speedup:>7.2f}x  {interp}")

        results[variant_name] = {
            "instruction": variant_instr,
            "instruction_chars": len(variant_instr),
            "instruction_words": word_count,
            "char_diff_from_donor": char_diff,
            "n": len(variant_ttfts),
            "ttfts_ms": variant_ttfts,
            "p50_ms": p50,
            "mean_ms": mean_v,
            "speedup_vs_cold": round(speedup, 3),
            "hit_count": hits,
            "hit_pct": round(hit_pct, 1),
            "interpretation": interp,
        }

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 65)
    print("Summary")
    print("=" * 65)

    if results:
        hit_rates = [r["hit_pct"] for r in results.values()]
        speedups = [r["speedup_vs_cold"] for r in results.values()]
        avg_hit = statistics.mean(hit_rates)
        avg_speedup = statistics.mean(speedups)
        high_hit = sum(1 for h in hit_rates if h >= 80)
        partial_hit = sum(1 for h in hit_rates if 40 <= h < 80)
        miss = sum(1 for h in hit_rates if h < 40)

        print(f"  Instruction variants tested: {len(results)}")
        print(f"  Mean hit rate:    {avg_hit:.1f}%")
        print(f"  Mean speedup:     {avg_speedup:.2f}x")
        print(f"  High hit (>=80%): {high_hit}/{len(results)}")
        print(f"  Partial (40-80%): {partial_hit}/{len(results)}")
        print(f"  Miss (<40%):      {miss}/{len(results)}")
        print()
        print("Interpretation:")
        if avg_hit > 80:
            print("  Semantic routing is robust to diverse instructions.")
            print("  Document content dominates the embedding, not instruction text.")
        elif avg_hit > 40:
            print("  Mixed results: some instruction types route successfully,")
            print("  others diverge too much. Instruction length/type matters.")
        else:
            print("  Low hit rates confirm that trivially similar instructions")
            print("  inflate cross-instruction hit rates. Real workloads need")
            print("  instruction-aware routing or instruction stripping.")
    else:
        print("  No results collected.")

    print()
    print("Comparison to cross_instruction_bench.py (paraphrase-only):")
    print("  That benchmark uses trivially similar summarization phrasings")
    print("  which produce nearly identical embeddings -> 100% hit rates.")
    print("  This benchmark uses genuinely different tasks to reveal the")
    print("  real operating envelope of semantic routing.")

    # ------------------------------------------------------------------
    # Bootstrap CI summary
    # ------------------------------------------------------------------
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

    for variant_name, vr in results.items():
        v_arr = np.array(vr["ttfts_ms"])
        v_hits = vr["hit_count"]
        v_total = vr["n"]
        print(f"\n  Variant: {variant_name}")
        print(f"    TTFT mean:  {bootstrap_mean(v_arr)}")
        print(f"    Hit rate:   {bootstrap_proportion(v_hits, v_total)}")
        if len(cold_arr) > 0 and len(v_arr) > 0:
            print(f"    Speedup:    {bootstrap_speedup(v_arr, cold_arr)}")
    print()

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    if args.output:
        output_data = {
            "benchmark": "realistic_cross_instruction",
            "endpoint": args.endpoint,
            "model": args.model,
            "n_articles": args.n_articles,
            "token_length": args.token_length,
            "cold_baseline": {
                "n": len(cold_ttfts),
                "mean_ms": round(cold_mean, 1),
                "p50_ms": round(cold_p50, 1),
                "stdev_ms": round(cold_stdev, 1),
                "ttfts_ms": cold_ttfts,
            },
            "donor_instruction": donor_instr,
            "variants": {
                k: {key: val for key, val in v.items() if key != "ttfts_ms"}
                for k, v in results.items()
            },
            "variants_with_ttfts": results,
        }
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
