#!/usr/bin/env python3
"""PartialAttention shifted-prefix benchmark: demonstrates Δ≠0 scenarios.

Tests scenarios where SemBlend's chunk-swap injection (which guarantees Δ=0)
cannot reuse KV, but a hypothetical PartialAttention path (with RoPE Δ≠0
correction) could. This benchmark documents the current limitation and
quantifies the lost reuse opportunity.

Scenarios:
  1. Instruction-shift: same document, different-length instruction prefix.
     The document tokens are identical but shifted by the instruction length
     difference → every LMCache chunk boundary is misaligned → 0% chunk hit.
  2. Header-shift: same document body, different header/metadata block.
     All body tokens are identical but shifted → chunk boundary mismatch.
  3. Padding-shift: identical content with different amounts of system
     prompt padding, shifting all subsequent tokens.

For each scenario, we measure:
  - Offline alignment reuse ratio at chunk level (current: should be ~0%)
  - Offline alignment reuse ratio at token level via Levenshtein (potential)
  - Live E2E TTFT to confirm chunk-swap gets no hit

This data supports the paper's "Limitations" section and motivates the
PartialAttention + RoPE correction path as future work.

Usage:
    python -m benchmarks.e2e.partial_attention_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-samples 32 \
        --token-length 4096 \
        --output results/partial_attention.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from synapse_kv_connector.alignment import (
    LMCACHE_CHUNK_SIZE,
    _fallback_prefix_alignment,
    compute_alignment,
    compute_chunk_alignment,
)


# ---------------------------------------------------------------------------
# Scenario generators
# ---------------------------------------------------------------------------

def _make_document_tokens(length: int, seed: int = 42) -> list[int]:
    """Generate deterministic pseudo-document token IDs."""
    import random
    rng = random.Random(seed)
    return [rng.randint(100, 30000) for _ in range(length)]


def generate_instruction_shift_pairs(
    n: int, doc_length: int, shift_range: tuple[int, int] = (10, 200),
) -> list[dict]:
    """Same document, different-length instruction prefix → shifted tokens."""
    import random
    rng = random.Random(42)
    pairs = []

    for i in range(n):
        doc = _make_document_tokens(doc_length, seed=1000 + i)
        instr_len_a = rng.randint(20, 60)
        shift = rng.randint(shift_range[0], shift_range[1])
        instr_len_b = instr_len_a + shift

        instr_a = list(range(1, instr_len_a + 1))
        instr_b = list(range(1, instr_len_b + 1))

        donor = instr_a + doc[:doc_length - instr_len_a]
        target = instr_b + doc[:doc_length - instr_len_b]

        pairs.append({
            "name": f"instr_shift_{i}",
            "scenario": "instruction_shift",
            "shift_tokens": shift,
            "donor": donor,
            "target": target,
        })

    return pairs


def generate_header_shift_pairs(
    n: int, doc_length: int,
) -> list[dict]:
    """Same body, different header block → body tokens shifted."""
    import random
    rng = random.Random(99)
    pairs = []

    for i in range(n):
        body = _make_document_tokens(doc_length - 200, seed=2000 + i)
        header_len_a = rng.randint(50, 150)
        header_len_b = rng.randint(50, 150)
        while header_len_b == header_len_a:
            header_len_b = rng.randint(50, 150)

        header_a = [50000 + j for j in range(header_len_a)]
        header_b = [60000 + j for j in range(header_len_b)]

        donor = header_a + body[:doc_length - header_len_a]
        target = header_b + body[:doc_length - header_len_b]

        pairs.append({
            "name": f"header_shift_{i}",
            "scenario": "header_shift",
            "shift_tokens": abs(header_len_b - header_len_a),
            "donor": donor,
            "target": target,
        })

    return pairs


def generate_padding_shift_pairs(
    n: int, doc_length: int,
) -> list[dict]:
    """Identical content with different system prompt padding amounts."""
    import random
    rng = random.Random(77)
    pairs = []

    for i in range(n):
        content = _make_document_tokens(doc_length - 300, seed=3000 + i)
        pad_a = rng.randint(50, 200)
        pad_b = rng.randint(50, 200)
        while pad_b == pad_a:
            pad_b = rng.randint(50, 200)

        # Padding is a repeated token (like whitespace)
        donor = [1] * pad_a + content[:doc_length - pad_a]
        target = [1] * pad_b + content[:doc_length - pad_b]

        pairs.append({
            "name": f"padding_shift_{i}",
            "scenario": "padding_shift",
            "shift_tokens": abs(pad_b - pad_a),
            "donor": donor,
            "target": target,
        })

    return pairs


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_offline(pairs: list[dict]) -> list[dict]:
    """Run offline alignment analysis for each pair."""
    results = []

    for pair in pairs:
        donor = pair["donor"]
        target = pair["target"]

        # Chunk-level alignment (current production path)
        chunk_result = compute_chunk_alignment(donor, target)

        # Token-level alignment (Levenshtein or fallback prefix)
        token_result = compute_alignment(donor, target)

        # Fallback prefix alignment (for comparison)
        prefix_result = _fallback_prefix_alignment(donor, target)

        results.append({
            "name": pair["name"],
            "scenario": pair["scenario"],
            "shift_tokens": pair["shift_tokens"],
            "donor_len": len(donor),
            "target_len": len(target),
            "chunk_reuse_ratio": chunk_result.reuse_ratio,
            "chunk_edit_distance": chunk_result.edit_distance,
            "token_reuse_ratio": token_result.reuse_ratio,
            "token_edit_distance": token_result.edit_distance,
            "prefix_reuse_ratio": prefix_result.reuse_ratio,
            "prefix_match_len": len(target) - prefix_result.edit_distance,
            "lost_reuse_tokens": int(
                (token_result.reuse_ratio - chunk_result.reuse_ratio)
                * len(target)
            ),
        })

    return results


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5,
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


PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{content}<|im_end|>"
)


def run_e2e(
    pairs: list[dict],
    endpoint: str,
    model: str,
    n_cold: int = 8,
) -> dict:
    """Run live E2E measurements for a subset of pairs."""
    # Cold baseline
    cold_ttfts: list[float] = []
    for i, pair in enumerate(pairs[:n_cold]):
        # Use donor as cold baseline (unique content)
        text = "".join(str(t) + " " for t in pair["donor"][:500])
        prompt = PROMPT_TEMPLATE.format(content=text)
        t, ok = ttft_request(endpoint, model, prompt)
        if ok:
            cold_ttfts.append(t)
        print(f"  cold [{i+1}/{n_cold}] {t:.0f}ms", end="\r")

    if not cold_ttfts:
        return {"error": "no cold measurements"}

    cold_mean = sum(cold_ttfts) / len(cold_ttfts)
    print(f"\n  Cold baseline: mean={cold_mean:.0f}ms (n={len(cold_ttfts)})")

    # Register donors then query targets
    hit_count = 0
    pair_results: list[dict] = []

    for i, pair in enumerate(pairs):
        # Register donor
        donor_text = "".join(str(t) + " " for t in pair["donor"][:500])
        donor_prompt = PROMPT_TEMPLATE.format(content=donor_text)
        ttft_request(endpoint, model, donor_prompt, max_tokens=50)

        # Query target
        target_text = "".join(str(t) + " " for t in pair["target"][:500])
        target_prompt = PROMPT_TEMPLATE.format(content=target_text)
        t, ok = ttft_request(endpoint, model, target_prompt)

        is_hit = ok and t < 0.70 * cold_mean
        if is_hit:
            hit_count += 1

        pair_results.append({
            "name": pair["name"],
            "scenario": pair["scenario"],
            "ttft_ms": t if ok else None,
            "hit": is_hit,
        })
        print(f"  e2e [{i+1}/{len(pairs)}] {t:.0f}ms {'HIT' if is_hit else 'miss'}", end="\r")

    hit_pct = hit_count / len(pairs) * 100 if pairs else 0
    print(f"\n  E2E: {hit_count}/{len(pairs)} hits ({hit_pct:.0f}%)")

    return {
        "cold_mean_ms": cold_mean,
        "n_pairs": len(pairs),
        "hits": hit_count,
        "hit_pct": hit_pct,
        "pair_results": pair_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-samples", type=int, default=32)
    parser.add_argument("--token-length", type=int, default=4096)
    parser.add_argument("--skip-e2e", action="store_true",
                        help="Run offline analysis only (no vLLM endpoint needed)")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    n_per_scenario = args.n_samples // 3
    doc_len = args.token_length

    print(f"\nPartialAttention Shifted-Prefix Benchmark")
    print(f"  n_per_scenario={n_per_scenario}, token_length={doc_len}")
    print()

    # Generate scenario pairs
    all_pairs = (
        generate_instruction_shift_pairs(n_per_scenario, doc_len)
        + generate_header_shift_pairs(n_per_scenario, doc_len)
        + generate_padding_shift_pairs(n_per_scenario, doc_len)
    )

    # Phase 1: Offline alignment analysis
    print("Phase 1: Offline alignment analysis")
    offline_results = analyze_offline(all_pairs)

    # Aggregate by scenario
    scenarios = {}
    for r in offline_results:
        s = r["scenario"]
        if s not in scenarios:
            scenarios[s] = {
                "chunk_reuse_ratios": [],
                "token_reuse_ratios": [],
                "lost_reuse_tokens": [],
                "shifts": [],
            }
        scenarios[s]["chunk_reuse_ratios"].append(r["chunk_reuse_ratio"])
        scenarios[s]["token_reuse_ratios"].append(r["token_reuse_ratio"])
        scenarios[s]["lost_reuse_tokens"].append(r["lost_reuse_tokens"])
        scenarios[s]["shifts"].append(r["shift_tokens"])

    print(f"\n  {'Scenario':<20} {'N':>4} {'Chunk Reuse':>12} {'Token Reuse':>12} "
          f"{'Lost Tokens':>12} {'Avg Shift':>10}")
    print("-" * 75)

    for scenario, data in scenarios.items():
        n = len(data["chunk_reuse_ratios"])
        chunk_mean = sum(data["chunk_reuse_ratios"]) / n
        token_mean = sum(data["token_reuse_ratios"]) / n
        lost_mean = sum(data["lost_reuse_tokens"]) / n
        shift_mean = sum(data["shifts"]) / n

        print(f"  {scenario:<20} {n:>4} {chunk_mean:>11.1%} {token_mean:>11.1%} "
              f"{lost_mean:>11.0f} {shift_mean:>9.0f}")

    total_chunk = sum(r["chunk_reuse_ratio"] for r in offline_results) / len(offline_results)
    total_token = sum(r["token_reuse_ratio"] for r in offline_results) / len(offline_results)
    total_lost = sum(r["lost_reuse_tokens"] for r in offline_results) / len(offline_results)

    print(f"\n  Overall: chunk={total_chunk:.1%}, token={total_token:.1%}, "
          f"lost={total_lost:.0f} tokens/pair")
    print(f"  → PartialAttention could recover {total_token - total_chunk:.1%} "
          f"additional reuse per request")

    # Phase 2: E2E (optional)
    e2e_results = None
    if not args.skip_e2e:
        try:
            requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
            print(f"\nPhase 2: E2E measurement against {args.endpoint}")
            e2e_results = run_e2e(all_pairs, args.endpoint, args.model)
        except Exception as e:
            print(f"\nSkipping E2E (endpoint not available): {e}")

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import (
        bootstrap_mean,
        bootstrap_proportion,
    )

    print()
    print("=" * 75)
    print("Bootstrap 95% Confidence Intervals")
    print("=" * 75)

    for scenario, data in scenarios.items():
        chunk_arr = np.array(data["chunk_reuse_ratios"])
        token_arr = np.array(data["token_reuse_ratios"])
        lost_arr = np.array(data["lost_reuse_tokens"])
        print(f"\n  Scenario: {scenario}")
        print(f"    Chunk reuse ratio mean: {bootstrap_mean(chunk_arr)}")
        print(f"    Token reuse ratio mean: {bootstrap_mean(token_arr)}")
        print(f"    Lost tokens mean:       {bootstrap_mean(lost_arr)}")

    if e2e_results is not None and "pair_results" in e2e_results:
        e2e_pairs = e2e_results["pair_results"]
        e2e_ttfts = np.array([p["ttft_ms"] for p in e2e_pairs if p["ttft_ms"] is not None])
        e2e_hits = sum(1 for p in e2e_pairs if p["hit"])
        e2e_total = len(e2e_pairs)
        print(f"\n  E2E results:")
        if len(e2e_ttfts) > 0:
            print(f"    TTFT mean:  {bootstrap_mean(e2e_ttfts)}")
        print(f"    Hit rate:   {bootstrap_proportion(e2e_hits, e2e_total)}")
    print()

    # Save results
    if args.output:
        out = {
            "n_per_scenario": n_per_scenario,
            "token_length": doc_len,
            "offline": {
                "per_pair": offline_results,
                "per_scenario": {
                    s: {
                        "chunk_reuse_mean": sum(d["chunk_reuse_ratios"]) / len(d["chunk_reuse_ratios"]),
                        "token_reuse_mean": sum(d["token_reuse_ratios"]) / len(d["token_reuse_ratios"]),
                        "lost_tokens_mean": sum(d["lost_reuse_tokens"]) / len(d["lost_reuse_tokens"]),
                        "avg_shift": sum(d["shifts"]) / len(d["shifts"]),
                    }
                    for s, d in scenarios.items()
                },
                "overall_chunk_reuse": total_chunk,
                "overall_token_reuse": total_token,
                "recoverable_reuse": total_token - total_chunk,
            },
        }
        if e2e_results:
            out["e2e"] = e2e_results

        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")

    print("\nInterpretation: chunk-swap injection (Δ=0) cannot reuse shifted tokens.")
    print("PartialAttention with RoPE Δ≠0 correction could recover the 'token_reuse'")
    print("fraction, reducing prefill compute by the 'recoverable_reuse' percentage.")


if __name__ == "__main__":
    main()
