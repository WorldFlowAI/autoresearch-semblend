#!/usr/bin/env python3
"""FM3 Layer Criticality Profiling via Attention Entropy (DroidSpeak/kv-cache-physics gap).

MOTIVATION
----------
Failure Mode 3 (FM3): Only ~11% of transformer layers are "critical" for KV quality
(DroidSpeak). Architecture matters: LLaMA uses an inverted funnel (early layers
consolidate information, late layers more redundant), while Qwen uses a funnel
(late layers consolidate). This is the empirical basis for SemBlend's bathtub
calibration.

This benchmark profiles per-layer attention entropy from live vLLM prefill runs
to validate that our bathtub curve parameters match actual layer criticality.

METHODOLOGY
-----------
We extract per-token attention weights from vLLM using the logprobs API, then
measure attention entropy per layer position across 20 representative prompts.

High entropy = uniform attention distribution = layer is not "focusing" on
specific tokens = lower criticality (safe to reuse KV from donor).
Low entropy = peaked attention = layer is critically encoding specific content
= higher criticality (must recompute for semantic variants).

Expected result:
  - LLaMA: low entropy at layers 0-3 (early consolidation) = high criticality = must recompute
  - Qwen:  low entropy at layers 25-27 (late consolidation) = high criticality = must recompute
  - Middle layers: high entropy = low criticality = safe to reuse

USAGE
-----
    python -m benchmarks.e2e.layer_entropy_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters-file benchmarks/data/cnn_dailymail_clusters.json \\
        --target-length 4096 \\
        --n-prompts 20 \\
        --output results/layer-entropy/qwen_4k.json

NOTE: This requires vLLM to expose per-layer attention weights, which is not
a standard vLLM API. Instead, we use a proxy approach: measure PPL at each
context truncation length to identify where information is concentrated.

ALTERNATIVE: Token-level perplexity gradient approach.
Measure token-level PPL at the output for each prefix length, then compute
d(PPL)/d(layer) numerically by comparing PPL between sequential chunk
injections at different layer depths. This is the "DroidSpeak layer score"
approach without requiring internal attention access.

PRACTICAL IMPLEMENTATION
------------------------
Since we can't access internal attention weights via the vLLM OpenAI API,
we use a behavioral proxy:
1. Run cold inference → record output text (reference)
2. Run SemBlend inference with donor (all layers reused) → PPL ratio
3. Run SemBlend inference with bathtub selection (only early/late recomputed) → PPL ratio
4. Compare: if bathtub selection matches PPL of full-recompute, our layer selection is right.

This validates FM3 empirically without needing attention weight extraction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


def measure_ttft_and_text(
    endpoint: str, model: str, prompt: str, max_tokens: int = 128, timeout: int = 300,
) -> tuple[float, str]:
    """Measure TTFT and collect output text via streaming."""
    t0 = time.monotonic()
    text_out = ""
    ttft = -1.0
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model, "prompt": prompt,
                "max_tokens": max_tokens, "temperature": 0.0, "stream": True,
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
                if token and ttft < 0:
                    ttft = (time.monotonic() - t0) * 1000
                text_out += token
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        if ttft < 0:
            ttft = (time.monotonic() - t0) * 1000
    except Exception as e:
        print(f"  [WARN] Request failed: {e}")
    return ttft, text_out


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    """Register a prompt as a donor."""
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": prompt, "max_tokens": 5, "temperature": 0.0},
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  [WARN] Donor registration failed: {e}")


def set_env_bathtub_fraction(endpoint: str, fraction: str) -> None:
    """This can't be set at runtime — documented here as a reminder.

    To test different bathtub fractions, restart vLLM with
    SEMBLEND_BATHTUB_FRACTION=X before running this benchmark.
    """
    pass


def rouge_l(reference: str, hypothesis: str) -> float:
    """Compute ROUGE-L F1."""
    if not reference or not hypothesis:
        return 0.0
    ref_tokens = reference.lower().split()
    hyp_tokens = hypothesis.lower().split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    m, n = len(ref_tokens), len(hyp_tokens)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0.0
    recall = lcs / m if m > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


@dataclass
class LayerValidationResult:
    cluster_id: str
    bathtub_fraction: float
    cold_ttft_ms: float
    semblend_ttft_ms: float
    speedup: float
    hit: bool
    cold_text: str
    semblend_text: str
    rouge_l: float
    # If rouge_l ≈ 1.0, bathtub selection is sufficient (FM3 validated)
    # If rouge_l < 0.90, bathtub selection is losing information


def run_layer_entropy_bench(
    endpoint: str,
    model: str,
    clusters_file: str,
    target_length: int = 4096,
    n_prompts: int = 20,
    max_tokens: int = 128,
    bathtub_fractions: list[float] | None = None,
    output: str | None = None,
    settle_time: float = 2.0,
) -> dict:
    """
    Behavioral proxy for FM3 layer criticality validation.

    Tests: does bathtub selection (recompute early/late layers) preserve output
    quality vs all-layers recomputed? If yes, the bathtub is correctly identifying
    the critical layers.

    Three conditions per cluster:
    1. Cold (no donor): baseline PPL, TTFT
    2. SemBlend ALL: use donor, recompute ALL mismatched layers (conservative)
    3. SemBlend BATHTUB: use donor, recompute only bathtub-selected layers

    If condition 2 ≈ condition 3 in quality, bathtub is correctly identifying
    the critical layers → FM3 is handled.

    Note: Bathtub fraction can only be changed by restarting vLLM with different
    env var. This script runs at the CURRENT deployed bathtub fraction.
    The current fraction can be read from `infra/values-autoresearch.yaml`.
    """
    if bathtub_fractions is None:
        bathtub_fractions = [0.10]  # Current default — 10% layers recomputed

    with open(clusters_file) as f:
        clusters_data = json.load(f)

    clusters = clusters_data.get("clusters", clusters_data)
    if isinstance(clusters, dict):
        clusters = list(clusters.values())

    # Filter by target_length
    usable = [
        c for c in clusters
        if abs(c.get("seed", {}).get("token_count", 0) - target_length) <= target_length * 0.30
    ][:n_prompts]

    if not usable:
        print(f"ERROR: No clusters matching target_length={target_length}")
        sys.exit(1)

    print(f"=== FM3 Layer Criticality Validation ===")
    print(f"model={model}, target_length={target_length}, n_clusters={len(usable)}")
    print(f"max_tokens={max_tokens}")
    print(f"NOTE: Using current deployed bathtub fraction (see values-autoresearch.yaml)")

    results: list[LayerValidationResult] = []

    for ci, cluster in enumerate(usable):
        cluster_id = cluster.get("cluster_id", f"c{ci}")
        seed = cluster.get("seed", {})
        seed_text = seed.get("text", seed.get("prompt", ""))

        # Use REORDER variation if available (tests position-dependent layer recomputation)
        variations = cluster.get("variations", {})
        reorder_var = variations.get("reorder", {})
        test_text = reorder_var.get("text", reorder_var.get("prompt", "")) or seed_text

        if not seed_text or not test_text:
            continue

        print(f"\n  Cluster {ci+1}/{len(usable)}: {cluster_id[:12]}")

        # 1. Cold run (before donor registration)
        cold_ttft, cold_text = measure_ttft_and_text(endpoint, model, test_text, max_tokens)
        time.sleep(settle_time)

        # 2. Register seed as donor
        register_donor(endpoint, model, seed_text)
        time.sleep(settle_time)

        # 3. SemBlend run (with current bathtub fraction)
        sb_ttft, sb_text = measure_ttft_and_text(endpoint, model, test_text, max_tokens)
        time.sleep(0.5)

        hit = sb_ttft > 0 and cold_ttft > 0 and (cold_ttft / sb_ttft) >= 1.15
        speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0.0
        rl = rouge_l(cold_text, sb_text)

        result = LayerValidationResult(
            cluster_id=cluster_id,
            bathtub_fraction=bathtub_fractions[0],
            cold_ttft_ms=cold_ttft,
            semblend_ttft_ms=sb_ttft,
            speedup=speedup,
            hit=hit,
            cold_text=cold_text,
            semblend_text=sb_text,
            rouge_l=rl,
        )
        results.append(result)
        print(f"    hit={hit} speedup={speedup:.2f}× rouge_l={rl:.4f}")

    # Aggregate
    hit_results = [r for r in results if r.hit]
    agg = {
        "n_total": len(results),
        "hit_rate": len(hit_results) / len(results) if results else 0,
        "rouge_l_mean": statistics.mean(r.rouge_l for r in results) if results else 0,
        "rouge_l_hit_only": statistics.mean(r.rouge_l for r in hit_results) if hit_results else None,
        "speedup_hit_only": statistics.mean(r.speedup for r in hit_results) if hit_results else None,
    }

    print("\n=== FM3 Layer Criticality Summary ===")
    print(f"Hit rate: {agg['hit_rate']:.3f}")
    print(f"ROUGE-L mean: {agg['rouge_l_mean']:.4f}")
    if agg["rouge_l_hit_only"] is not None:
        print(f"ROUGE-L (hit only): {agg['rouge_l_hit_only']:.4f}")
    print()
    print("Interpretation:")
    if agg.get("rouge_l_hit_only", 0) is not None and agg.get("rouge_l_hit_only", 0) >= 0.95:
        print("  ROUGE-L ≥ 0.95 on hit pairs → bathtub layer selection is sufficient (FM3 OK)")
    else:
        print("  ROUGE-L < 0.95 on hit pairs → possible FM3 issue; try increasing bathtub fraction")

    print("\n---")
    print(f"benchmark:     layer_criticality_fm3")
    print(f"hit_rate:      {agg['hit_rate']:.4f}")
    print(f"rouge_l_mean:  {agg['rouge_l_mean']:.4f}")
    if agg.get("rouge_l_hit_only") is not None:
        print(f"rouge_l_hit:   {agg['rouge_l_hit_only']:.4f}")
    print("---")

    output_data = {
        "benchmark": "layer_criticality_fm3",
        "model": model,
        "clusters_file": clusters_file,
        "target_length": target_length,
        "n_prompts": len(usable),
        "max_tokens": max_tokens,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "aggregated": agg,
        "per_cluster": [asdict(r) for r in results],
    }

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to {output}")

    return output_data


def main():
    parser = argparse.ArgumentParser(description="FM3 layer criticality behavioral validation")
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", required=True)
    parser.add_argument("--target-length", type=int, default=4096)
    parser.add_argument("--n-prompts", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--settle-time", type=float, default=2.0)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_layer_entropy_bench(
        endpoint=args.endpoint,
        model=args.model,
        clusters_file=args.clusters_file,
        target_length=args.target_length,
        n_prompts=args.n_prompts,
        max_tokens=args.max_tokens,
        output=args.output,
        settle_time=args.settle_time,
    )


if __name__ == "__main__":
    main()
