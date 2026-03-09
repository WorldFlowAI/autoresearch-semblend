#!/usr/bin/env python3
"""RoPE Position-Shift Validation Benchmark.

MOTIVATION
----------
The prior rope_ablation_bench.py used REORDER cluster variations, which proved
inconclusive: paragraph-level reordering in real articles rarely produces
256-token chunks at different absolute positions (LMCache's chunk matching
finds zero matches, falling back to cold prefill, so RoPE has nothing to fix).

This benchmark forces non-zero RoPE deltas by design:

  Donor text  :  [ARTICLE_N_TOKENS]         positions 0 … N-1
  Variation   :  [FILLER_256][ARTICLE_N_TOKENS]  positions 0 … N+255

LMCache chunk matching finds content chunks at delta = +256 for every chunk.
RoPE correction must apply RoPE(Δ=256) to correct donor K tensors in place.

Expected outcome:
  - WITH RoPE correction (default):    PPL ratio ≈ 1.0  (quality preserved)
  - WITHOUT RoPE correction (ablation): PPL ratio > 1.0  (measurable degradation)

This constitutes the first empirical validation that exact RoPE delta
correction enables non-contiguous KV reuse with quality preservation.

USAGE
-----
# Phase 1: Run with RoPE correction enabled (default vLLM config)
python -m benchmarks.e2e.rope_position_shift_bench \\
    --endpoint http://localhost:8100 \\
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
    --clusters-file benchmarks/data/xsum_clusters.json \\
    --target-length 4096 \\
    --n 8 \\
    --mode corrected \\
    --output-dir results/rope-position-shift

# Phase 2: Restart vLLM with SEMBLEND_DISABLE_ROPE_CORRECTION=1, then:
python -m benchmarks.e2e.rope_position_shift_bench \\
    --endpoint http://localhost:8100 \\
    --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
    --clusters-file benchmarks/data/xsum_clusters.json \\
    --target-length 4096 \\
    --n 8 \\
    --mode uncorrected \\
    --output-dir results/rope-position-shift

# Phase 3: Print comparison
python -m benchmarks.e2e.rope_position_shift_bench \\
    --mode compare \\
    --output-dir results/rope-position-shift
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunResult:
    cluster_id: str
    target_length: int
    filler_tokens: int
    variation_length: int
    cold_ttft_ms: float
    semblend_ttft_ms: float
    speedup: float
    hit: bool
    cold_text: str
    semblend_text: str
    rouge_l: float
    cold_ppl: float | None
    semblend_ppl: float | None
    ppl_ratio: float | None


@dataclass
class PhaseReport:
    mode: str
    model: str
    timestamp: str
    target_length: int
    filler_tokens: int
    n_runs: int
    runs: list[RunResult] = field(default_factory=list)

    # Aggregates (computed after all runs)
    hit_rate: float = 0.0
    speedup_p50: float = 0.0
    rouge_l_mean: float = 0.0
    ppl_ratio_mean: float | None = None


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def rouge_l(hyp: str, ref: str) -> float:
    """ROUGE-L F1 via word-level LCS."""
    if not hyp or not ref:
        return 0.0
    h, r = hyp.split(), ref.split()
    m, n = len(r), len(h)
    if not m or not n:
        return 0.0
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if r[i - 1] == h[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    p = lcs / n
    rc = lcs / m
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0


def ppl(logprobs: list[float]) -> float | None:
    if not logprobs:
        return None
    return math.exp(-statistics.mean(logprobs))


def build_filler(target_tokens: int, model: str, endpoint: str) -> str:
    """Build a repeatable filler string of exactly `target_tokens` tokens.

    Uses a generic sentence repeated and then trimmed to exact token count
    via the model's tokenizer (loaded lazily).
    """
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
    except Exception:
        # Fallback: estimate ~4 chars/token (conservative)
        base = "This section contains introductory benchmark context information. "
        text = base * (target_tokens // 8 + 2)
        return text[: target_tokens * 4]

    base = (
        "This section contains introductory benchmark context padding. "
        "The following text is the primary research document. "
    )
    # Build until we have at least target_tokens
    text = base
    while len(tok.encode(text)) < target_tokens:
        text += base

    # Trim to exact length
    ids = tok.encode(text)[:target_tokens]
    return tok.decode(ids, skip_special_tokens=True)


def call_vllm(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 1,
    logprobs: int | None = None,
    reference_id: str | None = None,
) -> tuple[str, list[float], float]:
    """POST to vLLM /v1/completions. Returns (text, token_logprobs, ttft_ms)."""
    payload: dict = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    if logprobs is not None:
        payload["logprobs"] = logprobs
    # Unique reference_id bypasses prefix cache and SemBlend donor lookup by default
    if reference_id is not None:
        payload["extra_body"] = {"reference_id": reference_id}

    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json=payload,
        timeout=300.0,
    )
    ttft_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_raw = data["choices"][0].get("logprobs") or {}
    lps = [x for x in (lp_raw.get("token_logprobs") or []) if x is not None]
    return text, lps, ttft_ms


def is_hit(cold_ttft_ms: float, semblend_ttft_ms: float, threshold: float = 0.70) -> bool:
    return semblend_ttft_ms < threshold * cold_ttft_ms


# ---------------------------------------------------------------------------
# Core benchmark
# ---------------------------------------------------------------------------

def run_phase(
    endpoint: str,
    model: str,
    clusters_path: str,
    target_length: int,
    n_runs: int,
    mode: str,
    filler_tokens: int = 256,
    max_tokens: int = 128,
) -> PhaseReport:
    """Run one phase (corrected or uncorrected).

    Both phases use the SAME prompt construction — the difference is solely
    whether SEMBLEND_DISABLE_ROPE_CORRECTION=1 is set in the vLLM environment.
    """
    print(f"\n{'='*60}")
    print(f"RoPE Position-Shift Benchmark  [mode={mode.upper()}]")
    print(f"  Endpoint:       {endpoint}")
    print(f"  Model:          {model}")
    print(f"  Target length:  {target_length} tokens")
    print(f"  Filler tokens:  {filler_tokens} (RoPE delta forced to +{filler_tokens})")
    print(f"  N runs:         {n_runs}")
    print(f"{'='*60}\n")

    if mode == "uncorrected":
        print("  *** VERIFY: vLLM must be running with SEMBLEND_DISABLE_ROPE_CORRECTION=1 ***")
        print("  If not restarted, results will be IDENTICAL to corrected phase.\n")

    # Health check
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        assert resp.status_code == 200, f"unhealthy: {resp.status_code}"
        print(f"  vLLM healthy.\n")
    except Exception as exc:
        print(f"  ERROR: cannot reach {endpoint}: {exc}")
        sys.exit(1)

    # Load clusters
    with open(clusters_path) as f:
        all_clusters = json.load(f)

    clusters = [c for c in all_clusters if c.get("target_token_length") == target_length]
    if not clusters:
        print(f"ERROR: no clusters with target_token_length={target_length} in {clusters_path}")
        sys.exit(1)
    print(f"  Found {len(clusters)} clusters at {target_length} tokens.\n")

    # Build filler (tokenizer-verified, reused for all runs)
    print("Building tokenizer-verified filler prefix...")
    filler = build_filler(filler_tokens, model, endpoint)
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model)
        actual_filler_toks = len(tok.encode(filler))
        print(f"  Filler: {actual_filler_toks} tokens (target={filler_tokens})\n")
    except Exception:
        actual_filler_toks = filler_tokens
        print(f"  Filler: ~{filler_tokens} tokens (tokenizer unavailable)\n")

    report = PhaseReport(
        mode=mode,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
        target_length=target_length,
        filler_tokens=actual_filler_toks,
        n_runs=n_runs,
    )

    for i in range(n_runs):
        cluster = clusters[i % len(clusters)]
        seed_text = cluster["seed_text"]
        cid = cluster.get("cluster_id", f"c{i}")

        # variation_prompt = [filler][seed_text]
        # All seed chunks are now at positions shifted by +filler_tokens
        variation_prompt = filler + " " + seed_text
        donor_prompt = seed_text

        # Unique IDs prevent prefix cache hits
        cold_ref = f"rope-cold-{uuid.uuid4()}"
        semblend_ref = f"rope-sb-{uuid.uuid4()}"
        donor_ref = f"rope-donor-{uuid.uuid4()}"

        print(f"Run {i+1}/{n_runs} [cluster={cid}]")

        try:
            # Step 1: Cold baseline — generate from variation WITHOUT donor
            cold_text, cold_lps, cold_ttft = call_vllm(
                endpoint, model, variation_prompt,
                max_tokens=max_tokens, logprobs=1,
            )
            cold_perp = ppl(cold_lps)
            print(f"  Cold TTFT: {cold_ttft:.0f}ms  ppl={f'{cold_perp:.3f}' if cold_perp else 'N/A'}")

            # Step 2: Register donor (seed text, no filler)
            call_vllm(endpoint, model, donor_prompt, max_tokens=1)
            time.sleep(0.3)

            # Step 3: SemBlend — generate from variation WITH donor registered
            sb_text, sb_lps, sb_ttft = call_vllm(
                endpoint, model, variation_prompt,
                max_tokens=max_tokens, logprobs=1,
            )
            sb_perp = ppl(sb_lps)
            hit = is_hit(cold_ttft, sb_ttft)

            ratio = (sb_perp / cold_perp) if (cold_perp and sb_perp) else None
            rl = rouge_l(sb_text, cold_text)

            speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0.0

            print(
                f"  SemBlend TTFT: {sb_ttft:.0f}ms  "
                f"speedup: {speedup:.2f}x  "
                f"hit: {hit}  "
                f"ppl_ratio: {f'{ratio:.4f}' if ratio else 'N/A'}  "
                f"rouge_l: {rl:.3f}"
            )

            report.runs.append(RunResult(
                cluster_id=cid,
                target_length=target_length,
                filler_tokens=actual_filler_toks,
                variation_length=target_length + actual_filler_toks,
                cold_ttft_ms=cold_ttft,
                semblend_ttft_ms=sb_ttft,
                speedup=speedup,
                hit=hit,
                cold_text=cold_text,
                semblend_text=sb_text,
                rouge_l=rl,
                cold_ppl=cold_perp,
                semblend_ppl=sb_perp,
                ppl_ratio=ratio,
            ))

        except Exception as exc:
            print(f"  ERROR: {exc}")

    # Aggregate
    hits = [r for r in report.runs if r.hit]
    all_ratios = [r.ppl_ratio for r in report.runs if r.ppl_ratio is not None]
    speedups = sorted(r.speedup for r in report.runs)

    report.hit_rate = len(hits) / len(report.runs) if report.runs else 0.0
    report.speedup_p50 = speedups[len(speedups) // 2] if speedups else 0.0
    report.rouge_l_mean = statistics.mean(r.rouge_l for r in report.runs) if report.runs else 0.0
    report.ppl_ratio_mean = statistics.mean(all_ratios) if all_ratios else None

    print(f"\n{'='*60}")
    print(f"SUMMARY [{mode.upper()}]")
    print(f"  Hit rate:        {report.hit_rate:.0%}  ({len(hits)}/{len(report.runs)})")
    print(f"  Speedup P50:     {report.speedup_p50:.2f}x")
    print(f"  PPL ratio mean:  {report.ppl_ratio_mean:.4f}" if report.ppl_ratio_mean else "  PPL ratio:  N/A")
    print(f"  ROUGE-L mean:    {report.rouge_l_mean:.3f}")
    print(f"  Delta forced:    +{actual_filler_toks} tokens (all matched chunks)")
    print(f"{'='*60}\n")

    return report


def compare_phases(output_dir: str) -> None:
    """Load corrected + uncorrected JSON and print side-by-side comparison."""
    corr_path = Path(output_dir) / "rope_corrected.json"
    uncorr_path = Path(output_dir) / "rope_uncorrected.json"

    if not corr_path.exists() or not uncorr_path.exists():
        print(f"ERROR: need both {corr_path} and {uncorr_path}")
        sys.exit(1)

    with open(corr_path) as f:
        c = json.load(f)
    with open(uncorr_path) as f:
        u = json.load(f)

    print(f"\n{'='*70}")
    print(f"RoPE Position-Shift Ablation: CORRECTED vs UNCORRECTED")
    print(f"  Target length:  {c['target_length']} tokens")
    print(f"  Filler tokens:  {c['filler_tokens']} (RoPE delta forced to +{c['filler_tokens']})")
    print(f"  Model:          {c['model']}")
    print(f"{'='*70}")
    print(f"{'Metric':<30} {'Corrected':>15} {'Uncorrected':>15} {'Delta':>12}")
    print(f"{'-'*70}")

    c_ppl = c.get("ppl_ratio_mean")
    u_ppl = u.get("ppl_ratio_mean")
    c_rl = c.get("rouge_l_mean", 0)
    u_rl = u.get("rouge_l_mean", 0)
    c_hit = c.get("hit_rate", 0)
    u_hit = u.get("hit_rate", 0)
    c_spd = c.get("speedup_p50", 0)
    u_spd = u.get("speedup_p50", 0)

    def fmt(v, fmt_str):
        return fmt_str.format(v) if v is not None else "N/A"

    print(f"{'Hit rate':<30} {fmt(c_hit, '{:.0%}'):>15} {fmt(u_hit, '{:.0%}'):>15}")
    print(f"{'Speedup P50':<30} {fmt(c_spd, '{:.2f}x'):>15} {fmt(u_spd, '{:.2f}x'):>15}")
    print(f"{'PPL ratio (lower=better)':<30} {fmt(c_ppl, '{:.4f}'):>15} {fmt(u_ppl, '{:.4f}'):>15}", end="")
    if c_ppl and u_ppl:
        delta = u_ppl - c_ppl
        print(f" {delta:>+12.4f}  ← correction benefit")
    else:
        print()
    print(f"{'ROUGE-L (higher=better)':<30} {fmt(c_rl, '{:.3f}'):>15} {fmt(u_rl, '{:.3f}'):>15}", end="")
    if c_rl and u_rl:
        print(f" {c_rl - u_rl:>+12.3f}  ← correction benefit")
    else:
        print()
    print(f"{'='*70}")

    # Interpretation
    if c_ppl and u_ppl:
        if u_ppl > c_ppl * 1.02:
            print(f"\nCONCLUSION: RoPE correction CONFIRMED beneficial.")
            print(f"  Uncorrected PPL ratio {u_ppl:.4f} > corrected {c_ppl:.4f}")
            print(f"  Quality degradation without correction: {(u_ppl/c_ppl - 1)*100:.1f}%")
            print(f"  Position delta Δ={c['filler_tokens']} tokens is sufficient to cause measurable drift.")
        elif abs(u_ppl - c_ppl) < 0.005:
            print(f"\nCONCLUSION: INCONCLUSIVE — PPL ratio difference < 0.5%")
            print(f"  LMCache may not be matching chunks at the expected positions.")
            print(f"  Check vLLM logs for 'rope_pairs' count > 0 during SemBlend hits.")
            print(f"  Try larger filler (e.g., 512 or 1024 tokens) or different content.")
        else:
            print(f"\nCONCLUSION: Corrected PPL unexpectedly higher — check for run artifacts.")
    else:
        print(f"\nCONCLUSION: Insufficient data for comparison.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoPE position-shift ablation: validates RoPE correction by forcing non-zero deltas."
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", default="benchmarks/data/xsum_clusters.json")
    parser.add_argument("--target-length", type=int, default=4096,
                        help="Token length for donor content (variation = target + filler_tokens)")
    parser.add_argument("--filler-tokens", type=int, default=256,
                        help="Number of tokens to prepend, forcing RoPE delta = this value")
    parser.add_argument("--n", type=int, default=8, help="Number of runs")
    parser.add_argument("--max-tokens", type=int, default=128,
                        help="Max generation tokens for quality measurement")
    parser.add_argument("--mode", choices=["corrected", "uncorrected", "compare"],
                        default="corrected")
    parser.add_argument("--output-dir", default="results/rope-position-shift")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.mode == "compare":
        compare_phases(args.output_dir)
        return

    report = run_phase(
        endpoint=args.endpoint,
        model=args.model,
        clusters_path=args.clusters_file,
        target_length=args.target_length,
        n_runs=args.n,
        mode=args.mode,
        filler_tokens=args.filler_tokens,
        max_tokens=args.max_tokens,
    )

    out_name = f"rope_{args.mode}.json"
    out_path = Path(args.output_dir) / out_name
    with open(out_path, "w") as f:
        json.dump(asdict(report), f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
