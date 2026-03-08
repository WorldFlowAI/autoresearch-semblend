"""Ablation C: RoPE Correction On/Off.

Validates the novel contribution: exact RoPE delta correction enables
non-contiguous KV reuse that was previously impossible.

Measures:
1. REORDER scenario WITH vs WITHOUT RoPE correction
2. Quality degradation from skipping correction (expected: significant)
3. Per-length comparison of corrected vs uncorrected quality
4. Position delta distribution and its effect on output quality

This is the key ablation that SemShareKV lacks -- they acknowledge
"sharing KV caches across semantically similar prompts" but don't
solve the position encoding problem for reordered content.

Since RoPE correction cannot be toggled at runtime (requires restarting
the vLLM pod with SEMBLEND_DISABLE_ROPE_CORRECTION=1), this benchmark
operates in three modes:

  --mode corrected     Run with RoPE correction enabled (default).
                       Save results to {output_dir}/rope_corrected.json.

  --mode uncorrected   Run with RoPE correction disabled.
                       Requires: restart vLLM with SEMBLEND_DISABLE_ROPE_CORRECTION=1
                       Save results to {output_dir}/rope_uncorrected.json.

  --mode compare       Load both JSON files and print side-by-side comparison.
                       No vLLM endpoint needed.

Usage:
    # Phase 1: Run with correction enabled (default vLLM config)
    python -m benchmarks.e2e.rope_ablation_bench \\
        --endpoint http://localhost:8000 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters benchmarks/data/semblend_clusters.json \\
        --mode corrected \\
        --output-dir results/rope-ablation

    # Phase 2: Restart vLLM with SEMBLEND_DISABLE_ROPE_CORRECTION=1, then:
    python -m benchmarks.e2e.rope_ablation_bench \\
        --endpoint http://localhost:8000 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --clusters benchmarks/data/semblend_clusters.json \\
        --mode uncorrected \\
        --output-dir results/rope-ablation

    # Phase 3: Compare results
    python -m benchmarks.e2e.rope_ablation_bench \\
        --mode compare \\
        --output-dir results/rope-ablation
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


@dataclass(frozen=True)
class RunMeasurement:
    """Single measurement of a REORDER variation."""

    cluster_id: str
    prompt_length: int
    target_token_length: int
    cold_ttft_ms: float
    semblend_ttft_ms: float
    speedup: float
    cold_text: str
    semblend_text: str
    rouge_l: float
    ppl_ratio: float | None
    cold_ppl: float | None
    semblend_ppl: float | None


@dataclass(frozen=True)
class LengthAggregate:
    """Aggregated results for a single prompt length."""

    target_length: int
    num_runs: int
    cold_ttft_p50_ms: float
    semblend_ttft_p50_ms: float
    speedup_mean: float
    rouge_l_mean: float
    rouge_l_stdev: float | None
    ppl_ratio_mean: float | None
    ppl_ratio_stdev: float | None


@dataclass
class PhaseReport:
    """Report for a single phase (corrected or uncorrected)."""

    mode: str  # "corrected" or "uncorrected"
    model: str
    hardware: str
    timestamp: str
    num_clusters: int
    num_runs: int
    length_aggregates: list[LengthAggregate] = field(default_factory=list)
    raw_measurements: list[RunMeasurement] = field(default_factory=list)


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
    """Generate text with logprobs. Returns (text, logprobs, ttft_ms)."""
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


def _find_reorder_pairs(clusters: list[dict]) -> list[tuple[dict, dict]]:
    """Extract (cluster, reorder_variation) pairs from cluster data."""
    pairs = []
    for c in clusters:
        for v in c.get("variations", []):
            if v["overlap_type"] == "reorder":
                pairs.append((c, v))
                break
    return pairs


def run_phase(
    endpoint: str,
    model: str,
    clusters_path: str,
    mode: str,
    num_runs: int = 20,
    max_tokens: int = 50,
    hardware: str = "unknown",
) -> PhaseReport:
    """Run one phase of the RoPE ablation (corrected or uncorrected).

    Both phases use the same methodology:
    1. For each REORDER cluster, register seed as donor.
    2. Generate from seed (cold baseline).
    3. Generate from reordered variation (SemBlend path).
    4. Compare quality.
    """
    with open(clusters_path) as f:
        clusters = json.load(f)

    reorder_pairs = _find_reorder_pairs(clusters)
    if not reorder_pairs:
        print("No REORDER variations found in clusters")
        return PhaseReport(
            mode=mode,
            model=model,
            hardware=hardware,
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_clusters=0,
            num_runs=0,
        )

    report = PhaseReport(
        mode=mode,
        model=model,
        hardware=hardware,
        timestamp=datetime.now(timezone.utc).isoformat(),
        num_clusters=len(reorder_pairs),
        num_runs=num_runs,
    )

    print(f"RoPE Ablation Phase: {mode.upper()}")
    print(f"  Model: {model}")
    print(f"  REORDER clusters: {len(reorder_pairs)}")
    print(f"  Runs per length: {num_runs}")

    if mode == "uncorrected":
        print()
        print("  NOTE: This phase requires vLLM to be running with:")
        print("    SEMBLEND_DISABLE_ROPE_CORRECTION=1")
        print("  If you haven't restarted vLLM with this env var,")
        print("  results will be identical to the corrected phase.")
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

    # Group by prompt length
    by_length: dict[int, list[tuple[dict, dict]]] = {}
    for c, v in reorder_pairs:
        tl = c.get("target_token_length", 0)
        by_length.setdefault(tl, []).append((c, v))

    measurements: list[RunMeasurement] = []

    for target_len, pairs in sorted(by_length.items()):
        print(f"\nToken length: {target_len}")

        for run_idx in range(min(num_runs, len(pairs))):
            cluster, variation = pairs[run_idx % len(pairs)]
            seed = cluster["seed_text"]
            reordered = variation["text"]
            cluster_id = cluster.get("cluster_id", f"cluster_{run_idx}")

            print(f"  Run {run_idx + 1}/{num_runs}...", end=" ", flush=True)

            try:
                # 1. Cold baseline
                cold_text, cold_lps, cold_ttft = generate_with_logprobs(
                    endpoint, model, seed, max_tokens,
                )
                cold_ppl = compute_perplexity(cold_lps)

                # 2. Register seed as donor
                register_donor(endpoint, model, seed)
                time.sleep(0.5)

                # 3. Send reordered variation
                sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                    endpoint, model, reordered, max_tokens,
                )
                sb_ppl = compute_perplexity(sb_lps)

                rl = rouge_l(sb_text, cold_text)
                speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0

                ppl_ratio = None
                if cold_ppl is not None and sb_ppl is not None and cold_ppl > 0:
                    ppl_ratio = sb_ppl / cold_ppl

                m = RunMeasurement(
                    cluster_id=cluster_id,
                    prompt_length=len(seed),
                    target_token_length=target_len,
                    cold_ttft_ms=cold_ttft,
                    semblend_ttft_ms=sb_ttft,
                    speedup=speedup,
                    cold_text=cold_text[:200],
                    semblend_text=sb_text[:200],
                    rouge_l=rl,
                    ppl_ratio=ppl_ratio,
                    cold_ppl=cold_ppl,
                    semblend_ppl=sb_ppl,
                )
                measurements.append(m)

                ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "ppl_r=N/A"
                print(
                    f"cold={cold_ttft:.0f}ms sb={sb_ttft:.0f}ms "
                    f"speedup={speedup:.1f}x RL={rl:.3f} {ppl_str}"
                )

            except Exception as e:
                print(f"FAILED: {e}")
                continue

            time.sleep(0.2)

    report.raw_measurements = measurements

    # Aggregate by length
    meas_by_len: dict[int, list[RunMeasurement]] = {}
    for m in measurements:
        meas_by_len.setdefault(m.target_token_length, []).append(m)

    for tl, items in sorted(meas_by_len.items()):
        cold_ttfts = sorted([m.cold_ttft_ms for m in items])
        sb_ttfts = sorted([m.semblend_ttft_ms for m in items])
        speedups = [m.speedup for m in items]
        rouges = [m.rouge_l for m in items]
        ppl_ratios = [m.ppl_ratio for m in items if m.ppl_ratio is not None]

        agg = LengthAggregate(
            target_length=tl,
            num_runs=len(items),
            cold_ttft_p50_ms=cold_ttfts[len(cold_ttfts) // 2],
            semblend_ttft_p50_ms=sb_ttfts[len(sb_ttfts) // 2],
            speedup_mean=statistics.mean(speedups) if speedups else 0,
            rouge_l_mean=statistics.mean(rouges) if rouges else 0,
            rouge_l_stdev=(
                statistics.stdev(rouges) if len(rouges) > 1 else None
            ),
            ppl_ratio_mean=(
                statistics.mean(ppl_ratios) if ppl_ratios else None
            ),
            ppl_ratio_stdev=(
                statistics.stdev(ppl_ratios) if len(ppl_ratios) > 1 else None
            ),
        )
        report.length_aggregates.append(agg)

    # Print phase summary
    _print_phase_summary(report)

    return report


def _print_phase_summary(report: PhaseReport) -> None:
    """Print summary table for a single phase."""
    print(f"\n{'=' * 75}")
    print(f"ROPE ABLATION: {report.mode.upper()} RESULTS")
    print(f"{'=' * 75}")
    print(
        f"{'Length':>8} {'N':>4} {'Cold p50':>10} {'SB p50':>10} "
        f"{'Speedup':>8} {'ROUGE-L':>9} {'PPL Ratio':>10}"
    )
    print("-" * 75)

    for a in report.length_aggregates:
        ppl_str = f"{a.ppl_ratio_mean:.4f}" if a.ppl_ratio_mean else "---"
        rl_std = f" +/-{a.rouge_l_stdev:.3f}" if a.rouge_l_stdev else ""
        print(
            f"{a.target_length:>8} {a.num_runs:>4} "
            f"{a.cold_ttft_p50_ms:>9.0f}ms "
            f"{a.semblend_ttft_p50_ms:>9.0f}ms "
            f"{a.speedup_mean:>7.1f}x "
            f"{a.rouge_l_mean:>8.3f}{rl_std} "
            f"{ppl_str:>10}"
        )


def run_comparison(output_dir: str) -> None:
    """Load corrected and uncorrected results and print comparison."""
    corr_path = os.path.join(output_dir, "rope_corrected.json")
    uncorr_path = os.path.join(output_dir, "rope_uncorrected.json")

    if not os.path.exists(corr_path):
        print(f"Missing: {corr_path}")
        print("Run with --mode corrected first.")
        return
    if not os.path.exists(uncorr_path):
        print(f"Missing: {uncorr_path}")
        print("Run with --mode uncorrected first.")
        return

    with open(corr_path) as f:
        corr_data = json.load(f)
    with open(uncorr_path) as f:
        uncorr_data = json.load(f)

    corr_aggs = {
        a["target_length"]: a for a in corr_data.get("length_aggregates", [])
    }
    uncorr_aggs = {
        a["target_length"]: a
        for a in uncorr_data.get("length_aggregates", [])
    }

    all_lengths = sorted(set(corr_aggs.keys()) | set(uncorr_aggs.keys()))

    print(f"\n{'=' * 100}")
    print("ROPE CORRECTION ABLATION: CORRECTED vs UNCORRECTED (REORDER scenarios)")
    print(f"{'=' * 100}")
    print(
        f"{'Length':>8} | "
        f"{'Corr Speedup':>13} {'Corr ROUGE-L':>13} {'Corr PPL-R':>11} | "
        f"{'Uncorr Speedup':>15} {'Uncorr ROUGE-L':>15} {'Uncorr PPL-R':>13}"
    )
    print("-" * 100)

    for tl in all_lengths:
        corr = corr_aggs.get(tl, {})
        uncorr = uncorr_aggs.get(tl, {})

        c_spd = corr.get("speedup_mean")
        c_rl = corr.get("rouge_l_mean")
        c_ppl = corr.get("ppl_ratio_mean")

        u_spd = uncorr.get("speedup_mean")
        u_rl = uncorr.get("rouge_l_mean")
        u_ppl = uncorr.get("ppl_ratio_mean")

        c_spd_s = f"{c_spd:.1f}x" if c_spd else "---"
        c_rl_s = f"{c_rl:.4f}" if c_rl else "---"
        c_ppl_s = f"{c_ppl:.4f}" if c_ppl else "---"

        u_spd_s = f"{u_spd:.1f}x" if u_spd else "---"
        u_rl_s = f"{u_rl:.4f}" if u_rl else "---"
        u_ppl_s = f"{u_ppl:.4f}" if u_ppl else "---"

        print(
            f"{tl:>8} | "
            f"{c_spd_s:>13} {c_rl_s:>13} {c_ppl_s:>11} | "
            f"{u_spd_s:>15} {u_rl_s:>15} {u_ppl_s:>13}"
        )

    # Compute overall deltas
    corr_rls = [
        a.get("rouge_l_mean", 0)
        for a in corr_aggs.values()
        if a.get("rouge_l_mean") is not None
    ]
    uncorr_rls = [
        a.get("rouge_l_mean", 0)
        for a in uncorr_aggs.values()
        if a.get("rouge_l_mean") is not None
    ]
    corr_ppls = [
        a.get("ppl_ratio_mean", 0)
        for a in corr_aggs.values()
        if a.get("ppl_ratio_mean") is not None
    ]
    uncorr_ppls = [
        a.get("ppl_ratio_mean", 0)
        for a in uncorr_aggs.values()
        if a.get("ppl_ratio_mean") is not None
    ]

    if corr_rls and uncorr_rls:
        avg_corr_rl = statistics.mean(corr_rls)
        avg_uncorr_rl = statistics.mean(uncorr_rls)
        rl_delta = avg_corr_rl - avg_uncorr_rl
        print(f"\n  ROUGE-L delta (corrected - uncorrected): {rl_delta:+.4f}")
        print(f"  Corrected avg: {avg_corr_rl:.4f}, Uncorrected avg: {avg_uncorr_rl:.4f}")

    if corr_ppls and uncorr_ppls:
        avg_corr_ppl = statistics.mean(corr_ppls)
        avg_uncorr_ppl = statistics.mean(uncorr_ppls)
        ppl_delta = avg_uncorr_ppl - avg_corr_ppl
        print(f"\n  PPL Ratio delta (uncorrected - corrected): {ppl_delta:+.4f}")
        print(f"  Corrected avg: {avg_corr_ppl:.4f}, Uncorrected avg: {avg_uncorr_ppl:.4f}")

    # Paper-ready table
    print(f"\n{'=' * 75}")
    print("PAPER TABLE: RoPE Correction Ablation on REORDER Scenarios")
    print(f"{'=' * 75}")
    print(
        f"{'Length':>8} | {'With Correction':^30} | {'Without Correction':^30}"
    )
    print(
        f"{'':>8} | {'ROUGE-L':>10} {'PPL Ratio':>10} {'Speedup':>8} | "
        f"{'ROUGE-L':>10} {'PPL Ratio':>10} {'Speedup':>8}"
    )
    print("-" * 75)

    for tl in all_lengths:
        corr = corr_aggs.get(tl, {})
        uncorr = uncorr_aggs.get(tl, {})

        c_rl = f"{corr.get('rouge_l_mean', 0):.3f}" if corr.get("rouge_l_mean") else "---"
        c_ppl = f"{corr.get('ppl_ratio_mean', 0):.3f}" if corr.get("ppl_ratio_mean") else "---"
        c_spd = f"{corr.get('speedup_mean', 0):.1f}x" if corr.get("speedup_mean") else "---"

        u_rl = f"{uncorr.get('rouge_l_mean', 0):.3f}" if uncorr.get("rouge_l_mean") else "---"
        u_ppl = f"{uncorr.get('ppl_ratio_mean', 0):.3f}" if uncorr.get("ppl_ratio_mean") else "---"
        u_spd = f"{uncorr.get('speedup_mean', 0):.1f}x" if uncorr.get("speedup_mean") else "---"

        print(
            f"{tl:>8} | {c_rl:>10} {c_ppl:>10} {c_spd:>8} | "
            f"{u_rl:>10} {u_ppl:>10} {u_spd:>8}"
        )

    print(f"\n  Expected: Without correction, REORDER quality degrades")
    print(f"  significantly (PPL ratio >> 1.0, ROUGE-L drops) because")
    print(f"  K tensors carry incorrect position encodings.")
    print(f"{'=' * 75}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ablation C: RoPE Correction On/Off",
    )
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--clusters",
        default="benchmarks/data/semblend_clusters.json",
    )
    parser.add_argument(
        "--mode",
        choices=["corrected", "uncorrected", "compare"],
        default="corrected",
        help=(
            "corrected: run with RoPE correction (default). "
            "uncorrected: run without correction (requires vLLM restart "
            "with SEMBLEND_DISABLE_ROPE_CORRECTION=1). "
            "compare: load both result files and print comparison."
        ),
    )
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--output-dir", default="results/rope-ablation")
    parser.add_argument("--hardware", default="unknown")
    args = parser.parse_args()

    if args.mode == "compare":
        run_comparison(args.output_dir)
        return

    report = run_phase(
        endpoint=args.endpoint,
        model=args.model,
        clusters_path=args.clusters,
        mode=args.mode,
        num_runs=args.num_runs,
        max_tokens=args.max_tokens,
        hardware=args.hardware,
    )

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"rope_{args.mode}.json")
    with open(out_file, "w") as f:
        json.dump(asdict(report), f, indent=2, default=str)
    print(f"\nSaved: {out_file}")

    if args.mode == "corrected":
        print(f"\n  Next step: Restart vLLM with SEMBLEND_DISABLE_ROPE_CORRECTION=1")
        print(f"  Then run: python -m benchmarks.e2e.rope_ablation_bench --mode uncorrected \\")
        print(f"    --output-dir {args.output_dir}")
    elif args.mode == "uncorrected":
        print(f"\n  Next step: Compare results:")
        print(f"  python -m benchmarks.e2e.rope_ablation_bench --mode compare \\")
        print(f"    --output-dir {args.output_dir}")


if __name__ == "__main__":
    main()
