#!/usr/bin/env python3
"""Bathtub curve calibration from real per-layer K-norm deviations.

SemBlend's bathtub curve (layer sensitivity function) was previously
calibrated from published data in other papers (CacheBlend, KVShare,
ProphetKV). This script calibrates it directly from Qwen2.5-7B-AWQ
and LLaMA-3.1-8B-AWQ running in the autoresearch deployment.

Methodology:
  1. Run pairs of semantically similar prompts against vLLM.
  2. Collect per-layer K-norm deviations logged to /tmp/semblend_deviations.jsonl
     (emitted by SemBlend's _log_layer_deviation hook when
     SEMBLEND_LOG_DEVIATIONS=1 is set).
  3. Fit the bathtub curve parameters (sigma_e, sigma_l, beta) to the
     measured deviation profile.
  4. Compare to the published presets:
       Qwen (funnel):          sigma_l=0.35, sigma_e=0.15
       LLaMA (inv. funnel):    sigma_e=0.45, sigma_l=0.15
  5. Output fitted parameters + goodness-of-fit.

Requires:
  - SEMBLEND_LOG_DEVIATIONS=1 set in vLLM pod env (via values-autoresearch.yaml)
  - Port-forward to autoresearch namespace vLLM (localhost:8100)
  - scipy for curve fitting (uv add scipy)

Usage:
    python -m benchmarks.e2e.bathtub_calibration_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --n-pairs 30 \\
        --deviation-log /tmp/semblend_deviations.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


def send_pair(endpoint: str, model: str, donor_text: str, query_text: str) -> bool:
    """Register donor then query with related text. Returns True if hit."""
    # Register donor
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": donor_text,
                  "max_tokens": 10, "temperature": 0.0},
            timeout=300,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"  Donor registration failed: {e}", file=sys.stderr)
        return False

    time.sleep(0.2)

    # Query with similar text
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": query_text,
                  "max_tokens": 10, "temperature": 0.0},
            timeout=300,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"  Query failed: {e}", file=sys.stderr)
        return False


def read_deviation_log(path: str) -> list[dict]:
    """Read JSONL deviation log from pod /tmp/semblend_deviations.jsonl."""
    lines = Path(path).read_text().strip().split("\n")
    entries = []
    for line in lines:
        line = line.strip()
        if line:
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def aggregate_deviations(entries: list[dict]) -> dict[int, list[float]]:
    """Aggregate per-layer K-norm deviations across all logged pairs."""
    by_layer: dict[int, list[float]] = {}
    for entry in entries:
        layer_idx = entry.get("layer_idx")
        k_norm = entry.get("k_norm")
        if layer_idx is not None and k_norm is not None:
            by_layer.setdefault(layer_idx, []).append(k_norm)
    return by_layer


def bathtub_curve(layer_idx: float, num_layers: float,
                  sigma_e: float, sigma_l: float, beta: float) -> float:
    """Bathtub deviation score: high at early/late layers, low in middle."""
    x = layer_idx / num_layers
    early = math.exp(-((x / sigma_e) ** 2))
    late = math.exp(-(((1 - x) / sigma_l) ** 2))
    return (early + late) ** beta


def fit_bathtub(layer_devs: dict[int, float]) -> dict:
    """Fit bathtub curve to observed mean deviations per layer."""
    try:
        from scipy.optimize import curve_fit
        import numpy as np
    except ImportError:
        print("scipy/numpy not available — skipping curve fit", file=sys.stderr)
        return {}

    layers = sorted(layer_devs.keys())
    if not layers:
        return {}

    num_layers = max(layers) + 1
    x_data = np.array([l / num_layers for l in layers])
    y_raw = np.array([layer_devs[l] for l in layers])
    # Normalize to [0,1] for fitting
    y_max = y_raw.max()
    y_data = y_raw / y_max if y_max > 0 else y_raw

    def model(x, sigma_e, sigma_l, beta):
        early = np.exp(-((x / sigma_e) ** 2))
        late = np.exp(-(((1 - x) / sigma_l) ** 2))
        return (early + late) ** beta

    try:
        popt, pcov = curve_fit(
            model, x_data, y_data,
            p0=[0.20, 0.20, 1.0],
            bounds=([0.01, 0.01, 0.1], [1.0, 1.0, 5.0]),
            maxfev=10000,
        )
        sigma_e, sigma_l, beta = popt
        residuals = y_data - model(x_data, *popt)
        r_squared = 1 - (residuals.var() / y_data.var())

        return {
            "sigma_e": float(sigma_e),
            "sigma_l": float(sigma_l),
            "beta": float(beta),
            "r_squared": float(r_squared),
            "num_layers": num_layers,
        }
    except Exception as e:
        print(f"Curve fit failed: {e}", file=sys.stderr)
        return {}


def fetch_deviation_log_from_pod(namespace: str = "autoresearch") -> str | None:
    """Fetch /tmp/semblend_deviations.jsonl from the vLLM pod via kubectl exec."""
    import subprocess
    try:
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace, "-l", "app=vllm",
             "--no-headers", "-o", "custom-columns=NAME:.metadata.name"],
            capture_output=True, text=True, timeout=10,
        )
        pod = result.stdout.strip().split("\n")[0]
        if not pod:
            return None

        fetch = subprocess.run(
            ["kubectl", "exec", "-n", namespace, pod, "--",
             "cat", "/tmp/semblend_deviations.jsonl"],
            capture_output=True, text=True, timeout=30,
        )
        if fetch.returncode == 0 and fetch.stdout.strip():
            local_path = "/tmp/semblend_deviations_fetched.jsonl"
            Path(local_path).write_text(fetch.stdout)
            return local_path
    except Exception as e:
        print(f"Could not fetch deviation log from pod: {e}", file=sys.stderr)
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-pairs", type=int, default=30)
    parser.add_argument("--deviation-log", default=None,
                        help="Local path to deviation JSONL (or auto-fetch from pod)")
    parser.add_argument("--namespace", default="autoresearch")
    args = parser.parse_args()

    print("Bathtub Calibration Benchmark")
    print(f"  model={args.model}, n_pairs={args.n_pairs}")
    print()

    # Health check
    requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()

    # Generate diverse prompt pairs for deviation collection
    base_texts = [
        "Climate change is accelerating due to greenhouse gas emissions. " * 50,
        "Machine learning models require large datasets for training. " * 50,
        "The human brain contains approximately 86 billion neurons. " * 50,
        "Global trade networks connect producers and consumers worldwide. " * 50,
        "Quantum computing uses superposition and entanglement principles. " * 50,
    ]

    variations = [
        lambda t: t,
        lambda t: t.replace("is ", "has been ").replace("are ", "have been "),
        lambda t: " ".join(t.split()[::-1][:len(t.split())//2]) + " " + " ".join(t.split()[len(t.split())//2:]),
    ]

    print(f"Sending {args.n_pairs} prompt pairs to trigger deviation logging...")
    print("(Requires SEMBLEND_LOG_DEVIATIONS=1 in vLLM pod env)")
    print()

    hits = 0
    for i in range(args.n_pairs):
        base = base_texts[i % len(base_texts)]
        var = variations[i % len(variations)]
        donor = f"Summarize: {base[:2000]}"
        query = f"Summarize: {var(base)[:2000]}"
        ok = send_pair(args.endpoint, args.model, donor, query)
        if ok:
            hits += 1
        print(f"  [{i+1}/{args.n_pairs}] {'hit' if ok else 'miss'}", end="\r")

    print(f"\n  Sent {args.n_pairs} pairs ({hits} successful)")

    # Fetch deviation log
    log_path = args.deviation_log
    if not log_path:
        print("\nFetching deviation log from pod...")
        log_path = fetch_deviation_log_from_pod(args.namespace)

    if not log_path or not Path(log_path).exists():
        print(
            "\nNo deviation log found. Enable SEMBLEND_LOG_DEVIATIONS=1 in the"
            " vLLM pod and redeploy, then re-run this benchmark.",
            file=sys.stderr,
        )
        print("\nTo enable: add to infra/values-autoresearch.yaml:")
        print("  vllm:")
        print("    extraEnv:")
        print("      SEMBLEND_LOG_DEVIATIONS: '1'")
        sys.exit(1)

    # Read and aggregate deviations
    entries = read_deviation_log(log_path)
    if not entries:
        print(f"No deviation entries in {log_path}", file=sys.stderr)
        sys.exit(1)

    print(f"\nRead {len(entries)} deviation entries from {log_path}")

    by_layer = aggregate_deviations(entries)
    mean_devs = {l: sum(vs) / len(vs) for l, vs in by_layer.items()}
    num_layers = max(mean_devs.keys()) + 1

    print(f"\nPer-layer mean K-norm deviation (n_layers={num_layers}):")
    print(f"{'Layer':>6} {'Mean K-norm':>12} {'Recompute?':>12}")
    for l in sorted(mean_devs.keys()):
        # Use current presets to predict recompute decision
        from synapse_kv_connector.bathtub import should_recompute_layer
        recomp = should_recompute_layer(l, num_layers)
        print(f"{l:>6} {mean_devs[l]:>12.4f} {'YES' if recomp else 'no':>12}")

    # Fit bathtub curve
    print("\nFitting bathtub curve to measured deviations...")
    fit = fit_bathtub(mean_devs)

    if fit:
        print(f"\nFitted parameters:")
        print(f"  sigma_e (early sensitivity) = {fit['sigma_e']:.3f}")
        print(f"  sigma_l (late sensitivity)  = {fit['sigma_l']:.3f}")
        print(f"  beta (sharpness)            = {fit['beta']:.3f}")
        print(f"  R² (goodness of fit)        = {fit['r_squared']:.4f}")

        # Compare to presets
        print(f"\nComparison to current presets:")
        if "Qwen" in args.model:
            print(f"  Preset (Qwen/funnel):      sigma_e=0.15, sigma_l=0.35")
        elif "Llama" in args.model or "llama" in args.model:
            print(f"  Preset (LLaMA/inv-funnel): sigma_e=0.45, sigma_l=0.15")
        print(f"  Measured:                  sigma_e={fit['sigma_e']:.3f}, sigma_l={fit['sigma_l']:.3f}")

        # Output for paper
        print(f"\nFor paper Table: sigma_e={fit['sigma_e']:.2f}, sigma_l={fit['sigma_l']:.2f}, "
              f"beta={fit['beta']:.1f}, R²={fit['r_squared']:.3f}")


if __name__ == "__main__":
    main()
