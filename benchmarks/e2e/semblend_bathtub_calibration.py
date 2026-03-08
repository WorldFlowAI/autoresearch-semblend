#!/usr/bin/env python3
"""SemBlend bathtub curve calibration — measures per-layer KV deviation.

Sends pairs of semantically similar requests (seed + variant), captures
per-layer KV fingerprints from /tmp/semblend_fingerprints/latest.json,
and computes the normalized per-layer deviation σ(ℓ) for fitting the
bathtub curve: σ(ℓ) = σ_base + σ_e·exp(-ℓ/τ_e) + σ_l·exp(-(L-ℓ)/τ_l)

Usage:
    python3 semblend_bathtub_calibration.py \
        --endpoint http://localhost:8000 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --kubectl-pod synapse-staging-vllm-75d6bcc86-8nl5b \
        --kubectl-ns synapse-staging \
        --token-lengths 2048,4096,8192 \
        --pairs 5 \
        --output results/bathtub_calibration.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from math import exp
from statistics import mean, stdev

import numpy as np

try:
    import requests
except ImportError:
    sys.exit("ERROR: pip install requests")

try:
    from scipy.optimize import curve_fit
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("WARNING: scipy not installed — will output raw deviations but skip curve fitting")


# Semantic variant types
VARIANT_TYPES = ["REORDER", "PARTIAL_75", "PARAPHRASE"]

CONTEXTS = [
    (
        "Machine learning has transformed natural language processing. "
        "Transformer architectures replaced recurrent neural networks with "
        "self-attention mechanisms allowing massive parallelization. Models "
        "grew from 110M parameters to hundreds of billions. Efficient "
        "inference became critical with techniques like quantization, KV "
        "cache optimization, continuous batching, and speculative decoding. "
        "Flash Attention reduced memory complexity from O(n²) to O(n). "
        "Rotary Position Embeddings encode positional information as "
        "rotations in complex space preserving relative position awareness."
    ),
    (
        "Kubernetes orchestrates containerized applications across clusters. "
        "The control plane consists of API server, etcd, scheduler, and "
        "controller manager. Worker nodes run kubelet and container runtime. "
        "Pods are smallest deployable units with shared networking. "
        "Deployments manage ReplicaSets for rolling updates. Services "
        "provide stable networking with ClusterIP, NodePort, LoadBalancer. "
        "GPU scheduling uses device plugins and node selectors. Helm "
        "packages applications as charts with templated YAML manifests."
    ),
    (
        "Distributed caching architectures form the backbone of high "
        "performance systems. Redis provides in-memory key-value storage "
        "with lists, sets, sorted sets, hashes, and streams. Redis Cluster "
        "enables horizontal scaling via hash slot partitioning across "
        "multiple nodes. Vector similarity search with RediSearch supports "
        "HNSW and FLAT indexing. Cache invalidation strategies include TTL, "
        "LRU eviction, and write-through patterns. Memcached offers simpler "
        "multi-threaded caching with consistent hashing for distribution."
    ),
    (
        "NVIDIA GPU architecture evolved from Kepler through Ampere to "
        "Hopper. Tensor Cores accelerate matrix operations for deep "
        "learning with mixed precision FP16/BF16/TF32. CUDA programming "
        "model uses grids of thread blocks with shared memory and warps "
        "of 32 threads. H100 SXM5 provides 80GB HBM3 at 3.35 TB/s "
        "bandwidth. Multi-Instance GPU partitions resources into isolated "
        "instances. NVLink interconnects GPUs at 900 GB/s bidirectional. "
        "CUDA Graphs reduce kernel launch overhead for repetitive workloads."
    ),
    (
        "Modern compiler optimization techniques transform code through "
        "multiple intermediate representations. LLVM uses SSA form for "
        "analysis and transformation passes including dead code elimination, "
        "loop unrolling, vectorization, and inlining. Just-in-time "
        "compilation in V8 and PyPy bridges interpretation and AOT "
        "compilation. Profile-guided optimization uses runtime data to "
        "improve branch prediction and code layout. Link-time optimization "
        "enables cross-module inlining and devirtualization."
    ),
]


def make_variant(text: str, variant_type: str) -> str:
    """Create a semantic variant of the input text."""
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if variant_type == "REORDER":
        # Reverse sentence order
        return ". ".join(reversed(sentences)) + "."
    elif variant_type == "PARTIAL_75":
        # Keep first 75% of sentences
        keep = max(1, int(len(sentences) * 0.75))
        return ". ".join(sentences[:keep]) + "."
    elif variant_type == "PARAPHRASE":
        # Simple word-level shuffle within sentences (approximation)
        result = []
        for s in sentences:
            words = s.split()
            if len(words) > 4:
                # Swap pairs of adjacent words
                for i in range(0, len(words) - 1, 2):
                    words[i], words[i + 1] = words[i + 1], words[i]
            result.append(" ".join(words))
        return ". ".join(result) + "."
    return text


def build_prompt(context: str, target_tokens: int, model: str) -> str:
    """Build prompt padded to target token count."""
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception:
        # Fallback: ~4 chars per token
        chars_needed = target_tokens * 4
        repeated = context
        while len(repeated) < chars_needed:
            repeated = repeated + " " + context
        return repeated[:chars_needed]

    prefix = "Analyze the following:\n"
    base = prefix + context
    base_tok = len(tokenizer.encode(base))

    if base_tok >= target_tokens:
        ids = tokenizer.encode(base)[:target_tokens]
        return tokenizer.decode(ids)

    repeated = context
    while len(tokenizer.encode(prefix + repeated)) < target_tokens + 100:
        repeated = repeated + " " + context

    full = prefix + repeated
    ids = tokenizer.encode(full)[:target_tokens]
    return tokenizer.decode(ids)


def send_request(endpoint: str, model: str, prompt: str) -> float:
    """Send completion request, return TTFT in ms."""
    t0 = time.monotonic()
    r = requests.post(
        f"{endpoint}/v1/completions",
        json={"model": model, "prompt": prompt, "max_tokens": 5, "temperature": 0},
        timeout=120,
    )
    r.raise_for_status()
    return (time.monotonic() - t0) * 1000


def read_fingerprint(pod: str, ns: str) -> dict | None:
    """Read latest.json from the vLLM pod via kubectl exec."""
    try:
        result = subprocess.run(
            [
                "kubectl", "exec", pod, "-n", ns, "--",
                "cat", "/tmp/semblend_fingerprints/latest.json",
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"  WARNING: failed to read fingerprint: {e}", file=sys.stderr)
    return None


def compute_layer_deviation(fp_seed: dict, fp_variant: dict) -> list[dict]:
    """Compute per-layer normalized deviation between two fingerprints.

    Returns list of {layer_idx, norm_seed, norm_variant, deviation, relative_deviation}
    """
    deviations = []
    num_layers = min(len(fp_seed), len(fp_variant))

    for layer_idx in range(num_layers):
        seed_data = fp_seed.get(str(layer_idx), {})
        var_data = fp_variant.get(str(layer_idx), {})

        seed_norm = seed_data.get("norm", 0.0)
        var_norm = var_data.get("norm", 0.0)

        # Absolute deviation in K-norm
        abs_dev = abs(seed_norm - var_norm)

        # Relative deviation (normalized by seed norm)
        rel_dev = abs_dev / max(seed_norm, 1e-8)

        deviations.append({
            "layer_idx": layer_idx,
            "norm_seed": seed_norm,
            "norm_variant": var_norm,
            "abs_deviation": abs_dev,
            "rel_deviation": rel_dev,
        })

    return deviations


def bathtub_model(layer_idx, sigma_base, sigma_e, tau_e, sigma_l, tau_l, L):
    """Bathtub curve: σ(ℓ) = σ_base + σ_e·exp(-ℓ/τ_e) + σ_l·exp(-(L-ℓ)/τ_l)"""
    return sigma_base + sigma_e * np.exp(-layer_idx / tau_e) + sigma_l * np.exp(-(L - layer_idx) / tau_l)


def fit_bathtub(layer_deviations: list[list[dict]], num_layers: int) -> dict:
    """Fit bathtub curve parameters from collected deviation data."""
    if not HAS_SCIPY:
        return {"error": "scipy not installed"}

    # Average relative deviation per layer across all pairs
    avg_deviation = []
    for layer_idx in range(num_layers):
        devs = [
            pair[layer_idx]["rel_deviation"]
            for pair in layer_deviations
            if layer_idx < len(pair)
        ]
        if devs:
            avg_deviation.append(mean(devs))
        else:
            avg_deviation.append(0.0)

    x = np.arange(num_layers, dtype=float)
    y = np.array(avg_deviation)
    L = float(num_layers)

    def model(ell, sigma_base, sigma_e, tau_e, sigma_l, tau_l):
        return bathtub_model(ell, sigma_base, sigma_e, tau_e, sigma_l, tau_l, L)

    try:
        popt, pcov = curve_fit(
            model, x, y,
            p0=[0.12, 0.35, 3.0, 0.20, 4.0],
            bounds=([0, 0, 0.1, 0, 0.1], [1, 2, 50, 2, 50]),
            maxfev=10000,
        )
        perr = np.sqrt(np.diag(pcov))

        params = {
            "sigma_base": float(popt[0]),
            "sigma_e": float(popt[1]),
            "tau_e": float(popt[2]),
            "sigma_l": float(popt[3]),
            "tau_l": float(popt[4]),
            "L": int(num_layers),
            "errors": {
                "sigma_base": float(perr[0]),
                "sigma_e": float(perr[1]),
                "tau_e": float(perr[2]),
                "sigma_l": float(perr[3]),
                "tau_l": float(perr[4]),
            },
        }

        # Compute fitted values
        fitted = [model(i, *popt) for i in range(num_layers)]
        residuals = [abs(y[i] - fitted[i]) for i in range(num_layers)]
        r_squared = 1 - sum(r ** 2 for r in residuals) / max(sum((yi - mean(avg_deviation)) ** 2 for yi in avg_deviation), 1e-10)

        params["r_squared"] = float(r_squared)
        params["fitted_curve"] = fitted
        params["avg_deviation"] = avg_deviation

        return params

    except Exception as e:
        return {
            "error": str(e),
            "avg_deviation": avg_deviation,
        }


def main():
    parser = argparse.ArgumentParser(description="SemBlend bathtub curve calibration")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--kubectl-pod", required=True, help="vLLM pod name")
    parser.add_argument("--kubectl-ns", default="synapse-staging")
    parser.add_argument("--token-lengths", default="2048,4096,8192")
    parser.add_argument("--pairs", type=int, default=5)
    parser.add_argument("--output", default="results/bathtub_calibration.json")
    args = parser.parse_args()

    token_lengths = [int(x) for x in args.token_lengths.split(",")]
    endpoint = args.endpoint
    model = args.model
    pod = args.kubectl_pod
    ns = args.kubectl_ns

    print(f"SemBlend Bathtub Curve Calibration")
    print(f"  Model: {model}")
    print(f"  Token lengths: {token_lengths}")
    print(f"  Pairs per length per variant: {args.pairs}")
    print(f"  Pod: {pod}")
    print()

    # Verify connectivity
    r = requests.get(f"{endpoint}/health", timeout=5)
    print(f"  vLLM healthy: {r.status_code}")

    fp_test = read_fingerprint(pod, ns)
    if fp_test is None:
        sys.exit("ERROR: Cannot read fingerprints from pod")
    num_layers = len(fp_test)
    print(f"  Model layers: {num_layers}")
    print()

    all_deviations = []  # list of (length, variant_type, deviations_per_layer)
    results = {"model": model, "num_layers": num_layers, "measurements": []}

    for target_tokens in token_lengths:
        print(f"{'=' * 60}")
        print(f"  {target_tokens} TOKENS")
        print(f"{'=' * 60}")

        for variant_type in VARIANT_TYPES:
            print(f"\n  --- {variant_type} ---")

            for pair_idx in range(args.pairs):
                ctx = CONTEXTS[pair_idx % len(CONTEXTS)]
                seed_prompt = build_prompt(ctx, target_tokens, model)
                variant_ctx = make_variant(ctx, variant_type)
                variant_prompt = build_prompt(variant_ctx, target_tokens, model)

                # Send seed, capture fingerprint
                try:
                    seed_ttft = send_request(endpoint, model, seed_prompt)
                except Exception as e:
                    print(f"    Pair {pair_idx + 1}: seed FAILED: {e}")
                    continue

                time.sleep(0.5)  # Let fingerprint flush
                fp_seed = read_fingerprint(pod, ns)

                # Send variant, capture fingerprint
                try:
                    var_ttft = send_request(endpoint, model, variant_prompt)
                except Exception as e:
                    print(f"    Pair {pair_idx + 1}: variant FAILED: {e}")
                    continue

                time.sleep(0.5)
                fp_variant = read_fingerprint(pod, ns)

                if fp_seed is None or fp_variant is None:
                    print(f"    Pair {pair_idx + 1}: fingerprint missing")
                    continue

                # Compute per-layer deviation
                devs = compute_layer_deviation(fp_seed, fp_variant)
                all_deviations.append(devs)

                avg_rel = mean(d["rel_deviation"] for d in devs)
                max_rel = max(d["rel_deviation"] for d in devs)
                max_layer = max(devs, key=lambda d: d["rel_deviation"])["layer_idx"]

                print(
                    f"    Pair {pair_idx + 1}: avg_dev={avg_rel:.4f} "
                    f"max_dev={max_rel:.4f} (layer {max_layer}) "
                    f"seed={seed_ttft:.0f}ms var={var_ttft:.0f}ms"
                )

                results["measurements"].append({
                    "token_length": target_tokens,
                    "variant_type": variant_type,
                    "pair_idx": pair_idx,
                    "seed_ttft_ms": seed_ttft,
                    "variant_ttft_ms": var_ttft,
                    "layer_deviations": devs,
                })

    # Fit bathtub curve
    print(f"\n{'=' * 60}")
    print(f"  BATHTUB CURVE FIT")
    print(f"{'=' * 60}")

    if all_deviations:
        fit_result = fit_bathtub(all_deviations, num_layers)
        results["bathtub_fit"] = fit_result

        if "error" not in fit_result:
            print(f"  σ_base = {fit_result['sigma_base']:.4f} ± {fit_result['errors']['sigma_base']:.4f}")
            print(f"  σ_e    = {fit_result['sigma_e']:.4f} ± {fit_result['errors']['sigma_e']:.4f}")
            print(f"  τ_e    = {fit_result['tau_e']:.2f} ± {fit_result['errors']['tau_e']:.2f}")
            print(f"  σ_l    = {fit_result['sigma_l']:.4f} ± {fit_result['errors']['sigma_l']:.4f}")
            print(f"  τ_l    = {fit_result['tau_l']:.2f} ± {fit_result['errors']['tau_l']:.2f}")
            print(f"  R²     = {fit_result['r_squared']:.4f}")
            print(f"  L      = {fit_result['L']} layers")
        else:
            print(f"  Fit failed: {fit_result['error']}")

        # Print per-layer summary
        if "avg_deviation" in fit_result:
            print(f"\n  Per-layer average relative deviation:")
            avg_dev = fit_result["avg_deviation"]
            for i, d in enumerate(avg_dev):
                bar = "█" * int(d * 200)
                print(f"    Layer {i:2d}: {d:.4f} {bar}")
    else:
        print("  No data collected")

    # Save
    import os
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
