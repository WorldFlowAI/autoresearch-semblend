#!/usr/bin/env python3
"""Chunk size ablation — live E2E across multiple chunk sizes.

Deploys vLLM+SemBlend with different LMCache chunk sizes (64, 128, 256, 512),
runs cross-instruction TTFT benchmarks at each, and compares results.

For each chunk size:
  1. Update ConfigMap with chunk_size
  2. Set LMCACHE_CHUNK_SIZE env var on the vLLM deployment
  3. Restart vLLM pods (picks up new config + env)
  4. Run E2E benchmark (cold, donor, cross-instruction)
  5. Record results

Usage:
    python -m benchmarks.e2e.chunk_size_e2e_ablation \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-articles 250 \
        --token-length 8192 \
        --chunk-sizes 64,128,256,512 \
        --output results/chunk_size_e2e_ablation.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

NAMESPACE = "autoresearch"
DEPLOYMENT = "autoresearch-synapse-vllm"
CONFIGMAP = "lmcache-config-large"

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


def _run(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run a shell command, print it, return result."""
    print(f"  $ {cmd}")
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, check=check)


def _update_chunk_size(chunk_size: int) -> None:
    """Update LMCache ConfigMap and deployment env var for new chunk size."""
    print(f"\n=== Deploying chunk_size={chunk_size} ===")

    # 1. Update ConfigMap via pipe
    config_yaml = (
        f"# LMCache config - chunk_size={chunk_size}\n"
        f"local_cpu: true\n"
        f"max_local_cpu_size: 15\n"
        f"enable_blending: false\n"
        f"chunk_size: {chunk_size}\n"
    )
    # Write to temp file, create configmap from it
    config_path = Path("/tmp/lmcache_config_ablation.yaml")
    config_path.write_text(config_yaml)
    _run(
        f"kubectl create configmap {CONFIGMAP} -n {NAMESPACE} "
        f"--from-file=lmcache_config.yaml={config_path} "
        f"--dry-run=client -o yaml | kubectl apply -f -"
    )

    # 2. Set env var on deployment
    _run(
        f"kubectl set env deployment/{DEPLOYMENT} -n {NAMESPACE} "
        f"LMCACHE_CHUNK_SIZE={chunk_size}"
    )

    # 3. Restart pods
    _run(f"kubectl rollout restart deployment/{DEPLOYMENT} -n {NAMESPACE}")
    _run(
        f"kubectl rollout status deployment/{DEPLOYMENT} -n {NAMESPACE} "
        f"--timeout=300s"
    )

    # 4. Wait for model to load
    print("  Waiting for model load...")
    time.sleep(15)


def _setup_port_forward(namespace: str) -> None:
    """Kill old port-forward and set up new one."""
    subprocess.run(
        f'pkill -f "kubectl port-forward.*-n {namespace}"',
        shell=True, capture_output=True,
    )
    time.sleep(3)
    pod = subprocess.run(
        f"kubectl get pods -n {namespace} -l app=vllm --no-headers "
        f"| awk '{{print $1}}' | head -1",
        shell=True, capture_output=True, text=True,
    ).stdout.strip()
    if not pod:
        print("ERROR: No vLLM pod found")
        sys.exit(1)
    subprocess.Popen(
        f"kubectl port-forward -n {namespace} pod/{pod} 8100:8000",
        shell=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(8)


def _health_check(endpoint: str, retries: int = 5) -> bool:
    """Wait for endpoint health."""
    for i in range(retries):
        try:
            r = requests.get(f"{endpoint}/health", timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


def ttft_request(
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


def _load_articles(n: int, token_length: int) -> list[str]:
    """Load articles from cluster data or generate synthetic."""
    data_path = Path(__file__).parent.parent / "data" / "semblend_clusters.json"
    articles: list[str] = []

    if data_path.exists():
        data = json.loads(data_path.read_text())
        clusters = [
            c for c in data
            if c.get("target_token_length") == token_length
            and c.get("source_dataset") == "xsum"
        ]
        if not clusters:
            clusters = [c for c in data if c.get("source_dataset") == "xsum"]

        for c in clusters[:n * 2]:
            seed = c.get("seed_text", "")
            marker = "Context:\n"
            idx = seed.find(marker)
            if idx < 0:
                continue
            article = seed[idx + len(marker):]
            for end in ["<|im_end|>", "\n\n<|im_end", "\n\nQuestion:"]:
                end_idx = article.rfind(end)
                if end_idx > 0:
                    article = article[:end_idx]
            articles.append(article.strip())
            if len(articles) >= n:
                break

    # Pad with synthetic if needed
    filler = (
        "Climate change effects are escalating across global regions. "
        "Scientists observe unprecedented temperature anomalies and weather "
        "disruptions. Policy responses remain fragmented despite growing "
        "evidence and urgency. "
    )
    while len(articles) < n:
        articles.append(filler * 120)

    return articles[:n]


def run_e2e_at_chunk_size(
    endpoint: str,
    model: str,
    articles: list[str],
    token_length: int,
    chunk_size: int,
) -> dict:
    """Run E2E cross-instruction benchmark at given chunk size."""
    target_chars = token_length * 4
    donor_instr = INSTRUCTION_VARIANTS[0][1]
    n = len(articles)

    # Phase 1: Cold baselines
    print(f"  Phase 1: Cold baselines (n={n})")
    cold_ttfts: list[float] = []
    for i, article in enumerate(articles):
        prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr, article=article[:target_chars]
        )
        t, ok = ttft_request(endpoint, model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"    [{i + 1}/{n}] cold {t:.0f}ms    ", end="\r")
    print()

    if not cold_ttfts:
        return {"error": "no cold measurements"}

    cold_mean = statistics.mean(cold_ttfts)
    cold_p50 = statistics.median(cold_ttfts)
    print(f"    Cold: mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Phase 2: Register donors
    print(f"  Phase 2: Register donors (n={n})")
    for i, article in enumerate(articles):
        prompt = PROMPT_TEMPLATE.format(
            instruction=donor_instr, article=article[:target_chars]
        )
        ttft_request(endpoint, model, prompt, max_tokens=50)
        print(f"    [{i + 1}/{n}] registered    ", end="\r")
    print()

    # Phase 3: Cross-instruction queries
    print("  Phase 3: Cross-instruction queries")
    all_variant_ttfts: list[float] = []
    variant_results: dict[str, dict] = {}

    for variant_name, variant_instr in INSTRUCTION_VARIANTS[1:]:
        ttfts: list[float] = []
        for i, article in enumerate(articles):
            prompt = PROMPT_TEMPLATE.format(
                instruction=variant_instr, article=article[:target_chars]
            )
            t, ok = ttft_request(endpoint, model, prompt, max_tokens=5)
            if ok:
                ttfts.append(t)
            print(f"    [{variant_name}] [{i+1}/{n}] {t:.0f}ms    ", end="\r")
        print()

        if not ttfts:
            continue

        all_variant_ttfts.extend(ttfts)
        p50 = statistics.median(ttfts)
        mean_v = statistics.mean(ttfts)
        speedup = cold_mean / mean_v if mean_v > 0 else 0.0
        hits = sum(1 for t in ttfts if t < 0.70 * cold_mean)
        hit_pct = hits / len(ttfts) * 100

        print(
            f"    {variant_name}: p50={p50:.0f}ms, speedup={speedup:.2f}x, "
            f"hits={hit_pct:.0f}%"
        )

        variant_results[variant_name] = {
            "p50_ms": round(p50, 1),
            "mean_ms": round(mean_v, 1),
            "speedup": round(speedup, 3),
            "hit_pct": round(hit_pct, 1),
            "hits": hits,
            "n": len(ttfts),
            "raw_ttfts": ttfts,
        }

    # Aggregate
    if all_variant_ttfts:
        agg_mean = statistics.mean(all_variant_ttfts)
        agg_p50 = statistics.median(all_variant_ttfts)
        agg_speedup = cold_mean / agg_mean if agg_mean > 0 else 0.0
        agg_hits = sum(1 for t in all_variant_ttfts if t < 0.70 * cold_mean)
        agg_hit_pct = agg_hits / len(all_variant_ttfts) * 100
    else:
        agg_mean = agg_p50 = agg_speedup = agg_hit_pct = 0.0
        agg_hits = 0

    return {
        "chunk_size": chunk_size,
        "cold_mean_ms": round(cold_mean, 1),
        "cold_p50_ms": round(cold_p50, 1),
        "agg_mean_ms": round(agg_mean, 1),
        "agg_p50_ms": round(agg_p50, 1),
        "agg_speedup": round(agg_speedup, 3),
        "agg_hit_pct": round(agg_hit_pct, 1),
        "agg_hits": agg_hits,
        "agg_total": len(all_variant_ttfts),
        "n_articles": len(articles),
        "variants": {
            k: {kk: vv for kk, vv in v.items() if kk != "raw_ttfts"}
            for k, v in variant_results.items()
        },
        "_raw_cold": cold_ttfts,
        "_raw_variant": all_variant_ttfts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Chunk size ablation — live E2E across multiple chunk sizes"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-articles", type=int, default=250)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument(
        "--chunk-sizes", default="64,128,256,512",
        help="Comma-separated chunk sizes to test",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    chunk_sizes = [int(x.strip()) for x in args.chunk_sizes.split(",")]

    print("=" * 72)
    print("Chunk Size E2E Ablation")
    print("=" * 72)
    print(f"  endpoint     = {args.endpoint}")
    print(f"  model        = {args.model}")
    print(f"  n_articles   = {args.n_articles}")
    print(f"  token_length = {args.token_length}")
    print(f"  chunk_sizes  = {chunk_sizes}")
    print()

    # Load articles once
    articles = _load_articles(args.n_articles, args.token_length)
    print(f"Loaded {len(articles)} articles")

    results: list[dict] = []

    for cs in chunk_sizes:
        # Deploy with this chunk size
        _update_chunk_size(cs)
        _setup_port_forward(NAMESPACE)

        if not _health_check(args.endpoint):
            print(f"  ERROR: health check failed for chunk_size={cs}, skipping")
            continue

        # Verify chunk size in pod
        pod_check = subprocess.run(
            f"kubectl exec -n {NAMESPACE} deploy/{DEPLOYMENT} -- "
            f"python3 -c \"import os; print('LMCACHE_CHUNK_SIZE=' + os.environ.get('LMCACHE_CHUNK_SIZE', 'not set'))\"",
            shell=True, capture_output=True, text=True,
        )
        print(f"  Pod env: {pod_check.stdout.strip()}")

        # Run benchmark
        result = run_e2e_at_chunk_size(
            args.endpoint, args.model, articles, args.token_length, cs
        )
        results.append(result)

        # Print summary for this chunk size
        if "error" not in result:
            print(
                f"\n  chunk_size={cs}: speedup={result['agg_speedup']:.2f}x, "
                f"hit_pct={result['agg_hit_pct']:.0f}%, "
                f"cold={result['cold_mean_ms']:.0f}ms, "
                f"variant={result['agg_mean_ms']:.0f}ms"
            )

    # Final summary table
    print()
    print("=" * 72)
    print("Summary: Chunk Size E2E Ablation")
    print("=" * 72)
    print(
        f"  {'Chunk':>6} {'Cold':>10} {'Variant':>10} {'Speedup':>8} "
        f"{'Hit%':>6} {'N':>6}"
    )
    print("-" * 54)
    for r in results:
        if "error" in r:
            print(f"  {r.get('chunk_size', '?'):>6} ERROR")
            continue
        print(
            f"  {r['chunk_size']:>6} {r['cold_mean_ms']:>9.0f}ms "
            f"{r['agg_mean_ms']:>9.0f}ms {r['agg_speedup']:>7.2f}x "
            f"{r['agg_hit_pct']:>5.0f}% {r['agg_total']:>6}"
        )
    print()

    # Bootstrap CIs
    from benchmarks.e2e.bootstrap_ci import bootstrap_mean, bootstrap_speedup

    print("Bootstrap 95% Confidence Intervals")
    print("=" * 72)
    for r in results:
        if "error" in r:
            continue
        cs = r["chunk_size"]
        cold_arr = np.array(r["_raw_cold"])
        var_arr = np.array(r["_raw_variant"])
        print(f"\n  chunk_size={cs}:")
        print(f"    Cold TTFT mean:    {bootstrap_mean(cold_arr)}")
        print(f"    Variant TTFT mean: {bootstrap_mean(var_arr)}")
        if len(var_arr) > 0:
            print(f"    Speedup:           {bootstrap_speedup(var_arr, cold_arr)}")
    print()

    # Restore default chunk size (256)
    print("Restoring default chunk_size=256...")
    _update_chunk_size(256)
    _setup_port_forward(NAMESPACE)

    if args.output:
        serializable = []
        for r in results:
            serializable.append(
                {k: v for k, v in r.items() if not k.startswith("_")}
            )
        out = {
            "endpoint": args.endpoint,
            "model": args.model,
            "n_articles": args.n_articles,
            "token_length": args.token_length,
            "chunk_sizes": chunk_sizes,
            "results": serializable,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
