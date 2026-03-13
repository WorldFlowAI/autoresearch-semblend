#!/usr/bin/env python3
"""Donor pool scaling benchmark for SemBlend.

Measures how hit rate and TTFT speedup improve as the number of
registered donors increases. This simulates production workloads
where the donor pool grows over time.

For each pool_size in [1, 2, 4, 8, 16, 32]:
  1. Restart clean (no donors)
  2. Register pool_size donors using seed texts from different clusters
  3. Query with variations from those clusters
  4. Measure hit rate and speedup

Usage:
    python -m benchmarks.e2e.donor_scaling_bench \
        --endpoint http://localhost:8100 \
        --clusters-file benchmarks/data/xsum_clusters.json \
        --target-length 8192 \
        --pool-sizes 1,2,4,8,16,32
"""
from __future__ import annotations

import json
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


def measure_streaming_ttft(endpoint: str, model: str, prompt: str) -> float:
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": 5, "temperature": 0.0,
            "stream": True,
        },
        timeout=300, stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8", errors="replace")
        if line_str.startswith("data: ") and line_str[6:].strip() != "[DONE]":
            ttft = (time.monotonic() - t0) * 1000
            for _ in resp.iter_lines():
                pass
            return ttft
    raise ValueError("No tokens received")


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": 1, "temperature": 0.0,
        },
        timeout=300,
    ).raise_for_status()


def _pick_variation(cluster: dict, preference: list[str] | None = None) -> dict | None:
    if preference is None:
        preference = ["reorder", "partial_80", "paraphrase"]
    by_type = {v["overlap_type"]: v for v in cluster.get("variations", [])}
    for pref in preference:
        if pref in by_type:
            return by_type[pref]
    for v in cluster.get("variations", []):
        if v["overlap_type"] != "exact":
            return v
    return None


def run_donor_scaling(
    endpoint: str,
    model: str,
    clusters: list[dict],
    target_length: int,
    pool_sizes: list[int],
    queries_per_pool: int = 8,
) -> dict:
    # Filter clusters to target length
    length_clusters = [c for c in clusters if c["target_token_length"] == target_length]
    if not length_clusters:
        print(f"No clusters for length {target_length}")
        return {}

    max_pool = max(pool_sizes)
    if len(length_clusters) < max_pool:
        print(f"WARNING: only {len(length_clusters)} clusters for max pool {max_pool}")

    print(f"\nDonor Pool Scaling Benchmark")
    print(f"  Model: {model}")
    print(f"  Target length: {target_length}")
    print(f"  Pool sizes: {pool_sizes}")
    print(f"  Queries per pool: {queries_per_pool}")
    print(f"  Available clusters: {len(length_clusters)}")
    print()

    # Warmup
    register_donor(endpoint, model, "Hello world warmup")
    time.sleep(1)

    # Measure cold baseline
    print("Measuring cold baselines...")
    cold_times = []
    for i in range(queries_per_pool):
        cluster = length_clusters[i % len(length_clusters)]
        variation = _pick_variation(cluster)
        if variation is None:
            continue
        try:
            ttft = measure_streaming_ttft(endpoint, model, variation["text"])
            cold_times.append(ttft)
            print(f"  Cold [{i+1}] {ttft:.0f}ms")
        except Exception as e:
            print(f"  Cold [{i+1}] FAILED: {e}")
        time.sleep(0.2)

    if not cold_times:
        print("No cold baselines measured")
        return {}

    cold_p50 = sorted(cold_times)[len(cold_times) // 2]
    miss_threshold = cold_p50 * 0.70
    print(f"  Cold p50: {cold_p50:.0f}ms, miss threshold: {miss_threshold:.0f}ms\n")

    results = {}

    for pool_size in pool_sizes:
        print(f"{'='*60}")
        print(f"Pool size: {pool_size}")
        print(f"{'='*60}")

        # Register donors from different clusters
        donor_clusters = length_clusters[:pool_size]
        print(f"  Registering {pool_size} donors...")
        for idx, cluster in enumerate(donor_clusters):
            try:
                register_donor(endpoint, model, cluster["seed_text"])
            except Exception as e:
                print(f"  Donor [{idx+1}] FAILED: {e}")
            time.sleep(0.3)

        print(f"  Settling 3s...")
        time.sleep(3)

        # Query with variations from the SAME clusters
        hits = 0
        total = 0
        sem_times = []
        for i in range(queries_per_pool):
            cluster = donor_clusters[i % len(donor_clusters)]
            variation = _pick_variation(cluster)
            if variation is None:
                continue
            try:
                ttft = measure_streaming_ttft(endpoint, model, variation["text"])
                sem_times.append(ttft)
                total += 1
                is_hit = ttft < miss_threshold
                if is_hit:
                    hits += 1
                speedup = cold_p50 / ttft if ttft > 0 else 0
                marker = "HIT" if is_hit else "MISS"
                print(f"  Query [{i+1}] {ttft:.0f}ms {speedup:.2f}x [{marker}]")
            except Exception as e:
                print(f"  Query [{i+1}] FAILED: {e}")
                total += 1
            time.sleep(0.2)

        hit_rate = hits / total if total > 0 else 0
        sem_p50 = sorted(sem_times)[len(sem_times) // 2] if sem_times else cold_p50
        hits_only = sorted([t for t in sem_times if t < miss_threshold])
        hit_p50 = hits_only[len(hits_only) // 2] if hits_only else sem_p50
        speedup = cold_p50 / sem_p50 if sem_p50 > 0 else 0
        hit_speedup = cold_p50 / hit_p50 if hit_p50 > 0 else 0

        results[pool_size] = {
            "pool_size": pool_size,
            "hit_rate": hit_rate,
            "hits": hits,
            "total": total,
            "speedup_p50": speedup,
            "speedup_hit_p50": hit_speedup,
            "sem_p50": sem_p50,
            "hit_p50": hit_p50,
            "cold_p50": cold_p50,
        }
        print(f"  Hit rate: {hits}/{total} ({100*hit_rate:.0f}%)")
        print(f"  Speedup: {speedup:.2f}x (hit-only: {hit_speedup:.2f}x)")
        print()

    # Summary
    print(f"\n{'='*60}")
    print(f"{'Pool':>6} {'Hit%':>6} {'Speedup':>9} {'Hit Spd':>9} {'n':>4}")
    print(f"{'-'*60}")
    for ps in pool_sizes:
        if ps in results:
            r = results[ps]
            print(f"{ps:>6} {100*r['hit_rate']:>5.0f}% {r['speedup_p50']:>8.2f}x {r['speedup_hit_p50']:>8.2f}x {r['total']:>4}")
    print(f"{'='*60}")

    return {
        "model": model,
        "target_length": target_length,
        "cold_p50": cold_p50,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", required=True)
    parser.add_argument("--target-length", type=int, default=8192)
    parser.add_argument("--pool-sizes", default="1,2,4,8,16,32")
    parser.add_argument("--queries-per-pool", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.clusters_file) as f:
        clusters = json.load(f)

    pool_sizes = [int(x) for x in args.pool_sizes.split(",")]

    report = run_donor_scaling(
        endpoint=args.endpoint,
        model=args.model,
        clusters=clusters,
        target_length=args.target_length,
        pool_sizes=pool_sizes,
        queries_per_pool=args.queries_per_pool,
    )

    if args.output and report:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
