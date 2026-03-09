#!/usr/bin/env python3
"""Long-context throughput benchmark for SemBlend.

Measures QPS and TTFT under concurrent load with 8K-token prompts,
using XSum cluster variations to exercise SemBlend donor reuse.

Methodology:
  1. Register seed texts from N clusters as donors
  2. Run concurrent queries using cluster variations at --target-length
  3. Report QPS, TTFT p50/p99, hit rate, errors per concurrency level

Usage:
    python -m benchmarks.e2e.longctx_throughput_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --clusters-file benchmarks/data/xsum_clusters.json \
        --target-length 8192 \
        --concurrency 1,4,8 \
        --queries-per-level 16
"""
from __future__ import annotations

import asyncio
import json
import random
import sys
import time
from dataclasses import dataclass

try:
    import aiohttp
except ImportError:
    print("ERROR: pip install aiohttp")
    sys.exit(1)


@dataclass(frozen=True)
class LevelResult:
    concurrency: int
    n_ok: int
    n_errors: int
    qps: float
    ttft_p50: float
    ttft_p99: float
    wall_secs: float


def _pick_variation(cluster: dict) -> dict | None:
    by_type = {v["overlap_type"]: v for v in cluster.get("variations", [])}
    for pref in ("reorder", "partial_80", "paraphrase"):
        if pref in by_type:
            return by_type[pref]
    for v in cluster.get("variations", []):
        if v["overlap_type"] != "exact":
            return v
    return None


async def _measure_ttft(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
) -> tuple[float, str]:
    """Returns (ttft_ms, status)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": prompt,
                  "max_tokens": 5, "temperature": 0.0, "stream": True},
            timeout=aiohttp.ClientTimeout(total=300),
        ) as resp:
            if resp.status != 200:
                return (0.0, f"HTTP {resp.status}")
            async for line in resp.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded.startswith("data:") and decoded[5:].strip() != "[DONE]":
                    return ((time.monotonic() - t0) * 1000, "ok")
            return (0.0, "no_tokens")
    except Exception as e:
        return (0.0, f"error: {e}")


async def _run_level(
    endpoint: str,
    model: str,
    concurrency: int,
    prompts: list[str],
) -> LevelResult:
    sem = asyncio.Semaphore(concurrency)
    ttfts: list[float] = []
    errors = 0
    t_wall = time.monotonic()

    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    async with aiohttp.ClientSession(connector=connector) as session:
        async def _send(prompt: str) -> None:
            nonlocal errors
            async with sem:
                ttft, status = await _measure_ttft(session, endpoint, model, prompt)
                if status == "ok":
                    ttfts.append(ttft)
                else:
                    errors += 1

        await asyncio.gather(*[_send(p) for p in prompts])

    wall = time.monotonic() - t_wall
    qps = len(prompts) / wall if wall > 0 else 0
    sorted_ttfts = sorted(ttfts)
    p50 = sorted_ttfts[len(sorted_ttfts) // 2] if sorted_ttfts else 0
    p99 = sorted_ttfts[int(len(sorted_ttfts) * 0.99)] if sorted_ttfts else 0

    return LevelResult(
        concurrency=concurrency,
        n_ok=len(ttfts),
        n_errors=errors,
        qps=qps,
        ttft_p50=p50,
        ttft_p99=p99,
        wall_secs=wall,
    )


def run_longctx_throughput(
    endpoint: str,
    model: str,
    clusters: list[dict],
    target_length: int,
    concurrency_levels: list[int],
    queries_per_level: int,
    n_donors: int,
) -> dict:
    length_clusters = [c for c in clusters if c["target_token_length"] == target_length]
    if not length_clusters:
        print(f"No clusters for length {target_length}")
        return {}

    print(f"\nLong-Context Throughput Benchmark")
    print(f"  Model: {model}")
    print(f"  Target length: {target_length}")
    print(f"  Concurrency levels: {concurrency_levels}")
    print(f"  Queries per level: {queries_per_level}")
    print(f"  Donors: {n_donors}")
    print(f"  Available clusters: {len(length_clusters)}\n")

    # Register donors
    import requests
    donor_clusters = length_clusters[:n_donors]
    print(f"Registering {n_donors} donors...")
    for i, cluster in enumerate(donor_clusters):
        requests.post(
            f"{endpoint}/v1/completions",
            json={"model": model, "prompt": cluster["seed_text"],
                  "max_tokens": 1, "temperature": 0.0},
            timeout=300,
        ).raise_for_status()
        print(f"  Donor [{i+1}/{n_donors}] registered")
        time.sleep(0.3)

    time.sleep(3)
    print("Donors registered, settling 3s\n")

    # Build query pool from variations
    rng = random.Random(42)
    results = {}

    for level in concurrency_levels:
        # Build query batch
        prompts = []
        for i in range(queries_per_level):
            cluster = donor_clusters[i % len(donor_clusters)]
            var = _pick_variation(cluster)
            if var:
                prompts.append(var["text"])
            else:
                prompts.append(cluster["seed_text"])
        rng.shuffle(prompts)

        print(f"{'='*60}")
        print(f"Concurrency = {level}")
        print(f"{'='*60}")

        result = asyncio.run(_run_level(endpoint, model, level, prompts))

        print(f"  QPS:      {result.qps:.2f}")
        print(f"  TTFT P50: {result.ttft_p50:.0f}ms")
        print(f"  TTFT P99: {result.ttft_p99:.0f}ms")
        print(f"  OK/Total: {result.n_ok}/{result.n_ok + result.n_errors}")
        print(f"  Errors:   {result.n_errors}\n")

        results[level] = result

    print(f"\n{'='*60}")
    print(f"{'Conc':>6} {'QPS':>8} {'P50ms':>8} {'P99ms':>8} {'Err':>5}")
    print(f"{'-'*60}")
    for level in concurrency_levels:
        if level in results:
            r = results[level]
            print(f"{r.concurrency:>6} {r.qps:>8.2f} {r.ttft_p50:>8.0f} {r.ttft_p99:>8.0f} {r.n_errors:>5}")
    print(f"{'='*60}")

    return {
        "model": model,
        "target_length": target_length,
        "n_donors": n_donors,
        "results": {
            str(level): {
                "concurrency": r.concurrency,
                "qps": r.qps,
                "ttft_p50": r.ttft_p50,
                "ttft_p99": r.ttft_p99,
                "n_ok": r.n_ok,
                "n_errors": r.n_errors,
            }
            for level, r in results.items()
        },
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--clusters-file", required=True)
    parser.add_argument("--target-length", type=int, default=8192)
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--queries-per-level", type=int, default=16)
    parser.add_argument("--n-donors", type=int, default=8)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    with open(args.clusters_file) as f:
        clusters = json.load(f)

    concurrency_levels = [int(x) for x in args.concurrency.split(",")]

    report = run_longctx_throughput(
        endpoint=args.endpoint,
        model=args.model,
        clusters=clusters,
        target_length=args.target_length,
        concurrency_levels=concurrency_levels,
        queries_per_level=args.queries_per_level,
        n_donors=args.n_donors,
    )

    if args.output and report:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")
