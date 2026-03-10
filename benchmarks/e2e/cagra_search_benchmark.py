#!/usr/bin/env python3
"""CAGRA vs numpy donor search benchmark for SemBlend.

Measures ANN search latency at scale across three backends:
  (a) numpy cosine scan — O(N) brute force, current production
  (b) cuVS brute_force — GPU-accelerated exact search
  (c) cuVS CAGRA — GPU-accelerated ANN, sub-ms at any N

Run this script INSIDE the vLLM container (on a GPU node) to get real GPU numbers.

Usage (on GPU pod):
    python -m benchmarks.e2e.cagra_search_benchmark \
        --donor-counts "100,1000,10000,100000,1000000" \
        --dim 384 --query-count 1000

Or via kubectl exec:
    POD=$(kubectl get pods -n autoresearch -l app=vllm --no-headers | awk '{print $1}')
    kubectl exec -n autoresearch $POD -- python3 /tmp/cagra_search_benchmark.py

Outputs a parseable results block:
    ---
    backend: cagra
    n_donors_10000_p50_ms: 0.12
    n_donors_100000_p50_ms: 0.13
    n_donors_1000000_p50_ms: 0.14
    numpy_10000_p50_ms: 1.1
    numpy_100000_p50_ms: 11.0
    cagra_recall_at10: 0.998
    ---
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np


def _numpy_cosine_search(
    embeddings: np.ndarray,
    query: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Brute-force cosine search on CPU via numpy matrix multiply."""
    q_norm = query / (np.linalg.norm(query) + 1e-8)
    sims = embeddings @ q_norm
    if len(sims) <= top_k:
        idx = np.argsort(-sims)
        return sims[idx], idx
    idx = np.argpartition(sims, -top_k)[-top_k:]
    idx = idx[np.argsort(-sims[idx])]
    return sims[idx], idx


def _measure_latency(
    fn,
    n_warmup: int = 10,
    n_measure: int = 100,
) -> dict[str, float]:
    """Measure call latency: warmup then measure."""
    for _ in range(n_warmup):
        fn()
    latencies_ms = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        fn()
        latencies_ms.append((time.perf_counter() - t0) * 1000)
    return {
        "p50": statistics.median(latencies_ms),
        "p99": statistics.quantiles(latencies_ms, n=100)[98],
        "mean": statistics.mean(latencies_ms),
        "min": min(latencies_ms),
    }


def benchmark_numpy(n: int, dim: int, top_k: int = 10, n_measure: int = 100) -> dict[str, float]:
    """Benchmark numpy cosine scan at N donors."""
    rng = np.random.default_rng(42)
    corpus = rng.standard_normal((n, dim)).astype(np.float32)
    # Normalize corpus rows
    norms = np.linalg.norm(corpus, axis=1, keepdims=True)
    corpus /= norms + 1e-8

    query = rng.standard_normal(dim).astype(np.float32)

    return _measure_latency(
        lambda: _numpy_cosine_search(corpus, query, top_k),
        n_warmup=10,
        n_measure=n_measure,
    )


def benchmark_cuvs_brute_force(n: int, dim: int, top_k: int = 10, n_measure: int = 100) -> dict[str, float]:
    """Benchmark cuVS brute_force exact search at N donors."""
    try:
        import cupy as cp
        from cuvs.neighbors import brute_force
    except ImportError as e:
        return {"error": str(e)}

    rng = np.random.default_rng(42)
    corpus_np = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(corpus_np, axis=1, keepdims=True)
    corpus_np /= norms + 1e-8

    query_np = rng.standard_normal((1, dim)).astype(np.float32)

    corpus_gpu = cp.asarray(corpus_np)
    query_gpu = cp.asarray(query_np)

    # Build index once
    idx = brute_force.build(corpus_gpu, metric="cosine")

    def _search():
        dists, inds = brute_force.search(idx, query_gpu, top_k)
        cp.cuda.Stream.null.synchronize()

    return _measure_latency(_search, n_warmup=20, n_measure=n_measure)


def benchmark_cagra(
    n: int,
    dim: int,
    top_k: int = 10,
    n_measure: int = 100,
    exact_neighbors: np.ndarray | None = None,
) -> dict[str, float]:
    """Benchmark cuVS CAGRA ANN search at N donors, optionally measure recall."""
    CAGRA_MIN_N = 64
    try:
        import cupy as cp
        from cuvs.neighbors import brute_force, cagra
    except ImportError as e:
        return {"error": str(e)}

    if n < CAGRA_MIN_N:
        return {"skipped": f"N={n} < CAGRA minimum {CAGRA_MIN_N}, use brute_force"}

    rng = np.random.default_rng(42)
    corpus_np = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(corpus_np, axis=1, keepdims=True)
    corpus_np /= norms + 1e-8

    query_np = rng.standard_normal((1, dim)).astype(np.float32)

    corpus_gpu = cp.asarray(corpus_np)
    query_gpu = cp.asarray(query_np)

    # Build CAGRA index
    t_build = time.perf_counter()
    build_params = cagra.IndexParams(metric="cosine")
    index = cagra.build(build_params, corpus_gpu)
    cp.cuda.Stream.null.synchronize()
    build_ms = (time.perf_counter() - t_build) * 1000

    search_params = cagra.SearchParams()

    def _search():
        dists, inds = cagra.search(search_params, index, query_gpu, top_k)
        cp.cuda.Stream.null.synchronize()

    latency = _measure_latency(_search, n_warmup=20, n_measure=n_measure)
    latency["build_ms"] = build_ms

    # Recall@top_k vs brute-force exact
    try:
        exact_idx_obj = brute_force.build(corpus_gpu, metric="cosine")
        exact_dists, exact_inds = brute_force.search(exact_idx_obj, query_gpu, top_k)
        approx_dists, approx_inds = cagra.search(search_params, index, query_gpu, top_k)

        exact_set = set(exact_inds[0].tolist())
        approx_set = set(approx_inds[0].tolist())
        recall = len(exact_set & approx_set) / len(exact_set)
        latency["recall_at_k"] = recall
    except Exception as e:
        latency["recall_error"] = str(e)

    return latency


def run_benchmark(
    donor_counts: list[int],
    dim: int,
    top_k: int = 10,
    n_measure: int = 100,
    skip_numpy_above: int = 100_000,
) -> None:
    """Run full CAGRA vs numpy benchmark across all donor counts."""
    print("\n=== SemBlend CAGRA vs Numpy Benchmark ===")
    print(f"dim={dim}, top_k={top_k}, n_measure={n_measure}")

    results: dict[str, dict] = {}

    for n in donor_counts:
        print(f"\n--- N={n:,} donors ---")

        # Numpy
        if n <= skip_numpy_above:
            try:
                npm = benchmark_numpy(n, dim, top_k, n_measure=min(n_measure, 50 if n >= 10_000 else n_measure))
                print(f"  numpy:       p50={npm['p50']:.3f}ms  p99={npm['p99']:.3f}ms")
                results[f"numpy_{n}"] = npm
            except Exception as e:
                print(f"  numpy:       ERROR: {e}")
        else:
            print(f"  numpy:       SKIPPED (N>{skip_numpy_above:,} — too slow)")

        # cuVS brute_force
        try:
            bf = benchmark_cuvs_brute_force(n, dim, top_k, n_measure=n_measure)
            if "error" in bf:
                print(f"  brute_force: UNAVAILABLE ({bf['error']})")
            else:
                print(f"  brute_force: p50={bf['p50']:.3f}ms  p99={bf['p99']:.3f}ms")
            results[f"brute_force_{n}"] = bf
        except Exception as e:
            print(f"  brute_force: ERROR: {e}")
            results[f"brute_force_{n}"] = {"error": str(e)}

        # CAGRA
        try:
            cg = benchmark_cagra(n, dim, top_k, n_measure=n_measure)
            if "error" in cg:
                print(f"  CAGRA:       UNAVAILABLE ({cg['error']})")
            elif "skipped" in cg:
                print(f"  CAGRA:       {cg['skipped']}")
            else:
                recall_str = f"  recall@{top_k}={cg.get('recall_at_k', 'N/A'):.4f}" if "recall_at_k" in cg else ""
                print(f"  CAGRA:       p50={cg['p50']:.3f}ms  p99={cg['p99']:.3f}ms  build={cg.get('build_ms', 0):.0f}ms{recall_str}")
            results[f"cagra_{n}"] = cg
        except Exception as e:
            print(f"  CAGRA:       ERROR: {e}")
            results[f"cagra_{n}"] = {"error": str(e)}

    # Print speedup table
    print("\n=== Speedup Table (numpy p50 / CAGRA p50) ===")
    print(f"{'N':>12} {'numpy p50':>12} {'CAGRA p50':>12} {'Speedup':>10} {'Recall@K':>10}")
    for n in donor_counts:
        npm_p50 = results.get(f"numpy_{n}", {}).get("p50")
        cg_p50 = results.get(f"cagra_{n}", {}).get("p50")
        recall = results.get(f"cagra_{n}", {}).get("recall_at_k")

        npm_str = f"{npm_p50:.3f}ms" if npm_p50 is not None else "—"
        cg_str = f"{cg_p50:.3f}ms" if cg_p50 is not None else "—"
        spd_str = f"{npm_p50/cg_p50:.1f}×" if npm_p50 and cg_p50 else "—"
        rec_str = f"{recall:.4f}" if recall is not None else "—"
        print(f"{n:>12,} {npm_str:>12} {cg_str:>12} {spd_str:>10} {rec_str:>10}")

    # Machine-parseable output
    print("\n---")
    print(f"benchmark:    cagra_scale")
    print(f"dim:          {dim}")
    print(f"top_k:        {top_k}")
    for n in donor_counts:
        for backend in ["numpy", "brute_force", "cagra"]:
            r = results.get(f"{backend}_{n}", {})
            if "p50" in r:
                print(f"{backend}_n{n}_p50_ms: {r['p50']:.4f}")
        if "recall_at_k" in results.get(f"cagra_{n}", {}):
            print(f"cagra_n{n}_recall_at_{top_k}: {results['cagra_' + str(n)]['recall_at_k']:.4f}")
    print("---")


def main():
    parser = argparse.ArgumentParser(description="CAGRA vs numpy donor search benchmark")
    parser.add_argument(
        "--donor-counts", default="100,1000,10000,100000",
        help="Comma-separated donor counts to benchmark",
    )
    parser.add_argument("--dim", type=int, default=384, help="Embedding dimension")
    parser.add_argument("--top-k", type=int, default=10, help="Top-K neighbors")
    parser.add_argument("--query-count", type=int, default=100, help="Number of queries per measurement")
    parser.add_argument(
        "--skip-numpy-above", type=int, default=100_000,
        help="Skip numpy benchmark for N above this (too slow)",
    )
    args = parser.parse_args()

    counts = [int(x.strip()) for x in args.donor_counts.split(",")]
    run_benchmark(
        donor_counts=counts,
        dim=args.dim,
        top_k=args.top_k,
        n_measure=args.query_count,
        skip_numpy_above=args.skip_numpy_above,
    )


if __name__ == "__main__":
    main()
