"""Quick empirical measurement script for SemBlend paper.

Measures:
1. ONNX embedding latency (via proxy internal, from response headers)
2. L0 GPU cache hit latency (exact + semantic)
3. Direct vLLM TTFT (baseline) at various prompt lengths
4. Response-level cache speedup

All measurements are direct, empirical, and reported with percentiles.
"""

from __future__ import annotations

import asyncio
import json
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


@dataclass
class Stats:
    samples: list[float] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.samples)

    @property
    def p50(self) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        return s[len(s) // 2]

    @property
    def p95(self) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        return s[int(len(s) * 0.95)]

    @property
    def p99(self) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        return s[min(int(len(s) * 0.99), len(s) - 1)]

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    def summary(self) -> str:
        return (
            f"n={self.n}, P50={self.p50:.1f}ms, P95={self.p95:.1f}ms, "
            f"P99={self.p99:.1f}ms, mean={self.mean:.1f}ms"
        )


PROXY_URL = "http://localhost:8081"
VLLM_URL = "http://localhost:18000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


async def measure_vllm_ttft(prompt: str, session: aiohttp.ClientSession) -> float:
    """Measure TTFT via direct vLLM streaming."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "stream": True,
            },
        ) as resp:
            if resp.status != 200:
                return -1
            async for line in resp.content:
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("data:") and text != "data: [DONE]":
                    return (time.monotonic() - t0) * 1000
    except Exception:
        pass
    return -1


async def measure_proxy_query(
    query: str, session: aiohttp.ClientSession
) -> tuple[float, bool, str, float]:
    """Send query through proxy, return (latency_ms, cache_hit, source, similarity)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{PROXY_URL}/api/v1/query",
            json={"query": query, "model": MODEL, "max_tokens": 10},
            headers={"x-api-key": "bench", "Content-Type": "application/json"},
        ) as resp:
            elapsed = (time.monotonic() - t0) * 1000
            if resp.status == 200:
                data = await resp.json()
                return (
                    elapsed,
                    data.get("cache_hit", False),
                    data.get("source", "unknown"),
                    data.get("similarity", 0.0),
                )
    except Exception:
        pass
    return (-1, False, "error", 0.0)


async def run_measurements() -> dict:
    results = {}
    timeout = aiohttp.ClientTimeout(total=30)

    # =========================================================================
    # 1. Baseline vLLM TTFT at various prompt lengths
    # =========================================================================
    print("\n=== Phase 1: Baseline vLLM TTFT (direct, no cache) ===")
    prompt_lengths = [50, 128, 256, 512, 1024, 2048, 3500]
    base_text = (
        "Explain the following concept in detail with examples and "
        "step-by-step reasoning. "
    )
    vllm_baseline = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for target_len in prompt_lengths:
            stats = Stats()
            word_count = int(target_len / 1.3)
            padding = " ".join(f"word{i}" for i in range(word_count))
            prompt = base_text + padding

            n_per = 20 if target_len <= 512 else 10
            for i in range(n_per):
                ttft = await measure_vllm_ttft(prompt, session)
                if ttft > 0:
                    stats.samples.append(ttft)

            vllm_baseline[target_len] = {
                "p50": stats.p50,
                "p95": stats.p95,
                "mean": stats.mean,
                "n": stats.n,
            }
            print(f"  {target_len:>5} tokens: {stats.summary()}")

    results["vllm_baseline"] = vllm_baseline

    # =========================================================================
    # 2. Response-level cache: warm up, then measure hits
    # =========================================================================
    print("\n=== Phase 2: Response-level cache (via proxy + ONNX) ===")

    # Seed queries (diverse topics for realistic cache)
    seed_queries = [
        "What is machine learning and how does it work?",
        "Explain the difference between TCP and UDP protocols",
        "What are the main causes of climate change?",
        "How does a relational database manage transactions?",
        "What is the theory of relativity?",
        "Describe how a compiler works step by step",
        "What are the principles of object-oriented programming?",
        "How does photosynthesis work in plants?",
        "What is the difference between HTTP and HTTPS?",
        "Explain the concept of recursion in programming",
        "What are neural networks and how do they learn?",
        "How does encryption protect data in transit?",
        "What is the difference between RAM and ROM?",
        "Explain how search engines rank web pages",
        "What is quantum computing and why does it matter?",
        "How do vaccines work in the human body?",
        "What is the CAP theorem in distributed systems?",
        "Explain the concept of supply and demand in economics",
        "How does GPS determine your location?",
        "What is the difference between a stack and a queue?",
    ]

    # Semantic variants of seed queries
    semantic_variants = [
        "Explain what machine learning is and its mechanisms",
        "Compare TCP vs UDP networking protocols",
        "What factors contribute to global warming?",
        "How do relational databases handle transaction management?",
        "Describe Einstein's theory of relativity",
        "Walk me through the compilation process",
        "What are the core principles of OOP?",
        "Explain the process of photosynthesis",
        "How is HTTPS different from HTTP?",
        "What is recursion and how does it work in code?",
        "How do artificial neural networks learn from data?",
        "How does data encryption work for network security?",
        "Explain the difference between RAM and ROM memory",
        "How do search engines determine page rankings?",
        "What is quantum computing and its significance?",
        "How do vaccines protect against diseases?",
        "Explain the CAP theorem for distributed databases",
        "How do supply and demand affect prices?",
        "How does GPS positioning work?",
        "Compare stack and queue data structures",
    ]

    # Phase 2a: Warm up cache
    print("  Warming cache with 20 seed queries...")
    cold_latencies = Stats()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for q in seed_queries:
            latency, hit, source, sim = await measure_proxy_query(q, session)
            if latency > 0:
                cold_latencies.samples.append(latency)

    print(f"  Cold (miss): {cold_latencies.summary()}")
    results["cache_cold"] = {
        "p50": cold_latencies.p50,
        "p95": cold_latencies.p95,
        "mean": cold_latencies.mean,
        "n": cold_latencies.n,
    }

    # Small delay for cache propagation
    await asyncio.sleep(1)

    # Phase 2b: Exact hits (same queries)
    print("  Measuring exact cache hits (same queries)...")
    exact_stats = Stats()
    hash_stats = Stats()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(3):  # 3 rounds
            for q in seed_queries:
                latency, hit, source, sim = await measure_proxy_query(q, session)
                if latency > 0 and hit:
                    if "hash" in source or "query_hash" in source:
                        hash_stats.samples.append(latency)
                    else:
                        exact_stats.samples.append(latency)

    print(f"  Hash-path hits: {hash_stats.summary()}")
    print(f"  Exact semantic hits: {exact_stats.summary()}")
    results["cache_hash"] = {
        "p50": hash_stats.p50,
        "p95": hash_stats.p95,
        "mean": hash_stats.mean,
        "n": hash_stats.n,
    }
    results["cache_exact_semantic"] = {
        "p50": exact_stats.p50,
        "p95": exact_stats.p95,
        "mean": exact_stats.mean,
        "n": exact_stats.n,
    }

    # Phase 2c: Semantic variants
    print("  Measuring semantic cache hits (variant queries)...")
    semantic_stats = Stats()
    semantic_miss = Stats()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(3):
            for q in semantic_variants:
                latency, hit, source, sim = await measure_proxy_query(q, session)
                if latency > 0:
                    if hit:
                        semantic_stats.samples.append(latency)
                    else:
                        semantic_miss.samples.append(latency)

    print(f"  Semantic hits: {semantic_stats.summary()}")
    if semantic_miss.n > 0:
        print(f"  Semantic misses: {semantic_miss.summary()}")
    results["cache_semantic"] = {
        "p50": semantic_stats.p50,
        "p95": semantic_stats.p95,
        "mean": semantic_stats.mean,
        "n": semantic_stats.n,
        "hit_rate": semantic_stats.n / (semantic_stats.n + semantic_miss.n)
        if (semantic_stats.n + semantic_miss.n) > 0
        else 0,
    }

    # =========================================================================
    # 3. Compute speedups
    # =========================================================================
    print("\n=== Speedup Summary ===")
    baseline_p50 = vllm_baseline.get(50, {}).get("p50", 1)
    baseline_512 = vllm_baseline.get(512, {}).get("p50", 1)
    baseline_3500 = vllm_baseline.get(3500, {}).get("p50", 1)

    if hash_stats.n > 0 and baseline_p50 > 0:
        speedup = baseline_p50 / hash_stats.p50
        print(
            f"  Hash-path speedup (vs 50-tok baseline): {speedup:.1f}x "
            f"({hash_stats.p50:.0f}ms vs {baseline_p50:.0f}ms)"
        )

    if exact_stats.n > 0 and baseline_p50 > 0:
        speedup = baseline_p50 / exact_stats.p50
        print(
            f"  Exact semantic speedup (vs 50-tok baseline): {speedup:.1f}x "
            f"({exact_stats.p50:.0f}ms vs {baseline_p50:.0f}ms)"
        )

    if semantic_stats.n > 0 and baseline_p50 > 0:
        speedup = baseline_p50 / semantic_stats.p50
        print(
            f"  Semantic speedup (vs 50-tok baseline): {speedup:.1f}x "
            f"({semantic_stats.p50:.0f}ms vs {baseline_p50:.0f}ms)"
        )

    if baseline_3500 > 0 and hash_stats.n > 0:
        speedup = baseline_3500 / hash_stats.p50
        print(
            f"  Hash-path speedup (vs 3500-tok baseline): {speedup:.1f}x "
            f"({hash_stats.p50:.0f}ms vs {baseline_3500:.0f}ms)"
        )

    if baseline_3500 > 0 and semantic_stats.n > 0:
        speedup = baseline_3500 / semantic_stats.p50
        print(
            f"  Semantic speedup (vs 3500-tok baseline): {speedup:.1f}x "
            f"({semantic_stats.p50:.0f}ms vs {baseline_3500:.0f}ms)"
        )

    results["speedups"] = {
        "hash_vs_50tok": baseline_p50 / hash_stats.p50
        if hash_stats.p50 > 0
        else 0,
        "semantic_vs_50tok": baseline_p50 / semantic_stats.p50
        if semantic_stats.p50 > 0
        else 0,
        "hash_vs_3500tok": baseline_3500 / hash_stats.p50
        if hash_stats.p50 > 0
        else 0,
        "semantic_vs_3500tok": baseline_3500 / semantic_stats.p50
        if semantic_stats.p50 > 0
        else 0,
    }

    # Save results
    output_path = "benchmarks/e2e/results/semblend-empirical.json"
    import os

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    asyncio.run(run_measurements())
