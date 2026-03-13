"""Comprehensive empirical measurement for SemBlend paper.

Uses proxy-reported internal latency (not external curl timing) for accurate
cache hit measurements. Measures baseline TTFT at multiple prompt lengths.
"""

from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from dataclasses import dataclass, field

import aiohttp


PROXY_URL = "http://localhost:8081"
VLLM_URL = "http://localhost:18000"
MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


@dataclass
class Stats:
    samples: list[float] = field(default_factory=list)

    def add(self, v: float) -> None:
        if v > 0:
            self.samples.append(v)

    @property
    def n(self) -> int:
        return len(self.samples)

    def pct(self, p: float) -> float:
        if not self.samples:
            return 0
        s = sorted(self.samples)
        return s[min(int(len(s) * p), len(s) - 1)]

    @property
    def p50(self) -> float:
        return self.pct(0.50)

    @property
    def p95(self) -> float:
        return self.pct(0.95)

    @property
    def p99(self) -> float:
        return self.pct(0.99)

    @property
    def mean(self) -> float:
        return statistics.mean(self.samples) if self.samples else 0

    @property
    def min_val(self) -> float:
        return min(self.samples) if self.samples else 0

    def __str__(self) -> str:
        return (
            f"n={self.n:>3}, P50={self.p50:>7.1f}ms, P95={self.p95:>7.1f}ms, "
            f"mean={self.mean:>7.1f}ms, min={self.min_val:>7.1f}ms"
        )


async def measure_vllm_baseline(
    session: aiohttp.ClientSession,
    prompt: str,
) -> float:
    """Measure TTFT via direct vLLM (non-streaming, total latency)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1,
                "stream": False,
            },
        ) as resp:
            if resp.status == 200:
                await resp.json()
                return (time.monotonic() - t0) * 1000
    except Exception as e:
        pass
    return -1


async def proxy_query(
    session: aiohttp.ClientSession,
    query: str,
) -> dict:
    """Send query through proxy and return full response data."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{PROXY_URL}/api/v1/query",
            json={"query": query, "model": MODEL, "max_tokens": 10},
            headers={"x-api-key": "bench", "Content-Type": "application/json"},
        ) as resp:
            ext_ms = (time.monotonic() - t0) * 1000
            if resp.status == 200:
                data = await resp.json()
                return {
                    "external_ms": ext_ms,
                    "internal_ms": data.get("latency_ms", ext_ms),
                    "cache_hit": data.get("cache_hit", False),
                    "source": data.get("source", "unknown"),
                    "similarity": data.get("similarity", 0),
                    "cache_tier": data.get("cache_tier", ""),
                }
    except Exception:
        pass
    return {"external_ms": -1, "internal_ms": -1, "cache_hit": False, "source": "error"}


# Customer support queries (Bitext-style)
SEED_QUERIES = [
    "I need to cancel my subscription, how do I do that?",
    "What are your shipping rates for international orders?",
    "I received a damaged product, can I get a replacement?",
    "How do I update my billing information on my account?",
    "Can you explain your return policy for electronics?",
    "I want to upgrade my plan to the premium tier",
    "My order hasn't arrived yet, it's been two weeks",
    "How do I reset my password for the mobile app?",
    "What payment methods do you accept for online purchases?",
    "I need a refund for my recent purchase, order number 12345",
    "Can I change the delivery address for an existing order?",
    "What is the warranty period for your products?",
    "How do I contact customer support by phone?",
    "I'm having trouble logging into my account on the website",
    "What are your business hours for customer service?",
    "Can I get a price match if I find a lower price elsewhere?",
    "How do I track my package after it's been shipped?",
    "I want to report a bug in your mobile application",
    "What discounts are available for first-time customers?",
    "How do I unsubscribe from your marketing emails?",
    "Can I schedule a pickup for a return shipment?",
    "What are the differences between your basic and pro plans?",
    "I need an invoice for my last three purchases",
    "How long does standard shipping take within the country?",
    "Can I use multiple coupon codes on a single order?",
    "I want to add an item to my existing subscription box",
    "What is your policy on exchanges for clothing items?",
    "How do I enable two-factor authentication on my account?",
    "Can I transfer my subscription to another person?",
    "I need help installing your desktop software on Windows",
]

# Semantic variants
VARIANT_QUERIES = [
    "How can I cancel my membership subscription?",
    "What do you charge for shipping to other countries?",
    "My item arrived broken, what can I do about it?",
    "How can I change my payment details on the account?",
    "What's the return policy on electronic devices?",
    "I'd like to move to your premium subscription level",
    "My package still hasn't shown up after two weeks of waiting",
    "I forgot my password for the app, how do I reset it?",
    "Which payment options do you offer for purchases?",
    "I'd like my money back for order number 12345",
    "Is it possible to modify the shipping address on my order?",
    "How long is the warranty on items you sell?",
    "What's the phone number for reaching customer support?",
    "I can't seem to sign into my account on your site",
    "When is your customer support team available?",
    "Do you offer price matching against competitors?",
    "Where can I find tracking information for my shipment?",
    "I found a bug in the mobile app and want to report it",
    "Are there any deals for new customers signing up?",
    "How do I stop receiving promotional emails from you?",
    "Can you arrange a pickup for my return?",
    "What features come with basic versus professional plans?",
    "I need receipts for my recent orders",
    "How many days does domestic standard delivery take?",
    "Can I stack discount codes on one purchase?",
    "I'd like to add something to my subscription",
    "What's your exchange policy for apparel?",
    "How do I set up 2FA on my user account?",
    "Can I give my subscription to someone else?",
    "I need assistance setting up the software on my PC",
]


async def run() -> dict:
    results: dict = {}
    timeout = aiohttp.ClientTimeout(total=60)

    # =========================================================================
    # Phase 1: Baseline vLLM TTFT at various prompt lengths
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Baseline vLLM TTFT (direct, no cache)")
    print("=" * 70)

    prompt_lengths = [50, 128, 256, 512, 1024]
    base_text = (
        "Explain the following concept in detail with examples and "
        "step-by-step reasoning. "
    )
    vllm_results = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for tlen in prompt_lengths:
            stats = Stats()
            word_count = int(tlen / 1.3)
            padding = " ".join(f"word{i}" for i in range(word_count))
            prompt = base_text + padding
            n_per = 30 if tlen <= 256 else 15
            for _ in range(n_per):
                ttft = await measure_vllm_baseline(session, prompt)
                stats.add(ttft)
            vllm_results[tlen] = {
                "p50": stats.p50, "p95": stats.p95, "mean": stats.mean, "n": stats.n
            }
            print(f"  {tlen:>5} tokens: {stats}")

    results["baseline_ttft"] = vllm_results

    # =========================================================================
    # Phase 2: Warm the response-level cache
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Warm response cache (30 seed queries)")
    print("=" * 70)

    cold_internal = Stats()
    cold_external = Stats()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for q in SEED_QUERIES:
            r = await proxy_query(session, q)
            cold_internal.add(r["internal_ms"])
            cold_external.add(r["external_ms"])

    print(f"  Cold internal: {cold_internal}")
    print(f"  Cold external: {cold_external}")
    results["cold_miss"] = {
        "internal_p50": cold_internal.p50,
        "internal_mean": cold_internal.mean,
        "external_p50": cold_external.p50,
        "n": cold_internal.n,
    }

    await asyncio.sleep(2)

    # =========================================================================
    # Phase 3: Measure exact cache hits
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Exact cache hits (repeat queries, 3 rounds)")
    print("=" * 70)

    exact_internal = Stats()
    exact_external = Stats()

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(3):
            for q in SEED_QUERIES:
                r = await proxy_query(session, q)
                if r["cache_hit"]:
                    exact_internal.add(r["internal_ms"])
                    exact_external.add(r["external_ms"])

    print(f"  Internal: {exact_internal}")
    print(f"  External: {exact_external}")
    results["exact_hit"] = {
        "internal_p50": exact_internal.p50,
        "internal_p95": exact_internal.p95,
        "internal_mean": exact_internal.mean,
        "internal_min": exact_internal.min_val,
        "external_p50": exact_external.p50,
        "n": exact_internal.n,
    }

    # =========================================================================
    # Phase 4: Measure semantic cache hits
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Semantic cache hits (variant queries, 3 rounds)")
    print("=" * 70)

    sem_internal = Stats()
    sem_external = Stats()
    sem_miss_internal = Stats()
    n_hits = 0
    n_total = 0

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for round_i in range(3):
            for q in VARIANT_QUERIES:
                r = await proxy_query(session, q)
                n_total += 1
                if r["cache_hit"]:
                    n_hits += 1
                    sem_internal.add(r["internal_ms"])
                    sem_external.add(r["external_ms"])
                else:
                    sem_miss_internal.add(r["internal_ms"])

    hit_rate = n_hits / n_total if n_total > 0 else 0
    print(f"  Internal: {sem_internal}")
    print(f"  External: {sem_external}")
    print(f"  Hit rate: {hit_rate:.1%} ({n_hits}/{n_total})")
    if sem_miss_internal.n > 0:
        print(f"  Miss internal: {sem_miss_internal}")

    results["semantic_hit"] = {
        "internal_p50": sem_internal.p50,
        "internal_p95": sem_internal.p95,
        "internal_mean": sem_internal.mean,
        "internal_min": sem_internal.min_val,
        "external_p50": sem_external.p50,
        "n": sem_internal.n,
        "hit_rate": hit_rate,
    }

    # =========================================================================
    # Phase 5: Compute speedups and paper table
    # =========================================================================
    print("\n" + "=" * 70)
    print("PAPER TABLE: SemBlend Response-Level Cache Speedup")
    print("=" * 70)

    # Use INTERNAL latency for cache (removes port-forward overhead)
    # Use baseline TTFT for comparison
    cache_p50 = exact_internal.p50
    sem_p50 = sem_internal.p50

    print(f"\n{'Prompt Length':<15} {'Baseline P50':<15} {'Cache P50':<12} {'Speedup':<10}")
    print("-" * 55)

    speedups = {}
    for tlen in prompt_lengths:
        bl = vllm_results.get(tlen, {}).get("p50", 0)
        if bl > 0 and cache_p50 > 0:
            sp = bl / cache_p50
            print(f"  {tlen:>5} tokens   {bl:>8.1f}ms     {cache_p50:>7.1f}ms    {sp:>5.1f}x")
            speedups[str(tlen)] = sp

    results["speedups"] = speedups
    results["component_latency"] = {
        "onnx_embedding_ms": 17,  # From proxy log warmup #2
        "cagra_search_ms": "<1",  # Sub-millisecond
        "cache_hit_total_internal_ms": cache_p50,
        "semantic_match_internal_ms": sem_p50,
    }

    print(f"\nComponent latencies (from proxy logs):")
    print(f"  ONNX embedding (BGE-M3): 17ms (warmup #2 steady-state)")
    print(f"  CAGRA search: <1ms")
    print(f"  Cache hit total (internal): {cache_p50:.1f}ms")
    print(f"  Semantic match total (internal): {sem_p50:.1f}ms")

    # Save
    output = "benchmarks/e2e/results/semblend-empirical-v2.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    asyncio.run(run())
