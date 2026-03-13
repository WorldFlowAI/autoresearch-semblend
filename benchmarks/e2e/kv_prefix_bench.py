"""Empirical measurement of vLLM's KV prefix caching speedup.

Measures TTFT with and without prefix cache hits to quantify the
KV-tensor-level caching benefit. Uses streaming responses to get
accurate time-to-first-token.

Three phases:
1. Baseline: Cold TTFT at various prompt lengths (no prefix cache hits)
2. Warm-up: Send initial prompts to populate prefix cache
3. Prefix hits: Re-send prompts with shared prefixes to measure speedup

All measurements are empirical. Reports P50/P95/mean with sample sizes.
"""

from __future__ import annotations

import asyncio
import json
import os
import random
import statistics
import string
import time
from dataclasses import dataclass, field

import aiohttp


VLLM_URL = os.environ.get("VLLM_URL", "http://localhost:18000")
MODEL = os.environ.get("MODEL", "Qwen/Qwen2.5-1.5B-Instruct")


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


async def measure_ttft_streaming(
    session: aiohttp.ClientSession,
    messages: list[dict],
    max_tokens: int = 1,
) -> float:
    """Measure TTFT via streaming response (time to first data chunk)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": True,
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                return -1
            async for line in resp.content:
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("data:") and text != "data: [DONE]":
                    return (time.monotonic() - t0) * 1000
    except Exception as e:
        pass
    return -1


async def measure_ttft_nonstreaming(
    session: aiohttp.ClientSession,
    messages: list[dict],
    max_tokens: int = 1,
) -> float:
    """Measure TTFT via non-streaming response (total latency with max_tokens=1)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{VLLM_URL}/v1/chat/completions",
            json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "stream": False,
            },
        ) as resp:
            if resp.status == 200:
                await resp.json()
                return (time.monotonic() - t0) * 1000
    except Exception:
        pass
    return -1


def make_prompt(n_words: int, seed: int = 0) -> str:
    """Generate a deterministic prompt of approximately n_words."""
    rng = random.Random(seed)
    words = [
        "".join(rng.choices(string.ascii_lowercase, k=rng.randint(3, 8)))
        for _ in range(n_words)
    ]
    return "Explain the following text in detail: " + " ".join(words)


# Shared-prefix prompts for testing prefix caching.
# The system prompt is the long shared prefix; user prompts vary.
SYSTEM_PROMPT_SHORT = (
    "You are a helpful customer support assistant for a technology company. "
    "You help users with billing, technical issues, account management, and "
    "product questions. Always be polite, concise, and helpful."
)

SYSTEM_PROMPT_LONG = (
    "You are a helpful customer support assistant for a technology company. "
    "You help users with billing, technical issues, account management, and "
    "product questions. Always be polite, concise, and helpful. "
    "Here are the company policies you must follow:\n"
    "1. Refunds are available within 30 days of purchase for all products.\n"
    "2. Premium support is available 24/7 for enterprise customers.\n"
    "3. Free tier users get email support with 48-hour response time.\n"
    "4. All data is encrypted at rest and in transit using AES-256.\n"
    "5. Account deletion requests must be processed within 72 hours.\n"
    "6. Two-factor authentication is required for all admin accounts.\n"
    "7. API rate limits are 1000 requests per minute for paid plans.\n"
    "8. Custom integrations require a professional plan or above.\n"
    "9. Annual billing provides a 20% discount over monthly billing.\n"
    "10. Educational institutions receive a 50% discount on all plans.\n"
    "11. Service level agreements guarantee 99.9% uptime for enterprise.\n"
    "12. Data retention policy: logs kept for 90 days, backups for 1 year.\n"
    "13. GDPR compliance: users can request data export at any time.\n"
    "14. SOC 2 Type II certification renewed annually.\n"
    "15. Bug bounty program pays up to $10,000 for critical vulnerabilities.\n"
)

USER_QUESTIONS = [
    "How do I cancel my subscription?",
    "What's your refund policy?",
    "I can't log in to my account.",
    "How do I upgrade to premium?",
    "What are your API rate limits?",
    "Can I export my data?",
    "How do I enable two-factor auth?",
    "What's the uptime guarantee?",
    "How long do you keep my logs?",
    "Do you offer educational discounts?",
    "How do I reset my password?",
    "What payment methods do you accept?",
    "Can I change my billing cycle?",
    "How do I contact support?",
    "What's included in the free tier?",
]


async def run() -> dict:
    results: dict = {}
    timeout = aiohttp.ClientTimeout(total=120)

    # =========================================================================
    # Phase 1: Cold baseline TTFT at various prompt lengths
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Cold baseline TTFT (unique prompts, no cache reuse)")
    print("=" * 70)

    prompt_lengths = [50, 128, 256, 512, 1024]
    cold_results = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for tlen in prompt_lengths:
            stats = Stats()
            n_per = 20 if tlen <= 256 else 10
            for i in range(n_per):
                # Each prompt is unique (different seed) to avoid prefix hits
                prompt = make_prompt(int(tlen / 1.3), seed=tlen * 1000 + i)
                messages = [{"role": "user", "content": prompt}]
                ttft = await measure_ttft_nonstreaming(session, messages)
                stats.add(ttft)
            cold_results[tlen] = {
                "p50": stats.p50,
                "p95": stats.p95,
                "mean": stats.mean,
                "n": stats.n,
            }
            print(f"  {tlen:>5} tokens: {stats}")

    results["cold_baseline"] = cold_results

    # =========================================================================
    # Phase 2: Warm prefix cache (shared system prompt)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Warm prefix cache with shared system prompt")
    print("=" * 70)

    # Send initial requests to populate the prefix cache
    warmup_stats = Stats()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for q in USER_QUESTIONS[:5]:
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT_LONG},
                {"role": "user", "content": q},
            ]
            ttft = await measure_ttft_nonstreaming(session, messages)
            warmup_stats.add(ttft)

    print(f"  Warmup (first 5 queries): {warmup_stats}")
    results["warmup"] = {
        "p50": warmup_stats.p50,
        "mean": warmup_stats.mean,
        "n": warmup_stats.n,
    }

    await asyncio.sleep(1)

    # =========================================================================
    # Phase 3: Prefix cache hits (shared system prompt, new user questions)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Prefix cache hits (shared system prompt, varied questions)")
    print("=" * 70)

    prefix_hit_stats = Stats()
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for round_i in range(3):
            for q in USER_QUESTIONS:
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT_LONG},
                    {"role": "user", "content": q},
                ]
                ttft = await measure_ttft_nonstreaming(session, messages)
                prefix_hit_stats.add(ttft)

    print(f"  Prefix cache hits: {prefix_hit_stats}")
    results["prefix_hit"] = {
        "p50": prefix_hit_stats.p50,
        "p95": prefix_hit_stats.p95,
        "mean": prefix_hit_stats.mean,
        "min": prefix_hit_stats.min_val,
        "n": prefix_hit_stats.n,
    }

    # =========================================================================
    # Phase 4: Exact repeat (same prompt, same user question)
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 4: Exact repeats (full prompt cached)")
    print("=" * 70)

    exact_stats = Stats()
    fixed_messages = [
        {"role": "system", "content": SYSTEM_PROMPT_LONG},
        {"role": "user", "content": USER_QUESTIONS[0]},
    ]
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for _ in range(30):
            ttft = await measure_ttft_nonstreaming(session, fixed_messages)
            exact_stats.add(ttft)

    print(f"  Exact repeat: {exact_stats}")
    results["exact_repeat"] = {
        "p50": exact_stats.p50,
        "p95": exact_stats.p95,
        "mean": exact_stats.mean,
        "min": exact_stats.min_val,
        "n": exact_stats.n,
    }

    # =========================================================================
    # Phase 5: Prefix cache at various shared-prefix lengths
    # =========================================================================
    print("\n" + "=" * 70)
    print("Phase 5: Prefix caching at various shared-prefix lengths")
    print("=" * 70)

    prefix_lengths = [50, 100, 200, 400]
    prefix_scaling = {}

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for plen in prefix_lengths:
            # Generate a shared prefix
            prefix = make_prompt(int(plen / 1.3), seed=9999)
            suffix_questions = [
                f"Question {i}: What does this mean?",
                f"Question {i}: Summarize this.",
                f"Question {i}: What are the key points?",
            ]

            # Warm the prefix
            for sq in suffix_questions[:1]:
                messages = [
                    {"role": "system", "content": prefix},
                    {"role": "user", "content": sq},
                ]
                await measure_ttft_nonstreaming(session, messages)

            await asyncio.sleep(0.5)

            # Measure with shared prefix
            stats = Stats()
            for round_i in range(5):
                for sq in suffix_questions:
                    messages = [
                        {"role": "system", "content": prefix},
                        {"role": "user", "content": sq},
                    ]
                    ttft = await measure_ttft_nonstreaming(session, messages)
                    stats.add(ttft)

            prefix_scaling[plen] = {
                "p50": stats.p50,
                "p95": stats.p95,
                "mean": stats.mean,
                "n": stats.n,
            }
            print(f"  {plen:>4}-word prefix: {stats}")

    results["prefix_scaling"] = prefix_scaling

    # =========================================================================
    # Phase 6: Compute speedups and summary
    # =========================================================================
    print("\n" + "=" * 70)
    print("PAPER TABLE: KV Prefix Caching Speedup")
    print("=" * 70)

    # Compare cold baseline vs prefix cache hits
    cold_512 = cold_results.get(512, {}).get("p50", 0)
    cold_256 = cold_results.get(256, {}).get("p50", 0)
    prefix_p50 = prefix_hit_stats.p50
    exact_p50 = exact_stats.p50

    print(f"\n{'Metric':<30} {'P50 (ms)':<12} {'Speedup vs cold-512':<20}")
    print("-" * 65)
    print(f"  Cold baseline (512 tok)      {cold_512:>8.1f}ms    1.0x")
    print(f"  Cold baseline (256 tok)      {cold_256:>8.1f}ms    {cold_512/cold_256:.1f}x" if cold_256 > 0 else "")
    print(f"  Prefix cache hit (shared)    {prefix_p50:>8.1f}ms    {cold_512/prefix_p50:.1f}x" if prefix_p50 > 0 else "")
    print(f"  Exact repeat (full cached)   {exact_p50:>8.1f}ms    {cold_512/exact_p50:.1f}x" if exact_p50 > 0 else "")

    speedups = {}
    for tlen in prompt_lengths:
        bl = cold_results.get(tlen, {}).get("p50", 0)
        if bl > 0 and prefix_p50 > 0:
            sp = bl / prefix_p50
            speedups[str(tlen)] = sp
            print(f"\n  Speedup at {tlen} tokens: {sp:.1f}x "
                  f"({bl:.0f}ms cold → {prefix_p50:.0f}ms cached)")

    results["speedups"] = speedups
    results["summary"] = {
        "cold_512_p50": cold_512,
        "prefix_hit_p50": prefix_p50,
        "exact_repeat_p50": exact_p50,
        "prefix_speedup_512": cold_512 / prefix_p50 if prefix_p50 > 0 else 0,
        "exact_speedup_512": cold_512 / exact_p50 if exact_p50 > 0 else 0,
    }

    # Save
    output = "benchmarks/e2e/results/kv-prefix-cache-v1.json"
    os.makedirs(os.path.dirname(output), exist_ok=True)
    with open(output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output}")

    return results


if __name__ == "__main__":
    asyncio.run(run())
