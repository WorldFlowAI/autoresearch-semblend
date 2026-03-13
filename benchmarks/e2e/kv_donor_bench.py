#!/usr/bin/env python3
"""KV-tensor donor discovery benchmark.

Tests SemBlend's actual KV cache reuse pipeline:
  vLLM + SynapseKVConnector + LMCache → gateway KV cache → semantic donor lookup

Phase 1: Seed - send queries to populate the KV cache (start_save_kv stores tensors)
Phase 2: Exact - resend identical queries (Tier 1 exact hash match)
Phase 3: Semantic - send paraphrased queries (Tier 3 semantic donor discovery)
Phase 4: Novel - send unrelated queries (both tiers miss → cold baseline)
Phase 5: Cold baseline - bypass cache entirely

Measures TTFT at each phase to quantify KV-tensor-level speedup.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

import aiohttp

# Test prompts: seed + semantic variants + novel
SEED_PROMPTS = [
    "You are a customer support agent. Respond helpfully.\n\nCustomer: How do I cancel my subscription?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I need to update my billing information.",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: Can I get a refund on my last purchase?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: My order hasn't arrived yet, can you help?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: How do I change my shipping address?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I want to upgrade my plan to premium.",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: What are your business hours?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I forgot my password, how do I reset it?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: Is there a student discount available?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: How do I contact technical support?",
]

# Exact repeats of seed prompts (should hit Tier 1 exact match)
EXACT_PROMPTS = SEED_PROMPTS.copy()

# Semantic variants (different wording, same intent — should hit Tier 3 donor)
SEMANTIC_PROMPTS = [
    "You are a customer support agent. Respond helpfully.\n\nCustomer: What's the process to cancel my subscription plan?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I'd like to change my payment details please.",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: Is it possible to get my money back for the recent order?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: My package is late and hasn't been delivered.",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I need to modify the delivery address on my account.",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: Can I switch to the premium tier?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: When is your customer service team available?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: I can't log in, how to recover my account?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: Do you offer any educational pricing?",
    "You are a customer support agent. Respond helpfully.\n\nCustomer: What's the best way to reach tech support?",
]

# Novel prompts (completely different topics — should miss both tiers)
NOVEL_PROMPTS = [
    "You are a history teacher. Explain the French Revolution briefly.",
    "You are a chef. Give me a recipe for chocolate cake.",
    "You are a fitness coach. What exercises build core strength?",
    "You are a travel guide. Recommend things to do in Tokyo.",
    "You are a financial advisor. How should I start investing?",
]

# Longer prompts to amplify prefill cost differences
LONG_SEED_PROMPTS = [
    (
        "You are an expert customer support agent for a large e-commerce company. "
        "You must be professional, empathetic, and provide detailed step-by-step "
        "instructions. Always verify the customer's identity before making account "
        "changes. If you cannot resolve the issue, escalate to a supervisor.\n\n"
        "Customer order history shows 5 previous purchases in the last 30 days. "
        "The customer has been a member since 2022 with a good standing account.\n\n"
        "Customer: I placed an order three days ago for a laptop and it still shows "
        "as processing. The estimated delivery was supposed to be yesterday. Can you "
        "check what's happening with order number 12345?"
    ),
    (
        "You are an expert customer support agent for a large e-commerce company. "
        "You must be professional, empathetic, and provide detailed step-by-step "
        "instructions. Always verify the customer's identity before making account "
        "changes. If you cannot resolve the issue, escalate to a supervisor.\n\n"
        "Customer order history shows 12 previous purchases in the last 60 days. "
        "The customer has been a member since 2021 with premium membership.\n\n"
        "Customer: I received the wrong item in my shipment. I ordered blue headphones "
        "but received red ones instead. Order number 67890. I need this resolved "
        "quickly as it was a gift."
    ),
]

LONG_SEMANTIC_PROMPTS = [
    (
        "You are an expert customer support agent for a large e-commerce company. "
        "You must be professional, empathetic, and provide detailed step-by-step "
        "instructions. Always verify the customer's identity before making account "
        "changes. If you cannot resolve the issue, escalate to a supervisor.\n\n"
        "Customer order history shows 5 previous purchases in the last 30 days. "
        "The customer has been a member since 2022 with a good standing account.\n\n"
        "Customer: My laptop order from a few days ago hasn't shipped yet. It was "
        "supposed to arrive already. Order 12345 - what's the status?"
    ),
    (
        "You are an expert customer support agent for a large e-commerce company. "
        "You must be professional, empathetic, and provide detailed step-by-step "
        "instructions. Always verify the customer's identity before making account "
        "changes. If you cannot resolve the issue, escalate to a supervisor.\n\n"
        "Customer order history shows 12 previous purchases in the last 60 days. "
        "The customer has been a member since 2021 with premium membership.\n\n"
        "Customer: The item I got doesn't match what I ordered. I wanted blue "
        "headphones but got red ones. Order 67890. This needs to be fixed ASAP "
        "since it's meant to be a gift."
    ),
]


@dataclass
class QueryResult:
    prompt: str
    phase: str
    ttft_ms: float
    total_ms: float
    tokens_generated: int = 0
    error: str | None = None


@dataclass
class BenchmarkResults:
    run_id: str
    vllm_endpoint: str
    model: str
    started_at: str
    completed_at: str = ""
    phases: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


async def measure_ttft(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
) -> tuple[float, float, int]:
    """Send a chat completion request and measure TTFT via SSE streaming."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = None
    token_count = 0

    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")

        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    token_count += 1
            except json.JSONDecodeError:
                continue

    total = (time.perf_counter() - start) * 1000
    return ttft or total, total, token_count


async def run_phase(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompts: list[str],
    phase_name: str,
    repeats: int = 3,
    max_tokens: int = 50,
) -> list[QueryResult]:
    """Run a benchmark phase: send prompts and measure TTFT."""
    results = []
    for rep in range(repeats):
        for prompt in prompts:
            try:
                ttft, total, tokens = await measure_ttft(
                    session, endpoint, model, prompt, max_tokens
                )
                results.append(QueryResult(
                    prompt=prompt[:80],
                    phase=phase_name,
                    ttft_ms=ttft,
                    total_ms=total,
                    tokens_generated=tokens,
                ))
            except Exception as e:
                results.append(QueryResult(
                    prompt=prompt[:80],
                    phase=phase_name,
                    ttft_ms=-1,
                    total_ms=-1,
                    error=str(e)[:200],
                ))
        # Small delay between repeats
        await asyncio.sleep(0.5)
    return results


def compute_stats(results: list[QueryResult]) -> dict:
    """Compute P50, P95, mean, etc. from query results."""
    valid = [r.ttft_ms for r in results if r.ttft_ms > 0]
    if not valid:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "errors": len(results)}

    valid_sorted = sorted(valid)
    n = len(valid_sorted)
    p50_idx = int(n * 0.50)
    p95_idx = min(int(n * 0.95), n - 1)

    return {
        "n": n,
        "p50": valid_sorted[p50_idx],
        "p95": valid_sorted[p95_idx],
        "mean": mean(valid),
        "min": valid_sorted[0],
        "max": valid_sorted[-1],
        "errors": len(results) - n,
    }


async def run_benchmark(
    endpoint: str,
    model: str,
    output_dir: str,
    repeats: int = 3,
    include_long: bool = True,
) -> BenchmarkResults:
    """Run the full KV donor discovery benchmark."""
    run_id = f"kv-donor-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    bench = BenchmarkResults(
        run_id=run_id,
        vllm_endpoint=endpoint,
        model=model,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Verify vLLM is up
        try:
            async with session.get(f"{endpoint}/health") as resp:
                if resp.status != 200:
                    print(f"vLLM health check failed: {resp.status}")
                    return bench
            print(f"vLLM healthy at {endpoint}")
        except Exception as e:
            print(f"Cannot connect to vLLM: {e}")
            return bench

        all_seeds = SEED_PROMPTS + (LONG_SEED_PROMPTS if include_long else [])
        all_semantic = SEMANTIC_PROMPTS + (LONG_SEMANTIC_PROMPTS if include_long else [])

        # Phase 1: Seed — populate KV cache
        print(f"\n--- Phase 1: Seeding ({len(all_seeds)} prompts) ---")
        seed_results = await run_phase(
            session, endpoint, model, all_seeds, "seed", repeats=1, max_tokens=100
        )
        bench.phases["seed"] = {
            "results": [asdict(r) for r in seed_results],
            "stats": compute_stats(seed_results),
        }
        print(f"  Seed TTFT P50: {bench.phases['seed']['stats'].get('p50', 'N/A'):.0f}ms")

        # Wait for KV cache propagation
        print("  Waiting 3s for KV cache propagation...")
        await asyncio.sleep(3)

        # Phase 2: Exact repeats — should hit Tier 1 exact match
        print(f"\n--- Phase 2: Exact repeats ({len(EXACT_PROMPTS)} x {repeats}) ---")
        exact_results = await run_phase(
            session, endpoint, model, EXACT_PROMPTS, "exact", repeats=repeats
        )
        bench.phases["exact"] = {
            "results": [asdict(r) for r in exact_results],
            "stats": compute_stats(exact_results),
        }
        print(f"  Exact TTFT P50: {bench.phases['exact']['stats'].get('p50', 'N/A'):.0f}ms")

        # Phase 3: Semantic variants — should hit Tier 3 donor discovery
        print(f"\n--- Phase 3: Semantic variants ({len(all_semantic)} x {repeats}) ---")
        semantic_results = await run_phase(
            session, endpoint, model, all_semantic, "semantic", repeats=repeats
        )
        bench.phases["semantic"] = {
            "results": [asdict(r) for r in semantic_results],
            "stats": compute_stats(semantic_results),
        }
        print(f"  Semantic TTFT P50: {bench.phases['semantic']['stats'].get('p50', 'N/A'):.0f}ms")

        # Phase 4: Novel queries — both tiers miss
        print(f"\n--- Phase 4: Novel queries ({len(NOVEL_PROMPTS)} x {repeats}) ---")
        novel_results = await run_phase(
            session, endpoint, model, NOVEL_PROMPTS, "novel", repeats=repeats
        )
        bench.phases["novel"] = {
            "results": [asdict(r) for r in novel_results],
            "stats": compute_stats(novel_results),
        }
        print(f"  Novel TTFT P50: {bench.phases['novel']['stats'].get('p50', 'N/A'):.0f}ms")

    # Compute summary
    seed_p50 = bench.phases["seed"]["stats"].get("p50", 0)
    exact_p50 = bench.phases["exact"]["stats"].get("p50", 0)
    semantic_p50 = bench.phases["semantic"]["stats"].get("p50", 0)
    novel_p50 = bench.phases["novel"]["stats"].get("p50", 0)

    bench.summary = {
        "seed_p50_ms": seed_p50,
        "exact_p50_ms": exact_p50,
        "semantic_p50_ms": semantic_p50,
        "novel_p50_ms": novel_p50,
        "exact_speedup": seed_p50 / exact_p50 if exact_p50 else None,
        "semantic_speedup": seed_p50 / semantic_p50 if semantic_p50 else None,
        "novel_speedup": seed_p50 / novel_p50 if novel_p50 else None,
    }

    bench.completed_at = datetime.now(timezone.utc).isoformat()

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{run_id}.json"
    with open(result_file, "w") as f:
        json.dump(asdict(bench), f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    # Print summary
    print("\n" + "=" * 60)
    print("KV Donor Discovery Benchmark Summary")
    print("=" * 60)
    print(f"  Seed (cold) TTFT P50:     {seed_p50:.0f}ms")
    print(f"  Exact repeat TTFT P50:    {exact_p50:.0f}ms  ({bench.summary.get('exact_speedup', 0):.1f}x speedup)")
    print(f"  Semantic donor TTFT P50:  {semantic_p50:.0f}ms  ({bench.summary.get('semantic_speedup', 0):.1f}x speedup)")
    print(f"  Novel (miss) TTFT P50:    {novel_p50:.0f}ms  ({bench.summary.get('novel_speedup', 0):.1f}x speedup)")
    print("=" * 60)

    return bench


def main():
    parser = argparse.ArgumentParser(description="KV Donor Discovery Benchmark")
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8000",
        help="vLLM endpoint URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/e2e/results",
        help="Output directory for results JSON",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of repeats per phase (default: 3)",
    )
    parser.add_argument(
        "--no-long",
        action="store_true",
        help="Skip long prompts (faster but less signal)",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        output_dir=args.output,
        repeats=args.repeats,
        include_long=not args.no_long,
    ))


if __name__ == "__main__":
    main()
