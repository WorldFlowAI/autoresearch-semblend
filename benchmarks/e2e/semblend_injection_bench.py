#!/usr/bin/env python3
"""SemBlend KV injection benchmark — targeted test of donor KV reuse.

Tests the specific scenario SemBlend is designed for: prompts with high
token overlap but different prefix ordering that breaks vLLM prefix cache.

Phase 1 (SEED): Send prompts to populate donor store + LMCache
Phase 2 (REORDER): Same context chunks in different order → prefix cache
    misses but SemBlend should find donor
Phase 3 (PARTIAL): 80% overlapping context → high Jaccard, SemBlend match
Phase 4 (COLD): Completely different context → baseline cold prefill

Usage:
    python3 semblend_injection_bench.py --endpoint http://localhost:8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import aiohttp

# --- RAG chunk pool (4 independent chunks) ---

CHUNK_A = """SEMICONDUCTOR MARKET Q3 2025
The global semiconductor industry posted record revenue of $178.2 billion in Q3 2025,
a 23.4% year-over-year increase. AI accelerator chips were the primary growth driver,
generating $42.8 billion in sales (+67% YoY). NVIDIA maintained 82% market share in
data center GPUs, while AMD gained ground with MI300X reaching 14% share. The total
addressable market for AI accelerators is projected to reach $120 billion by 2027.
Memory revenue hit $48.3 billion, with HBM3E demand far outstripping supply. SK Hynix
reported a 340% jump in HBM revenue. DDR5 server adoption reached 45% penetration.
NAND flash prices stabilized after prolonged decline."""

CHUNK_B = """FOUNDRY AND PACKAGING
TSMC posted a record $23.5 billion quarterly revenue. Advanced nodes (5nm and below)
made up 69% of wafer revenue. Arizona Fab 2 construction is accelerated for 3nm volume
production by Q4 2026. Over 90% of 2nm capacity is pre-committed through 2027.
Samsung Foundry struggles with 3nm GAA yields at 55% versus TSMC's 78% at N3E.
Intel Foundry demonstrated working 18A test chips. GlobalFoundries reported strong
demand for its FDX platform. CoWoS advanced packaging remains the main bottleneck.
TSMC plans to boost CoWoS capacity 2.5x by mid-2026 but demand may triple. UCIe
chiplet interconnects gained adoption with 15 new consortium members."""

CHUNK_C = """AUTOMOTIVE AND EDGE AI
The automotive chip shortage from 2020 has largely resolved. Lead times returned to
12-16 weeks. Average silicon content per vehicle grew from $712 to $834. L3 autonomy
platforms from Mobileye, NVIDIA DRIVE, and Qualcomm Snapdragon Ride entered mass
production, each requiring 200-500 TOPS of AI compute. The auto chip market is forecast
at $78 billion by 2027. Qualcomm Snapdragon X Elite PC chips secured 35 OEM design wins.
AI PC revenue reached $1.2 billion. Edge AI deployment is accelerating, creating
opportunities for Qualcomm, MediaTek, and specialized startups. Broadcom's custom AI
XPU business tripled, with Google and Meta as primary customers."""

CHUNK_D = """GEOPOLITICS AND SUPPLY CHAIN
US-China technology restrictions continued reshaping supply chains. China invested $47
billion in new fabs, mostly at 28nm and above. The CHIPS Act committed $32 billion to 15
projects. Japan's Rapidus reached tape-out for 2nm. European Chips Act allocated 12
billion euros. AI infrastructure spending shows no slowdown. Memory manufacturers are
best positioned for HBM demand. Software ecosystem moats (CUDA, ROCm, OpenVINO) are
increasingly decisive for market share. Carbon markets expanded with EU ETS averaging
85 euros per tonne. Climate finance flows reached $1.8 trillion in 2025."""

COLD_CHUNK = """QUANTUM COMPUTING PROGRESS 2025
IBM's 1,386-qubit Flamingo processor demonstrated quantum advantage on optimization
problems. Google achieved error rates below 0.1% on its Willow chip using surface codes.
Microsoft's topological qubits reached logical error rates of 10^-6, surpassing
conventional superconducting approaches. IonQ's trapped-ion systems processed 64 logical
qubits with all-to-all connectivity. Rigetti shipped its Ankaa-3 system to two Fortune
500 companies. PsiQuantum's photonic quantum computer completed fabrication at GlobalFoundries.
The quantum computing market is projected to reach $8.5 billion by 2028, growing at 35% CAGR.
Post-quantum cryptography standards (CRYSTALS-Kyber, CRYSTALS-Dilithium) entered mandatory
adoption in US government systems. European Quantum Communication Infrastructure deployed
across 12 member states."""

QUESTION = "What are the key trends and market dynamics described in this report?"


def build_prompt(chunks: list[str], question: str, target_tokens: int) -> str:
    system = "You are a senior analyst. Use ONLY the context below.\n\n"
    suffix = f"\n\nBased on the above, answer:\n{question}"
    context = "\n\n".join(chunks)
    available = target_tokens * 4 - len(system) - len(suffix)
    while len(context) < available:
        context += "\n\n--- CONTINUED ---\n\n" + "\n\n".join(chunks)
    return system + context[:available] + suffix


@dataclass
class Result:
    phase: str
    label: str
    ttft_ms: float
    total_ms: float
    tokens_gen: int
    prompt_tokens: int
    error: str | None = None


async def measure_ttft(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 20,
) -> tuple[float, float, int, int]:
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.1,
    }
    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    prompt_tokens = 0

    async with session.post(
        f"{endpoint}/v1/chat/completions", json=body
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")
        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            payload = decoded[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    token_count += 1
                usage = chunk.get("usage")
                if usage and "prompt_tokens" in usage:
                    prompt_tokens = usage["prompt_tokens"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    t_end = time.perf_counter()
    if t_first_token is None:
        raise RuntimeError("No tokens received")
    return (
        (t_first_token - t_start) * 1000,
        (t_end - t_start) * 1000,
        token_count,
        prompt_tokens,
    )


async def run_single(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    label: str,
    prompt: str,
    phase: str,
) -> Result:
    try:
        ttft, total, toks, ptoks = await measure_ttft(
            session, endpoint, model, prompt
        )
        print(f"    {label}: TTFT={ttft:.0f}ms, prompt={ptoks}tok")
        return Result(phase, label, ttft, total, toks, ptoks)
    except Exception as e:
        print(f"    {label}: ERROR {e}")
        return Result(phase, label, -1, -1, 0, 0, str(e)[:200])


async def main(args: argparse.Namespace) -> None:
    endpoint = args.endpoint
    model = args.model
    target = args.target_tokens

    # Build prompts with different chunk orderings
    seed_prompt = build_prompt(
        [CHUNK_A, CHUNK_B, CHUNK_C, CHUNK_D], QUESTION, target
    )
    reorder_prompt = build_prompt(
        [CHUNK_C, CHUNK_A, CHUNK_D, CHUNK_B], QUESTION, target
    )
    partial_prompt = build_prompt(
        [CHUNK_A, CHUNK_B, CHUNK_C], QUESTION, target  # Missing CHUNK_D
    )
    cold_prompt = build_prompt(
        [COLD_CHUNK], QUESTION, target
    )

    print(f"SemBlend Injection Benchmark")
    print(f"  Model: {model}")
    print(f"  Target tokens: {target}")
    print(f"  Seed prompt length: ~{len(seed_prompt)} chars")

    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(f"{endpoint}/health") as resp:
            if resp.status != 200:
                print(f"vLLM unhealthy: {resp.status}")
                return
        print(f"  vLLM healthy\n")

        results = {}

        # Phase 1: SEED — cold prefill, populates caches
        print("--- SEED (cold prefill, populates donor store) ---")
        seed_r = await run_single(
            session, endpoint, model, "seed", seed_prompt, "seed"
        )
        results["seed"] = [seed_r]
        await asyncio.sleep(2)

        # Phase 1b: EXACT — verify prefix cache works
        print("\n--- EXACT (same prompt, prefix cache ceiling) ---")
        exact_results = []
        for i in range(3):
            r = await run_single(
                session, endpoint, model, f"exact-{i}",
                seed_prompt, "exact"
            )
            exact_results.append(r)
        results["exact"] = exact_results

        # Phase 2: REORDER — same chunks, different order
        # vLLM prefix cache will miss (tokens differ from position 0)
        # SemBlend should find the donor (high Jaccard overlap)
        print("\n--- REORDER (chunks in different order) ---")
        reorder_results = []
        for i in range(3):
            r = await run_single(
                session, endpoint, model, f"reorder-{i}",
                reorder_prompt, "reorder"
            )
            reorder_results.append(r)
        results["reorder"] = reorder_results

        # Phase 3: PARTIAL — 3 of 4 chunks (75% overlap)
        print("\n--- PARTIAL (3/4 chunks, 75% overlap) ---")
        partial_results = []
        for i in range(3):
            r = await run_single(
                session, endpoint, model, f"partial-{i}",
                partial_prompt, "partial"
            )
            partial_results.append(r)
        results["partial"] = partial_results

        # Phase 4: COLD — completely different domain
        print("\n--- COLD (quantum computing, no overlap) ---")
        cold_results = []
        for i in range(3):
            r = await run_single(
                session, endpoint, model, f"cold-{i}",
                cold_prompt, "cold"
            )
            cold_results.append(r)
        results["cold"] = cold_results

    # Summary
    def first_ttft(phase: str) -> float:
        r = results.get(phase, [])
        return r[0].ttft_ms if r and r[0].ttft_ms > 0 else 0

    def median_ttft(phase: str) -> float:
        valid = sorted(
            r.ttft_ms for r in results.get(phase, []) if r.ttft_ms > 0
        )
        return valid[len(valid) // 2] if valid else 0

    seed_t = first_ttft("seed")
    exact_med = median_ttft("exact")
    reorder_first = first_ttft("reorder")
    reorder_med = median_ttft("reorder")
    partial_first = first_ttft("partial")
    partial_med = median_ttft("partial")
    cold_first = first_ttft("cold")

    print("\n" + "=" * 70)
    print("SemBlend Injection Benchmark Results")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Target: ~{target} tokens")
    print("-" * 70)
    print(f"  SEED    (cold prefill)    {seed_t:8.0f}ms")
    if exact_med:
        print(f"  EXACT   (prefix cache)   {exact_med:8.0f}ms  ({seed_t/exact_med:.1f}x faster)")
    print(f"  REORDER (first request)  {reorder_first:8.0f}ms  ({seed_t/reorder_first:.1f}x)" if reorder_first else "  REORDER  error")
    print(f"  REORDER (median)         {reorder_med:8.0f}ms  ({seed_t/reorder_med:.1f}x)" if reorder_med else "  REORDER  error")
    print(f"  PARTIAL (first request)  {partial_first:8.0f}ms  ({seed_t/partial_first:.1f}x)" if partial_first else "  PARTIAL  error")
    print(f"  PARTIAL (median)         {partial_med:8.0f}ms  ({seed_t/partial_med:.1f}x)" if partial_med else "  PARTIAL  error")
    print(f"  COLD    (first request)  {cold_first:8.0f}ms  (baseline)")
    print("-" * 70)
    if seed_t and exact_med:
        print(f"  Prefix cache speedup: {seed_t/exact_med:.1f}x")
    if reorder_first and seed_t:
        if reorder_first < seed_t * 0.5:
            print(f"  SemBlend REORDER speedup: {seed_t/reorder_first:.1f}x")
        else:
            print(f"  SemBlend REORDER: no speedup (cold: {reorder_first:.0f}ms vs seed: {seed_t:.0f}ms)")
    print("=" * 70)

    # Save results
    run_id = f"semblend-injection-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    output = {
        "run_id": run_id,
        "model": model,
        "target_tokens": target,
        "results": {
            phase: [asdict(r) for r in rlist]
            for phase, rlist in results.items()
        },
    }
    import os
    os.makedirs(args.output, exist_ok=True)
    out_file = f"{args.output}/{run_id}.json"
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--target-tokens", type=int, default=8000)
    parser.add_argument("--output", default="/tmp/semblend-results")
    asyncio.run(main(parser.parse_args()))
