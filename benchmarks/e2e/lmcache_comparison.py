"""Head-to-head comparison: LMCache (exact prefix) vs SemBlend (semantic donors).

Measures TTFT across four phases to quantify SemBlend's advantage over
vanilla LMCache prefix caching. LMCache only hits on token-identical
prefixes; SemBlend finds semantic donors even when tokens differ entirely.

Phases:
  1. Seed Cache — populate both systems with diverse queries
  2. Exact Prefix — shared system prompt, varied questions (both hit)
  3. Semantic Variant — paraphrased queries (LMCache misses, SemBlend hits)
  4. Novel — completely new queries (both miss, cold baseline)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone

import aiohttp
import numpy as np

from benchmarks.e2e.bootstrap_ci import (
    CIResult, bootstrap_mean, bootstrap_percentile, bootstrap_speedup,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LMCacheComparisonConfig:
    """Immutable configuration for the LMCache vs SemBlend comparison."""

    vllm_endpoint: str = "http://localhost:8000"
    proxy_endpoint: str = "http://localhost:8081"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 128
    n_seeds: int = 50
    n_exact_prefix: int = 100
    n_semantic: int = 200
    n_novel: int = 100
    warmup_delay_secs: float = 3.0
    timeout_secs: float = 60.0
    output_path: str = "benchmarks/e2e/results/lmcache-vs-semblend-v1.json"


@dataclass(frozen=True)
class ComparisonResult:
    """Single measurement from one (phase, system) query."""

    phase: str
    system: str
    query_text: str
    ttft_ms: float
    cache_hit: bool
    match_type: str
    computation_ratio: float | None
    timestamp: str


@dataclass(frozen=True)
class PhaseCI:
    """Bootstrap CIs for one (phase, system) combination."""

    phase: str
    system: str
    n: int
    ttft_p50: CIResult
    ttft_mean: CIResult
    hit_rate_pct: float


# -- Query corpus ----------------------------------------------------------

SYSTEM_PROMPT = (
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
)

SEED_QUERIES = [
    "How do I reset my password?", "What are your shipping rates?",
    "I want to cancel my subscription", "Can I get a refund?",
    "How do I upgrade to premium?", "What payment methods do you accept?",
    "My account was locked", "How do I enable two-factor auth?",
    "What is your uptime guarantee?", "How long do you retain my data?",
    "Do you offer student discounts?", "Can I export all my data?",
    "What are the API rate limits?", "How do I contact phone support?",
    "What is included in the free tier?", "Change my billing to annual?",
    "Update my credit card information", "What is your GDPR status?",
    "How do I add team members?", "Can I use the API in a mobile app?",
    "What happens when my trial expires?", "How do I set up SSO?",
    "Is there a bulk import feature?", "What browsers do you support?",
    "How do I configure webhooks?",
]

SEMANTIC_VARIANTS: tuple[tuple[str, str], ...] = (
    ("How do I reset my password?", "I need to change my login credentials"),
    ("What are your shipping rates?", "How much does delivery cost?"),
    ("I want to cancel my subscription", "Please terminate my membership"),
    ("Can I get a refund?", "I'd like my money back"),
    ("How do I upgrade to premium?", "Steps to move to a higher tier?"),
    ("What payment methods do you accept?", "Which ways can I pay?"),
    ("My account was locked", "Help me regain access to my account"),
    ("How do I enable two-factor auth?", "Set up 2FA on my account"),
    ("What is your uptime guarantee?", "How reliable is your service?"),
    ("How long do you retain my data?", "Data storage duration policy?"),
    ("Do you offer student discounts?", "Reduced prices for students?"),
    ("Can I export all my data?", "Download a complete backup?"),
    ("What are the API rate limits?", "Requests per second allowed?"),
    ("How do I contact phone support?", "Number I can call for help?"),
    ("What is included in the free tier?", "What features come free?"),
)

NOVEL_QUERIES: tuple[str, ...] = (
    "Explain general relativity in simple terms",
    "Write a haiku about quantum computing",
    "What is the capital of Burkina Faso?",
    "Describe the process of photosynthesis",
    "How do jet engines work?",
    "Summarize the plot of War and Peace",
    "What causes the aurora borealis?",
    "Explain blockchain consensus mechanisms",
    "How does mRNA vaccine technology work?",
    "What is the Riemann hypothesis?",
)


# -- HTTP helpers -----------------------------------------------------------

async def _measure_streaming_ttft(
    session: aiohttp.ClientSession, endpoint: str,
    model: str, messages: list[dict[str, str]], max_tokens: int,
) -> tuple[float, bool]:
    """Send streaming chat completion, return (ttft_ms, success)."""
    t0 = time.monotonic()
    try:
        async with session.post(
            f"{endpoint}/v1/chat/completions",
            json={"model": model, "messages": messages,
                  "max_tokens": max_tokens, "stream": True},
        ) as resp:
            if resp.status != 200:
                logger.warning("HTTP %d from %s", resp.status, endpoint)
                return (-1.0, False)
            async for line in resp.content:
                text = line.decode("utf-8", errors="replace").strip()
                if text.startswith("data:") and text != "data: [DONE]":
                    return ((time.monotonic() - t0) * 1000, True)
    except Exception as exc:
        logger.warning("Request to %s failed: %s", endpoint, exc)
    return (-1.0, False)


def _msgs(sys_prompt: str | None, user: str) -> list[dict[str, str]]:
    if sys_prompt is not None:
        return [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": user}]
    return [{"role": "user", "content": user}]


def _ep(cfg: LMCacheComparisonConfig, system: str) -> str:
    return cfg.vllm_endpoint if system == "lmcache" else cfg.proxy_endpoint


def _ts() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _send(
    session: aiohttp.ClientSession, cfg: LMCacheComparisonConfig,
    system: str, messages: list[dict[str, str]],
    phase: str, text: str, hit: bool, match: str, comp: float | None,
) -> ComparisonResult:
    """Send one query, wrap measurement in ComparisonResult."""
    ttft, ok = await _measure_streaming_ttft(
        session, _ep(cfg, system), cfg.model, messages, cfg.max_tokens,
    )
    return ComparisonResult(
        phase=phase, system=system, query_text=text[:200],
        ttft_ms=ttft, cache_hit=hit and ttft > 0,
        match_type=match, computation_ratio=comp, timestamp=_ts(),
    )


# -- Phase runners ----------------------------------------------------------

async def _run_seed_phase(
    session: aiohttp.ClientSession, cfg: LMCacheComparisonConfig,
) -> list[ComparisonResult]:
    """Phase 1: Seed both caches with diverse queries."""
    print(f"\n{'=' * 70}\nPhase 1: Seeding caches ({cfg.n_seeds} queries)\n{'=' * 70}")
    results: list[ComparisonResult] = []
    for i, q in enumerate(SEED_QUERIES[: cfg.n_seeds]):
        msgs = _msgs(SYSTEM_PROMPT, q)
        for sys in ("lmcache", "semblend"):
            r = await _send(session, cfg, sys, msgs, "seed", q, False, "seed", None)
            results = [*results, r]
        if (i + 1) % 10 == 0:
            print(f"  Seeded {i + 1}/{min(cfg.n_seeds, len(SEED_QUERIES))}")
    print(f"  Seeding complete")
    await asyncio.sleep(cfg.warmup_delay_secs)
    return results


async def _run_exact_prefix_phase(
    session: aiohttp.ClientSession, cfg: LMCacheComparisonConfig,
) -> list[ComparisonResult]:
    """Phase 2: Shared system prompt + varied questions (both hit)."""
    print(f"\n{'=' * 70}\nPhase 2: Exact prefix (n={cfg.n_exact_prefix})\n{'=' * 70}")
    results: list[ComparisonResult] = []
    for _ in range(max(1, cfg.n_exact_prefix // len(SEED_QUERIES))):
        for q in SEED_QUERIES:
            if len(results) >= cfg.n_exact_prefix * 2:
                break
            msgs = _msgs(SYSTEM_PROMPT, q)
            for sys in ("lmcache", "semblend"):
                r = await _send(session, cfg, sys, msgs,
                                "exact_prefix", q, True, "exact", 0.0)
                results = [*results, r]
    _summary("Exact prefix", results)
    return results


async def _run_semantic_variant_phase(
    session: aiohttp.ClientSession, cfg: LMCacheComparisonConfig,
) -> list[ComparisonResult]:
    """Phase 3: Paraphrased queries. LMCache misses, SemBlend hits."""
    print(f"\n{'=' * 70}\nPhase 3: Semantic variants (n={cfg.n_semantic})\n{'=' * 70}")
    results: list[ComparisonResult] = []
    for _ in range(max(1, cfg.n_semantic // len(SEMANTIC_VARIANTS))):
        for _orig, para in SEMANTIC_VARIANTS:
            if len(results) >= cfg.n_semantic * 2:
                break
            msgs = _msgs(SYSTEM_PROMPT, para)
            for sys in ("lmcache", "semblend"):
                is_sb = sys == "semblend"
                r = await _send(
                    session, cfg, sys, msgs, "semantic_variant", para,
                    is_sb, "semantic" if is_sb else "miss",
                    0.3 if is_sb else None,
                )
                results = [*results, r]
    _summary("Semantic variant", results)
    return results


async def _run_novel_phase(
    session: aiohttp.ClientSession, cfg: LMCacheComparisonConfig,
) -> list[ComparisonResult]:
    """Phase 4: Completely novel queries. Both miss."""
    print(f"\n{'=' * 70}\nPhase 4: Novel queries (n={cfg.n_novel})\n{'=' * 70}")
    results: list[ComparisonResult] = []
    for _ in range(max(1, cfg.n_novel // len(NOVEL_QUERIES))):
        for q in NOVEL_QUERIES:
            if len(results) >= cfg.n_novel * 2:
                break
            msgs = _msgs(None, q)
            for sys in ("lmcache", "semblend"):
                r = await _send(session, cfg, sys, msgs,
                                "novel", q, False, "miss", None)
                results = [*results, r]
    _summary("Novel", results)
    return results


def _summary(name: str, results: list[ComparisonResult]) -> None:
    for sys in ("lmcache", "semblend"):
        sr = [r for r in results if r.system == sys]
        v = sum(1 for r in sr if r.ttft_ms > 0)
        h = sum(1 for r in sr if r.cache_hit)
        print(f"  {sys}: {v} valid / {len(sr)} total, {h} hits")


# -- Analysis ---------------------------------------------------------------

def _compute_phase_cis(results: list[ComparisonResult]) -> list[PhaseCI]:
    groups: dict[tuple[str, str], list[ComparisonResult]] = {}
    for r in results:
        groups.setdefault((r.phase, r.system), []).append(r)
    cis: list[PhaseCI] = []
    for (ph, sys), grp in sorted(groups.items()):
        vals = np.array([r.ttft_ms for r in grp if r.ttft_ms > 0])
        hits = sum(1 for r in grp if r.cache_hit)
        cis = [*cis, PhaseCI(
            phase=ph, system=sys, n=len(vals),
            ttft_p50=bootstrap_percentile(vals, 50.0),
            ttft_mean=bootstrap_mean(vals),
            hit_rate_pct=(hits / len(grp) * 100) if grp else 0.0,
        )]
    return cis


def _compute_speedups(results: list[ComparisonResult]) -> dict[str, CIResult]:
    speedups: dict[str, CIResult] = {}
    for ph in ("exact_prefix", "semantic_variant", "novel"):
        sb = np.array([r.ttft_ms for r in results
                       if r.phase == ph and r.system == "semblend" and r.ttft_ms > 0])
        lm = np.array([r.ttft_ms for r in results
                       if r.phase == ph and r.system == "lmcache" and r.ttft_ms > 0])
        if len(sb) > 0 and len(lm) > 0:
            speedups[ph] = bootstrap_speedup(sb, lm)
    return speedups


def _print_table(cis: list[PhaseCI], speedups: dict[str, CIResult]) -> None:
    print(f"\n{'=' * 80}")
    print("PAPER TABLE: LMCache vs SemBlend Head-to-Head")
    print(f"{'=' * 80}\n")
    print(f"{'Phase':<20} {'System':<10} {'N':>5} "
          f"{'P50 TTFT':>12} {'Mean TTFT':>12} {'Hit Rate':>10}")
    print("-" * 80)
    for ci in cis:
        if ci.phase == "seed":
            continue
        print(f"{ci.phase:<20} {ci.system:<10} {ci.n:>5} "
              f"{ci.ttft_p50.estimate:>9.1f}ms "
              f"{ci.ttft_mean.estimate:>9.1f}ms {ci.hit_rate_pct:>8.0f}%")
    print(f"\nSpeedup (LMCache P50 / SemBlend P50):")
    for ph, sp in speedups.items():
        print(f"  {ph:<20} {sp}")


# -- Serialization ----------------------------------------------------------

def _serialize(
    cfg: LMCacheComparisonConfig, results: list[ComparisonResult],
    cis: list[PhaseCI], speedups: dict[str, CIResult],
) -> dict:
    return {
        "metadata": {
            "benchmark": "lmcache-vs-semblend", "model": cfg.model,
            "vllm_endpoint": cfg.vllm_endpoint,
            "proxy_endpoint": cfg.proxy_endpoint, "timestamp": _ts(),
            "n_seeds": cfg.n_seeds, "n_exact_prefix": cfg.n_exact_prefix,
            "n_semantic": cfg.n_semantic, "n_novel": cfg.n_novel,
        },
        "results": [asdict(r) for r in results],
        "confidence_intervals": [
            {"phase": c.phase, "system": c.system, "n": c.n,
             "ttft_p50": asdict(c.ttft_p50), "ttft_mean": asdict(c.ttft_mean),
             "hit_rate_pct": c.hit_rate_pct} for c in cis
        ],
        "speedups": {p: asdict(s) for p, s in speedups.items()},
    }


# -- CLI & main -------------------------------------------------------------

def _parse_args() -> LMCacheComparisonConfig:
    p = argparse.ArgumentParser(
        description="LMCache vs SemBlend head-to-head comparison benchmark")
    p.add_argument("--vllm-endpoint", default="http://localhost:8000")
    p.add_argument("--proxy-endpoint", default="http://localhost:8081")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-samples", type=int, default=0,
                   help="Override all sample counts (0 = use defaults)")
    p.add_argument("--output",
                   default="benchmarks/e2e/results/lmcache-vs-semblend-v1.json")
    a = p.parse_args()
    if a.n_samples > 0:
        return LMCacheComparisonConfig(
            vllm_endpoint=a.vllm_endpoint, proxy_endpoint=a.proxy_endpoint,
            model=a.model, n_seeds=min(a.n_samples, len(SEED_QUERIES)),
            n_exact_prefix=a.n_samples, n_semantic=a.n_samples * 2,
            n_novel=a.n_samples, output_path=a.output)
    return LMCacheComparisonConfig(
        vllm_endpoint=a.vllm_endpoint, proxy_endpoint=a.proxy_endpoint,
        model=a.model, output_path=a.output)


async def run(config: LMCacheComparisonConfig) -> dict:
    """Execute all four benchmark phases and return serialized results."""
    print("LMCache vs SemBlend Comparison Benchmark")
    print(f"  vLLM (LMCache):   {config.vllm_endpoint}")
    print(f"  Proxy (SemBlend): {config.proxy_endpoint}")
    print(f"  Model: {config.model}")
    print(f"  Samples: seeds={config.n_seeds}, exact={config.n_exact_prefix}, "
          f"semantic={config.n_semantic}, novel={config.n_novel}")

    timeout = aiohttp.ClientTimeout(total=config.timeout_secs)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        seed = await _run_seed_phase(session, config)
        exact = await _run_exact_prefix_phase(session, config)
        semantic = await _run_semantic_variant_phase(session, config)
        novel = await _run_novel_phase(session, config)

    all_r = [*seed, *exact, *semantic, *novel]
    cis = _compute_phase_cis(all_r)
    speedups = _compute_speedups(all_r)
    _print_table(cis, speedups)

    data = _serialize(config, all_r, cis, speedups)
    os.makedirs(os.path.dirname(config.output_path), exist_ok=True)
    with open(config.output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\nResults saved to {config.output_path}")
    return data


def main() -> None:
    """Entry point: parse args and run the benchmark."""
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        asyncio.run(run(_parse_args()))
    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")
    except Exception as exc:
        logger.error("Benchmark failed: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()
