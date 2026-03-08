"""Throughput and tail-latency benchmark under concurrent load for SemBlend paper.

Measures QPS, TTFT percentiles, and warmup convergence across concurrency levels.
Produces per-query timing data suitable for plotting throughput curves, tail-latency
CDFs, and warmup convergence charts.

Three phases per concurrency level:
  1. Warmup: seed the cache with representative prompts
  2. Sweep: send N queries at the target concurrency, measuring TTFT via SSE streaming
  3. Tail analysis: at max concurrency, annotate P99 outliers for root-cause plotting
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

import aiohttp
import numpy as np

from benchmarks.e2e.bootstrap_ci import CIResult, bootstrap_mean, bootstrap_percentile

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Seed prompts for warmup + query generation
# ---------------------------------------------------------------------------

SEED_PROMPTS: tuple[str, ...] = (
    "Explain the key differences between supervised and unsupervised learning.",
    "What are the main causes of climate change and their effects?",
    "Describe how a transformer neural network processes input tokens.",
    "Summarize the history of the internet from ARPANET to modern day.",
    "Explain quantum computing in simple terms for a college student.",
    "What are the principles of clean code and why do they matter?",
    "How does photosynthesis convert sunlight into chemical energy?",
    "Describe the process of protein folding and why it matters.",
    "What is the CAP theorem and how does it affect distributed systems?",
    "Explain how mRNA vaccines work step by step.",
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ThroughputConfig:
    """Immutable configuration for the throughput benchmark."""

    proxy_endpoint: str = "http://localhost:8081"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 64
    concurrency_levels: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    queries_per_level: int = 100
    n_warmup: int = 30
    timeout_secs: float = 30.0
    output_path: str = "benchmarks/e2e/results/throughput-v1.json"


# ---------------------------------------------------------------------------
# Per-query result (frozen for immutability)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class QueryTiming:
    """Timing record for a single query."""

    query_index: int
    concurrency: int
    query_type: str  # "exact_repeat", "semantic_variant", "novel"
    prompt_prefix: str
    ttft_ms: float
    total_ms: float
    status: str  # "ok", "timeout", "error"
    error_message: str = ""
    wall_clock_offset_ms: float = 0.0  # offset from sweep start


@dataclass(frozen=True)
class ConcurrencyResult:
    """Aggregate result for one concurrency level."""

    concurrency: int
    n_queries: int
    n_ok: int
    n_errors: int
    wall_clock_secs: float
    qps: float
    ttft_p50: CIResult
    ttft_p95: CIResult
    ttft_p99: CIResult
    ttft_mean: CIResult
    timings: tuple[QueryTiming, ...]


@dataclass(frozen=True)
class TailAnalysis:
    """Tail-latency breakdown for the highest concurrency level."""

    concurrency: int
    warmup_timings: tuple[QueryTiming, ...]
    steady_state_timings: tuple[QueryTiming, ...]
    p99_threshold_ms: float
    outlier_indices: tuple[int, ...]


@dataclass(frozen=True)
class WarmupConvergence:
    """Per-query TTFT during the warmup seed phase for convergence plotting."""

    timings: tuple[QueryTiming, ...]


@dataclass(frozen=True)
class ThroughputResults:
    """Complete throughput benchmark output."""

    run_id: str
    config: ThroughputConfig
    started_at: str
    completed_at: str
    sweep_results: tuple[ConcurrencyResult, ...]
    tail_analysis: TailAnalysis | None
    warmup_convergence: WarmupConvergence


# ---------------------------------------------------------------------------
# Query generation (immutable — returns new lists)
# ---------------------------------------------------------------------------

def _build_semantic_variant(prompt: str, rng: random.Random) -> str:
    """Create a semantically similar but textually different prompt."""
    prefixes = (
        "Can you explain: ",
        "Please elaborate on the following topic: ",
        "I'd like to understand: ",
        "Give me a detailed answer about: ",
        "Help me learn about: ",
    )
    suffixes = (
        " Be thorough.",
        " Include examples.",
        " Keep it concise but complete.",
        " Explain like I'm a graduate student.",
        "",
    )
    return rng.choice(prefixes) + prompt + rng.choice(suffixes)


def _generate_query_batch(
    seeds: tuple[str, ...],
    n: int,
    rng: random.Random,
) -> tuple[tuple[str, str], ...]:
    """Generate a batch of (prompt, query_type) pairs.

    Mix: 30% exact repeat, 50% semantic variant, 20% novel.
    """
    queries: list[tuple[str, str]] = []
    for _ in range(n):
        roll = rng.random()
        if roll < 0.30:
            prompt = rng.choice(seeds)
            queries.append((prompt, "exact_repeat"))
        elif roll < 0.80:
            base = rng.choice(seeds)
            prompt = _build_semantic_variant(base, rng)
            queries.append((prompt, "semantic_variant"))
        else:
            idx = rng.randint(0, len(seeds) - 1)
            prompt = f"Novel question #{rng.randint(1000, 9999)}: {seeds[idx]}"
            queries.append((prompt, "novel"))
    return tuple(queries)


# ---------------------------------------------------------------------------
# SSE streaming TTFT measurement
# ---------------------------------------------------------------------------

async def _measure_single_query(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    max_tokens: int,
    prompt: str,
    timeout_secs: float,
) -> tuple[float, float, str, str]:
    """Send one streaming chat completion and measure TTFT.

    Returns (ttft_ms, total_ms, status, error_message).
    TTFT is measured as time until the first SSE data chunk arrives.
    """
    url = f"{endpoint}/v1/chat/completions"
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    t_start = time.monotonic()
    ttft_ms = 0.0

    try:
        timeout = aiohttp.ClientTimeout(total=timeout_secs)
        async with session.post(url, json=body, timeout=timeout) as resp:
            if resp.status != 200:
                elapsed = (time.monotonic() - t_start) * 1000
                return (elapsed, elapsed, "error", f"HTTP {resp.status}")

            first_chunk_seen = False
            async for line in resp.content:
                decoded = line.decode("utf-8", errors="replace").strip()
                if not first_chunk_seen and decoded.startswith("data:"):
                    ttft_ms = (time.monotonic() - t_start) * 1000
                    first_chunk_seen = True

            total_ms = (time.monotonic() - t_start) * 1000

            if not first_chunk_seen:
                return (total_ms, total_ms, "error", "no SSE data received")

            return (ttft_ms, total_ms, "ok", "")

    except asyncio.TimeoutError:
        elapsed = (time.monotonic() - t_start) * 1000
        return (elapsed, elapsed, "timeout", "request timed out")
    except aiohttp.ClientError as exc:
        elapsed = (time.monotonic() - t_start) * 1000
        return (elapsed, elapsed, "error", str(exc))


# ---------------------------------------------------------------------------
# Concurrency sweep for one level
# ---------------------------------------------------------------------------

async def _run_concurrency_level(
    config: ThroughputConfig,
    concurrency: int,
    queries: tuple[tuple[str, str], ...],
) -> ConcurrencyResult:
    """Execute all queries at a given concurrency level and collect timings."""
    sem = asyncio.Semaphore(concurrency)
    connector = aiohttp.TCPConnector(limit=concurrency + 4)
    timings: list[QueryTiming] = []

    sweep_start = time.monotonic()

    async with aiohttp.ClientSession(connector=connector) as session:

        async def _send(idx: int, prompt: str, query_type: str) -> QueryTiming:
            async with sem:
                wall_offset = (time.monotonic() - sweep_start) * 1000
                ttft, total, status, err = await _measure_single_query(
                    session=session,
                    endpoint=config.proxy_endpoint,
                    model=config.model,
                    max_tokens=config.max_tokens,
                    prompt=prompt,
                    timeout_secs=config.timeout_secs,
                )
                return QueryTiming(
                    query_index=idx,
                    concurrency=concurrency,
                    query_type=query_type,
                    prompt_prefix=prompt[:120],
                    ttft_ms=ttft,
                    total_ms=total,
                    status=status,
                    error_message=err,
                    wall_clock_offset_ms=wall_offset,
                )

        tasks = [
            _send(i, prompt, qtype)
            for i, (prompt, qtype) in enumerate(queries)
        ]
        timings = list(await asyncio.gather(*tasks))

    wall_clock_secs = (time.monotonic() - sweep_start)

    ok_timings = [t for t in timings if t.status == "ok"]
    n_ok = len(ok_timings)
    n_errors = len(timings) - n_ok

    qps = n_ok / wall_clock_secs if wall_clock_secs > 0 else 0.0

    ttft_values = np.array([t.ttft_ms for t in ok_timings]) if ok_timings else np.array([])

    return ConcurrencyResult(
        concurrency=concurrency,
        n_queries=len(queries),
        n_ok=n_ok,
        n_errors=n_errors,
        wall_clock_secs=wall_clock_secs,
        qps=qps,
        ttft_p50=bootstrap_percentile(ttft_values, 50.0),
        ttft_p95=bootstrap_percentile(ttft_values, 95.0),
        ttft_p99=bootstrap_percentile(ttft_values, 99.0),
        ttft_mean=bootstrap_mean(ttft_values),
        timings=tuple(timings),
    )


# ---------------------------------------------------------------------------
# Warmup phase
# ---------------------------------------------------------------------------

async def _run_warmup(
    config: ThroughputConfig,
) -> WarmupConvergence:
    """Send seed queries sequentially to populate the cache.

    Returns per-query timings for convergence plotting.
    """
    warmup_prompts = tuple(
        SEED_PROMPTS[i % len(SEED_PROMPTS)]
        for i in range(config.n_warmup)
    )
    timings: list[QueryTiming] = []

    connector = aiohttp.TCPConnector(limit=4)
    async with aiohttp.ClientSession(connector=connector) as session:
        for idx, prompt in enumerate(warmup_prompts):
            ttft, total, status, err = await _measure_single_query(
                session=session,
                endpoint=config.proxy_endpoint,
                model=config.model,
                max_tokens=config.max_tokens,
                prompt=prompt,
                timeout_secs=config.timeout_secs,
            )
            timings.append(QueryTiming(
                query_index=idx,
                concurrency=1,
                query_type="warmup_seed",
                prompt_prefix=prompt[:120],
                ttft_ms=ttft,
                total_ms=total,
                status=status,
                error_message=err,
            ))
            logger.debug(
                "Warmup %d/%d: TTFT=%.1fms status=%s",
                idx + 1, config.n_warmup, ttft, status,
            )

    return WarmupConvergence(timings=tuple(timings))


# ---------------------------------------------------------------------------
# Tail-latency analysis
# ---------------------------------------------------------------------------

def _analyze_tail_latency(
    result: ConcurrencyResult,
    warmup_count: int = 50,
) -> TailAnalysis:
    """Split timings into warmup vs steady-state and identify P99 outliers."""
    all_timings = result.timings
    warmup = tuple(all_timings[:warmup_count])
    steady = tuple(all_timings[warmup_count:])

    steady_ok = [t.ttft_ms for t in steady if t.status == "ok"]
    p99_threshold = float(np.percentile(steady_ok, 99.0)) if steady_ok else 0.0

    outlier_indices = tuple(
        t.query_index
        for t in steady
        if t.status == "ok" and t.ttft_ms >= p99_threshold
    )

    return TailAnalysis(
        concurrency=result.concurrency,
        warmup_timings=warmup,
        steady_state_timings=steady,
        p99_threshold_ms=p99_threshold,
        outlier_indices=outlier_indices,
    )


# ---------------------------------------------------------------------------
# Serialization helpers (pure functions, no mutation)
# ---------------------------------------------------------------------------

def _ci_to_dict(ci: CIResult) -> dict[str, float]:
    return {
        "estimate": ci.estimate,
        "ci_lower": ci.ci_lower,
        "ci_upper": ci.ci_upper,
    }


def _timing_to_dict(t: QueryTiming) -> dict[str, Any]:
    return {
        "query_index": t.query_index,
        "concurrency": t.concurrency,
        "query_type": t.query_type,
        "prompt_prefix": t.prompt_prefix,
        "ttft_ms": t.ttft_ms,
        "total_ms": t.total_ms,
        "status": t.status,
        "error_message": t.error_message,
        "wall_clock_offset_ms": t.wall_clock_offset_ms,
    }


def _serialize_concurrency_result(r: ConcurrencyResult) -> dict[str, Any]:
    return {
        "concurrency": r.concurrency,
        "n_queries": r.n_queries,
        "n_ok": r.n_ok,
        "n_errors": r.n_errors,
        "wall_clock_secs": round(r.wall_clock_secs, 3),
        "qps": round(r.qps, 2),
        "ttft_p50": _ci_to_dict(r.ttft_p50),
        "ttft_p95": _ci_to_dict(r.ttft_p95),
        "ttft_p99": _ci_to_dict(r.ttft_p99),
        "ttft_mean": _ci_to_dict(r.ttft_mean),
        "timings": [_timing_to_dict(t) for t in r.timings],
    }


def _serialize_results(results: ThroughputResults) -> dict[str, Any]:
    """Serialize complete results to a JSON-compatible dict."""
    tail_dict: dict[str, Any] | None = None
    if results.tail_analysis is not None:
        ta = results.tail_analysis
        tail_dict = {
            "concurrency": ta.concurrency,
            "p99_threshold_ms": round(ta.p99_threshold_ms, 2),
            "outlier_indices": list(ta.outlier_indices),
            "warmup_timings": [_timing_to_dict(t) for t in ta.warmup_timings],
            "steady_state_timings": [
                _timing_to_dict(t) for t in ta.steady_state_timings
            ],
        }

    return {
        "run_id": results.run_id,
        "started_at": results.started_at,
        "completed_at": results.completed_at,
        "config": asdict(results.config),
        "warmup_convergence": [
            _timing_to_dict(t) for t in results.warmup_convergence.timings
        ],
        "sweep_results": [
            _serialize_concurrency_result(r) for r in results.sweep_results
        ],
        "tail_analysis": tail_dict,
        "summary": _build_summary(results),
    }


def _build_summary(results: ThroughputResults) -> dict[str, Any]:
    """Build a compact summary table from sweep results."""
    rows = []
    for r in results.sweep_results:
        rows.append({
            "concurrency": r.concurrency,
            "qps": round(r.qps, 2),
            "ttft_p50_ms": round(r.ttft_p50.estimate, 1),
            "ttft_p95_ms": round(r.ttft_p95.estimate, 1),
            "ttft_p99_ms": round(r.ttft_p99.estimate, 1),
            "error_rate": round(r.n_errors / max(r.n_queries, 1), 4),
        })
    return {"concurrency_sweep": rows}


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

async def run_throughput_benchmark(
    config: ThroughputConfig | None = None,
) -> ThroughputResults:
    """Execute the full throughput benchmark.

    Steps:
      1. Warmup: seed the cache with sequential queries
      2. Sweep: measure at each concurrency level
      3. Tail analysis: deep-dive at max concurrency
    """
    cfg = config if config is not None else ThroughputConfig()
    run_id = f"throughput-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    started_at = datetime.now(timezone.utc).isoformat()
    rng = random.Random(42)

    logger.info("Starting throughput benchmark: %s", run_id)
    logger.info("  Endpoint: %s", cfg.proxy_endpoint)
    logger.info("  Model: %s", cfg.model)
    logger.info("  Concurrency levels: %s", cfg.concurrency_levels)
    logger.info("  Queries per level: %d", cfg.queries_per_level)

    # Phase 1: Warmup
    logger.info("Phase 1: Warmup (%d queries)...", cfg.n_warmup)
    warmup_convergence = await _run_warmup(cfg)
    warmup_ok = sum(1 for t in warmup_convergence.timings if t.status == "ok")
    logger.info("  Warmup complete: %d/%d ok", warmup_ok, cfg.n_warmup)

    # Phase 2: Concurrency sweep
    sweep_results: list[ConcurrencyResult] = []
    for level in cfg.concurrency_levels:
        logger.info("Phase 2: Concurrency=%d ...", level)
        queries = _generate_query_batch(SEED_PROMPTS, cfg.queries_per_level, rng)
        result = await _run_concurrency_level(cfg, level, queries)
        sweep_results.append(result)
        logger.info(
            "  c=%d: QPS=%.1f  P50=%.1fms  P95=%.1fms  P99=%.1fms  errors=%d",
            level,
            result.qps,
            result.ttft_p50.estimate,
            result.ttft_p95.estimate,
            result.ttft_p99.estimate,
            result.n_errors,
        )

    # Phase 3: Tail-latency analysis at max concurrency
    tail_analysis: TailAnalysis | None = None
    if sweep_results:
        max_result = sweep_results[-1]
        tail_analysis = _analyze_tail_latency(max_result)
        logger.info(
            "Phase 3: Tail analysis at c=%d — P99 threshold=%.1fms, %d outliers",
            tail_analysis.concurrency,
            tail_analysis.p99_threshold_ms,
            len(tail_analysis.outlier_indices),
        )

    completed_at = datetime.now(timezone.utc).isoformat()

    return ThroughputResults(
        run_id=run_id,
        config=cfg,
        started_at=started_at,
        completed_at=completed_at,
        sweep_results=tuple(sweep_results),
        tail_analysis=tail_analysis,
        warmup_convergence=warmup_convergence,
    )


def save_results(results: ThroughputResults, path: str | None = None) -> str:
    """Serialize and write results to JSON. Returns the output path."""
    output_path = path if path is not None else results.config.output_path
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    data = _serialize_results(results)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info("Results saved to %s", output_path)
    return output_path


def print_summary(results: ThroughputResults) -> None:
    """Print a human-readable summary table to stdout."""
    print(f"\n{'='*72}")
    print(f"Throughput Benchmark: {results.run_id}")
    print(f"  Model: {results.config.model}")
    print(f"  Endpoint: {results.config.proxy_endpoint}")
    print(f"{'='*72}")
    print(f"{'Concurrency':>12} {'QPS':>8} {'P50':>10} {'P95':>10} {'P99':>10} {'Errors':>8}")
    print(f"{'-'*12:>12} {'-'*8:>8} {'-'*10:>10} {'-'*10:>10} {'-'*10:>10} {'-'*8:>8}")

    for r in results.sweep_results:
        print(
            f"{r.concurrency:>12d} "
            f"{r.qps:>8.1f} "
            f"{r.ttft_p50.estimate:>8.1f}ms "
            f"{r.ttft_p95.estimate:>8.1f}ms "
            f"{r.ttft_p99.estimate:>8.1f}ms "
            f"{r.n_errors:>8d}"
        )

    if results.tail_analysis is not None:
        ta = results.tail_analysis
        print(f"\nTail Analysis (c={ta.concurrency}):")
        print(f"  P99 threshold: {ta.p99_threshold_ms:.1f}ms")
        print(f"  Outliers: {len(ta.outlier_indices)} queries")

    warmup_ok = [
        t for t in results.warmup_convergence.timings if t.status == "ok"
    ]
    if warmup_ok:
        first_5 = [t.ttft_ms for t in warmup_ok[:5]]
        last_5 = [t.ttft_ms for t in warmup_ok[-5:]]
        print(f"\nWarmup Convergence:")
        print(f"  First 5 mean TTFT: {sum(first_5)/len(first_5):.1f}ms")
        print(f"  Last 5 mean TTFT:  {sum(last_5)/len(last_5):.1f}ms")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

async def _async_main() -> None:
    """Async entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="SemBlend throughput and tail-latency benchmark",
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8081",
        help="Proxy endpoint URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for chat completions",
    )
    parser.add_argument(
        "--queries-per-level",
        type=int,
        default=100,
        help="Number of queries per concurrency level",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=64,
        help="Max tokens per completion",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/e2e/results/throughput-v1.json",
        help="Output JSON path",
    )
    args = parser.parse_args()

    config = ThroughputConfig(
        proxy_endpoint=args.endpoint,
        model=args.model,
        queries_per_level=args.queries_per_level,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )

    results = await run_throughput_benchmark(config)
    output_path = save_results(results)
    print_summary(results)
    print(f"\nResults written to: {output_path}")


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    asyncio.run(_async_main())


if __name__ == "__main__":
    main()
