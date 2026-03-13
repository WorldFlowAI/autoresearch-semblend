"""Cache response quality benchmark for the SemBlend paper.

Measures three dimensions of cache quality:
  1. LLM-as-Judge: compares cached vs fresh responses using Claude as judge
  2. False Positive Rate: measures intent mismatch on Bitext holdout set
  3. Embedding Ablation: tests BGE-M3 at multiple Matryoshka dimensions

Usage:
    python -m benchmarks.e2e.quality_bench
    python -m benchmarks.e2e.quality_bench --skip-judge
    python -m benchmarks.e2e.quality_bench --n-judge-samples 50
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import aiohttp
import anthropic
import numpy as np
from rich.console import Console
from tqdm import tqdm

from benchmarks.e2e.bootstrap_ci import CIResult, bootstrap_mean, bootstrap_proportion
from benchmarks.e2e.clients.synapse_client import SynapseClient
from benchmarks.e2e.clients.vllm_client import VllmClient
from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.datasets.bitext import load_bitext

logger = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class QualityConfig:
    """Configuration for quality benchmark."""
    proxy_endpoint: str = "http://localhost:8081"
    vllm_endpoint: str = "http://localhost:8000"
    judge_model: str = "claude-sonnet-4-20250514"
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_tokens: int = 256
    n_judge_samples: int = 200
    n_false_positive_samples: int = 500
    embedding_dimensions: tuple[int, ...] = (384, 768, 1024)
    timeout_secs: float = 60.0
    output_path: str = "benchmarks/e2e/results/quality-analysis-v1.json"


@dataclass(frozen=True)
class JudgeVerdict:
    """Single LLM-as-judge evaluation result."""
    query: str
    cached_response: str
    fresh_response: str
    same_intent: int          # 0 or 1
    response_quality: int     # 1-5 Likert
    factual_consistency: int  # 1-5 Likert
    reasoning: str


@dataclass(frozen=True)
class JudgeReport:
    """Aggregated LLM-as-judge results with bootstrap CIs."""
    n_evaluated: int
    same_intent_pct: CIResult
    mean_quality: CIResult
    mean_factual_consistency: CIResult
    failures: list[dict[str, str]]


@dataclass(frozen=True)
class FalsePositiveReport:
    """Aggregated false positive rate results."""
    n_cache_hits: int
    n_wrong_intent: int
    fpr: CIResult


@dataclass(frozen=True)
class AblationPoint:
    """Results for a single embedding dimension."""
    dimension: int
    n_queries: int
    hit_rate: CIResult
    mean_quality: CIResult
    mean_embed_latency_ms: CIResult


@dataclass(frozen=True)
class QualityResults:
    """Complete quality benchmark results."""
    run_id: str
    started_at: str
    completed_at: str
    config: dict[str, object]
    judge: JudgeReport | None
    false_positive: FalsePositiveReport | None
    ablation: list[AblationPoint]


JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator comparing two LLM responses to the same user query.
Response A is from a semantic cache. Response B is freshly generated.

Evaluate Response A relative to Response B:
1. same_intent (0 or 1): Does Response A address the same intent?
2. response_quality (1-5): 5=equally helpful, 3=adequate, 1=misleading.
3. factual_consistency (1-5): 5=fully consistent, 3=differs, 1=contradicts.

Respond with ONLY a JSON object (no markdown):
{"same_intent": 0, "response_quality": 3, "factual_consistency": 4, "reasoning": "..."}"""

_ZERO_CI = CIResult(0.0, 0.0, 0.0)


def _bench_config(cfg: QualityConfig, bitext_size: int) -> BenchmarkConfig:
    """Build a BenchmarkConfig from QualityConfig for dataset loading."""
    return BenchmarkConfig(
        synapse_endpoint=cfg.proxy_endpoint, vllm_endpoint=cfg.vllm_endpoint,
        model=cfg.model, max_tokens=cfg.max_tokens, bitext_size=bitext_size,
    )


# -- Phase 1: LLM-as-Judge -------------------------------------------------

async def _call_judge(
    client: anthropic.Anthropic, model: str,
    query: str, cached: str, fresh: str,
) -> JudgeVerdict:
    """Call the judge model to evaluate a single cached vs fresh pair."""
    user_msg = (
        f"User Query:\n{query}\n\n"
        f"Response A (cached):\n{cached}\n\n"
        f"Response B (fresh):\n{fresh}"
    )
    message = client.messages.create(
        model=model, max_tokens=300, system=JUDGE_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw = message.content[0].text.strip()
    try:
        v = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Judge returned invalid JSON: {raw[:200]}") from exc

    return JudgeVerdict(
        query=query[:200], cached_response=cached[:300], fresh_response=fresh[:300],
        same_intent=int(v.get("same_intent", 0)),
        response_quality=int(v.get("response_quality", 1)),
        factual_consistency=int(v.get("factual_consistency", 1)),
        reasoning=str(v.get("reasoning", "")),
    )


async def _collect_pairs(cfg: QualityConfig) -> list[tuple[str, str, str]]:
    """Collect (query, cached_response, fresh_response) triples."""
    dataset = load_bitext(cfg.n_judge_samples * 3, _bench_config(cfg, cfg.n_judge_samples * 3))
    queries = dataset.test_queries[: cfg.n_judge_samples]
    pairs: list[tuple[str, str, str]] = []

    async with SynapseClient(
        endpoint=cfg.proxy_endpoint, model=cfg.model,
        max_tokens=cfg.max_tokens, timeout=cfg.timeout_secs,
    ) as proxy, VllmClient(
        endpoint=cfg.vllm_endpoint, model=cfg.model,
        max_tokens=cfg.max_tokens, timeout=cfg.timeout_secs,
    ) as vllm:
        for q in tqdm(queries, desc="  Collecting pairs", leave=False):
            try:
                cr = await proxy.query(q.prompt)
                fr = await vllm.query(q.prompt)
                if cr.text and fr.text:
                    pairs.append((q.prompt, cr.text, fr.text))
            except Exception:
                logger.warning("Failed to collect pair", exc_info=True)
    return pairs


async def run_judge_evaluation(cfg: QualityConfig) -> JudgeReport:
    """Run LLM-as-judge evaluation on cached vs fresh response pairs."""
    console.print("\n[bold]Phase 1: LLM-as-Judge Evaluation[/bold]")
    console.print(f"  Judge: {cfg.judge_model}  Samples: {cfg.n_judge_samples}")

    pairs = await _collect_pairs(cfg)
    console.print(f"  Collected {len(pairs)} response pairs")
    if not pairs:
        return JudgeReport(0, _ZERO_CI, _ZERO_CI, _ZERO_CI, [])

    client = anthropic.Anthropic()
    verdicts: list[JudgeVerdict] = []
    failures: list[dict[str, str]] = []

    for query, cached, fresh in tqdm(pairs, desc="  Judging", leave=False):
        try:
            v = await _call_judge(client, cfg.judge_model, query, cached, fresh)
            verdicts = [*verdicts, v]
        except Exception as exc:
            logger.warning("Judge call failed: %s", exc)
            failures = [*failures, {"query": query[:200], "error": str(exc)}]

    if not verdicts:
        return JudgeReport(0, _ZERO_CI, _ZERO_CI, _ZERO_CI, failures)

    n = len(verdicts)
    return JudgeReport(
        n_evaluated=n,
        same_intent_pct=bootstrap_proportion(sum(v.same_intent for v in verdicts), n),
        mean_quality=bootstrap_mean(np.array([v.response_quality for v in verdicts], dtype=float)),
        mean_factual_consistency=bootstrap_mean(np.array([v.factual_consistency for v in verdicts], dtype=float)),
        failures=failures[:10],
    )


# -- Phase 2: False Positive Rate -------------------------------------------

async def run_false_positive_analysis(cfg: QualityConfig) -> FalsePositiveReport:
    """Measure false positive rate on Bitext holdout intent categories.

    A false positive occurs when the cache returns a hit for a query
    whose intent category was never seeded (holdout intents).
    """
    console.print("\n[bold]Phase 2: False Positive Rate[/bold]")
    size = cfg.n_false_positive_samples * 3
    dataset = load_bitext(size, _bench_config(cfg, size))

    novel = [q for q in dataset.test_queries if q.query_type == "novel"][: cfg.n_false_positive_samples]
    console.print(f"  Novel (holdout) queries: {len(novel)}")

    n_wrong = 0
    async with SynapseClient(
        endpoint=cfg.proxy_endpoint, model=cfg.model,
        max_tokens=cfg.max_tokens, timeout=cfg.timeout_secs,
    ) as proxy:
        for q in tqdm(novel, desc="  FPR check", leave=False):
            try:
                resp = await proxy.query(q.prompt)
                if resp.cache_hit:
                    n_wrong += 1
            except Exception:
                logger.warning("FPR query failed", exc_info=True)

    total = len(novel)
    console.print(f"  Cache hits on holdout intents: {n_wrong}/{total}")
    fpr_ci = bootstrap_proportion(n_wrong, total) if total > 0 else _ZERO_CI
    return FalsePositiveReport(n_cache_hits=n_wrong, n_wrong_intent=n_wrong, fpr=fpr_ci)


# -- Phase 3: Embedding Ablation --------------------------------------------

async def _measure_dim(cfg: QualityConfig, dim: int, queries: list[str]) -> AblationPoint:
    """Measure cache performance at a specific Matryoshka dimension."""
    n_hits, n_total = 0, 0
    quality_scores: list[float] = []
    latencies: list[float] = []

    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=cfg.timeout_secs)
    ) as session:
        for text in tqdm(queries, desc=f"  dim={dim}", leave=False):
            try:
                t0 = time.monotonic()
                async with session.post(
                    f"{cfg.proxy_endpoint}/api/v1/query",
                    json={"query": text, "model": cfg.model, "max_tokens": cfg.max_tokens},
                    headers={"Content-Type": "application/json", "X-Synapse-Embed-Dim": str(dim)},
                ) as resp:
                    data = await resp.json()
                    latencies.append((time.monotonic() - t0) * 1000)
                    n_total += 1
                    if data.get("cache_hit", False):
                        n_hits += 1
                        quality_scores.append(data.get("similarity", 0.0))
            except Exception:
                logger.warning("Ablation query failed (dim=%d)", dim, exc_info=True)

    return AblationPoint(
        dimension=dim, n_queries=n_total,
        hit_rate=bootstrap_proportion(n_hits, max(n_total, 1)),
        mean_quality=bootstrap_mean(np.array(quality_scores or [0.0])),
        mean_embed_latency_ms=bootstrap_mean(np.array(latencies or [0.0])),
    )


async def run_embedding_ablation(cfg: QualityConfig) -> list[AblationPoint]:
    """Test BGE-M3 at multiple Matryoshka dimensions."""
    console.print("\n[bold]Phase 3: Embedding Ablation[/bold]")
    console.print(f"  Dimensions: {cfg.embedding_dimensions}")

    dataset = load_bitext(500, _bench_config(cfg, 500))
    prompts = [q.prompt for q in dataset.test_queries[:100]]
    console.print(f"  Queries per dimension: {len(prompts)}")

    results: list[AblationPoint] = []
    for dim in cfg.embedding_dimensions:
        pt = await _measure_dim(cfg, dim, prompts)
        results = [*results, pt]
        console.print(f"  dim={dim}: hit_rate={pt.hit_rate}, latency={pt.mean_embed_latency_ms}")
    return results


# -- Serialization & reporting ----------------------------------------------

def _ci(ci: CIResult) -> dict[str, float]:
    return {"estimate": ci.estimate, "ci_lower": ci.ci_lower, "ci_upper": ci.ci_upper}


def save_results(results: QualityResults, output_path: str) -> str:
    """Save quality benchmark results to JSON."""
    judge_d = None
    if results.judge is not None:
        jr = results.judge
        judge_d = {
            "n_evaluated": jr.n_evaluated, "same_intent_pct": _ci(jr.same_intent_pct),
            "mean_quality": _ci(jr.mean_quality),
            "mean_factual_consistency": _ci(jr.mean_factual_consistency),
            "failures": jr.failures,
        }
    fpr_d = None
    if results.false_positive is not None:
        fp = results.false_positive
        fpr_d = {"n_cache_hits": fp.n_cache_hits, "n_wrong_intent": fp.n_wrong_intent, "fpr": _ci(fp.fpr)}

    data = {
        "run_id": results.run_id, "started_at": results.started_at,
        "completed_at": results.completed_at, "config": results.config,
        "judge": judge_d, "false_positive": fpr_d,
        "ablation": [
            {"dimension": p.dimension, "n_queries": p.n_queries,
             "hit_rate": _ci(p.hit_rate), "mean_quality": _ci(p.mean_quality),
             "mean_embed_latency_ms": _ci(p.mean_embed_latency_ms)}
            for p in results.ablation
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return str(path)


def print_report(results: QualityResults) -> None:
    """Print a human-readable summary to the console."""
    console.print("\n" + "=" * 60)
    console.print("[bold]Quality Benchmark Results[/bold]")
    console.print("=" * 60)

    if results.judge is not None:
        jr = results.judge
        console.print(f"\n[bold]LLM-as-Judge[/bold] (n={jr.n_evaluated})")
        console.print(f"  Same intent:          {jr.same_intent_pct}%")
        console.print(f"  Response quality:     {jr.mean_quality}")
        console.print(f"  Factual consistency:  {jr.mean_factual_consistency}")
        if jr.failures:
            console.print(f"  Judge failures:       {len(jr.failures)}")
    if results.false_positive is not None:
        fp = results.false_positive
        console.print(f"\n[bold]False Positive Rate[/bold] (hits={fp.n_cache_hits}, wrong={fp.n_wrong_intent})")
        console.print(f"  FPR: {fp.fpr}%")
    if results.ablation:
        console.print("\n[bold]Embedding Ablation[/bold]")
        for pt in results.ablation:
            console.print(f"  {pt.dimension}-D: hit={pt.hit_rate}%, qual={pt.mean_quality}, lat={pt.mean_embed_latency_ms}ms")


# -- CLI & orchestration ----------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(description="SemBlend cache quality benchmark")
    p.add_argument("--proxy-endpoint", default="http://localhost:8081")
    p.add_argument("--vllm-endpoint", default="http://localhost:8000")
    p.add_argument("--judge-model", default="claude-sonnet-4-20250514")
    p.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--n-judge-samples", type=int, default=200)
    p.add_argument("--n-fp-samples", type=int, default=500)
    p.add_argument("--skip-judge", action="store_true", help="Skip the expensive LLM-as-judge step")
    p.add_argument("--output", default="benchmarks/e2e/results/quality-analysis-v1.json")
    return p.parse_args()


async def run_benchmark(cfg: QualityConfig, skip_judge: bool) -> QualityResults:
    """Execute all quality benchmark phases."""
    started_at = datetime.now(timezone.utc).isoformat()
    run_id = f"quality-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')}"

    console.print(f"[bold]SemBlend Quality Benchmark[/bold] -- {run_id}")
    console.print(f"  Proxy: {cfg.proxy_endpoint}  vLLM: {cfg.vllm_endpoint}")
    console.print(f"  Model: {cfg.model}  Judge: {cfg.judge_model}  Skip: {skip_judge}")

    judge_report: JudgeReport | None = None
    if not skip_judge:
        if not os.environ.get("ANTHROPIC_API_KEY"):
            console.print("[red]ANTHROPIC_API_KEY not set, skipping judge[/red]")
        else:
            judge_report = await run_judge_evaluation(cfg)

    fpr_report = await run_false_positive_analysis(cfg)
    ablation = await run_embedding_ablation(cfg)

    return QualityResults(
        run_id=run_id, started_at=started_at,
        completed_at=datetime.now(timezone.utc).isoformat(),
        config={
            "proxy_endpoint": cfg.proxy_endpoint, "vllm_endpoint": cfg.vllm_endpoint,
            "judge_model": cfg.judge_model, "model": cfg.model,
            "max_tokens": cfg.max_tokens, "n_judge_samples": cfg.n_judge_samples,
            "n_false_positive_samples": cfg.n_false_positive_samples,
            "embedding_dimensions": list(cfg.embedding_dimensions),
            "timeout_secs": cfg.timeout_secs,
        },
        judge=judge_report, false_positive=fpr_report, ablation=ablation,
    )


def main() -> None:
    """Entry point for the quality benchmark."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    args = parse_args()
    cfg = QualityConfig(
        proxy_endpoint=args.proxy_endpoint, vllm_endpoint=args.vllm_endpoint,
        judge_model=args.judge_model, model=args.model,
        n_judge_samples=args.n_judge_samples,
        n_false_positive_samples=args.n_fp_samples, output_path=args.output,
    )
    results = asyncio.run(run_benchmark(cfg, skip_judge=args.skip_judge))
    print_report(results)
    console.print(f"\nResults saved to: {save_results(results, cfg.output_path)}")


if __name__ == "__main__":
    main()
