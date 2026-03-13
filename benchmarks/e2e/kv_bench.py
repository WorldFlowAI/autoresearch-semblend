"""KV-cache benchmark mode — measures KV-tensor injection via LMCache.

Sends queries directly to vLLM (which has LMCache + SynapseLMCacheConnector
wired in). KV caching happens transparently — the benchmark observes the
effect via TTFT reduction on repeated/similar prompts.

Three phases:
  1. Warmup: populate the KV cache with seed prompts
  2. Test: measure KV reuse on semantic variants
  3. Baseline: cold vLLM TTFT without any cached KV state

Distinguishes:
  - Exact KV hit: identical prefix → full prefill skip
  - Semantic KV hit: similar prefix → partial prefill (PartialAttention)
  - Miss: no cached KV → full prefill
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import struct
import time
from dataclasses import dataclass, field

import aiohttp
from rich.console import Console
from tqdm import tqdm

from benchmarks.e2e.clients.vllm_client import VllmClient
from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.bench_datasets.loader import DatasetQueries, Query, load_dataset_queries

logger = logging.getLogger(__name__)
console = Console()


@dataclass(frozen=True)
class KvBenchResult:
    """Result of a single KV-cache benchmark query."""

    prompt: str
    query_type: str
    ttft_ms: float
    total_ms: float
    kv_match_type: str  # exact, semantic, miss
    computation_ratio: float  # 0.0 = full reuse, 1.0 = full compute


@dataclass
class KvBenchResults:
    """Aggregate KV-cache benchmark results."""

    run_id: str
    model: str
    dataset: str
    warmup_queries: int = 0
    test_results: list[KvBenchResult] = field(default_factory=list)
    baseline_ttft: list[float] = field(default_factory=list)

    @property
    def exact_hits(self) -> list[KvBenchResult]:
        return [r for r in self.test_results if r.kv_match_type == "exact"]

    @property
    def semantic_hits(self) -> list[KvBenchResult]:
        return [
            r for r in self.test_results if r.kv_match_type == "semantic"
        ]

    @property
    def misses(self) -> list[KvBenchResult]:
        return [r for r in self.test_results if r.kv_match_type == "miss"]

    def summary(self) -> dict:
        def _stats(values: list[float]) -> dict:
            if not values:
                return {"n": 0, "p50": 0, "p95": 0, "p99": 0, "mean": 0}
            s = sorted(values)
            n = len(s)
            return {
                "n": n,
                "p50": s[int(n * 0.50)],
                "p95": s[int(n * 0.95)],
                "p99": s[int(n * 0.99)] if n > 1 else s[-1],
                "mean": sum(s) / n,
            }

        baseline_stats = _stats(self.baseline_ttft)
        exact_ttft = [r.ttft_ms for r in self.exact_hits]
        semantic_ttft = [r.ttft_ms for r in self.semantic_hits]
        miss_ttft = [r.ttft_ms for r in self.misses]

        baseline_p50 = baseline_stats["p50"] if baseline_stats["n"] > 0 else 1

        exact_stats = _stats(exact_ttft)
        semantic_stats = _stats(semantic_ttft)

        return {
            "dataset": self.dataset,
            "warmup_queries": self.warmup_queries,
            "total_test_queries": len(self.test_results),
            "exact_hits": len(self.exact_hits),
            "semantic_hits": len(self.semantic_hits),
            "misses": len(self.misses),
            "baseline": baseline_stats,
            "exact": exact_stats,
            "semantic": semantic_stats,
            "miss": _stats(miss_ttft),
            "speedup_exact": (
                baseline_p50 / exact_stats["p50"]
                if exact_stats["p50"] > 0
                else 0
            ),
            "speedup_semantic": (
                baseline_p50 / semantic_stats["p50"]
                if semantic_stats["p50"] > 0
                else 0
            ),
            "avg_computation_ratio": (
                sum(r.computation_ratio for r in self.semantic_hits)
                / len(self.semantic_hits)
                if self.semantic_hits
                else 1.0
            ),
        }


class KvCacheBenchmark:
    """Benchmarks KV-tensor injection via LMCache + vLLM."""

    def __init__(
        self,
        config: BenchmarkConfig,
        run_id: str,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._gateway_url = config.synapse_endpoint

    async def run(self, dataset_name: str) -> KvBenchResults:
        """Run the full KV-cache benchmark for a dataset."""
        console.print(f"\n[bold]KV-Cache Benchmark[/bold] -- {self._run_id}")
        console.print(f"  vLLM: {self._config.vllm_endpoint}")
        console.print(f"  Gateway: {self._gateway_url}")
        console.print(f"  Dataset: {dataset_name}")

        dataset = load_dataset_queries(dataset_name, self._config)
        results = KvBenchResults(
            run_id=self._run_id,
            model=self._config.model,
            dataset=dataset_name,
        )

        async with VllmClient(
            endpoint=self._config.vllm_endpoint,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout_secs,
        ) as client:
            # Phase 1: Warmup — populate KV cache via normal inference
            await self._warmup(client, dataset, results)

            # Phase 2: Test — measure KV reuse
            await self._test(client, dataset, results)

        # Phase 3: Baseline — cold vLLM with cleared KV state
        await self._clear_kv_cache()
        async with VllmClient(
            endpoint=self._config.vllm_endpoint,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout_secs,
        ) as client:
            await self._baseline(client, dataset, results)

        return results

    async def _warmup(
        self,
        client: VllmClient,
        dataset: DatasetQueries,
        results: KvBenchResults,
    ) -> None:
        """Populate KV cache by running seed prompts through vLLM."""
        seeds = dataset.seeds
        if not seeds:
            return

        console.print(f"  Phase 1: Warming KV cache with {len(seeds)} seeds...")

        sem = asyncio.Semaphore(self._config.concurrency)
        completed = 0

        async def _send(query: Query) -> bool:
            async with sem:
                try:
                    await client.query(query.prompt)
                    return True
                except Exception:
                    return False

        tasks = [_send(q) for q in seeds]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="  KV Warmup",
            leave=False,
        ):
            ok = await coro
            if ok:
                completed += 1

        results.warmup_queries = completed
        console.print(f"  Warmed up {completed}/{len(seeds)} queries")

        # Allow KV state to propagate to gateway
        await asyncio.sleep(self._config.warmup_delay_secs)

    async def _test(
        self,
        client: VllmClient,
        dataset: DatasetQueries,
        results: KvBenchResults,
    ) -> None:
        """Send test queries and classify KV match type."""
        queries = dataset.test_queries
        if not queries:
            return

        console.print(f"  Phase 2: Testing {len(queries)} queries...")

        sem = asyncio.Semaphore(self._config.concurrency)

        async def _send(query: Query) -> KvBenchResult | None:
            async with sem:
                try:
                    t0 = time.monotonic()
                    resp = await client.query(query.prompt)
                    total_ms = (time.monotonic() - t0) * 1000

                    # Classify match type by checking gateway
                    match_type, comp_ratio = await self._classify_kv_match(
                        query.prompt
                    )

                    return KvBenchResult(
                        prompt=query.prompt[:200],
                        query_type=query.query_type,
                        ttft_ms=resp.ttft_ms,
                        total_ms=total_ms,
                        kv_match_type=match_type,
                        computation_ratio=comp_ratio,
                    )
                except Exception:
                    logger.warning("Test query failed", exc_info=True)
                    return None

        tasks = [_send(q) for q in queries]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="  KV Test",
            leave=False,
        ):
            result = await coro
            if result is not None:
                results.test_results.append(result)

        n_exact = len(results.exact_hits)
        n_semantic = len(results.semantic_hits)
        n_miss = len(results.misses)
        console.print(
            f"  Results: {n_exact} exact, {n_semantic} semantic, {n_miss} miss"
        )

    async def _baseline(
        self,
        client: VllmClient,
        dataset: DatasetQueries,
        results: KvBenchResults,
    ) -> None:
        """Measure cold TTFT with no cached KV state."""
        queries = dataset.test_queries
        if not queries:
            return

        console.print(
            f"  Phase 3: Baseline {len(queries)} queries (cold vLLM)..."
        )

        sem = asyncio.Semaphore(self._config.concurrency)

        async def _send(query: Query) -> float:
            async with sem:
                try:
                    resp = await client.query(query.prompt)
                    return resp.ttft_ms
                except Exception:
                    return 0.0

        tasks = [_send(q) for q in queries]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="  Baseline",
            leave=False,
        ):
            ttft = await coro
            if ttft > 0:
                results.baseline_ttft.append(ttft)

    async def _classify_kv_match(
        self, prompt: str
    ) -> tuple[str, float]:
        """Check gateway to classify KV match type for a prompt.

        Returns (match_type, computation_ratio).
        """
        # Compute token hash from prompt (simplified: hash the text)
        text_hash = hashlib.sha256(prompt.encode()).hexdigest()

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=5.0)
            ) as session:
                async with session.get(
                    f"{self._gateway_url}/api/v1/kv-cache/lmcache/"
                    f"{text_hash}/exists",
                ) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        if data.get("exact", False):
                            return ("exact", 0.0)
                        if data.get("semantic", False):
                            comp = data.get("computationRatio", 0.5)
                            return ("semantic", comp)
        except aiohttp.ClientError:
            pass

        return ("miss", 1.0)

    async def _clear_kv_cache(self) -> None:
        """Clear the gateway KV cache for clean baseline measurement."""
        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10.0)
            ) as session:
                async with session.delete(
                    f"{self._gateway_url}/api/v1/kv-cache/clear",
                ) as resp:
                    if resp.status in (200, 204):
                        logger.info("KV cache cleared for baseline")
                    else:
                        logger.warning(
                            "KV cache clear returned %d", resp.status
                        )
        except aiohttp.ClientError:
            logger.warning("Failed to clear KV cache", exc_info=True)
