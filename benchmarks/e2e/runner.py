"""Three-phase benchmark runner: warmup -> test -> baseline."""

from __future__ import annotations

import asyncio
import logging

from rich.console import Console
from tqdm import tqdm

from benchmarks.e2e.clients.synapse_client import SynapseClient
from benchmarks.e2e.clients.vllm_client import VllmClient
from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.bench_datasets.loader import DatasetQueries, Query, load_dataset_queries
from benchmarks.e2e.results import BenchmarkResults, QueryResult

logger = logging.getLogger(__name__)
console = Console()


class BenchmarkRunner:
    """Orchestrates the three-phase benchmark execution.

    Phase 1 (Warmup): Send seed prompts to populate the KV cache.
    Phase 2 (Test): Send test queries and measure TTFT/cache behavior.
    Phase 3 (Baseline): Send test queries directly to vLLM (bypass cache).
    """

    def __init__(
        self,
        config: BenchmarkConfig,
        run_id: str,
    ) -> None:
        self._config = config
        self._run_id = run_id
        self._results = BenchmarkResults(
            run_id=run_id,
            model=config.model,
            synapse_endpoint=config.synapse_endpoint,
        )

    async def run(self) -> BenchmarkResults:
        """Execute the full benchmark pipeline."""
        console.print(f"\n[bold]SemBlend E2E Benchmark[/bold] -- {self._run_id}")
        console.print(f"  Model: {self._config.model}")
        console.print(f"  Synapse: {self._config.synapse_endpoint}")
        console.print(f"  vLLM: {self._config.vllm_endpoint}")
        console.print(f"  Datasets: {', '.join(self._config.datasets)}")
        console.print()

        for ds_name in self._config.datasets:
            dataset = load_dataset_queries(ds_name, self._config)
            await self._run_dataset(dataset)

        self._results.finalize()
        return self._results

    async def _run_dataset(self, dataset: DatasetQueries) -> None:
        """Run all three phases for a single dataset."""
        console.rule(f"[bold]{dataset.dataset_name}[/bold]")

        if not dataset.seeds and not dataset.test_queries:
            console.print("[yellow]  No queries loaded, skipping[/yellow]")
            return

        async with SynapseClient(
            endpoint=self._config.synapse_endpoint,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout_secs,
        ) as synapse:
            # Phase 1: Warmup
            await self._warmup(synapse, dataset)

            # Phase 2: Test
            await self._test(synapse, dataset)

        # Phase 3: Baseline (direct vLLM, no cache)
        await self._baseline(dataset)

    async def _warmup(
        self,
        client: SynapseClient,
        dataset: DatasetQueries,
    ) -> None:
        """Phase 1: Send seed prompts to populate the cache.

        Sends seeds in small batches to avoid overwhelming the proxy,
        which can trigger liveness probe failures and pod restarts.
        """
        seeds = dataset.seeds
        if not seeds:
            return

        batch_size = self._config.warmup_batch_size
        batch_delay = self._config.warmup_batch_delay_secs
        n_batches = (len(seeds) + batch_size - 1) // batch_size

        console.print(
            f"  Phase 1: Warming up with {len(seeds)} seeds "
            f"({n_batches} batches of {batch_size})..."
        )

        sem = asyncio.Semaphore(self._config.concurrency)
        completed = 0
        failed = 0

        async def _send(query: Query) -> bool:
            async with sem:
                try:
                    await client.query(query.prompt)
                    return True
                except Exception:
                    logger.warning(
                        "Warmup query failed", exc_info=True
                    )
                    return False

        progress = tqdm(total=len(seeds), desc="  Warmup", leave=False)

        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, len(seeds))
            batch = seeds[start:end]

            tasks = [_send(q) for q in batch]
            results = await asyncio.gather(*tasks)

            batch_ok = sum(1 for r in results if r)
            batch_fail = len(results) - batch_ok
            completed += batch_ok
            failed += batch_fail
            progress.update(len(batch))

            # Pause between batches to let the proxy recover
            if batch_idx < n_batches - 1:
                await asyncio.sleep(batch_delay)

        progress.close()

        if failed > 0:
            console.print(
                f"  [yellow]Warmup: {completed} ok, {failed} failed[/yellow]"
            )

        # Wait for cache propagation
        delay = self._config.warmup_delay_secs
        console.print(f"  Waiting {delay}s for cache propagation...")
        await asyncio.sleep(delay)

    async def _test(
        self,
        client: SynapseClient,
        dataset: DatasetQueries,
    ) -> None:
        """Phase 2: Send test queries and measure TTFT/cache."""
        queries = dataset.test_queries
        if not queries:
            return

        console.print(
            f"  Phase 2: Testing with {len(queries)} queries..."
        )

        sem = asyncio.Semaphore(self._config.concurrency)
        results: list[tuple[int, QueryResult | None]] = []

        async def _send(idx: int, query: Query) -> tuple[int, QueryResult | None]:
            async with sem:
                try:
                    resp = await client.query(query.prompt)
                    qr = QueryResult(
                        prompt=query.prompt,
                        query_type=query.query_type,
                        source_dataset=dataset.dataset_name,
                        response_text=resp.text,
                        ttft_ms=resp.ttft_ms,
                        total_ms=resp.total_ms,
                        cache_hit=resp.cache_hit,
                        cache_tier=resp.cache_tier,
                    )
                    return (idx, qr)
                except Exception:
                    logger.warning(
                        "Test query %d failed", idx, exc_info=True
                    )
                    return (idx, None)

        tasks = [_send(i, q) for i, q in enumerate(queries)]
        for coro in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="  Test",
            leave=False,
        ):
            idx, qr = await coro
            if qr is not None:
                self._results.add_query_result(qr)
                results.append((idx, qr))

        hits = sum(1 for _, qr in results if qr and qr.cache_hit)
        console.print(
            f"  Cache hit rate: {hits}/{len(results)} "
            f"({100*hits/max(len(results),1):.1f}%)"
        )

    async def _baseline(self, dataset: DatasetQueries) -> None:
        """Phase 3: Send test queries directly to vLLM (bypass cache)."""
        queries = dataset.test_queries
        if not queries:
            return

        console.print(
            f"  Phase 3: Baseline with {len(queries)} queries "
            f"(direct vLLM at {self._config.vllm_endpoint})..."
        )

        async with VllmClient(
            endpoint=self._config.vllm_endpoint,
            model=self._config.model,
            max_tokens=self._config.max_tokens,
            timeout=self._config.timeout_secs,
        ) as client:
            sem = asyncio.Semaphore(self._config.concurrency)

            async def _send(idx: int, query: Query) -> tuple[int, str, float]:
                async with sem:
                    try:
                        resp = await client.query(query.prompt)
                        return (idx, resp.text, resp.ttft_ms)
                    except Exception:
                        logger.warning(
                            "Baseline query %d failed",
                            idx,
                            exc_info=True,
                        )
                        return (idx, "", 0.0)

            tasks = [_send(i, q) for i, q in enumerate(queries)]
            for coro in tqdm(
                asyncio.as_completed(tasks),
                total=len(tasks),
                desc="  Baseline",
                leave=False,
            ):
                idx, text, ttft = await coro
                if text:
                    self._results.add_baseline_result(
                        idx, dataset.dataset_name, text, ttft
                    )

        # Print TTFT comparison
        latency = self._results.latency
        if latency.baseline_ttft:
            console.print(
                f"  Baseline TTFT P50: {latency.baseline_p50:.1f}ms"
            )
            for tier, stats in latency.tiers.items():
                speedup = latency.speedup(tier)
                console.print(
                    f"  {tier} TTFT P50: {stats.p50:.1f}ms "
                    f"(speedup: {speedup:.1f}x)"
                )
