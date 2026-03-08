"""Component-level direct measurement for paper decomposition table.

Measures each Synapse component in isolation so the paper can report
empirical numbers instead of differential analysis:

- ONNX embedding latency (proxy L0 GPU embedder)
- CAGRA search latency (gateway segment search)
- KV load/save latency (gateway KV cache endpoints)
- Cold TTFT by prompt length (direct vLLM)

All measurements are direct, not derived by subtraction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from typing import Any

import aiohttp

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyStats:
    """Latency percentile statistics."""

    n: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    min_ms: float
    max_ms: float

    @staticmethod
    def from_samples(samples: list[float]) -> LatencyStats:
        if not samples:
            return LatencyStats(
                n=0, p50_ms=0, p95_ms=0, p99_ms=0,
                mean_ms=0, min_ms=0, max_ms=0,
            )
        s = sorted(samples)
        n = len(s)
        return LatencyStats(
            n=n,
            p50_ms=s[int(n * 0.50)],
            p95_ms=s[int(n * 0.95)],
            p99_ms=s[int(n * 0.99)] if n > 1 else s[-1],
            mean_ms=sum(s) / n,
            min_ms=s[0],
            max_ms=s[-1],
        )


@dataclass
class ComponentResults:
    """Results from component-level benchmarking."""

    onnx_embedding: LatencyStats | None = None
    cagra_search: LatencyStats | None = None
    kv_save: LatencyStats | None = None
    kv_load: LatencyStats | None = None
    cold_ttft: dict[int, LatencyStats] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if self.onnx_embedding:
            result["onnx_embedding"] = asdict(self.onnx_embedding)
        if self.cagra_search:
            result["cagra_search"] = asdict(self.cagra_search)
        if self.kv_save:
            result["kv_save"] = asdict(self.kv_save)
        if self.kv_load:
            result["kv_load"] = asdict(self.kv_load)
        if self.cold_ttft:
            result["cold_ttft"] = {
                str(k): asdict(v) for k, v in self.cold_ttft.items()
            }
        return result


class ComponentBenchmark:
    """Measures individual Synapse components in isolation."""

    def __init__(
        self,
        proxy_endpoint: str,
        gateway_endpoint: str,
        vllm_endpoint: str,
        model: str,
        n_samples: int = 200,
        timeout: float = 30.0,
    ) -> None:
        self._proxy = proxy_endpoint.rstrip("/")
        self._gateway = gateway_endpoint.rstrip("/")
        self._vllm = vllm_endpoint.rstrip("/")
        self._model = model
        self._n = n_samples
        self._timeout = aiohttp.ClientTimeout(total=timeout)

    async def run_all(self) -> ComponentResults:
        """Run all component benchmarks and return results."""
        results = ComponentResults()

        results.onnx_embedding = await self._bench_onnx_embedding()
        results.cagra_search = await self._bench_cagra_search()
        results.kv_save = await self._bench_kv_save()
        results.kv_load = await self._bench_kv_load()
        results.cold_ttft = await self._bench_cold_ttft()

        return results

    async def _bench_onnx_embedding(self) -> LatencyStats:
        """Measure ONNX embedding latency via proxy embed endpoint."""
        samples: list[float] = []
        test_texts = [
            f"This is test query number {i} for embedding latency measurement"
            for i in range(self._n)
        ]

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for text in test_texts:
                t0 = time.monotonic()
                try:
                    async with session.post(
                        f"{self._proxy}/api/v1/embed",
                        json={"text": text},
                    ) as resp:
                        if resp.status == 200:
                            elapsed = (time.monotonic() - t0) * 1000
                            samples.append(elapsed)
                        else:
                            logger.debug(
                                "Embed endpoint returned %d", resp.status
                            )
                except aiohttp.ClientError:
                    logger.debug("Embed request failed", exc_info=True)

        logger.info(
            "ONNX embedding: %d/%d samples collected", len(samples), self._n
        )
        return LatencyStats.from_samples(samples)

    async def _bench_cagra_search(self) -> LatencyStats:
        """Measure CAGRA vector search latency via gateway."""
        samples: list[float] = []
        # Use a dummy 1024-dim embedding for search
        dummy_embedding = [0.01] * 1024

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for _ in range(self._n):
                t0 = time.monotonic()
                try:
                    async with session.post(
                        f"{self._gateway}/api/v1/kv-cache/segments/search",
                        json={
                            "embedding": dummy_embedding,
                            "topK": 5,
                        },
                    ) as resp:
                        if resp.status in (200, 404):
                            elapsed = (time.monotonic() - t0) * 1000
                            samples.append(elapsed)
                except aiohttp.ClientError:
                    logger.debug("CAGRA search failed", exc_info=True)

        logger.info(
            "CAGRA search: %d/%d samples collected", len(samples), self._n
        )
        return LatencyStats.from_samples(samples)

    async def _bench_kv_save(self) -> LatencyStats:
        """Measure KV cache save latency via gateway."""
        import hashlib
        import struct

        samples: list[float] = []

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for i in range(self._n):
                # Generate unique token hash
                token_ids = list(range(i, i + 64))
                packed = struct.pack(f"<{len(token_ids)}I", *token_ids)
                token_hash = hashlib.sha256(packed).hexdigest()

                # Small dummy KV data (64 tokens, 1 layer)
                import base64

                dummy_kv = b"\x00" * (64 * 128 * 2)  # minimal KV blob
                kv_b64 = base64.b64encode(dummy_kv).decode("ascii")

                t0 = time.monotonic()
                try:
                    async with session.put(
                        f"{self._gateway}/api/v1/kv-cache/{token_hash}",
                        json={
                            "kvData": kv_b64,
                            "numTokens": 64,
                            "numLayers": 1,
                            "numHeads": 8,
                            "headDim": 64,
                        },
                    ) as resp:
                        elapsed = (time.monotonic() - t0) * 1000
                        if resp.status in (200, 201):
                            samples.append(elapsed)
                except aiohttp.ClientError:
                    logger.debug("KV save failed", exc_info=True)

        logger.info(
            "KV save: %d/%d samples collected", len(samples), self._n
        )
        return LatencyStats.from_samples(samples)

    async def _bench_kv_load(self) -> LatencyStats:
        """Measure KV cache load latency via gateway."""
        import hashlib
        import struct

        samples: list[float] = []

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for i in range(self._n):
                token_ids = list(range(i, i + 64))
                packed = struct.pack(f"<{len(token_ids)}I", *token_ids)
                token_hash = hashlib.sha256(packed).hexdigest()

                t0 = time.monotonic()
                try:
                    async with session.get(
                        f"{self._gateway}/api/v1/kv-cache/{token_hash}",
                    ) as resp:
                        elapsed = (time.monotonic() - t0) * 1000
                        # Both hit (200) and miss (404) are valid latency
                        if resp.status in (200, 404):
                            samples.append(elapsed)
                except aiohttp.ClientError:
                    logger.debug("KV load failed", exc_info=True)

        logger.info(
            "KV load: %d/%d samples collected", len(samples), self._n
        )
        return LatencyStats.from_samples(samples)

    async def _bench_cold_ttft(self) -> dict[int, LatencyStats]:
        """Measure cold TTFT at various prompt lengths via direct vLLM."""
        prompt_lengths = [128, 256, 512, 1024, 2048, 3500]
        results: dict[int, LatencyStats] = {}

        # Generate prompts of specific token lengths (approximate: 1 word ~ 1.3 tokens)
        base_text = (
            "Explain the following concept in detail with examples and "
            "step-by-step reasoning. "
        )

        async with aiohttp.ClientSession(timeout=self._timeout) as session:
            for target_len in prompt_lengths:
                # Approximate word count for target token length
                word_count = int(target_len / 1.3)
                padding = " ".join(
                    f"word{i}" for i in range(word_count)
                )
                prompt = base_text + padding

                n_per_length = min(self._n, 50)
                samples: list[float] = []

                for _ in range(n_per_length):
                    t0 = time.monotonic()
                    try:
                        async with session.post(
                            f"{self._vllm}/v1/chat/completions",
                            json={
                                "model": self._model,
                                "messages": [
                                    {"role": "user", "content": prompt}
                                ],
                                "max_tokens": 1,
                                "stream": False,
                            },
                        ) as resp:
                            if resp.status == 200:
                                elapsed = (time.monotonic() - t0) * 1000
                                samples.append(elapsed)
                    except aiohttp.ClientError:
                        logger.debug(
                            "Cold TTFT failed for %d tokens",
                            target_len,
                            exc_info=True,
                        )

                results[target_len] = LatencyStats.from_samples(samples)
                logger.info(
                    "Cold TTFT @%d tokens: %d samples, P50=%.1fms",
                    target_len,
                    len(samples),
                    results[target_len].p50_ms,
                )

        return results


async def run_component_benchmark(
    proxy_endpoint: str,
    gateway_endpoint: str,
    vllm_endpoint: str,
    model: str,
    n_samples: int = 200,
    output_path: str | None = None,
) -> ComponentResults:
    """Run component benchmark and optionally save results."""
    bench = ComponentBenchmark(
        proxy_endpoint=proxy_endpoint,
        gateway_endpoint=gateway_endpoint,
        vllm_endpoint=vllm_endpoint,
        model=model,
        n_samples=n_samples,
    )

    results = await bench.run_all()

    if output_path:
        import os

        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results.to_dict(), f, indent=2)
        logger.info("Component results saved to %s", output_path)

    return results
