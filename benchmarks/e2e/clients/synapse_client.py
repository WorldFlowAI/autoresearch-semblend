"""Async Synapse proxy client with TTFT measurement."""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class SynapseResponse:
    """Response from a Synapse proxy query."""

    text: str
    ttft_ms: float
    total_ms: float
    cache_hit: bool
    cache_tier: str  # "l0_gpu", "l1", "l2", "miss"
    similarity: float
    model: str
    latency_ms: float


class SynapseClient:
    """Async client for Synapse proxy.

    Uses the proxy's /api/v1/query endpoint and extracts cache
    metadata from the JSON response body.
    """

    def __init__(
        self,
        endpoint: str,
        model: str,
        max_tokens: int = 256,
        timeout: float = 60.0,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._model = model
        self._max_tokens = max_tokens
        self._timeout = aiohttp.ClientTimeout(total=timeout)
        self._session: aiohttp.ClientSession | None = None

    async def __aenter__(self) -> SynapseClient:
        connector = aiohttp.TCPConnector(limit=32)
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            connector=connector,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._session:
            await self._session.close()

    async def query(
        self,
        prompt: str,
        skip_cache: bool = False,
        retries: int = 2,
    ) -> SynapseResponse:
        """Send a query to the Synapse proxy and measure response time.

        Args:
            prompt: The user prompt.
            skip_cache: If True, sends X-Synapse-Skip-Cache header.
            retries: Number of retries on transient connection errors.

        Returns:
            SynapseResponse with timing, cache, and quality data.
        """
        if not self._session:
            raise RuntimeError("Client not initialized (use async with)")

        url = f"{self._endpoint}/api/v1/query"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if skip_cache:
            headers["X-Synapse-Skip-Cache"] = "true"

        body = {
            "query": prompt,
            "model": self._model,
            "max_tokens": self._max_tokens,
        }

        last_err: Exception | None = None
        for attempt in range(1 + retries):
            try:
                t_start = time.monotonic()

                async with self._session.post(
                    url, json=body, headers=headers
                ) as resp:
                    resp.raise_for_status()
                    data = await resp.json()

                total_ms = (time.monotonic() - t_start) * 1000

                response_text = data.get("response", "")
                cache_hit = data.get("cache_hit", False)
                cache_tier = data.get("cache_tier", "miss")
                similarity = data.get("similarity", 0.0)
                latency_ms = data.get("latency_ms", total_ms)

                ttft_ms = latency_ms if latency_ms > 0 else total_ms

                return SynapseResponse(
                    text=response_text,
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    cache_hit=cache_hit,
                    cache_tier=cache_tier if cache_hit else "miss",
                    similarity=similarity,
                    model=self._model,
                    latency_ms=latency_ms,
                )
            except (
                aiohttp.ClientConnectorError,
                aiohttp.ServerDisconnectedError,
            ) as exc:
                last_err = exc
                if attempt < retries:
                    wait = 3.0 * (attempt + 1)
                    logger.warning(
                        "Connection error (attempt %d/%d), retrying in %.0fs: %s",
                        attempt + 1, 1 + retries, wait, exc,
                    )
                    await asyncio.sleep(wait)

        raise last_err  # type: ignore[misc]
