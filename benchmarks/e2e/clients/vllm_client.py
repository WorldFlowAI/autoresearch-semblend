"""Direct vLLM client for baseline (no-cache) measurements."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import aiohttp

logger = logging.getLogger(__name__)


@dataclass
class VllmResponse:
    """Response from a direct vLLM query (no cache)."""

    text: str
    ttft_ms: float
    total_ms: float
    model: str


class VllmClient:
    """Async client for direct vLLM OpenAI-compatible API.

    Used as the baseline: queries bypass Synapse entirely so we can
    measure the TTFT improvement from caching.
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

    async def __aenter__(self) -> VllmClient:
        connector = aiohttp.TCPConnector(limit=32)
        self._session = aiohttp.ClientSession(
            timeout=self._timeout,
            connector=connector,
        )
        return self

    async def __aexit__(self, *args: object) -> None:
        if self._session:
            await self._session.close()

    async def query(self, prompt: str, retries: int = 2) -> VllmResponse:
        """Send a chat completion to vLLM and measure TTFT.

        Args:
            prompt: The user prompt.
            retries: Number of retries on transient connection errors.

        Returns:
            VllmResponse with timing and text data.
        """
        if not self._session:
            raise RuntimeError("Client not initialized (use async with)")

        url = f"{self._endpoint}/v1/chat/completions"
        body = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._max_tokens,
            "stream": True,
        }

        last_err: Exception | None = None
        for attempt in range(1 + retries):
            try:
                t_start = time.monotonic()
                ttft_ms = 0.0
                chunks: list[str] = []

                async with self._session.post(url, json=body) as resp:
                    first_token = True
                    async for line in resp.content:
                        decoded = line.decode("utf-8").strip()
                        if not decoded.startswith("data: "):
                            continue
                        data = decoded[6:]
                        if data == "[DONE]":
                            break

                        if first_token:
                            ttft_ms = (time.monotonic() - t_start) * 1000
                            first_token = False

                        try:
                            import json
                            chunk = json.loads(data)
                            delta = chunk.get("choices", [{}])[0].get(
                                "delta", {}
                            )
                            content = delta.get("content", "")
                            if content:
                                chunks.append(content)
                        except (ValueError, KeyError, IndexError):
                            continue

                total_ms = (time.monotonic() - t_start) * 1000

                return VllmResponse(
                    text="".join(chunks),
                    ttft_ms=ttft_ms,
                    total_ms=total_ms,
                    model=self._model,
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
