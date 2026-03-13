"""Unified dataset loader returning seed and test queries."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from benchmarks.e2e.config import BenchmarkConfig

logger = logging.getLogger(__name__)


@dataclass
class Query:
    """A benchmark query with metadata."""

    prompt: str
    query_type: str  # "exact_repeat", "semantic_variant", "novel"
    source_dataset: str
    seed_index: int | None = None  # index of seed this derives from
    reference_response: str | None = None  # gold answer if available


@dataclass
class DatasetQueries:
    """Seeds (for warmup) and test queries (for measurement)."""

    seeds: list[Query]
    test_queries: list[Query]
    dataset_name: str


def load_dataset_queries(
    name: str,
    config: BenchmarkConfig,
) -> DatasetQueries:
    """Load a named dataset and split into seeds + test queries.

    Args:
        name: One of "sharegpt", "multinews", "bitext".
        config: Benchmark configuration.

    Returns:
        DatasetQueries with seeds and test queries.
    """
    size = config.dataset_size(name)
    logger.info("Loading dataset %s (size=%d)", name, size)

    if name == "sharegpt":
        from benchmarks.e2e.bench_datasets.sharegpt import load_sharegpt
        return load_sharegpt(size, config)
    if name == "multinews":
        from benchmarks.e2e.bench_datasets.multinews import load_multinews
        return load_multinews(size, config)
    if name == "bitext":
        from benchmarks.e2e.bench_datasets.bitext import load_bitext
        return load_bitext(size, config)

    raise ValueError(f"Unknown dataset: {name}")
