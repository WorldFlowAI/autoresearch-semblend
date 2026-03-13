"""News summarization dataset loader (CNN/DailyMail)."""

from __future__ import annotations

import logging
import random

from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.bench_datasets.loader import DatasetQueries, Query

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a summarization assistant. Provide a concise summary "
    "of the following news article."
)


def load_multinews(size: int, config: BenchmarkConfig) -> DatasetQueries:
    """Load news articles and split into seeds/tests.

    Uses CNN/DailyMail for long-context summarization. Tests KV cache
    reuse for shared system-prompt + similar article patterns (RAG use case).
    """
    from datasets import load_dataset

    ds = load_dataset("cnn_dailymail", "3.0.0", split="test")
    articles = []
    for row in ds:
        doc = row.get("article", "")
        summary = row.get("highlights", "")
        if doc and len(doc) > 200:
            articles.append((doc[:1800], summary))  # within proxy 2000-char embedding limit
        if len(articles) >= size:
            break

    if not articles:
        logger.warning("No news articles loaded")
        return DatasetQueries(
            seeds=[], test_queries=[], dataset_name="multinews"
        )

    random.seed(42)
    random.shuffle(articles)
    split_idx = int(len(articles) * 0.4)
    seed_articles = articles[:split_idx]
    novel_pool = articles[split_idx:]

    seeds = [
        Query(
            prompt=f"{SYSTEM_PROMPT}\n\n{doc}",
            query_type="seed",
            source_dataset="multinews",
            reference_response=summary,
        )
        for doc, summary in seed_articles
    ]

    test_queries: list[Query] = []
    n_test = len(articles) - split_idx
    n_exact = int(n_test * config.exact_repeat_ratio)
    n_semantic = int(n_test * config.semantic_variant_ratio)
    n_novel = n_test - n_exact - n_semantic

    # Exact repeats
    for i in range(min(n_exact, len(seed_articles))):
        idx = i % len(seed_articles)
        doc, summary = seed_articles[idx]
        test_queries.append(Query(
            prompt=f"{SYSTEM_PROMPT}\n\n{doc}",
            query_type="exact_repeat",
            source_dataset="multinews",
            seed_index=idx,
            reference_response=summary,
        ))

    # Semantic variants: same articles, different instruction
    variant_instructions = [
        "Summarize the key points from this article:",
        "Provide a brief overview of the following news:",
        "What are the main takeaways from this article?",
    ]
    for i in range(min(n_semantic, len(seed_articles))):
        idx = i % len(seed_articles)
        doc, summary = seed_articles[idx]
        instruction = variant_instructions[i % len(variant_instructions)]
        test_queries.append(Query(
            prompt=f"{instruction}\n\n{doc}",
            query_type="semantic_variant",
            source_dataset="multinews",
            seed_index=idx,
            reference_response=summary,
        ))

    # Novel queries
    for i in range(min(n_novel, len(novel_pool))):
        doc, summary = novel_pool[i]
        test_queries.append(Query(
            prompt=f"{SYSTEM_PROMPT}\n\n{doc}",
            query_type="novel",
            source_dataset="multinews",
            reference_response=summary,
        ))

    random.shuffle(test_queries)
    logger.info(
        "MultiNews: %d seeds, %d test queries",
        len(seeds), len(test_queries),
    )

    return DatasetQueries(
        seeds=seeds,
        test_queries=test_queries,
        dataset_name="multinews",
    )
