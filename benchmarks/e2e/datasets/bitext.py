"""Bitext customer support dataset loader with held-out intent split."""

from __future__ import annotations

import logging
import random
from collections import defaultdict

from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.datasets.loader import DatasetQueries, Query

logger = logging.getLogger(__name__)

DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"

SYSTEM_PROMPT = (
    "You are a customer support agent. Respond helpfully and "
    "professionally to the customer's question."
)

# Fraction of intent categories held out entirely from seeding.
# Queries from held-out intents become "novel" test queries that the
# cache has never seen, producing a realistic miss rate.
HOLDOUT_INTENT_FRACTION = 0.25

# Per-intent sample size limit.  We load ALL intents first, then
# sample down so the total stays near `size`.
MAX_PER_INTENT = 60


def load_bitext(size: int, config: BenchmarkConfig) -> DatasetQueries:
    """Load Bitext customer support queries with held-out intent split.

    To avoid artificially inflating hit rates, 25% of intent categories
    are held out entirely from the seed set. Test queries from held-out
    intents are labeled 'novel' and should produce cache misses, while
    queries from seeded intents are labeled 'semantic_variant'.

    The loader reads ALL intents from the dataset (not just the first N
    rows) and samples proportionally to reach the target ``size``.
    """
    from datasets import load_dataset

    ds = load_dataset(DATASET_ID, split="train")

    # Collect ALL intents first — do NOT early-terminate by row count,
    # because dataset rows are grouped by intent.
    by_intent: dict[str, list[str]] = defaultdict(list)
    for row in ds:
        intent = row.get("intent", "unknown")
        instruction = row.get("instruction", "")
        if instruction:
            by_intent[intent].append(instruction)

    if not by_intent:
        logger.warning("No Bitext queries loaded")
        return DatasetQueries(
            seeds=[], test_queries=[], dataset_name="bitext"
        )

    random.seed(42)

    # Sample each intent down so the total is near `size`.
    n_intents = len(by_intent)
    per_intent = max(10, size // n_intents)
    per_intent = min(per_intent, MAX_PER_INTENT)
    for intent in by_intent:
        instructions = by_intent[intent]
        random.shuffle(instructions)
        by_intent[intent] = instructions[:per_intent]

    total_loaded = sum(len(v) for v in by_intent.values())
    logger.info(
        "Bitext: loaded %d queries across %d intents "
        "(~%d per intent, target size=%d)",
        total_loaded, n_intents, per_intent, size,
    )

    # Split intent categories into seeded vs. held-out
    all_intents = sorted(by_intent.keys())
    random.shuffle(all_intents)
    n_holdout = max(1, int(len(all_intents) * HOLDOUT_INTENT_FRACTION))
    holdout_intents = set(all_intents[:n_holdout])
    seeded_intents = set(all_intents[n_holdout:])

    logger.info(
        "Bitext intent split: %d seeded, %d held-out (%.0f%%)",
        len(seeded_intents),
        len(holdout_intents),
        100 * len(holdout_intents) / len(all_intents),
    )

    seeds: list[Query] = []
    test_queries: list[Query] = []
    seed_prompts: list[str] = []

    # Seeded intents: 30% seeds, 70% semantic_variant tests
    for intent in sorted(seeded_intents):
        instructions = by_intent[intent]
        random.shuffle(instructions)
        n_seed = max(1, int(len(instructions) * 0.3))
        intent_seeds = instructions[:n_seed]
        intent_tests = instructions[n_seed:]

        for inst in intent_seeds:
            prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {inst}"
            seeds.append(Query(
                prompt=prompt,
                query_type="seed",
                source_dataset="bitext",
            ))
            seed_prompts.append(prompt)

        for inst in intent_tests:
            prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {inst}"
            test_queries.append(Query(
                prompt=prompt,
                query_type="semantic_variant",
                source_dataset="bitext",
            ))

    # Held-out intents: ALL queries become novel test queries (no seeds)
    for intent in sorted(holdout_intents):
        instructions = by_intent[intent]
        for inst in instructions:
            prompt = f"{SYSTEM_PROMPT}\n\nCustomer: {inst}"
            test_queries.append(Query(
                prompt=prompt,
                query_type="novel",
                source_dataset="bitext",
            ))

    # Add exact repeats from seeds (small fraction)
    n_exact = int(len(test_queries) * config.exact_repeat_ratio)
    for i in range(min(n_exact, len(seed_prompts))):
        idx = i % len(seed_prompts)
        test_queries.append(Query(
            prompt=seed_prompts[idx],
            query_type="exact_repeat",
            source_dataset="bitext",
            seed_index=idx,
        ))

    random.shuffle(test_queries)
    n_novel = sum(1 for q in test_queries if q.query_type == "novel")
    n_variant = sum(1 for q in test_queries if q.query_type == "semantic_variant")
    n_repeat = sum(1 for q in test_queries if q.query_type == "exact_repeat")
    logger.info(
        "Bitext: %d seeds, %d test queries "
        "(%d variant, %d novel, %d repeat) across %d intents",
        len(seeds), len(test_queries),
        n_variant, n_novel, n_repeat, len(by_intent),
    )

    return DatasetQueries(
        seeds=seeds,
        test_queries=test_queries,
        dataset_name="bitext",
    )
