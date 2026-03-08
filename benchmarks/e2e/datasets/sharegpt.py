"""Conversation dataset loader (OpenAssistant/ShareGPT)."""

from __future__ import annotations

import logging
import random

from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.datasets.loader import DatasetQueries, Query

logger = logging.getLogger(__name__)

# Primary: OpenAssistant (publicly available)
# Fallback: ShareGPT Vicuna (may be unavailable)
DATASETS = [
    ("OpenAssistant/oasst1", "oasst1"),
    ("anon8231489123/ShareGPT_Vicuna_unfiltered", "sharegpt_vicuna"),
]


def load_sharegpt(size: int, config: BenchmarkConfig) -> DatasetQueries:
    """Load conversation prompts and split into seeds/tests.

    Tries OpenAssistant first (publicly available), falls back to ShareGPT.
    Extracts user/prompter messages as prompts for diverse topic coverage.
    """
    from datasets import load_dataset

    conversations: list[str] = []

    for dataset_id, ds_type in DATASETS:
        try:
            logger.info("Trying dataset: %s", dataset_id)
            if ds_type == "oasst1":
                conversations = _load_oasst1(dataset_id, size)
            else:
                conversations = _load_sharegpt_vicuna(dataset_id, size)

            if conversations:
                logger.info(
                    "Loaded %d conversations from %s",
                    len(conversations), dataset_id,
                )
                break
        except Exception:
            logger.warning(
                "Failed to load %s, trying next", dataset_id, exc_info=True
            )

    if not conversations:
        logger.warning("No conversation data loaded from any source")
        return DatasetQueries(
            seeds=[], test_queries=[], dataset_name="sharegpt"
        )

    # Split: 40% seeds, 60% test pool
    random.seed(42)
    random.shuffle(conversations)
    split_idx = int(len(conversations) * 0.4)
    seed_prompts = conversations[:split_idx]
    novel_pool = conversations[split_idx:]

    seeds = [
        Query(
            prompt=p,
            query_type="seed",
            source_dataset="sharegpt",
        )
        for p in seed_prompts
    ]

    # Build test queries with the configured mix
    test_queries: list[Query] = []
    n_test = len(conversations) - split_idx
    n_exact = int(n_test * config.exact_repeat_ratio)
    n_semantic = int(n_test * config.semantic_variant_ratio)
    n_novel = n_test - n_exact - n_semantic

    # Exact repeats: pick random seeds
    for i in range(min(n_exact, len(seed_prompts))):
        idx = i % len(seed_prompts)
        test_queries.append(Query(
            prompt=seed_prompts[idx],
            query_type="exact_repeat",
            source_dataset="sharegpt",
            seed_index=idx,
        ))

    # Semantic variants: minor prefix modification
    for i in range(min(n_semantic, len(seed_prompts))):
        idx = i % len(seed_prompts)
        variant = _make_semantic_variant(seed_prompts[idx])
        test_queries.append(Query(
            prompt=variant,
            query_type="semantic_variant",
            source_dataset="sharegpt",
            seed_index=idx,
        ))

    # Novel queries
    for i in range(min(n_novel, len(novel_pool))):
        test_queries.append(Query(
            prompt=novel_pool[i],
            query_type="novel",
            source_dataset="sharegpt",
        ))

    random.shuffle(test_queries)
    logger.info(
        "ShareGPT: %d seeds, %d test queries "
        "(exact=%d, semantic=%d, novel=%d)",
        len(seeds), len(test_queries), n_exact, n_semantic, n_novel,
    )

    return DatasetQueries(
        seeds=seeds,
        test_queries=test_queries,
        dataset_name="sharegpt",
    )


def _load_oasst1(dataset_id: str, size: int) -> list[str]:
    """Load prompter messages from OpenAssistant oasst1."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    conversations = []
    for row in ds:
        if row.get("role") != "prompter":
            continue
        text = row.get("text", "").strip()
        if len(text) < 20:
            continue
        # Filter to English
        if row.get("lang", "en") != "en":
            continue
        conversations.append(text)
        if len(conversations) >= size:
            break
    return conversations


def _load_sharegpt_vicuna(dataset_id: str, size: int) -> list[str]:
    """Load first-turn user messages from ShareGPT Vicuna."""
    from datasets import load_dataset

    ds = load_dataset(dataset_id, split="train")
    conversations = []
    for row in ds:
        convos = row.get("conversations", [])
        if not convos:
            continue
        user_msgs = [c for c in convos if c.get("from") == "human"]
        if user_msgs:
            text = user_msgs[0].get("value", "").strip()
            if len(text) >= 20:
                conversations.append(text)
        if len(conversations) >= size:
            break
    return conversations


def _make_semantic_variant(prompt: str) -> str:
    """Create a semantically similar variant of a prompt."""
    prefixes = [
        "Can you help me with this: ",
        "I'd like to know: ",
        "Please explain: ",
        "Could you tell me about: ",
    ]
    prefix = random.choice(prefixes)
    return prefix + prompt
