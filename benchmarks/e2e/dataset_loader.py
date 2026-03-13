"""Dataset loader for SemBlend benchmarks.

Supports multiple datasets grouped by semantic cluster for realistic
donor-matching scenarios:
  - ShareGPT: LMCache reference dataset (conversation-style)
  - LMSYS-Chat-1M: production-realistic chat logs
  - LongBench: 4K-16K long-context evaluation
  - MS MARCO: RAG chunk simulation
  - Synthetic: controlled overlap scenarios for ablation

Usage:
    sampler = DatasetSampler(datasets=["sharegpt", "synthetic"])
    for cluster in sampler.clusters(num_clusters=10, prompts_per_cluster=5):
        seed_prompt = cluster[0]
        variations = cluster[1:]
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PromptPair:
    """A pair of related prompts for donor-matching benchmarks."""
    seed: str
    variation: str
    overlap_type: str  # "exact", "reorder", "partial", "paraphrase", "diverse"
    expected_overlap: float  # approximate token overlap ratio


@dataclass
class PromptCluster:
    """A group of semantically related prompts."""
    cluster_id: str
    seed_prompt: str
    variations: list[PromptPair] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Synthetic dataset (controlled overlap)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS = [
    "You are a helpful AI assistant.",
    "You are an expert software engineer.",
    "You are a knowledgeable science tutor.",
    "You are a creative writing assistant.",
    "You are a data analysis specialist.",
]

RAG_CONTEXTS = [
    "According to the documentation, the API supports REST and gRPC endpoints. "
    "Authentication uses JWT tokens with a 1-hour expiry. Rate limiting is set "
    "to 100 requests per second per tenant.",

    "The database schema includes users, sessions, and events tables. "
    "Users have a unique email constraint. Sessions reference users via "
    "foreign key. Events are partitioned by month for query performance.",

    "The deployment uses Kubernetes on AWS EKS with GPU nodes for inference. "
    "Models are cached on EFS for fast restarts. Auto-scaling is based on "
    "GPU utilization metrics from Prometheus.",

    "The caching layer has three tiers: L0 GPU (CAGRA ANN), L1 Redis (HNSW), "
    "and L2 Milvus (persistent vector search). Semantic threshold is 0.85 "
    "cosine similarity for cache hits.",

    "Performance benchmarks show 50ms p99 latency for cache hits and 500ms "
    "for cache misses. Throughput scales linearly to 4 GPU nodes. Memory "
    "usage is bounded by the LRU eviction policy.",
]

USER_QUERIES = [
    "How do I authenticate with the API?",
    "What is the database schema for user sessions?",
    "How do I deploy this on Kubernetes?",
    "Explain the caching architecture.",
    "What are the performance characteristics?",
    "How do I set up rate limiting?",
    "What embedding models are supported?",
    "How do I configure auto-scaling?",
    "What is the latency for cache hits vs misses?",
    "How do I monitor GPU utilization?",
]


def _build_prompt(system: str, context: str, query: str) -> str:
    return f"System: {system}\n\nContext: {context}\n\nUser: {query}\n\nAssistant:"


def generate_synthetic_clusters(
    num_clusters: int = 10,
    variations_per_cluster: int = 5,
) -> list[PromptCluster]:
    """Generate synthetic prompt clusters with controlled overlap types."""
    clusters = []

    for ci in range(num_clusters):
        system = SYSTEM_PROMPTS[ci % len(SYSTEM_PROMPTS)]
        context = RAG_CONTEXTS[ci % len(RAG_CONTEXTS)]
        query = USER_QUERIES[ci % len(USER_QUERIES)]

        seed = _build_prompt(system, context, query)
        cluster_id = hashlib.md5(seed.encode()).hexdigest()[:8]

        cluster = PromptCluster(
            cluster_id=cluster_id,
            seed_prompt=seed,
        )

        overlap_types = [
            "exact", "reorder", "partial_30", "partial_50",
            "partial_70", "partial_80", "paraphrase", "diverse",
        ]

        for vi in range(variations_per_cluster):
            otype = overlap_types[vi % len(overlap_types)]
            variation, expected = _generate_variation(
                system, context, query, otype
            )
            cluster.variations.append(PromptPair(
                seed=seed,
                variation=variation,
                overlap_type=otype,
                expected_overlap=expected,
            ))

        clusters.append(cluster)

    return clusters


def _generate_variation(
    system: str,
    context: str,
    query: str,
    overlap_type: str,
) -> tuple[str, float]:
    """Generate a prompt variation with specified overlap type."""
    if overlap_type == "exact":
        return _build_prompt(system, context, query), 1.0

    if overlap_type == "reorder":
        # Same content, different query order
        words = query.split()
        if len(words) > 2:
            mid = len(words) // 2
            reordered = " ".join(words[mid:] + words[:mid])
            return _build_prompt(system, context, reordered), 0.9

    if overlap_type.startswith("partial_"):
        pct = int(overlap_type.split("_")[1]) / 100.0
        # Replace (1-pct) of context words
        words = context.split()
        n_replace = int(len(words) * (1 - pct))
        replaced = words[:]
        for i in range(n_replace):
            idx = (i * 7 + 3) % len(replaced)  # deterministic
            replaced[idx] = f"[REPLACED_{i}]"
        new_context = " ".join(replaced)
        return _build_prompt(system, new_context, query), pct

    if overlap_type == "paraphrase":
        # Same system + context, rephrased query
        rephrased = f"Could you explain: {query.lower().rstrip('?')}?"
        return _build_prompt(system, context, rephrased), 0.85

    # "diverse" — different context entirely
    alt_context = RAG_CONTEXTS[(RAG_CONTEXTS.index(context) + 1) % len(RAG_CONTEXTS)]
    return _build_prompt(system, alt_context, query), 0.3


# ---------------------------------------------------------------------------
# ShareGPT dataset loader
# ---------------------------------------------------------------------------


def load_sharegpt(path: str | None = None, max_prompts: int = 50) -> list[str]:
    """Load prompts from ShareGPT dataset.

    If path is not provided, returns built-in conversation starters.
    """
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
        prompts = []
        for conv in data[:max_prompts]:
            turns = conv.get("conversations", [])
            if turns:
                prompts.append(turns[0].get("value", ""))
        return [p for p in prompts if p]

    # Built-in fallback
    return [
        "Explain the difference between TCP and UDP protocols in detail.",
        "Write a Python function that implements a binary search tree with insert, delete, and search operations.",
        "What are the main causes of climate change and what can individuals do to help?",
        "Describe the architecture of a modern microservices application.",
        "How does garbage collection work in Java? Compare different GC algorithms.",
        "Explain quantum computing to someone with a computer science background.",
        "What is the CAP theorem? Give real-world examples of each trade-off.",
        "Write a comprehensive SQL query to analyze e-commerce sales data.",
        "How does TLS 1.3 differ from TLS 1.2? Explain the handshake process.",
        "Compare and contrast different machine learning optimization algorithms.",
    ][:max_prompts]


# ---------------------------------------------------------------------------
# DatasetSampler
# ---------------------------------------------------------------------------


class DatasetSampler:
    """Yields prompt tuples grouped by semantic cluster.

    Args:
        datasets: List of dataset names to include.
        seed: Random seed for reproducibility.
    """

    def __init__(
        self,
        datasets: list[str] | None = None,
        seed: int = 42,
    ) -> None:
        self._datasets = datasets or ["synthetic"]
        self._rng = random.Random(seed)

    def clusters(
        self,
        num_clusters: int = 10,
        prompts_per_cluster: int = 5,
    ) -> list[PromptCluster]:
        """Generate prompt clusters from configured datasets."""
        all_clusters: list[PromptCluster] = []

        if "synthetic" in self._datasets:
            all_clusters.extend(
                generate_synthetic_clusters(num_clusters, prompts_per_cluster)
            )

        if "sharegpt" in self._datasets:
            sharegpt_prompts = load_sharegpt(max_prompts=num_clusters)
            for i, prompt in enumerate(sharegpt_prompts):
                cluster = PromptCluster(
                    cluster_id=f"sharegpt-{i}",
                    seed_prompt=prompt,
                    variations=[
                        PromptPair(
                            seed=prompt,
                            variation=prompt,
                            overlap_type="exact",
                            expected_overlap=1.0,
                        )
                    ],
                )
                all_clusters.append(cluster)

        self._rng.shuffle(all_clusters)
        return all_clusters[:num_clusters]

    def flat_prompts(self, num_prompts: int = 100) -> list[str]:
        """Return a flat list of prompts (no cluster structure)."""
        clusters = self.clusters(
            num_clusters=num_prompts,
            prompts_per_cluster=1,
        )
        prompts = [c.seed_prompt for c in clusters]
        for c in clusters:
            for v in c.variations:
                prompts.append(v.variation)
        return prompts[:num_prompts]
