"""Benchmark results data model and serialization."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

from benchmarks.e2e.metrics.latency import LatencyReport
from benchmarks.e2e.metrics.quality import compute_quality_metrics


@dataclass
class QueryResult:
    """Result of a single benchmark query."""

    prompt: str
    query_type: str  # exact_repeat, semantic_variant, novel
    source_dataset: str
    response_text: str
    ttft_ms: float
    total_ms: float
    cache_hit: bool
    cache_tier: str
    baseline_text: str | None = None
    baseline_ttft_ms: float | None = None
    bleu: float | None = None
    rouge_l: float | None = None


@dataclass
class DatasetResults:
    """Results for a single dataset."""

    dataset_name: str
    queries: list[QueryResult] = field(default_factory=list)

    @property
    def hit_rate(self) -> float:
        if not self.queries:
            return 0.0
        hits = sum(1 for q in self.queries if q.cache_hit)
        return hits / len(self.queries)

    @property
    def hit_rate_by_type(self) -> dict[str, float]:
        by_type: dict[str, list[bool]] = {}
        for q in self.queries:
            by_type.setdefault(q.query_type, []).append(q.cache_hit)
        return {
            t: sum(hits) / len(hits) if hits else 0.0
            for t, hits in by_type.items()
        }

    @property
    def avg_bleu(self) -> float:
        scores = [q.bleu for q in self.queries if q.bleu is not None]
        return sum(scores) / len(scores) if scores else 0.0

    @property
    def avg_rouge_l(self) -> float:
        scores = [q.rouge_l for q in self.queries if q.rouge_l is not None]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class BenchmarkResults:
    """Complete benchmark results across all datasets."""

    run_id: str
    model: str
    synapse_endpoint: str
    started_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    completed_at: str | None = None
    datasets: dict[str, DatasetResults] = field(default_factory=dict)
    latency: LatencyReport = field(default_factory=LatencyReport)

    def add_query_result(self, result: QueryResult) -> None:
        ds = result.source_dataset
        if ds not in self.datasets:
            self.datasets[ds] = DatasetResults(dataset_name=ds)
        self.datasets[ds].queries.append(result)
        self.latency.add_result(result.cache_tier, result.ttft_ms)

    def add_baseline_result(
        self, query_idx: int, ds_name: str, text: str, ttft_ms: float
    ) -> None:
        """Attach baseline response to an existing query result."""
        if ds_name not in self.datasets:
            return
        ds = self.datasets[ds_name]
        if query_idx >= len(ds.queries):
            return

        qr = ds.queries[query_idx]
        qr.baseline_text = text
        qr.baseline_ttft_ms = ttft_ms
        self.latency.add_baseline(ttft_ms)

        # Compute quality only for cache hits
        if qr.cache_hit and text:
            metrics = compute_quality_metrics(qr.response_text, text)
            qr.bleu = metrics["bleu"]
            qr.rouge_l = metrics["rouge_l"]

    def finalize(self) -> None:
        self.completed_at = datetime.now(timezone.utc).isoformat()

    def summary(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "datasets": {
                name: {
                    "total_queries": len(ds.queries),
                    "hit_rate": ds.hit_rate,
                    "hit_rate_by_type": ds.hit_rate_by_type,
                    "avg_bleu": ds.avg_bleu,
                    "avg_rouge_l": ds.avg_rouge_l,
                }
                for name, ds in self.datasets.items()
            },
            "latency": self.latency.summary(),
        }

    def save(self, output_dir: str) -> str:
        """Save results to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{self.run_id}.json")

        # Custom serialization for non-serializable fields
        data = {
            "run_id": self.run_id,
            "model": self.model,
            "synapse_endpoint": self.synapse_endpoint,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "summary": self.summary(),
            "queries": {
                ds_name: [
                    {
                        "prompt": q.prompt[:200],  # truncate for storage
                        "query_type": q.query_type,
                        "response_text": q.response_text[:500],
                        "ttft_ms": q.ttft_ms,
                        "total_ms": q.total_ms,
                        "cache_hit": q.cache_hit,
                        "cache_tier": q.cache_tier,
                        "baseline_ttft_ms": q.baseline_ttft_ms,
                        "bleu": q.bleu,
                        "rouge_l": q.rouge_l,
                    }
                    for q in ds.queries
                ]
                for ds_name, ds in self.datasets.items()
            },
        }

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        return path
