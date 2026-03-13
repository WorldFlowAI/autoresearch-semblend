"""Benchmark configuration."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkConfig:
    """Configuration for a benchmark run."""

    # Connection
    synapse_endpoint: str = "http://localhost:8080"
    vllm_endpoint: str = "http://localhost:8000"
    model: str = "mistralai/Mistral-7B-Instruct-v0.2"
    max_tokens: int = 256

    # Dataset sizes
    sharegpt_size: int = 5000
    multinews_size: int = 2000
    bitext_size: int = 3000

    # Query mix (must sum to 1.0)
    exact_repeat_ratio: float = 0.2
    semantic_variant_ratio: float = 0.5
    novel_ratio: float = 0.3

    # Execution
    warmup_delay_secs: float = 5.0
    warmup_batch_size: int = 20  # Send warmup seeds in batches to avoid overloading proxy
    warmup_batch_delay_secs: float = 2.0  # Pause between warmup batches
    concurrency: int = 4
    timeout_secs: float = 60.0

    # Quality thresholds
    min_bleu: float = 0.7
    min_rouge_l: float = 0.8

    # Output
    output_dir: str = "results"
    datasets: list[str] = field(
        default_factory=lambda: ["sharegpt", "multinews", "bitext"]
    )

    def dataset_size(self, name: str) -> int:
        sizes = {
            "sharegpt": self.sharegpt_size,
            "multinews": self.multinews_size,
            "bitext": self.bitext_size,
        }
        return sizes.get(name, 1000)
