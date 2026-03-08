#!/usr/bin/env python3
"""Build real-dataset clusters for SemBlend benchmarks.

Generates per-dataset cluster files from CNN/DailyMail, MultiNews,
WikiHow, and SAMSum, plus a combined file for backward compatibility.

Usage:
    python -m benchmarks.e2e.build_clusters
    python -m benchmarks.e2e.build_clusters --datasets cnn_dailymail multinews
    python -m benchmarks.e2e.build_clusters --vllm-endpoint http://localhost:8000
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from benchmarks.e2e.real_dataset_builder import (
    build_clusters,
    load_cnn_dailymail,
    load_multinews,
    load_samsum,
    load_wikihow,
    load_xsum,
)

NUM_ARTICLES = 100  # per dataset
TARGET_LENGTHS = [2048, 4096, 8192, 16384]
DATA_DIR = Path("benchmarks/data")
COMBINED_OUTPUT = DATA_DIR / "semblend_clusters.json"

ALL_DATASETS = ["cnn_dailymail", "xsum", "multinews", "wikihow"]

DATASET_LOADERS = {
    "cnn_dailymail": load_cnn_dailymail,
    "xsum": load_xsum,
    "multinews": load_multinews,
    "wikihow": load_wikihow,
    "samsum": load_samsum,
}

DATASET_OUTPUT_FILES = {
    "cnn_dailymail": DATA_DIR / "cnn_dailymail_clusters.json",
    "xsum": DATA_DIR / "xsum_clusters.json",
    "multinews": DATA_DIR / "multinews_clusters.json",
    "wikihow": DATA_DIR / "wikihow_clusters.json",
    "samsum": DATA_DIR / "samsum_clusters.json",
}


def print_dataset_stats(dataset_name: str, clusters: list) -> None:
    """Print per-dataset statistics after building clusters."""
    if not clusters:
        print(f"  [{dataset_name}] No clusters built.", flush=True)
        return

    token_lengths = [c.seed_token_count for c in clusters]
    mean_len = statistics.mean(token_lengths)

    variation_types: Counter = Counter()
    for c in clusters:
        for v in c.variations:
            variation_types[v.overlap_type] += 1

    print(f"  [{dataset_name}] {len(clusters)} clusters, "
          f"mean token length: {mean_len:.0f}", flush=True)
    print(f"    variation types: {dict(sorted(variation_types.items()))}",
          flush=True)


def save_clusters(clusters: list, output_path: Path) -> None:
    """Serialize clusters to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out = [asdict(c) for c in clusters]
    with open(output_path, "w") as f:
        json.dump(out, f)
    print(f"  Saved {len(out)} clusters to {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Build real-dataset clusters for SemBlend benchmarks."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=ALL_DATASETS,
        choices=ALL_DATASETS,
        help="Datasets to build (default: all 4)",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default=None,
        help="Optional vLLM endpoint URL for LLM-based paraphrase variations",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for vLLM completions (default: auto-detect from endpoint)",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct-AWQ")

    all_clusters = []

    for ds_name in args.datasets:
        loader = DATASET_LOADERS[ds_name]
        print(f"Loading {ds_name}...", flush=True)
        articles = loader(max_articles=NUM_ARTICLES)
        print(f"  {len(articles)} articles", flush=True)

        print(f"Building clusters at {TARGET_LENGTHS}...", flush=True)
        t0 = time.time()
        clusters = build_clusters(
            articles,
            tok,
            target_lengths=TARGET_LENGTHS,
            vllm_endpoint=args.vllm_endpoint,
            vllm_model=args.model,
        )
        elapsed = time.time() - t0
        print(f"  Built {len(clusters)} clusters in {elapsed:.1f}s", flush=True)

        print_dataset_stats(ds_name, clusters)
        save_clusters(clusters, DATASET_OUTPUT_FILES[ds_name])

        all_clusters.extend(clusters)

    # Backward compatibility: combined file
    if all_clusters:
        save_clusters(all_clusters, COMBINED_OUTPUT)
        print(f"\nTotal: {len(all_clusters)} clusters across "
              f"{len(args.datasets)} datasets", flush=True)


if __name__ == "__main__":
    main()
