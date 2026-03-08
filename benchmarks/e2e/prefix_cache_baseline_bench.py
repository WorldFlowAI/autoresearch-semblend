#!/usr/bin/env python3
"""vLLM native prefix cache baseline benchmark for SemBlend paper.

Measures vLLM's built-in prefix cache performance across scenarios where
it helps (exact prefix match) and where it fails (reorder, partial overlap,
paraphrase). This establishes the baseline that SemBlend improves upon.

Key insight: vLLM prefix cache only matches exact token prefixes. Any
lexical change — reordering sentences, paraphrasing, or partial overlap —
defeats the prefix cache entirely. SemBlend's semantic KV reuse handles
all of these cases.

Usage:
    python -m benchmarks.e2e.prefix_cache_baseline_bench \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --token-lengths "2048,5120,8192,16000" \
        --runs 10 \
        --output-dir results/prefix_cache_baseline
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
except ImportError:
    print("ERROR: pip install transformers")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

class Scenario(str, Enum):
    EXACT_PREFIX = "exact_prefix"
    REORDER = "reorder"
    PARTIAL_75 = "partial_75"
    PARTIAL_50 = "partial_50"
    PARAPHRASE = "paraphrase"


# ---------------------------------------------------------------------------
# Context corpus — diverse technical paragraphs
# ---------------------------------------------------------------------------

CONTEXTS = [
    (
        "Machine learning has transformed natural language processing. "
        "Transformer architectures replaced recurrent neural networks with "
        "self-attention mechanisms allowing massive parallelization. Models "
        "grew from 110M parameters to hundreds of billions. Efficient "
        "inference became critical with techniques like quantization, KV "
        "cache optimization, continuous batching, and speculative decoding. "
        "Flash Attention reduced memory complexity from quadratic to linear. "
        "Rotary Position Embeddings encode positional information as rotations "
        "in the complex plane, enabling length extrapolation."
    ),
    (
        "Kubernetes orchestrates containerized applications across clusters. "
        "The control plane consists of API server, etcd, scheduler, and "
        "controller manager. Worker nodes run kubelet and container runtime. "
        "Pods are the smallest deployable units containing one or more containers. "
        "Deployments manage ReplicaSets for rolling updates and rollbacks. "
        "Services provide stable networking via ClusterIP, NodePort, or LoadBalancer. "
        "GPU scheduling uses device plugins and resource limits. "
        "Helm packages applications as charts with templated YAML manifests."
    ),
    (
        "Distributed caching architectures form the backbone of high "
        "performance systems. Redis provides in-memory key-value storage "
        "with rich data structures including lists, sets, sorted sets, hashes, "
        "and streams. Redis Cluster enables horizontal scaling via hash slot "
        "partitioning across multiple nodes. Vector similarity search uses "
        "HNSW graphs for approximate nearest neighbor queries with sublinear "
        "time complexity. Multi-tier caching stacks L0 GPU cache, L1 Redis, "
        "and L2 vector database for optimal latency and cost."
    ),
    (
        "Cloud GPU computing economics drive architectural decisions in AI "
        "infrastructure. T4 instances cost approximately $0.53 per hour with "
        "16GB HBM and 320 GB/s memory bandwidth. A10G costs $1.01 per hour "
        "with 24GB and 600 GB/s bandwidth. KV cache memory consumption grows "
        "linearly with sequence length and batch size. For 7B parameter models "
        "with 32 transformer layers, each token requires approximately 524KB "
        "of KV cache. GPU memory utilization targets are typically 85 to 90 "
        "percent. AWQ quantization reduces model size by 4x with minimal accuracy loss."
    ),
    (
        "Quantum computing leverages quantum mechanical phenomena for "
        "computation beyond classical limits. Qubits exist in superposition "
        "of zero and one states unlike deterministic classical bits. "
        "Entanglement creates strongly correlated qubit pairs across distance. "
        "Quantum gates manipulate qubit states through unitary transformations. "
        "Error correction codes protect quantum information against decoherence "
        "and operational errors. Shor's algorithm factors large integers in "
        "polynomial time. Grover's algorithm provides quadratic speedup for "
        "unstructured database search problems."
    ),
    (
        "Database systems provide persistent structured data storage and "
        "retrieval with strong consistency guarantees. Relational databases "
        "use SQL for querying normalized tables with foreign key relationships. "
        "B-tree indexes enable logarithmic lookup time for range and equality "
        "queries. Write-ahead logging ensures ACID transactions survive system "
        "crashes. Columnar storage formats like Parquet optimize analytical "
        "workloads by reducing I/O. LSM trees power high write throughput "
        "in key-value stores like RocksDB and LevelDB."
    ),
    (
        "Computer networking follows the OSI seven layer model for protocol "
        "abstraction. The physical layer handles bit transmission over copper "
        "fiber or wireless media. Data link layer manages frame delivery and "
        "MAC addressing. Network layer routes packets via IP addressing and "
        "forwarding tables. Transport layer ensures reliable ordered delivery "
        "through TCP with congestion control or fast unreliable delivery via "
        "UDP. HTTP operates at the application layer for web communication. "
        "TLS encrypts data in transit using asymmetric key exchange followed "
        "by symmetric encryption."
    ),
    (
        "Operating systems manage hardware resources for concurrent applications "
        "while providing isolation and security. Process scheduling algorithms "
        "include round-robin for fairness, priority-based for responsiveness, "
        "and the completely fair scheduler for proportional share allocation. "
        "Virtual memory maps logical addresses to physical addresses using "
        "multi-level page tables with TLB caching. File systems like ext4 "
        "and ZFS organize data on block storage devices with journaling for "
        "crash consistency. System calls provide the interface between user "
        "space applications and kernel space."
    ),
]

# Paraphrased versions — semantically equivalent, lexically different
CONTEXTS_PARAPHRASED = [
    (
        "Natural language processing was revolutionized by deep learning "
        "techniques. The transformer model supplanted RNNs by introducing "
        "attention across all positions simultaneously, enabling parallel "
        "training. Language models scaled from a hundred million weights to "
        "hundreds of billions. Making inference efficient became paramount "
        "through weight quantization, key-value caching, continuous request "
        "batching, and draft-model speculation. FlashAttention brought memory "
        "usage down from O(n^2) to O(n). RoPE represents positions as complex "
        "rotations, supporting sequence lengths beyond training distribution."
    ),
    (
        "Container orchestration at scale is handled by Kubernetes. Its "
        "control plane includes the API server, the etcd key-value store, "
        "the kube-scheduler, and various controllers. Each worker node runs "
        "a kubelet daemon alongside a container runtime engine. A pod wraps "
        "one or more tightly coupled containers as the atomic deployment unit. "
        "Deployment objects coordinate ReplicaSet updates for zero-downtime "
        "rollouts. Networking abstractions called Services expose pods via "
        "stable virtual IPs. GPU workloads leverage the NVIDIA device plugin "
        "and resource quotas. Helm is the package manager for Kubernetes, "
        "using templated manifests bundled into charts."
    ),
    (
        "High-throughput systems rely on distributed cache layers for "
        "performance. The Redis data store keeps everything in RAM and "
        "supports diverse structures: linked lists, hash maps, sorted "
        "sets, and event streams. Clustering in Redis uses consistent "
        "hashing across 16384 slots to spread load. For similarity-based "
        "retrieval, HNSW indexes provide sub-millisecond approximate nearest "
        "neighbor lookups. A multi-level caching hierarchy places the hottest "
        "data on GPU memory (L0), warm data in Redis (L1), and cold data in "
        "a vector store (L2)."
    ),
    (
        "Economic considerations around GPU cloud instances shape AI system "
        "design. An NVIDIA T4 card runs about fifty-three cents an hour, "
        "offering 16 gigabytes of high-bandwidth memory at 320 GB/s. The "
        "A10G is roughly a dollar per hour, providing 24 GB at 600 GB/s "
        "bandwidth. Memory consumed by the key-value cache scales linearly "
        "with both context window and concurrent requests. A seven-billion "
        "parameter transformer with 32 layers allocates roughly half a "
        "megabyte of KV state per token. Operators typically cap GPU memory "
        "use at 85 to 90 percent. Activation-aware weight quantization "
        "compresses the model four-fold with negligible quality impact."
    ),
    (
        "Harnessing quantum phenomena opens computational possibilities "
        "unreachable by classical machines. A qubit occupies a probabilistic "
        "blend of binary states, unlike a conventional bit. Quantum "
        "entanglement links particles so measuring one instantly determines "
        "the other. Logic operations on qubits are performed by unitary "
        "quantum gates. Because qubits are fragile, error-correcting codes "
        "protect stored quantum information from noise and decoherence. "
        "Peter Shor devised a quantum algorithm that breaks RSA by factoring "
        "large numbers efficiently. Lov Grover's search procedure achieves "
        "a square-root reduction in query complexity."
    ),
    (
        "Persistent data management is the domain of database engines that "
        "guarantee consistency under concurrent access. SQL-based relational "
        "systems query tables linked by foreign keys across normalized schemas. "
        "B-tree data structures enable O(log n) key lookups supporting both "
        "point and range queries. Crash recovery relies on write-ahead log "
        "records replayed after failure to restore ACID properties. Column-"
        "oriented formats such as Apache Parquet slash I/O for analytics by "
        "reading only needed columns. Log-structured merge trees, used by "
        "RocksDB and LevelDB, optimize for heavy write workloads."
    ),
    (
        "Network communication is conceptualized through the seven-layer "
        "OSI reference model. Layer one transmits raw bits across physical "
        "media such as copper, fiber optic cable, or radio frequencies. "
        "The data-link layer frames data and handles MAC-level addressing. "
        "IP routing and packet forwarding occur at the network layer. "
        "Reliable, ordered byte streams are provided by TCP at the transport "
        "layer, while UDP offers minimal-overhead datagram delivery. Web "
        "traffic flows over HTTP at layer seven. Secure connections use TLS, "
        "negotiating an asymmetric handshake before switching to symmetric "
        "cipher for bulk data encryption."
    ),
    (
        "An operating system arbitrates access to CPU, memory, and I/O "
        "devices among competing processes. Scheduling policies range from "
        "simple round-robin time slicing to priority queues and the Linux "
        "CFS that allocates CPU proportionally. The virtual memory subsystem "
        "translates per-process logical addresses into physical frames via "
        "hierarchical page tables accelerated by TLB hardware. Filesystems "
        "including ext4 and ZFS manage on-disk layout with journaling or "
        "copy-on-write semantics for crash safety. The syscall interface "
        "mediates all transitions from userland into the kernel."
    ),
]


# ---------------------------------------------------------------------------
# Tokenizer-verified prompt construction
# ---------------------------------------------------------------------------

_TOKENIZER = None


def _get_tokenizer(model_name: str) -> AutoTokenizer:
    global _TOKENIZER
    if _TOKENIZER is None:
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on period boundaries."""
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p for p in parts if p.strip()]


def build_prompt(
    context: str,
    question: str,
    target_tokens: int,
    model_name: str,
) -> str:
    """Build a prompt padded/truncated to exactly target_tokens via tokenizer."""
    tokenizer = _get_tokenizer(model_name)
    prefix = (
        f"<|im_start|>system\nYou are a helpful AI assistant. "
        f"Answer based only on the provided context.<|im_end|>\n"
        f"<|im_start|>user\nContext:\n"
    )
    suffix = (
        f"\n\nQuestion: {question}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )

    prefix_toks = len(tokenizer.encode(prefix))
    suffix_toks = len(tokenizer.encode(suffix))
    budget = target_tokens - prefix_toks - suffix_toks

    if budget <= 0:
        full = prefix + context + suffix
        ids = tokenizer.encode(full)
        return tokenizer.decode(ids[:target_tokens])

    # Repeat context to fill budget
    body = context
    while len(tokenizer.encode(body)) < budget + 200:
        body = body + " " + context

    body_ids = tokenizer.encode(body)
    if len(body_ids) > budget:
        body = tokenizer.decode(body_ids[:budget])

    full = prefix + body + suffix
    actual = len(tokenizer.encode(full))

    # Final trim if overshot
    if actual > target_tokens + 10:
        full_ids = tokenizer.encode(full)
        full = tokenizer.decode(full_ids[:target_tokens])

    return full


# ---------------------------------------------------------------------------
# Scenario variant generators
# ---------------------------------------------------------------------------

def make_exact_prefix_variant(
    seed_context: str,
    seed_question: str,
    variant_question: str,
    target_tokens: int,
    model_name: str,
) -> str:
    """Same context, different question — prefix cache should hit."""
    return build_prompt(seed_context, variant_question, target_tokens, model_name)


def make_reorder_variant(
    seed_context: str,
    question: str,
    target_tokens: int,
    model_name: str,
    rng: random.Random,
) -> str:
    """Shuffle sentences in the context — prefix cache will miss."""
    sentences = _split_sentences(seed_context)
    shuffled = list(sentences)
    rng.shuffle(shuffled)
    # Ensure it actually changed order
    if shuffled == sentences and len(sentences) > 1:
        shuffled[0], shuffled[-1] = shuffled[-1], shuffled[0]
    reordered = " ".join(shuffled)
    return build_prompt(reordered, question, target_tokens, model_name)


def make_partial_variant(
    seed_context: str,
    question: str,
    target_tokens: int,
    model_name: str,
    overlap_pct: float,
    rng: random.Random,
) -> str:
    """Keep overlap_pct of original sentences, replace rest with filler."""
    sentences = _split_sentences(seed_context)
    n_keep = max(1, int(len(sentences) * overlap_pct))
    kept_indices = sorted(rng.sample(range(len(sentences)), min(n_keep, len(sentences))))
    filler = (
        "Additional background information is provided for context. "
        "This supplementary material covers related topics and findings. "
        "The data has been collected from multiple authoritative sources. "
        "Cross-referencing confirms the reliability of these observations."
    )
    result_sentences = []
    for i, sent in enumerate(sentences):
        if i in kept_indices:
            result_sentences.append(sent)
        else:
            result_sentences.append(filler)
    partial = " ".join(result_sentences)
    return build_prompt(partial, question, target_tokens, model_name)


def make_paraphrase_variant(
    paraphrased_context: str,
    question: str,
    target_tokens: int,
    model_name: str,
) -> str:
    """Use pre-written paraphrased context — prefix cache will miss."""
    return build_prompt(paraphrased_context, question, target_tokens, model_name)


# ---------------------------------------------------------------------------
# TTFT measurement (streaming, synchronous requests)
# ---------------------------------------------------------------------------

def measure_ttft(endpoint: str, model: str, prompt: str) -> float:
    """Send a streaming completion request and return TTFT in milliseconds."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": 5,
            "temperature": 0.0,
            "stream": True,
        },
        timeout=300.0,
        stream=True,
    )
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8", errors="replace")
        if decoded.startswith("data: ") and decoded[6:].strip() != "[DONE]":
            ttft = (time.monotonic() - t0) * 1000
            # Drain remaining stream
            for _ in resp.iter_lines():
                pass
            return ttft
    raise ValueError("No tokens received from model")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class ScenarioResult:
    scenario: str
    target_tokens: int
    seed_ttft_ms: float
    variant_ttfts_ms: list[float] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def variant_p50(self) -> Optional[float]:
        if not self.variant_ttfts_ms:
            return None
        s = sorted(self.variant_ttfts_ms)
        return s[len(s) // 2]

    @property
    def variant_mean(self) -> Optional[float]:
        if not self.variant_ttfts_ms:
            return None
        return statistics.mean(self.variant_ttfts_ms)

    @property
    def variant_stdev(self) -> Optional[float]:
        if len(self.variant_ttfts_ms) < 2:
            return None
        return statistics.stdev(self.variant_ttfts_ms)

    @property
    def speedup(self) -> Optional[float]:
        p50 = self.variant_p50
        if p50 is None or p50 <= 0 or self.seed_ttft_ms <= 0:
            return None
        return self.seed_ttft_ms / p50


# ---------------------------------------------------------------------------
# Questions pool
# ---------------------------------------------------------------------------

SEED_QUESTIONS = [
    "What are the key technical concepts described in the context?",
    "Summarize the main topics covered in the passage above.",
    "What are the most important points mentioned in this text?",
    "Explain the primary technologies discussed in the context.",
]

VARIANT_QUESTIONS = [
    "What are the practical implications of the technologies described?",
    "How do the concepts in this passage relate to modern computing?",
    "What performance characteristics are mentioned in the context?",
    "What trends or developments does this passage highlight?",
]


# ---------------------------------------------------------------------------
# Core benchmark runner
# ---------------------------------------------------------------------------

def run_scenario(
    endpoint: str,
    model: str,
    scenario: Scenario,
    context_idx: int,
    target_tokens: int,
    runs: int,
    rng: random.Random,
) -> ScenarioResult:
    """Run a single scenario: seed + n variant measurements."""
    context = CONTEXTS[context_idx % len(CONTEXTS)]
    paraphrased = CONTEXTS_PARAPHRASED[context_idx % len(CONTEXTS_PARAPHRASED)]
    seed_q = SEED_QUESTIONS[context_idx % len(SEED_QUESTIONS)]
    variant_q = VARIANT_QUESTIONS[context_idx % len(VARIANT_QUESTIONS)]

    # Build seed prompt and populate prefix cache
    seed_prompt = build_prompt(context, seed_q, target_tokens, model)
    seed_ttft = measure_ttft(endpoint, model, seed_prompt)
    time.sleep(0.5)  # Let prefix cache settle

    result = ScenarioResult(
        scenario=scenario.value,
        target_tokens=target_tokens,
        seed_ttft_ms=seed_ttft,
    )

    for run_idx in range(runs):
        try:
            if scenario == Scenario.EXACT_PREFIX:
                variant_prompt = make_exact_prefix_variant(
                    context, seed_q, variant_q, target_tokens, model,
                )
            elif scenario == Scenario.REORDER:
                variant_prompt = make_reorder_variant(
                    context, seed_q, target_tokens, model, rng,
                )
            elif scenario == Scenario.PARTIAL_75:
                variant_prompt = make_partial_variant(
                    context, seed_q, target_tokens, model, 0.75, rng,
                )
            elif scenario == Scenario.PARTIAL_50:
                variant_prompt = make_partial_variant(
                    context, seed_q, target_tokens, model, 0.50, rng,
                )
            elif scenario == Scenario.PARAPHRASE:
                variant_prompt = make_paraphrase_variant(
                    paraphrased, seed_q, target_tokens, model,
                )
            else:
                raise ValueError(f"Unknown scenario: {scenario}")

            ttft = measure_ttft(endpoint, model, variant_prompt)
            result.variant_ttfts_ms.append(ttft)
            speedup = seed_ttft / ttft if ttft > 0 else 0
            print(f"      run {run_idx + 1}/{runs}: {ttft:.0f}ms (vs seed {seed_ttft:.0f}ms, {speedup:.2f}x)")
        except Exception as exc:
            result.errors.append(f"run {run_idx + 1}: {str(exc)[:200]}")
            print(f"      run {run_idx + 1}/{runs}: ERROR {exc}")
        time.sleep(0.3)

    return result


def run_length_suite(
    endpoint: str,
    model: str,
    target_tokens: int,
    runs: int,
    rng: random.Random,
) -> dict[str, list[ScenarioResult]]:
    """Run all scenarios for a single token length."""
    scenarios_results: dict[str, list[ScenarioResult]] = {}

    for scenario in Scenario:
        print(f"\n  --- {scenario.value.upper()} ---")
        results_for_scenario = []

        # Use multiple contexts to get diverse measurements
        n_contexts = min(4, len(CONTEXTS))
        for ctx_idx in range(n_contexts):
            print(f"    context {ctx_idx + 1}/{n_contexts}:")
            try:
                sr = run_scenario(
                    endpoint, model, scenario, ctx_idx,
                    target_tokens, runs, rng,
                )
                results_for_scenario.append(sr)
            except Exception as exc:
                print(f"    ERROR (context {ctx_idx}): {exc}")
            time.sleep(1.0)

        scenarios_results[scenario.value] = results_for_scenario

    return scenarios_results


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def aggregate_scenario(results: list[ScenarioResult]) -> dict:
    """Compute aggregate stats across all context runs for a scenario."""
    all_seed = [r.seed_ttft_ms for r in results if r.seed_ttft_ms > 0]
    all_variant = []
    for r in results:
        all_variant.extend(r.variant_ttfts_ms)

    seed_p50 = sorted(all_seed)[len(all_seed) // 2] if all_seed else 0
    variant_sorted = sorted(all_variant)
    variant_p50 = variant_sorted[len(variant_sorted) // 2] if variant_sorted else 0
    variant_mean_val = statistics.mean(all_variant) if all_variant else 0
    variant_stdev_val = statistics.stdev(all_variant) if len(all_variant) > 1 else 0

    speedup = seed_p50 / variant_p50 if variant_p50 > 0 and seed_p50 > 0 else None
    total_errors = sum(len(r.errors) for r in results)

    return {
        "n_seeds": len(all_seed),
        "n_variants": len(all_variant),
        "n_errors": total_errors,
        "seed_p50_ms": round(seed_p50, 1),
        "seed_mean_ms": round(statistics.mean(all_seed), 1) if all_seed else 0,
        "variant_p50_ms": round(variant_p50, 1),
        "variant_mean_ms": round(variant_mean_val, 1),
        "variant_stdev_ms": round(variant_stdev_val, 1),
        "variant_min_ms": round(min(all_variant), 1) if all_variant else 0,
        "variant_max_ms": round(max(all_variant), 1) if all_variant else 0,
        "speedup_p50": round(speedup, 3) if speedup else None,
        "raw_seed_ms": [round(v, 1) for v in all_seed],
        "raw_variant_ms": [round(v, 1) for v in all_variant],
    }


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(
    all_results: dict[int, dict[str, list[ScenarioResult]]],
) -> dict:
    """Print the final summary table and return aggregated data."""
    aggregated = {}

    print(f"\n{'=' * 90}")
    print("vLLM PREFIX CACHE BASELINE — SUMMARY")
    print(f"{'=' * 90}")
    print(
        f"{'Tokens':>8}  {'Scenario':<16}  {'Seed p50':>10}  "
        f"{'Variant p50':>12}  {'Speedup':>8}  {'n':>4}  {'Prefix Hit?':<12}"
    )
    print("-" * 90)

    for token_len in sorted(all_results.keys()):
        scenarios = all_results[token_len]
        aggregated[token_len] = {}
        for scenario_name in [s.value for s in Scenario]:
            results = scenarios.get(scenario_name, [])
            if not results:
                continue
            agg = aggregate_scenario(results)
            aggregated[token_len][scenario_name] = agg

            speedup_str = f"{agg['speedup_p50']:.2f}x" if agg["speedup_p50"] else "---"

            # Determine expected prefix cache behavior
            if scenario_name == Scenario.EXACT_PREFIX.value:
                hit_label = "YES (exact)"
            else:
                hit_label = "NO"

            print(
                f"{token_len:>8}  {scenario_name:<16}  "
                f"{agg['seed_p50_ms']:>9.0f}ms  "
                f"{agg['variant_p50_ms']:>11.0f}ms  "
                f"{speedup_str:>8}  "
                f"{agg['n_variants']:>4}  "
                f"{hit_label:<12}"
            )
        print("-" * 90)

    # Paper-ready comparison table
    print(f"\n{'=' * 75}")
    print("PAPER TABLE: Prefix Cache Coverage Gap")
    print("(Scenarios where prefix cache CANNOT help but SemBlend CAN)")
    print(f"{'=' * 75}")
    print(f"{'Tokens':>8}  {'EXACT (hit)':>12}  {'REORDER':>10}  "
          f"{'PARTIAL75':>10}  {'PARTIAL50':>10}  {'PARAPHRASE':>10}")
    print("-" * 75)

    for token_len in sorted(aggregated.keys()):
        vals = aggregated[token_len]

        def _get_p50(name: str) -> str:
            agg = vals.get(name)
            if agg and agg["variant_p50_ms"] > 0:
                return f"{agg['variant_p50_ms']:.0f}ms"
            return "---"

        def _get_speedup(name: str) -> str:
            agg = vals.get(name)
            if agg and agg["speedup_p50"]:
                return f"({agg['speedup_p50']:.1f}x)"
            return ""

        exact = _get_p50(Scenario.EXACT_PREFIX.value)
        exact_s = _get_speedup(Scenario.EXACT_PREFIX.value)
        reorder = _get_p50(Scenario.REORDER.value)
        reorder_s = _get_speedup(Scenario.REORDER.value)
        p75 = _get_p50(Scenario.PARTIAL_75.value)
        p75_s = _get_speedup(Scenario.PARTIAL_75.value)
        p50 = _get_p50(Scenario.PARTIAL_50.value)
        p50_s = _get_speedup(Scenario.PARTIAL_50.value)
        para = _get_p50(Scenario.PARAPHRASE.value)
        para_s = _get_speedup(Scenario.PARAPHRASE.value)

        print(
            f"{token_len:>8}  {exact:>7}{exact_s:>5}  {reorder:>6}{reorder_s:>4}  "
            f"{p75:>6}{p75_s:>4}  {p50:>6}{p50_s:>4}  {para:>6}{para_s:>4}"
        )

    print(f"\nExpected: EXACT_PREFIX shows speedup (prefix cache hit).")
    print(f"          All others show ~1.0x (prefix cache miss = cold prefill).")
    print(f"          SemBlend should show speedup for ALL scenarios.")
    print(f"{'=' * 75}")

    return aggregated


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_clusters(clusters_file: str) -> list[dict]:
    """Load cluster data from a JSON file produced by build_clusters.py."""
    with open(clusters_file) as f:
        return json.load(f)


# Mapping from cluster overlap_type to Scenario
_OVERLAP_TO_SCENARIO = {
    "exact": Scenario.EXACT_PREFIX,
    "reorder": Scenario.REORDER,
    "partial_80": Scenario.PARTIAL_75,
    "partial_60": Scenario.PARTIAL_50,
    "paraphrase": Scenario.PARAPHRASE,
}


def run_length_suite_clusters(
    endpoint: str,
    model: str,
    clusters_for_length: list[dict],
    target_tokens: int,
    runs: int,
) -> dict[str, list[ScenarioResult]]:
    """Run all scenarios for a single token length using cluster data."""
    scenarios_results: dict[str, list[ScenarioResult]] = {}

    # Group variations across clusters by scenario
    # For each scenario, use multiple clusters as different "contexts"
    n_contexts = min(4, len(clusters_for_length))

    for scenario in Scenario:
        print(f"\n  --- {scenario.value.upper()} ---")
        results_for_scenario = []

        for ctx_idx in range(n_contexts):
            cluster = clusters_for_length[ctx_idx]
            print(f"    context {ctx_idx + 1}/{n_contexts}:")

            # Find matching variation for this scenario
            target_overlap = None
            for overlap_type, sc in _OVERLAP_TO_SCENARIO.items():
                if sc == scenario:
                    target_overlap = overlap_type
                    break

            if target_overlap is None:
                print(f"      no mapping for {scenario.value}")
                continue

            # Find variation with matching overlap_type
            variation = None
            for v in cluster.get("variations", []):
                if v["overlap_type"] == target_overlap:
                    variation = v
                    break

            if variation is None:
                print(f"      no {target_overlap} variation in cluster")
                continue

            try:
                # Seed: measure TTFT for the seed_text
                seed_prompt = cluster["seed_text"]
                seed_ttft = measure_ttft(endpoint, model, seed_prompt)
                time.sleep(0.5)

                result = ScenarioResult(
                    scenario=scenario.value,
                    target_tokens=target_tokens,
                    seed_ttft_ms=seed_ttft,
                )

                # Variant measurements using the cluster variation
                variant_prompt = variation["text"]
                for run_idx in range(runs):
                    try:
                        ttft = measure_ttft(endpoint, model, variant_prompt)
                        result.variant_ttfts_ms.append(ttft)
                        speedup = seed_ttft / ttft if ttft > 0 else 0
                        print(f"      run {run_idx + 1}/{runs}: {ttft:.0f}ms "
                              f"(vs seed {seed_ttft:.0f}ms, {speedup:.2f}x)")
                    except Exception as exc:
                        result.errors.append(f"run {run_idx + 1}: {str(exc)[:200]}")
                        print(f"      run {run_idx + 1}/{runs}: ERROR {exc}")
                    time.sleep(0.3)

                results_for_scenario.append(result)
            except Exception as exc:
                print(f"    ERROR (context {ctx_idx}): {exc}")
            time.sleep(1.0)

        scenarios_results[scenario.value] = results_for_scenario

    return scenarios_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="vLLM native prefix cache baseline benchmark for SemBlend paper"
    )
    parser.add_argument(
        "--endpoint", default="http://localhost:8001",
        help="vLLM endpoint URL (default: http://localhost:8001)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ",
        help="Model name served by vLLM",
    )
    parser.add_argument(
        "--token-lengths", default="2048,5120,8192,16000",
        help="Comma-separated target prompt token lengths",
    )
    parser.add_argument(
        "--runs", type=int, default=10,
        help="Number of variant measurements per context per scenario",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/prefix_cache_baseline",
        help="Directory for JSON output files",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--clusters-file", default=None,
        help="Path to cluster JSON file from build_clusters.py. "
             "When provided, uses real-dataset prompts instead of hardcoded CONTEXTS.",
    )
    parser.add_argument(
        "--dataset-name", default=None,
        help="Optional dataset label for output metadata.",
    )
    args = parser.parse_args()

    lengths = [int(x) for x in args.token_lengths.split(",")]
    rng = random.Random(args.seed)

    print("vLLM Native Prefix Cache Baseline Benchmark")
    print(f"  Endpoint:      {args.endpoint}")
    print(f"  Model:         {args.model}")
    print(f"  Token lengths: {lengths}")
    print(f"  Runs/context:  {args.runs}")
    print(f"  Random seed:   {args.seed}")
    print(f"  Scenarios:     {[s.value for s in Scenario]}")

    # Health check
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        if resp.status_code != 200:
            print(f"\nERROR: vLLM health check failed (HTTP {resp.status_code})")
            sys.exit(1)
        print(f"  vLLM healthy.\n")
    except Exception as exc:
        print(f"\nERROR: Cannot reach vLLM at {args.endpoint}: {exc}")
        sys.exit(1)

    # Warmup: single short request to initialize model
    print("Warmup...")
    try:
        measure_ttft(args.endpoint, args.model, "Hello, this is a warmup request.")
    except Exception as exc:
        print(f"  Warmup failed (non-fatal): {exc}")
    time.sleep(1)

    cluster_mode = args.clusters_file is not None

    # Load clusters if provided
    clusters_by_length: dict[int, list[dict]] = {}
    if cluster_mode:
        all_clusters = _load_clusters(args.clusters_file)
        for c in all_clusters:
            tl = c["target_token_length"]
            clusters_by_length.setdefault(tl, []).append(c)
        available_lengths = sorted(clusters_by_length.keys())
        if lengths:
            lengths = [tl for tl in lengths if tl in clusters_by_length]
        if not lengths:
            lengths = available_lengths
        print(f"  Cluster file: {args.clusters_file}")
        print(f"  Cluster lengths: {available_lengths}")
        print(f"  Using lengths: {lengths}")
    if args.dataset_name:
        print(f"  Dataset: {args.dataset_name}")

    # Run all lengths
    all_results: dict[int, dict[str, list[ScenarioResult]]] = {}

    for token_len in lengths:
        print(f"\n{'=' * 70}")
        print(f"  TOKEN LENGTH: {token_len}")
        print(f"{'=' * 70}")

        if cluster_mode:
            suite = run_length_suite_clusters(
                args.endpoint, args.model,
                clusters_by_length.get(token_len, []),
                token_len, args.runs,
            )
        else:
            suite = run_length_suite(
                args.endpoint, args.model, token_len, args.runs, rng,
            )
        all_results[token_len] = suite

    # Summary
    aggregated = print_summary_table(all_results)

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    run_id = f"prefix-cache-baseline-{timestamp}"

    # Serialize ScenarioResult objects
    serializable_results = {}
    for token_len, scenarios in all_results.items():
        serializable_results[str(token_len)] = {}
        for scenario_name, results_list in scenarios.items():
            serializable_results[str(token_len)][scenario_name] = [
                {
                    "scenario": r.scenario,
                    "target_tokens": r.target_tokens,
                    "seed_ttft_ms": round(r.seed_ttft_ms, 1),
                    "variant_ttfts_ms": [round(v, 1) for v in r.variant_ttfts_ms],
                    "variant_p50_ms": round(r.variant_p50, 1) if r.variant_p50 else None,
                    "variant_mean_ms": round(r.variant_mean, 1) if r.variant_mean else None,
                    "speedup": round(r.speedup, 3) if r.speedup else None,
                    "errors": r.errors,
                }
                for r in results_list
            ]

    output = {
        "run_id": run_id,
        "timestamp": timestamp,
        "config": {
            "endpoint": args.endpoint,
            "model": args.model,
            "token_lengths": lengths,
            "runs_per_context": args.runs,
            "contexts_per_scenario": min(4, len(CONTEXTS)) if not cluster_mode else min(4, max((len(v) for v in clusters_by_length.values()), default=0)),
            "random_seed": args.seed,
            "cluster_mode": cluster_mode,
        },
        "aggregated": {
            str(k): v for k, v in aggregated.items()
        },
        "raw_results": serializable_results,
    }
    if args.dataset_name:
        output["dataset_name"] = args.dataset_name

    out_file = os.path.join(args.output_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {out_file}")


if __name__ == "__main__":
    main()
