#!/usr/bin/env python3
"""Focused TTFT scaling benchmark for SemBlend paper.

Hybrid design:
  Phase 1: ALL cold baselines measured first (empty donor store)
  Phase 2: Per-length register+measure (donors for one length, then measure,
           then next length) to keep KV fresh in LMCache

Reports both raw p50 and hit-only p50.
"""
from __future__ import annotations

import hashlib
import json
import statistics
import sys
import time

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


CONTEXTS = [
    (
        "Machine learning has transformed natural language processing. "
        "Transformer architectures replaced recurrent neural networks with "
        "self-attention mechanisms allowing massive parallelization. Models "
        "grew from 110M parameters to hundreds of billions. Efficient "
        "inference became critical with techniques like quantization, KV "
        "cache optimization, continuous batching, and speculative decoding. "
        "Flash Attention reduced memory complexity. Rotary Position "
        "Embeddings encode positional information as rotations."
    ),
    (
        "Kubernetes orchestrates containerized applications across clusters. "
        "The control plane consists of API server, etcd, scheduler, and "
        "controller manager. Worker nodes run kubelet and container runtime. "
        "Pods are smallest deployable units. Deployments manage ReplicaSets "
        "for rolling updates. Services provide stable networking. GPU "
        "scheduling uses device plugins. Helm packages applications as "
        "charts with templated YAML."
    ),
    (
        "Distributed caching architectures form the backbone of high "
        "performance systems. Redis provides in-memory key-value storage "
        "with lists, sets, sorted sets, hashes, streams. Redis Cluster "
        "enables horizontal scaling via hash slot partitioning. Vector "
        "similarity search uses HNSW graphs for approximate nearest "
        "neighbor queries. Multi-tier caching stacks L0 GPU cache, L1 "
        "Redis, and L2 vector DB."
    ),
    (
        "Cloud GPU computing economics drive architectural decisions. T4 "
        "instances cost $0.53/hour with 16GB HBM and 320 GB/s bandwidth. "
        "A10G costs $1.01/hour with 24GB and 600 GB/s. KV cache memory "
        "grows linearly with sequence length. For 7B models with 32 layers "
        "each token requires 524KB. GPU memory utilization is typically "
        "set to 85-90 percent. AWQ quantization reduces model size 4x."
    ),
    (
        "Quantum computing leverages quantum mechanical phenomena for "
        "computation. Qubits exist in superposition of states unlike "
        "classical bits. Entanglement creates correlated qubit pairs. "
        "Quantum gates manipulate qubit states through unitary operations. "
        "Error correction codes protect against decoherence. Shor's "
        "algorithm factors integers exponentially faster. Grover's "
        "algorithm provides quadratic speedup for unstructured search."
    ),
    (
        "Database systems provide persistent structured data storage. "
        "Relational databases use SQL for querying normalized tables. "
        "B-tree indexes enable logarithmic lookup time. Write-ahead "
        "logging ensures ACID transactions survive crashes. Columnar "
        "storage optimizes analytical workloads. LSM trees power high "
        "write throughput in key-value stores like RocksDB and LevelDB."
    ),
    (
        "Computer networking follows the OSI seven layer model. Physical "
        "layer handles bit transmission. Data link manages frame delivery. "
        "Network layer routes packets via IP addressing. Transport layer "
        "ensures reliable delivery through TCP or fast delivery via UDP. "
        "HTTP operates at the application layer. TLS encrypts data in "
        "transit using asymmetric key exchange and symmetric encryption."
    ),
    (
        "Operating systems manage hardware resources for applications. "
        "Process scheduling algorithms include round-robin, priority-based, "
        "and completely fair scheduler. Virtual memory maps logical to "
        "physical addresses using page tables. File systems like ext4 "
        "and ZFS organize data on storage devices. System calls provide "
        "the interface between user space and kernel space."
    ),
    (
        "Compiler design involves multiple transformation stages from source "
        "code to machine instructions. Lexical analysis produces tokens from "
        "character streams. Parsing builds abstract syntax trees using context "
        "free grammars. Type checking validates semantic correctness. Code "
        "generation maps intermediate representation to target architecture "
        "instructions with register allocation and instruction scheduling."
    ),
    (
        "Cryptographic hash functions map arbitrary input to fixed-size output "
        "with collision resistance and preimage resistance. SHA-256 produces "
        "256-bit digests used in blockchain proof of work. Merkle trees enable "
        "efficient verification of large datasets. Digital signatures combine "
        "hash functions with asymmetric keys for authentication and integrity. "
        "Zero-knowledge proofs verify statements without revealing data."
    ),
    (
        "Distributed consensus protocols enable fault-tolerant agreement across "
        "unreliable networks. Paxos achieves safety with majority quorums. Raft "
        "simplifies leader election and log replication. Byzantine fault tolerance "
        "handles malicious nodes with 3f+1 replicas. Practical BFT optimizes "
        "for the common case using speculative execution and batching."
    ),
    (
        "Graph databases model relationships as first-class citizens. Property "
        "graphs store key-value pairs on nodes and edges. Cypher query language "
        "uses pattern matching for traversal. Graph algorithms include shortest "
        "path, pagerank, community detection, and centrality measures. Native "
        "graph storage uses index-free adjacency for constant-time traversal."
    ),
]


_TOKENIZER = None

def _get_tokenizer(model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ"):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def build_prompt(
    context: str, uid: int, target_tokens: int,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
) -> str:
    """Build a prompt with exactly target_tokens token count.

    Uses the model tokenizer to verify token count and adjusts padding.
    """
    tokenizer = _get_tokenizer(model_name)
    ref_id = hashlib.md5(f"ref-{uid}".encode()).hexdigest()[:12]
    sys_prompt = f"You are a helpful AI assistant. Reference: {ref_id}."
    prefix = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\n"
    )
    suffix = (
        "\nQuestion: Summarize the key concepts described above."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    base_prompt = prefix + suffix
    base_tok_count = len(tokenizer.encode(base_prompt))

    if base_tok_count >= target_tokens:
        return base_prompt

    # Build extended context, then trim to exact token count
    pad_needed_tokens = target_tokens - base_tok_count
    repeated = context
    while len(tokenizer.encode(repeated)) < pad_needed_tokens + 200:
        repeated = repeated + " " + context

    pad_text = repeated
    pad_ids = tokenizer.encode(pad_text)
    if len(pad_ids) > pad_needed_tokens:
        pad_text = tokenizer.decode(pad_ids[:pad_needed_tokens])

    full_prompt = prefix + f"Extended context:\n{pad_text}\n" + suffix
    actual_tokens = len(tokenizer.encode(full_prompt))

    if actual_tokens > target_tokens + 10:
        full_ids = tokenizer.encode(full_prompt)
        return tokenizer.decode(full_ids[:target_tokens])
    return full_prompt


def api_call(
    endpoint: str, model: str, prompt: str,
    max_tokens: int = 5, stream: bool = False,
    timeout: float = 300.0, retries: int = 3,
) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model, "prompt": prompt,
                    "max_tokens": max_tokens, "temperature": 0.0,
                    "stream": stream,
                },
                timeout=timeout, stream=stream,
            )
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout) as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise


def measure_streaming_ttft(endpoint: str, model: str, prompt: str) -> float:
    t0 = time.monotonic()
    resp = api_call(endpoint, model, prompt, max_tokens=5, stream=True)
    for line in resp.iter_lines():
        if not line:
            continue
        line_str = line.decode("utf-8", errors="replace")
        if line_str.startswith("data: ") and line_str[6:].strip() != "[DONE]":
            ttft = (time.monotonic() - t0) * 1000
            for _ in resp.iter_lines():
                pass
            return ttft
    raise ValueError("No tokens received")


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    api_call(endpoint, model, prompt, max_tokens=1, stream=False)


def _load_clusters(clusters_file: str) -> list[dict]:
    """Load cluster data from a JSON file produced by build_clusters.py."""
    with open(clusters_file) as f:
        return json.load(f)


def _pick_variation(cluster: dict, preference: list[str] | None = None) -> dict | None:
    """Pick a variation from a cluster, preferring certain overlap_types."""
    if preference is None:
        preference = ["reorder", "partial_80", "paraphrase"]
    by_type = {v["overlap_type"]: v for v in cluster.get("variations", [])}
    for pref in preference:
        if pref in by_type:
            return by_type[pref]
    # Fallback: first non-exact variation
    for v in cluster.get("variations", []):
        if v["overlap_type"] != "exact":
            return v
    return None


def run_scaling_bench_clusters(
    endpoint: str,
    model: str,
    clusters: list[dict],
    runs_per_length: int = 8,
    donors_per_context: int = 2,
    output_path: str | None = None,
    dataset_name: str | None = None,
) -> dict:
    """Run TTFT scaling benchmark using pre-built cluster data."""
    # Group clusters by target_token_length
    by_length: dict[int, list[dict]] = {}
    for c in clusters:
        tl = c["target_token_length"]
        by_length.setdefault(tl, []).append(c)

    token_lengths = sorted(by_length.keys())
    results = {}

    print(f"\nSemBlend TTFT Scaling Benchmark (Cluster Mode)")
    print(f"  Model: {model}")
    print(f"  Token lengths: {token_lengths}")
    print(f"  Runs per length: {runs_per_length}")
    print(f"  Donors per context: {donors_per_context}")
    print(f"  Total clusters: {len(clusters)}")
    if dataset_name:
        print(f"  Dataset: {dataset_name}")
    print()

    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)

    # ================================================================
    # PHASE 1: All cold baselines (empty donor store)
    # ================================================================
    print("=" * 70)
    print("PHASE 1: Cold baselines (empty donor store)")
    print("=" * 70)

    cold_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}

    for target_tokens in token_lengths:
        length_clusters = by_length[target_tokens][:runs_per_length]
        print(f"\n  Cold @ {target_tokens} tokens:")
        for run_idx, cluster in enumerate(length_clusters):
            cold_prompt = cluster["seed_text"]
            try:
                ttft = measure_streaming_ttft(endpoint, model, cold_prompt)
                cold_data[target_tokens].append(ttft)
                print(f"    [{run_idx+1}] {ttft:.0f}ms")
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
            time.sleep(0.2)

    print(f"\n{'='*70}")
    print("Cold Summary:")
    for tl in token_lengths:
        if cold_data[tl]:
            p50 = sorted(cold_data[tl])[len(cold_data[tl]) // 2]
            print(f"  {tl:>6} tokens: p50={p50:.0f}ms  (n={len(cold_data[tl])})")
    print(f"{'='*70}")

    # ================================================================
    # PHASE 2: Per-length register + measure (keeps KV fresh)
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Per-length donor registration + SemBlend measurement")
    print("=" * 70)

    semblend_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}

    for target_tokens in token_lengths:
        length_clusters = by_length[target_tokens][:runs_per_length]
        cold_p50 = sorted(cold_data[target_tokens])[len(cold_data[target_tokens]) // 2] if cold_data[target_tokens] else 1
        miss_threshold = cold_p50 * 0.70

        print(f"\n--- {target_tokens} tokens (cold_p50={cold_p50:.0f}ms) ---")

        # Register donors (seed_text) for THIS length
        print(f"  Registering {len(length_clusters) * donors_per_context} donors...")
        for run_idx, cluster in enumerate(length_clusters):
            for d in range(donors_per_context):
                try:
                    register_donor(endpoint, model, cluster["seed_text"])
                except Exception as e:
                    print(f"    [{run_idx+1}] donor {d+1} FAILED: {e}")
                time.sleep(0.3)

        # Wait for all request_finished callbacks
        print(f"  Settling 3s...")
        time.sleep(3)

        # Measure SemBlend using variation text for THIS length
        hits = 0
        total = 0
        for run_idx, cluster in enumerate(length_clusters):
            variation = _pick_variation(cluster)
            if variation is None:
                print(f"    [{run_idx+1}] SKIPPED: no variation available")
                continue
            sem_prompt = variation["text"]
            try:
                ttft = measure_streaming_ttft(endpoint, model, sem_prompt)
                semblend_data[target_tokens].append(ttft)
                total += 1
                is_hit = ttft < miss_threshold
                if is_hit:
                    hits += 1
                speedup = cold_p50 / ttft if ttft > 0 else 0
                marker = "HIT" if is_hit else "MISS"
                print(f"    [{run_idx+1}] {ttft:.0f}ms  {speedup:.2f}x  [{marker}]")
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
                total += 1
            time.sleep(0.2)

        if total > 0:
            print(f"  Hit rate: {hits}/{total} ({100*hits/total:.0f}%)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"{'Tokens':>8} {'Cold p50':>10} {'SB p50':>10} {'SB hit p50':>12} "
          f"{'Speedup':>9} {'Hit spd':>9} {'Hit%':>6} {'n':>4}")
    print("-" * 70)

    for tl in token_lengths:
        if cold_data[tl] and semblend_data[tl]:
            cold_sorted = sorted(cold_data[tl])
            sem_sorted = sorted(semblend_data[tl])
            cold_p50 = cold_sorted[len(cold_sorted) // 2]
            sem_p50 = sem_sorted[len(sem_sorted) // 2]

            miss_threshold = cold_p50 * 0.70
            hits_only = sorted([t for t in semblend_data[tl] if t < miss_threshold])
            hit_p50 = hits_only[len(hits_only) // 2] if hits_only else sem_p50
            hit_rate = len(hits_only) / len(semblend_data[tl])

            speedup = cold_p50 / sem_p50 if sem_p50 > 0 else 0
            hit_speedup = cold_p50 / hit_p50 if hit_p50 > 0 else 0

            result_entry = {
                "cold_p50": cold_p50,
                "cold_mean": statistics.mean(cold_data[tl]),
                "cold_stdev": statistics.stdev(cold_data[tl]) if len(cold_data[tl]) > 1 else 0,
                "semblend_p50": sem_p50,
                "semblend_mean": statistics.mean(semblend_data[tl]),
                "semblend_hit_p50": hit_p50,
                "speedup_p50": speedup,
                "speedup_hit_p50": hit_speedup,
                "hit_rate": hit_rate,
                "cold_raw": cold_data[tl],
                "semblend_raw": semblend_data[tl],
                "semblend_hits": hits_only,
                "n_cold": len(cold_data[tl]),
                "n_sem": len(semblend_data[tl]),
                "n_hits": len(hits_only),
            }
            # Include source_dataset info from clusters
            length_clusters = by_length[tl][:runs_per_length]
            sources = list({c["source_dataset"] for c in length_clusters})
            if sources:
                result_entry["source_dataset"] = sources[0] if len(sources) == 1 else sources

            results[tl] = result_entry
            print(f"{tl:>8} {cold_p50:>9.0f}ms {sem_p50:>9.0f}ms {hit_p50:>11.0f}ms "
                  f"{speedup:>8.2f}x {hit_speedup:>8.2f}x {100*hit_rate:>5.0f}% {len(semblend_data[tl]):>4}")

    print(f"{'='*70}")

    report = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "A10G",
        "methodology": "cluster-based: real-dataset prompts, cold-first then per-length register+measure",
        "donors_per_context": donors_per_context,
        "cluster_mode": True,
        "results": {str(k): v for k, v in results.items()},
    }
    if dataset_name:
        report["dataset_name"] = dataset_name

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {output_path}")

    return report


def run_scaling_bench(
    endpoint: str,
    model: str,
    token_lengths: list[int],
    runs_per_length: int = 8,
    donors_per_context: int = 2,
    output_path: str | None = None,
) -> dict:
    results = {}

    print(f"\nSemBlend TTFT Scaling Benchmark (Hybrid)")
    print(f"  Model: {model}")
    print(f"  Token lengths: {token_lengths}")
    print(f"  Runs per length: {runs_per_length}")
    print(f"  Donors per context: {donors_per_context}")
    print(f"  Contexts available: {len(CONTEXTS)}")
    print()

    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)
    uid = 10000

    # ================================================================
    # PHASE 1: All cold baselines (empty donor store)
    # ================================================================
    print("=" * 70)
    print("PHASE 1: Cold baselines (empty donor store)")
    print("=" * 70)

    cold_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}

    for target_tokens in token_lengths:
        target_toks = target_tokens
        print(f"\n  Cold @ {target_tokens} tokens:")
        for run_idx in range(runs_per_length):
            ctx = CONTEXTS[run_idx % len(CONTEXTS)]
            cold_uid = uid
            uid += 1
            cold_prompt = build_prompt(ctx, cold_uid, target_toks)
            try:
                ttft = measure_streaming_ttft(endpoint, model, cold_prompt)
                cold_data[target_tokens].append(ttft)
                print(f"    [{run_idx+1}] {ttft:.0f}ms")
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
            time.sleep(0.2)

    print(f"\n{'='*70}")
    print("Cold Summary:")
    for tl in token_lengths:
        if cold_data[tl]:
            p50 = sorted(cold_data[tl])[len(cold_data[tl]) // 2]
            print(f"  {tl:>6} tokens: p50={p50:.0f}ms  (n={len(cold_data[tl])})")
    print(f"{'='*70}")

    # ================================================================
    # PHASE 2: Per-length register + measure (keeps KV fresh)
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: Per-length donor registration + SemBlend measurement")
    print("=" * 70)

    semblend_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}

    for target_tokens in token_lengths:
        target_toks = target_tokens
        cold_p50 = sorted(cold_data[target_tokens])[len(cold_data[target_tokens]) // 2] if cold_data[target_tokens] else 1
        miss_threshold = cold_p50 * 0.70

        print(f"\n--- {target_tokens} tokens (cold_p50={cold_p50:.0f}ms) ---")

        # Register donors for THIS length
        print(f"  Registering {runs_per_length * donors_per_context} donors...")
        for run_idx in range(runs_per_length):
            ctx = CONTEXTS[run_idx % len(CONTEXTS)]
            for d in range(donors_per_context):
                donor_uid = uid
                uid += 1
                donor_prompt = build_prompt(ctx, donor_uid, target_toks)
                try:
                    register_donor(endpoint, model, donor_prompt)
                except Exception as e:
                    print(f"    [{run_idx+1}] donor {d+1} FAILED: {e}")
                time.sleep(0.3)

        # Wait for all request_finished callbacks
        print(f"  Settling 3s...")
        time.sleep(3)

        # Measure SemBlend for THIS length
        hits = 0
        total = 0
        for run_idx in range(runs_per_length):
            ctx = CONTEXTS[run_idx % len(CONTEXTS)]
            sem_uid = uid
            uid += 1
            sem_prompt = build_prompt(ctx, sem_uid, target_toks)
            try:
                ttft = measure_streaming_ttft(endpoint, model, sem_prompt)
                semblend_data[target_tokens].append(ttft)
                total += 1
                is_hit = ttft < miss_threshold
                if is_hit:
                    hits += 1
                speedup = cold_p50 / ttft if ttft > 0 else 0
                marker = "HIT" if is_hit else "MISS"
                print(f"    [{run_idx+1}] {ttft:.0f}ms  {speedup:.2f}x  [{marker}]")
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
                total += 1
            time.sleep(0.2)

        if total > 0:
            print(f"  Hit rate: {hits}/{total} ({100*hits/total:.0f}%)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'='*70}")
    print(f"{'Tokens':>8} {'Cold p50':>10} {'SB p50':>10} {'SB hit p50':>12} "
          f"{'Speedup':>9} {'Hit spd':>9} {'Hit%':>6} {'n':>4}")
    print("-" * 70)

    for tl in token_lengths:
        if cold_data[tl] and semblend_data[tl]:
            cold_sorted = sorted(cold_data[tl])
            sem_sorted = sorted(semblend_data[tl])
            cold_p50 = cold_sorted[len(cold_sorted) // 2]
            sem_p50 = sem_sorted[len(sem_sorted) // 2]

            miss_threshold = cold_p50 * 0.70
            hits_only = sorted([t for t in semblend_data[tl] if t < miss_threshold])
            hit_p50 = hits_only[len(hits_only) // 2] if hits_only else sem_p50
            hit_rate = len(hits_only) / len(semblend_data[tl])

            speedup = cold_p50 / sem_p50 if sem_p50 > 0 else 0
            hit_speedup = cold_p50 / hit_p50 if hit_p50 > 0 else 0

            results[tl] = {
                "cold_p50": cold_p50,
                "cold_mean": statistics.mean(cold_data[tl]),
                "cold_stdev": statistics.stdev(cold_data[tl]) if len(cold_data[tl]) > 1 else 0,
                "semblend_p50": sem_p50,
                "semblend_mean": statistics.mean(semblend_data[tl]),
                "semblend_hit_p50": hit_p50,
                "speedup_p50": speedup,
                "speedup_hit_p50": hit_speedup,
                "hit_rate": hit_rate,
                "cold_raw": cold_data[tl],
                "semblend_raw": semblend_data[tl],
                "semblend_hits": hits_only,
                "n_cold": len(cold_data[tl]),
                "n_sem": len(semblend_data[tl]),
                "n_hits": len(hits_only),
            }
            print(f"{tl:>8} {cold_p50:>9.0f}ms {sem_p50:>9.0f}ms {hit_p50:>11.0f}ms "
                  f"{speedup:>8.2f}x {hit_speedup:>8.2f}x {100*hit_rate:>5.0f}% {len(semblend_data[tl]):>4}")

    print(f"{'='*70}")

    report = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "A10G",
        "methodology": "hybrid: cold-first, then per-length register+measure",
        "donors_per_context": donors_per_context,
        "results": {str(k): v for k, v in results.items()},
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {output_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--token-lengths", default="2048,4096,8192,16000")
    parser.add_argument("--runs", type=int, default=8)
    parser.add_argument("--donors-per-context", type=int, default=2)
    parser.add_argument("--output", default=None)
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

    if args.clusters_file:
        clusters = _load_clusters(args.clusters_file)
        run_scaling_bench_clusters(
            endpoint=args.endpoint,
            model=args.model,
            clusters=clusters,
            runs_per_length=args.runs,
            donors_per_context=args.donors_per_context,
            output_path=args.output,
            dataset_name=args.dataset_name,
        )
    else:
        lengths = [int(x) for x in args.token_lengths.split(",")]
        run_scaling_bench(
            endpoint=args.endpoint,
            model=args.model,
            token_lengths=lengths,
            runs_per_length=args.runs,
            donors_per_context=args.donors_per_context,
            output_path=args.output,
        )
