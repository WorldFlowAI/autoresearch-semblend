#!/usr/bin/env python3
"""Comprehensive SemBlend benchmark for paper evaluation.

Combines TTFT scaling, quality measurement, and scale testing in a single
reproducible run. Designed to generate all data needed for the paper:

1. TTFT scaling at 1K, 2K, 4K, 8K, 16K tokens
2. Quality: ROUGE-L, exact match, PPL ratio via logprobs
3. Scale: 100 donors vs 1000 donors using real dataset clusters
4. Comparison targets for SemShareKV (arxiv 2509.24832)

Methodology:
  - Pod restart before each scale tier (clean donor store + LMCache)
  - Per-length register+measure (hybrid) to keep KV fresh in LMCache
  - Streaming TTFT measurement (first SSE token)
  - Quality via /v1/completions with logprobs=1

Usage:
    python -m benchmarks.e2e.semblend_comprehensive_bench \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --output results/comprehensive.json
"""
from __future__ import annotations

import hashlib
import json
import math
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
    This replaces the old chars-based approximation which was ~6.5x off
    (Qwen's BPE tokenizes ~6.5 chars per token, not 4).
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
    base_tokens = len(tokenizer.encode(base_prompt))

    if base_tokens >= target_tokens:
        return base_prompt

    # Build extended context, then trim to exact token count
    pad_needed_tokens = target_tokens - base_tokens
    # Over-generate padding text (each context is ~90 tokens)
    repeated = context
    while len(tokenizer.encode(repeated)) < pad_needed_tokens + 200:
        repeated = repeated + " " + context

    # Binary search for exact padding length
    pad_text = repeated
    pad_ids = tokenizer.encode(pad_text)
    if len(pad_ids) > pad_needed_tokens:
        # Decode exactly pad_needed_tokens tokens worth of text
        pad_text = tokenizer.decode(pad_ids[:pad_needed_tokens])

    full_prompt = prefix + f"Extended context:\n{pad_text}\n" + suffix
    actual_tokens = len(tokenizer.encode(full_prompt))

    # Fine-tune: if slightly off, adjust
    if actual_tokens > target_tokens + 10:
        excess = actual_tokens - target_tokens
        full_ids = tokenizer.encode(full_prompt)
        # Remove excess tokens from the middle of the padding
        trimmed = tokenizer.decode(full_ids[:target_tokens])
        return trimmed
    return full_prompt


def api_call(
    endpoint: str, model: str, prompt: str,
    max_tokens: int = 5, stream: bool = False,
    timeout: float = 300.0, retries: int = 3,
    logprobs: int | None = None,
) -> requests.Response:
    body = {
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "stream": stream,
    }
    if logprobs is not None:
        body["logprobs"] = logprobs
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json=body,
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


def measure_quality(
    endpoint: str, model: str, prompt: str,
    max_tokens: int = 50,
) -> tuple[str, float | None]:
    """Generate output with logprobs for quality measurement."""
    resp = api_call(
        endpoint, model, prompt,
        max_tokens=max_tokens, stream=False, logprobs=1,
    )
    data = resp.json()
    choices = data.get("choices", [{}])
    text = choices[0].get("text", "")
    logprobs_data = choices[0].get("logprobs", {})
    token_logprobs = logprobs_data.get("token_logprobs", []) if logprobs_data else []

    avg_logprob = None
    if token_logprobs:
        valid = [lp for lp in token_logprobs if lp is not None]
        if valid:
            avg_logprob = statistics.mean(valid)

    return text, avg_logprob


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    hyp_words = hypothesis.split()
    ref_words = reference.split()
    if not hyp_words or not ref_words:
        return 0.0
    m, n = len(ref_words), len(hyp_words)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_words[i - 1] == hyp_words[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs_len = dp[m][n]
    precision = lcs_len / n if n > 0 else 0
    recall = lcs_len / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    api_call(endpoint, model, prompt, max_tokens=1, stream=False)


def run_comprehensive_bench(
    endpoint: str,
    model: str,
    token_lengths: list[int],
    runs_per_length: int = 12,
    donors_per_context: int = 2,
    quality_tokens: int = 50,
    output_path: str | None = None,
) -> dict:
    results = {}

    print(f"\nSemBlend Comprehensive Benchmark")
    print(f"  Model: {model}")
    print(f"  Token lengths: {token_lengths}")
    print(f"  Runs per length: {runs_per_length}")
    print(f"  Donors per context: {donors_per_context}")
    print(f"  Quality gen tokens: {quality_tokens}")
    print()

    # Warmup
    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)
    uid = 10000

    # ================================================================
    # PHASE 1: Cold baselines (empty donor store)
    # ================================================================
    print("=" * 70)
    print("PHASE 1: Cold baselines + quality baselines")
    print("=" * 70)

    cold_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}
    cold_quality: dict[int, list[tuple[str, float | None]]] = {
        tl: [] for tl in token_lengths
    }

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

                # Quality baseline (only for first 6 runs to save time)
                if run_idx < 6:
                    text, logprob = measure_quality(
                        endpoint, model, cold_prompt,
                        max_tokens=quality_tokens,
                    )
                    cold_quality[target_tokens].append((text, logprob))
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
            time.sleep(0.2)

    # ================================================================
    # PHASE 2: Per-length register + measure
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 2: SemBlend measurement (per-length donor registration)")
    print("=" * 70)

    semblend_data: dict[int, list[float]] = {tl: [] for tl in token_lengths}
    semblend_quality: dict[int, list[tuple[str, float | None, int]]] = {
        tl: [] for tl in token_lengths
    }

    for target_tokens in token_lengths:
        target_toks = target_tokens
        cold_sorted = sorted(cold_data[target_tokens]) if cold_data[target_tokens] else [1]
        cold_p50 = cold_sorted[len(cold_sorted) // 2]
        miss_threshold = cold_p50 * 0.70

        print(f"\n--- {target_tokens} tokens (cold_p50={cold_p50:.0f}ms, "
              f"miss_threshold={miss_threshold:.0f}ms) ---")

        # Register donors
        num_donors = runs_per_length * donors_per_context
        print(f"  Registering {num_donors} donors...")
        for run_idx in range(runs_per_length):
            ctx = CONTEXTS[run_idx % len(CONTEXTS)]
            for d in range(donors_per_context):
                donor_uid = uid
                uid += 1
                donor_prompt = build_prompt(ctx, donor_uid, target_toks)
                try:
                    register_donor(endpoint, model, donor_prompt)
                except Exception as e:
                    print(f"    donor FAILED: {e}")
                time.sleep(0.3)

        print(f"  Settling 3s...")
        time.sleep(3)

        # Measure SemBlend
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

                # Quality measurement for hits (first 6)
                if run_idx < 6:
                    text, logprob = measure_quality(
                        endpoint, model, sem_prompt,
                        max_tokens=quality_tokens,
                    )
                    semblend_quality[target_tokens].append(
                        (text, logprob, run_idx)
                    )
            except Exception as e:
                print(f"    [{run_idx+1}] FAILED: {e}")
                total += 1
            time.sleep(0.2)

        if total > 0:
            print(f"  Hit rate: {hits}/{total} ({100*hits/total:.0f}%)")

    # ================================================================
    # FINAL ANALYSIS
    # ================================================================
    print(f"\n{'='*70}")
    print("RESULTS")
    print("=" * 70)

    print(f"\n{'Tokens':>8} {'Cold p50':>10} {'SB p50':>10} {'SB hit p50':>12} "
          f"{'Speedup':>9} {'Hit spd':>9} {'Hit%':>6} {'n':>4}")
    print("-" * 75)

    for tl in token_lengths:
        if not cold_data[tl] or not semblend_data[tl]:
            continue

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

        # Quality analysis
        quality_metrics = _compute_quality_metrics(
            cold_quality.get(tl, []),
            semblend_quality.get(tl, []),
        )

        results[tl] = {
            "cold_p50": cold_p50,
            "cold_mean": statistics.mean(cold_data[tl]),
            "cold_stdev": statistics.stdev(cold_data[tl]) if len(cold_data[tl]) > 1 else 0,
            "cold_raw": cold_data[tl],
            "semblend_p50": sem_p50,
            "semblend_mean": statistics.mean(semblend_data[tl]),
            "semblend_hit_p50": hit_p50,
            "semblend_raw": semblend_data[tl],
            "semblend_hits_raw": hits_only,
            "speedup_p50": speedup,
            "speedup_hit_p50": hit_speedup,
            "hit_rate": hit_rate,
            "n_cold": len(cold_data[tl]),
            "n_sem": len(semblend_data[tl]),
            "n_hits": len(hits_only),
            **quality_metrics,
        }

        print(f"{tl:>8} {cold_p50:>9.0f}ms {sem_p50:>9.0f}ms {hit_p50:>11.0f}ms "
              f"{speedup:>8.2f}x {hit_speedup:>8.2f}x {100*hit_rate:>5.0f}% "
              f"{len(semblend_data[tl]):>4}")

    print("=" * 75)

    # Quality summary
    print(f"\nQUALITY METRICS:")
    print(f"{'Tokens':>8} {'ROUGE-L':>10} {'Exact%':>8} {'PPL Ratio':>10}")
    print("-" * 40)
    for tl in token_lengths:
        if tl not in results:
            continue
        r = results[tl]
        rl = f"{r.get('rouge_l', 0):.3f}" if r.get('rouge_l') else "---"
        em = f"{r.get('exact_match_rate', 0):.0%}" if 'exact_match_rate' in r else "---"
        ppl = f"{r.get('ppl_ratio', 0):.3f}" if r.get('ppl_ratio') else "---"
        print(f"{tl:>8} {rl:>10} {em:>8} {ppl:>10}")

    # SemShareKV comparison
    print(f"\nSemShareKV COMPARISON:")
    print(f"  SemShareKV: 6.25x peak at 5K on A100")
    print(f"  SemBlend results (A10G, cheaper hardware):")
    for tl in sorted(results.keys()):
        r = results[tl]
        marker = " <<" if r["speedup_hit_p50"] >= 6.25 else ""
        print(f"    {tl:>6} tokens: {r['speedup_hit_p50']:.2f}x (hit-only){marker}")

    report = {
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "A10G (g5.xlarge)",
        "methodology": "hybrid: cold-first, per-length register+measure",
        "donors_per_context": donors_per_context,
        "runs_per_length": runs_per_length,
        "quality_tokens": quality_tokens,
        "results": {str(k): v for k, v in results.items()},
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {output_path}")

    return report


def _compute_quality_metrics(
    cold_quality: list[tuple[str, float | None]],
    semblend_quality: list[tuple[str, float | None, int]],
) -> dict:
    """Compute ROUGE-L, exact match, and PPL ratio."""
    if not cold_quality or not semblend_quality:
        return {}

    rouge_ls = []
    exact_matches = 0
    ppl_ratios = []
    total_pairs = 0

    for i, (sb_text, sb_logprob, run_idx) in enumerate(semblend_quality):
        cold_idx = run_idx % len(cold_quality)
        cold_text, cold_logprob = cold_quality[cold_idx]
        total_pairs += 1

        # ROUGE-L
        rl = compute_rouge_l(sb_text, cold_text)
        rouge_ls.append(rl)

        # Exact match
        if sb_text.strip() == cold_text.strip():
            exact_matches += 1

        # PPL ratio
        if (cold_logprob is not None and sb_logprob is not None
                and cold_logprob < 0 and sb_logprob < 0):
            ppl_cold = math.exp(-cold_logprob)
            ppl_sb = math.exp(-sb_logprob)
            if ppl_cold > 0:
                ppl_ratios.append(ppl_sb / ppl_cold)

    result = {}
    if rouge_ls:
        result["rouge_l"] = statistics.mean(rouge_ls)
        result["rouge_l_raw"] = rouge_ls
    if total_pairs > 0:
        result["exact_match_rate"] = exact_matches / total_pairs
    if ppl_ratios:
        result["ppl_ratio"] = statistics.mean(ppl_ratios)
        result["ppl_ratio_raw"] = ppl_ratios

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--token-lengths", default="1024,2048,4096,8192,16000")
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--donors-per-context", type=int, default=2)
    parser.add_argument("--quality-tokens", type=int, default=50)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.token_lengths.split(",")]
    run_comprehensive_bench(
        endpoint=args.endpoint,
        model=args.model,
        token_lengths=lengths,
        runs_per_length=args.runs,
        donors_per_context=args.donors_per_context,
        quality_tokens=args.quality_tokens,
        output_path=args.output,
    )
