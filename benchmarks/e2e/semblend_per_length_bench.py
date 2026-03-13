#!/usr/bin/env python3
"""Per-length SemBlend benchmark with pod restarts for clean measurements.

Each token length gets a fresh pod (clean donor store + LMCache) to
eliminate cross-length cold baseline contamination. This produces the
most rigorous TTFT numbers for the paper.

Usage:
    python -m benchmarks.e2e.semblend_per_length_bench \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --token-lengths "2048,5120,8192,16000" \
        --runs 10 \
        --output results/per_length_v10.json

Note: requires kubectl access to restart the vLLM pod between lengths.
Each length takes ~3-7 minutes (pod restart + cold + donor reg + measure).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import subprocess
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

    pad_needed_tokens = target_tokens - base_tok_count
    repeated = context
    while len(tokenizer.encode(repeated)) < pad_needed_tokens + 200:
        repeated = repeated + " " + context

    pad_ids = tokenizer.encode(repeated)
    if len(pad_ids) > pad_needed_tokens:
        repeated = tokenizer.decode(pad_ids[:pad_needed_tokens])

    full_prompt = prefix + f"Extended context:\n{repeated}\n" + suffix
    actual_tokens = len(tokenizer.encode(full_prompt))

    if actual_tokens > target_tokens + 10:
        full_ids = tokenizer.encode(full_prompt)
        return tokenizer.decode(full_ids[:target_tokens])
    return full_prompt


def measure_streaming_ttft(endpoint: str, model: str, prompt: str) -> float:
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": 5, "temperature": 0.0,
            "stream": True,
        },
        timeout=300.0, stream=True,
    )
    resp.raise_for_status()
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
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": 1, "temperature": 0.0,
        },
        timeout=300.0,
    )
    resp.raise_for_status()


def measure_quality(
    endpoint: str, model: str, prompt: str, max_tokens: int = 50,
) -> tuple[str, float | None]:
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0,
            "logprobs": 1,
        },
        timeout=300.0,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_data = data["choices"][0].get("logprobs", {})
    token_lps = lp_data.get("token_logprobs", []) if lp_data else []
    valid = [lp for lp in token_lps if lp is not None]
    avg_lp = statistics.mean(valid) if valid else None
    return text, avg_lp


def compute_rouge_l(hypothesis: str, reference: str) -> float:
    if not hypothesis or not reference:
        return 0.0
    hyp = hypothesis.split()
    ref = reference.split()
    if not hyp or not ref:
        return 0.0
    m, n = len(ref), len(hyp)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref[i-1] == hyp[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p = lcs / n if n > 0 else 0
    r = lcs / m if m > 0 else 0
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def restart_pod_and_portforward(
    namespace: str = "synapse-staging",
    port: int = 8001,
) -> None:
    """Restart vLLM pod and re-establish port-forward."""
    print("  Restarting vLLM pod...")
    subprocess.run(
        ["kubectl", "rollout", "restart",
         "deployment/synapse-staging-vllm", "-n", namespace],
        check=True, capture_output=True,
    )

    # Wait for rollout to complete (handles old pod termination)
    print("  Waiting for rollout...")
    subprocess.run(
        ["kubectl", "rollout", "status",
         "deployment/synapse-staging-vllm", "-n", namespace,
         "--timeout=300s"],
        capture_output=True, timeout=310,
    )

    # Extra wait for model loading
    time.sleep(10)

    # Wait for pod ready
    print("  Waiting for pod ready...")
    for attempt in range(60):
        result = subprocess.run(
            ["kubectl", "get", "pods", "-n", namespace,
             "-l", "app=vllm", "-o",
             "jsonpath={.items[0].status.conditions[?(@.type=='Ready')].status}"],
            capture_output=True, text=True,
        )
        if result.stdout.strip() == "True":
            break
        time.sleep(5)

    # Kill and restart port-forward
    subprocess.run(
        f"lsof -i :{port} -t 2>/dev/null | xargs kill -9 2>/dev/null",
        shell=True,
    )
    time.sleep(2)
    subprocess.Popen(
        ["kubectl", "port-forward", "-n", namespace,
         "svc/synapse-staging-vllm", f"{port}:8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )

    # Wait for health endpoint
    print("  Waiting for endpoint health...")
    for attempt in range(60):
        try:
            resp = requests.get(f"http://localhost:{port}/health", timeout=5)
            if resp.status_code == 200:
                print("  Pod ready.")
                return
        except Exception:
            pass
        time.sleep(3)
    raise RuntimeError("Pod failed to become healthy")


def run_single_length(
    endpoint: str,
    model: str,
    target_tokens: int,
    runs: int = 10,
    donors_per_context: int = 2,
    quality_tokens: int = 50,
) -> dict:
    """Run benchmark for a single token length on a clean pod."""
    uid = 10000
    cold_ttfts = []
    sb_ttfts = []
    quality_data = []  # [(cold_text, cold_lp, sb_text, sb_lp)]

    # Warmup
    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)

    # === COLD BASELINES ===
    print(f"    Cold baselines ({runs} runs):")
    for i in range(runs):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        prompt = build_prompt(ctx, uid, target_tokens, model)
        uid += 1
        try:
            ttft = measure_streaming_ttft(endpoint, model, prompt)
            cold_ttfts.append(ttft)
            print(f"      [{i+1}] {ttft:.0f}ms")

            # Quality baseline (first 6)
            if i < 6:
                text, lp = measure_quality(
                    endpoint, model, prompt, max_tokens=quality_tokens,
                )
                quality_data.append({"cold_text": text, "cold_lp": lp, "ctx_idx": i})
        except Exception as e:
            print(f"      [{i+1}] FAILED: {e}")
        time.sleep(0.2)

    cold_sorted = sorted(cold_ttfts)
    cold_p50 = cold_sorted[len(cold_sorted) // 2] if cold_sorted else 0
    miss_threshold = cold_p50 * 0.70

    # === DONOR REGISTRATION ===
    num_donors = runs * donors_per_context
    print(f"    Registering {num_donors} donors...")
    for i in range(runs):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        for d in range(donors_per_context):
            donor_prompt = build_prompt(ctx, uid, target_tokens, model)
            uid += 1
            try:
                register_donor(endpoint, model, donor_prompt)
            except Exception as e:
                print(f"      donor FAILED: {e}")
            time.sleep(0.3)

    print("    Settling 3s...")
    time.sleep(3)

    # === SEMBLEND MEASUREMENT ===
    hits = 0
    print(f"    SemBlend measurement ({runs} runs):")
    for i in range(runs):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        prompt = build_prompt(ctx, uid, target_tokens, model)
        uid += 1
        try:
            ttft = measure_streaming_ttft(endpoint, model, prompt)
            sb_ttfts.append(ttft)
            is_hit = ttft < miss_threshold
            if is_hit:
                hits += 1
            speedup = cold_p50 / ttft if ttft > 0 else 0
            marker = "HIT" if is_hit else "MISS"
            print(f"      [{i+1}] {ttft:.0f}ms  {speedup:.2f}x  [{marker}]")

            # Quality (first 6)
            if i < 6 and i < len(quality_data):
                text, lp = measure_quality(
                    endpoint, model, prompt, max_tokens=quality_tokens,
                )
                quality_data[i]["sb_text"] = text
                quality_data[i]["sb_lp"] = lp
        except Exception as e:
            print(f"      [{i+1}] FAILED: {e}")
        time.sleep(0.2)

    # Compute results
    sb_sorted = sorted(sb_ttfts)
    sb_p50 = sb_sorted[len(sb_sorted) // 2] if sb_sorted else 0
    hits_only = sorted([t for t in sb_ttfts if t < miss_threshold])
    hit_p50 = hits_only[len(hits_only) // 2] if hits_only else sb_p50
    hit_rate = len(hits_only) / len(sb_ttfts) if sb_ttfts else 0

    speedup = cold_p50 / sb_p50 if sb_p50 > 0 else 0
    hit_speedup = cold_p50 / hit_p50 if hit_p50 > 0 else 0

    # Quality
    rouge_ls = []
    exact_matches = 0
    ppl_ratios = []
    for qd in quality_data:
        if "sb_text" not in qd:
            continue
        rl = compute_rouge_l(qd["sb_text"], qd["cold_text"])
        rouge_ls.append(rl)
        if qd["sb_text"].strip() == qd["cold_text"].strip():
            exact_matches += 1
        if (qd["cold_lp"] is not None and qd["sb_lp"] is not None
                and qd["cold_lp"] < 0 and qd["sb_lp"] < 0):
            ppl_c = math.exp(-qd["cold_lp"])
            ppl_s = math.exp(-qd["sb_lp"])
            if ppl_c > 0:
                ppl_ratios.append(ppl_s / ppl_c)

    n_quality = len([qd for qd in quality_data if "sb_text" in qd])

    return {
        "target_tokens": target_tokens,
        "cold_p50": cold_p50,
        "cold_mean": statistics.mean(cold_ttfts) if cold_ttfts else 0,
        "cold_stdev": statistics.stdev(cold_ttfts) if len(cold_ttfts) > 1 else 0,
        "cold_raw": cold_ttfts,
        "semblend_p50": sb_p50,
        "semblend_hit_p50": hit_p50,
        "semblend_raw": sb_ttfts,
        "semblend_hits_raw": hits_only,
        "speedup_p50": speedup,
        "speedup_hit_p50": hit_speedup,
        "hit_rate": hit_rate,
        "n_cold": len(cold_ttfts),
        "n_sem": len(sb_ttfts),
        "n_hits": len(hits_only),
        "rouge_l": statistics.mean(rouge_ls) if rouge_ls else None,
        "exact_match_rate": exact_matches / n_quality if n_quality > 0 else None,
        "ppl_ratio": statistics.mean(ppl_ratios) if ppl_ratios else None,
        "rouge_l_raw": rouge_ls,
        "ppl_ratio_raw": ppl_ratios,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--token-lengths", default="2048,5120,8192,16000")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--donors-per-context", type=int, default=2)
    parser.add_argument("--quality-tokens", type=int, default=50)
    parser.add_argument("--output", default=None)
    parser.add_argument("--no-restart", action="store_true",
                        help="Skip pod restarts (for debugging)")
    args = parser.parse_args()

    lengths = [int(x) for x in args.token_lengths.split(",")]

    print(f"SemBlend Per-Length Benchmark (pod restart per length)")
    print(f"  Model: {args.model}")
    print(f"  Token lengths: {lengths}")
    print(f"  Runs: {args.runs}")
    print(f"  Donors per context: {args.donors_per_context}")
    print()

    all_results = {}

    for target_tokens in lengths:
        print(f"\n{'='*60}")
        print(f"  {target_tokens} TOKENS")
        print(f"{'='*60}")

        if not args.no_restart:
            restart_pod_and_portforward()

        result = run_single_length(
            endpoint=args.endpoint,
            model=args.model,
            target_tokens=target_tokens,
            runs=args.runs,
            donors_per_context=args.donors_per_context,
            quality_tokens=args.quality_tokens,
        )
        all_results[target_tokens] = result

    # Summary
    print(f"\n{'='*75}")
    print("FINAL RESULTS (per-length, clean pod)")
    print(f"{'='*75}")
    print(f"{'Tokens':>8} {'Cold p50':>10} {'SB p50':>10} {'SB hit p50':>12} "
          f"{'Speedup':>9} {'Hit spd':>9} {'Hit%':>6}")
    print("-" * 75)

    for tl in sorted(all_results.keys()):
        r = all_results[tl]
        print(f"{tl:>8} {r['cold_p50']:>9.0f}ms {r['semblend_p50']:>9.0f}ms "
              f"{r['semblend_hit_p50']:>11.0f}ms "
              f"{r['speedup_p50']:>8.2f}x {r['speedup_hit_p50']:>8.2f}x "
              f"{100*r['hit_rate']:>5.0f}%")

    print(f"\nQUALITY:")
    print(f"{'Tokens':>8} {'ROUGE-L':>10} {'Exact%':>8} {'PPL Ratio':>10}")
    print("-" * 40)
    for tl in sorted(all_results.keys()):
        r = all_results[tl]
        rl = f"{r['rouge_l']:.3f}" if r['rouge_l'] is not None else "---"
        em = f"{r['exact_match_rate']:.0%}" if r['exact_match_rate'] is not None else "---"
        ppl = f"{r['ppl_ratio']:.3f}" if r['ppl_ratio'] is not None else "---"
        print(f"{tl:>8} {rl:>10} {em:>8} {ppl:>10}")

    print(f"\nSemShareKV COMPARISON (6.25x at 5K on A100):")
    for tl in sorted(all_results.keys()):
        r = all_results[tl]
        marker = " << EXCEEDS" if r['speedup_hit_p50'] >= 6.25 else ""
        print(f"  {tl:>6} tokens: {r['speedup_hit_p50']:.2f}x{marker}")

    report = {
        "model": args.model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "hardware": "A10G (g5.xlarge)",
        "methodology": "per-length pod restart, clean cold baselines",
        "donors_per_context": args.donors_per_context,
        "runs_per_length": args.runs,
        "results": {str(k): v for k, v in all_results.items()},
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
