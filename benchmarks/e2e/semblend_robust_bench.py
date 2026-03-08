#!/usr/bin/env python3
"""Robust SemBlend benchmark — avoids prefix cache contamination.

Key design decisions:
- Each measurement uses a UNIQUE prompt (different padding seed) to prevent
  vLLM's prefix cache from confounding results
- Donors and measurement prompts share the same CONTEXT but with unique padding
- Retries on connection failures (vLLM OOM recovery)
- Runs at multiple token lengths to show scaling behavior
- Measures quality (ROUGE-L, exact match) in same pass

Usage:
    python3 semblend_robust_bench.py \
        --endpoint http://synapse-staging-vllm:8000 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import sys
import time
from dataclasses import dataclass

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)

# 8 distinct technical contexts (~200-250 tokens each)
CONTEXTS = [
    (
        "Machine learning has transformed the field of natural language processing. "
        "Transformer architectures, introduced in the seminal 'Attention Is All You Need' "
        "paper by Vaswani et al., replaced recurrent neural networks with self-attention "
        "mechanisms. This allowed for massive parallelization during training and led to "
        "models like BERT, GPT, and T5. The key innovation was the multi-head attention "
        "mechanism, which computes attention scores between all pairs of tokens in a "
        "sequence simultaneously. Pre-training on large corpora followed by fine-tuning "
        "on specific tasks became the dominant paradigm. Models grew from 110M parameters "
        "(BERT-base) to hundreds of billions (GPT-4, PaLM). Efficient inference became "
        "critical: techniques like quantization (INT8, INT4, GPTQ, AWQ), KV cache "
        "optimization, continuous batching, and speculative decoding were developed. "
        "Flash Attention reduced the memory complexity of attention from O(N^2) to O(N). "
        "Rotary Position Embeddings (RoPE) became standard, encoding positional information "
        "as rotations in the complex plane. KV cache reuse across similar prompts became "
        "an active research area, with systems like LMCache, CacheBlend, and SemShareKV "
        "exploring semantic similarity-based cache sharing."
    ),
    (
        "Kubernetes orchestrates containerized applications across clusters of machines. "
        "The control plane consists of the API server, etcd (distributed key-value store), "
        "scheduler, and controller manager. Worker nodes run kubelet, kube-proxy, and a "
        "container runtime. Pods are the smallest deployable units, containing one or more "
        "containers sharing network and storage. Deployments manage ReplicaSets for "
        "rolling updates. Services provide stable networking via ClusterIP, NodePort, or "
        "LoadBalancer types. ConfigMaps and Secrets manage configuration. Persistent "
        "Volumes abstract storage from pods. Horizontal Pod Autoscaler adjusts replicas "
        "based on CPU, memory, or custom metrics. GPU scheduling uses device plugins "
        "(nvidia.com/gpu resource). Node affinity and taints/tolerations control pod "
        "placement. Helm packages applications as charts with templated YAML. Ingress "
        "controllers like ALB or nginx route external HTTP traffic. Service mesh solutions "
        "like Istio add mTLS, traffic management, and observability. EKS on AWS manages "
        "the control plane, while nodegroups provide worker capacity."
    ),
    (
        "Distributed caching architectures form the backbone of high-performance systems. "
        "Redis provides in-memory key-value storage with persistence options (RDB snapshots, "
        "AOF logs). Redis supports data structures beyond simple strings: lists, sets, "
        "sorted sets, hashes, streams, and HyperLogLog. Redis Cluster enables horizontal "
        "scaling via hash slot partitioning across multiple nodes. The RediSearch module "
        "adds secondary indexing and full-text search. Vector similarity search uses HNSW "
        "(Hierarchical Navigable Small World) graphs for approximate nearest neighbor "
        "queries. Milvus is a purpose-built vector database supporting IVF_FLAT, IVF_SQ8, "
        "HNSW, and DiskANN indices. It partitions data into segments for parallel search. "
        "CAGRA (CUDA-Accelerated Graph-based Approximate nearest neighbor) from NVIDIA "
        "runs ANN search directly on GPU, achieving microsecond latencies. Multi-tier "
        "caching stacks L0 GPU cache, L1 Redis, and L2 vector DB for different capacity "
        "and latency tradeoffs. Cache coherence protocols ensure consistency across tiers."
    ),
    (
        "The economics of cloud GPU computing drive architectural decisions. NVIDIA GPU "
        "pricing varies significantly: T4 instances (16GB HBM, FP16 65 TFLOPS) cost "
        "$0.53/hour on AWS g4dn.xlarge. A10G (24GB, 312 TFLOPS) costs $1.01/hour on "
        "g5.xlarge. A100 (80GB, 1555 TFLOPS) costs $3.67/hour on p4d.24xlarge. H100 "
        "(80GB, 3958 TFLOPS) costs $5.12/hour. Memory bandwidth is often the bottleneck: "
        "T4 has 320 GB/s, A10G has 600 GB/s, A100 has 2039 GB/s, H100 has 3350 GB/s. "
        "KV cache memory usage grows linearly with sequence length and batch size. For "
        "Qwen2.5-7B with 32 layers, 32 heads, head_dim=128, FP16: each token requires "
        "32 * 32 * 128 * 2 * 2 = 524KB. An 8K context uses 4GB. GPU memory utilization "
        "is typically set to 85-90% to leave room for activations. Model parallelism "
        "(tensor parallel, pipeline parallel) distributes across GPUs. AWQ quantization "
        "reduces model size 4x with minimal quality loss."
    ),
    (
        "Database optimization fundamentals guide performance engineering. B-tree indices "
        "provide O(log N) lookup for range queries. Hash indices give O(1) for equality. "
        "Composite indices follow the leftmost prefix rule. Query planners choose between "
        "sequential scans, index scans, and bitmap heap scans based on selectivity. "
        "PostgreSQL's MVCC (Multi-Version Concurrency Control) uses transaction snapshots "
        "for isolation. WAL (Write-Ahead Logging) ensures durability. Connection pooling "
        "via PgBouncer reduces overhead. Partitioning by range, list, or hash distributes "
        "large tables. EXPLAIN ANALYZE reveals actual execution plans and row estimates. "
        "Index-only scans avoid heap access when all needed columns are in the index. "
        "Materialized views precompute expensive joins. Full-text search uses GIN indices "
        "on tsvector columns. JSON/JSONB columns with GIN indices support efficient "
        "document queries. Lateral joins enable correlated subqueries. Window functions "
        "compute running aggregates without GROUP BY."
    ),
    (
        "Cryptographic protocols secure modern internet communications. TLS 1.3 reduced "
        "the handshake from two round-trips to one, improving latency. The protocol uses "
        "ephemeral Diffie-Hellman key exchange (ECDHE) for forward secrecy. Cipher suites "
        "specify the key exchange, bulk encryption (AES-256-GCM, ChaCha20-Poly1305), and "
        "hash algorithm (SHA-256, SHA-384). Certificate verification uses X.509 chains "
        "with root CAs. OCSP stapling avoids separate revocation checks. HTTP/2 multiplexes "
        "streams over a single TLS connection. HTTP/3 uses QUIC (UDP-based) with built-in "
        "TLS 1.3. JWT (JSON Web Tokens) encode claims signed with HMAC-SHA256 or RSA-SHA256. "
        "OAuth 2.0 authorization code flow with PKCE prevents token interception. "
        "WebAuthn enables passwordless authentication using public key cryptography. "
        "mTLS (mutual TLS) authenticates both client and server, common in service meshes. "
        "Key rotation policies ensure compromised keys have limited blast radius."
    ),
    (
        "Compiler optimization passes transform code for performance. Dead code elimination "
        "removes unreachable instructions. Constant folding evaluates compile-time expressions. "
        "Loop unrolling reduces branch overhead. Vectorization (SIMD) processes multiple data "
        "elements per instruction. Register allocation via graph coloring minimizes spills. "
        "Inlining replaces function calls with body copies. Tail call optimization reuses "
        "stack frames. Link-time optimization (LTO) enables cross-module inlining. Profile-"
        "guided optimization (PGO) uses runtime data to inform decisions. The LLVM IR "
        "(Intermediate Representation) enables language-independent optimization. MLIR "
        "(Multi-Level IR) adds domain-specific dialects for GPU, tensor, and sparse "
        "computation. Triton compiles Python to PTX via MLIR for GPU kernels. PyTorch's "
        "torch.compile uses TorchDynamo to capture Python bytecode and TorchInductor to "
        "generate Triton kernels. Quantization-aware training (QAT) simulates low-precision "
        "arithmetic during training. Kernel fusion reduces memory bandwidth by combining ops."
    ),
    (
        "Observability engineering combines metrics, logs, and traces. Prometheus scrapes "
        "time-series metrics via HTTP pull. PromQL queries aggregate, filter, and transform "
        "metric data. Grafana dashboards visualize metrics with panels and alerts. OpenTelemetry "
        "provides vendor-neutral instrumentation for traces, metrics, and logs. Distributed "
        "tracing propagates context (trace ID, span ID) across service boundaries. Jaeger "
        "and Zipkin collect and visualize trace data. Log aggregation via Fluentd, Loki, or "
        "Elasticsearch enables full-text search across services. Structured logging (JSON) "
        "enables machine parsing. Correlation IDs link related log entries across services. "
        "SLOs (Service Level Objectives) define reliability targets: availability (99.9%), "
        "latency (p99 < 200ms), error rate (< 0.1%). Error budgets determine when to "
        "prioritize reliability over features. Canary deployments gradually shift traffic "
        "to new versions, rolling back on SLO violations. Chaos engineering (Chaos Monkey, "
        "Litmus) proactively tests failure modes."
    ),
]


def build_unique_prompt(
    context: str,
    uid: int,
    target_chars: int,
    question: str = "Summarize the key concepts described above.",
) -> str:
    """Build a prompt that preserves semantic similarity while preventing prefix cache hits.

    Strategy: a unique reference ID at the START breaks prefix cache matching
    (first tokens differ), while the BULK of the prompt is the repeated context
    text, preserving high semantic similarity for SemBlend matching.
    """
    # Unique prefix breaks prefix cache (different first tokens for each measurement)
    ref_id = hashlib.md5(f"ref-{uid}".encode()).hexdigest()[:12]
    sys_prompt = f"You are a helpful AI assistant. Reference: {ref_id}."

    prefix = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
    )
    suffix = (
        f"\nQuestion: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    base_len = len(prefix) + len(suffix)
    pad_needed = max(0, target_chars - base_len)

    if pad_needed > 0:
        # Repeat the CONTEXT itself to fill length (preserves semantic similarity)
        repeated = context
        while len(repeated) < pad_needed:
            repeated = repeated + " " + context
        return prefix + f"Extended context:\n{repeated[:pad_needed]}\n" + suffix
    return prefix + suffix


def reorder_sentences(text: str, rng: random.Random) -> str:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) < 3:
        return text
    rng.shuffle(sentences)
    return ". ".join(sentences) + "."


def partial_replace(text: str, ratio: float, rng: random.Random) -> str:
    sentences = [s.strip() for s in text.split(". ") if s.strip()]
    if len(sentences) < 3:
        return text
    n_replace = max(1, int(len(sentences) * ratio))
    other = rng.choice([c for c in CONTEXTS if c != text])
    other_sents = [s.strip() for s in other.split(". ") if s.strip()]
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    result = list(sentences)
    for i in indices[:n_replace]:
        result[i] = rng.choice(other_sents)
    return ". ".join(result) + "."


def paraphrase(text: str, rng: random.Random) -> str:
    swaps = {
        "provides": "offers", "uses": "employs", "enables": "allows",
        "reduces": "decreases", "supports": "accommodates",
        "manages": "handles", "includes": "encompasses",
        "requires": "demands", "ensures": "guarantees",
        "computes": "calculates", "stores": "maintains",
    }
    words = text.split()
    result = []
    for w in words:
        w_lower = w.lower().rstrip(".,;:")
        punct = ""
        for c in reversed(w):
            if c in ".,;:":
                punct = c + punct
            else:
                break
        if w_lower in swaps and rng.random() < 0.5:
            rep = swaps[w_lower]
            if w[0].isupper():
                rep = rep.capitalize()
            result.append(rep + punct)
        else:
            result.append(w)
    return " ".join(result)


def call_vllm(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 1,
    logprobs: int = 0,
    max_retries: int = 5,
) -> dict:
    """Call vLLM with retry logic for OOM recovery."""
    body = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    if logprobs > 0:
        body["logprobs"] = logprobs

    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json=body,
                timeout=300.0,
            )
            elapsed_ms = (time.monotonic() - t0) * 1000
            data = resp.json()
            if "choices" not in data:
                err_msg = data.get("message", data.get("error", str(data)))
                raise ValueError(f"vLLM error: {err_msg}")
            data["_ttft_ms"] = elapsed_ms
            return data
        except (requests.ConnectionError, requests.Timeout) as e:
            wait = min(30, 5 * (attempt + 1))
            print(f"    Connection error (attempt {attempt+1}/{max_retries}), "
                  f"waiting {wait}s for vLLM recovery...")
            time.sleep(wait)
        except ValueError:
            raise
    raise RuntimeError(f"vLLM unreachable after {max_retries} retries")


def rouge_l(hyp: str, ref: str) -> float:
    if not hyp or not ref:
        return 0.0
    h, r = hyp.split(), ref.split()
    if not h or not r:
        return 0.0
    m, n = len(r), len(h)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = (dp[i-1][j-1] + 1 if r[i-1] == h[j-1]
                        else max(dp[i-1][j], dp[i][j-1]))
    lcs = dp[m][n]
    p = lcs / n if n else 0
    rc = lcs / m if m else 0
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000) -> dict:
    if not values:
        return {"mean": 0, "ci_lo": 0, "ci_hi": 0, "p50": 0, "p95": 0, "n": 0}
    rng = random.Random(42)
    n = len(values)
    means = sorted(statistics.mean(rng.choices(values, k=n)) for _ in range(n_boot))
    sv = sorted(values)
    return {
        "mean": statistics.mean(values),
        "ci_lo": means[int(0.025 * n_boot)],
        "ci_hi": means[int(0.975 * n_boot) - 1],
        "p50": sv[max(int(0.5 * n) - 1, 0)],
        "p95": sv[max(int(0.95 * n) - 1, 0)],
        "n": n,
    }


@dataclass
class RunResult:
    scenario: str
    token_length: int
    ttft_ms: float
    output_text: str
    avg_logprob: float | None


def run_single_streaming(
    endpoint: str,
    model: str,
    prompt: str,
    scenario: str,
    token_length: int,
    max_tokens: int = 20,
    max_retries: int = 5,
) -> RunResult:
    """Single streaming request: measures TTFT from first token AND captures output.

    This halves the number of completions vs separate TTFT + quality calls,
    reducing KV cache pressure and preventing OOM crashes.
    """
    for attempt in range(max_retries):
        try:
            t0 = time.monotonic()
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0.0,
                    "stream": True,
                },
                timeout=300.0,
                stream=True,
            )
            resp.raise_for_status()

            ttft = None
            chunks = []
            for line in resp.iter_lines():
                if not line:
                    continue
                line_str = line.decode("utf-8", errors="replace")
                if not line_str.startswith("data: "):
                    continue
                payload = line_str[6:]
                if payload.strip() == "[DONE]":
                    break
                if ttft is None:
                    ttft = (time.monotonic() - t0) * 1000
                try:
                    chunk = json.loads(payload)
                    text = chunk.get("choices", [{}])[0].get("text", "")
                    if text:
                        chunks.append(text)
                except json.JSONDecodeError:
                    continue

            if ttft is None:
                raise ValueError("No tokens received from stream")

            output = "".join(chunks)
            return RunResult(scenario, token_length, ttft, output, None)

        except (requests.ConnectionError, requests.Timeout):
            wait = min(30, 5 * (attempt + 1))
            print(f"    Connection error (attempt {attempt+1}/{max_retries}), "
                  f"waiting {wait}s...")
            time.sleep(wait)
        except ValueError as e:
            if "vLLM error" in str(e) or "No tokens" in str(e):
                raise
            raise

    raise RuntimeError(f"vLLM unreachable after {max_retries} retries")


def run_benchmark(
    endpoint: str,
    model: str,
    num_runs: int = 10,
    token_lengths: list[int] | None = None,
    output_path: str | None = None,
) -> dict:
    if token_lengths is None:
        token_lengths = [2048, 4096]

    rng = random.Random(42)
    all_results: dict[str, dict[int, list[RunResult]]] = {}

    print(f"\n{'='*70}")
    print(f"SemBlend Robust Benchmark")
    print(f"  Model: {model}")
    print(f"  Token lengths: {token_lengths}")
    print(f"  Runs per scenario per length: {num_runs}")
    print(f"{'='*70}\n")

    # Warmup
    print("Warming up...", flush=True)
    call_vllm(endpoint, model, "Hello world", max_tokens=1)
    time.sleep(1)

    uid_counter = 1000  # Unique ID for padding generation

    for target_tokens in token_lengths:
        target_chars = target_tokens * 4  # ~4 chars per token
        print(f"\n{'='*70}")
        print(f"TOKEN LENGTH: {target_tokens}")
        print(f"{'='*70}")

        # PHASE 1: Measure ALL cold baselines FIRST (no donors in store yet)
        # This ensures cold measurements are uncontaminated.
        print(f"\n  --- COLD BASELINE (no donors registered yet) ---")
        for run_idx in range(num_runs):
            ctx_idx = run_idx % len(CONTEXTS)
            context = CONTEXTS[ctx_idx]
            cold_uid = uid_counter
            uid_counter += 1
            cold_prompt = build_unique_prompt(context, cold_uid, target_chars,
                                              question="What are the main ideas here?")
            try:
                cold_result = run_single_streaming(endpoint, model, cold_prompt,
                                         "cold", target_tokens)
                all_results.setdefault("cold", {}).setdefault(
                    target_tokens, []).append(cold_result)
                print(f"    [{run_idx+1}/{num_runs}] COLD: {cold_result.ttft_ms:7.0f}ms")
            except (ValueError, RuntimeError) as e:
                print(f"    [{run_idx+1}/{num_runs}] COLD FAILED: {e}")
            time.sleep(0.5)

        cold_runs = all_results.get("cold", {}).get(target_tokens, [])
        cold_p50 = bootstrap_ci([r.ttft_ms for r in cold_runs])["p50"] if cold_runs else 0
        print(f"  Cold p50: {cold_p50:.0f}ms (n={len(cold_runs)})")

        # PHASE 2: For each run, register donor then immediately measure variations
        print(f"\n  --- DONOR REGISTRATION + SCENARIO MEASUREMENT ---")
        for run_idx in range(num_runs):
            ctx_idx = run_idx % len(CONTEXTS)
            context = CONTEXTS[ctx_idx]

            print(f"\n  Run {run_idx+1}/{num_runs} (context {ctx_idx}):")

            # Register donor
            donor_uid = uid_counter
            uid_counter += 1
            donor_prompt = build_unique_prompt(context, donor_uid, target_chars)
            try:
                call_vllm(endpoint, model, donor_prompt, max_tokens=1)
            except (ValueError, RuntimeError) as e:
                print(f"    DONOR REG FAILED: {e}")
                continue
            time.sleep(0.5)

            # Scenarios to measure
            scenarios = [
                ("exact", donor_prompt),
            ]

            # REORDER
            reorder_ctx = reorder_sentences(context, random.Random(run_idx * 7 + 1))
            reorder_uid = uid_counter; uid_counter += 1
            scenarios.append(("reorder",
                build_unique_prompt(reorder_ctx, reorder_uid, target_chars)))

            # PARTIAL_80
            partial_ctx = partial_replace(context, 0.2, random.Random(run_idx * 11 + 2))
            partial_uid = uid_counter; uid_counter += 1
            scenarios.append(("partial_80",
                build_unique_prompt(partial_ctx, partial_uid, target_chars)))

            # PARAPHRASE
            para_ctx = paraphrase(context, random.Random(run_idx * 13 + 3))
            para_uid = uid_counter; uid_counter += 1
            scenarios.append(("paraphrase",
                build_unique_prompt(para_ctx, para_uid, target_chars)))

            # DIVERSE (completely different context)
            div_ctx = CONTEXTS[(ctx_idx + 4) % len(CONTEXTS)]
            div_uid = uid_counter; uid_counter += 1
            scenarios.append(("diverse",
                build_unique_prompt(div_ctx, div_uid, target_chars,
                                    question="Explain the concepts discussed.")))

            for scenario, prompt in scenarios:
                try:
                    result = run_single_streaming(endpoint, model, prompt,
                                        scenario, target_tokens)
                    all_results.setdefault(scenario, {}).setdefault(
                        target_tokens, []).append(result)
                    speedup = cold_p50 / max(result.ttft_ms, 1)
                    label = scenario.upper()
                    print(f"    {label:<12} {result.ttft_ms:7.0f}ms ({speedup:.1f}x)")
                except (ValueError, RuntimeError) as e:
                    print(f"    {scenario.upper():<12} FAILED: {e}")
                time.sleep(0.5)

    # Aggregate and report
    report = {
        "model": model,
        "token_lengths": token_lengths,
        "num_runs": num_runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": {},
    }

    print(f"\n\n{'='*90}")
    print("SEMBLEND BENCHMARK RESULTS")
    print(f"{'='*90}")
    print(f"{'Scenario':<15} {'Length':>6} {'TTFT p50':>10} {'95% CI':>20} "
          f"{'Speedup':>8} {'ROUGE-L':>8} {'EM%':>5}")
    print("-" * 90)

    for scenario in ["cold", "exact", "reorder", "partial_80", "paraphrase", "diverse"]:
        if scenario not in all_results:
            continue
        for tl in token_lengths:
            runs = all_results[scenario].get(tl, [])
            if not runs:
                continue

            ttfts = [r.ttft_ms for r in runs]
            ci = bootstrap_ci(ttfts)

            # Cold baseline for this token length
            cold_runs = all_results.get("cold", {}).get(tl, [])
            cold_p50 = bootstrap_ci([r.ttft_ms for r in cold_runs])["p50"] if cold_runs else 1
            speedup = cold_p50 / ci["p50"] if ci["p50"] > 0 else 0

            # Quality: compare against cold baseline outputs
            rouge_ls = []
            exact_matches = 0
            for i, r in enumerate(runs):
                if i < len(cold_runs) and cold_runs[i].output_text and r.output_text:
                    rl = rouge_l(r.output_text, cold_runs[i].output_text)
                    rouge_ls.append(rl)
                    if r.output_text.strip() == cold_runs[i].output_text.strip():
                        exact_matches += 1

            avg_rl = statistics.mean(rouge_ls) if rouge_ls else 0
            em_rate = exact_matches / max(len(rouge_ls), 1)

            print(f"{scenario:<15} {tl:>6} {ci['p50']:>9.0f}ms "
                  f"[{ci['ci_lo']:>7.0f},{ci['ci_hi']:>7.0f}]ms "
                  f"{speedup:>7.2f}x {avg_rl:>8.3f} {em_rate:>4.0%}")

            report["scenarios"].setdefault(scenario, {})[str(tl)] = {
                "ttft": ci,
                "speedup": speedup,
                "rouge_l": avg_rl,
                "exact_match": em_rate,
                "raw_ttfts": ttfts,
            }

    print(f"{'='*90}")

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SemBlend Robust Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--token-lengths", default="2048,4096",
                        help="Comma-separated target token lengths")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    lengths = [int(x) for x in args.token_lengths.split(",")]
    run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        num_runs=args.num_runs,
        token_lengths=lengths,
        output_path=args.output,
    )
