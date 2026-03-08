#!/usr/bin/env python3
"""SemBlend benchmark script designed to run inside a k8s pod.

Self-contained — only requires `requests` (available in most Python images).
Generates its own test prompts from templates instead of requiring HuggingFace datasets.

Produces:
- TTFT measurements across scenarios (COLD, EXACT, REORDER, PARTIAL, PARAPHRASE, DIVERSE)
- Quality measurements (ROUGE-L via logprobs comparison)
- Speedup ratios vs cold baseline
- JSON results file

Usage inside pod:
    python3 /bench/semblend_k8s_bench.py \
        --endpoint http://synapse-staging-vllm:8000 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --num-runs 10 \
        --output /results/bench_results.json
"""
from __future__ import annotations

import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Prompt generation (self-contained, no external datasets needed)
# ---------------------------------------------------------------------------

# Real-world-style RAG contexts (long, realistic documents)
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

SYSTEM_PROMPTS = [
    "You are a helpful AI assistant that provides accurate, detailed answers.",
    "You are an expert technical analyst. Provide thorough explanations.",
    "You are a research assistant. Base your answers on the provided context.",
    "You are a knowledgeable tutor. Explain concepts step by step.",
]

QUESTIONS = [
    "Summarize the key concepts described in the context above.",
    "What are the most important technical details mentioned?",
    "Explain the main ideas and their practical implications.",
    "Provide a detailed analysis of the information presented.",
    "What are the critical tradeoffs discussed in this context?",
]


def build_prompt(system: str, context: str, question: str, pad_to_tokens: int = 0) -> str:
    """Build a chat-formatted prompt."""
    base = (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    if pad_to_tokens > 0:
        # Rough approximation: 1 token ≈ 4 chars
        target_chars = pad_to_tokens * 4
        if len(base) < target_chars:
            # Repeat context to pad
            padding = context
            while len(base) + len(padding) < target_chars:
                padding = padding + " " + context
            base = (
                f"<|im_start|>system\n{system}<|im_end|>\n"
                f"<|im_start|>user\n"
                f"Context:\n{context}\n\n"
                f"Additional context:\n{padding[:target_chars - len(base)]}\n\n"
                f"Question: {question}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )
    return base


def reorder_context(context: str, rng: random.Random) -> str:
    """Shuffle sentences in context (same tokens, different positions)."""
    sentences = [s.strip() for s in context.split(". ") if s.strip()]
    if len(sentences) < 3:
        return context
    rng.shuffle(sentences)
    return ". ".join(sentences) + "."


def partial_replace(context: str, replace_ratio: float, rng: random.Random) -> str:
    """Replace a fraction of sentences with content from a different context."""
    sentences = [s.strip() for s in context.split(". ") if s.strip()]
    if len(sentences) < 3:
        return context
    n_replace = max(1, int(len(sentences) * replace_ratio))
    other = rng.choice(CONTEXTS)
    other_sents = [s.strip() for s in other.split(". ") if s.strip()]
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    for i in indices[:n_replace]:
        if other_sents:
            sentences[i] = rng.choice(other_sents)
    return ". ".join(sentences) + "."


def paraphrase_context(context: str, rng: random.Random) -> str:
    """Simple synonym substitution."""
    swaps = {
        "provides": "offers", "uses": "employs", "enables": "allows",
        "reduces": "decreases", "supports": "accommodates",
        "manages": "handles", "includes": "encompasses",
        "requires": "demands", "ensures": "guarantees",
        "allows": "permits", "computes": "calculates",
    }
    words = context.split()
    result = []
    for w in words:
        w_lower = w.lower().rstrip(".,;:")
        punct = ""
        for c in reversed(w):
            if c in ".,;:":
                punct = c + punct
            else:
                break
        if w_lower in swaps and rng.random() < 0.4:
            rep = swaps[w_lower]
            if w[0].isupper():
                rep = rep.capitalize()
            result.append(rep + punct)
        else:
            result.append(w)
    return " ".join(result)


@dataclass
class BenchPrompt:
    seed: str
    variation: str
    scenario: str
    context_idx: int
    approx_tokens: int


def generate_prompts(
    num_per_scenario: int = 10,
    target_tokens: int = 2048,
    rng: random.Random | None = None,
) -> list[BenchPrompt]:
    """Generate benchmark prompts for all scenarios."""
    if rng is None:
        rng = random.Random(42)

    prompts = []
    scenarios = {
        "cold": lambda ctx, r: ctx,  # Same context (but first-ever request)
        "exact": lambda ctx, r: ctx,
        "reorder": reorder_context,
        "partial_80": lambda ctx, r: partial_replace(ctx, 0.2, r),
        "partial_60": lambda ctx, r: partial_replace(ctx, 0.4, r),
        "partial_40": lambda ctx, r: partial_replace(ctx, 0.6, r),
        "paraphrase": paraphrase_context,
        "diverse": lambda ctx, r: rng.choice([c for c in CONTEXTS if c != ctx]),
    }

    for scenario, transform_fn in scenarios.items():
        for i in range(num_per_scenario):
            ctx_idx = (i * 3 + hash(scenario)) % len(CONTEXTS)
            ctx = CONTEXTS[ctx_idx]
            sys = SYSTEM_PROMPTS[i % len(SYSTEM_PROMPTS)]
            q = QUESTIONS[i % len(QUESTIONS)]

            seed = build_prompt(sys, ctx, q, pad_to_tokens=target_tokens)
            var_ctx = transform_fn(ctx, rng)
            variation = build_prompt(sys, var_ctx, q, pad_to_tokens=target_tokens)

            prompts.append(BenchPrompt(
                seed=seed,
                variation=variation,
                scenario=scenario,
                context_idx=ctx_idx,
                approx_tokens=target_tokens,
            ))

    return prompts


# ---------------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------------

def measure_ttft(endpoint: str, model: str, prompt: str, max_tokens: int = 1) -> float:
    """Measure TTFT in ms."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": False,
        },
        timeout=300.0,
    )
    ttft = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return ttft


def measure_with_output(
    endpoint: str, model: str, prompt: str, max_tokens: int = 50,
) -> tuple[float, str, float | None]:
    """Returns (ttft_ms, output_text, avg_logprob)."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 1,
            "stream": False,
        },
        timeout=300.0,
    )
    ttft = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_data = data["choices"][0].get("logprobs", {})
    token_lps = lp_data.get("token_logprobs", []) if lp_data else []
    valid = [lp for lp in token_lps if lp is not None]
    avg_lp = statistics.mean(valid) if valid else None
    return ttft, text, avg_lp


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
            dp[i][j] = dp[i-1][j-1] + 1 if r[i-1] == h[j-1] else max(dp[i-1][j], dp[i][j-1])
    lcs = dp[m][n]
    p, rc = lcs / n if n else 0, lcs / m if m else 0
    return 2 * p * rc / (p + rc) if (p + rc) > 0 else 0.0


def bootstrap_ci(values: list[float], n_boot: int = 2000) -> dict:
    if not values:
        return {"mean": 0, "ci_lo": 0, "ci_hi": 0, "p50": 0, "p95": 0, "p99": 0, "n": 0}
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
        "p99": sv[max(int(0.99 * n) - 1, 0)],
        "n": n,
    }


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_benchmark(
    endpoint: str,
    model: str,
    num_runs: int = 10,
    target_tokens: int = 2048,
    measure_quality: bool = True,
    output_path: str | None = None,
) -> dict:
    """Run the full benchmark suite.

    TTFT is ALWAYS measured with max_tokens=1 (pure prefill).
    Quality is measured in a separate pass with max_tokens=50.
    A 0.5s delay between requests prevents KV cache exhaustion.
    """
    rng = random.Random(42)
    prompts = generate_prompts(num_runs, target_tokens, rng)

    # Group by scenario
    by_scenario: dict[str, list[BenchPrompt]] = {}
    for p in prompts:
        by_scenario.setdefault(p.scenario, []).append(p)

    results = {}
    all_cold_ttfts = []

    print(f"\n{'='*70}")
    print(f"SemBlend Scale Benchmark")
    print(f"  Model: {model}")
    print(f"  Target tokens: {target_tokens}")
    print(f"  Runs per scenario: {num_runs}")
    print(f"  Quality measurement: {measure_quality}")
    print(f"{'='*70}\n")

    # Warm up
    print("Warming up...", flush=True)
    try:
        measure_ttft(endpoint, model, "Hello world", max_tokens=1)
    except Exception as e:
        print(f"Warmup failed: {e}")
        return {}

    # Phase 1: COLD baseline (unique prompts, no donor matching)
    print(f"\n--- COLD baseline (max_tokens=1, pure prefill) ---")
    cold_prompts = by_scenario.get("cold", [])
    for i, p in enumerate(cold_prompts):
        try:
            ttft = measure_ttft(endpoint, model, p.seed, max_tokens=1)
            all_cold_ttfts.append(ttft)
            print(f"  [{i+1}/{len(cold_prompts)}] TTFT={ttft:.0f}ms")
            time.sleep(0.3)
        except Exception as e:
            print(f"  [{i+1}] FAILED: {e}")

    cold_ci = bootstrap_ci(all_cold_ttfts)
    results["cold"] = {
        "ttft": cold_ci,
        "speedup": 1.0,
        "raw_ttfts": all_cold_ttfts,
    }
    print(f"  COLD p50={cold_ci['p50']:.0f}ms p99={cold_ci['p99']:.0f}ms")

    # Phase 2: Register donors (send seed prompts, max_tokens=1)
    print(f"\n--- Registering donors ---")
    registered = set()
    for scenario, sp_list in by_scenario.items():
        if scenario == "cold":
            continue
        for p in sp_list:
            if p.context_idx not in registered:
                try:
                    measure_ttft(endpoint, model, p.seed, max_tokens=1)
                    registered.add(p.context_idx)
                    time.sleep(0.3)
                except Exception as e:
                    print(f"  Failed to register donor {p.context_idx}: {e}")
    print(f"  Registered {len(registered)} unique donors")

    # Phase 3: Measure TTFT for each scenario (max_tokens=1)
    for scenario in ["exact", "reorder", "partial_80", "partial_60",
                      "partial_40", "paraphrase", "diverse"]:
        sp_list = by_scenario.get(scenario, [])
        if not sp_list:
            continue

        print(f"\n--- {scenario.upper()} (TTFT, max_tokens=1) ---")
        ttfts = []

        for i, p in enumerate(sp_list):
            try:
                ttft = measure_ttft(endpoint, model, p.variation, max_tokens=1)
                ttfts.append(ttft)
                print(f"  [{i+1}/{len(sp_list)}] TTFT={ttft:.0f}ms")
                time.sleep(0.3)
            except Exception as e:
                print(f"  [{i+1}] FAILED: {e}")

        ttft_ci = bootstrap_ci(ttfts)
        speedup = cold_ci["p50"] / ttft_ci["p50"] if ttft_ci["p50"] > 0 else 0

        scenario_result = {
            "ttft": ttft_ci,
            "speedup": speedup,
            "raw_ttfts": ttfts,
        }
        results[scenario] = scenario_result
        print(f"  {scenario.upper()} p50={ttft_ci['p50']:.0f}ms "
              f"speedup={speedup:.2f}x")

    # Phase 4: Quality measurement (separate pass, max_tokens=20)
    if measure_quality:
        print(f"\n--- QUALITY MEASUREMENT (max_tokens=20) ---")
        for scenario in ["exact", "reorder", "partial_80", "paraphrase"]:
            sp_list = by_scenario.get(scenario, [])
            if not sp_list:
                continue

            rouge_ls = []
            exact_matches = 0
            ppl_ratios = []
            n_measured = min(5, len(sp_list))  # Limit quality runs

            print(f"  {scenario.upper()}:")
            for i in range(n_measured):
                p = sp_list[i]
                try:
                    _, seed_output, seed_lp = measure_with_output(
                        endpoint, model, p.seed, max_tokens=20
                    )
                    time.sleep(0.3)
                    _, var_output, var_lp = measure_with_output(
                        endpoint, model, p.variation, max_tokens=20
                    )
                    time.sleep(0.3)

                    rl = rouge_l(var_output, seed_output)
                    rouge_ls.append(rl)
                    if var_output.strip() == seed_output.strip():
                        exact_matches += 1
                    if seed_lp is not None and var_lp is not None and seed_lp != 0:
                        ppl_base = math.exp(-seed_lp)
                        ppl_var = math.exp(-var_lp)
                        if ppl_base > 0:
                            ppl_ratios.append(ppl_var / ppl_base)
                    print(f"    [{i+1}/{n_measured}] ROUGE-L={rl:.3f}")
                except Exception as e:
                    print(f"    [{i+1}] FAILED: {e}")

            if rouge_ls and scenario in results:
                results[scenario]["quality"] = {
                    "rouge_l_mean": statistics.mean(rouge_ls),
                    "rouge_l_min": min(rouge_ls),
                    "rouge_l_max": max(rouge_ls),
                    "exact_match_rate": exact_matches / max(n_measured, 1),
                    "ppl_ratio_mean": statistics.mean(ppl_ratios) if ppl_ratios else None,
                    "n_quality": len(rouge_ls),
                }

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Scenario':<15} {'TTFT p50':>10} {'TTFT p99':>10} {'Speedup':>8} "
          f"{'ROUGE-L':>8} {'EM%':>5} {'PPL':>8}")
    print(f"{'-'*80}")

    for scenario in ["cold", "exact", "reorder", "partial_80", "partial_60",
                      "partial_40", "paraphrase", "diverse"]:
        r = results.get(scenario)
        if not r:
            continue
        q = r.get("quality", {})
        rl = f"{q.get('rouge_l_mean', 0):.3f}" if q else "---"
        em = f"{q.get('exact_match_rate', 0):.0%}" if q else "---"
        ppl = f"{q.get('ppl_ratio_mean', 0):.3f}" if q and q.get('ppl_ratio_mean') else "---"
        print(f"{scenario:<15} {r['ttft']['p50']:>9.0f}ms {r['ttft']['p99']:>9.0f}ms "
              f"{r.get('speedup', 0):>7.2f}x {rl:>8} {em:>5} {ppl:>8}")

    print(f"{'='*80}")

    # Save
    report = {
        "model": model,
        "target_tokens": target_tokens,
        "num_runs": num_runs,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": results,
    }

    if output_path:
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    return report


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--num-runs", type=int, default=10)
    parser.add_argument("--target-tokens", type=int, default=2048)
    parser.add_argument("--no-quality", action="store_true")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        num_runs=args.num_runs,
        target_tokens=args.target_tokens,
        measure_quality=not args.no_quality,
        output_path=args.output,
    )
