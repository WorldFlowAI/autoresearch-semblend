#!/usr/bin/env python3
"""SemBlend output quality benchmark — measures generation fidelity with long outputs.

Compares SemBlend-accelerated outputs against cold baseline outputs using
longer generation (default max_tokens=256) to detect any quality degradation
from KV cache donor reuse. Uses tokenizer-verified prompt lengths.

Methodology per (length, run):
  1. COLD_A: Generate with unique prompt (uid=A) -> baseline output + logprobs
  2. COLD_B: Generate with unique prompt (uid=B) -> control output (no donor)
  3. REGISTER: Register donors for this context
  4. SEMBLEND: Generate with unique prompt (uid=D) -> semblend output + logprobs
  5. Compare:
     - control_rouge = ROUGE-L(COLD_A, COLD_B)   # baseline sensitivity
     - semblend_rouge = ROUGE-L(COLD_A, SEMBLEND) # quality under SemBlend
     - PPL ratio = PPL(SEMBLEND) / PPL(COLD_A)

Metrics reported per length:
  - ROUGE-L (via simple LCS, no external deps)
  - Exact match rate
  - Perplexity ratio (via logprobs from vLLM)

Usage:
    python -m benchmarks.e2e.semblend_quality_bench \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --token-lengths "2048,5120,8192,16000" \
        --runs 10 \
        --max-tokens 256 \
        --output-dir results/quality
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone

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

QUESTION = "Summarize the key concepts described above in detail."

_TOKENIZER = None


def _get_tokenizer(model_name: str):
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
    return _TOKENIZER


def build_prompt(
    context: str, uid: int, target_tokens: int, model_name: str,
) -> str:
    """Build a prompt of approximately target_tokens length (tokenizer-verified)."""
    tokenizer = _get_tokenizer(model_name)
    ref_id = hashlib.md5(f"ref-{uid}".encode()).hexdigest()[:12]
    sys_prompt = f"You are a helpful AI assistant. Reference: {ref_id}."
    prefix = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\n"
    )
    suffix = (
        f"\nQuestion: {QUESTION}"
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


def rouge_l(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 using longest common subsequence (no external deps)."""
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
            if ref[i - 1] == hyp[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[m][n]
    precision = lcs / n if n > 0 else 0
    recall = lcs / m if m > 0 else 0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    """Send a prompt to populate the KV cache donor store."""
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": 1, "temperature": 0.0,
        },
        timeout=300.0,
    )
    resp.raise_for_status()


def generate_with_logprobs(
    endpoint: str, model: str, prompt: str, max_tokens: int,
    min_tokens: int = 0,
) -> tuple[str, list[float], float]:
    """Generate text and collect per-token log probabilities.

    Args:
        min_tokens: If >0, forces the model to generate at least this many tokens
            before emitting EOS. Use to prevent EOS-collapse when KV injection from
            a near-identical donor places the model at a sentence boundary.

    Returns (text, token_logprobs, ttft_ms).
    """
    t0 = time.monotonic()
    payload: dict = {
        "model": model, "prompt": prompt,
        "max_tokens": max_tokens, "temperature": 0.0,
        "logprobs": 1,
    }
    if min_tokens > 0:
        payload["min_tokens"] = min_tokens
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json=payload,
        timeout=300.0,
    )
    ttft_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_data = data["choices"][0].get("logprobs", {})
    token_lps = lp_data.get("token_logprobs", []) if lp_data else []
    valid_lps = [lp for lp in token_lps if lp is not None]
    return text, valid_lps, ttft_ms


def compute_perplexity(logprobs: list[float]) -> float | None:
    """Compute perplexity from a list of token log probabilities."""
    if not logprobs:
        return None
    avg_neg_lp = -statistics.mean(logprobs)
    return math.exp(avg_neg_lp)


@dataclass
class RunSample:
    run_idx: int
    context_idx: int
    cold_text: str
    cold_perplexity: float | None
    cold_ttft_ms: float
    control_text: str
    control_perplexity: float | None
    semblend_text: str
    semblend_perplexity: float | None
    semblend_ttft_ms: float
    control_rouge_l: float
    semblend_rouge_l: float
    exact_match: bool
    perplexity_ratio: float | None


@dataclass
class LengthResult:
    target_tokens: int
    n_runs: int
    max_tokens: int
    # ROUGE-L stats (semblend vs cold baseline)
    rouge_l_mean: float | None
    rouge_l_stdev: float | None
    rouge_l_min: float | None
    rouge_l_max: float | None
    # Control ROUGE-L stats (cold_b vs cold_a, baseline sensitivity)
    control_rouge_l_mean: float | None
    # Quality delta: semblend - control (positive = SemBlend as good or better)
    quality_delta: float | None
    # Exact match
    exact_match_rate: float | None
    # Perplexity
    perplexity_ratio_mean: float | None
    perplexity_ratio_stdev: float | None
    cold_perplexity_mean: float | None
    semblend_perplexity_mean: float | None
    # Timing
    cold_ttft_p50_ms: float | None
    semblend_ttft_p50_ms: float | None
    speedup_mean: float | None
    # Verdict
    quality_preserved: bool
    samples: list[dict] = field(default_factory=list)


def run_quality_for_length(
    endpoint: str,
    model: str,
    target_tokens: int,
    runs: int,
    max_tokens: int,
    donors_per_context: int = 2,
    min_tokens: int = 0,
) -> LengthResult:
    """Run quality comparison for a single token length.

    Args:
        min_tokens: Minimum tokens to generate before EOS is allowed. Set to
            prevent EOS-collapse when KV-injection places model at a sentence boundary.
    """
    uid = 20000 + target_tokens
    samples: list[RunSample] = []

    # Warmup
    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)

    for i in range(runs):
        ctx_idx = i % len(CONTEXTS)
        ctx = CONTEXTS[ctx_idx]

        print(f"    Run {i + 1}/{runs}: ", end="", flush=True)

        try:
            # STEP 1: COLD_A -- baseline generation
            cold_a_prompt = build_prompt(ctx, uid, target_tokens, model)
            uid += 1
            cold_text, cold_lps, cold_ttft = generate_with_logprobs(
                endpoint, model, cold_a_prompt, max_tokens, min_tokens=min_tokens,
            )
            cold_ppl = compute_perplexity(cold_lps)
            time.sleep(0.2)

            # STEP 2: COLD_B -- control (same context, different uid/ref_id)
            ctrl_prompt = build_prompt(ctx, uid, target_tokens, model)
            uid += 1
            ctrl_text, ctrl_lps, _ = generate_with_logprobs(
                endpoint, model, ctrl_prompt, max_tokens, min_tokens=min_tokens,
            )
            ctrl_ppl = compute_perplexity(ctrl_lps)
            time.sleep(0.2)

            # STEP 3: Register donors for this context
            for d in range(donors_per_context):
                donor_prompt = build_prompt(ctx, uid, target_tokens, model)
                uid += 1
                register_donor(endpoint, model, donor_prompt)
                time.sleep(0.3)

            time.sleep(1)  # Let donor store settle

            # STEP 4: SEMBLEND -- generation with donor KV available
            sb_prompt = build_prompt(ctx, uid, target_tokens, model)
            uid += 1
            sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                endpoint, model, sb_prompt, max_tokens, min_tokens=min_tokens,
            )
            sb_ppl = compute_perplexity(sb_lps)
            time.sleep(0.2)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Compute quality metrics
        ctrl_rl = rouge_l(ctrl_text, cold_text)
        sem_rl = rouge_l(sb_text, cold_text)
        exact = sb_text.strip() == cold_text.strip()

        ppl_ratio = None
        if cold_ppl is not None and sb_ppl is not None and cold_ppl > 0:
            ppl_ratio = sb_ppl / cold_ppl

        sample = RunSample(
            run_idx=i,
            context_idx=ctx_idx,
            cold_text=cold_text,
            cold_perplexity=cold_ppl,
            cold_ttft_ms=cold_ttft,
            control_text=ctrl_text,
            control_perplexity=ctrl_ppl,
            semblend_text=sb_text,
            semblend_perplexity=sb_ppl,
            semblend_ttft_ms=sb_ttft,
            control_rouge_l=ctrl_rl,
            semblend_rouge_l=sem_rl,
            exact_match=exact,
            perplexity_ratio=ppl_ratio,
        )
        samples.append(sample)

        # EOS-collapse detection: SemBlend output is abnormally short
        # Threshold: < max_tokens/8 words (e.g., <64 words for max_tokens=512)
        collapse_word_threshold = max(10, max_tokens // 8)
        is_collapse = len(sb_text.split()) < collapse_word_threshold
        collapse_tag = " [COLLAPSE]" if is_collapse else ""
        em_tag = " EXACT" if exact else ""
        ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "ppl_r=N/A"
        print(
            f"ctrl_RL={ctrl_rl:.3f} sem_RL={sem_rl:.3f} "
            f"{ppl_str} "
            f"cold={cold_ttft:.0f}ms sb={sb_ttft:.0f}ms{em_tag}{collapse_tag}"
        )

    # Aggregate
    n = len(samples)
    if n == 0:
        return LengthResult(
            target_tokens=target_tokens, n_runs=0, max_tokens=max_tokens,
            rouge_l_mean=None, rouge_l_stdev=None, rouge_l_min=None,
            rouge_l_max=None, control_rouge_l_mean=None, quality_delta=None,
            exact_match_rate=None, perplexity_ratio_mean=None,
            perplexity_ratio_stdev=None, cold_perplexity_mean=None,
            semblend_perplexity_mean=None, cold_ttft_p50_ms=None,
            semblend_ttft_p50_ms=None, speedup_mean=None,
            quality_preserved=False,
        )

    # Separate collapse vs non-collapse runs for accurate quality reporting
    collapse_word_threshold = max(10, max_tokens // 8)
    non_collapse = [
        s for s in samples if len(s.semblend_text.split()) >= collapse_word_threshold
    ]
    collapse_count = n - len(non_collapse)
    if collapse_count > 0:
        nc_ppls = [s.perplexity_ratio for s in non_collapse if s.perplexity_ratio]
        nc_rouges = [s.semblend_rouge_l for s in non_collapse]
        nc_ppl_str = f"{statistics.mean(nc_ppls):.4f}" if nc_ppls else "N/A"
        nc_rl_str = f"{statistics.mean(nc_rouges):.4f}" if nc_rouges else "N/A"
        print(
            f"  [EOS-collapse: {collapse_count}/{n} runs collapsed "
            f"(<{collapse_word_threshold} words). "
            f"Non-collapse only: PPL={nc_ppl_str} ROUGE-L={nc_rl_str}]"
        )

    sem_rouges = [s.semblend_rouge_l for s in samples]
    ctrl_rouges = [s.control_rouge_l for s in samples]
    exact_count = sum(1 for s in samples if s.exact_match)
    ppl_ratios = [
        s.perplexity_ratio for s in samples if s.perplexity_ratio is not None
    ]
    cold_ppls = [
        s.cold_perplexity for s in samples if s.cold_perplexity is not None
    ]
    sb_ppls = [
        s.semblend_perplexity for s in samples
        if s.semblend_perplexity is not None
    ]
    cold_ttfts = sorted([s.cold_ttft_ms for s in samples])
    sb_ttfts = sorted([s.semblend_ttft_ms for s in samples])
    speedups = [
        s.cold_ttft_ms / s.semblend_ttft_ms
        for s in samples if s.semblend_ttft_ms > 0
    ]

    avg_sem_rl = statistics.mean(sem_rouges)
    avg_ctrl_rl = statistics.mean(ctrl_rouges)
    quality_delta = avg_sem_rl - avg_ctrl_rl
    # Quality preserved if SemBlend ROUGE-L >= 0.85 or within 5% of control
    preserved = avg_sem_rl >= 0.85 or quality_delta >= -0.05

    return LengthResult(
        target_tokens=target_tokens,
        n_runs=n,
        max_tokens=max_tokens,
        rouge_l_mean=avg_sem_rl,
        rouge_l_stdev=(
            statistics.stdev(sem_rouges) if len(sem_rouges) > 1 else None
        ),
        rouge_l_min=min(sem_rouges),
        rouge_l_max=max(sem_rouges),
        control_rouge_l_mean=avg_ctrl_rl,
        quality_delta=quality_delta,
        exact_match_rate=exact_count / n,
        perplexity_ratio_mean=(
            statistics.mean(ppl_ratios) if ppl_ratios else None
        ),
        perplexity_ratio_stdev=(
            statistics.stdev(ppl_ratios) if len(ppl_ratios) > 1 else None
        ),
        cold_perplexity_mean=(
            statistics.mean(cold_ppls) if cold_ppls else None
        ),
        semblend_perplexity_mean=(
            statistics.mean(sb_ppls) if sb_ppls else None
        ),
        cold_ttft_p50_ms=(
            cold_ttfts[len(cold_ttfts) // 2] if cold_ttfts else None
        ),
        semblend_ttft_p50_ms=(
            sb_ttfts[len(sb_ttfts) // 2] if sb_ttfts else None
        ),
        speedup_mean=statistics.mean(speedups) if speedups else None,
        quality_preserved=preserved,
        samples=[asdict(s) for s in samples],
    )


def _load_clusters(clusters_file: str) -> list[dict]:
    """Load cluster data from a JSON file produced by build_clusters.py."""
    with open(clusters_file) as f:
        return json.load(f)


def _pick_semblend_variation(cluster: dict) -> dict | None:
    """Pick the best variation for SemBlend measurement.

    Preference: reorder > partial_80 > paraphrase > any non-exact.
    """
    preference = ["reorder", "partial_80", "paraphrase"]
    by_type = {v["overlap_type"]: v for v in cluster.get("variations", [])}
    for pref in preference:
        if pref in by_type:
            return by_type[pref]
    for v in cluster.get("variations", []):
        if v["overlap_type"] != "exact":
            return v
    return None


def run_quality_for_length_clusters(
    endpoint: str,
    model: str,
    clusters_for_length: list[dict],
    target_tokens: int,
    runs: int,
    max_tokens: int,
    donors_per_context: int = 2,
    min_tokens: int = 0,
) -> LengthResult:
    """Run quality comparison for a single token length using cluster data."""
    uid = 30000 + target_tokens
    samples: list[RunSample] = []

    # Warmup
    register_donor(endpoint, model, "Hello world warmup test")
    time.sleep(1)

    for i in range(min(runs, len(clusters_for_length))):
        cluster = clusters_for_length[i]

        print(f"    Run {i + 1}/{runs}: ", end="", flush=True)

        try:
            # STEP 1: COLD_A -- baseline generation using seed_text with unique uid
            ref_id = hashlib.md5(f"cold-a-{uid}".encode()).hexdigest()[:12]
            cold_a_prompt = cluster["seed_text"].replace(
                "<|im_start|>system\n",
                f"<|im_start|>system\nRef: {ref_id}. ",
                1,
            ) if "<|im_start|>system\n" in cluster["seed_text"] else cluster["seed_text"]
            uid += 1
            cold_text, cold_lps, cold_ttft = generate_with_logprobs(
                endpoint, model, cold_a_prompt, max_tokens, min_tokens=min_tokens,
            )
            cold_ppl = compute_perplexity(cold_lps)
            time.sleep(0.2)

            # STEP 2: COLD_B -- control (same seed_text, different uid)
            ref_id = hashlib.md5(f"cold-b-{uid}".encode()).hexdigest()[:12]
            ctrl_prompt = cluster["seed_text"].replace(
                "<|im_start|>system\n",
                f"<|im_start|>system\nRef: {ref_id}. ",
                1,
            ) if "<|im_start|>system\n" in cluster["seed_text"] else cluster["seed_text"]
            uid += 1
            ctrl_text, ctrl_lps, _ = generate_with_logprobs(
                endpoint, model, ctrl_prompt, max_tokens, min_tokens=min_tokens,
            )
            ctrl_ppl = compute_perplexity(ctrl_lps)
            time.sleep(0.2)

            # STEP 3: Register seed_text as donor
            for d in range(donors_per_context):
                register_donor(endpoint, model, cluster["seed_text"])
                time.sleep(0.3)

            time.sleep(1)  # Let donor store settle

            # STEP 4: SEMBLEND -- use a variation text
            variation = _pick_semblend_variation(cluster)
            if variation is None:
                print("SKIPPED: no variation")
                continue
            sb_prompt = variation["text"]
            uid += 1
            sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                endpoint, model, sb_prompt, max_tokens, min_tokens=min_tokens,
            )
            sb_ppl = compute_perplexity(sb_lps)
            time.sleep(0.2)

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        # Compute quality metrics
        ctrl_rl = rouge_l(ctrl_text, cold_text)
        sem_rl = rouge_l(sb_text, cold_text)
        exact = sb_text.strip() == cold_text.strip()

        ppl_ratio = None
        if cold_ppl is not None and sb_ppl is not None and cold_ppl > 0:
            ppl_ratio = sb_ppl / cold_ppl

        sample = RunSample(
            run_idx=i,
            context_idx=i,
            cold_text=cold_text,
            cold_perplexity=cold_ppl,
            cold_ttft_ms=cold_ttft,
            control_text=ctrl_text,
            control_perplexity=ctrl_ppl,
            semblend_text=sb_text,
            semblend_perplexity=sb_ppl,
            semblend_ttft_ms=sb_ttft,
            control_rouge_l=ctrl_rl,
            semblend_rouge_l=sem_rl,
            exact_match=exact,
            perplexity_ratio=ppl_ratio,
        )
        samples.append(sample)

        collapse_word_threshold = max(10, max_tokens // 8)
        is_collapse = len(sb_text.split()) < collapse_word_threshold
        collapse_tag = " [COLLAPSE]" if is_collapse else ""
        em_tag = " EXACT" if exact else ""
        ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "ppl_r=N/A"
        print(
            f"ctrl_RL={ctrl_rl:.3f} sem_RL={sem_rl:.3f} "
            f"{ppl_str} "
            f"cold={cold_ttft:.0f}ms sb={sb_ttft:.0f}ms{em_tag}{collapse_tag}"
        )

    # Aggregate (same logic as run_quality_for_length)
    n = len(samples)
    if n == 0:
        return LengthResult(
            target_tokens=target_tokens, n_runs=0, max_tokens=max_tokens,
            rouge_l_mean=None, rouge_l_stdev=None, rouge_l_min=None,
            rouge_l_max=None, control_rouge_l_mean=None, quality_delta=None,
            exact_match_rate=None, perplexity_ratio_mean=None,
            perplexity_ratio_stdev=None, cold_perplexity_mean=None,
            semblend_perplexity_mean=None, cold_ttft_p50_ms=None,
            semblend_ttft_p50_ms=None, speedup_mean=None,
            quality_preserved=False,
        )

    collapse_word_threshold = max(10, max_tokens // 8)
    non_collapse = [
        s for s in samples if len(s.semblend_text.split()) >= collapse_word_threshold
    ]
    collapse_count = n - len(non_collapse)
    if collapse_count > 0:
        nc_ppls = [s.perplexity_ratio for s in non_collapse if s.perplexity_ratio]
        nc_rouges = [s.semblend_rouge_l for s in non_collapse]
        nc_ppl_str = f"{statistics.mean(nc_ppls):.4f}" if nc_ppls else "N/A"
        nc_rl_str = f"{statistics.mean(nc_rouges):.4f}" if nc_rouges else "N/A"
        print(
            f"  [EOS-collapse: {collapse_count}/{n} runs collapsed "
            f"(<{collapse_word_threshold} words). "
            f"Non-collapse only: PPL={nc_ppl_str} ROUGE-L={nc_rl_str}]"
        )

    sem_rouges = [s.semblend_rouge_l for s in samples]
    ctrl_rouges = [s.control_rouge_l for s in samples]
    exact_count = sum(1 for s in samples if s.exact_match)
    ppl_ratios = [
        s.perplexity_ratio for s in samples if s.perplexity_ratio is not None
    ]
    cold_ppls = [
        s.cold_perplexity for s in samples if s.cold_perplexity is not None
    ]
    sb_ppls = [
        s.semblend_perplexity for s in samples
        if s.semblend_perplexity is not None
    ]
    cold_ttfts = sorted([s.cold_ttft_ms for s in samples])
    sb_ttfts = sorted([s.semblend_ttft_ms for s in samples])
    speedups = [
        s.cold_ttft_ms / s.semblend_ttft_ms
        for s in samples if s.semblend_ttft_ms > 0
    ]

    avg_sem_rl = statistics.mean(sem_rouges)
    avg_ctrl_rl = statistics.mean(ctrl_rouges)
    quality_delta = avg_sem_rl - avg_ctrl_rl
    preserved = avg_sem_rl >= 0.85 or quality_delta >= -0.05

    return LengthResult(
        target_tokens=target_tokens,
        n_runs=n,
        max_tokens=max_tokens,
        rouge_l_mean=avg_sem_rl,
        rouge_l_stdev=(
            statistics.stdev(sem_rouges) if len(sem_rouges) > 1 else None
        ),
        rouge_l_min=min(sem_rouges),
        rouge_l_max=max(sem_rouges),
        control_rouge_l_mean=avg_ctrl_rl,
        quality_delta=quality_delta,
        exact_match_rate=exact_count / n,
        perplexity_ratio_mean=(
            statistics.mean(ppl_ratios) if ppl_ratios else None
        ),
        perplexity_ratio_stdev=(
            statistics.stdev(ppl_ratios) if len(ppl_ratios) > 1 else None
        ),
        cold_perplexity_mean=(
            statistics.mean(cold_ppls) if cold_ppls else None
        ),
        semblend_perplexity_mean=(
            statistics.mean(sb_ppls) if sb_ppls else None
        ),
        cold_ttft_p50_ms=(
            cold_ttfts[len(cold_ttfts) // 2] if cold_ttfts else None
        ),
        semblend_ttft_p50_ms=(
            sb_ttfts[len(sb_ttfts) // 2] if sb_ttfts else None
        ),
        speedup_mean=statistics.mean(speedups) if speedups else None,
        quality_preserved=preserved,
        samples=[asdict(s) for s in samples],
    )


def _fmt(val, f_str: str) -> str:
    """Format a value, returning '---' if None."""
    return f_str.format(val) if val is not None else "---"


def main():
    parser = argparse.ArgumentParser(
        description="SemBlend output quality benchmark (Sprint 1)",
    )
    parser.add_argument("--endpoint", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--token-lengths", default="2048,5120,8192,16000",
        help="Comma-separated target token lengths",
    )
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--min-tokens", type=int, default=0,
        help="Force model to generate at least this many tokens before EOS is allowed. "
             "Use to prevent EOS-collapse when KV injection places model at a sentence "
             "boundary (e.g., set to 64 or 128 when max-tokens >= 512).",
    )
    parser.add_argument("--donors-per-context", type=int, default=2)
    parser.add_argument("--output-dir", default="/tmp/semblend-quality-results")
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

    cluster_mode = args.clusters_file is not None

    print("SemBlend Quality Benchmark (Sprint 1)")
    print(f"  Model: {args.model}")
    print(f"  Token lengths: {lengths}")
    print(f"  Runs per length: {args.runs}")
    print(f"  Max tokens (generation): {args.max_tokens}")
    print(f"  Donors per context: {args.donors_per_context}")
    if cluster_mode:
        print(f"  Cluster file: {args.clusters_file}")
    if args.dataset_name:
        print(f"  Dataset: {args.dataset_name}")
    print()

    # Health check
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        if resp.status_code != 200:
            print(f"vLLM unhealthy: {resp.status_code}")
            return
        print(f"  vLLM healthy at {args.endpoint}\n")
    except Exception as e:
        print(f"Cannot reach vLLM: {e}")
        return

    # Load clusters if provided
    clusters_by_length: dict[int, list[dict]] = {}
    if cluster_mode:
        all_clusters = _load_clusters(args.clusters_file)
        for c in all_clusters:
            tl = c["target_token_length"]
            clusters_by_length.setdefault(tl, []).append(c)
        # Override lengths to those available in cluster file
        available_lengths = sorted(clusters_by_length.keys())
        if lengths:
            # Filter to requested lengths that exist in cluster data
            lengths = [tl for tl in lengths if tl in clusters_by_length]
        if not lengths:
            lengths = available_lengths
        print(f"  Cluster lengths available: {available_lengths}")
        print(f"  Using lengths: {lengths}\n")

    all_results: dict[int, LengthResult] = {}

    for target_tokens in lengths:
        print(f"\n{'=' * 65}")
        print(f"  {target_tokens} TOKENS (max_tokens={args.max_tokens})")
        print(f"{'=' * 65}")

        if cluster_mode:
            result = run_quality_for_length_clusters(
                endpoint=args.endpoint,
                model=args.model,
                clusters_for_length=clusters_by_length.get(target_tokens, []),
                target_tokens=target_tokens,
                runs=args.runs,
                max_tokens=args.max_tokens,
                donors_per_context=args.donors_per_context,
                min_tokens=args.min_tokens,
            )
        else:
            result = run_quality_for_length(
                endpoint=args.endpoint,
                model=args.model,
                target_tokens=target_tokens,
                runs=args.runs,
                max_tokens=args.max_tokens,
                donors_per_context=args.donors_per_context,
                min_tokens=args.min_tokens,
            )
        all_results[target_tokens] = result

    # --- Summary table ---
    print(f"\n{'=' * 100}")
    print("QUALITY RESULTS SUMMARY")
    print(f"{'=' * 100}")
    print(
        f"{'Tokens':>8} {'N':>4} {'Sem RL':>8} {'Ctrl RL':>9} "
        f"{'Delta':>7} {'Exact%':>8} {'PPL Ratio':>10} "
        f"{'Cold PPL':>9} {'SB PPL':>9} {'Speedup':>8} {'Status':>7}"
    )
    print("-" * 100)

    all_pass = True
    for tl in sorted(all_results.keys()):
        r = all_results[tl]
        if not r.quality_preserved:
            all_pass = False

        status = "PASS" if r.quality_preserved else "FAIL"
        print(
            f"{tl:>8} {r.n_runs:>4} "
            f"{_fmt(r.rouge_l_mean, '{:.4f}'):>8} "
            f"{_fmt(r.control_rouge_l_mean, '{:.4f}'):>9} "
            f"{_fmt(r.quality_delta, '{:+.4f}'):>7} "
            f"{_fmt(r.exact_match_rate, '{:.0%}'):>8} "
            f"{_fmt(r.perplexity_ratio_mean, '{:.4f}'):>10} "
            f"{_fmt(r.cold_perplexity_mean, '{:.2f}'):>9} "
            f"{_fmt(r.semblend_perplexity_mean, '{:.2f}'):>9} "
            f"{_fmt(r.speedup_mean, '{:.2f}x'):>8} "
            f"{status:>7}"
        )

    print(f"\n  Interpretation:")
    print(f"    PPL Ratio:  1.0 = identical quality, <1.05 = negligible degradation")
    print(f"    ROUGE-L:    1.0 = identical output, >0.8 = high similarity")
    print(f"    Delta:      Sem RL - Ctrl RL. Positive = SemBlend as good or better")
    print(f"    Ctrl RL:    Baseline sensitivity (two cold calls, different ref_id)")
    print(f"\n  Verdict: {'QUALITY PRESERVED' if all_pass else 'QUALITY DEGRADATION DETECTED'}")
    print(f"{'=' * 100}")

    # --- Save JSON ---
    run_id = (
        f"semblend-quality-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    report = {
        "run_id": run_id,
        "model": args.model,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "max_tokens": args.max_tokens,
        "donors_per_context": args.donors_per_context,
        "runs_per_length": args.runs,
        "token_lengths": lengths,
        "cluster_mode": cluster_mode,
        "results": {
            str(tl): asdict(r) for tl, r in all_results.items()
        },
        "verdict": (
            "QUALITY PRESERVED" if all_pass
            else "QUALITY DEGRADATION DETECTED"
        ),
    }
    if args.dataset_name:
        report["dataset_name"] = args.dataset_name

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
