#!/usr/bin/env python3
"""SemBlend memory savings benchmark — measures GPU memory efficiency from KV reuse.

Quantifies GPU memory savings from SemBlend donor KV cache sharing by:
  1. Querying vLLM's /metrics endpoint for GPU memory utilization
  2. Measuring how many concurrent contexts fit with vs without donor sharing
  3. Tracking LMCache CPU buffer utilization

Reports:
  - GPU memory saved per donor reuse event
  - Maximum concurrent contexts before capacity exhaustion
  - CPU buffer usage from LMCache
  - Memory efficiency ratio (with donors vs without)

Usage:
    python -m benchmarks.e2e.semblend_memory_bench \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --num-donors 10 \
        --output-dir results/memory
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
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
]

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
    ref_id = hashlib.md5(f"mem-{uid}".encode()).hexdigest()[:12]
    sys_prompt = f"You are a helpful AI assistant. Reference: {ref_id}."
    prefix = (
        f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
        f"<|im_start|>user\nContext:\n{context}\n\n"
    )
    suffix = (
        "\nQuestion: Summarize the key concepts."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    base_prompt = prefix + suffix
    base_count = len(tokenizer.encode(base_prompt))

    if base_count >= target_tokens:
        return base_prompt

    pad_needed = target_tokens - base_count
    repeated = context
    while len(tokenizer.encode(repeated)) < pad_needed + 200:
        repeated = repeated + " " + context

    pad_ids = tokenizer.encode(repeated)
    if len(pad_ids) > pad_needed:
        repeated = tokenizer.decode(pad_ids[:pad_needed])

    full = prefix + f"Extended context:\n{repeated}\n" + suffix
    full_ids = tokenizer.encode(full)
    if len(full_ids) > target_tokens + 10:
        return tokenizer.decode(full_ids[:target_tokens])
    return full


@dataclass
class GpuMemorySnapshot:
    """GPU memory state from vLLM /metrics endpoint."""
    timestamp: float
    gpu_cache_usage_pct: float | None
    num_preemptions: int | None
    num_running: int | None
    num_waiting: int | None
    gpu_kv_cache_blocks_used: int | None
    gpu_kv_cache_blocks_total: int | None
    cpu_kv_cache_blocks_used: int | None
    cpu_kv_cache_blocks_total: int | None
    raw_metrics: dict = field(default_factory=dict)


def _parse_metric(text: str, name: str) -> float | None:
    """Extract a single Prometheus metric value from /metrics output."""
    pattern = rf'^{re.escape(name)}\s+([\d.eE+\-]+)'
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            return None
    return None


def scrape_gpu_metrics(endpoint: str) -> GpuMemorySnapshot:
    """Scrape vLLM's /metrics endpoint for GPU memory utilization."""
    resp = requests.get(f"{endpoint}/metrics", timeout=10)
    resp.raise_for_status()
    text = resp.text

    # vLLM metric names vary by version; try both known patterns
    gpu_cache = _parse_metric(text, "vllm:gpu_cache_usage_perc")
    if gpu_cache is None:
        gpu_cache = _parse_metric(text, "vllm:kv_cache_usage_perc")

    preemptions = _parse_metric(text, "vllm:num_preemptions_total")
    running = _parse_metric(text, "vllm:num_requests_running")
    waiting = _parse_metric(text, "vllm:num_requests_waiting")

    # KV cache block counts
    gpu_blocks_used = _parse_metric(text, "vllm:num_gpu_blocks_used")
    gpu_blocks_total = _parse_metric(text, "vllm:num_gpu_blocks")
    cpu_blocks_used = _parse_metric(text, "vllm:num_cpu_blocks_used")
    cpu_blocks_total = _parse_metric(text, "vllm:num_cpu_blocks")

    # Collect all vllm: metrics for raw storage
    raw = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        if line.startswith("vllm:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    raw[parts[0]] = float(parts[-1])
                except ValueError:
                    raw[parts[0]] = parts[-1]

    return GpuMemorySnapshot(
        timestamp=time.time(),
        gpu_cache_usage_pct=gpu_cache,
        num_preemptions=(
            int(preemptions) if preemptions is not None else None
        ),
        num_running=int(running) if running is not None else None,
        num_waiting=int(waiting) if waiting is not None else None,
        gpu_kv_cache_blocks_used=(
            int(gpu_blocks_used) if gpu_blocks_used is not None else None
        ),
        gpu_kv_cache_blocks_total=(
            int(gpu_blocks_total) if gpu_blocks_total is not None else None
        ),
        cpu_kv_cache_blocks_used=(
            int(cpu_blocks_used) if cpu_blocks_used is not None else None
        ),
        cpu_kv_cache_blocks_total=(
            int(cpu_blocks_total) if cpu_blocks_total is not None else None
        ),
        raw_metrics=raw,
    )


def send_completion(
    endpoint: str, model: str, prompt: str, max_tokens: int = 1,
) -> float:
    """Send a completion request. Returns TTFT in ms."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model, "prompt": prompt,
            "max_tokens": max_tokens, "temperature": 0.0,
        },
        timeout=300.0,
    )
    ttft = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    return ttft


@dataclass
class PhaseResult:
    phase: str
    num_requests: int
    gpu_cache_before_pct: float | None
    gpu_cache_after_pct: float | None
    gpu_delta_pct: float | None
    gpu_blocks_before: int | None
    gpu_blocks_after: int | None
    cpu_blocks_used: int | None
    cpu_blocks_total: int | None
    ttft_p50_ms: float | None
    ttft_mean_ms: float | None


@dataclass
class MemoryReport:
    model: str
    num_donors: int
    target_tokens: int
    max_contexts: int
    baseline: PhaseResult
    donor_registration: PhaseResult
    semblend_queries: PhaseResult
    gpu_memory_saved_pct: float | None
    gpu_blocks_saved: int | None
    capacity_ratio: float | None
    estimated_gpu_pct_per_donor: float | None
    estimated_max_donors: int | None
    snapshots: list = field(default_factory=list)


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "---"
    # vLLM reports cache usage as a fraction (0.0-1.0) or percentage
    if val <= 1.0:
        return f"{val:.1%}"
    return f"{val:.1f}%"


def _fmt(val, f_str: str) -> str:
    return f_str.format(val) if val is not None else "---"


def _safe_sub(a: float | None, b: float | None) -> float | None:
    if a is not None and b is not None:
        return a - b
    return None


def _median(vals: list[float]) -> float | None:
    if not vals:
        return None
    return statistics.median(vals)


def _mean(vals: list[float]) -> float | None:
    if not vals:
        return None
    return statistics.mean(vals)


def run_memory_bench(
    endpoint: str,
    model: str,
    num_donors: int,
    target_tokens: int = 4096,
    max_contexts: int = 20,
) -> MemoryReport:
    """Measure GPU memory impact of SemBlend donor sharing."""
    uid = 40000

    # --- Baseline snapshot ---
    snap_initial = scrape_gpu_metrics(endpoint)
    all_snapshots = [("initial", asdict(snap_initial))]
    print(f"  Initial GPU cache: {_fmt_pct(snap_initial.gpu_cache_usage_pct)}")

    # --- Phase 1: Baseline requests (no donors, unique prompts) ---
    print(
        f"\n  Phase 1: Baseline "
        f"({max_contexts} unique contexts, no donors)..."
    )
    baseline_ttfts: list[float] = []
    for i in range(max_contexts):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        prompt = build_prompt(ctx, uid, target_tokens, model)
        uid += 1
        try:
            ttft = send_completion(endpoint, model, prompt, max_tokens=5)
            baseline_ttfts.append(ttft)
        except Exception as e:
            print(f"    [{i + 1}] FAILED: {e}")
            break

        if (i + 1) % 5 == 0:
            snap = scrape_gpu_metrics(endpoint)
            all_snapshots.append((f"baseline_{i + 1}", asdict(snap)))
            print(
                f"    [{i + 1}/{max_contexts}] "
                f"GPU: {_fmt_pct(snap.gpu_cache_usage_pct)}  "
                f"TTFT: {ttft:.0f}ms"
            )
        time.sleep(0.1)

    snap_after_baseline = scrape_gpu_metrics(endpoint)
    all_snapshots.append(("after_baseline", asdict(snap_after_baseline)))

    baseline_result = PhaseResult(
        phase="baseline",
        num_requests=len(baseline_ttfts),
        gpu_cache_before_pct=snap_initial.gpu_cache_usage_pct,
        gpu_cache_after_pct=snap_after_baseline.gpu_cache_usage_pct,
        gpu_delta_pct=_safe_sub(
            snap_after_baseline.gpu_cache_usage_pct,
            snap_initial.gpu_cache_usage_pct,
        ),
        gpu_blocks_before=snap_initial.gpu_kv_cache_blocks_used,
        gpu_blocks_after=snap_after_baseline.gpu_kv_cache_blocks_used,
        cpu_blocks_used=snap_after_baseline.cpu_kv_cache_blocks_used,
        cpu_blocks_total=snap_after_baseline.cpu_kv_cache_blocks_total,
        ttft_p50_ms=_median(baseline_ttfts),
        ttft_mean_ms=_mean(baseline_ttfts),
    )

    time.sleep(2)

    # --- Phase 2: Donor registration ---
    snap_pre_donors = scrape_gpu_metrics(endpoint)
    all_snapshots.append(("pre_donors", asdict(snap_pre_donors)))

    print(f"\n  Phase 2: Registering {num_donors} donors...")
    donor_ttfts: list[float] = []
    for i in range(num_donors):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        prompt = build_prompt(ctx, uid, target_tokens, model)
        uid += 1
        try:
            ttft = send_completion(endpoint, model, prompt, max_tokens=1)
            donor_ttfts.append(ttft)
        except Exception as e:
            print(f"    [{i + 1}] FAILED: {e}")

        if (i + 1) % 10 == 0:
            snap = scrape_gpu_metrics(endpoint)
            all_snapshots.append((f"donors_{i + 1}", asdict(snap)))
            print(
                f"    [{i + 1}/{num_donors}] "
                f"GPU: {_fmt_pct(snap.gpu_cache_usage_pct)}  "
                f"TTFT: {ttft:.0f}ms"
            )
        time.sleep(0.1)

    snap_after_donors = scrape_gpu_metrics(endpoint)
    all_snapshots.append(("after_donors", asdict(snap_after_donors)))

    donor_result = PhaseResult(
        phase="donor_registration",
        num_requests=len(donor_ttfts),
        gpu_cache_before_pct=snap_pre_donors.gpu_cache_usage_pct,
        gpu_cache_after_pct=snap_after_donors.gpu_cache_usage_pct,
        gpu_delta_pct=_safe_sub(
            snap_after_donors.gpu_cache_usage_pct,
            snap_pre_donors.gpu_cache_usage_pct,
        ),
        gpu_blocks_before=snap_pre_donors.gpu_kv_cache_blocks_used,
        gpu_blocks_after=snap_after_donors.gpu_kv_cache_blocks_used,
        cpu_blocks_used=snap_after_donors.cpu_kv_cache_blocks_used,
        cpu_blocks_total=snap_after_donors.cpu_kv_cache_blocks_total,
        ttft_p50_ms=_median(donor_ttfts),
        ttft_mean_ms=_mean(donor_ttfts),
    )

    print(f"\n  After donors:")
    print(
        f"    GPU cache: {_fmt_pct(snap_pre_donors.gpu_cache_usage_pct)} "
        f"-> {_fmt_pct(snap_after_donors.gpu_cache_usage_pct)}"
    )

    time.sleep(3)  # Let donor store settle

    # --- Phase 3: SemBlend queries (should reuse donor KV) ---
    snap_pre_queries = scrape_gpu_metrics(endpoint)
    all_snapshots.append(("pre_semblend", asdict(snap_pre_queries)))

    num_queries = min(max_contexts, num_donors)
    print(f"\n  Phase 3: SemBlend queries ({num_queries} runs)...")
    sb_ttfts: list[float] = []
    for i in range(num_queries):
        ctx = CONTEXTS[i % len(CONTEXTS)]
        prompt = build_prompt(ctx, uid, target_tokens, model)
        uid += 1
        try:
            ttft = send_completion(endpoint, model, prompt, max_tokens=5)
            sb_ttfts.append(ttft)
            print(f"    [{i + 1}] {ttft:.0f}ms")
        except Exception as e:
            print(f"    [{i + 1}] FAILED: {e}")
        time.sleep(0.1)

    snap_after_queries = scrape_gpu_metrics(endpoint)
    all_snapshots.append(("after_semblend", asdict(snap_after_queries)))

    semblend_result = PhaseResult(
        phase="semblend_queries",
        num_requests=len(sb_ttfts),
        gpu_cache_before_pct=snap_pre_queries.gpu_cache_usage_pct,
        gpu_cache_after_pct=snap_after_queries.gpu_cache_usage_pct,
        gpu_delta_pct=_safe_sub(
            snap_after_queries.gpu_cache_usage_pct,
            snap_pre_queries.gpu_cache_usage_pct,
        ),
        gpu_blocks_before=snap_pre_queries.gpu_kv_cache_blocks_used,
        gpu_blocks_after=snap_after_queries.gpu_kv_cache_blocks_used,
        cpu_blocks_used=snap_after_queries.cpu_kv_cache_blocks_used,
        cpu_blocks_total=snap_after_queries.cpu_kv_cache_blocks_total,
        ttft_p50_ms=_median(sb_ttfts),
        ttft_mean_ms=_mean(sb_ttfts),
    )

    # --- Derived metrics ---
    gpu_saved = _safe_sub(
        baseline_result.gpu_delta_pct,
        semblend_result.gpu_delta_pct,
    )
    gpu_blocks_saved = None
    if (baseline_result.gpu_blocks_after is not None
            and semblend_result.gpu_blocks_after is not None):
        gpu_blocks_saved = (
            baseline_result.gpu_blocks_after
            - semblend_result.gpu_blocks_after
        )

    capacity_ratio = None
    if (baseline_result.num_requests > 0
            and semblend_result.num_requests > 0):
        capacity_ratio = (
            semblend_result.num_requests / baseline_result.num_requests
        )

    # Per-donor memory cost estimate
    est_pct_per_donor = None
    est_max_donors = None
    if (donor_result.gpu_delta_pct is not None
            and len(donor_ttfts) > 0
            and donor_result.gpu_delta_pct > 0):
        est_pct_per_donor = donor_result.gpu_delta_pct / len(donor_ttfts)
        if (snap_initial.gpu_cache_usage_pct is not None
                and est_pct_per_donor > 0):
            available = 1.0 - snap_initial.gpu_cache_usage_pct
            est_max_donors = int(available / est_pct_per_donor)

    return MemoryReport(
        model=model,
        num_donors=num_donors,
        target_tokens=target_tokens,
        max_contexts=max_contexts,
        baseline=baseline_result,
        donor_registration=donor_result,
        semblend_queries=semblend_result,
        gpu_memory_saved_pct=gpu_saved,
        gpu_blocks_saved=gpu_blocks_saved,
        capacity_ratio=capacity_ratio,
        estimated_gpu_pct_per_donor=est_pct_per_donor,
        estimated_max_donors=est_max_donors,
        snapshots=all_snapshots,
    )


def main():
    parser = argparse.ArgumentParser(
        description="SemBlend memory savings benchmark (Sprint 1)",
    )
    parser.add_argument("--endpoint", default="http://localhost:8001")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--num-donors", type=int, default=10)
    parser.add_argument(
        "--max-contexts", type=int, default=20,
        help="Maximum contexts to send per measurement phase",
    )
    parser.add_argument(
        "--target-tokens", type=int, default=4096,
        help="Target prompt length in tokens",
    )
    parser.add_argument(
        "--output-dir", default="/tmp/semblend-memory-results",
    )
    args = parser.parse_args()

    print("SemBlend Memory Savings Benchmark (Sprint 1)")
    print(f"  Model: {args.model}")
    print(f"  Num donors: {args.num_donors}")
    print(f"  Target tokens: {args.target_tokens}")
    print(f"  Max contexts per phase: {args.max_contexts}")
    print()

    # Health check
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        if resp.status_code != 200:
            print(f"vLLM unhealthy: {resp.status_code}")
            return
        print(f"  vLLM healthy at {args.endpoint}")
    except Exception as e:
        print(f"Cannot reach vLLM: {e}")
        return

    # Verify /metrics accessible
    try:
        resp = requests.get(f"{args.endpoint}/metrics", timeout=10)
        if resp.status_code != 200:
            print(f"  WARNING: /metrics returned {resp.status_code}")
        else:
            print(f"  /metrics endpoint accessible")
    except Exception as e:
        print(f"  WARNING: /metrics not accessible: {e}")

    print()

    report = run_memory_bench(
        endpoint=args.endpoint,
        model=args.model,
        num_donors=args.num_donors,
        target_tokens=args.target_tokens,
        max_contexts=args.max_contexts,
    )

    # --- Summary table ---
    print(f"\n{'=' * 75}")
    print("MEMORY SAVINGS SUMMARY")
    print(f"{'=' * 75}")
    print(
        f"{'Phase':<22} {'GPU Before':>11} {'GPU After':>10} "
        f"{'Delta':>8} {'Requests':>9} {'TTFT p50':>9}"
    )
    print("-" * 75)

    for phase in [
        report.baseline, report.donor_registration, report.semblend_queries,
    ]:
        delta_str = _fmt(phase.gpu_delta_pct, "{:+.4f}")
        print(
            f"{phase.phase:<22} "
            f"{_fmt_pct(phase.gpu_cache_before_pct):>11} "
            f"{_fmt_pct(phase.gpu_cache_after_pct):>10} "
            f"{delta_str:>8} "
            f"{phase.num_requests:>9} "
            f"{_fmt(phase.ttft_p50_ms, '{:.0f}ms'):>9}"
        )

    print(f"\n  GPU KV CACHE BLOCKS:")
    for phase in [
        report.baseline, report.donor_registration, report.semblend_queries,
    ]:
        print(
            f"    {phase.phase:<22} "
            f"before={_fmt(phase.gpu_blocks_before, '{}')} "
            f"after={_fmt(phase.gpu_blocks_after, '{}')} "
            f"cpu_used={_fmt(phase.cpu_blocks_used, '{}')} "
            f"cpu_total={_fmt(phase.cpu_blocks_total, '{}')}"
        )

    print(f"\n  DERIVED METRICS:")
    print(
        f"    GPU memory saved (baseline - semblend delta): "
        f"{_fmt(report.gpu_memory_saved_pct, '{:.4f}')}"
    )
    print(f"    GPU blocks saved: {_fmt(report.gpu_blocks_saved, '{}')}")
    print(f"    Capacity ratio: {_fmt(report.capacity_ratio, '{:.2f}x')}")
    print(
        f"    Est. GPU % per donor: "
        f"{_fmt(report.estimated_gpu_pct_per_donor, '{:.6f}')}"
    )
    print(
        f"    Est. max donors in GPU cache: "
        f"{_fmt(report.estimated_max_donors, '{}')}"
    )

    # Speedup from donor reuse
    if (report.baseline.ttft_p50_ms is not None
            and report.semblend_queries.ttft_p50_ms is not None
            and report.semblend_queries.ttft_p50_ms > 0):
        speedup = (
            report.baseline.ttft_p50_ms
            / report.semblend_queries.ttft_p50_ms
        )
        print(f"    TTFT speedup (baseline vs semblend): {speedup:.2f}x")

    print(f"{'=' * 75}")

    # --- Save JSON ---
    run_id = (
        f"semblend-memory-"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    )
    output = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **asdict(report),
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, f"{run_id}.json")
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    main()
