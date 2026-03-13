#!/usr/bin/env python3
"""RoPE forced-delta E2E benchmark — proves correction necessity at Δ≠0.

Artificially injects RoPE position offset (Δ) into K values AFTER LMCache
loads donor KV, then measures quality with and without correction.

Three conditions per Δ value:
  baseline:    SEMBLEND_FORCE_DELTA=0        → normal SemBlend (Δ=0)
  uncorrected: SEMBLEND_FORCE_DELTA=Δ        → K corrupted, NOT fixed
  corrected:   SEMBLEND_FORCE_DELTA=Δ + _CORRECT=1 → K corrupted THEN fixed

Expected results:
  - baseline PPL ratio ≈ 1.0 (same as normal SemBlend)
  - uncorrected PPL ratio >> 1.0 (quality degrades with Δ)
  - corrected PPL ratio ≈ 1.0 (correction restores quality)

Success criteria (from program.md Gap 2):
  At Δ≥256, uncorrected PPL > 1.10, corrected PPL < 1.05.

Modes:
  --mode run       Run one condition, save to JSON. Requires --condition.
  --mode auto      Automated: runs all conditions via kubectl (default).
  --mode compare   Load saved results and print comparison table.

Usage:
    # Automated full run (handles kubectl set env + rollout restart):
    python -m benchmarks.e2e.rope_force_delta_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --mode auto --deltas 256,1024 --runs 16

    # Single condition (env vars already set on pod):
    python -m benchmarks.e2e.rope_force_delta_bench \\
        --endpoint http://localhost:8100 \\
        --condition baseline --runs 16

    # Compare saved results:
    python -m benchmarks.e2e.rope_force_delta_bench --mode compare
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

try:
    import requests
except ImportError:
    print("ERROR: pip install requests", file=sys.stderr)
    sys.exit(1)


# Diverse context paragraphs — each sample uses a UNIQUE context
# to prevent cross-batch LMCache contamination.
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
        "for rolling updates. Services provide stable networking."
    ),
    (
        "Distributed caching architectures form the backbone of high "
        "performance systems. Redis provides in-memory key-value storage "
        "with lists, sets, sorted sets, hashes, streams. Redis Cluster "
        "enables horizontal scaling via hash slot partitioning."
    ),
    (
        "Cloud GPU computing economics drive architectural decisions. T4 "
        "instances cost $0.53/hour with 16GB HBM and 320 GB/s bandwidth. "
        "A10G costs $1.01/hour with 24GB and 600 GB/s. KV cache memory "
        "grows linearly with sequence length."
    ),
    (
        "Quantum computing leverages quantum mechanical phenomena for "
        "computation. Qubits exist in superposition of states unlike "
        "classical bits. Entanglement creates correlated qubit pairs. "
        "Quantum gates manipulate qubit states through unitary operations."
    ),
    (
        "Database systems provide persistent structured data storage. "
        "Relational databases use SQL for querying normalized tables. "
        "B-tree indexes enable logarithmic lookup time. Write-ahead "
        "logging ensures ACID transactions survive crashes."
    ),
    (
        "Computer networking follows the OSI seven layer model. Physical "
        "layer handles bit transmission. Data link manages frame delivery. "
        "Network layer routes packets via IP addressing. Transport layer "
        "ensures reliable delivery through TCP."
    ),
    (
        "Operating systems manage hardware resources for applications. "
        "Process scheduling algorithms include round-robin, priority-based, "
        "and completely fair scheduler. Virtual memory maps logical to "
        "physical addresses using page tables."
    ),
    (
        "Compiler design involves multiple transformation stages from source "
        "code to machine instructions. Lexical analysis produces tokens from "
        "character streams. Parsing builds abstract syntax trees."
    ),
    (
        "Cryptographic hash functions map arbitrary input to fixed-size output "
        "with collision resistance and preimage resistance. SHA-256 produces "
        "256-bit digests used in blockchain proof of work."
    ),
    (
        "Distributed consensus protocols enable fault-tolerant agreement across "
        "unreliable networks. Paxos achieves safety with majority quorums. Raft "
        "simplifies leader election and log replication."
    ),
    (
        "Graph databases model relationships as first-class citizens. Property "
        "graphs store key-value pairs on nodes and edges. Cypher query language "
        "uses pattern matching for traversal."
    ),
    (
        "Microservices architecture decomposes monolithic applications into small "
        "independently deployable services. Each service owns its data store. "
        "Inter-service communication uses REST, gRPC, or message queues."
    ),
    (
        "Reinforcement learning agents learn optimal policies through trial "
        "and error interaction with environments. Q-learning updates action-value "
        "estimates from temporal difference errors. Policy gradient methods "
        "directly optimize the expected return."
    ),
    (
        "Computer vision applies deep learning to image understanding. "
        "Convolutional neural networks extract hierarchical spatial features. "
        "Object detection combines classification and localization. Semantic "
        "segmentation assigns labels to every pixel."
    ),
    (
        "Information retrieval systems enable efficient document search. "
        "Inverted indexes map terms to document lists. TF-IDF weights combine "
        "term frequency and inverse document frequency. BM25 extends TF-IDF "
        "with document length normalization."
    ),
    (
        "Functional programming emphasizes immutable data and pure functions. "
        "Higher-order functions accept and return functions. Monads chain "
        "computations with side effects. Pattern matching provides concise "
        "control flow. Lazy evaluation defers computation until needed."
    ),
    (
        "Robotics integrates perception, planning, and control. Simultaneous "
        "localization and mapping creates environment models while tracking "
        "position. Motion planning finds collision-free paths through "
        "configuration space. PID controllers stabilize actuator output."
    ),
    (
        "Natural language generation systems produce human-readable text from "
        "structured data. Beam search explores multiple hypotheses during "
        "autoregressive decoding. Nucleus sampling truncates the probability "
        "distribution for diverse outputs."
    ),
    (
        "Time series analysis studies temporal data patterns. Autoregressive "
        "models predict future values from past observations. Fourier transforms "
        "decompose signals into frequency components. Change point detection "
        "identifies regime shifts in streaming data."
    ),
]


@dataclass
class Sample:
    """One measurement: cold vs SemBlend quality."""
    context_idx: int
    cold_ppl: float | None
    semblend_ppl: float | None
    ppl_ratio: float | None
    cold_ttft_ms: float
    semblend_ttft_ms: float
    speedup: float
    cold_text: str
    semblend_text: str


@dataclass
class ConditionResult:
    """Results for one condition (baseline/uncorrected/corrected)."""
    condition: str
    force_delta: int
    force_correct: bool
    model: str
    timestamp: str
    samples: list[Sample] = field(default_factory=list)
    ppl_ratio_mean: float | None = None
    ppl_ratio_median: float | None = None
    ppl_ratio_stdev: float | None = None
    speedup_mean: float | None = None
    num_valid: int = 0


def build_prompt(context: str, uid: str, target_chars: int = 32000) -> str:
    """Build a prompt of ~8K tokens using repeated context."""
    prefix = (
        f"<|im_start|>system\nYou are a helpful assistant. Ref: {uid}.<|im_end|>\n"
        f"<|im_start|>user\nContext:\n"
    )
    suffix = (
        f"\nQuestion: Summarize the key concepts above in detail."
        "<|im_end|>\n<|im_start|>assistant\n"
    )

    # Repeat context to reach target length
    body = context
    while len(body) < target_chars:
        body = body + " " + context

    # Trim to target
    if len(body) > target_chars:
        body = body[:target_chars]

    return prefix + body + suffix


def generate_with_logprobs(
    endpoint: str, model: str, prompt: str,
    max_tokens: int = 50, timeout: float = 300.0,
) -> tuple[str, list[float], float]:
    """Generate text with logprobs. Returns (text, logprobs, ttft_ms)."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": 1,
        },
        timeout=timeout,
    )
    ttft_ms = (time.monotonic() - t0) * 1000
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0].get("text", "")
    lp_data = data["choices"][0].get("logprobs", {})
    token_lps = lp_data.get("token_logprobs", []) if lp_data else []
    valid_lps = [lp for lp in token_lps if lp is not None]
    return text, valid_lps, ttft_ms


def compute_ppl(logprobs: list[float]) -> float | None:
    """Compute perplexity from token log probs."""
    if not logprobs:
        return None
    return math.exp(-statistics.mean(logprobs))


def health_check(endpoint: str) -> bool:
    """Check vLLM health."""
    try:
        resp = requests.get(f"{endpoint}/health", timeout=10)
        return resp.status_code == 200
    except Exception:
        return False


def wait_for_health(endpoint: str, timeout_s: int = 300) -> bool:
    """Wait for vLLM to become healthy."""
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout_s:
        if health_check(endpoint):
            return True
        time.sleep(5)
    return False


def kubectl_set_env(deploy: str, namespace: str, env_vars: dict[str, str]) -> bool:
    """Set env vars on a deployment and restart."""
    env_args = [f"{k}={v}" for k, v in env_vars.items()]
    cmd = ["kubectl", "set", "env", f"deploy/{deploy}", "-n", namespace] + env_args
    print(f"  kubectl: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  kubectl set env failed: {result.stderr}", file=sys.stderr)
        return False

    # Rollout restart
    cmd2 = ["kubectl", "rollout", "restart", f"deploy/{deploy}", "-n", namespace]
    result2 = subprocess.run(cmd2, capture_output=True, text=True)
    if result2.returncode != 0:
        print(f"  rollout restart failed: {result2.stderr}", file=sys.stderr)
        return False

    print("  Waiting for rollout...")
    cmd3 = [
        "kubectl", "rollout", "status", f"deploy/{deploy}",
        "-n", namespace, "--timeout=300s",
    ]
    result3 = subprocess.run(cmd3, capture_output=True, text=True)
    return result3.returncode == 0


def run_condition(
    endpoint: str,
    model: str,
    condition: str,
    force_delta: int,
    force_correct: bool,
    num_runs: int = 16,
    max_tokens: int = 50,
    target_chars: int = 32000,
) -> ConditionResult:
    """Run quality measurements for one condition."""
    result = ConditionResult(
        condition=condition,
        force_delta=force_delta,
        force_correct=force_correct,
        model=model,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    print(f"\n{'='*60}")
    print(f"Condition: {condition} (FORCE_DELTA={force_delta}, CORRECT={force_correct})")
    print(f"Runs: {num_runs}, max_tokens: {max_tokens}")
    print(f"{'='*60}")

    if not wait_for_health(endpoint, timeout_s=30):
        print(f"ERROR: vLLM not healthy at {endpoint}")
        return result

    for i in range(num_runs):
        ctx_idx = i % len(CONTEXTS)
        context = CONTEXTS[ctx_idx]

        # Unique prompt IDs to avoid cross-sample LMCache contamination
        uid_cold = f"force-delta-cold-{condition}-{force_delta}-{i}-{int(time.time())}"
        uid_sb = f"force-delta-sb-{condition}-{force_delta}-{i}-{int(time.time())}"

        prompt_cold = build_prompt(context, uid_cold, target_chars)
        prompt_sb = build_prompt(context, uid_sb, target_chars)

        print(f"  [{i+1}/{num_runs}] ctx={ctx_idx}", end=" ", flush=True)

        try:
            # 1. Cold baseline (no donor match)
            cold_text, cold_lps, cold_ttft = generate_with_logprobs(
                endpoint, model, prompt_cold, max_tokens,
            )
            cold_ppl = compute_ppl(cold_lps)

            # 2. Register donor (same context, different uid → semantic match)
            # Use prompt_cold as donor — SemBlend will match prompt_sb to it
            time.sleep(0.3)

            # 3. SemBlend path (should match donor from step 1)
            sb_text, sb_lps, sb_ttft = generate_with_logprobs(
                endpoint, model, prompt_sb, max_tokens,
            )
            sb_ppl = compute_ppl(sb_lps)

            ppl_ratio = None
            if cold_ppl and sb_ppl and cold_ppl > 0:
                ppl_ratio = sb_ppl / cold_ppl

            speedup = cold_ttft / sb_ttft if sb_ttft > 0 else 0

            sample = Sample(
                context_idx=ctx_idx,
                cold_ppl=cold_ppl,
                semblend_ppl=sb_ppl,
                ppl_ratio=ppl_ratio,
                cold_ttft_ms=cold_ttft,
                semblend_ttft_ms=sb_ttft,
                speedup=speedup,
                cold_text=cold_text[:200],
                semblend_text=sb_text[:200],
            )
            result.samples.append(sample)

            ppl_str = f"ppl_r={ppl_ratio:.3f}" if ppl_ratio else "N/A"
            print(
                f"cold={cold_ttft:.0f}ms sb={sb_ttft:.0f}ms "
                f"spd={speedup:.2f}x {ppl_str}"
            )

        except Exception as e:
            print(f"FAILED: {e}")
            continue

        time.sleep(0.2)

    # Aggregate
    valid_ratios = [
        s.ppl_ratio for s in result.samples
        if s.ppl_ratio is not None
    ]
    valid_speedups = [s.speedup for s in result.samples]

    if valid_ratios:
        result.ppl_ratio_mean = statistics.mean(valid_ratios)
        result.ppl_ratio_median = statistics.median(valid_ratios)
        result.ppl_ratio_stdev = (
            statistics.stdev(valid_ratios) if len(valid_ratios) > 1 else 0.0
        )
    if valid_speedups:
        result.speedup_mean = statistics.mean(valid_speedups)
    result.num_valid = len(valid_ratios)

    print(f"\n  Results: n={result.num_valid}")
    if result.ppl_ratio_mean is not None:
        print(
            f"  PPL ratio: mean={result.ppl_ratio_mean:.4f} "
            f"median={result.ppl_ratio_median:.4f} "
            f"std={result.ppl_ratio_stdev:.4f}"
        )
    if result.speedup_mean is not None:
        print(f"  Speedup: mean={result.speedup_mean:.2f}x")

    return result


def run_auto(
    endpoint: str,
    model: str,
    deltas: list[int],
    num_runs: int,
    max_tokens: int,
    target_chars: int,
    deploy: str,
    namespace: str,
    output_dir: str,
) -> None:
    """Automated: run all conditions with kubectl env var changes."""
    os.makedirs(output_dir, exist_ok=True)
    all_results: list[ConditionResult] = []

    conditions = [("baseline", 0, False)]
    for d in deltas:
        conditions.append((f"uncorrected_d{d}", d, False))
        conditions.append((f"corrected_d{d}", d, True))

    for cond_name, delta, correct in conditions:
        print(f"\n{'#'*60}")
        print(f"# Setting up condition: {cond_name}")
        print(f"{'#'*60}")

        env_vars = {
            "SEMBLEND_FORCE_DELTA": str(delta),
            "SEMBLEND_FORCE_DELTA_CORRECT": "1" if correct else "0",
        }

        if not kubectl_set_env(deploy, namespace, env_vars):
            print(f"FAILED to set env vars for {cond_name}, skipping")
            continue

        # Wait for pod to be ready (model load ~2-3 min)
        print(f"  Waiting for vLLM to become healthy...")
        if not wait_for_health(endpoint, timeout_s=360):
            print(f"  vLLM did not become healthy within 6 min, skipping")
            continue
        # Extra wait for model stabilization
        time.sleep(10)

        result = run_condition(
            endpoint=endpoint,
            model=model,
            condition=cond_name,
            force_delta=delta,
            force_correct=correct,
            num_runs=num_runs,
            max_tokens=max_tokens,
            target_chars=target_chars,
        )
        all_results.append(result)

        # Save individual result
        out_file = os.path.join(output_dir, f"{cond_name}.json")
        with open(out_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"  Saved: {out_file}")

    # Print comparison
    if len(all_results) > 1:
        _print_comparison(all_results)

    # Save combined results
    combined = os.path.join(output_dir, "all_conditions.json")
    with open(combined, "w") as f:
        json.dump([asdict(r) for r in all_results], f, indent=2, default=str)
    print(f"\nAll results saved to: {combined}")

    # Restore baseline (FORCE_DELTA=0) after all runs
    print("\nRestoring baseline (FORCE_DELTA=0)...")
    kubectl_set_env(deploy, namespace, {
        "SEMBLEND_FORCE_DELTA": "0",
        "SEMBLEND_FORCE_DELTA_CORRECT": "0",
    })


def _print_comparison(results: list[ConditionResult]) -> None:
    """Print paper-ready comparison table."""
    print(f"\n{'='*80}")
    print("RoPE FORCED-DELTA E2E RESULTS")
    print(f"{'='*80}")
    print(
        f"{'Condition':<25} {'Delta':>6} {'Correct':>8} {'N':>4} "
        f"{'PPL Ratio':>10} {'Std':>8} {'Speedup':>8}"
    )
    print("-" * 80)

    for r in results:
        ppl_str = f"{r.ppl_ratio_mean:.4f}" if r.ppl_ratio_mean else "---"
        std_str = f"{r.ppl_ratio_stdev:.4f}" if r.ppl_ratio_stdev else "---"
        spd_str = f"{r.speedup_mean:.2f}x" if r.speedup_mean else "---"
        print(
            f"{r.condition:<25} {r.force_delta:>6} "
            f"{'yes' if r.force_correct else 'no':>8} {r.num_valid:>4} "
            f"{ppl_str:>10} {std_str:>8} {spd_str:>8}"
        )

    # Paper-ready LaTeX-compatible table
    print(f"\n{'='*80}")
    print("PAPER TABLE: RoPE Correction at Forced Position Offset")
    print(f"{'='*80}")
    print(f"{'Condition':<20} | {'PPL Ratio':>10} | {'Verdict':>20}")
    print("-" * 60)

    for r in results:
        if r.ppl_ratio_mean is None:
            continue
        if r.condition == "baseline":
            verdict = "baseline"
        elif not r.force_correct and r.force_delta > 0:
            verdict = "DEGRADED" if r.ppl_ratio_mean > 1.10 else "minimal impact"
        else:
            verdict = "RESTORED" if r.ppl_ratio_mean < 1.05 else "partial restore"
        print(f"{r.condition:<20} | {r.ppl_ratio_mean:>10.4f} | {verdict:>20}")

    # Check success criteria
    baseline = next((r for r in results if r.condition == "baseline"), None)
    uncorrected = [r for r in results if not r.force_correct and r.force_delta > 0]
    corrected = [r for r in results if r.force_correct and r.force_delta > 0]

    print(f"\n--- Gap 2 Success Criteria ---")
    if uncorrected:
        for u in uncorrected:
            if u.ppl_ratio_mean and u.ppl_ratio_mean > 1.10:
                print(f"  PASS: uncorrected Δ={u.force_delta} PPL={u.ppl_ratio_mean:.3f} > 1.10")
            elif u.ppl_ratio_mean:
                print(f"  FAIL: uncorrected Δ={u.force_delta} PPL={u.ppl_ratio_mean:.3f} <= 1.10")
    if corrected:
        for c in corrected:
            if c.ppl_ratio_mean and c.ppl_ratio_mean < 1.05:
                print(f"  PASS: corrected Δ={c.force_delta} PPL={c.ppl_ratio_mean:.3f} < 1.05")
            elif c.ppl_ratio_mean:
                print(f"  FAIL: corrected Δ={c.force_delta} PPL={c.ppl_ratio_mean:.3f} >= 1.05")


def run_compare(output_dir: str) -> None:
    """Load saved results and print comparison."""
    combined_path = os.path.join(output_dir, "all_conditions.json")
    if os.path.exists(combined_path):
        with open(combined_path) as f:
            data = json.load(f)
        results = [
            ConditionResult(
                condition=d["condition"],
                force_delta=d["force_delta"],
                force_correct=d["force_correct"],
                model=d["model"],
                timestamp=d["timestamp"],
                ppl_ratio_mean=d.get("ppl_ratio_mean"),
                ppl_ratio_median=d.get("ppl_ratio_median"),
                ppl_ratio_stdev=d.get("ppl_ratio_stdev"),
                speedup_mean=d.get("speedup_mean"),
                num_valid=d.get("num_valid", 0),
            )
            for d in data
        ]
        _print_comparison(results)
        return

    # Try individual files
    results = []
    for fname in sorted(os.listdir(output_dir)):
        if fname.endswith(".json") and fname != "all_conditions.json":
            with open(os.path.join(output_dir, fname)) as f:
                d = json.load(f)
            results.append(
                ConditionResult(
                    condition=d["condition"],
                    force_delta=d["force_delta"],
                    force_correct=d["force_correct"],
                    model=d["model"],
                    timestamp=d["timestamp"],
                    ppl_ratio_mean=d.get("ppl_ratio_mean"),
                    ppl_ratio_median=d.get("ppl_ratio_median"),
                    ppl_ratio_stdev=d.get("ppl_ratio_stdev"),
                    speedup_mean=d.get("speedup_mean"),
                    num_valid=d.get("num_valid", 0),
                )
            )

    if results:
        _print_comparison(results)
    else:
        print(f"No results found in {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="RoPE forced-delta E2E benchmark",
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument(
        "--mode", choices=["run", "auto", "compare"], default="auto",
    )
    parser.add_argument(
        "--condition",
        choices=["baseline", "uncorrected", "corrected"],
        help="Condition to run (for --mode run)",
    )
    parser.add_argument(
        "--force-delta", type=int, default=256,
        help="Forced RoPE delta value",
    )
    parser.add_argument(
        "--deltas", default="256,1024",
        help="Comma-separated delta values (for --mode auto)",
    )
    parser.add_argument("--runs", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument("--target-chars", type=int, default=32000,
                        help="Target prompt length in characters (~8K tokens)")
    parser.add_argument("--output-dir", default="results/rope-force-delta")
    parser.add_argument("--deploy", default="autoresearch-synapse-vllm",
                        help="Kubernetes deployment name")
    parser.add_argument("--namespace", default="autoresearch")
    args = parser.parse_args()

    if args.mode == "compare":
        run_compare(args.output_dir)
        return

    if args.mode == "run":
        if not args.condition:
            parser.error("--condition required for --mode run")

        delta = args.force_delta if args.condition != "baseline" else 0
        correct = args.condition == "corrected"

        result = run_condition(
            endpoint=args.endpoint,
            model=args.model,
            condition=args.condition,
            force_delta=delta,
            force_correct=correct,
            num_runs=args.runs,
            max_tokens=args.max_tokens,
            target_chars=args.target_chars,
        )

        os.makedirs(args.output_dir, exist_ok=True)
        label = f"{args.condition}_d{delta}"
        out_file = os.path.join(args.output_dir, f"{label}.json")
        with open(out_file, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        print(f"\nSaved: {out_file}")
        return

    # Auto mode
    deltas = [int(d.strip()) for d in args.deltas.split(",")]
    run_auto(
        endpoint=args.endpoint,
        model=args.model,
        deltas=deltas,
        num_runs=args.runs,
        max_tokens=args.max_tokens,
        target_chars=args.target_chars,
        deploy=args.deploy,
        namespace=args.namespace,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
