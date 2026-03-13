#!/usr/bin/env python3
"""SemBlend real-dataset benchmark — MS MARCO RAG scenarios at multiple lengths.

METHODOLOGY: For each (prompt_length, scenario) combination, we generate N
independent prompt instances using different passage selections. Each instance
is measured ONCE — this captures the first-encounter latency where SemBlend
discovers a donor and injects its KV cache. Repeated measurements of the same
prompt would just hit LMCache prefix cache (not SemBlend).

Reports TWO baselines per scenario:
  1. vs cold prefill (completely different passages, no cache benefit)
  2. vs prefix-cached vLLM (same prompt with unique prefix — production-realistic)

Usage:
    python benchmarks/e2e/semblend_real_bench.py \
        --endpoint http://localhost:8001 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --runs 7
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import numpy as np


@dataclass
class Passage:
    id: str
    text: str
    token_estimate: int


@dataclass
class TTFTResult:
    ttft_ms: float
    prompt_tokens: int
    output_text: str
    error: str | None = None


def send_completion(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 50,
    temperature: float = 0.0,
) -> TTFTResult:
    """Send a completion request and measure TTFT via streaming."""
    try:
        with httpx.Client(timeout=120.0) as client:
            t0 = time.monotonic()
            first_token_time = None
            output_chunks = []

            with client.stream(
                "POST",
                f"{endpoint}/v1/completions",
                json={
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": temperature,
                    "stream": True,
                },
            ) as resp:
                if resp.status_code != 200:
                    return TTFTResult(
                        ttft_ms=0, prompt_tokens=0, output_text="",
                        error=f"HTTP {resp.status_code}: {resp.read().decode()[:200]}",
                    )

                for line in resp.iter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                        text = chunk["choices"][0].get("text", "")
                        if text and first_token_time is None:
                            first_token_time = time.monotonic()
                        output_chunks.append(text)
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

            ttft_ms = (
                (first_token_time - t0) * 1000 if first_token_time else 0
            )
            prompt_tokens = int(len(prompt.split()) * 1.3)

            return TTFTResult(
                ttft_ms=ttft_ms,
                prompt_tokens=prompt_tokens,
                output_text="".join(output_chunks),
            )
    except Exception as e:
        return TTFTResult(
            ttft_ms=0, prompt_tokens=0, output_text="",
            error=str(e),
        )


def load_passages(max_passages: int = 5000, seed: int = 42) -> list[Passage]:
    """Load MS MARCO passages from HuggingFace."""
    cache_dir = Path.home() / ".cache" / "semblend_datasets"
    cache_path = cache_dir / "msmarco_passages.jsonl"

    if not cache_path.exists():
        print("Downloading MS MARCO passages...", flush=True)
        try:
            from datasets import load_dataset
            ds = load_dataset(
                "microsoft/ms_marco", "v2.1", split="train",
                streaming=True,
            )
            cache_dir.mkdir(parents=True, exist_ok=True)
            count = 0
            with open(cache_path, "w") as f:
                for item in ds:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    count += 1
                    if count >= 50_000:
                        break
                    if count % 10_000 == 0:
                        print(f"  {count} rows...", flush=True)
            print(f"Downloaded {count} rows", flush=True)
        except Exception as e:
            print(f"Download failed: {e}", file=sys.stderr)
            return _fallback_passages()

    passages = []
    rng = random.Random(seed)
    lines = cache_path.read_text().splitlines()
    rng.shuffle(lines)

    for line in lines:
        if len(passages) >= max_passages:
            break
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue

        passage_list = item.get("passages", {})
        if isinstance(passage_list, dict):
            texts = passage_list.get("passage_text", [])
        elif isinstance(passage_list, list):
            texts = [p.get("passage_text", p.get("text", "")) for p in passage_list]
        else:
            continue

        for text in texts:
            if not text or len(text) < 100:
                continue
            tok_est = int(len(text.split()) * 1.3)
            if tok_est < 50 or tok_est > 500:
                continue
            passages.append(Passage(
                id=f"p{len(passages)}",
                text=text,
                token_estimate=tok_est,
            ))
            if len(passages) >= max_passages:
                break

    print(f"Loaded {len(passages)} passages", flush=True)
    return passages


def _fallback_passages() -> list[Passage]:
    """Simple fallback if MS MARCO unavailable."""
    topics = [
        "Semiconductor manufacturing uses extreme ultraviolet lithography at 13.5nm wavelength. TSMC's N3 node achieves 1.6x logic density over N5. Advanced packaging like CoWoS integrates HBM stacks with 2.5D interposer.",
        "Battery technology advances with CATL's Qilin 3.0 achieving 255 Wh/kg at pack level. Solid-state batteries target 500 Wh/kg by 2027. Sodium-ion cells offer 30% lower cost than LFP chemistry.",
        "Autonomous vehicles use transformer-based perception. Tesla FSD v12 replaces 300K lines of C++ with neural nets. Waymo uses custom TPU v5e for real-time 3D reconstruction from 29 cameras.",
        "Quantum computing hardware diverges across modalities. IBM Heron achieves 133 qubits with 99.5% gate fidelity. Google Willow demonstrates below-threshold error correction with 105 qubits.",
        "AI governance frameworks vary globally. EU AI Act establishes risk-based regulation. US Executive Order requires safety testing for models above 10^26 FLOPs. China mandates content review and algorithm filing.",
        "Cloud computing infrastructure scales with custom silicon. AWS Graviton4 provides 30% better price-performance. Google TPU v5p delivers 459 TFLOPS for ML training workloads.",
        "Cybersecurity threats evolve with AI-powered attacks. Ransomware-as-a-service models enable less sophisticated actors. Zero-trust architecture replaces perimeter-based security models.",
        "Space technology advances with SpaceX Starship's fully reusable design. Blue Origin's New Glenn targets 2025 maiden flight. Commercial space stations from Axiom and Orbital Reef succeed ISS.",
    ]
    return [
        Passage(id=f"fb{i}", text=t, token_estimate=int(len(t.split()) * 1.3))
        for i, t in enumerate(topics * 10)
    ]


SYSTEM_PROMPT = (
    "You are a helpful research assistant. Answer the user's question "
    "based on the provided context documents. Be specific and cite facts."
)

QUESTIONS = [
    "Summarize the key points from the documents.",
    "What are the main findings or conclusions?",
    "What challenges or risks are discussed?",
    "What are the practical implications?",
    "What trends or patterns emerge?",
    "Compare and contrast the approaches discussed.",
    "What evidence supports the claims made?",
    "What are the limitations mentioned?",
]


def build_rag_prompt(passages: list[Passage], question: str) -> str:
    """Build a RAG prompt from passages."""
    parts = [f"[Document {i+1}]\n{p.text}" for i, p in enumerate(passages)]
    context = "\n\n".join(parts)
    return (
        f"System: {SYSTEM_PROMPT}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        f"Answer:"
    )


def estimate_tokens(text: str) -> int:
    return int(len(text.split()) * 1.3)


def bootstrap_ci(data: list[float], n_resamples: int = 10000) -> tuple[float, float, float]:
    """Bootstrap 95% CI for median."""
    arr = np.array(data)
    rng = np.random.default_rng(42)
    medians = [float(np.median(rng.choice(arr, size=len(arr), replace=True)))
               for _ in range(n_resamples)]
    medians.sort()
    return (
        float(np.median(arr)),
        medians[int(0.025 * n_resamples)],
        medians[int(0.975 * n_resamples)],
    )


def main():
    parser = argparse.ArgumentParser(description="SemBlend real dataset benchmark")
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--runs", type=int, default=7,
                        help="Independent prompt instances per scenario")
    parser.add_argument("--max-tokens", type=int, default=50)
    parser.add_argument(
        "--chunks", nargs="+", type=int, default=[8, 16, 24, 32],
        help="Number of passages per prompt (controls length)",
    )
    parser.add_argument("--output", default="results/semblend_real")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    passages = load_passages(max_passages=5000, seed=args.seed)
    if len(passages) < 200:
        print("Not enough passages", file=sys.stderr)
        sys.exit(1)

    # Pipeline warmup: send a throwaway prompt to initialize ONNX/CUDA
    print("Pipeline warmup...", flush=True)
    warmup_p = rng.sample(passages, 8)
    warmup_prompt = build_rag_prompt(warmup_p, "Warmup question.")
    wr = send_completion(args.endpoint, args.model, warmup_prompt, args.max_tokens)
    if wr.error:
        print(f"  Warmup error: {wr.error}")
    else:
        print(f"  Warmup TTFT: {wr.ttft_ms:.0f}ms")
    time.sleep(2.0)

    results = {}
    scenarios = ["EXACT", "REORDER", "PARTIAL_75", "PARTIAL_50", "COLD"]

    for n_chunks in args.chunks:
        needed_passages = n_chunks * (args.runs + 1) * 3
        if len(passages) < needed_passages:
            print(f"Skipping {n_chunks} chunks (need {needed_passages}, have {len(passages)})")
            continue

        print(f"\n{'='*70}")
        print(f"CHUNK COUNT: {n_chunks} passages")
        print(f"{'='*70}")

        chunk_results = {}

        for scenario in scenarios:
            semblend_times = []
            prefixed_baseline_times = []
            cold_baseline_times = []
            quality_matches = []
            tok_estimates = []

            print(f"\n  {scenario} (n={args.runs} independent instances):", flush=True)

            for run_idx in range(args.runs):
                # For each run, create a FRESH seed + test pair using different passages
                # This ensures each measurement is independent (no prefix cache reuse)

                # Partition passages into pools for this run
                pool = rng.sample(passages, n_chunks * 4)
                seed_pool = pool[:n_chunks]
                extra_pool = pool[n_chunks:]

                question = rng.choice(QUESTIONS)
                seed_prompt = build_rag_prompt(seed_pool, question)
                seed_tok = estimate_tokens(seed_prompt)
                tok_estimates.append(seed_tok)

                # 1. Seed the donor store with this prompt
                seed_result = send_completion(
                    args.endpoint, args.model, seed_prompt, args.max_tokens,
                )
                if seed_result.error:
                    print(f"    Run {run_idx}: Seed error: {seed_result.error}")
                    continue
                time.sleep(1.0)

                # 2. Build scenario-specific test prompt
                if scenario == "EXACT":
                    test_prompt = seed_prompt
                elif scenario == "REORDER":
                    reordered = list(seed_pool)
                    for _ in range(10):  # ensure actual reordering
                        rng.shuffle(reordered)
                        if reordered != seed_pool:
                            break
                    test_prompt = build_rag_prompt(reordered, question)
                elif scenario == "PARTIAL_75":
                    n_keep = n_chunks - max(1, n_chunks // 4)
                    kept = seed_pool[:n_keep]
                    new_p = rng.sample(extra_pool, n_chunks - n_keep)
                    test_passages = kept + new_p
                    rng.shuffle(test_passages)
                    test_prompt = build_rag_prompt(test_passages, question)
                elif scenario == "PARTIAL_50":
                    n_keep = n_chunks // 2
                    kept = seed_pool[:n_keep]
                    new_p = rng.sample(extra_pool, n_chunks - n_keep)
                    test_passages = kept + new_p
                    rng.shuffle(test_passages)
                    test_prompt = build_rag_prompt(test_passages, question)
                elif scenario == "COLD":
                    cold_passages = rng.sample(extra_pool, n_chunks)
                    alt_q = rng.choice([q for q in QUESTIONS if q != question])
                    test_prompt = build_rag_prompt(cold_passages, alt_q)
                else:
                    continue

                # 3. SemBlend measurement (first encounter of test prompt)
                sb_result = send_completion(
                    args.endpoint, args.model, test_prompt, args.max_tokens,
                )
                time.sleep(0.5)

                # 4. Prefix-cached baseline (same test content, unique prefix)
                pf_prompt = (
                    f"[bl-{n_chunks}-{scenario}-{run_idx}-{time.monotonic():.6f}] "
                    + test_prompt
                )
                pf_result = send_completion(
                    args.endpoint, args.model, pf_prompt, args.max_tokens,
                )
                time.sleep(0.5)

                # 5. Cold baseline (completely different passages, fresh)
                cold_pool = rng.sample(
                    [p for p in passages if p not in pool],
                    n_chunks,
                )
                cold_q = rng.choice(QUESTIONS)
                cold_prompt = build_rag_prompt(cold_pool, cold_q)
                cl_result = send_completion(
                    args.endpoint, args.model, cold_prompt, args.max_tokens,
                )
                time.sleep(0.5)

                if sb_result.error or pf_result.error or cl_result.error:
                    errs = [r.error for r in [sb_result, pf_result, cl_result] if r.error]
                    print(f"    Run {run_idx}: Error: {errs[0]}")
                    continue

                semblend_times.append(sb_result.ttft_ms)
                prefixed_baseline_times.append(pf_result.ttft_ms)
                cold_baseline_times.append(cl_result.ttft_ms)

                match = sb_result.output_text.strip()[:100] == pf_result.output_text.strip()[:100]
                quality_matches.append(match)

                sp_pf = pf_result.ttft_ms / max(sb_result.ttft_ms, 1)
                sp_cl = cl_result.ttft_ms / max(sb_result.ttft_ms, 1)
                print(
                    f"    Run {run_idx} (~{seed_tok}tok): "
                    f"SB={sb_result.ttft_ms:.0f}ms  "
                    f"PfxBL={pf_result.ttft_ms:.0f}ms ({sp_pf:.2f}x)  "
                    f"ColdBL={cl_result.ttft_ms:.0f}ms ({sp_cl:.2f}x)",
                    flush=True,
                )

            if len(semblend_times) >= 3:
                sb_med, sb_lo, sb_hi = bootstrap_ci(semblend_times)
                pf_med, pf_lo, pf_hi = bootstrap_ci(prefixed_baseline_times)
                cl_med, cl_lo, cl_hi = bootstrap_ci(cold_baseline_times)
                sp_pf = pf_med / max(sb_med, 1)
                sp_cl = cl_med / max(sb_med, 1)
                eq_rate = sum(1 for m in quality_matches if m) / len(quality_matches) * 100
                avg_tok = int(np.mean(tok_estimates))

                print(
                    f"    SUMMARY (~{avg_tok} tokens):\n"
                    f"      SemBlend:    {sb_med:>7.0f}ms [{sb_lo:.0f}, {sb_hi:.0f}]\n"
                    f"      vs Prefix:   {pf_med:>7.0f}ms [{pf_lo:.0f}, {pf_hi:.0f}]  "
                    f"Speedup={sp_pf:.2f}x\n"
                    f"      vs Cold:     {cl_med:>7.0f}ms [{cl_lo:.0f}, {cl_hi:.0f}]  "
                    f"Speedup={sp_cl:.2f}x\n"
                    f"      Quality:     {eq_rate:.0f}% output match"
                )

                chunk_results[scenario] = {
                    "semblend_ms": semblend_times,
                    "prefixed_baseline_ms": prefixed_baseline_times,
                    "cold_baseline_ms": cold_baseline_times,
                    "semblend_median": sb_med,
                    "semblend_ci": [sb_lo, sb_hi],
                    "prefixed_baseline_median": pf_med,
                    "prefixed_baseline_ci": [pf_lo, pf_hi],
                    "cold_baseline_median": cl_med,
                    "cold_baseline_ci": [cl_lo, cl_hi],
                    "speedup_vs_prefixed": sp_pf,
                    "speedup_vs_cold": sp_cl,
                    "quality_match_rate": eq_rate,
                    "avg_tokens": avg_tok,
                    "n": len(semblend_times),
                }

        results[f"{n_chunks}_chunks"] = chunk_results

    # Save results
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    with open(out_dir / f"results_{ts}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / "results_latest.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_dir}/results_{ts}.json")

    # Summary table
    print(f"\n{'='*105}")
    print("SUMMARY TABLE")
    print(f"{'='*105}")
    print(f"{'Chunks':>8s}  {'~Tokens':>8s}  {'Scenario':>12s}  "
          f"{'SemBlend':>10s}  {'vs Prefix':>10s}  {'vs Cold':>10s}  {'Quality':>8s}  n")
    print(f"{'-'*8}  {'-'*8}  {'-'*12}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*8}  -")

    for config_key, scenario_results in results.items():
        for scenario, data in scenario_results.items():
            print(
                f"{config_key:>8s}  {data['avg_tokens']:>7d}   {scenario:>12s}  "
                f"{data['semblend_median']:>8.0f}ms  "
                f"{data['speedup_vs_prefixed']:>8.2f}x   "
                f"{data['speedup_vs_cold']:>8.2f}x   "
                f"{data['quality_match_rate']:>6.0f}%  {data['n']}"
            )


if __name__ == "__main__":
    main()
