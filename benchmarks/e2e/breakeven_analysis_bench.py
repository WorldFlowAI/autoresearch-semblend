#!/usr/bin/env python3
"""Break-even hit-rate analysis: at what hit rate does SemBlend become net-positive?

For each context length, measures three conditions:
  C = cold TTFT    (no SemBlend pipeline, baseline prefill)
  H = hit  TTFT    (SemBlend found a matching donor)
  M = miss TTFT    (SemBlend searched but found nothing)

From these, the break-even hit probability is:
  P_h* = overhead_miss / (overhead_miss + savings_hit)
       = (M - C) / ((M - C) + (C - H))

At any deployment hit rate P_h > P_h*, SemBlend delivers net latency savings.

Usage:
    python -m benchmarks.e2e.breakeven_analysis_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-samples 16 \
        --token-lengths 2048,4096,8192,16384
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n"
    "Document ID: {uid}\n\n"
    "{context}\n\n"
    "Summarize the key points above.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Pool of diverse sentences for generating unique context per prompt.
# Each sentence ~100-120 chars (~25-30 tokens). Using a large pool ensures
# that each prompt's random sampling produces text that does NOT match any
# other prompt at the 256-token chunk level, preventing cross-batch LMCache hits.
_CONTEXT_SENTENCES = [
    "The research committee reviewed the quarterly findings and recommended further investigation into the observed anomalies. ",
    "Tropical rainforests host approximately half of all known species on Earth despite covering less than seven percent of the planet. ",
    "Advances in quantum error correction have brought fault-tolerant quantum computing closer to practical deployment timelines. ",
    "The International Space Station orbits at roughly four hundred kilometers altitude completing sixteen revolutions per day. ",
    "Renewable energy capacity additions exceeded fossil fuel additions globally for the third consecutive year in recent reports. ",
    "Archaeologists discovered a previously unknown trade route connecting ancient Mesopotamian city-states to Indus Valley settlements. ",
    "Machine learning models trained on protein sequences can now predict three-dimensional folding structures with atomic accuracy. ",
    "The deep ocean floor contains vast deposits of polymetallic nodules rich in manganese cobalt nickel and rare earth elements. ",
    "Climate reconstructions from ice cores reveal abrupt temperature shifts occurring within decades during the last glacial period. ",
    "High-temperature superconductors operating above liquid nitrogen temperatures remain an active area of condensed matter research. ",
    "The human gut microbiome contains trillions of bacteria that play essential roles in digestion immunity and mental health regulation. ",
    "Autonomous underwater vehicles equipped with sonar mapping systems survey uncharted seafloor terrain at depths exceeding six kilometers. ",
    "Gravitational wave detectors have confirmed the existence of binary black hole mergers occurring billions of light years from Earth. ",
    "Agricultural soil degradation threatens food security as topsoil erosion rates accelerate beyond natural replenishment capacity worldwide. ",
    "The development of mRNA vaccine platforms demonstrated remarkable adaptability during the recent global pandemic response efforts. ",
    "Satellite constellations in low Earth orbit provide broadband internet connectivity to previously underserved rural and remote communities. ",
    "Volcanic eruptions inject sulfur dioxide particles into the stratosphere creating temporary cooling effects on regional climate patterns. ",
    "Neuroplasticity research shows that adult brains retain the ability to form new neural connections throughout the entire lifespan. ",
    "Coral reef bleaching events have increased in frequency and severity as ocean temperatures rise beyond historical seasonal averages. ",
    "Synthetic biology enables engineering of microorganisms capable of producing pharmaceuticals biofuels and industrial chemicals efficiently. ",
    "The discovery of water ice deposits in permanently shadowed lunar craters has renewed interest in establishing permanent Moon bases. ",
    "Continental drift theory evolved into plate tectonics providing a unified framework for understanding earthquakes volcanoes and mountain formation. ",
    "Electric vehicle battery technology continues advancing with solid-state designs promising higher energy density and faster charging rates. ",
    "Ancient DNA analysis has revolutionized our understanding of human migration patterns and population mixing over the last hundred thousand years. ",
    "Microplastic contamination has been detected in drinking water sources marine sediments atmospheric samples and human blood specimens globally. ",
    "Precision agriculture uses satellite imagery drone surveys and soil sensors to optimize crop yields while minimizing fertilizer and water usage. ",
    "The standard model of particle physics successfully describes three of the four fundamental forces but does not incorporate gravity. ",
    "Geothermal energy systems tap into heat stored deep within the Earth providing reliable baseload power generation with minimal carbon emissions. ",
    "Recent paleontological discoveries suggest that many dinosaur species possessed feathers challenging traditional cold-blooded reptilian depictions. ",
    "Blockchain technology applications have expanded beyond cryptocurrency to include supply chain verification digital identity and voting systems. ",
    "Ocean acidification caused by dissolved carbon dioxide threatens shell-forming marine organisms including oysters mussels and coral polyps. ",
    "The James Webb Space Telescope has captured infrared images of galaxies formed within the first few hundred million years after the Big Bang. ",
    "Epigenetic modifications allow environmental factors to influence gene expression without altering the underlying DNA sequence itself permanently. ",
    "Wildfire intensity and burned area have increased dramatically in Mediterranean climates due to prolonged drought conditions and rising temperatures. ",
    "CRISPR gene editing technology offers potential treatments for genetic disorders including sickle cell disease and certain forms of hereditary blindness. ",
    "Tidal energy converters installed in narrow coastal channels harness predictable ocean currents to generate electricity with high capacity factors. ",
    "The global semiconductor shortage exposed critical vulnerabilities in just-in-time manufacturing supply chains across automotive and electronics industries. ",
    "Permafrost thaw across Arctic regions releases stored methane and carbon dioxide accelerating positive feedback loops in the global climate system. ",
    "Artificial photosynthesis research aims to replicate natural light-harvesting processes for direct solar fuel production from water and carbon dioxide. ",
    "The human genome project completed in 2003 mapped approximately three billion base pairs enabling subsequent advances in personalized medicine approaches. ",
]


def _build_unique_context(target_chars: int, seed: int) -> str:
    """Return seeded pseudo-random text of approximately *target_chars* characters.

    Each seed produces a unique sequence of sentences from _CONTEXT_SENTENCES,
    ensuring no two prompts share the same 256-token chunk content in LMCache.
    """
    import random
    rng = random.Random(seed)
    parts: list[str] = []
    total = 0
    while total < target_chars:
        sentence = rng.choice(_CONTEXT_SENTENCES)
        parts.append(sentence)
        total += len(sentence)
    return "".join(parts)[:target_chars]


def _build_prompt(uid: str, context: str) -> str:
    return _PROMPT_TEMPLATE.format(uid=uid, context=context)


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Send a non-streaming completions request. Returns (latency_ms, ok)."""
    t0 = time.monotonic()
    try:
        resp = requests.post(
            f"{endpoint}/v1/completions",
            json={
                "model": model,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": 0.0,
                "stream": False,
            },
            timeout=300,
        )
        ms = (time.monotonic() - t0) * 1000
        resp.raise_for_status()
        return ms, True
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"    request failed ({elapsed:.0f}ms): {exc}")
        return 0.0, False


def _percentile(values: list[float], pct: int) -> float:
    """Return the *pct*-th percentile from a sorted copy of *values*."""
    s = sorted(values)
    idx = max(0, min(len(s) - 1, int(len(s) * pct / 100)))
    return s[idx]


def run_condition(
    label: str,
    endpoint: str,
    model: str,
    prompts: list[str],
    max_tokens: int = 5,
) -> list[float]:
    """Run a batch of requests and return the list of successful latencies."""
    latencies: list[float] = []
    total = len(prompts)
    for i, prompt in enumerate(prompts):
        t, ok = ttft_request(endpoint, model, prompt, max_tokens=max_tokens)
        tag = f"{t:.0f}ms" if ok else "FAIL"
        print(f"    [{i + 1}/{total}] {label} {tag}    ", end="\r")
        if ok:
            latencies.append(t)
    print()
    return latencies


def analyze_breakeven(
    cold: list[float], hit: list[float], miss: list[float]
) -> dict:
    """Compute break-even statistics from the three measurement vectors."""
    c_mean = statistics.mean(cold) if cold else 0.0
    h_mean = statistics.mean(hit) if hit else 0.0
    m_mean = statistics.mean(miss) if miss else 0.0

    c_p50 = _percentile(cold, 50) if cold else 0.0
    h_p50 = _percentile(hit, 50) if hit else 0.0
    m_p50 = _percentile(miss, 50) if miss else 0.0

    overhead_miss = max(0.0, m_mean - c_mean)
    savings_hit = max(0.0, c_mean - h_mean)

    denominator = overhead_miss + savings_hit
    breakeven_ph = overhead_miss / denominator if denominator > 0 else 1.0

    # Expected speedup at various hit rates
    speedup_at_ph: dict[str, float] = {}
    for ph in [0.25, 0.50, 0.75, 0.90, 1.00]:
        expected_ttft = ph * h_mean + (1.0 - ph) * m_mean
        spd = c_mean / expected_ttft if expected_ttft > 0 else 0.0
        speedup_at_ph[f"{ph:.2f}"] = round(spd, 3)

    return {
        "cold_mean_ms": round(c_mean, 1),
        "cold_p50_ms": round(c_p50, 1),
        "hit_mean_ms": round(h_mean, 1),
        "hit_p50_ms": round(h_p50, 1),
        "miss_mean_ms": round(m_mean, 1),
        "miss_p50_ms": round(m_p50, 1),
        "overhead_miss_ms": round(overhead_miss, 1),
        "savings_hit_ms": round(savings_hit, 1),
        "breakeven_hit_rate": round(breakeven_ph, 4),
        "speedup_at_hit_rate": speedup_at_ph,
        "n_cold": len(cold),
        "n_hit": len(hit),
        "n_miss": len(miss),
    }


_MISS_FILLER_SENTENCE = (
    "Quantum computing breakthroughs enable molecular simulation at "
    "unprecedented scales, transforming pharmaceutical drug discovery. "
)


def _build_miss_context(target_chars: int, seed: int) -> str:
    """Return unique miss-condition text that is topically different from donors."""
    import random
    rng = random.Random(seed)
    topics = [
        "Deep ocean exploration reveals bioluminescent organisms thriving "
        "near hydrothermal vents at extreme pressures and temperatures. ",
        "Ancient astronomical observatories demonstrate sophisticated "
        "mathematical understanding of celestial mechanics and seasonal cycles. ",
        "Sustainable urban agriculture integrates vertical farming technology "
        "with renewable energy systems to maximize food production density. ",
        "Advances in materials science yield self-healing polymers capable "
        "of restoring structural integrity after mechanical damage events. ",
        _MISS_FILLER_SENTENCE,
    ]
    parts = []
    while len("".join(parts)) < target_chars:
        parts.append(rng.choice(topics))
    return "".join(parts)[:target_chars]


def _compute_batch_size(token_length: int, buffer_gb: float = 15.0) -> int:
    """Max donors that fit in the LMCache CPU buffer at given token length.

    KV per token per donor ≈ 28 layers × 2 (K+V) × 128 (head_dim) × 2 (fp16)
    = 14,336 bytes.  With 70% safety margin to avoid OOMKill.
    """
    kv_bytes_per_token = 28 * 2 * 128 * 2  # 14,336
    kv_per_donor = token_length * kv_bytes_per_token
    max_donors = int(buffer_gb * 1e9 * 0.70 / kv_per_donor)
    return max(10, min(max_donors, 500))


def run_length(
    token_length: int,
    endpoint: str,
    model: str,
    n_samples: int,
) -> dict:
    """Run the full three-condition measurement for one context length.

    Processes in batches that fit within the LMCache buffer to avoid OOMKill.
    For each batch: cold → register donors → hit → miss, then aggregate.
    """
    target_chars = token_length * 4  # ~4 chars per token

    batch_size = _compute_batch_size(token_length)
    n_batches = (n_samples + batch_size - 1) // batch_size

    print(f"  LMCache batch size: {batch_size} donors "
          f"({n_batches} batches for {n_samples} samples)")

    # --- Generate ALL unique prompts up front ---
    # Each prompt gets a UNIQUE context body (seeded) to prevent cross-batch
    # LMCache contamination: batch N's stored KV must never match batch N+1's
    # "cold" prompts.
    prompts_a = []
    for i in range(n_samples):
        uid = f"brk-a-{uuid.uuid4().hex[:12]}"
        context = _build_unique_context(target_chars, seed=1_000_000 + i)
        prompts_a.append(_build_prompt(uid, context))

    # Hit prompts: exact same prompts_a re-sent. LMCache matches on exact
    # 256-token chunk content — same prompt text = same tokens = hit.

    # Miss prompts: completely different seed range, no chunk overlap.
    prompts_b = []
    for i in range(n_samples):
        uid = f"brk-b-{uuid.uuid4().hex[:12]}"
        miss_body = _build_miss_context(target_chars, seed=i)
        prompts_b.append(_build_prompt(uid, miss_body))

    print(f"  Prompt length: ~{len(prompts_a[0])} chars "
          f"(~{len(prompts_a[0]) // 4} tokens)")

    all_cold: list[float] = []
    all_hit: list[float] = []
    all_miss: list[float] = []

    for batch_idx in range(n_batches):
        start = batch_idx * batch_size
        end = min(start + batch_size, n_samples)
        batch_a = prompts_a[start:end]
        batch_b = prompts_b[start:end]
        batch_n = end - start

        print(f"\n  --- Batch {batch_idx + 1}/{n_batches} "
              f"(samples {start + 1}-{end}) ---")

        # Phase 1: Cold TTFT — each prompt has unique context body,
        # so LMCache cannot match chunks from any prior batch.
        print(f"  Phase 1: Cold TTFT (n={batch_n})")
        cold_lat = run_condition("cold", endpoint, model, batch_a)
        all_cold.extend(cold_lat)

        # Phase 2: Register donors — re-send cold prompts to store KV
        print(f"  Phase 2: Register donors (n={batch_n})")
        run_condition("donor", endpoint, model, batch_a, max_tokens=50)

        # Phase 3: Hit TTFT — re-send exact same prompts as cold phase.
        # LMCache matches on exact 256-token chunk content → hit.
        print(f"  Phase 3: Hit TTFT (n={batch_n})")
        hit_lat = run_condition("hit", endpoint, model, batch_a)
        all_hit.extend(hit_lat)

        # Phase 4: Miss TTFT — topically different content, no chunk overlap
        print(f"  Phase 4: Miss TTFT (n={batch_n})")
        miss_lat = run_condition("miss", endpoint, model, batch_b)
        all_miss.extend(miss_lat)

    result = analyze_breakeven(all_cold, all_hit, all_miss)
    result["token_length"] = token_length
    result["_raw_cold"] = all_cold
    result["_raw_hit"] = all_hit
    result["_raw_miss"] = all_miss
    return result


def print_table(results: list[dict]) -> None:
    """Print a formatted summary table."""
    header = (
        f"{'Tokens':>8} {'Cold':>10} {'Hit':>10} {'Miss':>10} "
        f"{'Overhead':>10} {'Savings':>10} {'BrkEven':>8} "
        f"{'Spd@50%':>8} {'Spd@75%':>8} {'Spd@90%':>8}"
    )
    print()
    print("=" * len(header))
    print("Break-Even Hit-Rate Analysis")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for r in results:
        sp = r.get("speedup_at_hit_rate", {})
        print(
            f"{r['token_length']:>8} "
            f"{r['cold_mean_ms']:>9.0f}ms "
            f"{r['hit_mean_ms']:>9.0f}ms "
            f"{r['miss_mean_ms']:>9.0f}ms "
            f"{r['overhead_miss_ms']:>9.0f}ms "
            f"{r['savings_hit_ms']:>9.0f}ms "
            f"{r['breakeven_hit_rate']:>7.1%} "
            f"{sp.get('0.50', 0):>7.2f}x "
            f"{sp.get('0.75', 0):>7.2f}x "
            f"{sp.get('0.90', 0):>7.2f}x"
        )

    print("-" * len(header))
    print()
    print("Interpretation:")
    print("  BrkEven = minimum hit rate for SemBlend to be net-positive vs cold prefill")
    print("  Overhead = extra latency on misses (M - C)")
    print("  Savings  = latency saved on hits (C - H)")
    print("  P_h* = Overhead / (Overhead + Savings)")
    print("  Spd@X%  = expected speedup at X% hit rate")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Break-even hit-rate analysis for SemBlend"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-samples", type=int, default=500,
                        help="Number of prompts per condition per length")
    parser.add_argument("--token-lengths", default="2048,4096,8192,16384",
                        help="Comma-separated context lengths in tokens")
    parser.add_argument("--output", default=None,
                        help="Path to save JSON results")
    args = parser.parse_args()

    token_lengths = [int(x.strip()) for x in args.token_lengths.split(",")]

    print()
    print("Break-Even Hit-Rate Analysis")
    print(f"  endpoint={args.endpoint}")
    print(f"  model={args.model}")
    print(f"  n_samples={args.n_samples}")
    print(f"  token_lengths={token_lengths}")
    print()

    # Health check
    try:
        requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()
    except Exception as exc:
        print(f"Endpoint health check failed: {exc}")
        sys.exit(1)

    results: list[dict] = []

    for tl in token_lengths:
        print(f"\n--- Context length: {tl} tokens ---")
        result = run_length(tl, args.endpoint, args.model, args.n_samples)
        results.append(result)

        # Print intermediate result
        be = result["breakeven_hit_rate"]
        print(f"  => Cold={result['cold_mean_ms']:.0f}ms, "
              f"Hit={result['hit_mean_ms']:.0f}ms, "
              f"Miss={result['miss_mean_ms']:.0f}ms")
        print(f"  => Break-even P_h* = {be:.1%}")

    print_table(results)

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import bootstrap_mean, bootstrap_speedup

    print("Bootstrap 95% Confidence Intervals")
    print("=" * 60)
    for r in results:
        tl = r["token_length"]
        cold_arr = np.array(r["_raw_cold"])
        hit_arr = np.array(r["_raw_hit"])
        miss_arr = np.array(r["_raw_miss"])
        print(f"\n  Context length: {tl} tokens")
        print(f"    Cold TTFT mean:  {bootstrap_mean(cold_arr)}")
        print(f"    Hit TTFT mean:   {bootstrap_mean(hit_arr)}")
        print(f"    Miss TTFT mean:  {bootstrap_mean(miss_arr)}")
        if len(cold_arr) > 0 and len(hit_arr) > 0:
            print(f"    Hit speedup:     {bootstrap_speedup(hit_arr, cold_arr)}")
        print(f"    Break-even P_h*: {r['breakeven_hit_rate']:.4f}")
    print()

    if args.output:
        # Strip internal raw arrays before serializing
        serializable_results = [
            {k: v for k, v in r.items() if not k.startswith("_")}
            for r in results
        ]
        output_data = {
            "endpoint": args.endpoint,
            "model": args.model,
            "n_samples": args.n_samples,
            "results": serializable_results,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
