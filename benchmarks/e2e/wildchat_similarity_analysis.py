#!/usr/bin/env python3
"""WildChat-1M semantic similarity analysis for SemBlend paper.

Two-part analysis:
  Part 1: Compute MiniLM embeddings for all conversations, measure cosine
          similarity between consecutive conversations from the same user.
          Reports the distribution (key paper metric: how many user pairs
          share sufficient semantic overlap for SemBlend KV reuse).

  Part 2: E2E TTFT benchmark on pairs with cosine sim >= threshold,
          using padded prompts to reach meaningful lengths (4K/8K tokens)
          where SemBlend's speedup is significant.

Paper claim: "In WildChat-1M, X% of consecutive user conversation pairs
share cosine similarity >= 0.60, indicating that most production users
benefit from SemBlend's semantic KV reuse."
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# Embedder (MiniLM-L6-v2) — same model as SemBlend uses for donor lookup
# ---------------------------------------------------------------------------

_EMBEDDER = None


def get_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        try:
            from sentence_transformers import SentenceTransformer
            # Force CPU to avoid competing with vLLM for GPU memory
            _EMBEDDER = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2",
                device="cpu",
            )
            print("  Embedder: MiniLM-L6-v2 loaded on CPU")
        except ImportError:
            print("  ERROR: pip install sentence-transformers")
            sys.exit(1)
    return _EMBEDDER


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a list of texts using MiniLM-L6-v2. Returns (N, 384) float32 array."""
    embedder = get_embedder()
    embeddings = embedder.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via dot product
    )
    return embeddings.astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two normalized vectors (dot product)."""
    return float(np.dot(a, b))


# ---------------------------------------------------------------------------
# WildChat data loading
# ---------------------------------------------------------------------------

def load_wildchat(data_path: str, max_rows: int = 100000) -> list[dict]:
    rows = []
    with open(data_path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_rows and len(rows) >= max_rows:
                break
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue  # Skip malformed lines (invalid unicode escapes)
    return rows


def build_user_conv_texts(rows: list[dict]) -> dict[str, list[str]]:
    """Group conversation texts by user IP."""
    user_convos: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ip = row.get("hashed_ip", "unknown")
        # Use full_text (user + assistant) for richer context
        text = row.get("full_text", "")
        if not text:
            texts = row.get("user_texts", [])
            text = " ".join(texts).strip()
        if text and len(text.strip()) > 20:
            user_convos[ip].append(text)
    return dict(user_convos)


# ---------------------------------------------------------------------------
# Part 1: Similarity distribution analysis
# ---------------------------------------------------------------------------

def analyze_similarity_distribution(
    user_convos: dict[str, list[str]],
    threshold: float = 0.60,
    max_users: int = 5000,
    output_path: str | None = None,
) -> dict:
    """Compute cosine similarity for all consecutive conversation pairs.

    Returns statistics about the distribution.
    """
    print(f"\nPart 1: Semantic Similarity Distribution Analysis")
    print(f"  Threshold: {threshold}")
    print(f"  Max users: {max_users}")

    # Filter users with >=2 conversations
    eligible = {ip: convos for ip, convos in user_convos.items()
                if len(convos) >= 2}
    print(f"  Users with >=2 conversations: {len(eligible)}")

    if not eligible:
        print("  ERROR: No eligible users")
        return {}

    # Sample up to max_users
    import random
    random.seed(42)
    sampled_ips = list(eligible.keys())
    if len(sampled_ips) > max_users:
        sampled_ips = random.sample(sampled_ips, max_users)

    # Build all consecutive pairs
    pairs: list[tuple[str, str, str]] = []  # (ip, conv_n, conv_n+1)
    for ip in sampled_ips:
        convos = eligible[ip]
        for i in range(len(convos) - 1):
            pairs.append((ip, convos[i], convos[i + 1]))

    print(f"  Total pairs to analyze: {len(pairs)}")

    # Embed all texts (donor + query)
    all_texts = [donor for _, donor, _ in pairs] + [query for _, _, query in pairs]
    print(f"  Computing embeddings for {len(all_texts)} texts...")
    all_embeddings = embed_texts(all_texts)

    donor_embeddings = all_embeddings[:len(pairs)]
    query_embeddings = all_embeddings[len(pairs):]

    # Compute pairwise similarities
    similarities = [
        cosine_similarity(donor_embeddings[i], query_embeddings[i])
        for i in range(len(pairs))
    ]

    # Also compute random-pair baseline (different users)
    print(f"  Computing random-pair baseline...")
    n_random = min(len(pairs), 1000)
    random_sims = []
    random_ips = list(eligible.keys())
    for _ in range(n_random):
        ip_a, ip_b = random.sample(random_ips, 2)
        text_a = random.choice(eligible[ip_a])
        text_b = random.choice(eligible[ip_b])
        # Quick embedding of individual texts
        emb_a = embed_texts([text_a])[0]
        emb_b = embed_texts([text_b])[0]
        random_sims.append(cosine_similarity(emb_a, emb_b))

    # Statistics
    sims = np.array(similarities)
    rand_sims = np.array(random_sims)

    above_threshold = float((sims >= threshold).mean())
    above_50 = float((sims >= 0.50).mean())
    above_70 = float((sims >= 0.70).mean())
    above_80 = float((sims >= 0.80).mean())

    print(f"\n  Results:")
    print(f"  Consecutive user pairs: {len(pairs)}")
    print(f"  Similarity >= {threshold}: {above_threshold:.1%}")
    print(f"  Similarity >= 0.50: {above_50:.1%}")
    print(f"  Similarity >= 0.70: {above_70:.1%}")
    print(f"  Similarity >= 0.80: {above_80:.1%}")
    print(f"  Median similarity: {float(np.median(sims)):.3f}")
    print(f"  Mean similarity: {float(np.mean(sims)):.3f}")
    print(f"  p90 similarity: {float(np.percentile(sims, 90)):.3f}")
    print(f"  Random-pair median: {float(np.median(rand_sims)):.3f} (baseline)")

    result = {
        "n_pairs": len(pairs),
        "n_users": len(sampled_ips),
        "threshold": threshold,
        "hit_rate_at_threshold": above_threshold,
        "hit_rate_at_50": above_50,
        "hit_rate_at_70": above_70,
        "hit_rate_at_80": above_80,
        "median_similarity": float(np.median(sims)),
        "mean_similarity": float(np.mean(sims)),
        "p10_similarity": float(np.percentile(sims, 10)),
        "p25_similarity": float(np.percentile(sims, 25)),
        "p75_similarity": float(np.percentile(sims, 75)),
        "p90_similarity": float(np.percentile(sims, 90)),
        "random_baseline_median": float(np.median(rand_sims)),
        "lift_over_random": float(np.median(sims)) - float(np.median(rand_sims)),
        # Return high-similarity pairs for E2E TTFT test
        "high_sim_pairs": [
            {"ip": pairs[i][0], "donor": pairs[i][1], "query": pairs[i][2],
             "similarity": similarities[i]}
            for i in range(len(pairs))
            if similarities[i] >= threshold
        ][:100],  # Top 100 high-similarity pairs for E2E
    }

    if output_path:
        # Save without the full pairs (too large)
        save_result = {k: v for k, v in result.items() if k != "high_sim_pairs"}
        save_result["n_high_sim_pairs"] = len(result["high_sim_pairs"])
        # Save distribution as histogram
        hist, edges = np.histogram(sims, bins=20, range=(0, 1))
        save_result["similarity_histogram"] = {
            "counts": hist.tolist(),
            "bin_edges": edges.tolist(),
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(save_result, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    return result


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def api_call(endpoint, model, prompt, max_tokens=5, stream=False, timeout=300.0):
    for attempt in range(3):
        try:
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json={"model": model, "prompt": prompt,
                      "max_tokens": max_tokens, "temperature": 0.0, "stream": stream},
                timeout=timeout, stream=stream,
            )
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                raise


def measure_streaming_ttft(endpoint, model, prompt):
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


def register_donor(endpoint, model, prompt):
    api_call(endpoint, model, prompt, max_tokens=1, stream=False)


# ---------------------------------------------------------------------------
# Part 2: E2E TTFT benchmark on high-similarity pairs
# ---------------------------------------------------------------------------

# A realistic 2048-token system prompt (simulates production chatbot context)
SYSTEM_PROMPT = """You are a helpful, harmless, and honest AI assistant. Your goal is to provide accurate, thoughtful, and nuanced responses to user queries. You excel at:

- Explaining complex topics clearly and at the appropriate level for the user
- Providing step-by-step guidance for technical problems
- Offering balanced perspectives on ambiguous questions
- Generating creative content including writing, code, and brainstorming
- Analyzing documents and extracting key information
- Debugging code and suggesting improvements
- Translating between languages with cultural sensitivity
- Answering factual questions with appropriate caveats about uncertainty

You approach sensitive topics with care and nuance. When you're unsure about something, you say so clearly. You don't make up information or hallucinate citations. When asked to do something outside your capabilities, you explain what you can't do and suggest alternatives.

Your responses are structured clearly, using markdown formatting when appropriate (headers, bullet points, code blocks). You adapt your communication style to match the user's apparent expertise level and needs.

For technical questions, you prefer to show working code with explanations rather than just describing what to do. For creative writing, you aim to match the tone and style requested. For research tasks, you help organize information and identify what's well-established vs. uncertain.

Context about this deployment: This assistant is used by a diverse global user base across many domains including software development, creative writing, education, research, data analysis, customer service, and personal productivity. Users may communicate in any language, and you should respond in the same language as the user when possible.

Remember: accuracy over confidence. If you don't know something, say so. If a question is ambiguous, ask for clarification. If there are multiple valid approaches, present the tradeoffs.

The following is a conversation with a user. Please provide a helpful and informative response to their query. Take care to understand the full context of what they're asking before responding. If the user's message is unclear, ask a clarifying question rather than guessing at their intent.

Begin your response directly with relevant content — do not start with "Of course!", "Sure!", "Great question!" or similar filler phrases. Be direct and substantive.

User query follows:
"""  # ~500 tokens — will be combined with user content to reach ~2K tokens total


def pad_prompt_to_target(text: str, target_tokens: int, tokenizer) -> str:
    """Combine system prompt + user text, truncated to target_tokens."""
    combined = SYSTEM_PROMPT + "\n\n" + text
    if tokenizer is None:
        # Rough char estimate
        return combined[:target_tokens * 4]
    ids = tokenizer.encode(combined)
    if len(ids) <= target_tokens:
        return combined
    return tokenizer.decode(ids[:target_tokens - 8])


def run_e2e_ttft(
    endpoint: str,
    model: str,
    high_sim_pairs: list[dict],
    target_tokens: int = 4096,
    n_pairs: int = 30,
    settle_time: float = 5.0,
    cold_samples: int = 8,
) -> dict:
    """Run E2E TTFT benchmark on high-similarity WildChat pairs.

    Uses a system prompt to pad prompts to meaningful lengths.
    """
    print(f"\nPart 2: E2E TTFT Benchmark on High-Similarity Pairs")
    print(f"  Target tokens: {target_tokens}")
    print(f"  Pairs: {n_pairs}")
    print(f"  Endpoint: {endpoint}")

    # Load tokenizer
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        print(f"  Tokenizer: loaded")
    except Exception:
        tokenizer = None
        print(f"  Tokenizer: unavailable, using char estimate")

    # Sample pairs
    import random
    random.seed(42)
    sampled = random.sample(high_sim_pairs, min(n_pairs, len(high_sim_pairs)))

    # Build padded prompts
    donor_prompts = []
    query_prompts = []
    for pair in sampled:
        donor_prompts.append(pad_prompt_to_target(pair["donor"], target_tokens, tokenizer))
        query_prompts.append(pad_prompt_to_target(pair["query"], target_tokens, tokenizer))

    # Measure cold baselines
    print(f"\n  Measuring cold baseline ({cold_samples} samples)...")
    cold_times = []
    for i, prompt in enumerate(donor_prompts[:cold_samples]):
        try:
            ttft = measure_streaming_ttft(endpoint, model, prompt)
            cold_times.append(ttft)
            print(f"    [{i+1}] {ttft:.0f}ms")
        except Exception as e:
            print(f"    [{i+1}] ERROR: {e}")

    if not cold_times:
        print("  ERROR: No cold measurements succeeded")
        return {}

    cold_p50 = statistics.median(cold_times)
    print(f"  Cold p50: {cold_p50:.0f}ms")

    # Run SemBlend pairs
    print(f"\n  Running {len(sampled)} SemBlend pairs...")
    results = []
    hits = 0

    for i, (pair, donor_prompt, query_prompt) in enumerate(
        zip(sampled, donor_prompts, query_prompts)
    ):
        sim = pair.get("similarity", 0.0)
        print(f"  [{i+1}/{len(sampled)}] ip={pair['ip'][:8]}... sim={sim:.3f}")
        try:
            register_donor(endpoint, model, donor_prompt)
            time.sleep(settle_time)
            ttft = measure_streaming_ttft(endpoint, model, query_prompt)
            speedup = cold_p50 / ttft
            hit = ttft < cold_p50 * 0.75
            if hit:
                hits += 1
            print(f"    TTFT: {ttft:.0f}ms, speedup: {speedup:.2f}x [{'HIT' if hit else 'MISS'}]")
            results.append({"sim": sim, "ttft_ms": ttft, "speedup": speedup, "hit": hit})
        except Exception as e:
            print(f"    ERROR: {e}")

    valid = [r for r in results if "error" not in r]
    hit_results = [r for r in valid if r["hit"]]
    hit_rate = hits / len(valid) if valid else 0.0
    hit_speedups = [r["speedup"] for r in hit_results]

    summary = {
        "n_pairs": len(sampled),
        "n_valid": len(valid),
        "hit_rate": hit_rate,
        "cold_p50_ms": cold_p50,
        "target_tokens": target_tokens,
    }
    if hit_speedups:
        summary["hit_speedup_p50"] = statistics.median(hit_speedups)
        summary["hit_speedup_max"] = max(hit_speedups)

    print(f"\n  E2E Results:")
    print(f"  Hit rate: {hit_rate:.1%} ({hits}/{len(valid)})")
    print(f"  Cold p50: {cold_p50:.0f}ms")
    if hit_speedups:
        print(f"  Hit speedup p50: {summary['hit_speedup_p50']:.2f}x")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="WildChat Similarity Analysis")
    p.add_argument("--data-path", required=True,
                   help="Path to WildChat JSONL data file")
    p.add_argument("--threshold", type=float, default=0.60,
                   help="Cosine similarity threshold for 'hit' (default: 0.60)")
    p.add_argument("--max-users", type=int, default=5000,
                   help="Max users to analyze (default: 5000)")
    p.add_argument("--output", default=None,
                   help="Path to save similarity analysis JSON")
    # E2E benchmark args (optional)
    p.add_argument("--run-e2e", action="store_true",
                   help="Also run E2E TTFT benchmark on high-similarity pairs")
    p.add_argument("--endpoint", default="http://localhost:8100")
    p.add_argument("--model",
                   default="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4")
    p.add_argument("--target-tokens", type=int, default=4096)
    p.add_argument("--e2e-pairs", type=int, default=30)
    p.add_argument("--settle-time", type=float, default=5.0)
    p.add_argument("--e2e-output", default=None,
                   help="Path to save E2E TTFT results JSON")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    print("Loading WildChat data...")
    rows = load_wildchat(args.data_path)
    print(f"Loaded {len(rows)} rows")

    user_convos = build_user_conv_texts(rows)
    print(f"Unique users: {len(user_convos)}")
    multi = sum(1 for c in user_convos.values() if len(c) >= 2)
    print(f"Users with >=2 conversations: {multi}")

    # Part 1: Similarity analysis
    sim_result = analyze_similarity_distribution(
        user_convos,
        threshold=args.threshold,
        max_users=args.max_users,
        output_path=args.output,
    )

    # Print key paper metric
    hit_rate = sim_result.get("hit_rate_at_threshold", 0)
    median_sim = sim_result.get("median_similarity", 0)
    rand_baseline = sim_result.get("random_baseline_median", 0)
    lift = sim_result.get("lift_over_random", 0)
    print(f"\n{'='*70}")
    print(f"KEY PAPER METRICS:")
    print(f"  Hit rate at τ={args.threshold}: {hit_rate:.1%}")
    print(f"  Median similarity (consecutive): {median_sim:.3f}")
    print(f"  Median similarity (random pairs): {rand_baseline:.3f}")
    print(f"  Lift over random: {lift:+.3f}")
    print(f"{'='*70}")

    # Part 2: E2E TTFT (optional)
    if args.run_e2e:
        high_sim_pairs = sim_result.get("high_sim_pairs", [])
        if not high_sim_pairs:
            print("No high-similarity pairs found for E2E benchmark")
        else:
            print(f"\nRunning E2E on {len(high_sim_pairs)} high-similarity pairs...")
            e2e_result = run_e2e_ttft(
                endpoint=args.endpoint,
                model=args.model,
                high_sim_pairs=high_sim_pairs,
                target_tokens=args.target_tokens,
                n_pairs=args.e2e_pairs,
                settle_time=args.settle_time,
            )
            if args.e2e_output and e2e_result:
                Path(args.e2e_output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.e2e_output, "w") as f:
                    json.dump(e2e_result, f, indent=2)
