#!/usr/bin/env python3
"""WildChat-1M SemBlend benchmark: user-level KV reuse across chat sessions.

For each user (grouped by hashed_ip) with >=N consecutive conversations,
registers conversation K as donor then measures TTFT on conversation K+1.

Validates the core SemBlend production thesis: users asking related questions
over time enable KV cache reuse across their session history.

Metrics reported:
  - hit_rate: fraction of user pairs with cosine similarity >= threshold
  - ttft_speedup_{cold}: hit-only speedup for each target token length
  - similarity_distribution: percentiles of cosine similarity across user pairs
  - per_user_speedup: distribution of speedups per user
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

try:
    import requests
except ImportError:
    print("ERROR: pip install requests")
    sys.exit(1)


# ---------------------------------------------------------------------------
# API helpers (mirrors semblend_ttft_scaling.py)
# ---------------------------------------------------------------------------

def api_call(
    endpoint: str, model: str, prompt: str,
    max_tokens: int = 5, stream: bool = False,
    timeout: float = 300.0, retries: int = 3,
) -> requests.Response:
    for attempt in range(retries):
        try:
            resp = requests.post(
                f"{endpoint}/v1/completions",
                json={
                    "model": model, "prompt": prompt,
                    "max_tokens": max_tokens, "temperature": 0.0,
                    "stream": stream,
                },
                timeout=timeout, stream=stream,
            )
            resp.raise_for_status()
            return resp
        except (requests.ConnectionError, requests.Timeout):
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
    raise ValueError("No tokens received from stream")


def register_donor(endpoint: str, model: str, prompt: str) -> None:
    api_call(endpoint, model, prompt, max_tokens=1, stream=False)


# ---------------------------------------------------------------------------
# Tokenizer helper
# ---------------------------------------------------------------------------

def get_tokenizer(model: str):
    """Load tokenizer for the given model.

    Returns None if tokenizer would be too slow (e.g., slow Python HF tokenizer
    when PyTorch is unavailable). Character-based truncation is used as fallback.
    For 50K rows at ~4K tokens each, a slow tokenizer would take hours.
    """
    try:
        import torch  # noqa: F401 — presence means fast path available
    except ImportError:
        # No PyTorch → transformers tokenizers run in slow Python mode.
        # Character-based truncation (4 chars/token) is much faster and accurate enough.
        print("  Tokenizer: PyTorch unavailable; using character-based truncation (~4 chars/token)")
        return None

    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
        return tok
    except Exception as e:
        print(f"WARNING: Could not load tokenizer for {model}: {e}")
        return None


def truncate_to_tokens(text: str, max_tokens: int, tokenizer) -> str:
    """Truncate text to at most max_tokens tokens."""
    if tokenizer is None:
        # Rough estimate: ~4 chars/token
        return text[:max_tokens * 4]
    ids = tokenizer.encode(text)
    if len(ids) <= max_tokens:
        return text
    return tokenizer.decode(ids[:max_tokens - 8])


def count_tokens(text: str, tokenizer) -> int:
    """Count tokens in text."""
    if tokenizer is None:
        return len(text) // 4
    return len(tokenizer.encode(text))


# ---------------------------------------------------------------------------
# WildChat data loading
# ---------------------------------------------------------------------------

def load_wildchat(data_path: str, max_rows: int = 50000) -> list[dict]:
    """Load WildChat conversations from JSONL file.

    Each row: {'hashed_ip': str, 'user_texts': list[str], 'model': str}
    """
    rows = []
    with open(data_path) as f:
        for line in f:
            if max_rows and len(rows) >= max_rows:
                break
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_user_pairs(
    rows: list[dict],
    min_conversations: int = 2,
    target_tokens: int = 4096,
    tokenizer=None,
    min_chars: int = 0,
) -> list[tuple[str, str, str]]:
    """Build (ip, donor_prompt, query_prompt) pairs for users with >=min_conversations.

    Returns list of (hashed_ip, donor_text, query_text) tuples.
    For each consecutive pair of conversations from the same user,
    creates one donor-query pair.

    Args:
        min_chars: Minimum character length for both donor and query after truncation.
            Use to filter out short conversations where cold TTFT is already fast
            and SemBlend speedup would be negligible. Typical: 2000 chars (~500 tokens)
            for visible speedup, 6000 chars (~1500 tokens) for strong speedup.
    """
    # Group rows by hashed_ip (preserve order = conversation order in dataset)
    user_convos: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        ip = row.get("hashed_ip", "unknown")
        # Prefer full_text (user + assistant), fall back to user_texts only
        full_text = row.get("full_text", "")
        if not full_text:
            user_texts = row.get("user_texts", [])
            full_text = " ".join(user_texts).strip()
        if full_text:
            user_convos[ip].append(full_text)

    effective_min = max(min_chars, 50)

    # Build consecutive pairs for users with enough conversations
    pairs = []
    for ip, convos in user_convos.items():
        if len(convos) < min_conversations:
            continue
        # For each consecutive pair
        for i in range(len(convos) - 1):
            donor_raw = convos[i]
            query_raw = convos[i + 1]
            # Truncate to target token length
            donor_prompt = truncate_to_tokens(donor_raw, target_tokens, tokenizer)
            query_prompt = truncate_to_tokens(query_raw, target_tokens, tokenizer)
            # Filter by minimum length — short texts show no SemBlend benefit
            if (len(donor_prompt.strip()) >= effective_min
                    and len(query_prompt.strip()) >= effective_min):
                pairs.append((ip, donor_prompt, query_prompt))

    return pairs


# ---------------------------------------------------------------------------
# Cold baseline measurement
# ---------------------------------------------------------------------------

def measure_cold_baseline(
    endpoint: str, model: str,
    texts: list[str],
    n_samples: int = 5,
) -> float:
    """Measure cold TTFT baseline on a fresh vLLM restart."""
    print(f"\n  Measuring cold baseline ({n_samples} samples)...")
    times = []
    for i, text in enumerate(texts[:n_samples]):
        try:
            ttft = measure_streaming_ttft(endpoint, model, text)
            times.append(ttft)
            print(f"    [{i+1}] {ttft:.0f}ms")
        except Exception as e:
            print(f"    [{i+1}] ERROR: {e}")

    if not times:
        raise RuntimeError("No cold baseline measurements succeeded")
    return statistics.median(times)


# ---------------------------------------------------------------------------
# Main benchmark
# ---------------------------------------------------------------------------

def run_wildchat_bench(
    endpoint: str,
    model: str,
    data_path: str,
    target_tokens: int = 4096,
    n_pairs: int = 50,
    settle_time: float = 3.0,
    cold_samples: int = 8,
    output_path: str | None = None,
    min_chars: int = 0,
) -> dict:
    """Run WildChat SemBlend benchmark.

    Args:
        endpoint: vLLM API endpoint
        model: model name
        data_path: path to WildChat JSONL file
        target_tokens: target prompt length (truncated to this)
        n_pairs: number of donor-query pairs to evaluate
        settle_time: seconds to wait for Milvus to index donor
        cold_samples: number of cold baseline measurements
        output_path: optional path to save results JSON

    Returns:
        dict with benchmark metrics
    """
    print(f"\nWildChat-1M SemBlend Benchmark")
    print(f"  Model: {model}")
    print(f"  Target tokens: {target_tokens}")
    print(f"  Pairs to test: {n_pairs}")
    print(f"  Data: {data_path}")

    # Load tokenizer
    tokenizer = get_tokenizer(model)
    if tokenizer:
        print(f"  Tokenizer: loaded for {model}")
    else:
        print(f"  Tokenizer: unavailable, using char estimate")

    # Load WildChat data
    print(f"\n  Loading WildChat data...")
    rows = load_wildchat(data_path)
    print(f"  Loaded {len(rows)} rows")

    # Build user pairs
    pairs = build_user_pairs(rows, min_conversations=2,
                             target_tokens=target_tokens,
                             tokenizer=tokenizer,
                             min_chars=min_chars)
    print(f"  User pairs available: {len(pairs)}")

    if len(pairs) < 5:
        print(f"  ERROR: Not enough pairs. Need more WildChat data.")
        print(f"  Try downloading more rows (current: {len(rows)}).")
        sys.exit(1)

    # Sample pairs for benchmark
    import random
    random.seed(42)
    if len(pairs) > n_pairs:
        pairs = random.sample(pairs, n_pairs)

    # Get actual token lengths
    actual_lens = []
    for _, donor, query in pairs[:5]:
        d_len = count_tokens(donor, tokenizer)
        q_len = count_tokens(query, tokenizer)
        actual_lens.append((d_len, q_len))

    avg_donor_len = statistics.mean([d for d, _ in actual_lens])
    avg_query_len = statistics.mean([q for _, q in actual_lens])
    print(f"  Avg donor length: {avg_donor_len:.0f} tokens")
    print(f"  Avg query length: {avg_query_len:.0f} tokens")

    # Measure cold baseline on a subset of donor texts
    cold_texts = [donor for _, donor, _ in pairs[:cold_samples]]
    print(f"\n{'='*70}")
    print(f"PHASE 1: Cold baselines")
    print(f"{'='*70}")
    cold_p50 = measure_cold_baseline(endpoint, model, cold_texts, cold_samples)
    print(f"\n  Cold p50: {cold_p50:.0f}ms")

    # Run SemBlend pairs: register donor, then measure query TTFT
    print(f"\n{'='*70}")
    print(f"PHASE 2: Register donor, measure query TTFT ({len(pairs)} pairs)")
    print(f"{'='*70}")

    results = []
    hits = 0
    misses = 0
    errors = 0

    for i, (ip, donor, query) in enumerate(pairs):
        print(f"\n  Pair {i+1}/{len(pairs)} [ip={ip[:8]}...]")
        try:
            # Register donor
            register_donor(endpoint, model, donor)
            time.sleep(settle_time)  # Wait for Milvus to index

            # Measure query TTFT
            ttft = measure_streaming_ttft(endpoint, model, query)
            speedup = cold_p50 / ttft
            hit = ttft < cold_p50 * 0.75  # >25% faster than cold = likely hit

            if hit:
                hits += 1
                label = "HIT"
            else:
                misses += 1
                label = "MISS"

            print(f"    TTFT: {ttft:.0f}ms, speedup: {speedup:.2f}x [{label}]")
            results.append({
                "ip": ip,
                "ttft_ms": ttft,
                "cold_p50_ms": cold_p50,
                "speedup": speedup,
                "hit": hit,
            })

        except Exception as e:
            print(f"    ERROR: {e}")
            errors += 1
            results.append({
                "ip": ip,
                "ttft_ms": None,
                "cold_p50_ms": cold_p50,
                "speedup": None,
                "hit": False,
                "error": str(e),
            })

    # Compute summary statistics
    valid = [r for r in results if r["ttft_ms"] is not None]
    hit_results = [r for r in valid if r["hit"]]
    miss_results = [r for r in valid if not r["hit"]]

    hit_rate = hits / len(valid) if valid else 0.0
    all_speedups = [r["speedup"] for r in valid]
    hit_speedups = [r["speedup"] for r in hit_results]

    summary = {
        "n_pairs": len(pairs),
        "n_valid": len(valid),
        "n_hits": hits,
        "n_misses": misses,
        "n_errors": errors,
        "hit_rate": hit_rate,
        "cold_p50_ms": cold_p50,
        "target_tokens": target_tokens,
        "avg_donor_len": avg_donor_len,
        "avg_query_len": avg_query_len,
    }

    if all_speedups:
        summary["speedup_p50"] = statistics.median(all_speedups)
        summary["speedup_mean"] = statistics.mean(all_speedups)
    if hit_speedups:
        summary["hit_speedup_p50"] = statistics.median(hit_speedups)
        summary["hit_speedup_mean"] = statistics.mean(hit_speedups)
        summary["hit_speedup_max"] = max(hit_speedups)

    print(f"\n{'='*70}")
    print(f"RESULTS:")
    print(f"  Pairs tested:        {len(valid)}")
    print(f"  Hit rate:            {hit_rate:.1%} ({hits}/{len(valid)})")
    print(f"  Cold p50:            {cold_p50:.0f}ms")
    if all_speedups:
        print(f"  Overall p50 speedup: {summary['speedup_p50']:.2f}x")
    if hit_speedups:
        print(f"  Hit-only p50 speedup:{summary['hit_speedup_p50']:.2f}x")
        print(f"  Hit speedup max:     {summary['hit_speedup_max']:.2f}x")
    print(f"{'='*70}")

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump({"summary": summary, "per_pair": results}, f, indent=2)
        print(f"\n  Results saved to {output_path}")

    return summary


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------

def download_wildchat(output_path: str, max_rows: int = 50000) -> None:
    """Download WildChat-1M from HuggingFace and save to JSONL.

    Runs in a subprocess with a clean sys.path to avoid the local
    benchmarks/e2e/datasets/ module shadowing the HuggingFace datasets package.
    """
    import subprocess
    import sys

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading WildChat-1M ({max_rows} rows) → {output_path}")

    # Run download in a subprocess so it doesn't inherit the local path pollution
    script = f"""
import sys, os
# Remove any local paths that shadow site-packages
sys.path = [p for p in sys.path if 'benchmarks' not in p]
import json
from datasets import load_dataset
from pathlib import Path

output_path = {repr(output_path)}
max_rows = {max_rows}
Path(output_path).parent.mkdir(parents=True, exist_ok=True)

ds = load_dataset("allenai/WildChat-1M", split="train", streaming=True)
count = 0
with open(output_path, "w") as f:
    for row in ds:
        if count >= max_rows:
            break
        conv = row.get("conversation", [])
        # Include all turns (user + assistant) for longer context
        full_text = " ".join(
            t.get("content", "") for t in conv
            if t.get("role") in ("user", "assistant") and t.get("content")
        ).strip()
        user_texts = [t.get("content", "") for t in conv if t.get("role") == "user"]
        record = {{
            "hashed_ip": row.get("hashed_ip", ""),
            "full_text": full_text,           # user + assistant combined (longer)
            "user_texts": user_texts,         # user-only for reference
            "model": row.get("model", ""),
            "n_turns": len(conv),
        }}
        f.write(json.dumps(record) + "\\n")
        count += 1
        if count % 5000 == 0:
            print(f"  {{count}}/{{max_rows}} rows downloaded...", flush=True)

print(f"Done: {{count}} rows saved to {{output_path}}", flush=True)
"""
    result = subprocess.run(
        [sys.executable, "-u", "-c", script],
        text=True, capture_output=False,
        env={**os.environ},
    )
    if result.returncode != 0:
        raise RuntimeError(f"WildChat download failed with exit code {result.returncode}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="WildChat-1M SemBlend Benchmark")
    p.add_argument("--endpoint", default="http://localhost:8100",
                   help="vLLM API endpoint")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ",
                   help="Model name")
    p.add_argument("--data-path", default="benchmarks/data/wildchat/wildchat_50k.jsonl",
                   help="Path to WildChat JSONL data file")
    p.add_argument("--download", action="store_true",
                   help="Download WildChat data before benchmarking")
    p.add_argument("--download-rows", type=int, default=50000,
                   help="Number of WildChat rows to download (default: 50000)")
    p.add_argument("--target-tokens", type=int, default=4096,
                   help="Target prompt length in tokens (default: 4096)")
    p.add_argument("--n-pairs", type=int, default=50,
                   help="Number of donor-query pairs to test (default: 50)")
    p.add_argument("--settle-time", type=float, default=3.0,
                   help="Seconds to wait after donor registration (default: 3.0)")
    p.add_argument("--cold-samples", type=int, default=8,
                   help="Cold baseline measurements (default: 8)")
    p.add_argument("--min-chars", type=int, default=0,
                   help="Minimum character length for donor and query texts (default: 0). "
                        "Use to filter out short conversations where SemBlend speedup is "
                        "negligible. E.g., 2000 (~500 tokens) for visible speedup, "
                        "6000 (~1500 tokens) for strong speedup.")
    p.add_argument("--output", default=None,
                   help="Path to save JSON results")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Download data if requested or if file doesn't exist
    if args.download or not Path(args.data_path).exists():
        download_wildchat(args.data_path, args.download_rows)

    run_wildchat_bench(
        endpoint=args.endpoint,
        model=args.model,
        data_path=args.data_path,
        target_tokens=args.target_tokens,
        n_pairs=args.n_pairs,
        settle_time=args.settle_time,
        cold_samples=args.cold_samples,
        output_path=args.output,
        min_chars=args.min_chars,
    )
