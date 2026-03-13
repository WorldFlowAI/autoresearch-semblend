#!/usr/bin/env python3
"""E2E benchmark: RoPE correction for REORDER scenarios.

Demonstrates that SemBlend's RoPE delta correction improves quality when
donor KV is reused at different positions. Creates document pairs with
identical paragraphs in different order, chunk-boundary-aligned so that
LMCache chunk hashes match across reordering.

Key requirement: each paragraph must be exactly N×256 tokens so that
chunk boundaries align with paragraph boundaries. This ensures REORDER
produces chunk-level matches with Δ≠0 position mappings.

Scenario:
  Donor:  [instruction_256tok][para_A_768tok][para_B_768tok][para_C_768tok][para_D_768tok]
  Target: [instruction_256tok][para_D_768tok][para_C_768tok][para_B_768tok][para_A_768tok]

  For para_A: donor_pos = 256..1023, target_pos = 2560..3327 → Δ = 2304
  RoPE correction: K_corrected = RoPE(Δ) × K_donor → exact position fix.

Usage:
  python benchmarks/e2e/rope_reorder_bench.py \\
      --endpoint http://localhost:8100 \\
      --n 8 --para-chunks 3 \\
      [--max-tokens 256]
"""

import argparse
import json
import math
import os
import random
import sys
import time

import requests

CHUNK_SIZE = 256  # LMCache chunk size in tokens


# ── Paragraph texts ──────────────────────────────────────────────────────
# Distinct topics so paragraphs are clearly differentiable.

_PARA_TEXTS = [
    (
        "Climate change represents one of the most pressing challenges facing "
        "modern civilization. Rising global temperatures are driving more frequent "
        "and severe weather events, from devastating hurricanes to prolonged "
        "droughts that threaten agricultural systems worldwide. The polar ice caps "
        "continue to shrink at an accelerating rate, contributing to rising sea "
        "levels that endanger coastal communities housing hundreds of millions of "
        "people. Scientists emphasize that reducing greenhouse gas emissions "
        "through transitioning to renewable energy sources, improving energy "
        "efficiency, and implementing carbon capture technologies is essential."
    ),
    (
        "Quantum computing has emerged as a transformative technology with the "
        "potential to revolutionize fields ranging from cryptography to drug "
        "discovery. Unlike classical computers that process information in binary "
        "bits, quantum computers leverage qubits that can exist in superposition "
        "states, enabling them to explore multiple solutions simultaneously. "
        "Recent breakthroughs in error correction and qubit coherence times have "
        "brought practical quantum computing closer to reality, with companies "
        "racing to achieve quantum advantage for commercially relevant problems."
    ),
    (
        "The world's oceans serve as critical regulators of Earth's climate "
        "system, absorbing approximately thirty percent of anthropogenic carbon "
        "dioxide emissions and storing vast quantities of thermal energy. Deep "
        "ocean currents act as a global conveyor belt, redistributing heat from "
        "the tropics to higher latitudes and influencing weather patterns across "
        "continents. Marine biodiversity faces unprecedented threats from "
        "acidification, plastic pollution, and overfishing, with coral reef "
        "ecosystems experiencing mass bleaching events at increasing frequency."
    ),
    (
        "Neuroscience research has made remarkable strides in understanding the "
        "human brain, revealing intricate networks of billions of neurons "
        "connected by trillions of synapses that give rise to consciousness, "
        "memory, and behavior. Advanced imaging technologies like functional MRI "
        "and optogenetics allow researchers to observe and manipulate neural "
        "activity with unprecedented precision. Brain-computer interfaces are "
        "enabling paralyzed individuals to control prosthetic limbs through "
        "thought alone, while deep brain stimulation offers breakthrough treatments."
    ),
]


def _load_tokenizer(model: str):
    """Load HuggingFace tokenizer for token-level control."""
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Cannot load tokenizer for {model}: {e}", file=sys.stderr)
        sys.exit(1)


def _pad_to_chunks(token_ids: list[int], pad_token: int) -> list[int]:
    """Pad token list to exact multiple of CHUNK_SIZE."""
    remainder = len(token_ids) % CHUNK_SIZE
    if remainder == 0:
        return token_ids
    pad_count = CHUNK_SIZE - remainder
    return token_ids + [pad_token] * pad_count


def _expand_text_to_tokens(
    text: str,
    tokenizer,
    target_chunks: int,
    seed: int,
) -> list[int]:
    """Tokenize text and pad/extend to exactly target_chunks × 256 tokens.

    If the text is too short, repeat it with variations. If too long, truncate.
    Then pad to exact chunk boundary.
    """
    target_tokens = target_chunks * CHUNK_SIZE
    rng = random.Random(seed)

    # Tokenize base text
    base_ids = tokenizer.encode(text, add_special_tokens=False)

    # Extend if needed by repeating with sentence shuffling
    sentences = text.split(". ")
    all_ids = list(base_ids)
    while len(all_ids) < target_tokens:
        rng.shuffle(sentences)
        variation = ". ".join(sentences) + ". "
        var_ids = tokenizer.encode(variation, add_special_tokens=False)
        all_ids.extend(var_ids)

    # Truncate to exact target
    all_ids = all_ids[:target_tokens]

    # Verify length
    assert len(all_ids) == target_tokens, (
        f"Expected {target_tokens} tokens, got {len(all_ids)}"
    )
    return all_ids


def _build_chat_token_ids(
    tokenizer,
    instruction_ids: list[int],
    paragraph_id_lists: list[list[int]],
    model: str,
) -> list[int]:
    """Build full chat-template token sequence from pre-tokenized components.

    Constructs the chat template tokens manually to preserve exact chunk
    alignment of paragraphs.
    """
    sep_ids = tokenizer.encode("\n\n", add_special_tokens=False)
    instr_text_ids = tokenizer.encode(
        "Summarize the key themes in each section above.",
        add_special_tokens=False,
    )

    if "llama" in model.lower():
        prefix = tokenizer.encode(
            "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            "You are a helpful assistant.<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.encode(
            "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
            add_special_tokens=False,
        )
    else:
        prefix = tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
            "<|im_start|>user\n",
            add_special_tokens=False,
        )
        suffix = tokenizer.encode(
            "<|im_end|>\n<|im_start|>assistant\n",
            add_special_tokens=False,
        )

    # Build: prefix + instruction_pad + para1 + sep + para2 + ... + instr_text + suffix
    # The instruction_ids are already padded to chunk boundary.
    # We embed them BEFORE the paragraphs to keep chunk alignment clean.
    result = list(instruction_ids)

    for i, para_ids in enumerate(paragraph_id_lists):
        if i > 0:
            # Don't add separators — they'd break chunk alignment
            pass
        result.extend(para_ids)

    return result


def _send_request(
    endpoint: str,
    token_ids: list[int],
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict:
    """Send a completion request using token IDs and return timing + output."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=timeout,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()

    choice = data["choices"][0]
    usage = data.get("usage", {})

    return {
        "text": choice["text"],
        "total_ms": elapsed_ms,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def _send_text_request(
    endpoint: str,
    prompt: str,
    model: str,
    max_tokens: int = 256,
    temperature: float = 0.0,
    timeout: float = 120.0,
) -> dict:
    """Send a text completion request."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        },
        timeout=timeout,
    )
    elapsed_ms = (time.monotonic() - t0) * 1000.0
    resp.raise_for_status()
    data = resp.json()
    choice = data["choices"][0]
    usage = data.get("usage", {})
    return {
        "text": choice["text"],
        "total_ms": elapsed_ms,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
    }


def run_reorder_benchmark(
    endpoint: str,
    model: str,
    n_samples: int,
    para_chunks: int,
    max_tokens: int,
    output_path: str | None = None,
):
    """Run the REORDER RoPE correction benchmark.

    Args:
        endpoint: vLLM API endpoint.
        model: Model name/ID.
        n_samples: Number of document pairs.
        para_chunks: Number of 256-token chunks per paragraph.
        max_tokens: Max generation tokens.
        output_path: Where to save JSON results.
    """
    tokenizer = _load_tokenizer(model)
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    instr_chunks = 1  # 256 tokens for instruction
    total_chunks = instr_chunks + 4 * para_chunks
    total_tokens = total_chunks * CHUNK_SIZE

    print(f"\n{'='*80}")
    print(f"REORDER RoPE Correction Benchmark")
    print(f"  endpoint:     {endpoint}")
    print(f"  model:        {model}")
    print(f"  n_samples:    {n_samples}")
    print(f"  para_chunks:  {para_chunks} (×256 = {para_chunks * 256} tokens/para)")
    print(f"  total_tokens: {total_tokens}")
    print(f"  max_tokens:   {max_tokens}")
    print(f"  chunk_size:   {CHUNK_SIZE}")
    print(f"{'='*80}\n")

    results = []

    for i in range(n_samples):
        seed = 42 + i * 1000
        print(f"\n--- Sample {i+1}/{n_samples} (seed={seed}) ---")

        # Build instruction (padded to chunk boundary)
        if "llama" in model.lower():
            instr_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Summarize the key themes discussed in each section below.\n\n"
            )
        else:
            instr_text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nSummarize the key themes discussed in each "
                "section below.\n\n"
            )
        instr_ids = tokenizer.encode(instr_text, add_special_tokens=False)
        instr_ids = _pad_to_chunks(instr_ids, pad_token)
        print(f"  Instruction: {len(instr_ids)} tokens ({len(instr_ids)//CHUNK_SIZE} chunks)")

        # Build 4 paragraphs (each padded to para_chunks × 256 tokens)
        para_ids_list = []
        for j, para_text in enumerate(_PARA_TEXTS):
            pids = _expand_text_to_tokens(
                para_text, tokenizer, para_chunks, seed + j
            )
            para_ids_list.append(pids)
            print(f"  Para {j} ({['A','B','C','D'][j]}): {len(pids)} tokens ({len(pids)//CHUNK_SIZE} chunks)")

        # Build suffix
        if "llama" in model.lower():
            suffix_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            suffix_text = "<|im_end|>\n<|im_start|>assistant\n"
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Donor: A-B-C-D order
        donor_ids = list(instr_ids)
        for pids in para_ids_list:
            donor_ids.extend(pids)
        donor_ids.extend(suffix_ids)

        # Target: D-C-B-A order (reversed paragraphs)
        target_ids = list(instr_ids)
        for pids in reversed(para_ids_list):
            target_ids.extend(pids)
        target_ids.extend(suffix_ids)

        print(f"  Donor total: {len(donor_ids)} tokens")
        print(f"  Target total: {len(target_ids)} tokens")

        # Verify chunk alignment: check that corresponding paragraph chunks
        # have the same token content (and thus the same hash)
        mismatches = 0
        for j in range(4):
            d_start = len(instr_ids) + j * (para_chunks * CHUNK_SIZE)
            t_start = len(instr_ids) + (3 - j) * (para_chunks * CHUNK_SIZE)
            for c in range(para_chunks):
                d_chunk = donor_ids[d_start + c * CHUNK_SIZE: d_start + (c + 1) * CHUNK_SIZE]
                t_chunk = target_ids[t_start + c * CHUNK_SIZE: t_start + (c + 1) * CHUNK_SIZE]
                if d_chunk != t_chunk:
                    mismatches += 1
        if mismatches > 0:
            print(f"  WARNING: {mismatches} chunk mismatches (alignment broken)")
        else:
            print(f"  Chunk alignment verified: all {4 * para_chunks} para chunks match")

        # Phase 1: Send donor (cold — registers in cache)
        print(f"  Sending donor (cold)...", end="", flush=True)
        try:
            donor_result = _send_request(
                endpoint, donor_ids, model, max_tokens=max_tokens
            )
            print(
                f" {donor_result['total_ms']:.0f}ms, "
                f"prompt={donor_result['prompt_tokens']}tok, "
                f"out={donor_result['completion_tokens']}tok"
            )
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Brief pause for registration
        time.sleep(0.5)

        # Phase 2: Send target (reversed order — should find donor semantically)
        print(f"  Sending target (reorder)...", end="", flush=True)
        try:
            target_result = _send_request(
                endpoint, target_ids, model, max_tokens=max_tokens
            )
            print(
                f" {target_result['total_ms']:.0f}ms, "
                f"prompt={target_result['prompt_tokens']}tok, "
                f"out={target_result['completion_tokens']}tok"
            )
        except Exception as e:
            print(f" ERROR: {e}")
            continue

        # Phase 3: Cold reference (unique content to avoid cache hit)
        # Use a different instruction that changes all chunk hashes
        if "llama" in model.lower():
            cold_instr = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                f"Analyze each section's arguments in detail (ref={seed}).\n\n"
            )
        else:
            cold_instr = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                f"<|im_start|>user\nAnalyze each section's arguments (ref={seed}).\n\n"
            )
        cold_instr_ids = tokenizer.encode(cold_instr, add_special_tokens=False)
        cold_instr_ids = _pad_to_chunks(cold_instr_ids, pad_token)
        cold_ids = list(cold_instr_ids)
        for pids in reversed(para_ids_list):
            cold_ids.extend(pids)
        cold_ids.extend(suffix_ids)

        print(f"  Sending cold reference...", end="", flush=True)
        try:
            cold_result = _send_request(
                endpoint, cold_ids, model, max_tokens=max_tokens
            )
            print(
                f" {cold_result['total_ms']:.0f}ms, "
                f"prompt={cold_result['prompt_tokens']}tok"
            )
        except Exception as e:
            print(f" ERROR: {e}")
            cold_result = None

        speedup = donor_result["total_ms"] / target_result["total_ms"]

        # Calculate expected position deltas
        deltas = {}
        for j in range(4):
            label = ["A", "B", "C", "D"][j]
            d_start = len(instr_ids) + j * (para_chunks * CHUNK_SIZE)
            t_start = len(instr_ids) + (3 - j) * (para_chunks * CHUNK_SIZE)
            delta = t_start - d_start
            deltas[label] = delta

        result = {
            "sample": i,
            "seed": seed,
            "donor_ms": round(donor_result["total_ms"], 1),
            "target_ms": round(target_result["total_ms"], 1),
            "cold_ms": round(cold_result["total_ms"], 1) if cold_result else None,
            "speedup": round(speedup, 3),
            "donor_tokens": donor_result["prompt_tokens"],
            "target_tokens": target_result["prompt_tokens"],
            "donor_completion": len(donor_result["text"]),
            "target_completion": len(target_result["text"]),
            "position_deltas": deltas,
            "chunk_mismatches": mismatches,
        }
        results.append(result)

        print(
            f"  → speedup: {speedup:.2f}x | "
            f"deltas: {deltas}"
        )

    # Summary
    if results:
        donor_times = [r["donor_ms"] for r in results]
        target_times = [r["target_ms"] for r in results]
        speedups = [r["speedup"] for r in results]
        cold_times = [r["cold_ms"] for r in results if r["cold_ms"]]

        print(f"\n{'='*80}")
        print("REORDER Benchmark Results")
        print(f"{'='*80}")
        print(f"  Samples:         {len(results)}")
        print(f"  Token length:    ~{results[0]['donor_tokens']}")
        print(f"  Para chunks:     {para_chunks} (×256 = {para_chunks * 256} tokens)")
        print(
            f"  Donor (cold):    mean={sum(donor_times)/len(donor_times):.0f}ms, "
            f"p50={sorted(donor_times)[len(donor_times)//2]:.0f}ms"
        )
        print(
            f"  Target (reorder): mean={sum(target_times)/len(target_times):.0f}ms, "
            f"p50={sorted(target_times)[len(target_times)//2]:.0f}ms"
        )
        if cold_times:
            print(
                f"  Cold reference:  mean={sum(cold_times)/len(cold_times):.0f}ms, "
                f"p50={sorted(cold_times)[len(cold_times)//2]:.0f}ms"
            )
        print(
            f"  Speedup:         mean={sum(speedups)/len(speedups):.2f}x, "
            f"p50={sorted(speedups)[len(speedups)//2]:.2f}x"
        )
        print(f"  Position deltas: {results[0]['position_deltas']}")
        print()
        print("  Key: speedup = donor_cold_ms / target_reorder_ms")
        print("  If target < cold_ref → SemBlend donor reuse is working")
        print("  Watch vLLM logs for '[SemBlend] RoPE correction APPLIED' messages")

        # Save results
        if output_path:
            output = {
                "endpoint": endpoint,
                "model": model,
                "n_samples": n_samples,
                "para_chunks": para_chunks,
                "max_tokens": max_tokens,
                "chunk_size": CHUNK_SIZE,
                "results": results,
                "summary": {
                    "donor_mean_ms": round(
                        sum(donor_times) / len(donor_times), 1
                    ),
                    "target_mean_ms": round(
                        sum(target_times) / len(target_times), 1
                    ),
                    "cold_mean_ms": round(
                        sum(cold_times) / len(cold_times), 1
                    )
                    if cold_times
                    else None,
                    "speedup_mean": round(
                        sum(speedups) / len(speedups), 3
                    ),
                },
            }
            os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
            with open(output_path, "w") as f:
                json.dump(output, f, indent=2)
            print(f"\nSaved to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="REORDER RoPE correction E2E benchmark"
    )
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8100",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument(
        "--para-chunks",
        type=int,
        default=3,
        help="Number of 256-token chunks per paragraph (3 = 768 tok/para, total ~3.3K)",
    )
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument(
        "--output",
        default="results/rope_reorder.json",
    )
    args = parser.parse_args()

    run_reorder_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        n_samples=args.n,
        para_chunks=args.para_chunks,
        max_tokens=args.max_tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
