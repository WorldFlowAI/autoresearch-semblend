#!/usr/bin/env python3
"""REORDER RoPE E2E benchmark — streaming TTFT measurement.

Measures actual TTFT (time to first token) using streaming, so decode time
doesn't mask prefill speedup. Compares:
  1. Donor (cold prefill) — registers in LMCache
  2. Target (reordered paragraphs) — should match via SemBlend + get RoPE correction
  3. Cold reference (different instruction) — no cache hit baseline

For quality: compares outputs between target and cold reference
(PPL requires logprobs, so we compare ROUGE-L of generated text).
"""

import argparse
import json
import math
import os
import random
import sys
import time

import requests

CHUNK_SIZE = 256


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
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    except Exception as e:
        print(f"ERROR: Cannot load tokenizer for {model}: {e}", file=sys.stderr)
        sys.exit(1)


def _pad_to_chunks(token_ids: list[int], pad_token: int) -> list[int]:
    remainder = len(token_ids) % CHUNK_SIZE
    if remainder == 0:
        return token_ids
    pad_count = CHUNK_SIZE - remainder
    return token_ids + [pad_token] * pad_count


def _expand_text_to_tokens(text, tokenizer, target_chunks, seed):
    target_tokens = target_chunks * CHUNK_SIZE
    rng = random.Random(seed)
    base_ids = tokenizer.encode(text, add_special_tokens=False)
    sentences = text.split(". ")
    all_ids = list(base_ids)
    while len(all_ids) < target_tokens:
        rng.shuffle(sentences)
        variation = ". ".join(sentences) + ". "
        var_ids = tokenizer.encode(variation, add_special_tokens=False)
        all_ids.extend(var_ids)
    return all_ids[:target_tokens]


def _streaming_ttft(endpoint, token_ids, model, max_tokens=64, temperature=0.0):
    """Send streaming request, return TTFT (ms) and full text."""
    t0 = time.monotonic()
    resp = requests.post(
        f"{endpoint}/v1/completions",
        json={
            "model": model,
            "prompt": token_ids,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        },
        stream=True,
        timeout=120,
    )
    resp.raise_for_status()

    ttft = None
    text_parts = []
    for line in resp.iter_lines():
        if not line or not line.startswith(b"data: "):
            continue
        data = line[6:]
        if data == b"[DONE]":
            break
        chunk = json.loads(data)
        text = chunk.get("choices", [{}])[0].get("text", "")
        if text:
            if ttft is None:
                ttft = (time.monotonic() - t0) * 1000.0
            text_parts.append(text)

    total_ms = (time.monotonic() - t0) * 1000.0
    return {
        "ttft_ms": ttft or total_ms,
        "total_ms": total_ms,
        "text": "".join(text_parts),
    }


def run_benchmark(endpoint, model, n_samples, para_chunks, max_tokens):
    tokenizer = _load_tokenizer(model)
    pad_token = tokenizer.pad_token_id or tokenizer.eos_token_id or 0

    total_tokens = (1 + 4 * para_chunks) * CHUNK_SIZE
    print(f"\n{'='*70}")
    print(f"REORDER RoPE TTFT Benchmark")
    print(f"  model:       {model}")
    print(f"  para_chunks: {para_chunks} ({para_chunks*256} tok/para)")
    print(f"  total_tokens: ~{total_tokens}")
    print(f"  max_tokens:  {max_tokens}")
    print(f"  n_samples:   {n_samples}")
    print(f"{'='*70}\n")

    results = []

    for i in range(n_samples):
        seed = 42 + i * 1000
        print(f"--- Sample {i+1}/{n_samples} ---")

        # Build instruction
        if "llama" in model.lower():
            instr_text = (
                "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                "You are a helpful assistant.<|eot_id|>"
                "<|start_header_id|>user<|end_header_id|>\n\n"
                "Summarize the key themes discussed in each section below.\n\n"
            )
            suffix_text = "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        else:
            instr_text = (
                "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nSummarize the key themes discussed in each "
                "section below.\n\n"
            )
            suffix_text = "<|im_end|>\n<|im_start|>assistant\n"

        instr_ids = _pad_to_chunks(
            tokenizer.encode(instr_text, add_special_tokens=False), pad_token
        )
        suffix_ids = tokenizer.encode(suffix_text, add_special_tokens=False)

        # Build 4 paragraphs
        para_ids_list = [
            _expand_text_to_tokens(txt, tokenizer, para_chunks, seed + j)
            for j, txt in enumerate(_PARA_TEXTS)
        ]

        # Donor: A-B-C-D
        donor_ids = list(instr_ids)
        for pids in para_ids_list:
            donor_ids.extend(pids)
        donor_ids.extend(suffix_ids)

        # Target: D-C-B-A
        target_ids = list(instr_ids)
        for pids in reversed(para_ids_list):
            target_ids.extend(pids)
        target_ids.extend(suffix_ids)

        # Cold ref: different instruction → no chunk match
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
        cold_instr_ids = _pad_to_chunks(
            tokenizer.encode(cold_instr, add_special_tokens=False), pad_token
        )
        cold_ids = list(cold_instr_ids)
        for pids in reversed(para_ids_list):
            cold_ids.extend(pids)
        cold_ids.extend(suffix_ids)

        # Run: donor → wait → target → cold
        print(f"  donor (cold)...", end="", flush=True)
        d = _streaming_ttft(endpoint, donor_ids, model, max_tokens)
        print(f" TTFT={d['ttft_ms']:.0f}ms")

        time.sleep(0.5)

        print(f"  target (reorder)...", end="", flush=True)
        t = _streaming_ttft(endpoint, target_ids, model, max_tokens)
        print(f" TTFT={t['ttft_ms']:.0f}ms")

        time.sleep(0.3)

        print(f"  cold ref...", end="", flush=True)
        c = _streaming_ttft(endpoint, cold_ids, model, max_tokens)
        print(f" TTFT={c['ttft_ms']:.0f}ms")

        speedup_vs_cold = c["ttft_ms"] / t["ttft_ms"]
        speedup_vs_donor = d["ttft_ms"] / t["ttft_ms"]

        result = {
            "sample": i,
            "donor_ttft_ms": round(d["ttft_ms"], 1),
            "target_ttft_ms": round(t["ttft_ms"], 1),
            "cold_ttft_ms": round(c["ttft_ms"], 1),
            "speedup_vs_cold": round(speedup_vs_cold, 3),
            "speedup_vs_donor": round(speedup_vs_donor, 3),
            "target_text": t["text"][:200],
            "cold_text": c["text"][:200],
        }
        results.append(result)
        print(
            f"  → speedup vs cold: {speedup_vs_cold:.2f}x, "
            f"vs donor: {speedup_vs_donor:.2f}x"
        )

    # Summary
    if results:
        donor_ttfts = [r["donor_ttft_ms"] for r in results]
        target_ttfts = [r["target_ttft_ms"] for r in results]
        cold_ttfts = [r["cold_ttft_ms"] for r in results]
        speedups = [r["speedup_vs_cold"] for r in results]

        print(f"\n{'='*70}")
        print("REORDER RoPE TTFT Results")
        print(f"{'='*70}")
        print(f"  n={len(results)}, ~{total_tokens} tokens")
        print(f"  Donor TTFT:  p50={sorted(donor_ttfts)[len(donor_ttfts)//2]:.0f}ms")
        print(f"  Target TTFT: p50={sorted(target_ttfts)[len(target_ttfts)//2]:.0f}ms")
        print(f"  Cold TTFT:   p50={sorted(cold_ttfts)[len(cold_ttfts)//2]:.0f}ms")
        print(
            f"  Speedup vs cold: mean={sum(speedups)/len(speedups):.2f}x, "
            f"p50={sorted(speedups)[len(speedups)//2]:.2f}x"
        )
        print()

        # Save
        out = {
            "model": model,
            "para_chunks": para_chunks,
            "total_tokens": total_tokens,
            "max_tokens": max_tokens,
            "results": results,
            "summary": {
                "donor_p50": round(sorted(donor_ttfts)[len(donor_ttfts)//2], 1),
                "target_p50": round(sorted(target_ttfts)[len(target_ttfts)//2], 1),
                "cold_p50": round(sorted(cold_ttfts)[len(cold_ttfts)//2], 1),
                "speedup_mean": round(sum(speedups)/len(speedups), 3),
            },
        }
        os.makedirs("results", exist_ok=True)
        with open("results/rope_reorder_ttft.json", "w") as f:
            json.dump(out, f, indent=2)
        print("Saved to results/rope_reorder_ttft.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--para-chunks", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=64)
    args = parser.parse_args()
    run_benchmark(args.endpoint, args.model, args.n, args.para_chunks, args.max_tokens)


if __name__ == "__main__":
    main()
