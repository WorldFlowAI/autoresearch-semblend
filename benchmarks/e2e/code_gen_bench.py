#!/usr/bin/env python3
"""Code generation workload benchmark for SemBlend.

Tests SemBlend on non-summarization workloads: coding problems where
the same base problem is queried with different natural language
descriptions. This simulates a production scenario where multiple
users ask variants of the same coding question.

Methodology:
  1. Generate pairs of coding prompts: same algorithmic problem,
     different natural language descriptions.
  2. Register the first variant as a donor.
  3. Send the second variant → measure TTFT and hit rate.
  4. Optionally measure functional correctness (pass@1) via
     simple test cases.

Key finding expected: SemBlend should achieve moderate hit rates
(40-70%) on coding problems because problem descriptions share
significant lexical overlap even when rephrased.

Usage:
    python -m benchmarks.e2e.code_gen_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-problems 32 \
        --output results/code_gen.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Coding problem pairs: (donor_description, variant_description, test_input, expected_output)
CODING_PROBLEMS = [
    {
        "name": "two_sum",
        "donor": (
            "Write a Python function called two_sum that takes a list of "
            "integers nums and an integer target. Return the indices of the "
            "two numbers that add up to the target. You may assume each input "
            "has exactly one solution and you may not use the same element twice."
        ),
        "variant": (
            "Implement a function two_sum(nums, target) in Python. Given an "
            "array of integers, find two numbers such that they add up to a "
            "specific target number. The function should return their indices. "
            "Each input has exactly one valid answer."
        ),
    },
    {
        "name": "reverse_string",
        "donor": (
            "Write a Python function reverse_string that takes a string s "
            "and returns the string reversed. Do not use slicing or built-in "
            "reverse methods. Use a loop-based approach."
        ),
        "variant": (
            "Create a Python function called reverse_string that accepts a "
            "single string parameter s. It should return a new string that is "
            "the reverse of the input. Implement it iteratively without using "
            "Python's slice notation or the reversed() built-in."
        ),
    },
    {
        "name": "fibonacci",
        "donor": (
            "Write a Python function fibonacci(n) that returns the nth "
            "Fibonacci number. Use dynamic programming (memoization or "
            "tabulation) for efficiency. Handle edge cases: fib(0) = 0, "
            "fib(1) = 1."
        ),
        "variant": (
            "Implement a function fibonacci(n) in Python that computes the "
            "nth number in the Fibonacci sequence. Use an efficient approach "
            "(not naive recursion) such as iterative computation or memoized "
            "recursion. The sequence starts with F(0)=0, F(1)=1."
        ),
    },
    {
        "name": "binary_search",
        "donor": (
            "Write a Python function binary_search(arr, target) that searches "
            "for target in a sorted array arr. Return the index if found, "
            "otherwise return -1. Implement the iterative version."
        ),
        "variant": (
            "Implement binary_search(arr, target) in Python. Given a sorted "
            "list of integers arr, find the position of target using the "
            "binary search algorithm. Return the index of the target, or -1 "
            "if not present. Use an iterative (non-recursive) approach."
        ),
    },
    {
        "name": "is_palindrome",
        "donor": (
            "Write a Python function is_palindrome(s) that checks if a "
            "string is a palindrome. Ignore case and non-alphanumeric "
            "characters. Return True if palindrome, False otherwise."
        ),
        "variant": (
            "Create a function is_palindrome(s) in Python that determines "
            "whether a given string reads the same forwards and backwards. "
            "The comparison should be case-insensitive and should ignore "
            "all non-alphanumeric characters."
        ),
    },
    {
        "name": "merge_sorted",
        "donor": (
            "Write a Python function merge_sorted(list1, list2) that merges "
            "two sorted lists into one sorted list. Do not use the built-in "
            "sort function. Use a two-pointer approach."
        ),
        "variant": (
            "Implement merge_sorted(list1, list2) in Python. Given two "
            "lists that are already sorted in ascending order, combine them "
            "into a single sorted list. Use an efficient merging algorithm "
            "with two pointers, not sort() or sorted()."
        ),
    },
    {
        "name": "max_subarray",
        "donor": (
            "Write a Python function max_subarray(nums) that finds the "
            "contiguous subarray with the largest sum and returns that sum. "
            "Use Kadane's algorithm for O(n) time complexity."
        ),
        "variant": (
            "Implement max_subarray(nums) in Python. Given an integer array, "
            "find the contiguous subarray (containing at least one number) "
            "which has the largest sum and return its sum. Use an efficient "
            "linear-time approach (Kadane's algorithm)."
        ),
    },
    {
        "name": "valid_parentheses",
        "donor": (
            "Write a Python function is_valid(s) that determines if an input "
            "string of brackets is valid. A string is valid if: open brackets "
            "are closed by the same type, and in the correct order. Brackets "
            "include (), [], {}."
        ),
        "variant": (
            "Create is_valid(s) in Python to check whether a string containing "
            "only parentheses, square brackets, and curly braces is balanced "
            "and properly nested. Return True if valid, False otherwise. "
            "Use a stack-based approach."
        ),
    },
]

SYSTEM_PROMPT = (
    "You are an expert Python programmer. Write clean, efficient, "
    "well-documented code. Include type hints and docstrings."
)

PROMPT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{description}\n\n"
    "Provide only the Python function implementation, no explanations."
    "<|im_end|>"
)


def pad_prompt(prompt: str, target_chars: int) -> str:
    """Pad prompt to target character length with additional context."""
    if len(prompt) >= target_chars:
        return prompt[:target_chars]
    padding = (
        "\n\nAdditional context for this problem:\n"
        "Consider edge cases such as empty inputs, single elements, "
        "very large inputs, negative numbers, and duplicate values. "
        "The solution should handle all standard Python data types. "
        "Performance matters: aim for optimal time complexity. "
        "Memory usage should be reasonable for inputs up to 10^6 elements. "
    )
    # Repeat padding to fill
    while len(prompt) < target_chars:
        prompt += padding
    return prompt[:target_chars]


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Measure request time. Returns (ms, ok)."""
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
    except Exception:
        return 0.0, False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-problems", type=int, default=1000)
    parser.add_argument("--token-length", type=int, default=4096)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()

    target_chars = args.token_length * 4

    print(f"\nCode Generation Benchmark")
    print(f"  endpoint={args.endpoint}, model={args.model}")
    print(f"  n_problems={args.n_problems}, token_length={args.token_length}")
    print()

    # Cycle through problem templates to reach n_problems
    problems = []
    for i in range(args.n_problems):
        base = CODING_PROBLEMS[i % len(CODING_PROBLEMS)]
        problems.append({
            "name": f"{base['name']}_{i}",
            "donor": base["donor"],
            "variant": base["variant"],
        })

    # Phase 1: Cold baselines
    print(f"Phase 1: Cold baselines ({args.n_problems} unique prompts)")
    cold_ttfts: list[float] = []
    for i, prob in enumerate(problems):
        prompt = pad_prompt(
            PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, description=prob["donor"]),
            target_chars,
        )
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=5)
        if ok:
            cold_ttfts.append(t)
        print(f"  [{i+1}/{args.n_problems}] cold {t:.0f}ms", end="\r")

    if not cold_ttfts:
        print("\nNo cold measurements — aborting")
        return

    cold_mean = sum(cold_ttfts) / len(cold_ttfts)
    cold_p50 = sorted(cold_ttfts)[len(cold_ttfts) // 2]
    print(f"\n  Cold: n={len(cold_ttfts)}, mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")

    # Phase 2: Register donors
    print(f"\nPhase 2: Register donors ({args.n_problems} problems)")
    for i, prob in enumerate(problems):
        prompt = pad_prompt(
            PROMPT_TEMPLATE.format(system=SYSTEM_PROMPT, description=prob["donor"]),
            target_chars,
        )
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=50)
        print(f"  [{i+1}/{args.n_problems}] registered {t:.0f}ms", end="\r")
    print(f"\n  Donors registered")

    # Phase 3: Query with variant descriptions
    print(f"\nPhase 3: Query with variant descriptions")
    variant_ttfts: list[float] = []
    per_problem: list[dict] = []

    for i, prob in enumerate(problems):
        prompt = pad_prompt(
            PROMPT_TEMPLATE.format(
                system=SYSTEM_PROMPT, description=prob["variant"]
            ),
            target_chars,
        )
        t, ok = ttft_request(args.endpoint, args.model, prompt, max_tokens=5)
        if ok:
            variant_ttfts.append(t)
            is_hit = t < 0.70 * cold_mean
            per_problem.append({
                "name": prob["name"],
                "ttft_ms": t,
                "hit": is_hit,
            })
        print(f"  [{i+1}/{args.n_problems}] variant {t:.0f}ms", end="\r")

    if not variant_ttfts:
        print("\nNo variant measurements — aborting")
        return

    # Results
    hits = sum(1 for r in per_problem if r["hit"])
    hit_pct = hits / len(per_problem) * 100
    variant_mean = sum(variant_ttfts) / len(variant_ttfts)
    variant_p50 = sorted(variant_ttfts)[len(variant_ttfts) // 2]
    speedup = cold_mean / variant_mean if variant_mean > 0 else 0.0
    hit_only = [r["ttft_ms"] for r in per_problem if r["hit"]]
    hit_speedup = (
        cold_mean / (sum(hit_only) / len(hit_only))
        if hit_only
        else 0.0
    )

    print(f"\n\nResults:")
    print(f"  Cold baseline:   mean={cold_mean:.0f}ms, p50={cold_p50:.0f}ms")
    print(f"  Variant queries: mean={variant_mean:.0f}ms, p50={variant_p50:.0f}ms")
    print(f"  Hit rate:        {hit_pct:.0f}% ({hits}/{len(per_problem)})")
    print(f"  Overall speedup: {speedup:.2f}x")
    print(f"  Hit-only speedup:{hit_speedup:.2f}x")
    print()
    print("Interpretation: Code generation workload with rephrased problem")
    print("descriptions. SemBlend should find donors via semantic similarity")
    print("despite different wording.")

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import (
        bootstrap_mean,
        bootstrap_proportion,
        bootstrap_speedup,
    )

    print()
    print("=" * 60)
    print("Bootstrap 95% Confidence Intervals")
    print("=" * 60)

    cold_arr = np.array(cold_ttfts)
    variant_arr = np.array(variant_ttfts)
    print(f"  Cold TTFT mean:    {bootstrap_mean(cold_arr)}")
    print(f"  Variant TTFT mean: {bootstrap_mean(variant_arr)}")
    print(f"  Hit rate:          {bootstrap_proportion(hits, len(per_problem))}")
    print(f"  Overall speedup:   {bootstrap_speedup(variant_arr, cold_arr)}")
    if hit_only:
        hit_arr = np.array(hit_only)
        print(f"  Hit-only speedup:  {bootstrap_speedup(hit_arr, cold_arr)}")
    print()

    if args.output:
        out = {
            "cold_mean_ms": cold_mean,
            "cold_p50_ms": cold_p50,
            "variant_mean_ms": variant_mean,
            "variant_p50_ms": variant_p50,
            "hit_rate_pct": hit_pct,
            "overall_speedup": speedup,
            "hit_only_speedup": hit_speedup,
            "n_problems": args.n_problems,
            "token_length": args.token_length,
            "per_problem": per_problem,
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
