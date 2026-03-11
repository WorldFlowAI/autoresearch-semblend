#!/usr/bin/env python3
"""Multi-turn dialogue benchmark for SemBlend KV cache reuse.

Tests SemBlend on multi-turn chat — a non-summarization workload.
In multi-turn conversations, each successive turn shares a growing prefix
with the prior turn. SemBlend should find turn N as a semantic donor for
turn N+1 because 90%+ of the text is identical (same prefix + new message).

Expected behavior:
  Turn 1: cold prefill (no donor)
  Turn 2: SemBlend hit on turn 1 → speedup
  Turn 3: SemBlend hit on turn 2 → speedup
  ...
  Turn N: SemBlend hit on turn N-1 → speedup (growing prefix reuse)

Usage:
    python -m benchmarks.e2e.multiturn_dialogue_bench \\
        --endpoint http://localhost:8100 \\
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \\
        --n-conversations 32 \\
        --max-turns 6 \\
        --token-length 4096
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from statistics import mean, median

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# ---------------------------------------------------------------------------
# Conversation templates — alternating user/assistant messages
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = "You are a helpful AI assistant."

CONVERSATION_TEMPLATES = [
    # Tech support conversation
    [
        "I'm having trouble with my laptop. It's running very slowly and the fan is always on.",
        "That could be several things. Can you open Task Manager and tell me what's using the most CPU?",
        "It says Chrome is using 85% of CPU with 47 tabs open.",
        "That's likely the issue. Try closing unnecessary tabs and see if performance improves.",
        "I closed most tabs and it's much better now. But the fan is still loud.",
        "The fan might need cleaning. When was the last time you cleaned the laptop vents?",
    ],
    # Travel planning conversation
    [
        "I'm planning a trip to Japan in spring. What should I know?",
        "Spring is wonderful for cherry blossoms! Peak season is late March to mid-April.",
        "Where are the best spots for cherry blossoms in Tokyo?",
        "Ueno Park, Shinjuku Gyoen, and Chidorigafuchi are top spots. Book hotels early.",
        "What about food recommendations? I'm vegetarian.",
        "Japanese cuisine is challenging for vegetarians. Look for shojin ryori (temple cuisine).",
    ],
    # Cooking conversation
    [
        "I want to learn how to make authentic Italian pasta from scratch. Where do I start?",
        "Start with a simple egg pasta: 100g flour per egg. Make a well, crack eggs in, mix with a fork.",
        "How long should I knead the dough?",
        "Knead for about 10 minutes until smooth and elastic. Then rest it wrapped for 30 minutes.",
        "What sauce pairs best with fresh tagliatelle?",
        "A classic ragu bolognese is perfect with tagliatelle. Use a mix of beef and pork mince.",
    ],
    # Programming help conversation
    [
        "I'm getting a segmentation fault in my C program. How do I debug it?",
        "Use valgrind or GDB to find the exact line. Run: valgrind --track-origins=yes ./program",
        "Valgrind says invalid read of size 4 at line 42. What does that mean?",
        "You're reading memory that wasn't allocated or was already freed. Check array bounds at line 42.",
        "I found it — I was accessing array[10] but only allocated 10 elements (0-9). Fixed!",
        "Great catch! Off-by-one errors are very common. Consider using sanitizers: -fsanitize=address.",
    ],
    # Fitness advice conversation
    [
        "I want to start running but I'm completely out of shape. Any advice?",
        "Start with a walk-run program: alternate 1 minute running, 2 minutes walking for 20 minutes.",
        "How many times per week should I do this?",
        "Three times per week with rest days between. Your body needs recovery time to adapt.",
        "After two weeks I can run for 3 minutes straight. Should I increase distance or speed?",
        "Focus on distance first. Add 10% more total time each week. Speed comes naturally later.",
    ],
    # Home renovation conversation
    [
        "I'm thinking about renovating my kitchen. Where should I start planning?",
        "Start with your budget and layout. The work triangle between sink, stove, and fridge is key.",
        "My budget is around $25,000. Is that realistic for a full kitchen remodel?",
        "For a mid-range remodel, $25K is workable but tight. Cabinets take 30-40% of the budget.",
        "Should I go with custom cabinets or stock ones from a big box store?",
        "Stock or semi-custom cabinets save 40-60% over full custom. RTA cabinets are another option.",
    ],
    # Financial planning conversation
    [
        "I just got my first real job making $65,000 a year. How should I manage my money?",
        "Follow the 50/30/20 rule: 50% needs, 30% wants, 20% savings. Start an emergency fund first.",
        "How much should I keep in my emergency fund?",
        "Aim for 3-6 months of expenses. At your income, that's roughly $8,000 to $16,000.",
        "Should I pay off my student loans faster or invest in my 401k?",
        "If your employer matches 401k contributions, get the full match first. It's free money.",
    ],
    # Pet care conversation
    [
        "I'm adopting a rescue dog next week. What do I need to prepare?",
        "Get the basics: crate, bed, food bowls, leash, collar with ID tag, and age-appropriate food.",
        "The shelter said the dog is anxious around strangers. How should I handle introductions?",
        "Give the dog space and let them approach people on their own terms. No forced interactions.",
        "What about introducing the dog to my cat?",
        "Keep them separated for the first week. Use scent swapping — trade blankets between them.",
    ],
    # Photography conversation
    [
        "I bought my first DSLR camera. What settings should a beginner learn first?",
        "Start with the exposure triangle: aperture, shutter speed, and ISO. Try aperture priority mode.",
        "What aperture should I use for portraits?",
        "Use f/1.8 to f/2.8 for blurry backgrounds, or f/5.6 for group shots where everyone's sharp.",
        "My indoor photos keep coming out blurry. What am I doing wrong?",
        "Likely too slow a shutter speed. Use at least 1/focal length. Bump ISO to 800-1600 indoors.",
    ],
    # Gardening conversation
    [
        "I want to start a vegetable garden but I only have a small balcony. Is it possible?",
        "Absolutely! Container gardening works great. Tomatoes, herbs, peppers, and lettuce all do well.",
        "What size containers do I need for tomatoes?",
        "At least 5-gallon containers for determinate varieties. Indeterminate types need 10+ gallons.",
        "How often should I water container vegetables?",
        "Check soil daily — containers dry out fast. Water when the top inch feels dry, usually daily in summer.",
    ],
]

# Extended context padding for longer token targets. This background document
# gets prepended to the system prompt to inflate the context window.
BACKGROUND_DOCUMENT = """COMPREHENSIVE REFERENCE GUIDE

Section 1: Communication Best Practices
Effective communication requires clarity, empathy, and active listening. When
responding to questions, consider the context, the user's level of expertise,
and what additional information might be helpful. Use concrete examples when
explaining abstract concepts. Break complex topics into manageable steps.
Avoid jargon unless the audience is technical. Always verify understanding
by summarizing key points. Non-verbal cues in written communication include
formatting, tone words, and structural organization. Prioritize actionable
advice over theoretical frameworks when users seek practical help.

Section 2: Problem-Solving Methodology
When approaching problems, start by clearly defining the issue. Gather all
relevant information before proposing solutions. Consider multiple approaches
and evaluate trade-offs. Break large problems into smaller sub-problems.
Test solutions incrementally rather than implementing everything at once.
Document what works and what doesn't for future reference. Collaborate with
others when stuck — fresh perspectives often reveal blind spots. Use the
scientific method: hypothesize, test, observe, adjust. Be willing to
abandon approaches that aren't working rather than sinking more time into
them. Celebrate progress, not just completion.

Section 3: Knowledge Organization
Information is most useful when it's organized for retrieval. Use categories,
tags, and cross-references to build a personal knowledge system. Spaced
repetition helps with retention of factual information. Connect new knowledge
to existing frameworks to deepen understanding. Teaching others is one of the
most effective ways to solidify your own understanding. Write summaries of
important topics in your own words. Maintain a distinction between facts,
interpretations, and opinions. Update your knowledge regularly as fields
evolve. Be comfortable with uncertainty — knowing the limits of your
knowledge is itself valuable knowledge.

Section 4: Time Management and Productivity
The most productive people focus on high-impact tasks first. Use time-boxing
to prevent tasks from expanding to fill available time. Batch similar tasks
together to reduce context-switching costs. Protect deep work time from
interruptions. The Pomodoro technique (25 minutes work, 5 minutes break)
helps maintain focus. Review your priorities weekly and adjust based on
changing circumstances. Learn to say no to low-priority requests. Automate
repetitive tasks whenever possible. Energy management is as important as
time management — schedule demanding work for your peak hours.

Section 5: Health and Wellness Fundamentals
Physical health forms the foundation for cognitive performance. Regular
exercise improves memory, focus, and mood through increased BDNF production.
Sleep 7-9 hours consistently — sleep debt cannot be fully repaid. Nutrition
affects cognition: omega-3 fatty acids, antioxidants, and adequate hydration
support brain function. Manage stress through regular breaks, social
connection, and mindfulness practices. Screen time before bed disrupts
circadian rhythms. Stand and move every 30-60 minutes during sedentary work.
Regular health checkups catch issues early when they're most treatable.

Section 6: Learning and Skill Acquisition
The most effective learners use deliberate practice: focused effort on
specific weaknesses with immediate feedback. The 10,000-hour rule is
misleading — quality of practice matters more than quantity. Use interleaving
(mixing different topics) rather than blocking (one topic at a time) for
deeper learning. Retrieval practice (testing yourself) is more effective than
re-reading. Set specific, measurable learning goals with deadlines. Find
mentors who can accelerate your learning by sharing their experience. Accept
that discomfort is a sign of growth — if learning feels easy, you're probably
not pushing hard enough. Build learning habits that are sustainable over years.

Section 7: Technology and Digital Literacy
Understanding technology requires both breadth and depth. Learn fundamental
concepts (networking, data structures, algorithms) rather than memorizing
specific tools. Stay current with major technology trends but don't chase
every new framework. Security hygiene is essential: use strong unique
passwords, enable two-factor authentication, keep software updated. Back up
important data using the 3-2-1 rule (3 copies, 2 media types, 1 offsite).
Understand privacy trade-offs when using free services. Automate backups
and updates to reduce friction. Learn to evaluate technology claims
critically — not every innovation lives up to its marketing.

Section 8: Environmental Awareness
Individual actions matter when multiplied across populations. Reduce
consumption before recycling — the most sustainable product is the one
you don't buy. Energy efficiency improvements often pay for themselves.
Understand the carbon footprint of common activities: flying is high-impact,
dietary choices matter more than most people realize. Support systemic
changes through voting and civic engagement. Local food systems reduce
transportation emissions and support community resilience. Water conservation
becomes more critical as climate change affects precipitation patterns.
Educate others through example rather than lectures."""


def build_chat_prompt(system: str, turns: list[str], turn_index: int) -> str:
    """Build a ChatML-formatted prompt from conversation turns up to turn_index.

    Even-indexed turns are user messages, odd-indexed are assistant messages.
    The final message is always a user turn (so the model generates assistant).
    """
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]

    # Include all completed turn pairs plus the current user turn
    for i in range(turn_index + 1):
        role = "user" if i % 2 == 0 else "assistant"
        parts.append(f"<|im_start|>{role}\n{turns[i]}<|im_end|>")

    # If the last included turn was an assistant turn, we need to set up
    # for the next user turn — but we only build prompts ending with a user turn
    # so the model generates the next assistant response.

    return "\n".join(parts)


def pad_system_prompt(base_system: str, target_chars: int, current_chars: int) -> str:
    """Pad the system prompt to reach the target total character count.

    Adds chunks of BACKGROUND_DOCUMENT to the system prompt until the total
    prompt (system + conversation) would reach approximately target_chars.
    """
    deficit = target_chars - current_chars
    if deficit <= 0:
        return base_system

    padding = BACKGROUND_DOCUMENT
    while len(padding) < deficit:
        padding += "\n\n" + BACKGROUND_DOCUMENT
    padding = padding[:deficit]

    return base_system + "\n\n" + padding


def generate_conversations(
    n_conversations: int,
    max_turns: int,
    target_chars: int,
) -> list[dict]:
    """Generate synthetic multi-turn conversations.

    Returns a list of conversation dicts, each containing:
      - system: the (possibly padded) system prompt
      - turns: list of alternating user/assistant message strings
      - turn_prompts: list of full ChatML prompts for each user turn
    """
    conversations = []
    n_templates = len(CONVERSATION_TEMPLATES)

    for conv_idx in range(n_conversations):
        template = CONVERSATION_TEMPLATES[conv_idx % n_templates]
        # Use only up to max_turns messages, ensuring we end on a user turn
        # (even index = user). We need pairs, so max_turns user messages means
        # 2*max_turns - 1 total messages (but templates may be shorter).
        usable_turns = min(len(template), max_turns * 2)
        turns = list(template[:usable_turns])

        # Build the prompt for the LAST user turn (largest context) to
        # measure how much padding we need
        last_user_idx = usable_turns - 1 if usable_turns % 2 == 1 else usable_turns - 2
        test_prompt = build_chat_prompt(SYSTEM_PROMPT, turns, last_user_idx)
        current_chars = len(test_prompt)

        # Pad system prompt if needed
        system = pad_system_prompt(SYSTEM_PROMPT, target_chars, current_chars)

        # Build prompts for each user turn (even indices: 0, 2, 4, ...)
        turn_prompts = []
        for t_idx in range(0, usable_turns, 2):
            prompt = build_chat_prompt(system, turns, t_idx)
            turn_prompts.append({
                "turn_number": t_idx // 2 + 1,
                "prompt": prompt,
                "prompt_chars": len(prompt),
                "new_content": turns[t_idx],
                "new_chars": len(turns[t_idx]),
            })

        conversations.append({
            "conv_idx": conv_idx,
            "template_idx": conv_idx % n_templates,
            "system": system,
            "turns": turns,
            "turn_prompts": turn_prompts,
        })

    return conversations


def ttft_request(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 5,
) -> tuple[float, bool]:
    """Send a /v1/completions request and measure total time. Returns (ms, ok)."""
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


def estimate_tokens(char_count: int) -> int:
    """Rough token estimate: ~4 chars per token for English text."""
    return char_count // 4


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Multi-turn dialogue benchmark for SemBlend KV cache reuse"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-conversations", type=int, default=200)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--token-length", type=int, default=4096,
                        help="Target token length for the final turn context")
    parser.add_argument("--output", default=None, help="JSON output path")
    args = parser.parse_args()

    # Health check
    try:
        resp = requests.get(f"{args.endpoint}/health", timeout=10)
        resp.raise_for_status()
    except Exception as exc:
        print(f"Cannot reach endpoint {args.endpoint}: {exc}")
        return

    target_chars = args.token_length * 4  # ~4 chars/token

    print(f"\nMulti-Turn Dialogue Benchmark")
    print(f"  endpoint      = {args.endpoint}")
    print(f"  model         = {args.model}")
    print(f"  conversations = {args.n_conversations}")
    print(f"  max_turns     = {args.max_turns}")
    print(f"  token_length  = {args.token_length}")
    print()
    print("Theory: Each turn N+1 shares 90%+ of its context with turn N.")
    print("        SemBlend should find turn N as a semantic donor, reusing")
    print("        cached KV entries and reducing TTFT for subsequent turns.")
    print()

    # Generate conversations
    conversations = generate_conversations(
        args.n_conversations, args.max_turns, target_chars
    )

    # Determine the maximum number of user turns across all conversations
    max_user_turns = max(len(c["turn_prompts"]) for c in conversations)

    # Collect per-turn measurements across all conversations
    # turn_results[turn_number] = list of (ttft_ms, prompt_chars, new_chars)
    turn_results: dict[int, list[tuple[float, int, int]]] = {}
    for turn_num in range(1, max_user_turns + 1):
        turn_results[turn_num] = []

    # Run benchmark
    total_convs = len(conversations)
    for c_idx, conv in enumerate(conversations):
        n_turns = len(conv["turn_prompts"])
        print(f"  Conversation [{c_idx + 1}/{total_convs}] "
              f"({n_turns} user turns)", end="")

        for turn_info in conv["turn_prompts"]:
            turn_num = turn_info["turn_number"]
            prompt = turn_info["prompt"]

            # For turn 1, use more max_tokens to ensure LMCache stores the KV
            # (longer generation = donor registration via SemBlend connector)
            gen_tokens = 50 if turn_num == 1 else 5

            ttft_ms, ok = ttft_request(
                args.endpoint, args.model, prompt, max_tokens=gen_tokens
            )

            if ok:
                turn_results[turn_num].append((
                    ttft_ms,
                    turn_info["prompt_chars"],
                    turn_info["new_chars"],
                ))
                print(f"  T{turn_num}={ttft_ms:.0f}ms", end="")
            else:
                print(f"  T{turn_num}=ERR", end="")

        print()

    # Compute and display results
    print()
    print("=" * 85)
    print("Multi-Turn Dialogue Benchmark Results")
    print("=" * 85)
    print(f"  Model: {args.model}")
    print(f"  Conversations: {total_convs}")
    print()

    header = (f"{'Turn':>4} | {'Context Tokens':>14} | {'New Tokens':>10} | "
              f"{'Hit Rate':>8} | {'P50 TTFT':>10} | {'Mean TTFT':>10} | "
              f"{'Speedup':>7}")
    print(header)
    print("-" * 85)

    turn_summaries = []
    cold_p50 = None

    for turn_num in sorted(turn_results.keys()):
        measurements = turn_results[turn_num]
        if not measurements:
            continue

        ttfts = [m[0] for m in measurements]
        context_chars = [m[1] for m in measurements]
        new_chars = [m[2] for m in measurements]

        sorted_ttfts = sorted(ttfts)
        p50 = sorted_ttfts[len(sorted_ttfts) // 2]
        mean_ttft = mean(ttfts)
        med_context_tokens = estimate_tokens(
            sorted(context_chars)[len(context_chars) // 2]
        )
        med_new_tokens = estimate_tokens(
            sorted(new_chars)[len(new_chars) // 2]
        )

        # Turn 1 is always cold (no donor exists)
        if turn_num == 1:
            cold_p50 = p50
            hit_rate = 0.0
            speedup = 1.0
        else:
            # Hit detection: TTFT < 70% of cold P50 → likely cache hit
            if cold_p50 and cold_p50 > 0:
                hits = sum(1 for t in ttfts if t < 0.70 * cold_p50)
                hit_rate = hits / len(ttfts) * 100
                speedup = cold_p50 / p50 if p50 > 0 else 0.0
            else:
                hit_rate = 0.0
                speedup = 0.0

        print(f"{turn_num:>4} | {med_context_tokens:>14} | {med_new_tokens:>10} | "
              f"{hit_rate:>7.0f}% | {p50:>8.0f}ms | {mean_ttft:>8.0f}ms | "
              f"{speedup:>6.1f}x{'  (cold)' if turn_num == 1 else ''}")

        turn_summaries.append({
            "turn": turn_num,
            "n": len(ttfts),
            "context_tokens_p50": med_context_tokens,
            "new_tokens_p50": med_new_tokens,
            "p50_ms": round(p50, 1),
            "mean_ms": round(mean_ttft, 1),
            "min_ms": round(min(ttfts), 1),
            "max_ms": round(max(ttfts), 1),
            "hit_rate_pct": round(hit_rate, 1),
            "speedup_vs_cold": round(speedup, 2),
        })

    print("-" * 85)

    # Overall summary for turns 2+
    subsequent_turns = [s for s in turn_summaries if s["turn"] > 1]
    if subsequent_turns:
        avg_speedup = mean(s["speedup_vs_cold"] for s in subsequent_turns)
        avg_hit_rate = mean(s["hit_rate_pct"] for s in subsequent_turns)
        print(f"\nSubsequent turns (2+): avg speedup = {avg_speedup:.2f}x, "
              f"avg hit rate = {avg_hit_rate:.0f}%")
    print()
    print("Interpretation:")
    print("  Turn 1 is always cold (no prior context to reuse).")
    print("  Turns 2+ should show SemBlend hits because each turn's context")
    print("  is ~90%+ identical to the prior turn (same prefix + new message).")
    if subsequent_turns:
        if avg_speedup > 1.3:
            print(f"  Result: SemBlend provides {avg_speedup:.1f}x speedup on "
                  f"multi-turn dialogue.")
        else:
            print("  Result: Speedup below expected — check SemBlend configuration "
                  "and LMCache chunk alignment.")

    # Bootstrap CI summary
    from benchmarks.e2e.bootstrap_ci import (
        bootstrap_mean,
        bootstrap_proportion,
        bootstrap_speedup,
    )

    print()
    print("=" * 85)
    print("Bootstrap 95% Confidence Intervals")
    print("=" * 85)

    cold_ttft_arr = np.array([m[0] for m in turn_results.get(1, [])])
    if len(cold_ttft_arr) > 0:
        print(f"  Turn 1 (cold) TTFT mean: {bootstrap_mean(cold_ttft_arr)}")

    for turn_num in sorted(turn_results.keys()):
        if turn_num == 1:
            continue
        measurements = turn_results[turn_num]
        if not measurements:
            continue
        ttfts = np.array([m[0] for m in measurements])
        hits = int(sum(1 for t in ttfts if cold_p50 and t < 0.70 * cold_p50))
        total = len(ttfts)
        print(f"\n  Turn {turn_num}:")
        print(f"    TTFT mean:  {bootstrap_mean(ttfts)}")
        print(f"    Hit rate:   {bootstrap_proportion(hits, total)}")
        if len(cold_ttft_arr) > 0 and len(ttfts) > 0:
            print(f"    Speedup:    {bootstrap_speedup(ttfts, cold_ttft_arr)}")
    print()

    # Save results
    if args.output:
        output_data = {
            "config": {
                "endpoint": args.endpoint,
                "model": args.model,
                "n_conversations": args.n_conversations,
                "max_turns": args.max_turns,
                "token_length": args.token_length,
            },
            "cold_p50_ms": round(cold_p50, 1) if cold_p50 else None,
            "turn_summaries": turn_summaries,
        }
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(output_data, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
