"""Diagnostic: verify SemBlend REORDER donor matching after v6 fixes.

Tests that registering a donor and sending a reordered version of the same
content results in a SemBlend HIT (not MISS). This validates:
  1. _register_donor only uses prompt tokens (not output tokens)
  2. ONNX max_length=256 captures enough text for stable embeddings
  3. _order_invariant_text makes embedding invariant to chunk reordering
"""
import json
import random
import sys
import time
import requests

VLLM_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

# 4 substantial chunks (~150 tokens each) for >256 token prompts
CHUNKS = [
    (
        "The semiconductor industry has entered a transformative era driven by advanced "
        "extreme ultraviolet lithography systems. These EUV machines, operating at 13.5nm "
        "wavelength, enable the fabrication of transistors at 3nm and below process nodes. "
        "The optical systems require multilayer mirrors with over 100 alternating layers of "
        "molybdenum and silicon, each deposited with sub-angstrom precision. The light source "
        "uses a tin droplet plasma that must be maintained at temperatures exceeding 500,000 "
        "degrees Celsius to produce adequate photon flux for wafer exposure."
    ),
    (
        "Quantum error correction represents one of the most significant challenges in "
        "building practical quantum computers. The surface code architecture requires "
        "thousands of physical qubits to encode a single logical qubit with sufficiently "
        "low error rates. Recent breakthroughs in superconducting transmon qubits have "
        "demonstrated two-qubit gate fidelities exceeding 99.5 percent, approaching the "
        "threshold needed for fault-tolerant computation. The cryogenic infrastructure "
        "required to maintain these systems at millikelvin temperatures adds substantial "
        "engineering complexity to any commercial quantum computing deployment."
    ),
    (
        "Advanced battery technology for electric vehicles continues to evolve rapidly "
        "with solid-state electrolytes emerging as the leading candidate for next-generation "
        "energy storage. Unlike conventional lithium-ion batteries that use liquid organic "
        "electrolytes, solid-state designs employ ceramic or polymer electrolytes that "
        "eliminate the risk of thermal runaway and enable the use of lithium metal anodes. "
        "These anodes offer theoretical energy densities exceeding 500 watt-hours per "
        "kilogram, roughly double that of current graphite-anode designs. Manufacturing "
        "challenges include maintaining interfacial contact between solid components."
    ),
    (
        "The development of neuromorphic computing architectures represents a fundamental "
        "shift in how we approach computational problems. Unlike von Neumann machines that "
        "separate memory and processing, neuromorphic chips integrate computation directly "
        "with memory using artificial synaptic connections. Intel's Loihi 2 processor "
        "contains over one million neurons with 120 million synaptic connections per chip, "
        "consuming only 1 watt of power during typical inference workloads. These systems "
        "excel at sparse temporal pattern recognition tasks where traditional GPU-based "
        "approaches require orders of magnitude more energy per inference operation."
    ),
]


def make_prompt(chunks: list[str], system: str = "You are a technical analyst.") -> list[dict]:
    context = "\n\n".join(chunks)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Analyze the following technical developments:\n\n{context}\n\nProvide a brief synthesis."},
    ]


def send_request(messages: list[dict], max_tokens: int = 50) -> dict:
    resp = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


def get_recent_logs(n: int = 100) -> str:
    import subprocess
    result = subprocess.run(
        ["kubectl", "logs", "--tail", str(n), "-n", "synapse-staging",
         "-l", "app=synapse-vllm", "-c", "vllm"],
        capture_output=True, text=True, timeout=15,
    )
    return result.stdout


def main():
    print("=== SemBlend REORDER Diagnostic (v6) ===\n")

    # Step 1: Seed — send original order to register as donor
    original_order = CHUNKS[:]
    seed_msgs = make_prompt(original_order)
    print(f"[1/3] SEED: sending {len(CHUNKS)} chunks in original order...")
    t0 = time.time()
    seed_resp = send_request(seed_msgs)
    seed_ttft = time.time() - t0
    print(f"  TTFT: {seed_ttft*1000:.0f}ms")
    print(f"  Response: {seed_resp['choices'][0]['message']['content'][:80]}...")

    # Wait for donor registration
    time.sleep(2)

    # Step 2: Check donor store
    logs_after_seed = get_recent_logs(50)
    donor_registered = "donor registered" in logs_after_seed.lower() or "register_donor" in logs_after_seed.lower()
    store_size_lines = [l for l in logs_after_seed.split('\n') if 'store_size' in l.lower() or 'donor_count' in l.lower()]
    print(f"\n  Donor registered: {donor_registered}")
    if store_size_lines:
        print(f"  Store info: {store_size_lines[-1].strip()}")

    # Step 3: REORDER — shuffle chunks and send
    reordered = CHUNKS[:]
    random.seed(42)
    random.shuffle(reordered)
    reorder_msgs = make_prompt(reordered)

    # Show the reordering
    orig_first_words = [c.split()[0:3] for c in original_order]
    reord_first_words = [c.split()[0:3] for c in reordered]
    print(f"\n[2/3] REORDER: shuffled chunk order")
    print(f"  Original: {[' '.join(w) for w in orig_first_words]}")
    print(f"  Reordered: {[' '.join(w) for w in reord_first_words]}")

    print("  Sending reordered prompt...")
    t0 = time.time()
    reorder_resp = send_request(reorder_msgs)
    reorder_ttft = time.time() - t0
    print(f"  TTFT: {reorder_ttft*1000:.0f}ms")

    time.sleep(1)

    # Step 4: Check logs for SemBlend HIT/MISS
    logs_after_reorder = get_recent_logs(80)
    hit_lines = [l for l in logs_after_reorder.split('\n') if 'SemBlend' in l and ('HIT' in l or 'MISS' in l or 'hit' in l.lower())]
    sim_lines = [l for l in logs_after_reorder.split('\n') if 'sim=' in l or 'similarity' in l.lower()]
    donor_hit_lines = [l for l in logs_after_reorder.split('\n') if 'donor_hit' in l or 'donor hit' in l]
    tok_lines = [l for l in logs_after_reorder.split('\n') if 'tok=' in l]

    print(f"\n[3/3] RESULTS:")
    print(f"  Seed TTFT:    {seed_ttft*1000:.0f}ms")
    print(f"  Reorder TTFT: {reorder_ttft*1000:.0f}ms")
    print(f"  Speedup:      {seed_ttft/reorder_ttft:.2f}x")

    if hit_lines:
        print(f"\n  SemBlend log entries:")
        for l in hit_lines[-5:]:
            print(f"    {l.strip()}")

    if sim_lines:
        print(f"\n  Similarity log entries:")
        for l in sim_lines[-3:]:
            print(f"    {l.strip()}")

    if donor_hit_lines:
        print(f"\n  Donor hit log entries:")
        for l in donor_hit_lines[-3:]:
            print(f"    {l.strip()}")

    if tok_lines:
        print(f"\n  Token log entries:")
        for l in tok_lines[-3:]:
            print(f"    {l.strip()}")

    # Verdict
    got_hit = any('HIT' in l for l in hit_lines)
    got_miss = any('MISS' in l for l in hit_lines) and not got_hit
    print(f"\n  === VERDICT: {'PASS - REORDER got HIT' if got_hit else 'FAIL - REORDER got MISS' if got_miss else 'UNCLEAR - check logs'} ===")

    if not got_hit:
        print("\n  Full recent logs for debugging:")
        for l in logs_after_reorder.split('\n')[-30:]:
            if l.strip():
                print(f"    {l.strip()}")

    return 0 if got_hit else 1


if __name__ == "__main__":
    sys.exit(main())
