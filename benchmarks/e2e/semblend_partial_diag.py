"""Diagnostic: verify SemBlend PARTIAL scenario donor matching.

Tests the realistic RAG use case: same context chunks with a different user query.
The prefix (system + context) is identical, only the final question differs.
This should be a PARTIAL match with high reuse.
"""
import json
import sys
import time
import requests

VLLM_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

SHARED_CONTEXT = (
    "The semiconductor industry has entered a transformative era driven by advanced "
    "extreme ultraviolet lithography systems. These EUV machines, operating at 13.5nm "
    "wavelength, enable the fabrication of transistors at 3nm and below process nodes. "
    "The optical systems require multilayer mirrors with over 100 alternating layers of "
    "molybdenum and silicon, each deposited with sub-angstrom precision. The light source "
    "uses a tin droplet plasma that must be maintained at temperatures exceeding 500,000 "
    "degrees Celsius to produce adequate photon flux for wafer exposure. "
    "Quantum error correction represents one of the most significant challenges in "
    "building practical quantum computers. The surface code architecture requires "
    "thousands of physical qubits to encode a single logical qubit with sufficiently "
    "low error rates. Recent breakthroughs in superconducting transmon qubits have "
    "demonstrated two-qubit gate fidelities exceeding 99.5 percent, approaching the "
    "threshold needed for fault-tolerant computation. The cryogenic infrastructure "
    "required to maintain these systems at millikelvin temperatures adds substantial "
    "engineering complexity to any commercial quantum computing deployment. "
    "Advanced battery technology for electric vehicles continues to evolve rapidly "
    "with solid-state electrolytes emerging as the leading candidate for next-generation "
    "energy storage. Unlike conventional lithium-ion batteries that use liquid organic "
    "electrolytes, solid-state designs employ ceramic or polymer electrolytes that "
    "eliminate the risk of thermal runaway and enable the use of lithium metal anodes. "
    "These anodes offer theoretical energy densities exceeding 500 watt-hours per "
    "kilogram, roughly double that of current graphite-anode designs. Manufacturing "
    "challenges include maintaining interfacial contact between solid components. "
    "The development of neuromorphic computing architectures represents a fundamental "
    "shift in how we approach computational problems. Unlike von Neumann machines that "
    "separate memory and processing, neuromorphic chips integrate computation directly "
    "with memory using artificial synaptic connections. Intel's Loihi 2 processor "
    "contains over one million neurons with 120 million synaptic connections per chip, "
    "consuming only 1 watt of power during typical inference workloads. These systems "
    "excel at sparse temporal pattern recognition tasks where traditional GPU-based "
    "approaches require orders of magnitude more energy per inference operation."
)


def make_prompt(query: str) -> list[dict]:
    return [
        {"role": "system", "content": "You are a technical analyst. Answer concisely based on the provided context."},
        {"role": "user", "content": f"Context:\n{SHARED_CONTEXT}\n\nQuestion: {query}"},
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
         "deployment/synapse-staging-vllm"],
        capture_output=True, text=True, timeout=15,
    )
    return result.stdout


def main():
    print("=== SemBlend PARTIAL Diagnostic ===\n")

    # Query 1: Seed with first question
    q1 = "What are the main challenges in EUV lithography manufacturing?"
    msgs1 = make_prompt(q1)
    print(f"[1/4] SEED: '{q1}'")
    t0 = time.time()
    resp1 = send_request(msgs1)
    seed_ttft = time.time() - t0
    print(f"  TTFT: {seed_ttft*1000:.0f}ms")
    print(f"  Response: {resp1['choices'][0]['message']['content'][:80]}...")

    time.sleep(2)

    # Check registration
    logs = get_recent_logs(30)
    reg_lines = [l for l in logs.split('\n') if 'registered donor' in l.lower() or 'donor reg' in l.lower()]
    print(f"  Registered: {bool(reg_lines)}")
    if reg_lines:
        print(f"    {reg_lines[-1].strip()[-120:]}")

    # Query 2: PARTIAL — same context, different question
    q2 = "How do solid-state batteries compare to lithium-ion in terms of energy density?"
    msgs2 = make_prompt(q2)
    print(f"\n[2/4] PARTIAL: '{q2}'")
    t0 = time.time()
    resp2 = send_request(msgs2)
    partial_ttft = time.time() - t0
    print(f"  TTFT: {partial_ttft*1000:.0f}ms")

    time.sleep(1)

    # Check SemBlend results
    logs = get_recent_logs(50)
    hit_lines = [l for l in logs.split('\n') if 'SemBlend' in l and ('HIT' in l or 'MISS' in l)]
    sim_lines = [l for l in logs.split('\n') if 'max_sim=' in l or 'DonorStore:' in l]
    chunk_lines = [l for l in logs.split('\n') if 'chunk_alignment' in l or 'chunk' in l.lower() and 'matched' in l.lower()]
    donor_hit_lines = [l for l in logs.split('\n') if 'donor hit=' in l]

    print(f"\n[3/4] LOG ANALYSIS:")
    if sim_lines:
        for l in sim_lines[-3:]:
            print(f"  Similarity: {l.strip()[-120:]}")
    if chunk_lines:
        for l in chunk_lines[-3:]:
            print(f"  Chunk: {l.strip()[-120:]}")
    if hit_lines:
        for l in hit_lines[-5:]:
            print(f"  SemBlend: {l.strip()[-120:]}")
    if donor_hit_lines:
        for l in donor_hit_lines[-3:]:
            print(f"  Injection: {l.strip()[-120:]}")

    # Query 3: Cold baseline (completely different prompt for comparison)
    q3 = "Describe the history of space exploration from the 1960s to present day, including key missions and technological milestones."
    cold_msgs = [
        {"role": "system", "content": "You are a historian."},
        {"role": "user", "content": q3},
    ]
    print(f"\n[4/4] COLD BASELINE:")
    t0 = time.time()
    resp3 = send_request(cold_msgs)
    cold_ttft = time.time() - t0
    print(f"  TTFT: {cold_ttft*1000:.0f}ms")

    print(f"\n=== RESULTS ===")
    print(f"  Seed TTFT:    {seed_ttft*1000:.0f}ms")
    print(f"  Partial TTFT: {partial_ttft*1000:.0f}ms")
    print(f"  Cold TTFT:    {cold_ttft*1000:.0f}ms")

    got_hit = any('HIT' in l for l in hit_lines)
    if got_hit:
        speedup_vs_seed = seed_ttft / partial_ttft if partial_ttft > 0 else 0
        print(f"  Speedup vs seed: {speedup_vs_seed:.2f}x")
        print(f"\n  === VERDICT: PASS - PARTIAL got SemBlend HIT ===")
    else:
        print(f"\n  === VERDICT: FAIL - PARTIAL got MISS ===")
        # Dump recent logs
        print("\n  Recent logs:")
        for l in get_recent_logs(30).split('\n')[-20:]:
            if l.strip():
                print(f"    {l.strip()[-150:]}")

    return 0 if got_hit else 1


if __name__ == "__main__":
    sys.exit(main())
