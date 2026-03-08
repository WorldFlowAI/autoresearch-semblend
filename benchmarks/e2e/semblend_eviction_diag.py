"""Diagnostic: SemBlend value when vLLM prefix cache is evicted.

Tests the scenario where:
  1. SEED prompt fills vLLM's KV cache → LMCache saves to CPU
  2. Many different prompts evict the SEED from vLLM's prefix cache
  3. PARTIAL query (same context as SEED, different question) arrives
  4. vLLM prefix cache MISSES → SemBlend finds donor in LMCache CPU tier

This is SemBlend's primary value: extending KV reuse beyond vLLM's GPU
cache capacity through the CPU offload tier.
"""
import sys
import time
import re
import requests
import subprocess

VLLM_URL = "http://localhost:8000"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"

CONTEXT = (
    "The semiconductor industry has entered a transformative era driven by advanced "
    "extreme ultraviolet lithography systems. These EUV machines, operating at 13.5nm "
    "wavelength, enable the fabrication of transistors at 3nm and below process nodes. "
    "The optical systems require multilayer mirrors with over 100 alternating layers of "
    "molybdenum and silicon, each deposited with sub-angstrom precision. The light source "
    "uses a tin droplet plasma that must be maintained at temperatures exceeding 500,000 "
    "degrees Celsius to produce adequate photon flux for wafer exposure. Current production "
    "EUV systems cost approximately 380 million dollars each. "
    "Quantum error correction represents one of the most significant challenges in "
    "building practical quantum computers. The surface code architecture requires "
    "thousands of physical qubits to encode a single logical qubit with sufficiently "
    "low error rates. Recent breakthroughs in superconducting transmon qubits have "
    "demonstrated two-qubit gate fidelities exceeding 99.5 percent, approaching the "
    "threshold needed for fault-tolerant computation. "
    "Advanced battery technology for electric vehicles continues to evolve rapidly "
    "with solid-state electrolytes emerging as the leading candidate for next-generation "
    "energy storage. Unlike conventional lithium-ion batteries that use liquid organic "
    "electrolytes, solid-state designs employ ceramic or polymer electrolytes that "
    "eliminate the risk of thermal runaway and enable the use of lithium metal anodes. "
    "These anodes offer theoretical energy densities exceeding 500 watt-hours per kilogram."
)

# Many diverse prompts to flush vLLM's prefix cache
FILLER_PROMPTS = [
    f"Explain the history and development of {topic} in detail, covering key milestones, major contributors, current state, and future outlook."
    for topic in [
        "quantum computing",
        "artificial intelligence",
        "blockchain technology",
        "gene therapy",
        "autonomous vehicles",
        "space colonization",
        "nuclear fusion",
        "nanotechnology",
        "augmented reality",
        "brain-computer interfaces",
        "sustainable agriculture",
        "ocean energy harvesting",
        "supersonic travel",
        "quantum internet",
        "synthetic biology",
        "dark energy research",
        "neuromorphic engineering",
        "carbon nanotube manufacturing",
        "holographic displays",
        "gravitational wave astronomy",
    ]
]


def send_request(messages: list[dict], max_tokens: int = 30) -> tuple[float, dict]:
    t0 = time.time()
    resp = requests.post(
        f"{VLLM_URL}/v1/chat/completions",
        json={"model": MODEL, "messages": messages, "max_tokens": max_tokens, "temperature": 0.0},
        timeout=120,
    )
    ttft = time.time() - t0
    resp.raise_for_status()
    return ttft, resp.json()


def get_recent_logs(n: int = 50) -> str:
    result = subprocess.run(
        ["kubectl", "logs", "--tail", str(n), "-n", "synapse-staging",
         "deployment/synapse-staging-vllm"],
        capture_output=True, text=True, timeout=15,
    )
    return result.stdout


def main():
    print("=== SemBlend Eviction Scenario Diagnostic ===\n")

    # Step 1: SEED — register donor with shared context
    seed_msgs = [
        {"role": "system", "content": "You are a technical analyst."},
        {"role": "user", "content": f"Context:\n{CONTEXT}\n\nWhat are the key challenges in EUV lithography?"},
    ]
    print("[1/4] SEED: registering donor...")
    seed_ttft, _ = send_request(seed_msgs)
    print(f"  TTFT: {seed_ttft*1000:.0f}ms")
    time.sleep(2)

    logs = get_recent_logs(20)
    for l in logs.split('\n'):
        if 'registered donor' in l.lower():
            print(f"  {l.strip()[-100:]}")
            break

    # Step 2: Flood with diverse prompts to evict from vLLM prefix cache
    print(f"\n[2/4] FILLER: sending {len(FILLER_PROMPTS)} diverse prompts to evict prefix cache...")
    for i, prompt in enumerate(FILLER_PROMPTS):
        msgs = [{"role": "system", "content": "You are an expert."}, {"role": "user", "content": prompt}]
        ttft, _ = send_request(msgs, max_tokens=20)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(FILLER_PROMPTS)} done ({ttft*1000:.0f}ms)")
        time.sleep(0.5)

    # Check prefix cache hit rate
    logs = get_recent_logs(5)
    for l in logs.split('\n'):
        if 'Prefix cache hit rate' in l:
            print(f"  {l.strip()[-120:]}")

    # Step 3: PARTIAL — same context as SEED, different question
    print(f"\n[3/4] PARTIAL: same context, different question...")
    partial_msgs = [
        {"role": "system", "content": "You are a technical analyst."},
        {"role": "user", "content": f"Context:\n{CONTEXT}\n\nCompare solid-state batteries to conventional lithium-ion batteries."},
    ]
    partial_ttft, _ = send_request(partial_msgs)
    print(f"  TTFT: {partial_ttft*1000:.0f}ms")

    time.sleep(1)

    # Check SemBlend result
    logs = get_recent_logs(30)
    for l in logs.split('\n'):
        if 'SemBlend' in l and ('HIT' in l or 'MISS' in l):
            print(f"  {l.strip()[-120:]}")
        if 'donor hit=' in l:
            print(f"  {l.strip()[-120:]}")
        if 'num_computed' in l and 'PARTIAL' not in l:
            m = re.search(r'num_computed=(\d+), prompt_len=(\d+)', l)
            if m:
                nc, pl = int(m.group(1)), int(m.group(2))
                print(f"  Prefix cache: {nc}/{pl} tokens ({100*nc/pl:.0f}%)")

    # Step 4: Cold baseline
    print(f"\n[4/4] COLD baseline...")
    cold_msgs = [
        {"role": "system", "content": "You are a historian."},
        {"role": "user", "content": "Describe the Roman Empire's decline and fall."},
    ]
    cold_ttft, _ = send_request(cold_msgs)
    print(f"  TTFT: {cold_ttft*1000:.0f}ms")

    print(f"\n=== RESULTS ===")
    print(f"  Seed TTFT:    {seed_ttft*1000:.0f}ms")
    print(f"  Partial TTFT: {partial_ttft*1000:.0f}ms")
    print(f"  Cold TTFT:    {cold_ttft*1000:.0f}ms")
    speedup = seed_ttft / partial_ttft if partial_ttft > 0 else 0
    print(f"  Speedup (partial vs seed): {speedup:.2f}x")


if __name__ == "__main__":
    main()
