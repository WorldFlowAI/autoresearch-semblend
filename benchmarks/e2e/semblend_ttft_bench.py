#!/usr/bin/env python3
"""SemBlend TTFT benchmark — long-context KV cache reuse measurement.

Constructs prompts with ~8000 tokens to make prefill time dominant.
Measures TTFT across 5 scenarios to quantify KV cache reuse benefit.

Usage:
    python3 semblend_ttft_bench.py \
        --endpoint http://synapse-staging-vllm:8000 \
        --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
        --target-tokens 8000
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean, median

import aiohttp

# --- Long RAG context (~4000 words) ---

REPORT_SECTION_A = """EXECUTIVE SUMMARY
The global semiconductor industry experienced significant growth in Q3 2025,
driven primarily by accelerating demand for AI training and inference chips.
Total industry revenue reached $178.2 billion, representing a 23.4% year-over-year
increase. The AI accelerator segment alone contributed $42.8 billion, up 67%
from the same period last year. The memory segment contributed $48.3 billion
(+31% YoY), while analog and mixed-signal revenue reached $21.7 billion.

MARKET DYNAMICS
1. AI CHIP DEMAND: Enterprise adoption of large language models continued to surge,
with hyperscalers increasing their capital expenditure on AI infrastructure by 45%.
NVIDIA maintained its dominant position with an estimated 82% market share in the
data center GPU segment, though AMD gained ground with its MI300X accelerator
achieving 14% share, up from 8% in Q2. The total addressable market for AI
accelerators is projected to reach $120 billion by 2027, driven by both training
and inference workloads. Training still accounts for 60% of compute demand, but
inference is growing faster at 85% CAGR versus 45% for training. The shift toward
mixture-of-experts architectures and speculative decoding has increased the compute
efficiency of inference by roughly 3x, but this efficiency gain has been more than
offset by the explosion in inference volume as more applications deploy LLMs in
production.

2. MEMORY MARKET: HBM3E demand outstripped supply throughout the quarter. SK Hynix
reported a 340% increase in HBM revenue, while Samsung announced plans to triple
its HBM production capacity by Q1 2026. Average selling prices for HBM3E modules
increased 28% quarter-over-quarter. The standard DRAM market also strengthened,
with DDR5 adoption in servers reaching 45% penetration, up from 28% a year ago.
NAND flash prices stabilized after a period of decline, with enterprise SSD demand
growing 22% on the back of AI storage requirements. Micron reported that AI-related
memory revenue now accounts for 35% of its total revenue, up from 18% in Q3 2024.
The company shipped 50% more HBM units than originally planned, and raised its
FY2026 HBM revenue guidance to $12 billion.

3. FOUNDRY SERVICES: TSMC reported record quarterly revenue of $23.5 billion,
with advanced nodes (5nm and below) accounting for 69% of total wafer revenue.
The company announced accelerated construction of its Arizona Fab 2, now expected
to begin volume production of 3nm chips by Q4 2026. TSMC also disclosed that it
has received commitments for over 90% of its 2nm wafer capacity through 2027.
Samsung Foundry continues to struggle with yield issues at its 3nm GAA (Gate All
Around) node, achieving only 55% yield compared to TSMC's 78% at N3E. Intel
Foundry (formerly IFS) made progress on Intel 18A, demonstrating functional
test chips with yields described as competitive with industry though specific
numbers were not disclosed. GlobalFoundries reported strong demand for its
specialty FDX platform, particularly for automotive radar and IoT applications.

4. AUTOMOTIVE SEMICONDUCTORS: The automotive chip shortage that plagued the industry
since 2020 has largely resolved, with lead times returning to pre-pandemic levels
of 12-16 weeks. However, the shift toward electric vehicles and autonomous driving
is creating new demand patterns, with the average silicon content per vehicle
increasing from $712 in 2024 to $834 in 2025. L3 autonomous driving systems from
Mobileye, NVIDIA DRIVE, and Qualcomm Snapdragon Ride are entering mass production,
each requiring 200-500 TOPS of dedicated AI compute. The automotive semiconductor
market is expected to reach $78 billion by 2027, with ADAS and infotainment
accounting for 45% of the total.

5. GEOPOLITICAL FACTORS: US-China technology restrictions continued to reshape
supply chains. China domestic semiconductor industry invested $47 billion in
new fabrication facilities, though most remain at 28nm and above process nodes.
The CHIPS Act funding in the US has now committed $32 billion to 15 projects,
with Intel, TSMC, Samsung, and Micron as primary recipients. Japan semiconductor
revival continued with Rapidus announcing it had reached tape-out stage for its
2nm process in collaboration with IBM. The European Chips Act has allocated 12
billion euros, with key investments going to Intel Magdeburg fab, TSMC Dresden
facility, and STMicroelectronics expansion in Crolles, France."""

REPORT_SECTION_B = """COMPANY HIGHLIGHTS
- NVIDIA: Launched the B200 Ultra accelerator with 2x the inference throughput
  of the H100. Data center revenue reached $18.4 billion (+78% YoY). The company
  also unveiled its Blackwell Ultra architecture roadmap, promising 4x inference
  throughput improvement by Q2 2026. Software ecosystem (CUDA, TensorRT-LLM)
  remains the key moat, with 4.7 million active developers.
- AMD: MI300X gained traction with Microsoft Azure and Oracle Cloud. Total data
  center revenue was $3.9 billion (+112% YoY). The company announced the MI350X
  with HBM3E and PCIe Gen6 support, expected to sample in Q1 2026. AMD ROCm
  software stack achieved functional parity with CUDA for 80% of common LLM
  training workloads, though performance gaps remain at 10-15% for specialized
  kernels.
- Intel: Foundry services secured a major win with a US defense contract worth
  $4.2 billion over 5 years. However, its data center GPU (Gaudi 3) captured
  only 2% market share. Intel 18A showed promising yield improvements and the
  company secured new foundry customers including MediaTek.
- Qualcomm: Snapdragon X Elite PC chips gained momentum, with 35 design wins
  from major OEMs. AI PC revenue reached $1.2 billion in the quarter.
- Broadcom: Custom AI accelerator (XPU) business grew 3x, with Google and Meta
  as key customers. Total AI revenue was $4.1 billion.

SUPPLY CHAIN AND PACKAGING OUTLOOK
Advanced packaging (CoWoS, InFO) remains the primary bottleneck for AI chip
production. TSMC plans to increase CoWoS capacity by 2.5x by mid-2026, but
demand is expected to grow 3x in the same period. This supply-demand gap
suggests continued allocation constraints through 2026. New packaging approaches
including silicon photonics interconnects and chiplet-based architectures using
UCIe (Universal Chiplet Interconnect Express) are gaining industry adoption,
with 15 major companies joining the UCIe consortium this quarter.

INVESTMENT THEMES AND FORWARD OUTLOOK
1. AI infrastructure spending shows no signs of slowing; expect continued
   momentum through 2026 as enterprise adoption broadens beyond hyperscalers.
   Total AI chip spending could reach $200 billion annually by 2027.
2. Memory manufacturers (SK Hynix, Samsung, Micron) are best positioned to
   capture the HBM demand wave with structurally higher margins.
3. Edge AI deployment is accelerating, creating opportunities for Qualcomm,
   MediaTek, and specialized AI chip startups.
4. Geopolitical risk premium remains elevated for companies with significant
   China exposure.
5. Software ecosystem moats (CUDA, ROCm, OpenVINO) are becoming as important
   as hardware performance in determining market share."""

# Rephrased version — semantically identical, lexically different
REPORT_SECTION_A_REPHRASED = """OVERVIEW AND KEY METRICS
During the third quarter of 2025, the worldwide chip industry posted remarkable
expansion fuelled by surging appetite for artificial intelligence compute hardware.
Aggregate sector revenues hit $178.2B, marking a year-on-year jump of 23.4 percent.
AI accelerator chips alone generated $42.8B in sales, which represents a 67 percent
surge compared to the prior-year quarter. Memory products accounted for $48.3B (up
31% annually), and the analog plus mixed-signal category brought in $21.7B.

KEY MARKET THEMES
1. ARTIFICIAL INTELLIGENCE HARDWARE: The enterprise rollout of large language model
technology accelerated further, as major cloud providers boosted their AI capex by
45 percent. NVIDIA continued to lead the data-center GPU space with roughly 82%
market share, while AMD made inroads through the MI300X processor which climbed
to 14% share from 8% last quarter. Analysts forecast the AI accelerator TAM will
expand to $120B by calendar year 2027, with both model training and serving driving
demand. Training currently consumes 60% of compute cycles, yet inference demand is
outpacing it with an 85% compound annual growth rate versus 45% for training. New
architectural paradigms like mixture-of-experts and speculative decoding have
boosted inference efficiency by approximately 3x, but this has been more than
counterbalanced by the rapid proliferation of production LLM deployments.

2. MEMORY SECTOR: High-bandwidth memory (HBM3E) remained chronically undersupplied
all quarter long. SK Hynix disclosed a 340% jump in HBM sales, and Samsung committed
to tripling its HBM output capacity before Q1 2026. HBM3E module ASPs climbed 28%
sequentially. Standard DRAM improved too, with server DDR5 penetration rising to 45%
from 28% twelve months earlier. Enterprise SSD orders grew 22% driven by AI data
storage needs. Micron stated that AI-linked memory now comprises 35% of revenue
(versus 18% a year ago), shipped 50% extra HBM units beyond plan, and lifted its
FY2026 HBM revenue target to $12B.

3. WAFER FABRICATION: TSMC posted $23.5B in quarterly revenue, a new record, with
leading-edge processes (5nm and finer) representing 69% of wafer sales. Construction
on Arizona Fab 2 is ahead of schedule, targeting 3nm volume output by late 2026.
TSMC has pre-sold over 90% of 2nm wafer supply through 2027. Samsung Foundry
continued to face yield difficulties at its 3nm gate-all-around node, reaching
just 55% versus TSMC N3E at 78%. Intel Foundry demonstrated working 18A test
silicon with yields termed competitive though not quantified. GlobalFoundries
saw robust uptake of its specialty FDX process for automotive radar and IoT.

4. AUTOMOTIVE CHIPS: The chip shortages from 2020 onward have substantially
unwound, with delivery lead times back to the normal 12-to-16-week range. Yet
the transition to EVs and self-driving tech is reshaping the demand mix; average
semiconductor content per car climbed from $712 in 2024 to $834 this year. Level 3
autonomy platforms from Mobileye, NVIDIA DRIVE, and Qualcomm Snapdragon Ride are
entering series production, each demanding 200 to 500 TOPS of AI processing. The
auto chip market is forecast at $78B by 2027, with ADAS plus infotainment making
up 45% of that figure.

5. TRADE AND POLICY: US-China tech curbs kept reshaping global supply chains.
Chinese domestic chipmakers poured $47B into new fabs, predominantly at 28nm
and larger geometries. The US CHIPS Act has disbursed $32B across 15 projects
so far, mainly to Intel, TSMC, Samsung, and Micron. Japan Rapidus reached
tape-out on its 2nm node co-developed with IBM. Europe Chips Act funding of
EUR 12B is flowing to Intel Magdeburg, TSMC Dresden, and ST Crolles."""

REPORT_SECTION_B_REPHRASED = """NOTABLE COMPANY UPDATES
- NVIDIA released the B200 Ultra chip delivering double the inference throughput
  versus H100. Datacenter sales reached $18.4B, up 78% year-over-year. The
  Blackwell Ultra roadmap promises a further 4x inference gain by mid-2026.
  The CUDA developer community now exceeds 4.7 million.
- AMD saw MI300X adoption at Azure and Oracle Cloud; datacenter revenue hit
  $3.9B (+112% YoY). MI350X with HBM3E and PCIe Gen6 is sampling Q1 2026.
  ROCm now matches CUDA on 80% of mainstream LLM training tasks.
- Intel won a $4.2B US defense foundry contract but Gaudi 3 managed only 2%
  GPU market share. 18A yields are improving; MediaTek signed as a customer.
- Qualcomm Snapdragon X Elite PCs secured 35 OEM design wins; AI PC revenue
  hit $1.2B.
- Broadcom custom AI XPU business tripled; Google and Meta are primary clients.
  AI revenues totalled $4.1B.

PACKAGING CONSTRAINTS AND TECHNOLOGY TRENDS
CoWoS and InFO advanced packaging capacity is the main production chokepoint
for AI processors. TSMC aims to raise CoWoS output 2.5x by mid-2026 yet demand
may triple, implying persistent allocation pressure through 2026. Silicon-photonic
interconnects and chiplet approaches via UCIe are gaining momentum, with 15 firms
joining the UCIe consortium during Q3.

INVESTMENT PERSPECTIVE
1. AI infrastructure capex is poised to grow through 2026 as enterprise LLM
   adoption extends well beyond the hyperscale cloud operators. Annual AI chip
   expenditure could approach $200B by 2027.
2. HBM suppliers (SK Hynix, Samsung, Micron) offer the clearest earnings
   leverage to the AI hardware cycle.
3. Edge-AI silicon demand is ramping, benefiting Qualcomm, MediaTek, and
   start-ups.
4. Geopolitical risk remains a factor for firms with heavy China exposure.
5. Developer-ecosystem lock-in (CUDA, ROCm, OpenVINO) is increasingly
   decisive for chip market share."""

# Context for a completely different domain (cold baseline)
CLIMATE_REPORT = """GLOBAL CLIMATE AND ENERGY TRANSITION REPORT - H2 2025

TEMPERATURE AND EMISSIONS DATA
Global mean surface temperature in 2025 is tracking 1.48 degrees Celsius above
pre-industrial levels, making it likely the hottest year on record. Atmospheric
CO2 concentration reached 427 ppm in September 2025, a 2.8 ppm annual increase.
Methane levels continued their upward trajectory at 1,935 ppb (+14 ppb YoY).
Total anthropogenic greenhouse gas emissions are estimated at 57.4 GtCO2e for
2025, a 0.8% increase despite accelerating renewable deployment. The carbon
budget for 1.5C is now estimated at 120 GtCO2 remaining, implying roughly
3 years at current emission rates.

RENEWABLE ENERGY DEPLOYMENT
Solar PV installations reached 580 GW globally in 2025, up 35% from 2024.
China accounted for 60% of new solar capacity, followed by the US (12%),
India (8%), and the EU (7%). Wind power additions totaled 145 GW, with
offshore wind contributing 32 GW (+45% YoY). Battery energy storage
installations hit 120 GWh, led by utility-scale projects in China,
the US, and Australia. The levelized cost of solar reached $23/MWh
globally, and onshore wind hit $32/MWh, both new record lows. Lithium
iron phosphate (LFP) battery pack prices fell to $92/kWh, crossing below
the $100 threshold for the first time.

FOSSIL FUEL TRANSITION
Global coal consumption peaked in 2024 and declined 2.1% in 2025, primarily
driven by plant retirements in Europe and reduced growth in China. However,
India and Southeast Asia continued to add coal capacity. Natural gas demand
grew 1.5%, supported by LNG exports to Asia and data center power demand.
Oil demand growth slowed to 0.6 million barrels per day, as EV adoption
displaced approximately 1.8 million barrels per day of gasoline demand
globally. OPEC maintained production cuts, keeping Brent crude in the
$75-85 range. The International Energy Agency projects oil demand will
plateau by 2028 under current policies.

ELECTRIC VEHICLE MARKET
Global EV sales reached 22 million units in 2025, representing 25% of all
new car sales. China led with 12 million units (45% domestic share), followed
by Europe (5.2 million, 35% share) and the US (3.1 million, 15% share). BYD
overtook Volkswagen to become the world second-largest automaker by revenue.
Tesla maintained EV market leadership at 18% global share but faced increasing
competition from Chinese manufacturers. Average EV prices fell 12% year-over-year,
reaching parity with ICE vehicles in China and approaching parity in Europe.
Commercial EV adoption accelerated with electric trucks and buses reaching
8% of new registrations in China.

POLICY AND FINANCE
Carbon markets expanded significantly. The EU ETS carbon price averaged 85
euros per tonne. China launched Phase 2 of its ETS covering steel and cement.
Climate finance flows reached $1.8 trillion in 2025, with private sector
contributing $1.2 trillion. Green bond issuance totaled $650 billion.
Major policy developments included the US finalizing EPA tailpipe emission
standards requiring 56% EV share by 2032, the EU adopting a carbon border
adjustment mechanism, and India launching a $50 billion green hydrogen mission.
Climate litigation increased 30% with notable cases against major oil companies
in the Netherlands and the UK.

TECHNOLOGY BREAKTHROUGHS
Enhanced geothermal systems achieved commercial viability in Nevada, producing
25 MW at competitive costs. Direct air capture reached 10,000 tonnes per year
capacity across 5 commercial plants, though costs remain at $400-600 per tonne.
Perovskite-silicon tandem solar cells demonstrated 33.9% lab efficiency, with
commercial modules expected by 2027. Solid-state batteries from Toyota and
Samsung SDI began pilot production, offering 2x energy density improvement.
Nuclear fusion startup Commonwealth Fusion Systems reported sustained plasma
at MIT sparc tokamak for 8 seconds, a significant milestone toward commercial
fusion power."""


def build_prompt(context: str, question: str, target_tokens: int) -> str:
    """Build a prompt that is approximately target_tokens long."""
    system = "You are a senior analyst. Use ONLY the report below to answer.\n\n"
    suffix = f"\n\nBased on the report above, answer this question:\n{question}"

    # Estimate tokens: ~1.3 tokens per word, ~4 chars per token
    available_chars = target_tokens * 4 - len(system) - len(suffix)

    # Repeat context to fill the target length
    full_context = context
    while len(full_context) < available_chars:
        full_context += "\n\n--- CONTINUED ANALYSIS ---\n\n" + context

    full_context = full_context[:available_chars]
    return system + full_context + suffix


@dataclass
class Result:
    phase: str
    label: str
    ttft_ms: float
    total_ms: float
    tokens_gen: int
    prompt_tokens: int
    error: str | None = None


async def measure_ttft(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 20,
) -> tuple[float, float, int, int]:
    """Send a chat completion with streaming and measure TTFT.

    Returns (ttft_ms, total_ms, tokens_generated, prompt_tokens).
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.1,
    }

    t_start = time.perf_counter()
    t_first_token = None
    token_count = 0
    prompt_tokens = 0

    async with session.post(
        f"{endpoint}/v1/chat/completions", json=body
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")

        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            payload = decoded[6:]
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
                delta = chunk["choices"][0].get("delta", {})
                if delta.get("content"):
                    if t_first_token is None:
                        t_first_token = time.perf_counter()
                    token_count += 1
                usage = chunk.get("usage")
                if usage and "prompt_tokens" in usage:
                    prompt_tokens = usage["prompt_tokens"]
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    t_end = time.perf_counter()

    if t_first_token is None:
        raise RuntimeError("No tokens received")

    ttft = (t_first_token - t_start) * 1000
    total = (t_end - t_start) * 1000
    return ttft, total, token_count, prompt_tokens


async def run_phase(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompts: list[tuple[str, str]],
    phase_name: str,
    repeats: int = 1,
    max_tokens: int = 20,
) -> list[Result]:
    results = []
    for rep in range(repeats):
        for label, prompt in prompts:
            try:
                ttft, total, toks, ptoks = await measure_ttft(
                    session, endpoint, model, prompt, max_tokens
                )
                results.append(Result(
                    phase=phase_name, label=f"{label}-r{rep}",
                    ttft_ms=ttft, total_ms=total,
                    tokens_gen=toks, prompt_tokens=ptoks,
                ))
                print(f"    {label}-r{rep}: TTFT={ttft:.0f}ms, prompt={ptoks}tok")
            except Exception as e:
                results.append(Result(
                    phase=phase_name, label=f"{label}-r{rep}",
                    ttft_ms=-1, total_ms=-1, tokens_gen=0,
                    prompt_tokens=0, error=str(e)[:200],
                ))
                print(f"    {label}-r{rep}: ERROR {e}")
        await asyncio.sleep(0.5)
    return results


def stats(results: list[Result]) -> dict:
    valid = [r.ttft_ms for r in results if r.ttft_ms > 0]
    if not valid:
        return {"n": 0, "p50": None, "mean": None}
    valid.sort()
    return {
        "n": len(valid),
        "p50": valid[len(valid) // 2],
        "mean": mean(valid),
        "min": valid[0],
        "max": valid[-1],
        "prompt_tokens": results[0].prompt_tokens if results else 0,
    }


async def main(args: argparse.Namespace) -> None:
    endpoint = args.endpoint
    model = args.model
    target_tok = args.target_tokens
    repeats = args.repeats

    questions = [
        "What is NVIDIA's data center revenue and how does it compare year-over-year?",
        "What are the key trends in the HBM memory market?",
        "How does TSMC's foundry revenue break down by node?",
    ]

    follow_questions = [
        "What is AMD's market share trajectory in AI accelerators?",
        "What is the outlook for automotive semiconductor demand?",
        "What are the main geopolitical risks for the chip industry?",
    ]

    climate_questions = [
        "What is the current state of global CO2 emissions?",
        "How fast is EV adoption growing globally?",
        "What policy developments are driving the energy transition?",
    ]

    # Build prompts at target token length
    original_context = REPORT_SECTION_A + "\n\n" + REPORT_SECTION_B
    rephrased_context = REPORT_SECTION_A_REPHRASED + "\n\n" + REPORT_SECTION_B_REPHRASED

    seed_prompts = [
        (f"seed-q{i}", build_prompt(original_context, q, target_tok))
        for i, q in enumerate(questions)
    ]
    exact_prompts = seed_prompts.copy()

    prefix_vary = [
        (f"pfx-q{i}", build_prompt(original_context, q, target_tok))
        for i, q in enumerate(follow_questions)
    ]

    semantic_prompts = [
        (f"sem-q{i}", build_prompt(rephrased_context, q, target_tok))
        for i, q in enumerate(questions)
    ]

    cold_prompts = [
        (f"cold-q{i}", build_prompt(CLIMATE_REPORT, q, target_tok))
        for i, q in enumerate(climate_questions)
    ]

    print(f"SemBlend TTFT Benchmark")
    print(f"  Model: {model}")
    print(f"  Target tokens: {target_tok}")
    print(f"  Repeats: {repeats}")
    print(f"  Prompt length (chars): ~{len(seed_prompts[0][1])}")

    timeout = aiohttp.ClientTimeout(total=180)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Health check
        try:
            async with session.get(f"{endpoint}/health") as resp:
                if resp.status != 200:
                    print(f"vLLM unhealthy: {resp.status}")
                    return
            print(f"  vLLM healthy\n")
        except Exception as e:
            print(f"Cannot reach vLLM: {e}")
            return

        all_phases = {}

        # Phase 1: SEED (cold prefill)
        print(f"--- SEED ({len(seed_prompts)} prompts, 1 pass) ---")
        seed_r = await run_phase(
            session, endpoint, model, seed_prompts, "seed", 1
        )
        all_phases["seed"] = seed_r
        s = stats(seed_r)
        print(f"  SEED P50: {s['p50']:.0f}ms ({s.get('prompt_tokens',0)} tokens)\n")

        await asyncio.sleep(3)  # Let LMCache propagate

        # Phase 2: EXACT (prefix cache hit ceiling)
        print(f"--- EXACT ({len(exact_prompts)} x {repeats}) ---")
        exact_r = await run_phase(
            session, endpoint, model, exact_prompts, "exact", repeats
        )
        all_phases["exact"] = exact_r
        s = stats(exact_r)
        print(f"  EXACT P50: {s['p50']:.0f}ms\n")

        # Phase 3: PREFIX_VARY (same context, different question)
        print(f"--- PREFIX_VARY ({len(prefix_vary)} x {repeats}) ---")
        pv_r = await run_phase(
            session, endpoint, model, prefix_vary, "prefix_vary", repeats
        )
        all_phases["prefix_vary"] = pv_r
        s = stats(pv_r)
        print(f"  PREFIX_VARY P50: {s['p50']:.0f}ms\n")

        # Phase 4: SEMANTIC (rephrased context, same questions)
        print(f"--- SEMANTIC ({len(semantic_prompts)} x {repeats}) ---")
        sem_r = await run_phase(
            session, endpoint, model, semantic_prompts, "semantic", repeats
        )
        all_phases["semantic"] = sem_r
        s = stats(sem_r)
        print(f"  SEMANTIC P50: {s['p50']:.0f}ms\n")

        # Phase 5: COLD (different domain entirely)
        print(f"--- COLD ({len(cold_prompts)} x {repeats}) ---")
        cold_r = await run_phase(
            session, endpoint, model, cold_prompts, "cold", repeats
        )
        all_phases["cold"] = cold_r
        s = stats(cold_r)
        print(f"  COLD P50: {s['p50']:.0f}ms\n")

    # Summary
    def p50(phase: str) -> float:
        return stats(all_phases.get(phase, [])).get("p50", 0) or 0

    seed_p50 = p50("seed")
    exact_p50 = p50("exact")
    pv_p50 = p50("prefix_vary")
    sem_p50 = p50("semantic")
    cold_p50 = p50("cold")

    summary = {
        "model": model,
        "target_tokens": target_tok,
        "repeats": repeats,
        "seed_p50_ms": seed_p50,
        "exact_p50_ms": exact_p50,
        "prefix_vary_p50_ms": pv_p50,
        "semantic_p50_ms": sem_p50,
        "cold_p50_ms": cold_p50,
        "exact_speedup": seed_p50 / exact_p50 if exact_p50 else None,
        "prefix_vary_speedup": seed_p50 / pv_p50 if pv_p50 else None,
        "semantic_speedup": seed_p50 / sem_p50 if sem_p50 else None,
        "prefill_saveable_ms": cold_p50 - exact_p50 if cold_p50 and exact_p50 else None,
        "prompt_tokens": stats(all_phases.get("seed", [])).get("prompt_tokens", 0),
    }

    print("=" * 70)
    print("SemBlend TTFT Benchmark Results")
    print("=" * 70)
    ptok = summary["prompt_tokens"]
    print(f"  Model:  {model}")
    print(f"  Prompt: ~{ptok} tokens")
    print("-" * 70)
    print(f"  SEED   (cold)        {seed_p50:8.0f}ms")
    print(f"  EXACT  (cache hit)   {exact_p50:8.0f}ms  ({seed_p50/exact_p50:.1f}x)" if exact_p50 else "  EXACT  error")
    print(f"  PREFIX_VARY          {pv_p50:8.0f}ms  ({seed_p50/pv_p50:.1f}x)" if pv_p50 else "  PREFIX_VARY error")
    print(f"  SEMANTIC             {sem_p50:8.0f}ms  ({seed_p50/sem_p50:.1f}x)" if sem_p50 else "  SEMANTIC error")
    print(f"  COLD   (no cache)    {cold_p50:8.0f}ms  ({seed_p50/cold_p50:.1f}x)" if cold_p50 else "  COLD   error")
    print("-" * 70)
    if cold_p50 and exact_p50:
        saveable = cold_p50 - exact_p50
        print(f"  Saveable prefill time: {saveable:.0f}ms")
        print(f"  SemBlend target: bring SEMANTIC from {sem_p50:.0f}ms toward {exact_p50:.0f}ms")
    print("=" * 70)

    # Save results
    run_id = f"semblend-ttft-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    output = {
        "run_id": run_id,
        "summary": summary,
        "phases": {
            phase: [asdict(r) for r in results]
            for phase, results in all_phases.items()
        },
    }
    out_file = f"{args.output}/{run_id}.json"
    import os
    os.makedirs(args.output, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--target-tokens", type=int, default=8000)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", default="/tmp/results")
    asyncio.run(main(parser.parse_args()))
