#!/usr/bin/env python3
"""SemBlend KV cache reuse benchmark — empirical TTFT measurement.

Measures the real speedup from KV cache reuse at different levels:

Phase 1: SEED — Send long prompts to populate KV cache (cold prefill)
Phase 2: EXACT — Resend identical prompts (prefix caching ceiling)
Phase 3: PREFIX_VARY — Same long prefix, different suffix (prefix cache partial hit)
Phase 4: SEMANTIC — Paraphrased prompts (same intent, different tokens)
Phase 5: COLD — Completely unrelated prompts (baseline, no cache benefit)

Each prompt is ~1000-2000 tokens to make prefill time dominant over decode.
Uses streaming SSE to measure TTFT precisely.

Results show:
- Exact match speedup (best case — what prefix caching gives)
- Prefix-vary speedup (partial prefix match)
- Semantic speedup (what SemBlend aims to provide)
- Cold baseline (no cache, full prefill)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import aiohttp

# --- Prompt construction ---
# We build prompts with a long shared context (~1000 tokens) to make
# prefill time measurable. The context simulates a RAG retrieval scenario.

RAG_CONTEXT_A = """You are a senior financial analyst assistant. Below is a detailed research report
that you must use to answer the user's question accurately and completely.

=== RESEARCH REPORT: Q3 2025 SEMICONDUCTOR INDUSTRY ANALYSIS ===

EXECUTIVE SUMMARY
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
test chips with yields described as "competitive with industry" though specific
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
supply chains. China's domestic semiconductor industry invested $47 billion in
new fabrication facilities, though most remain at 28nm and above process nodes.
The CHIPS Act funding in the US has now committed $32 billion to 15 projects,
with Intel, TSMC, Samsung, and Micron as primary recipients. Japan's semiconductor
revival continued with Rapidus announcing it had reached tape-out stage for its
2nm process in collaboration with IBM. The European Chips Act has allocated 12
billion euros, with key investments going to Intel's Magdeburg fab, TSMC's Dresden
facility, and STMicroelectronics' expansion in Crolles, France. Export controls
on advanced equipment continued to tighten, with ASML reporting a 40% decline
in China bookings for its most advanced EUV lithography systems.

COMPANY HIGHLIGHTS
- NVIDIA: Launched the B200 Ultra accelerator with 2x the inference throughput
  of the H100. Data center revenue reached $18.4 billion (+78% YoY). The company
  also unveiled its Blackwell Ultra architecture roadmap, promising 4x inference
  throughput improvement by Q2 2026. Software ecosystem (CUDA, TensorRT-LLM)
  remains the key moat, with 4.7 million active developers. Networking (InfiniBand
  and Spectrum-X) revenue reached $3.2 billion, driven by AI cluster interconnect
  demand. Grace-Blackwell superchips are shipping to all major cloud providers.
- AMD: MI300X gained traction with Microsoft Azure and Oracle Cloud. Total data
  center revenue was $3.9 billion (+112% YoY). The company announced the MI350X
  with HBM3E and PCIe Gen6 support, expected to sample in Q1 2026. AMD's ROCm
  software stack achieved functional parity with CUDA for 80% of common LLM
  training workloads, though performance gaps remain at 10-15% for specialized
  kernels. Xilinx (now AMD Adaptive Computing) contributed $1.4 billion in revenue,
  with FPGA-based AI inference accelerators finding niche adoption.
- Intel: Foundry services secured a major win with a US defense contract worth
  $4.2 billion over 5 years. However, its data center GPU (Gaudi 3) captured
  only 2% market share. The company announced restructuring that will reduce
  headcount by 15,000 employees by end of 2025. Intel 18A showed promising yield
  improvements, and the company secured new foundry customers including MediaTek
  and a major US defense contractor. Altera (standalone FPGA unit) is performing
  ahead of plan with $1.8 billion quarterly revenue.
- Qualcomm: Snapdragon X Elite PC chips gained momentum, with 35 design wins
  from major OEMs. AI PC revenue reached $1.2 billion in the quarter. The company
  also announced Snapdragon 8 Gen 4, featuring a custom Oryon CPU and an NPU
  capable of 75 TOPS for on-device AI. Automotive design-win pipeline reached
  $45 billion lifetime value.
- Broadcom: Custom AI accelerator (XPU) business grew 3x, with Google and Meta
  as key customers. Total AI revenue was $4.1 billion. The company also benefits
  from its VMware acquisition, with infrastructure software revenue reaching $5.8
  billion (+43% YoY). Networking silicon for AI clusters (Memory, Jericho3-AI,
  Ramon) is growing rapidly as hyperscalers build larger clusters.

SUPPLY CHAIN OUTLOOK
Advanced packaging (CoWoS, InFO) remains the primary bottleneck for AI chip
production. TSMC plans to increase CoWoS capacity by 2.5x by mid-2026, but
demand is expected to grow 3x in the same period. This supply-demand gap
suggests continued allocation constraints through 2026. New packaging approaches
including silicon photonics interconnects and chiplet-based architectures using
UCIe (Universal Chiplet Interconnect Express) are gaining industry adoption,
with 15 major companies joining the UCIe consortium this quarter.

INVESTMENT THEMES
1. AI infrastructure spending shows no signs of slowing; expect continued
   momentum through 2026 as enterprise adoption broadens beyond hyperscalers.
   Total AI chip spending could reach $200 billion annually by 2027.
2. Memory manufacturers (SK Hynix, Samsung, Micron) are best positioned to
   capture the HBM demand wave with structurally higher margins. HBM revenue
   alone could reach $40 billion by 2026.
3. Edge AI deployment is accelerating, creating opportunities for Qualcomm,
   MediaTek, and specialized AI chip startups. The edge AI chip market is
   projected to reach $25 billion by 2027.
4. Geopolitical risk premium remains elevated for companies with significant
   China exposure. Diversification of supply chains continues to benefit
   foundries in Japan, US, and Europe.
5. Software ecosystem moats (CUDA, ROCm, OpenVINO) are becoming as important
   as hardware performance in determining market share.

=== END OF REPORT ===

Using ONLY the information in the report above, answer the user's question.
Be specific and cite numbers from the report."""

RAG_CONTEXT_A_VARIANT = """You are a senior financial analyst assistant. Below is a detailed research report
that you must use to answer the user's question accurately and completely.

=== RESEARCH REPORT: Q3 2025 SEMICONDUCTOR SECTOR REVIEW ===

OVERVIEW AND KEY FIGURES
Global chip industry revenues totaled $178.2 billion during Q3 2025, marking
a robust 23.4% expansion year-over-year. The artificial intelligence accelerator
segment was the primary growth engine, generating $42.8 billion in sales, a
remarkable 67% increase from the year-ago period. Memory products added $48.3
billion (+31% YoY) and analog/mixed-signal chips contributed $21.7 billion.

DETAILED MARKET ANALYSIS
1. ARTIFICIAL INTELLIGENCE PROCESSORS: The adoption of foundation models across
enterprises accelerated substantially, with major cloud providers boosting their
AI infrastructure capital expenditure by an impressive 45% year-over-year. In the
data center GPU arena, NVIDIA continued its market leadership with approximately
82% share, while AMD's MI300X accelerator made notable gains, climbing to 14%
market share from just 8% in the previous quarter. Industry analysts project the
AI accelerator addressable market will expand to $120 billion by calendar year
2027, propelled by both model training and real-time inference demands. While
training currently represents 60% of total AI compute consumption, inference
workloads are expanding at a blistering 85% compound annual rate, significantly
outpacing training's 45% CAGR. The proliferation of mixture-of-experts model
designs and speculative decoding algorithms has tripled inference computational
efficiency, yet the sheer volume of inference requests in production deployments
has more than compensated for these efficiency improvements.

2. SEMICONDUCTOR MEMORY: High-bandwidth memory (HBM3E) demand consistently
exceeded available supply throughout the entire quarter. SK Hynix disclosed a
stunning 340% year-over-year surge in HBM-related revenue, and Samsung unveiled
plans to expand its HBM manufacturing capacity threefold by the first quarter of
2026. HBM3E module average selling prices climbed 28% sequentially. In conventional
memory, DDR5 server adoption reached 45% penetration (versus 28% one year prior).
NAND flash pricing found a floor following earlier declines, buoyed by 22% growth
in enterprise solid-state drive shipments tied to AI data storage needs. Micron
Technology reported that AI-linked memory now represents 35% of total company
revenue, up sharply from 18% in Q3 2024, and the firm exceeded its original HBM
shipment target by 50%. Management subsequently increased fiscal year 2026 HBM
revenue projections to $12 billion.

3. WAFER FABRICATION: TSMC achieved record-breaking quarterly revenue of $23.5
billion, with cutting-edge process nodes (5 nanometers and smaller) comprising
69% of total wafer sales. The foundry giant accelerated the timeline for its
second Arizona fabrication plant, now slated to commence high-volume 3nm chip
production by Q4 2026. Additionally, TSMC revealed that over 90% of its upcoming
2nm production capacity has already been spoken for through 2027. Samsung's foundry
division continues to face challenges with its 3nm gate-all-around technology,
managing only 55% yield rates versus TSMC's 78% on its N3E process. Intel
Foundry demonstrated working silicon on the Intel 18A node with yields management
described as industry-competitive, though no specific metrics were shared.
GlobalFoundries reported healthy demand for its specialty FDX technology platform,
especially in automotive radar and Internet of Things applications.

4. AUTOMOTIVE CHIPS: The semiconductor supply crunch affecting automakers since
2020 has substantially eased, with component lead times reverting to the historical
norm of 12 to 16 weeks. Nonetheless, the transition to battery electric vehicles
and advanced driver assistance systems is reshaping demand, pushing average per-
vehicle semiconductor content from $712 in calendar 2024 to $834 in 2025. Level 3
self-driving platforms from Mobileye, NVIDIA DRIVE, and Qualcomm Snapdragon Ride
are entering volume manufacturing, each needing 200 to 500 trillion operations
per second of specialized AI processing power. The automotive chip market is
forecast to hit $78 billion by 2027, with ADAS and in-vehicle entertainment
systems accounting for roughly 45% of the total value.

5. TRADE AND POLICY: US-China technology export restrictions have continued to
reconfigure global supply chains. Chinese domestic chipmakers invested $47 billion
in new fabrication capacity, although the vast majority targets mature 28nm and
larger geometries. America's CHIPS and Science Act has now directed $32 billion to
15 separate manufacturing and R&D projects, primarily benefiting Intel, TSMC,
Samsung, and Micron. Japan's semiconductor renaissance progresses with Rapidus
reaching tape-out on its 2nm process node in partnership with IBM. The European
Chips Act has earmarked 12 billion euros for strategic projects including Intel's
Magdeburg facility, TSMC's Dresden plant, and STMicroelectronics' Crolles campus
expansion. Controls on advanced semiconductor equipment exports tightened further,
with ASML reporting a 40% drop in Chinese orders for cutting-edge EUV systems.

INDIVIDUAL COMPANY PERFORMANCE
- NVIDIA: Debuted the B200 Ultra GPU, delivering twice the inference throughput of
  its predecessor H100. Data center segment revenue was $18.4 billion, up 78% from
  a year earlier. The firm outlined its Blackwell Ultra architectural roadmap,
  targeting a fourfold inference speed improvement by mid-2026. NVIDIA's developer
  ecosystem (CUDA, TensorRT-LLM) now counts 4.7 million active participants,
  representing a formidable competitive advantage. High-speed networking products
  (InfiniBand, Spectrum-X) generated $3.2 billion, driven by AI data center fabric
  requirements. Grace-Blackwell integrated modules are now deployed across all
  tier-one cloud infrastructure providers.
- AMD: The MI300X accelerator secured meaningful deployments with Microsoft Azure
  and Oracle Cloud Infrastructure. Data center division revenue reached $3.9
  billion, more than doubling year-over-year (+112%). AMD announced the upcoming
  MI350X, incorporating HBM3E memory and PCIe Gen6 connectivity, with engineering
  samples expected in Q1 2026. The company's ROCm open-source software platform
  has achieved feature parity with CUDA for approximately 80% of mainstream LLM
  training scenarios, though a 10-15% performance deficit persists for specialized
  compute kernels. The Xilinx-derived adaptive computing unit delivered $1.4 billion
  in quarterly sales, with FPGA-based AI inference processing gaining targeted
  adoption in specific use cases.
- Intel: The foundry business landed a landmark $4.2 billion five-year US defense
  manufacturing agreement. The Gaudi 3 data center accelerator managed only about
  2% market penetration. A corporate restructuring will eliminate 15,000 positions
  by year-end 2025. Progress on Intel 18A manufacturing showed encouraging yield
  trajectory, attracting new foundry clients including MediaTek and additional US
  defense industry customers. The Altera programmable logic subsidiary exceeded
  expectations with $1.8 billion in quarterly revenue.
- Qualcomm: The Snapdragon X Elite processor for personal computers accumulated
  35 design commitments from leading device manufacturers. Revenue from AI-capable
  PCs hit $1.2 billion during the period. Qualcomm also introduced Snapdragon 8
  Gen 4, featuring proprietary Oryon CPU cores and a neural processing unit rated
  at 75 TOPS for edge AI inference. The automotive division's cumulative design-win
  pipeline expanded to $45 billion in projected lifetime revenue.
- Broadcom: Custom AI silicon (XPU) operations tripled in scale, with Google and
  Meta serving as anchor customers. AI-related total revenue was $4.1 billion.
  The VMware integration is paying dividends, as infrastructure software revenues
  climbed to $5.8 billion (+43% YoY). Networking ASIC products designed for AI
  cluster interconnect (Memory, Jericho3-AI, Ramon families) are experiencing
  rapid growth as hyperscale operators construct increasingly larger GPU clusters.

PACKAGING AND SUPPLY CHAIN
Advanced chip packaging technologies (CoWoS, InFO) continue to constrain AI
processor output. TSMC intends to boost CoWoS capacity by 2.5x before mid-2026,
but anticipated demand will grow 3x over that period, ensuring tight allocations
through at least 2026. Novel approaches such as silicon photonic interconnects
and chiplet-based designs leveraging the Universal Chiplet Interconnect Express
(UCIe) standard are gaining momentum, with 15 major semiconductor firms joining
the UCIe consortium during the quarter.

KEY INVESTMENT CONSIDERATIONS
1. Enterprise and hyperscaler spending on AI hardware infrastructure maintains
   strong momentum, expected to persist through 2026 and potentially beyond.
   Annual AI chip expenditure may approach $200 billion by 2027.
2. Memory industry leaders (SK Hynix, Samsung, Micron) occupy the most favorable
   position to monetize the HBM supercycle, benefiting from structurally elevated
   profit margins. Standalone HBM revenue could reach $40 billion annually by 2026.
3. AI processing at the network edge is gaining traction rapidly, opening market
   opportunities for Qualcomm, MediaTek, and emerging AI chip ventures. The edge
   AI processor segment may grow to $25 billion by 2027.
4. Companies with substantial China revenue face ongoing geopolitical uncertainty
   and risk premiums. Manufacturing diversification toward Japanese, American, and
   European locations continues to accelerate.
5. Developer toolchain lock-in (CUDA ecosystem, ROCm, OpenVINO) is emerging as
   a strategic differentiator on par with raw silicon performance.

=== END OF REPORT ===

Using ONLY the information in the report above, answer the user's question.
Be specific and cite numbers from the report."""

RAG_CONTEXT_B = """You are a healthcare policy expert assistant. Below is a detailed briefing
that you must use to answer the user's question accurately and completely.

=== POLICY BRIEFING: US HEALTHCARE REFORM PROPOSALS 2025 ===

OVERVIEW
The 119th Congress has introduced 47 healthcare-related bills in the current
session, with three major reform packages gaining bipartisan traction. Total
US healthcare expenditure reached $4.8 trillion in 2024 (17.8% of GDP), and
projections indicate it will exceed $5.1 trillion by 2026 without structural
reform.

PROPOSAL 1: PRESCRIPTION DRUG PRICING ACT (S.234)
Sponsors: Sen. Murray (D-WA), Sen. Collins (R-ME)
Key provisions:
- Extends Medicare drug price negotiation to 50 additional drugs per year
  (up from 20 under current law)
- Caps annual out-of-pocket drug spending at $2,000 for all insured Americans
- Requires pharmaceutical companies to justify price increases exceeding
  inflation by submitting cost transparency reports
- Creates an international reference pricing mechanism benchmarked to
  Canada, UK, Germany, Japan, and Australia
- Estimated CBO score: $287 billion in federal savings over 10 years
- Pharmaceutical industry opposition: PhRMA estimates it would reduce R&D
  investment by $180 billion over the same period

PROPOSAL 2: RURAL HEALTHCARE ACCESS ACT (H.R.891)
Sponsors: Rep. Thompson (R-PA), Rep. Sewell (D-AL)
Key provisions:
- $15 billion over 5 years for rural hospital stabilization grants
- Expands telehealth reimbursement parity to all Medicare services permanently
- Creates 5,000 new residency positions in rural and underserved areas
- Establishes a Rural Health Innovation Fund for AI-assisted diagnostics
- Raises Medicare reimbursement rates for Critical Access Hospitals by 12%
- Estimated to prevent closure of 120+ rural hospitals facing insolvency

PROPOSAL 3: MENTAL HEALTH INTEGRATION ACT (S.567)
Sponsors: Sen. Cassidy (R-LA), Sen. Stabenow (D-MI)
Key provisions:
- Mandates insurance coverage parity for mental health and substance abuse
  treatment with physical health services
- $8 billion for community mental health center expansion
- Integrates behavioral health screening into all primary care visits
- Creates a national 24/7 crisis response network (building on 988 Lifeline)
- Tax credits for employers providing comprehensive mental health benefits

BUDGET IMPACT
Combined 10-year cost: approximately $98 billion in new federal spending,
partially offset by $287 billion in drug pricing savings.

STAKEHOLDER POSITIONS
- AMA: Supports all three proposals with modifications
- PhRMA: Strongly opposes S.234, neutral on H.R.891 and S.567
- AARP: Strong support for drug pricing reform and mental health integration
- Rural health advocates: Enthusiastic support for H.R.891
- Insurance industry: Cautious support, concerns about mandate costs

=== END OF BRIEFING ===

Using ONLY the information in the briefing above, answer the user's question.
Be specific and cite provisions and numbers from the briefing."""

# User queries for each context
QUERIES_FOR_A = [
    "What was NVIDIA's data center revenue and market share in Q3 2025?",
    "How did the HBM memory market perform this quarter?",
    "What is the outlook for advanced packaging supply constraints?",
    "Compare AMD and Intel's data center performance.",
    "What are the key investment themes from this report?",
]

QUERIES_FOR_A_FOLLOWUP = [
    "What was NVIDIA's data center revenue growth rate year over year?",
    "How much did HBM3E average selling prices change quarter over quarter?",
    "When will TSMC's Arizona Fab 2 begin volume production?",
    "How much market share did AMD's MI300X accelerator gain compared to Q2?",
    "What is the estimated total AI infrastructure spending trajectory?",
]

QUERIES_FOR_B = [
    "What are the three major healthcare reform proposals?",
    "How much would the drug pricing act save over 10 years?",
    "What provisions does the Rural Healthcare Access Act include?",
    "What is the total healthcare expenditure projection for 2026?",
    "What are the key stakeholder positions on these proposals?",
]

@dataclass
class QueryResult:
    prompt_label: str
    phase: str
    ttft_ms: float
    total_ms: float
    tokens_generated: int = 0
    prompt_tokens_approx: int = 0
    error: str | None = None


@dataclass
class BenchmarkResults:
    run_id: str
    vllm_endpoint: str
    model: str
    started_at: str
    completed_at: str = ""
    phases: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)


async def measure_ttft(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int = 30,
) -> tuple[float, float, int]:
    """Send a streaming chat completion and measure TTFT."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
    }

    start = time.perf_counter()
    ttft = None
    token_count = 0

    async with session.post(
        f"{endpoint}/v1/chat/completions",
        json=payload,
        headers={"Content-Type": "application/json"},
    ) as resp:
        if resp.status != 200:
            text = await resp.text()
            raise RuntimeError(f"HTTP {resp.status}: {text[:200]}")

        async for line in resp.content:
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data: "):
                continue
            data = decoded[6:]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft is None:
                        ttft = (time.perf_counter() - start) * 1000
                    token_count += 1
            except json.JSONDecodeError:
                continue

    total = (time.perf_counter() - start) * 1000
    return ttft or total, total, token_count


async def run_phase(
    session: aiohttp.ClientSession,
    endpoint: str,
    model: str,
    prompts: list[tuple[str, str]],
    phase_name: str,
    repeats: int = 3,
    max_tokens: int = 30,
    warmup: bool = False,
) -> list[QueryResult]:
    """Run a benchmark phase with labeled prompts."""
    results = []
    for rep in range(repeats):
        for label, prompt in prompts:
            if warmup and rep == 0:
                # Warmup run — don't record
                try:
                    await measure_ttft(session, endpoint, model, prompt, max_tokens)
                except Exception:
                    pass
                continue
            try:
                ttft, total, tokens = await measure_ttft(
                    session, endpoint, model, prompt, max_tokens
                )
                results.append(QueryResult(
                    prompt_label=label,
                    phase=phase_name,
                    ttft_ms=ttft,
                    total_ms=total,
                    tokens_generated=tokens,
                    prompt_tokens_approx=len(prompt.split()) * 4 // 3,
                ))
            except Exception as e:
                results.append(QueryResult(
                    prompt_label=label,
                    phase=phase_name,
                    ttft_ms=-1,
                    total_ms=-1,
                    error=str(e)[:200],
                ))
        await asyncio.sleep(0.3)
    return results


def compute_stats(results: list[QueryResult]) -> dict:
    valid = [r.ttft_ms for r in results if r.ttft_ms > 0]
    if not valid:
        return {"n": 0, "p50": None, "p95": None, "mean": None, "errors": len(results)}
    valid_sorted = sorted(valid)
    n = len(valid_sorted)
    p50_idx = int(n * 0.50)
    p95_idx = min(int(n * 0.95), n - 1)
    return {
        "n": n,
        "p50": valid_sorted[p50_idx],
        "p95": valid_sorted[p95_idx],
        "mean": mean(valid),
        "min": valid_sorted[0],
        "max": valid_sorted[-1],
        "errors": len(results) - n,
    }


async def run_benchmark(
    endpoint: str,
    model: str,
    output_dir: str,
    repeats: int = 5,
) -> BenchmarkResults:
    run_id = f"semblend-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    bench = BenchmarkResults(
        run_id=run_id,
        vllm_endpoint=endpoint,
        model=model,
        started_at=datetime.now(timezone.utc).isoformat(),
    )

    timeout = aiohttp.ClientTimeout(total=120)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # Health check
        try:
            async with session.get(f"{endpoint}/health") as resp:
                if resp.status != 200:
                    print(f"vLLM health check failed: {resp.status}")
                    return bench
            print(f"vLLM healthy at {endpoint}")
        except Exception as e:
            print(f"Cannot connect to vLLM: {e}")
            return bench

        # Build prompt sets
        seed_prompts = [
            (f"seed-A-q{i}", f"{RAG_CONTEXT_A}\n\nQuestion: {q}")
            for i, q in enumerate(QUERIES_FOR_A)
        ]
        exact_prompts = seed_prompts.copy()

        # Prefix-vary: same RAG context, different questions
        prefix_vary_prompts = [
            (f"prefix-vary-A-q{i}", f"{RAG_CONTEXT_A}\n\nQuestion: {q}")
            for i, q in enumerate(QUERIES_FOR_A_FOLLOWUP)
        ]

        # Semantic: same information, rephrased context + same questions
        semantic_prompts = [
            (f"semantic-A-q{i}", f"{RAG_CONTEXT_A_VARIANT}\n\nQuestion: {q}")
            for i, q in enumerate(QUERIES_FOR_A)
        ]

        # Cold: completely different domain
        cold_prompts = [
            (f"cold-B-q{i}", f"{RAG_CONTEXT_B}\n\nQuestion: {q}")
            for i, q in enumerate(QUERIES_FOR_B)
        ]

        # Phase 1: Seed
        print(f"\n--- Phase 1: SEED ({len(seed_prompts)} prompts, 1 pass) ---")
        seed_results = await run_phase(
            session, endpoint, model, seed_prompts, "seed",
            repeats=1, max_tokens=50,
        )
        bench.phases["seed"] = {
            "results": [asdict(r) for r in seed_results],
            "stats": compute_stats(seed_results),
        }
        p50 = bench.phases["seed"]["stats"].get("p50")
        print(f"  Seed TTFT P50: {p50:.0f}ms" if p50 else "  Seed: errors")

        print("  Waiting 2s for cache propagation...")
        await asyncio.sleep(2)

        # Phase 2: Exact repeat
        print(f"\n--- Phase 2: EXACT ({len(exact_prompts)} x {repeats}) ---")
        exact_results = await run_phase(
            session, endpoint, model, exact_prompts, "exact", repeats=repeats,
        )
        bench.phases["exact"] = {
            "results": [asdict(r) for r in exact_results],
            "stats": compute_stats(exact_results),
        }
        p50 = bench.phases["exact"]["stats"].get("p50")
        print(f"  Exact TTFT P50: {p50:.0f}ms" if p50 else "  Exact: errors")

        # Phase 3: Prefix-vary (same context, different question)
        print(f"\n--- Phase 3: PREFIX_VARY ({len(prefix_vary_prompts)} x {repeats}) ---")
        pv_results = await run_phase(
            session, endpoint, model, prefix_vary_prompts, "prefix_vary", repeats=repeats,
        )
        bench.phases["prefix_vary"] = {
            "results": [asdict(r) for r in pv_results],
            "stats": compute_stats(pv_results),
        }
        p50 = bench.phases["prefix_vary"]["stats"].get("p50")
        print(f"  Prefix-vary TTFT P50: {p50:.0f}ms" if p50 else "  Prefix-vary: errors")

        # Phase 4: Semantic (rephrased context, same questions)
        print(f"\n--- Phase 4: SEMANTIC ({len(semantic_prompts)} x {repeats}) ---")
        sem_results = await run_phase(
            session, endpoint, model, semantic_prompts, "semantic", repeats=repeats,
        )
        bench.phases["semantic"] = {
            "results": [asdict(r) for r in sem_results],
            "stats": compute_stats(sem_results),
        }
        p50 = bench.phases["semantic"]["stats"].get("p50")
        print(f"  Semantic TTFT P50: {p50:.0f}ms" if p50 else "  Semantic: errors")

        # Phase 5: Cold (different domain entirely)
        print(f"\n--- Phase 5: COLD ({len(cold_prompts)} x {repeats}) ---")
        cold_results = await run_phase(
            session, endpoint, model, cold_prompts, "cold", repeats=repeats,
        )
        bench.phases["cold"] = {
            "results": [asdict(r) for r in cold_results],
            "stats": compute_stats(cold_results),
        }
        p50 = bench.phases["cold"]["stats"].get("p50")
        print(f"  Cold TTFT P50: {p50:.0f}ms" if p50 else "  Cold: errors")

    # Compute summary
    def get_p50(phase: str) -> float:
        return bench.phases.get(phase, {}).get("stats", {}).get("p50", 0) or 0

    seed_p50 = get_p50("seed")
    exact_p50 = get_p50("exact")
    pv_p50 = get_p50("prefix_vary")
    sem_p50 = get_p50("semantic")
    cold_p50 = get_p50("cold")

    bench.summary = {
        "seed_p50_ms": seed_p50,
        "exact_p50_ms": exact_p50,
        "prefix_vary_p50_ms": pv_p50,
        "semantic_p50_ms": sem_p50,
        "cold_p50_ms": cold_p50,
        "exact_speedup_vs_seed": seed_p50 / exact_p50 if exact_p50 else None,
        "prefix_vary_speedup_vs_seed": seed_p50 / pv_p50 if pv_p50 else None,
        "semantic_speedup_vs_seed": seed_p50 / sem_p50 if sem_p50 else None,
        "cold_speedup_vs_seed": seed_p50 / cold_p50 if cold_p50 else None,
        "semantic_vs_cold_speedup": cold_p50 / sem_p50 if sem_p50 else None,
        "prefix_vary_vs_cold_speedup": cold_p50 / pv_p50 if pv_p50 else None,
    }

    bench.completed_at = datetime.now(timezone.utc).isoformat()

    # Save results
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"{run_id}.json"
    with open(result_file, "w") as f:
        json.dump(asdict(bench), f, indent=2, default=str)
    print(f"\nResults saved to {result_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SemBlend KV Cache Reuse Benchmark Summary")
    print("=" * 70)
    print(f"  Model: {model}")
    print(f"  Endpoint: {endpoint}")
    print(f"  Repeats per phase: {repeats}")
    print("-" * 70)
    print(f"  SEED (cold prefill)    TTFT P50: {seed_p50:7.0f}ms")
    print(f"  EXACT (full cache hit) TTFT P50: {exact_p50:7.0f}ms  ({bench.summary.get('exact_speedup_vs_seed', 0):.2f}x vs seed)")
    print(f"  PREFIX_VARY            TTFT P50: {pv_p50:7.0f}ms  ({bench.summary.get('prefix_vary_speedup_vs_seed', 0):.2f}x vs seed)")
    print(f"  SEMANTIC (SemBlend)    TTFT P50: {sem_p50:7.0f}ms  ({bench.summary.get('semantic_speedup_vs_seed', 0):.2f}x vs seed)")
    print(f"  COLD (no cache)        TTFT P50: {cold_p50:7.0f}ms  ({bench.summary.get('cold_speedup_vs_seed', 0):.2f}x vs seed)")
    print("-" * 70)
    print(f"  Semantic vs Cold speedup: {bench.summary.get('semantic_vs_cold_speedup', 0):.2f}x")
    print(f"  Prefix-vary vs Cold:      {bench.summary.get('prefix_vary_vs_cold_speedup', 0):.2f}x")
    print("=" * 70)

    return bench


def main():
    parser = argparse.ArgumentParser(description="SemBlend KV Cache Reuse Benchmark")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--output", default="benchmarks/e2e/results")
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        endpoint=args.endpoint,
        model=args.model,
        output_dir=args.output,
        repeats=args.repeats,
    ))


if __name__ == "__main__":
    main()
