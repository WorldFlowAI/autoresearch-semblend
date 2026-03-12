#!/usr/bin/env python3
"""Empirical break-even measurement using sequential single-donor processing.

Processes one prompt at a time: cold → hit → miss, measuring all three conditions
cleanly. Uses diverse topics per prompt to prevent cross-prompt LMCache contamination.

Each measurement cycle:
1. Send prompt A (cold) — no donors match, pure prefill
2. Send prompt A again (hit) — same tokens, LMCache matches
3. Send prompt B (miss, completely different topic) — SemBlend searches but no match

The vector store grows with each cycle, but since each prompt has unique content
and the hit phase always uses the exact same prompt text, the nearest-neighbor
search returns the correct donor.

Usage:
    python -m benchmarks.e2e.breakeven_empirical_bench \
        --endpoint http://localhost:8100 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --n-samples 50 \
        --token-length 8192
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# Prompt template (Qwen ChatML format)
_PROMPT_TEMPLATE = (
    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
    "<|im_start|>user\n{context}\n\n"
    "Summarize the key points above.<|im_end|>\n"
    "<|im_start|>assistant\n"
)

# Each topic is a standalone paragraph with ~500 chars (~125 tokens).
# Topics are from maximally different domains to prevent cross-prompt
# semantic similarity from exceeding the SemBlend threshold (τ=0.60).
_TOPICS = [
    # 1. Underwater archaeology
    "Marine archaeologists discovered a 3,000-year-old Phoenician shipwreck off the coast "
    "of Malta containing intact amphorae filled with preserved olive oil and wine residue. "
    "The vessel's cedar hull planking reveals advanced shipbuilding techniques predating "
    "Greek naval architecture by centuries. Radiocarbon dating of rope fragments confirms "
    "the Bronze Age origin. The cargo manifest inscribed on a bronze tablet lists trading "
    "ports across the ancient Mediterranean basin. ",
    # 2. Quantum computing
    "IBM's latest quantum processor achieves 1,121 physical qubits arranged in a heavy-hex "
    "lattice topology with median two-qubit gate fidelity exceeding 99.5 percent. Error "
    "correction codes running on this chip demonstrate logical qubit lifetimes three orders "
    "of magnitude longer than physical qubit coherence times. The cryo-CMOS control "
    "electronics operating at 4 kelvin reduce wiring complexity between room temperature "
    "and the dilution refrigerator mixing chamber stage. ",
    # 3. Veterinary medicine
    "Feline hypertrophic cardiomyopathy remains the most common cardiac disease in domestic "
    "cats affecting approximately fifteen percent of the population. Recent genetic studies "
    "identified mutations in the MYBPC3 gene as a primary risk factor in Maine Coon and "
    "Ragdoll breeds. Echocardiographic screening using B-mode imaging can detect left "
    "ventricular wall thickening before clinical symptoms appear. Treatment protocols "
    "combining atenolol with clopidogrel show improved survival rates in longitudinal trials. ",
    # 4. Medieval cooking
    "The Forme of Cury compiled by Richard II's master cooks in 1390 contains 196 recipes "
    "documenting late medieval English cuisine. Dishes such as blank manger combined "
    "shredded capon with rice flour almond milk and sugar reflecting the sweet-savory "
    "profile characteristic of the period. Spice imports of cinnamon ginger and grains "
    "of paradise arrived via Venetian merchants trading at Alexandria. The manuscript "
    "reveals that saffron was the most expensive ingredient at twelve shillings per pound. ",
    # 5. Antarctic glaciology
    "Ice cores extracted from the West Antarctic Ice Sheet at WAIS Divide reach 3,405 "
    "meters depth spanning 68,000 years of climate history. Annual layer counting at "
    "centimeter resolution reveals abrupt warming events of 8-16 degrees Celsius occurring "
    "within decades during Dansgaard-Oeschger cycles. Trapped air bubbles preserve "
    "atmospheric composition samples showing pre-industrial carbon dioxide levels "
    "fluctuating between 180 and 280 parts per million across glacial-interglacial transitions. ",
    # 6. Jazz music theory
    "Coltrane's Giant Steps introduced a chord progression based on major third root motion "
    "dividing the octave into three equal parts creating the iconic B-G-Eb tonal axis. "
    "This tritone substitution pattern at maximum velocity challenged improvisers to abandon "
    "traditional ii-V-I vocabulary. The Coltrane changes influenced subsequent compositions "
    "by Wayne Shorter and McCoy Tyner who extended the concept into modal frameworks. "
    "Analysis of Coltrane's solo reveals systematic use of 1-2-3-5 digital patterns. ",
    # 7. Polymer chemistry
    "Ring-opening metathesis polymerization using Grubbs second-generation ruthenium "
    "catalysts enables precise synthesis of functionalized polynorbornene derivatives with "
    "narrow molecular weight distributions. The living character of ROMP allows block "
    "copolymer architectures through sequential monomer addition. Recent advances in "
    "aqueous ROMP extend this methodology to biocompatible hydrogel networks for tissue "
    "engineering scaffolds. Catalyst loadings as low as 0.01 mol percent achieve "
    "complete conversion within minutes under ambient conditions. ",
    # 8. Railway engineering
    "Slab track systems using continuously reinforced concrete provide superior geometric "
    "stability compared to ballasted track for high-speed rail operations exceeding 300 "
    "kilometers per hour. The Rheda 2000 design incorporates bi-block sleepers embedded "
    "in a concrete layer on a hydraulically bound base achieving maintenance-free operation "
    "for 60 years. Rail grinding profiles optimized for Hertzian contact stress reduce "
    "rolling contact fatigue crack initiation. Gauge corner lubrication systems decrease "
    "wheel flange wear by 70 percent on tight radius curves. ",
    # 9. Perfumery
    "Natural ambergris forms in the digestive tract of sperm whales as a waxy concretion "
    "around indigestible squid beaks. After decades of ocean exposure and photodegradation "
    "the material develops its characteristic sweet musky scent through oxidation of "
    "ambrein to ambrinol and ambroxide. Synthetic substitutes including Ambroxan derived "
    "from sclareol provide consistent olfactory profiles at a fraction of the cost. "
    "In fine perfumery ambroxide serves as a fixative extending the longevity of top notes. ",
    # 10. Mycology
    "Cordyceps militaris produces the bioactive compound cordycepin (3-deoxyadenosine) "
    "which exhibits broad-spectrum antiviral activity through inhibition of RNA-dependent "
    "RNA polymerase. Solid-state fermentation on brown rice substrate at 20 degrees Celsius "
    "and 85 percent relative humidity maximizes cordycepin yield to 8.3 milligrams per gram "
    "of dried fruiting body. The UPLC-MS/MS quantification method achieves detection limits "
    "of 0.1 nanograms per milliliter with 98 percent recovery rate. ",
    # 11. Textile restoration
    "Conservation of the Bayeux Tapestry requires monitoring thread tension across its "
    "68.38 meter length using fiber optic strain gauges embedded in the backing fabric. "
    "The original 10th century embroidery uses laid-and-couched technique with wool yarns "
    "dyed using woad (blue) madder (red) and weld (yellow) on a linen tabby-weave ground. "
    "Digital multispectral imaging reveals underdrawings and later repair campaigns "
    "distinguishable by their different thread twist directions (S-twist vs Z-twist). ",
    # 12. Volcanology
    "Infrasound monitoring arrays at distances of 50 to 250 kilometers from Kilauea "
    "volcano detect Strombolian explosion signals with dominant frequencies between 0.5 "
    "and 5 Hertz. Cross-correlation of signals from multiple array elements provides "
    "bearing estimates to the active vent with angular resolution of 0.3 degrees. "
    "Integration with seismic tremor amplitude and SO2 emission rates enables short-term "
    "eruption forecasting. The 2018 lower East Rift Zone eruption produced infrasound "
    "signals detectable at 9,000 kilometers distance. ",
    # 13. Numismatics
    "The 1933 Saint-Gaudens double eagle gold coin remains the most valuable numismatic "
    "item ever sold at auction reaching 18.9 million dollars in 2021. Of the 445,500 "
    "specimens minted none were officially released into circulation after Executive Order "
    "6102 mandated gold surrendering. Secret Service agents confiscated surviving examples "
    "over subsequent decades. The sole legally privately owned specimen traces its "
    "provenance through King Farouk of Egypt's collection acquired at a 1944 auction. ",
    # 14. Behavioral ecology
    "Honeybee waggle dances encode both the direction and distance of food sources relative "
    "to the hive entrance and solar azimuth. The duration of the waggle run correlates "
    "linearly with foraging distance at approximately 75 milliseconds per 100 meters. "
    "Neural recordings from the mushroom body reveal that recruits decode dance information "
    "through mechanosensory vibrations detected by Johnston's organ on the antenna. "
    "Colony-level foraging efficiency increases 30 percent when scouts perform dances "
    "for high-reward patches exceeding 2.0 molar sucrose concentration. ",
    # 15. Paper conservation
    "Iron gall ink corrosion threatens manuscripts dating from the 5th through 19th "
    "centuries as excess ferrous sulfate catalyzes cellulose chain scission through "
    "Fenton-type radical oxidation. Deacidification treatments using magnesium hydrogen "
    "carbonate solutions neutralize acid while depositing an alkaline reserve of 1-2 "
    "percent calcium carbonate equivalent. Nano-sized calcium hydroxide particles "
    "dispersed in isopropanol penetrate paper fibers without causing tidelines. The "
    "treatment raises pH from 3.5 to 8.0 extending paper lifetime by factor of ten. ",
    # 16. Stellar astrophysics
    "Type Ia supernovae originate from carbon-oxygen white dwarfs accreting mass from "
    "binary companions until reaching the Chandrasekhar limit of 1.44 solar masses. "
    "The thermonuclear detonation synthesizes approximately 0.6 solar masses of nickel-56 "
    "whose radioactive decay to cobalt-56 and iron-56 powers the optical light curve "
    "peak luminosity. Silicon II absorption at 6,150 angstroms in the photospheric "
    "spectrum serves as the primary classification criterion. These standardizable "
    "candles enabled the discovery of accelerating cosmic expansion in 1998. ",
    # 17. Cheese making
    "Gruyere AOP production begins with raw cow's milk heated to 33 degrees Celsius in "
    "copper vats before inoculation with thermophilic starter cultures containing "
    "Lactobacillus helveticus and Streptococcus thermophilus. Calf rennet coagulates "
    "the curd within 35 minutes after which it is cut to corn-kernel size and slowly "
    "heated to 57 degrees Celsius while stirring. The pressed wheels weighing 35 "
    "kilograms are brined for 24 hours then aged on spruce boards in climate-controlled "
    "cellars at 13 degrees and 95 percent humidity for minimum 5 months. ",
    # 18. Forensic entomology
    "Blow fly colonization sequences on human remains provide minimum postmortem interval "
    "estimates based on accumulated degree hours of larval development. Calliphora "
    "vicina typically arrives within minutes of death depositing egg masses of 150-200 "
    "eggs in natural body openings. At 20 degrees Celsius first instar larvae emerge "
    "after 24 hours progress through three instars over 6 days then pupate for 11 days. "
    "Species identification using mitochondrial COI barcoding achieves 99 percent accuracy "
    "even from degraded specimens recovered months after death. ",
    # 19. Architectural acoustics
    "Concert hall reverberation time of 1.8 to 2.2 seconds at 500 Hertz optimizes "
    "orchestral music clarity and warmth. The Boston Symphony Hall designed by Wallace "
    "Sabine in 1900 achieves a shoe-box geometry with 18,740 cubic meters volume and "
    "2,625 seats yielding 7.1 cubic meters per seat. Lateral reflections from parallel "
    "walls arriving within 80 milliseconds create acoustic intimacy measured by the "
    "interaural cross-correlation coefficient. Diffusing coffers on the ceiling "
    "prevent flutter echoes between parallel surfaces. ",
    # 20. Helicopter aerodynamics
    "Retreating blade stall limits conventional helicopter forward flight speed to "
    "approximately 170 knots as the advancing blade tip approaches Mach 0.9 while "
    "the retreating blade requires increased angle of attack beyond its stall margin. "
    "Coaxial rigid rotor designs eliminate this dissymmetry of lift by canceling "
    "lateral rolling moments between counter-rotating rotors. The ABC (Advancing "
    "Blade Concept) developed by Sikorsky offloads the retreating side to a pusher "
    "propeller enabling 250 knot cruise speeds demonstrated by the X2 Technology "
    "Demonstrator and S-97 Raider compound helicopter platforms. ",
]


def _build_context(topic_idx: int, target_chars: int) -> str:
    """Build unique context by repeating a specific topic paragraph."""
    topic = _TOPICS[topic_idx % len(_TOPICS)]
    repeats = max(1, target_chars // len(topic) + 1)
    return (topic * repeats)[:target_chars]


def _build_miss_context(seed: int, target_chars: int) -> str:
    """Build miss context from a different topic than the main prompt."""
    import random
    rng = random.Random(seed + 99999)
    # Use topics from a shifted range to minimize overlap
    shifted = _TOPICS[10:] + _TOPICS[:10]
    parts: list[str] = []
    total = 0
    while total < target_chars:
        parts.append(rng.choice(shifted))
        total += len(parts[-1])
    return "".join(parts)[:target_chars]


def ttft_request(
    endpoint: str, model: str, prompt: str, max_tokens: int = 5
) -> tuple[float, bool]:
    """Send a completions request, return (latency_ms, ok)."""
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
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        print(f"    request failed ({elapsed:.0f}ms): {exc}")
        return 0.0, False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Empirical break-even measurement (sequential)"
    )
    parser.add_argument("--endpoint", default="http://localhost:8100")
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    parser.add_argument("--n-samples", type=int, default=50)
    parser.add_argument("--token-length", type=int, default=8192)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    target_chars = args.token_length * 4
    print(f"\nEmpirical Break-Even Measurement")
    print(f"  endpoint={args.endpoint}")
    print(f"  model={args.model}")
    print(f"  n_samples={args.n_samples}")
    print(f"  token_length={args.token_length}")
    print()

    # Health check
    requests.get(f"{args.endpoint}/health", timeout=10).raise_for_status()

    cold_latencies: list[float] = []
    hit_latencies: list[float] = []
    miss_latencies: list[float] = []

    for i in range(args.n_samples):
        topic_idx = i % len(_TOPICS)
        context = _build_context(topic_idx, target_chars)
        prompt = _PROMPT_TEMPLATE.format(context=context)

        miss_context = _build_miss_context(i, target_chars)
        miss_prompt = _PROMPT_TEMPLATE.format(context=miss_context)

        # 1. Cold: first time seeing this exact text
        t_cold, ok_cold = ttft_request(args.endpoint, args.model, prompt)
        cold_tag = f"{t_cold:.0f}ms" if ok_cold else "FAIL"

        # 2. Hit: re-send exact same prompt (LMCache should match)
        t_hit, ok_hit = ttft_request(args.endpoint, args.model, prompt)
        hit_tag = f"{t_hit:.0f}ms" if ok_hit else "FAIL"

        # 3. Miss: different topic entirely
        t_miss, ok_miss = ttft_request(args.endpoint, args.model, miss_prompt)
        miss_tag = f"{t_miss:.0f}ms" if ok_miss else "FAIL"

        ratio = ""
        if ok_cold and ok_hit:
            ratio = f"speedup={t_cold / t_hit:.2f}x"

        print(f"  [{i + 1}/{args.n_samples}] topic={topic_idx:>2} "
              f"cold={cold_tag:>8} hit={hit_tag:>8} miss={miss_tag:>8} {ratio}")

        if ok_cold:
            cold_latencies.append(t_cold)
        if ok_hit:
            hit_latencies.append(t_hit)
        if ok_miss:
            miss_latencies.append(t_miss)

    # Results
    c_mean = statistics.mean(cold_latencies) if cold_latencies else 0
    h_mean = statistics.mean(hit_latencies) if hit_latencies else 0
    m_mean = statistics.mean(miss_latencies) if miss_latencies else 0

    c_p50 = statistics.median(cold_latencies) if cold_latencies else 0
    h_p50 = statistics.median(hit_latencies) if hit_latencies else 0
    m_p50 = statistics.median(miss_latencies) if miss_latencies else 0

    overhead = max(0, m_mean - c_mean)
    savings = max(0, c_mean - h_mean)
    denom = overhead + savings
    breakeven = overhead / denom if denom > 0 else 1.0

    print(f"\n{'=' * 60}")
    print(f"Results ({args.token_length} tokens, n={args.n_samples})")
    print(f"{'=' * 60}")
    print(f"  Cold  mean={c_mean:.0f}ms  p50={c_p50:.0f}ms  n={len(cold_latencies)}")
    print(f"  Hit   mean={h_mean:.0f}ms  p50={h_p50:.0f}ms  n={len(hit_latencies)}")
    print(f"  Miss  mean={m_mean:.0f}ms  p50={m_p50:.0f}ms  n={len(miss_latencies)}")
    print(f"  Overhead (M-C):  {overhead:.0f}ms")
    print(f"  Savings  (C-H):  {savings:.0f}ms")
    print(f"  Break-even P_h*: {breakeven:.1%}")

    for ph in [0.25, 0.50, 0.75, 1.00]:
        exp = ph * h_mean + (1 - ph) * m_mean
        spd = c_mean / exp if exp > 0 else 0
        print(f"  Speedup@{ph:.0%}: {spd:.2f}x")

    print()

    if args.output:
        result = {
            "endpoint": args.endpoint,
            "model": args.model,
            "n_samples": args.n_samples,
            "token_length": args.token_length,
            "cold_mean_ms": round(c_mean, 1),
            "cold_p50_ms": round(c_p50, 1),
            "hit_mean_ms": round(h_mean, 1),
            "hit_p50_ms": round(h_p50, 1),
            "miss_mean_ms": round(m_mean, 1),
            "miss_p50_ms": round(m_p50, 1),
            "overhead_miss_ms": round(overhead, 1),
            "savings_hit_ms": round(savings, 1),
            "breakeven_hit_rate": round(breakeven, 4),
            "raw_cold": [round(x, 1) for x in cold_latencies],
            "raw_hit": [round(x, 1) for x in hit_latencies],
            "raw_miss": [round(x, 1) for x in miss_latencies],
        }
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(result, indent=2))
        print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
