#!/usr/bin/env python3
"""RoPE FORCE_DELTA E2E benchmark — Gap 2 validation.

Proves that at Δ≥256 tokens:
  - Uncorrected: PPL > 1.10 (garbled output)
  - Corrected:   PPL < 1.05 (quality preserved)

Restarts vLLM between conditions to clear prefix cache.
Uses /tmp/semblend_force_delta.json for runtime config.

Usage:
    python rope_force_delta_e2e.py [--n N] [--pod POD] [--namespace NS]
"""

import argparse
import json
import math
import os
import subprocess
import sys
import time

import requests

BASE_URL = "http://localhost:8100/v1"
MODEL = "Qwen/Qwen2.5-7B-Instruct-AWQ"
LOCAL_FILES = [
    ("synapse_kv_connector/semblend_connector.py", "synapse_kv_connector/semblend_connector.py"),
    ("synapse_kv_connector/model_runner_hook.py", "synapse_kv_connector/model_runner_hook.py"),
    ("synapse_kv_connector/rope_correction.py", "synapse_kv_connector/rope_correction.py"),
    ("semblend_core/rope_correction.py", "semblend_core/rope_correction.py"),
    ("semblend_core/__init__.py", "semblend_core/__init__.py"),
]
REPO_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 10 diverse long documents (~1200-1500 tokens each)
DOCUMENTS = [
    # Doc 0: Climate change
    """Climate change refers to long-term shifts in temperatures and weather patterns across the globe. While climate change can occur naturally through volcanic eruptions, solar activity variations, and natural greenhouse gas fluctuations, the current period of global warming is primarily driven by human activities, particularly the burning of fossil fuels such as coal, oil, and natural gas. These activities release large quantities of greenhouse gases into the atmosphere, trapping heat and raising the planet's average temperature. The scientific consensus on climate change is overwhelming. The Intergovernmental Panel on Climate Change (IPCC) has concluded with high confidence that human influence has warmed the climate at a rate unprecedented in at least the last 2,000 years. Global surface temperature has increased by approximately 1.1 degrees Celsius above pre-industrial levels. The impacts of climate change are wide-ranging and increasingly severe. Rising temperatures are causing glaciers and ice sheets to melt, contributing to sea level rise that threatens coastal communities. Extreme weather events are becoming more frequent and intense. Heat waves, droughts, and wildfires are occurring with greater regularity, while heavy precipitation events and flooding are increasing. Oceans have absorbed over 90 percent of the excess heat, leading to ocean warming and acidification that disrupts marine ecosystems. The transition to clean energy is central to addressing climate change. Renewable energy sources, particularly solar and wind power, have seen dramatic cost reductions, making them competitive with fossil fuels. The Paris Agreement set the goal of limiting warming to well below 2 degrees Celsius. Carbon removal technologies are gaining attention as a complement to emissions reductions. Climate justice is an increasingly important dimension, as developing countries are often the most vulnerable to impacts despite contributing least to emissions.""",

    # Doc 1: Artificial intelligence
    """Artificial intelligence (AI) has emerged as one of the most transformative technologies of the 21st century, fundamentally changing how we interact with computers and process information. At its core, AI refers to computer systems designed to perform tasks that typically require human intelligence, including visual perception, speech recognition, decision-making, and language translation. The field has evolved dramatically since its inception in the 1950s, driven by advances in computing power, data availability, and algorithmic innovation. Machine learning, a subset of AI, enables systems to learn from data without being explicitly programmed. Deep learning, using neural networks with multiple layers, has achieved remarkable breakthroughs in image recognition, natural language processing, and game playing. The development of transformer architectures has revolutionized NLP, leading to large language models capable of generating human-like text, answering questions, and writing code. These models are trained on vast amounts of text data and can perform a wide range of tasks with minimal fine-tuning. The applications of AI span virtually every industry. In healthcare, AI systems assist with diagnosis, drug discovery, and personalized treatment plans. Autonomous vehicles rely on AI for perception and decision-making. Financial institutions use AI for fraud detection and algorithmic trading. The creative arts are being transformed by generative AI that can produce images, music, and text. However, AI also raises significant ethical concerns, including bias in training data, privacy implications, job displacement, and the potential for misuse. The development of artificial general intelligence (AGI), which would match or exceed human cognitive abilities across all domains, remains a subject of intense research and debate about its timeline and implications for society.""",

    # Doc 2: Space exploration
    """Space exploration has captivated human imagination for millennia and has become one of humanity's most ambitious scientific endeavors. From the early days of rocket development in the mid-20th century to today's plans for Mars colonization, the exploration of space has yielded profound scientific discoveries and technological innovations. The Space Age began in earnest with the Soviet Union's launch of Sputnik in 1957, followed by Yuri Gagarin's historic first human spaceflight in 1961. The Apollo program culminated in the first Moon landing in 1969, when Neil Armstrong and Buzz Aldrin walked on the lunar surface. The Space Shuttle program operated from 1981 to 2011, enabling the construction of the International Space Station (ISS), which has been continuously inhabited since 2000. Recent developments in space exploration have been driven by both government agencies and private companies. SpaceX has revolutionized the industry with reusable rockets, dramatically reducing launch costs. Blue Origin, Virgin Galactic, and other companies are pursuing various approaches to space tourism and commercial launch services. NASA's Artemis program aims to return humans to the Moon and establish a sustainable lunar presence. Mars exploration remains a major focus, with rovers like Curiosity and Perseverance providing detailed data about the Red Planet's geology and potential for past life. The James Webb Space Telescope has opened new windows into the cosmos, capturing unprecedented images of distant galaxies and exoplanets. Future missions aim to explore the icy moons of Jupiter and Saturn, where subsurface oceans may harbor conditions suitable for life. The search for extraterrestrial intelligence continues to capture public imagination, while advances in propulsion technology may eventually make interstellar travel feasible.""",

    # Doc 3: Renewable energy
    """Renewable energy has undergone a remarkable transformation over the past two decades, evolving from a niche alternative to a mainstream power source that is reshaping the global energy landscape. The transition from fossil fuels to renewable sources is driven by the dual imperatives of addressing climate change and meeting growing energy demand in a sustainable manner. Solar photovoltaic technology has experienced the most dramatic cost reduction, with prices falling by approximately 90 percent since 2010. Utility-scale solar farms now produce electricity at costs competitive with or below natural gas and coal in most markets. Rooftop solar installations have made distributed generation accessible to homeowners and businesses, transforming the traditional utility business model. Wind energy has similarly seen significant growth, with both onshore and offshore installations expanding rapidly. Modern wind turbines are larger and more efficient than their predecessors, with offshore turbines now reaching capacities of 12-15 megawatts. Energy storage is crucial for addressing the intermittency of solar and wind power. Lithium-ion battery costs have fallen dramatically, making grid-scale storage increasingly viable. Flow batteries, compressed air energy storage, and green hydrogen are emerging as additional storage options for longer-duration needs. The electrification of transportation is accelerating, with electric vehicle sales growing rapidly in major markets. Smart grid technologies and demand response programs are helping to balance supply and demand on electricity networks with high penetrations of variable renewable energy. The green hydrogen economy represents a promising pathway for decarbonizing hard-to-abate sectors such as heavy industry, shipping, and aviation. Nuclear energy, while controversial, continues to play a role in low-carbon electricity generation, with advanced reactor designs promising improved safety and efficiency.""",

    # Doc 4: Neuroscience
    """Neuroscience, the scientific study of the nervous system, has made extraordinary advances in recent decades, transforming our understanding of how the brain works and how neural processes give rise to cognition, emotion, and behavior. The human brain, with its approximately 86 billion neurons connected by trillions of synapses, is arguably the most complex structure in the known universe. Modern neuroscience draws on tools and concepts from biology, chemistry, physics, computer science, and psychology to investigate neural function at multiple levels, from individual molecules to large-scale brain networks. Neuroimaging technologies have revolutionized the field. Functional magnetic resonance imaging (fMRI) allows researchers to observe brain activity in real time by measuring blood flow changes. Magnetoencephalography (MEG) provides millisecond-scale temporal resolution for tracking neural dynamics. Advances in electron microscopy have enabled the mapping of neural circuits at synaptic resolution, contributing to the emerging field of connectomics. Optogenetics, which uses light to control genetically modified neurons, has become an indispensable tool for establishing causal relationships between neural activity and behavior. The study of neural plasticity has revealed that the brain continues to change throughout life, adapting to experience and learning. Understanding the molecular mechanisms of synaptic plasticity has implications for treating neurological disorders and enhancing cognitive function. Neurodegenerative diseases like Alzheimer's and Parkinson's affect millions of people worldwide, and research into their underlying mechanisms is yielding potential therapeutic targets. Brain-computer interfaces are advancing rapidly, offering hope for individuals with paralysis and other neurological conditions. The intersection of neuroscience and artificial intelligence is producing insights in both directions, with neural network architectures inspired by brain function and AI tools helping to analyze complex neural data.""",

    # Doc 5: Quantum computing
    """Quantum computing represents a fundamentally different approach to computation that harnesses the principles of quantum mechanics to solve problems that are intractable for classical computers. Unlike classical bits, which can exist only in states of 0 or 1, quantum bits (qubits) can exist in superpositions of both states simultaneously, enabling massive parallelism. Quantum entanglement allows qubits to be correlated in ways that have no classical analog, providing computational advantages for certain classes of problems. The development of quantum computers has progressed from theoretical proposals in the 1980s to working prototypes with hundreds of qubits today. Superconducting circuits, trapped ions, photonic systems, and topological qubits represent the leading hardware approaches, each with distinct advantages and challenges. Google claimed quantum supremacy in 2019 when its Sycamore processor completed a calculation in 200 seconds that would have taken classical supercomputers thousands of years. IBM, Microsoft, and numerous startups are investing heavily in quantum hardware and software ecosystems. Quantum error correction remains one of the most significant challenges, as qubits are extremely sensitive to environmental noise and decoherence. Logical qubits, which are encoded using many physical qubits, are needed for fault-tolerant quantum computation. Quantum algorithms like Shor's algorithm for factoring large numbers and Grover's algorithm for unstructured search demonstrate theoretical speedups, with implications for cryptography, optimization, drug discovery, and materials science. Quantum machine learning explores the intersection of quantum computing and artificial intelligence, potentially offering exponential speedups for certain learning tasks. Post-quantum cryptography is being developed to protect against future quantum attacks on current encryption standards.""",

    # Doc 6: Ocean ecosystems
    """The world's oceans cover approximately 71 percent of Earth's surface and contain 97 percent of the planet's water. These vast marine ecosystems support an incredible diversity of life, from microscopic phytoplankton that produce roughly half of the world's oxygen to the largest animals ever to have lived, the blue whales. Ocean ecosystems are structured by depth, temperature, salinity, and light availability, creating distinct zones from the sunlit euphotic zone near the surface to the abyssal depths thousands of meters below. Coral reefs, often called the rainforests of the sea, support approximately 25 percent of all marine species despite covering less than 1 percent of the ocean floor. These complex ecosystems are built by tiny coral polyps that secrete calcium carbonate skeletons over thousands of years. Ocean acidification, caused by the absorption of atmospheric carbon dioxide, is weakening coral structures and threatening reef ecosystems worldwide. Rising ocean temperatures are causing mass coral bleaching events with increasing frequency. Deep-sea hydrothermal vents, discovered in 1977, support unique ecosystems based on chemosynthesis rather than photosynthesis. The mesopelagic zone, between 200 and 1,000 meters depth, contains an enormous biomass of organisms that migrate vertically each day in the largest animal migration on Earth. Ocean circulation patterns, driven by temperature, salinity, and wind, distribute heat and nutrients globally. The thermohaline circulation acts as a conveyor belt, connecting surface and deep waters across ocean basins. Overfishing, plastic pollution, habitat destruction, and climate change represent major threats to marine ecosystems. Marine protected areas and sustainable fishing practices are critical tools for conservation.""",

    # Doc 7: Ancient civilizations
    """Ancient civilizations have left an indelible mark on human history, establishing the foundations of agriculture, writing, governance, architecture, and scientific inquiry that continue to influence modern society. The Fertile Crescent, spanning modern-day Iraq, Syria, and surrounding regions, gave rise to some of the earliest complex societies, including Sumer, Babylon, and Assyria. The Sumerians developed cuneiform writing around 3400 BCE, one of the earliest writing systems, used for recording economic transactions, laws, and literary works such as the Epic of Gilgamesh. Ancient Egypt flourished along the Nile River for over three millennia, producing monumental architecture including the Great Pyramid of Giza, built around 2560 BCE, and developing sophisticated systems of mathematics, astronomy, and medicine. The Indus Valley Civilization, centered in present-day Pakistan and northwestern India, created remarkably planned urban centers like Mohenjo-daro and Harappa with advanced drainage and water management systems. Chinese civilization developed independently along the Yellow River, producing innovations including paper, gunpowder, the compass, and printing. The Greek city-states, particularly Athens, laid the groundwork for Western philosophy, democracy, and scientific thought. Thinkers like Socrates, Plato, and Aristotle explored questions that remain central to philosophical inquiry. The Roman Empire built upon Greek foundations, creating an extensive network of roads, aqueducts, and legal systems that shaped European civilization. In the Americas, the Maya, Aztec, and Inca civilizations developed complex societies with advanced mathematics, astronomy, agricultural techniques, and monumental architecture independently of Old World influences.""",

    # Doc 8: Genetic engineering
    """Genetic engineering has transformed biological research and opened new frontiers in medicine, agriculture, and industrial biotechnology. The ability to precisely modify DNA sequences has evolved from early recombinant DNA techniques in the 1970s to today's sophisticated genome editing tools. The discovery of CRISPR-Cas9 in 2012 by Jennifer Doudna and Emmanuelle Charpentier, for which they received the Nobel Prize in Chemistry in 2020, revolutionized the field by providing an affordable, accurate, and efficient method for editing genomes in virtually any organism. CRISPR-Cas9 works by using a guide RNA to direct the Cas9 enzyme to a specific DNA sequence, where it creates a double-strand break that can be repaired by the cell's natural mechanisms, allowing researchers to insert, delete, or modify specific genes. Applications in medicine include the development of gene therapies for genetic disorders such as sickle cell disease and beta-thalassemia. CAR-T cell therapy, which involves genetically modifying a patient's immune cells to target cancer, has shown remarkable success in treating certain blood cancers. Agricultural biotechnology has produced crops with improved yields, pest resistance, and nutritional profiles. Golden rice, engineered to produce beta-carotene, addresses vitamin A deficiency in developing countries. Genetically modified organisms (GMOs) remain controversial despite scientific consensus on their safety. Emerging applications include gene drives for controlling disease-carrying mosquitoes, synthetic biology for producing biofuels and pharmaceuticals, and xenotransplantation research using genetically modified pig organs. Base editing and prime editing represent next-generation tools that can make precise single-base changes without creating double-strand breaks, reducing the risk of unintended mutations.""",

    # Doc 9: Urban planning
    """Urban planning has evolved dramatically over the past century as cities worldwide grapple with rapid population growth, environmental sustainability, transportation challenges, and the need to create livable communities. More than half of the world's population now lives in urban areas, a figure projected to reach 68 percent by 2050 according to the United Nations. This urbanization trend has intensified pressure on city infrastructure, housing, public services, and natural environments. Modern urban planning integrates transportation, land use, environmental protection, social equity, and economic development into comprehensive frameworks for city growth. Transit-oriented development concentrates mixed-use buildings around public transportation nodes, reducing car dependency and promoting walkability. Complete streets policies ensure that roads accommodate pedestrians, cyclists, and public transit users alongside motor vehicles. Green infrastructure strategies incorporate parks, urban forests, green roofs, and permeable surfaces to manage stormwater, reduce heat island effects, and improve air quality. Smart city technologies leverage sensors, data analytics, and the Internet of Things to optimize energy use, traffic flow, waste management, and public safety. However, the digital divide and surveillance concerns require careful consideration of equity and privacy. Affordable housing remains one of the most pressing urban challenges, with many cities experiencing housing crises driven by speculation, limited supply, and growing inequality. Zoning reform movements advocate for allowing greater housing density, mixed-use development, and the elimination of single-family-only zones. Climate resilience planning prepares cities for sea level rise, extreme heat events, flooding, and other climate impacts through infrastructure adaptation and emergency preparedness.""",
]

# Same instruction for seed and hit. System message isolation prevents
# prefix cache reuse while keeping user content identical for LMCache
# chunk matching. FORCE_DELTA then operates on correctly-loaded KV.
INSTRUCTION = "Given the following document, provide a comprehensive summary of the main points discussed.\n\nDocument:\n"

# Counter for unique system messages
_msg_counter = 0


def kubectl(*args, capture=True):
    """Run kubectl command."""
    cmd = ["kubectl"] + list(args)
    return subprocess.run(cmd, capture_output=capture, text=True)


def get_pod(ns):
    """Get vLLM pod name."""
    r = kubectl("get", "pods", "-n", ns, "-l", "app=vllm",
                "--no-headers", "-o", "custom-columns=:metadata.name")
    pod = r.stdout.strip().split("\n")[0]
    return pod if pod else None


def restart_vllm(ns, timeout=300):
    """Restart vLLM by killing process and re-injecting code files."""
    pod = get_pod(ns)
    if not pod:
        print("ERROR: No pod found")
        return None

    # Kill vLLM
    print(f"  Killing vLLM in {pod}...")
    kubectl("exec", "-n", ns, pod, "--", "pkill", "-f", "vllm")

    # Wait for container restart — CrashLoopBackOff adds increasing delay
    # (10s, 20s, 40s...). We need to wait for the container to come back
    # and enter the WAITING state.
    print("  Waiting for container restart (CrashLoopBackOff delay)...")
    for attempt in range(60):
        time.sleep(5)
        pod = get_pod(ns)
        if not pod:
            continue
        # Check if container is running and in WAITING state
        r = kubectl("logs", "-n", ns, pod, "--tail=1")
        if "WAITING" in r.stdout:
            print(f"  Container ready after {(attempt+1)*5}s")
            break
        # Also check if still in CrashLoopBackOff (container not started yet)
        r2 = kubectl("get", "pods", "-n", ns, pod, "--no-headers")
        status = r2.stdout.strip()
        if "CrashLoopBackOff" in status:
            continue  # Still waiting for backoff to expire
        if "Running" in status and "0/1" in status:
            continue  # Container starting
    else:
        print("ERROR: Pod never entered WAITING state")
        return None

    # Copy files
    print(f"  Copying source files to {pod}...")
    for local_rel, remote_rel in LOCAL_FILES:
        local_path = os.path.join(REPO_DIR, local_rel)
        remote_path = f"/opt/synapse/{remote_rel}"
        kubectl("cp", local_path, f"{ns}/{pod}:{remote_path}")

    # Clear pycache and trigger start
    kubectl("exec", "-n", ns, pod, "--", "bash", "-c",
            "find /opt/synapse -name '__pycache__' -exec rm -rf {} + 2>/dev/null; "
            "touch /tmp/code-ready")

    # Re-establish port-forward BEFORE health check
    print("  Setting up port-forward...")
    subprocess.run(["pkill", "-f", "kubectl port-forward.*autoresearch"],
                   capture_output=True)
    time.sleep(1)
    subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, pod, "8100:8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)

    # Wait for API server ready (model load ~60-90s)
    print(f"  Waiting for vLLM to load model ({timeout}s timeout)...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get("http://localhost:8100/health", timeout=3)
            if r.status_code == 200:
                print(f"  vLLM ready in {time.time()-start:.0f}s")
                return pod
        except Exception:
            pass
        time.sleep(5)

    print("ERROR: vLLM did not become ready")
    return None


def set_force_delta(pod, ns, delta, correct):
    """Write config to pod's /tmp/semblend_force_delta.json."""
    cfg = json.dumps({"delta": delta, "correct": correct})
    kubectl("exec", "-n", ns, pod, "--", "bash", "-c",
            f"echo '{cfg}' > /tmp/semblend_force_delta.json")


def recover_port_forward(ns):
    """Kill and re-establish port-forward."""
    subprocess.run(["pkill", "-f", "kubectl port-forward.*autoresearch"],
                   capture_output=True)
    time.sleep(1)
    pod = get_pod(ns)
    if not pod:
        return False
    subprocess.Popen(
        ["kubectl", "port-forward", "-n", ns, pod, "8100:8000"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    try:
        r = requests.get("http://localhost:8100/health", timeout=5)
        return r.status_code == 200
    except Exception:
        return False


def query(doc_idx, ns="autoresearch", system_msg=None, max_tokens=100, retries=3):
    """Send a chat completion request with optional system message isolation."""
    doc = DOCUMENTS[doc_idx]
    prompt = f"{INSTRUCTION}{doc}"
    messages = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": prompt})
    for attempt in range(retries + 1):
        try:
            t0 = time.time()
            r = requests.post(f"{BASE_URL}/chat/completions", json={
                "model": MODEL,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": 0,
                "logprobs": True,
                "top_logprobs": 1,
            }, timeout=180)
            elapsed = time.time() - t0
            if r.status_code != 200:
                return {"error": r.text, "elapsed": elapsed}
            d = r.json()
            choice = d["choices"][0]
            logprobs_list = []
            if "logprobs" in choice and choice["logprobs"]:
                content_lp = choice["logprobs"].get("content", [])
                for tok in content_lp:
                    lp = tok.get("logprob")
                    if lp is not None:
                        logprobs_list.append(lp)
            ppl = math.exp(-sum(logprobs_list) / len(logprobs_list)) if logprobs_list else float("nan")
            return {
                "text": choice["message"]["content"],
                "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"],
                "ppl": ppl,
                "elapsed": elapsed,
                "n_logprobs": len(logprobs_list),
            }
        except requests.exceptions.Timeout:
            if attempt < retries:
                print(f"    Timeout on attempt {attempt+1}, recovering port-forward...")
                recover_port_forward(ns)
                continue
            return {"error": "timeout after retries", "elapsed": time.time() - t0}
        except requests.exceptions.ConnectionError:
            if attempt < retries:
                print(f"    Connection error on attempt {attempt+1}, recovering port-forward...")
                recover_port_forward(ns)
                continue
            return {"error": "connection error after retries", "elapsed": 0}


def run_condition(name, pod, ns, delta, correct, n):
    """Run N document pairs under a specific condition.

    Uses same instruction for seed and hit. System messages provide
    prefix cache isolation: unique per request so prefix cache can't
    short-circuit, but user content is identical so LMCache chunks match.
    """
    global _msg_counter
    set_force_delta(pod, ns, delta, correct)
    print(f"\n{'='*60}")
    print(f"Condition: {name} (delta={delta}, correct={correct})")
    print(f"{'='*60}")

    results = []
    for i in range(n):
        doc_idx = i % len(DOCUMENTS)
        _msg_counter += 1
        # Seed with unique system message (creates donor)
        seed_sys = f"Request #{_msg_counter:06d}. You are a helpful assistant."
        seed = query(doc_idx, ns=ns, system_msg=seed_sys, max_tokens=5)
        if "error" in seed:
            print(f"  [{i}] SEED ERROR: {seed['error'][:100]}")
            continue
        time.sleep(2)
        _msg_counter += 1
        # Hit with DIFFERENT system message (prevents prefix cache reuse,
        # but same user content → LMCache chunks match after system msg)
        hit_sys = f"Request #{_msg_counter:06d}. You are a helpful assistant."
        hit = query(doc_idx, ns=ns, system_msg=hit_sys, max_tokens=100)
        if "error" in hit:
            print(f"  [{i}] HIT ERROR: {hit['error'][:100]}")
            continue
        results.append(hit)
        print(
            f"  [{i}] doc={doc_idx} PPL={hit['ppl']:.3f} "
            f"tokens={hit['completion_tokens']} "
            f"time={hit['elapsed']:.1f}s "
            f"text={hit['text'][:80]}..."
        )
        time.sleep(1)

    if results:
        ppls = [r["ppl"] for r in results if not math.isnan(r["ppl"])]
        avg_ppl = sum(ppls) / len(ppls) if ppls else float("nan")
        print(f"\n  >>> {name}: avg PPL = {avg_ppl:.4f} (n={len(ppls)})")
    return results


def main():
    parser = argparse.ArgumentParser(description="RoPE FORCE_DELTA E2E benchmark")
    parser.add_argument("--n", type=int, default=5, help="Docs per condition")
    parser.add_argument("--namespace", type=str, default="autoresearch")
    parser.add_argument("--deltas", type=str, default="256", help="Comma-separated deltas")
    parser.add_argument("--no-restart", action="store_true",
                        help="Skip vLLM restart between conditions (risk cache contamination)")
    args = parser.parse_args()

    ns = args.namespace
    deltas = [int(d) for d in args.deltas.split(",")]

    # Check port-forward
    try:
        r = requests.get("http://localhost:8100/health", timeout=5)
        assert r.status_code == 200
        print("vLLM is healthy")
    except Exception:
        print("ERROR: vLLM not reachable. Set up port-forward first:")
        print(f"  kubectl port-forward -n {ns} <pod> 8100:8000 &")
        sys.exit(1)

    pod = get_pod(ns)
    print(f"Pod: {pod}")

    # Define conditions: (name, delta, correct)
    conditions = [("baseline", 0, False)]
    for d in deltas:
        conditions.append((f"uncorrected_d{d}", d, False))
    for d in deltas:
        conditions.append((f"corrected_d{d}", d, True))

    all_results = {}

    for ci, (cname, delta, correct) in enumerate(conditions):
        if ci > 0 and not args.no_restart:
            print(f"\n--- Restarting vLLM for condition: {cname} ---")
            pod = restart_vllm(ns)
            if not pod:
                print(f"FATAL: Cannot restart for {cname}")
                break

        all_results[cname] = run_condition(cname, pod, ns, delta, correct, args.n)

    # Restore baseline
    if pod:
        set_force_delta(pod, ns, 0, False)

    # Compute baseline PPL for ratio-based pass criteria
    baseline_ppls = [r["ppl"] for r in all_results.get("baseline", []) if not math.isnan(r["ppl"])]
    baseline_avg = sum(baseline_ppls) / len(baseline_ppls) if baseline_ppls else 1.0

    # Summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY — RoPE FORCE_DELTA E2E Validation")
    print("=" * 80)
    print(f"{'Condition':<25} {'Avg PPL':>10} {'Ratio':>8} {'Stdev':>8} {'N':>4} {'Pass?':>8}")
    print("-" * 80)

    gap2_pass = True
    for key, results in all_results.items():
        ppls = [r["ppl"] for r in results if not math.isnan(r["ppl"])]
        if not ppls:
            print(f"{key:<25} {'N/A':>10} {'N/A':>8} {'N/A':>8} {0:>4} {'N/A':>8}")
            continue
        avg = sum(ppls) / len(ppls)
        std = (sum((p - avg) ** 2 for p in ppls) / len(ppls)) ** 0.5
        ratio = avg / baseline_avg
        if "uncorrected" in key:
            # Uncorrected must degrade: PPL ratio > 1.3x baseline
            ok = ratio > 1.3
            if not ok:
                gap2_pass = False
            mark = "PASS" if ok else "FAIL"
        elif "corrected" in key:
            # Corrected must preserve quality: PPL ratio < 1.05x baseline
            ok = ratio < 1.05
            if not ok:
                gap2_pass = False
            mark = "PASS" if ok else "FAIL"
        else:
            mark = "REF"
        print(f"{key:<25} {avg:>10.4f} {ratio:>7.3f}x {std:>8.4f} {len(ppls):>4} {mark:>8}")

    print("-" * 80)
    print(f"Baseline PPL: {baseline_avg:.4f}")
    print(f"Gap 2 overall: {'PASS' if gap2_pass else 'FAIL'}")
    print(f"  Criteria: uncorrected/baseline > 1.3x, corrected/baseline < 1.05x")
    print()

    # Save raw results
    out = "rope_force_delta_results.json"
    with open(out, "w") as f:
        json.dump({
            k: [{"ppl": r["ppl"], "tokens": r.get("completion_tokens", 0),
                 "text": r.get("text", "")[:200]} for r in v]
            for k, v in all_results.items()
        }, f, indent=2)
    print(f"Raw results saved to {out}")


if __name__ == "__main__":
    main()
