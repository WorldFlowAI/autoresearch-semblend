#!/usr/bin/env python3
"""SemBlend GPU memory savings benchmark — measures KV cache block reduction.

Polls vLLM /metrics at 100ms intervals DURING active inference to capture
peak GPU KV cache usage (vllm:gpu_cache_usage_perc reads 0 between requests).

Phases: BASELINE (cold) -> DONOR (register) -> SEMBLEND (reorder) -> COLD

Usage:
    python3 semblend_gpu_memory_bench.py \
        --endpoint http://localhost:8000 \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --target-tokens 2048,4096,8192
"""
from __future__ import annotations

import argparse, json, os, re, sys, threading, time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from statistics import mean

try:
    import requests
except ImportError:
    sys.exit("ERROR: pip install requests")

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

# -- RAG chunks for reorder variants --
CHUNKS = [
    "SEMICONDUCTOR MARKET Q3 2025: Global revenue $178.2B (+23.4% YoY). AI accelerators "
    "$42.8B (+67%). NVIDIA 82% GPU share, AMD MI300X 14%. TAM projected $120B by 2027. "
    "Memory $48.3B, HBM3E undersupplied. SK Hynix HBM +340%. DDR5 45% server penetration. "
    "Inference growing 85% CAGR vs 45% training. MoE and speculative decoding 3x efficiency "
    "but offset by production LLM volume explosion. Micron AI memory 35% of revenue.",

    "FOUNDRY AND PACKAGING: TSMC record $23.5B revenue. 5nm-and-below 69% of wafer sales. "
    "Arizona Fab 2 accelerated for 3nm by Q4 2026. 90% of 2nm pre-committed through 2027. "
    "Samsung 3nm GAA yields 55% vs TSMC N3E 78%. Intel 18A test chips working. CoWoS main "
    "bottleneck, capacity +2.5x by mid-2026 but demand may triple. UCIe chiplets gaining "
    "with 15 new consortium members. GlobalFoundries FDX for automotive and IoT.",

    "AUTOMOTIVE AND EDGE AI: Chip shortage resolved, lead times 12-16 weeks. Silicon per "
    "vehicle $712->$834. L3 autonomy from Mobileye, NVIDIA DRIVE, Qualcomm entering mass "
    "production, 200-500 TOPS each. Auto chip market $78B by 2027. Snapdragon X Elite 35 "
    "OEM wins. AI PC revenue $1.2B. Edge AI accelerating for Qualcomm, MediaTek, startups. "
    "Broadcom XPU tripled with Google and Meta. Total AI revenue $4.1B.",

    "GEOPOLITICS AND SUPPLY CHAIN: US-China restrictions reshaping chains. China $47B in "
    "new fabs at 28nm+. CHIPS Act $32B to 15 projects. Rapidus 2nm tape-out. EU Chips Act "
    "EUR 12B. ASML 40% decline in China EUV bookings. NVIDIA B200 Ultra 2x H100 throughput. "
    "Data center revenue $18.4B (+78%). AMD MI300X at Azure/Oracle, $3.9B (+112%). Intel "
    "Foundry $4.2B defense contract. Software moats CUDA/ROCm increasingly decisive.",
]

COLD_CHUNK = (
    "CLIMATE H2 2025: Temperature +1.48C above pre-industrial. CO2 427ppm. Solar 580GW "
    "(+35%). Wind 145GW. Battery storage 120GWh. Solar LCOE $23/MWh. LFP $92/kWh. EV "
    "sales 22M units (25% of new cars). BYD overtook VW. EU ETS 85 EUR/tonne. Climate "
    "finance $1.8T. Perovskite-silicon tandem 33.9% lab efficiency. Enhanced geothermal "
    "25MW commercial in Nevada. Direct air capture 10kt/yr across 5 plants."
)

QUESTION = "Summarize the key trends and market dynamics described in this report."


def build_prompt(chunks, question, target_tokens, ref_id, tokenizer=None):
    """Build prompt near target_tokens. ref_id defeats prefix cache."""
    header = f"You are a senior analyst. Reference: {ref_id}\nUse ONLY the context below.\n\n"
    suffix = f"\n\nBased on the above, answer:\n{question}"
    context = "\n\n".join(chunks)
    available = target_tokens * 4 - len(header) - len(suffix)
    while len(context) < available:
        context += "\n\n--- CONTINUED ---\n\n" + "\n\n".join(chunks)
    prompt = header + context[:available] + suffix
    if tokenizer is not None:
        tids = tokenizer.encode(prompt, add_special_tokens=False)
        if len(tids) > target_tokens:
            prompt = tokenizer.decode(tids[:target_tokens], skip_special_tokens=True) + suffix
    return prompt


# -- Metrics polling --
_PATTERNS = {
    "gpu_cache": re.compile(r'^vllm:gpu_cache_usage_perc\s+([\d.eE+-]+)', re.M),
    "kv_cache": re.compile(r'^vllm:kv_cache_usage_perc\s+([\d.eE+-]+)', re.M),
    "running": re.compile(r'^vllm:num_requests_running\s+([\d.eE+-]+)', re.M),
    "preempt": re.compile(r'^vllm:num_preemptions_total\s+([\d.eE+-]+)', re.M),
    "prefix_hit": re.compile(r'^vllm:gpu_prefix_cache_hit_rate\s+([\d.eE+-]+)', re.M),
}

@dataclass
class Sample:
    ts: float
    gpu_cache: float = 0.0
    kv_cache: float = 0.0
    running: float = 0.0
    preempt: float = 0.0
    prefix_hit: float = 0.0


def scrape(endpoint):
    try:
        text = requests.get(f"{endpoint}/metrics", timeout=2).text
    except Exception:
        return Sample(ts=time.time())
    s = Sample(ts=time.time())
    for key, pat in _PATTERNS.items():
        m = pat.search(text)
        if m:
            setattr(s, key, float(m.group(1)))
    return s


class Poller:
    def __init__(self, endpoint, interval=0.1):
        self._ep, self._iv = endpoint, interval
        self._samples, self._stop = [], threading.Event()
        self._t = None

    def start(self):
        self._samples.clear(); self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True); self._t.start()

    def stop(self):
        self._stop.set()
        if self._t: self._t.join(timeout=3)
        return list(self._samples)

    def _run(self):
        while not self._stop.is_set():
            self._samples.append(scrape(self._ep)); self._stop.wait(self._iv)


# -- Request with metrics --
@dataclass
class Result:
    phase: str; label: str; ttft_ms: float; total_ms: float; prompt_tokens: int
    peak_gpu: float; peak_kv: float; mean_gpu: float; mean_kv: float
    n_samples: int; peak_running: float; preempt_delta: float
    prefix_hit: float; error: str | None = None


def _err_result(phase, label, samples, err):
    return Result(phase, label, -1, -1, 0, 0, 0, 0, 0, len(samples), 0, 0, 0, err)


def send_with_metrics(endpoint, model, prompt, phase, label, max_tokens=20):
    poller = Poller(endpoint)
    poller.start()
    body = {"model": model, "prompt": prompt, "max_tokens": max_tokens,
            "temperature": 0.1, "stream": True}
    t0 = time.perf_counter()
    ttft = None; ptok = 0
    try:
        resp = requests.post(f"{endpoint}/v1/completions", json=body, stream=True, timeout=180)
        resp.raise_for_status()
        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "): continue
            p = line[6:]
            if p == "[DONE]": break
            try:
                c = json.loads(p)
                if c.get("choices", [{}])[0].get("text") and ttft is None:
                    ttft = (time.perf_counter() - t0) * 1000
                u = c.get("usage")
                if u and "prompt_tokens" in u: ptok = u["prompt_tokens"]
            except (json.JSONDecodeError, KeyError, IndexError):
                pass
        total = (time.perf_counter() - t0) * 1000
    except Exception as e:
        return _err_result(phase, label, poller.stop(), str(e)[:200])

    ss = poller.stop()
    gv = [s.gpu_cache for s in ss if s.gpu_cache > 0]
    kv = [s.kv_cache for s in ss if s.kv_cache > 0]
    ag = [s.gpu_cache for s in ss]; ak = [s.kv_cache for s in ss]
    pg = max(gv) if gv else (max(ag) if ag else 0)
    pk = max(kv) if kv else (max(ak) if ak else 0)
    pv = [s.preempt for s in ss]
    return Result(
        phase, label, ttft if ttft is not None else total, total, ptok,
        pg, pk, mean(gv) if gv else 0, mean(kv) if kv else 0, len(ss),
        max((s.running for s in ss), default=0),
        (pv[-1] - pv[0]) if len(pv) >= 2 else 0,
        max((s.prefix_hit for s in ss), default=0),
    )


# -- Main --
def run_benchmark(args):
    endpoint, model = args.endpoint, args.model
    num_ctx = args.num_contexts
    tok_list = [int(t) for t in args.target_tokens.split(",")]

    tokenizer = None
    if AutoTokenizer is not None:
        try:
            tokenizer = AutoTokenizer.from_pretrained(model)
            print(f"Loaded tokenizer: {model}")
        except Exception as e:
            print(f"Tokenizer unavailable ({e}), using char estimate")

    try:
        r = requests.get(f"{endpoint}/health", timeout=5)
        if r.status_code != 200: sys.exit(f"vLLM unhealthy: {r.status_code}")
        print(f"vLLM healthy at {endpoint}")
    except Exception as e:
        sys.exit(f"Cannot reach vLLM: {e}")

    all_results = {}
    for tgt in tok_list:
        print(f"\n{'='*70}\nTARGET TOKENS: {tgt}\n{'='*70}")
        pr = {"baseline": [], "donor_reg": [], "semblend": [], "cold": []}

        ctx_sets = [[CHUNKS[(i + j) % len(CHUNKS)] for j in range(4)] for i in range(num_ctx)]

        for phase, desc, get_chunks, wait in [
            ("baseline", "cold prefill", lambda i, c: c, 0),
            ("donor_reg", "donor registration", lambda i, c: c, 3),
            ("semblend", "reordered chunks", lambda i, c: [c[2], c[0], c[3], c[1]], 0),
        ]:
            print(f"\n--- {phase.upper()} ({num_ctx} prompts, {desc}) ---")
            for i, cs in enumerate(ctx_sets):
                chunks = get_chunks(i, cs)
                ref = f"{phase}-{tgt}-{i}"
                prompt = build_prompt(chunks, QUESTION, tgt, ref, tokenizer)
                res = send_with_metrics(endpoint, model, prompt, phase, ref)
                pr[phase].append(res)
                print(f"  {ref}: TTFT={res.ttft_ms:.0f}ms peak_gpu={res.peak_gpu:.4f} "
                      f"peak_kv={res.peak_kv:.4f} samples={res.n_samples}")
                time.sleep(0.5)
            if wait:
                print(f"  Waiting {wait}s for cache propagation...")
                time.sleep(wait)

        print(f"\n--- COLD ({num_ctx} prompts, unrelated) ---")
        for i in range(num_ctx):
            ref = f"cold-{tgt}-{i}"
            prompt = build_prompt([COLD_CHUNK], QUESTION, tgt, ref, tokenizer)
            res = send_with_metrics(endpoint, model, prompt, "cold", ref)
            pr["cold"].append(res)
            print(f"  {ref}: TTFT={res.ttft_ms:.0f}ms peak_gpu={res.peak_gpu:.4f}")
            time.sleep(0.5)

        all_results[tgt] = pr

    # -- Summary table --
    run_id = f"semblend-gpu-mem-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
    rows = []
    print(f"\n{'='*90}\nSemBlend GPU Memory Savings\n{'='*90}")
    print(f"  Model: {model}  |  Contexts: {num_ctx}")
    hdr = f"  {'Tok':>6} {'Phase':<12} {'TTFT P50':>9} {'PeakGPU%':>9} {'MeanGPU%':>9} {'PeakKV%':>9} {'Preempt':>8}"
    print(f"\n{hdr}\n  {'-'*72}")

    for tgt in tok_list:
        for pn in ["baseline", "semblend", "cold"]:
            rl = all_results[tgt][pn]
            vt = sorted(r.ttft_ms for r in rl if r.ttft_ms > 0)
            p50 = vt[len(vt)//2] if vt else 0
            gp = [r.peak_gpu for r in rl]; gm = [r.mean_gpu for r in rl if r.mean_gpu > 0]
            kp = [r.peak_kv for r in rl]; pe = sum(r.preempt_delta for r in rl)
            row = {"tokens": tgt, "phase": pn, "ttft_p50": p50,
                   "peak_gpu": max(gp, default=0), "mean_gpu": mean(gm) if gm else 0,
                   "peak_kv": max(kp, default=0), "preemptions": pe}
            rows.append(row)
            print(f"  {tgt:>6} {pn:<12} {p50:>8.0f}ms {row['peak_gpu']:>8.4f} "
                  f"{row['mean_gpu']:>8.4f} {row['peak_kv']:>8.4f} {pe:>7.0f}")
        print(f"  {'-'*72}")

    print(f"\n  Memory savings (baseline vs semblend peak GPU cache %):")
    for tgt in tok_list:
        bl = max((r.peak_gpu for r in all_results[tgt]["baseline"]), default=0)
        sb = max((r.peak_gpu for r in all_results[tgt]["semblend"]), default=0)
        if bl > 0:
            print(f"    {tgt:>6}tok: baseline={bl:.4f} semblend={sb:.4f} reduction={((1-sb/bl)*100):+.1f}%")
        else:
            print(f"    {tgt:>6}tok: no GPU cache data captured")
    print(f"{'='*90}")

    os.makedirs(args.output_dir, exist_ok=True)
    out = {"run_id": run_id, "model": model, "endpoint": endpoint,
           "num_contexts": num_ctx, "target_tokens": tok_list,
           "started_at": datetime.now(timezone.utc).isoformat(),
           "summary": rows,
           "phases": {str(t): {p: [asdict(r) for r in rl] for p, rl in ph.items()}
                      for t, ph in all_results.items()}}
    fp = os.path.join(args.output_dir, f"{run_id}.json")
    with open(fp, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {fp}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SemBlend GPU memory savings benchmark")
    p.add_argument("--endpoint", default="http://localhost:8000")
    p.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ")
    p.add_argument("--num-contexts", type=int, default=3)
    p.add_argument("--target-tokens", default="2048,4096,8192",
                   help="Comma-separated token lengths")
    p.add_argument("--output-dir", default="benchmarks/e2e/results")
    run_benchmark(p.parse_args())
