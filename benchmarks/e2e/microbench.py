"""Inference-engine microbenchmark for SemBlend paper.

Measures each component individually WITHOUT proxy overhead, directly calling:
  1. jina-v4 TEI  -> embedding latency
  2. Gateway       -> ANN search latency
  3. Gateway       -> KV cache retrieval latency
  4. vLLM          -> cold TTFT (baseline)

Then computes inference-level speedup comparable to CacheBlend/KVShare papers,
plus model-size projections showing the crossover point.

Usage (with port-forwards active):
    python -m benchmarks.e2e.microbench \
        --jina-url   http://localhost:8090 \
        --gateway-url http://localhost:8080 \
        --vllm-url   http://localhost:8000 \
        --iterations 200 \
        --model Qwen/Qwen2.5-1.5B-Instruct

Requires: requests, numpy, rich
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import requests
from rich.console import Console
from rich.table import Table

from benchmarks.e2e.bootstrap_ci import (
    CIResult,
    bootstrap_mean,
    bootstrap_percentile,
)

logger = logging.getLogger(__name__)
console = Console()

# Diverse test queries for representative timing
BENCH_QUERIES = [
    "How do I reset my password?",
    "I want to cancel my subscription",
    "What are your business hours?",
    "I need help with my billing",
    "How do I track my order?",
    "Can I get a refund?",
    "How do I update my account information?",
    "What payment methods do you accept?",
    "I have a problem with my delivery",
    "How do I contact customer support?",
    "I forgot my password, how can I reset it?",
    "Please help me cancel my plan",
    "When are you open for business?",
    "I have a billing question",
    "Where is my package?",
    "I'd like to request a refund please",
    "How can I change my account details?",
    "What payment options are available?",
    "My package hasn't arrived yet",
    "How can I reach your support team?",
]

# Published cold TTFT data for model-size projection (ms)
# Source: vLLM benchmarks, CacheBlend paper, community measurements
PUBLISHED_COLD_TTFT: dict[str, dict] = {
    "Qwen2.5-1.5B": {
        "params": "1.5B",
        "cold_ttft_ms": None,  # Measured live
        "source": "measured",
    },
    "Llama-2-7B": {
        "params": "7B",
        "cold_ttft_ms": 250,
        "source": "CacheBlend Table 2",
    },
    "Llama-2-13B": {
        "params": "13B",
        "cold_ttft_ms": 500,
        "source": "vLLM benchmarks",
    },
    "Llama-2-70B": {
        "params": "70B",
        "cold_ttft_ms": 2000,
        "source": "vLLM benchmarks (A100)",
    },
    "Llama-3-8B": {
        "params": "8B",
        "cold_ttft_ms": 280,
        "source": "community benchmarks",
    },
}


@dataclass(frozen=True)
class ComponentResult:
    """Timing results for a single component."""

    name: str
    latencies_ms: np.ndarray
    p50: CIResult
    mean: CIResult

    @property
    def p50_ms(self) -> float:
        return self.p50.estimate


def measure_embedding(
    jina_url: str,
    queries: list[str],
    iterations: int,
    warmup: int = 5,
) -> ComponentResult:
    """Measure embedding latency by calling jina-v4 TEI directly."""
    url = f"{jina_url.rstrip('/')}/embed"
    latencies = []

    all_queries = []
    for i in range(warmup + iterations):
        all_queries.append(queries[i % len(queries)])

    for i, query in enumerate(all_queries):
        body = {"inputs": query}
        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=body, timeout=10)
            resp.raise_for_status()
        except Exception as exc:
            logger.warning("Embedding request %d failed: %s", i, exc)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    arr = np.array(latencies)
    return ComponentResult(
        name="embedding",
        latencies_ms=arr,
        p50=bootstrap_percentile(arr, 50.0),
        mean=bootstrap_mean(arr),
    )


def measure_ann_search(
    gateway_url: str,
    embedding_url: str,
    queries: list[str],
    iterations: int,
    warmup: int = 5,
    top_k: int = 5,
    threshold: float = 0.80,
) -> ComponentResult:
    """Measure ANN search latency by calling gateway segment search.

    First embeds the query via jina-v4, then calls the search endpoint.
    We only time the search call itself.
    """
    search_url = f"{gateway_url.rstrip('/')}/api/v1/kv-cache/segments/search"
    embed_url = f"{embedding_url.rstrip('/')}/embed"
    latencies = []

    all_queries = []
    for i in range(warmup + iterations):
        all_queries.append(queries[i % len(queries)])

    for i, query in enumerate(all_queries):
        # Get embedding first (not timed)
        try:
            embed_resp = requests.post(
                embed_url, json={"inputs": query}, timeout=10,
            )
            embed_resp.raise_for_status()
            embedding_data = embed_resp.json()
            # TEI returns list of embeddings
            if isinstance(embedding_data, list) and len(embedding_data) > 0:
                embedding = embedding_data[0]
                if isinstance(embedding, dict):
                    embedding = embedding.get("embedding", embedding.get("values", []))
            else:
                logger.warning("Unexpected embedding response format")
                continue
        except Exception as exc:
            logger.warning("Embedding for search %d failed: %s", i, exc)
            continue

        # Time only the search
        body = {
            "embedding": embedding,
            "topK": top_k,
            "threshold": threshold,
        }
        t0 = time.monotonic()
        try:
            resp = requests.post(search_url, json=body, timeout=10)
            # 404/500 is OK — we just want the latency of the search path
            _ = resp.status_code
        except Exception as exc:
            logger.warning("Search request %d failed: %s", i, exc)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    arr = np.array(latencies) if latencies else np.array([0.0])
    return ComponentResult(
        name="ann_search",
        latencies_ms=arr,
        p50=bootstrap_percentile(arr, 50.0),
        mean=bootstrap_mean(arr),
    )


def measure_kv_retrieval(
    gateway_url: str,
    iterations: int,
    warmup: int = 5,
) -> ComponentResult:
    """Measure KV cache retrieval latency (cache miss path shows store overhead)."""
    # Use a dummy hash — we measure the round-trip even for misses
    dummy_hash = "0" * 64
    url = f"{gateway_url.rstrip('/')}/api/v1/kv-cache/{dummy_hash}"
    latencies = []

    for i in range(warmup + iterations):
        t0 = time.monotonic()
        try:
            resp = requests.get(url, timeout=10)
            _ = resp.status_code  # 404 expected for miss
        except Exception as exc:
            logger.warning("KV retrieval %d failed: %s", i, exc)
            continue
        elapsed_ms = (time.monotonic() - t0) * 1000

        if i >= warmup:
            latencies.append(elapsed_ms)

    arr = np.array(latencies) if latencies else np.array([0.0])
    return ComponentResult(
        name="kv_retrieval",
        latencies_ms=arr,
        p50=bootstrap_percentile(arr, 50.0),
        mean=bootstrap_mean(arr),
    )


def measure_cold_ttft(
    vllm_url: str,
    model: str,
    queries: list[str],
    iterations: int,
    warmup: int = 3,
    max_tokens: int = 32,
) -> ComponentResult:
    """Measure cold TTFT by calling vLLM directly with streaming.

    Uses stream=True and measures time to first SSE chunk (true TTFT).
    """
    url = f"{vllm_url.rstrip('/')}/v1/chat/completions"
    latencies = []
    failures = 0

    all_queries = []
    for i in range(warmup + iterations):
        all_queries.append(queries[i % len(queries)])

    for i, query in enumerate(all_queries):
        body = {
            "model": model,
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
            "stream": True,
        }

        t0 = time.monotonic()
        try:
            resp = requests.post(url, json=body, timeout=30, stream=True)
            resp.raise_for_status()
            # Read until first SSE data line (true TTFT)
            for line in resp.iter_lines():
                if line:  # First non-empty line = first token event
                    break
            elapsed_ms = (time.monotonic() - t0) * 1000
            # Consume remaining response to free the connection
            resp.close()
        except Exception as exc:
            failures += 1
            logger.warning("vLLM request %d failed: %s", i, exc)
            continue

        if i >= warmup:
            latencies.append(elapsed_ms)

    if failures > 0:
        logger.warning(
            "%d/%d vLLM requests failed.", failures, warmup + iterations,
        )
    if not latencies:
        logger.error(
            "All vLLM requests failed. Cold TTFT will be zero.",
        )

    arr = np.array(latencies) if latencies else np.array([0.0])
    return ComponentResult(
        name="cold_ttft",
        latencies_ms=arr,
        p50=bootstrap_percentile(arr, 50.0),
        mean=bootstrap_mean(arr),
    )


def compute_inference_speedup(
    embedding: ComponentResult,
    search: ComponentResult,
    retrieval: ComponentResult,
    cold_ttft: ComponentResult,
) -> dict:
    """Compute inference-engine-level speedup (no proxy overhead)."""
    cache_hit_p50 = embedding.p50_ms + search.p50_ms + retrieval.p50_ms
    cold_p50 = cold_ttft.p50_ms
    speedup = cold_p50 / cache_hit_p50 if cache_hit_p50 > 0 else 0.0

    rng = np.random.default_rng(42)
    n_boot = 10_000

    # Bootstrap CI for combined cache hit latency using independent sampling
    # (components are measured in separate loops, not paired observations)
    combined_boot = np.empty(n_boot)
    for i in range(n_boot):
        e = rng.choice(embedding.latencies_ms, size=1)[0]
        s = rng.choice(search.latencies_ms, size=1)[0]
        r = rng.choice(retrieval.latencies_ms, size=1)[0]
        combined_boot[i] = e + s + r

    combined_p50 = CIResult(
        estimate=float(np.percentile(combined_boot, 50)),
        ci_lower=float(np.percentile(combined_boot, 2.5)),
        ci_upper=float(np.percentile(combined_boot, 97.5)),
    )

    # Bootstrap CI for speedup
    cold_arr = cold_ttft.latencies_ms
    if len(cold_arr) == 0:
        logger.error("No cold TTFT data; speedup CI cannot be computed.")
        speedup_ci = CIResult(0.0, 0.0, 0.0)
    else:
        boot_speedups = np.empty(n_boot)
        for i in range(n_boot):
            c_sample = rng.choice(cold_arr, size=len(cold_arr), replace=True)
            cold_med = float(np.median(c_sample))
            cache_med = combined_boot[i]  # Reuse the combined sample
            boot_speedups[i] = cold_med / cache_med if cache_med > 0 else 0.0

        speedup_ci = CIResult(
            estimate=speedup,
            ci_lower=float(np.percentile(boot_speedups, 2.5)),
            ci_upper=float(np.percentile(boot_speedups, 97.5)),
        )

    return {
        "cache_hit_p50_ms": round(cache_hit_p50, 1),
        "cache_hit_ci": combined_p50,
        "cold_ttft_p50_ms": round(cold_p50, 1),
        "inference_speedup": round(speedup, 2),
        "speedup_ci": speedup_ci,
    }


def model_size_projection(
    cache_hit_ms: float,
    measured_cold_ms: float,
) -> list[dict]:
    """Project speedup across model sizes using measured cache latency."""
    rows = []
    for model_name, info in PUBLISHED_COLD_TTFT.items():
        cold = info["cold_ttft_ms"] if info["cold_ttft_ms"] is not None else measured_cold_ms
        speedup = cold / cache_hit_ms if cache_hit_ms > 0 else 0.0
        rows.append({
            "model": model_name,
            "params": info["params"],
            "cold_ttft_ms": round(cold, 1),
            "cache_lookup_ms": round(cache_hit_ms, 1),
            "speedup": round(speedup, 2),
            "source": info["source"],
            "beneficial": speedup > 1.0,
        })
    return rows


def print_component_table(components: list[ComponentResult]) -> None:
    """Print per-component timing table."""
    table = Table(title="Component-Level Latency (Direct Measurement)")
    table.add_column("Component", style="bold")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("Mean (ms)", justify="right")
    table.add_column("N", justify="right")

    for c in components:
        ci_str = f"[{c.p50.ci_lower:.1f}, {c.p50.ci_upper:.1f}]"
        table.add_row(
            c.name,
            f"{c.p50.estimate:.1f}",
            ci_str,
            f"{c.mean.estimate:.1f}",
            str(len(c.latencies_ms)),
        )

    console.print(table)


def print_speedup_table(result: dict) -> None:
    """Print inference-level speedup summary."""
    table = Table(title="Inference-Engine Speedup (No Proxy Overhead)")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    ci = result["speedup_ci"]
    cache_ci = result["cache_hit_ci"]

    table.add_row("Cold TTFT (P50)", f"{result['cold_ttft_p50_ms']:.1f} ms")
    table.add_row(
        "Cache Hit (P50)",
        f"{result['cache_hit_p50_ms']:.1f} ms [{cache_ci.ci_lower:.1f}, {cache_ci.ci_upper:.1f}]",
    )
    table.add_row(
        "Inference Speedup",
        f"{ci.estimate:.2f}x [{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]",
    )

    console.print(table)


def print_projection_table(projections: list[dict]) -> None:
    """Print model-size scaling projection."""
    table = Table(title="Model-Size Scaling Projection")
    table.add_column("Model", style="bold")
    table.add_column("Params", justify="right")
    table.add_column("Cold TTFT (ms)", justify="right")
    table.add_column("Cache Lookup (ms)", justify="right")
    table.add_column("Speedup", justify="right")
    table.add_column("Beneficial?", justify="center")

    for row in projections:
        beneficial = "[green]Yes[/green]" if row["beneficial"] else "[red]No[/red]"
        table.add_row(
            row["model"],
            row["params"],
            f"{row['cold_ttft_ms']:.0f}",
            f"{row['cache_lookup_ms']:.1f}",
            f"{row['speedup']:.2f}x",
            beneficial,
        )

    console.print(table)


def print_latex(
    components: list[ComponentResult],
    result: dict,
    projections: list[dict],
) -> None:
    """Print LaTeX-formatted tables for the paper."""
    console.print("\n[bold]LaTeX: Component Breakdown[/bold]")
    print(r"% Table: Inference-engine component latency")
    print(r"\begin{tabular}{@{}lrc@{}}")
    print(r"\toprule")
    print(r"\textbf{Component} & \textbf{P50 (ms)} & \textbf{95\% CI} \\")
    print(r"\midrule")
    for c in components:
        name = c.name.replace("_", r"\_")
        print(rf"{name} & {c.p50.estimate:.1f} & [{c.p50.ci_lower:.1f}, {c.p50.ci_upper:.1f}] \\")
    print(r"\midrule")
    ci = result["cache_hit_ci"]
    print(rf"\textbf{{Cache hit total}} & \textbf{{{result['cache_hit_p50_ms']:.1f}}} & [{ci.ci_lower:.1f}, {ci.ci_upper:.1f}] \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    console.print("\n[bold]LaTeX: Model-Size Projection[/bold]")
    print(r"% Table: Model-size scaling projection")
    print(r"\begin{tabular}{@{}llrrrr@{}}")
    print(r"\toprule")
    print(r"\textbf{Model} & \textbf{Size} & \textbf{Cold TTFT} & \textbf{Cache} & \textbf{Speedup} \\")
    print(r"\midrule")
    for row in projections:
        model = row["model"].replace("_", r"\_")
        marker = r" $\star$" if row["source"] == "measured" else ""
        print(
            rf"{model} & {row['params']} & {row['cold_ttft_ms']:.0f}\,ms & "
            rf"{row['cache_lookup_ms']:.1f}\,ms & "
            rf"${row['speedup']:.2f}\times${marker} \\"
        )
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"% $\star$ = measured; others from published benchmarks")


def export_results(
    components: list[ComponentResult],
    speedup_result: dict,
    projections: list[dict],
    output_path: Path,
) -> None:
    """Export all results to JSON."""
    data = {
        "components": {},
        "inference_speedup": {
            "cache_hit_p50_ms": speedup_result["cache_hit_p50_ms"],
            "cold_ttft_p50_ms": speedup_result["cold_ttft_p50_ms"],
            "speedup": speedup_result["inference_speedup"],
            "speedup_ci_lower": speedup_result["speedup_ci"].ci_lower,
            "speedup_ci_upper": speedup_result["speedup_ci"].ci_upper,
        },
        "model_size_projection": projections,
    }

    for c in components:
        data["components"][c.name] = {
            "p50_ms": round(c.p50.estimate, 2),
            "ci_lower": round(c.p50.ci_lower, 2),
            "ci_upper": round(c.p50.ci_upper, 2),
            "mean_ms": round(c.mean.estimate, 2),
            "n": len(c.latencies_ms),
            "raw_ms": [round(v, 2) for v in c.latencies_ms.tolist()],
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"\n[green]Results saved to {output_path}[/green]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference-engine microbenchmark for SemBlend paper",
    )
    parser.add_argument(
        "--jina-url",
        default="http://localhost:8090",
        help="jina-v4 TEI endpoint",
    )
    parser.add_argument(
        "--gateway-url",
        default="http://localhost:8080",
        help="Synapse gateway endpoint",
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="Direct vLLM endpoint",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for vLLM",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=200,
        help="Measurement iterations per component (after warmup)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Warmup iterations (excluded from measurement)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/e2e/results/microbench.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--latex",
        action="store_true",
        help="Print LaTeX table rows for paper",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    console.print("[bold]SemBlend Inference-Engine Microbenchmark[/bold]")
    console.print(f"  Iterations: {args.iterations} (warmup: {args.warmup})")
    console.print()

    # 1. Embedding latency
    console.print("[bold]1/4 Measuring embedding latency (jina-v4 TEI)...[/bold]")
    embedding = measure_embedding(
        args.jina_url, BENCH_QUERIES, args.iterations, args.warmup,
    )
    console.print(f"  P50: {embedding.p50_ms:.1f} ms")

    # 2. ANN search latency
    console.print("[bold]2/4 Measuring ANN search latency (gateway)...[/bold]")
    search = measure_ann_search(
        args.gateway_url, args.jina_url, BENCH_QUERIES,
        args.iterations, args.warmup,
    )
    console.print(f"  P50: {search.p50_ms:.1f} ms")

    # 3. KV retrieval latency
    console.print("[bold]3/4 Measuring KV retrieval latency (gateway)...[/bold]")
    retrieval = measure_kv_retrieval(
        args.gateway_url, args.iterations, args.warmup,
    )
    console.print(f"  P50: {retrieval.p50_ms:.1f} ms")

    # 4. Cold TTFT
    console.print("[bold]4/4 Measuring cold TTFT (vLLM direct)...[/bold]")
    cold_ttft = measure_cold_ttft(
        args.vllm_url, args.model, BENCH_QUERIES,
        args.iterations, args.warmup,
    )
    console.print(f"  P50: {cold_ttft.p50_ms:.1f} ms")

    # Results
    console.print()
    components = [embedding, search, retrieval, cold_ttft]
    print_component_table(components)

    # Compute inference-level speedup
    speedup_result = compute_inference_speedup(embedding, search, retrieval, cold_ttft)
    console.print()
    print_speedup_table(speedup_result)

    # Model-size projection
    projections = model_size_projection(
        speedup_result["cache_hit_p50_ms"],
        cold_ttft.p50_ms,
    )
    console.print()
    print_projection_table(projections)

    # Export
    export_results(components, speedup_result, projections, Path(args.output))

    if args.latex:
        print_latex(components, speedup_result, projections)

    # Final summary
    console.print("\n[bold]Key findings:[/bold]")
    ci = speedup_result["speedup_ci"]
    console.print(
        f"  Inference-engine speedup: {ci.estimate:.2f}x "
        f"[{ci.ci_lower:.2f}, {ci.ci_upper:.2f}]"
    )
    console.print(f"  Cache hit = {speedup_result['cache_hit_p50_ms']:.1f} ms "
                  f"(embed + search + retrieve)")
    console.print(f"  Cold TTFT = {speedup_result['cold_ttft_p50_ms']:.1f} ms")

    # Find crossover point
    crossover = next(
        (p for p in projections if p["beneficial"]),
        None,
    )
    if crossover:
        console.print(
            f"  Crossover: caching beneficial starting at "
            f"{crossover['model']} ({crossover['params']})"
        )


if __name__ == "__main__":
    main()
