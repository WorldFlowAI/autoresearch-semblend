"""CLI entry point for the SemBlend E2E benchmark.

Supports three modes:
  response-cache  — Response-level semantic cache (default, original mode)
  kv-cache        — KV-tensor injection via LMCache + vLLM
  full            — Run both modes + component measurements, produce
                    side-by-side comparison table
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import uuid

from rich.console import Console
from rich.table import Table

from benchmarks.e2e.config import BenchmarkConfig
from benchmarks.e2e.runner import BenchmarkRunner

console = Console()


def _common_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared across all subcommands."""
    parser.add_argument(
        "--endpoint",
        default="http://localhost:8080",
        help="Synapse proxy endpoint (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--gateway-endpoint",
        default="http://localhost:8080",
        help="Synapse gateway endpoint (default: http://localhost:8080)",
    )
    parser.add_argument(
        "--vllm-endpoint",
        default="http://localhost:8000",
        help="Direct vLLM endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for chat completions",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["bitext"],
        choices=["sharegpt", "multinews", "bitext"],
        help="Datasets to benchmark (default: bitext)",
    )
    parser.add_argument(
        "--bitext-size", type=int, default=921,
        help="Number of Bitext queries (default: 921 = holdout set)",
    )
    parser.add_argument(
        "--sharegpt-size", type=int, default=300,
        help="Number of ShareGPT queries",
    )
    parser.add_argument(
        "--multinews-size", type=int, default=180,
        help="Number of MultiNews queries",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256,
        help="Max tokens per response",
    )
    parser.add_argument(
        "--concurrency", type=int, default=4,
        help="Concurrent requests",
    )
    parser.add_argument(
        "--warmup-batch-size", type=int, default=20,
        help="Warmup seeds sent per batch",
    )
    parser.add_argument(
        "--output-dir", default="results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-id", default=None,
        help="Unique run identifier (auto-generated if not provided)",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Enable debug logging",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SemBlend E2E Benchmark — measure Synapse semantic KV cache "
            "performance across response-level and KV-tensor caching"
        ),
    )

    subparsers = parser.add_subparsers(dest="mode", help="Benchmark mode")

    # response-cache (default / original mode)
    rc = subparsers.add_parser(
        "response-cache",
        help="Response-level semantic cache benchmark",
    )
    _common_args(rc)

    # kv-cache mode
    kv = subparsers.add_parser(
        "kv-cache",
        help="KV-tensor injection benchmark (via LMCache + vLLM)",
    )
    _common_args(kv)

    # component mode
    comp = subparsers.add_parser(
        "component",
        help="Component-level direct measurement",
    )
    _common_args(comp)
    comp.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples per component measurement",
    )

    # full comparison mode
    full = subparsers.add_parser(
        "full",
        help="Run all modes and produce comparison table",
    )
    _common_args(full)
    full.add_argument(
        "--n-samples", type=int, default=200,
        help="Number of samples for component measurements",
    )

    args = parser.parse_args()

    # Default to response-cache if no mode specified
    if args.mode is None:
        _common_args(parser)
        args = parser.parse_args()
        args.mode = "response-cache"

    return args


def _make_config(args: argparse.Namespace) -> BenchmarkConfig:
    return BenchmarkConfig(
        synapse_endpoint=args.endpoint,
        vllm_endpoint=args.vllm_endpoint,
        model=args.model,
        max_tokens=args.max_tokens,
        sharegpt_size=args.sharegpt_size,
        multinews_size=args.multinews_size,
        bitext_size=args.bitext_size,
        concurrency=args.concurrency,
        warmup_batch_size=args.warmup_batch_size,
        output_dir=args.output_dir,
        datasets=args.datasets,
    )


def print_summary(results_path: str) -> None:
    """Print a formatted summary table from a results JSON file."""
    with open(results_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})

    # Dataset table
    table = Table(title="Dataset Results")
    table.add_column("Dataset", style="bold")
    table.add_column("Queries", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("Avg BLEU", justify="right")
    table.add_column("Avg ROUGE-L", justify="right")

    for name, ds in summary.get("datasets", {}).items():
        hit_pct = f"{100*ds['hit_rate']:.1f}%"
        table.add_row(
            name,
            str(ds["total_queries"]),
            hit_pct,
            f"{ds['avg_bleu']:.3f}",
            f"{ds['avg_rouge_l']:.3f}",
        )

    console.print(table)

    # Latency table
    latency = summary.get("latency", {})
    lat_table = Table(title="TTFT Latency")
    lat_table.add_column("Tier", style="bold")
    lat_table.add_column("Count", justify="right")
    lat_table.add_column("P50 (ms)", justify="right")
    lat_table.add_column("P95 (ms)", justify="right")
    lat_table.add_column("P99 (ms)", justify="right")
    lat_table.add_column("Speedup", justify="right")

    baseline = latency.get("baseline", {})
    if baseline.get("count", 0) > 0:
        lat_table.add_row(
            "baseline",
            str(baseline["count"]),
            f"{baseline['p50_ms']:.1f}",
            "---",
            f"{baseline['p99_ms']:.1f}",
            "1.0x",
        )

    speedups = latency.get("speedups", {})
    for tier_name, tier_data in latency.get("tiers", {}).items():
        sp = speedups.get(tier_name, 0)
        lat_table.add_row(
            tier_name,
            str(int(tier_data["count"])),
            f"{tier_data['p50_ms']:.1f}",
            f"{tier_data['p95_ms']:.1f}",
            f"{tier_data['p99_ms']:.1f}",
            f"{sp:.1f}x",
        )

    console.print(lat_table)


def print_kv_summary(summary: dict) -> None:
    """Print KV-cache benchmark summary."""
    table = Table(title="KV-Cache Benchmark Results")
    table.add_column("Category", style="bold")
    table.add_column("Count", justify="right")
    table.add_column("P50 (ms)", justify="right")
    table.add_column("P95 (ms)", justify="right")
    table.add_column("Speedup", justify="right")

    baseline = summary.get("baseline", {})
    if baseline.get("n", 0) > 0:
        table.add_row(
            "Baseline (cold)",
            str(baseline["n"]),
            f"{baseline['p50']:.1f}",
            f"{baseline['p95']:.1f}",
            "1.0x",
        )

    for label, key, speedup_key in [
        ("Exact KV hit", "exact", "speedup_exact"),
        ("Semantic KV hit", "semantic", "speedup_semantic"),
        ("Miss", "miss", None),
    ]:
        data = summary.get(key, {})
        if data.get("n", 0) > 0:
            sp = f"{summary.get(speedup_key, 0):.1f}x" if speedup_key else "---"
            table.add_row(
                label,
                str(data["n"]),
                f"{data['p50']:.1f}",
                f"{data['p95']:.1f}",
                sp,
            )

    console.print(table)

    avg_comp = summary.get("avg_computation_ratio", 1.0)
    console.print(
        f"  Avg computation ratio (semantic): {avg_comp:.2f} "
        f"({100*(1-avg_comp):.0f}% FLOPs saved)"
    )


def print_comparison(
    response_path: str | None,
    kv_summary: dict | None,
    component_path: str | None,
) -> None:
    """Print side-by-side comparison table for the paper."""
    table = Table(title="SemBlend Full Comparison")
    table.add_column("Metric", style="bold")
    table.add_column("Baseline", justify="right")
    table.add_column("Response Cache", justify="right")
    table.add_column("KV Cache", justify="right")

    # Load response-cache results
    rc_baseline_p50 = "---"
    rc_hit_p50 = "---"
    rc_speedup = "---"
    rc_hit_rate = "---"

    if response_path and os.path.exists(response_path):
        with open(response_path) as f:
            rc_data = json.load(f)
        rc_summary = rc_data.get("summary", {})
        rc_latency = rc_summary.get("latency", {})
        rc_bl = rc_latency.get("baseline", {})
        if rc_bl.get("count", 0) > 0:
            rc_baseline_p50 = f"{rc_bl['p50_ms']:.1f}"

        # Use first tier available
        tiers = rc_latency.get("tiers", {})
        speedups = rc_latency.get("speedups", {})
        for tier_name, tier_data in tiers.items():
            if tier_data.get("count", 0) > 0:
                rc_hit_p50 = f"{tier_data['p50_ms']:.1f}"
                sp = speedups.get(tier_name, 0)
                rc_speedup = f"{sp:.1f}x"
                break

        ds_summary = rc_summary.get("datasets", {})
        total_q = sum(d.get("total_queries", 0) for d in ds_summary.values())
        total_hits = sum(
            d.get("total_queries", 0) * d.get("hit_rate", 0)
            for d in ds_summary.values()
        )
        if total_q > 0:
            rc_hit_rate = f"{100*total_hits/total_q:.1f}%"

    # KV-cache results
    kv_baseline_p50 = "---"
    kv_hit_p50 = "---"
    kv_speedup = "---"
    kv_hit_rate = "---"

    if kv_summary:
        kv_bl = kv_summary.get("baseline", {})
        if kv_bl.get("n", 0) > 0:
            kv_baseline_p50 = f"{kv_bl['p50']:.1f}"

        # Prefer semantic hit stats (the paper's core contribution)
        sem = kv_summary.get("semantic", {})
        if sem.get("n", 0) > 0:
            kv_hit_p50 = f"{sem['p50']:.1f}"
            kv_speedup = f"{kv_summary.get('speedup_semantic', 0):.1f}x"

        total = kv_summary.get("total_test_queries", 0)
        hits = kv_summary.get("exact_hits", 0) + kv_summary.get(
            "semantic_hits", 0
        )
        if total > 0:
            kv_hit_rate = f"{100*hits/total:.1f}%"

    baseline_p50 = kv_baseline_p50 if kv_baseline_p50 != "---" else rc_baseline_p50

    table.add_row("TTFT P50 (ms)", baseline_p50, rc_hit_p50, kv_hit_p50)
    table.add_row("Speedup", "1.0x", rc_speedup, kv_speedup)
    table.add_row("Hit Rate", "---", rc_hit_rate, kv_hit_rate)

    if kv_summary:
        comp_ratio = kv_summary.get("avg_computation_ratio", 1.0)
        table.add_row(
            "Computation Ratio",
            "1.0",
            "0.0 (full skip)",
            f"{comp_ratio:.2f}",
        )

    console.print(table)

    # Component measurements
    if component_path and os.path.exists(component_path):
        with open(component_path) as f:
            comp_data = json.load(f)

        comp_table = Table(title="Component Latency Breakdown")
        comp_table.add_column("Component", style="bold")
        comp_table.add_column("P50 (ms)", justify="right")
        comp_table.add_column("P95 (ms)", justify="right")
        comp_table.add_column("P99 (ms)", justify="right")

        for name, key in [
            ("ONNX Embedding", "onnx_embedding"),
            ("CAGRA Search", "cagra_search"),
            ("KV Save", "kv_save"),
            ("KV Load", "kv_load"),
        ]:
            data = comp_data.get(key, {})
            if data.get("n", 0) > 0:
                comp_table.add_row(
                    name,
                    f"{data['p50_ms']:.1f}",
                    f"{data['p95_ms']:.1f}",
                    f"{data['p99_ms']:.1f}",
                )

        console.print(comp_table)

        # Cold TTFT by prompt length
        cold_ttft = comp_data.get("cold_ttft", {})
        if cold_ttft:
            ttft_table = Table(title="Cold TTFT by Prompt Length")
            ttft_table.add_column("Tokens", justify="right", style="bold")
            ttft_table.add_column("P50 (ms)", justify="right")
            ttft_table.add_column("P95 (ms)", justify="right")
            ttft_table.add_column("N", justify="right")

            for tok_len in sorted(cold_ttft.keys(), key=int):
                data = cold_ttft[tok_len]
                if data.get("n", 0) > 0:
                    ttft_table.add_row(
                        tok_len,
                        f"{data['p50_ms']:.1f}",
                        f"{data['p95_ms']:.1f}",
                        str(data["n"]),
                    )

            console.print(ttft_table)


def cmd_response_cache(args: argparse.Namespace) -> None:
    """Run response-level cache benchmark."""
    config = _make_config(args)
    run_id = args.run_id or f"bench-{uuid.uuid4().hex[:8]}"
    runner = BenchmarkRunner(config=config, run_id=run_id)
    results = asyncio.run(runner.run())
    path = results.save(config.output_dir)
    console.print(f"\n[green]Results saved to {path}[/green]")
    print_summary(path)


def cmd_kv_cache(args: argparse.Namespace) -> None:
    """Run KV-tensor injection benchmark."""
    from benchmarks.e2e.kv_bench import KvCacheBenchmark

    config = _make_config(args)
    run_id = args.run_id or f"kv-bench-{uuid.uuid4().hex[:8]}"
    bench = KvCacheBenchmark(config=config, run_id=run_id)

    all_summaries = {}
    for ds_name in config.datasets:
        kv_results = asyncio.run(bench.run(ds_name))
        summary = kv_results.summary()
        all_summaries[ds_name] = summary
        print_kv_summary(summary)

    # Save combined results
    os.makedirs(config.output_dir, exist_ok=True)
    out_path = os.path.join(config.output_dir, f"{run_id}.json")
    with open(out_path, "w") as f:
        json.dump({"mode": "kv-cache", "datasets": all_summaries}, f, indent=2)
    console.print(f"\n[green]KV results saved to {out_path}[/green]")


def cmd_component(args: argparse.Namespace) -> None:
    """Run component-level direct measurement."""
    from benchmarks.e2e.component_bench import run_component_benchmark

    run_id = args.run_id or f"comp-{uuid.uuid4().hex[:8]}"
    out_path = os.path.join(args.output_dir, f"{run_id}-components.json")

    asyncio.run(
        run_component_benchmark(
            proxy_endpoint=args.endpoint,
            gateway_endpoint=args.gateway_endpoint,
            vllm_endpoint=args.vllm_endpoint,
            model=args.model,
            n_samples=args.n_samples,
            output_path=out_path,
        )
    )
    console.print(f"\n[green]Component results saved to {out_path}[/green]")


def cmd_full(args: argparse.Namespace) -> None:
    """Run all benchmark modes and produce comparison table."""
    from benchmarks.e2e.component_bench import run_component_benchmark
    from benchmarks.e2e.kv_bench import KvCacheBenchmark

    config = _make_config(args)
    run_id = args.run_id or f"full-{uuid.uuid4().hex[:8]}"
    out_dir = os.path.join(config.output_dir, run_id)
    os.makedirs(out_dir, exist_ok=True)

    console.rule("[bold]SemBlend Full Benchmark Suite[/bold]")

    # 1. Response-level cache
    console.rule("[bold]Mode 1: Response-Level Cache[/bold]")
    rc_runner = BenchmarkRunner(
        config=config, run_id=f"{run_id}-response-cache"
    )
    rc_results = asyncio.run(rc_runner.run())
    rc_path = rc_results.save(out_dir)
    console.print(f"  Saved to {rc_path}")

    # 2. KV-tensor injection
    console.rule("[bold]Mode 2: KV-Tensor Injection[/bold]")
    kv_bench = KvCacheBenchmark(config=config, run_id=f"{run_id}-kv-cache")
    kv_summaries = {}
    for ds_name in config.datasets:
        kv_results = asyncio.run(kv_bench.run(ds_name))
        kv_summaries[ds_name] = kv_results.summary()

    # Use first dataset's summary for comparison
    first_ds = config.datasets[0]
    kv_summary = kv_summaries.get(first_ds, {})

    kv_path = os.path.join(out_dir, f"{run_id}-kv-cache.json")
    with open(kv_path, "w") as f:
        json.dump(
            {"mode": "kv-cache", "datasets": kv_summaries}, f, indent=2
        )

    # 3. Component measurements
    console.rule("[bold]Mode 3: Component Measurements[/bold]")
    comp_path = os.path.join(out_dir, f"{run_id}-components.json")
    asyncio.run(
        run_component_benchmark(
            proxy_endpoint=args.endpoint,
            gateway_endpoint=args.gateway_endpoint,
            vllm_endpoint=args.vllm_endpoint,
            model=args.model,
            n_samples=args.n_samples,
            output_path=comp_path,
        )
    )

    # 4. Print comparison
    console.rule("[bold]Comparison[/bold]")
    print_comparison(rc_path, kv_summary, comp_path)

    # 5. Export LaTeX table
    latex_path = os.path.join(out_dir, f"{run_id}-comparison.tex")
    _export_latex_table(rc_path, kv_summary, comp_path, latex_path)
    console.print(f"\n[green]LaTeX table saved to {latex_path}[/green]")
    console.print(f"[green]All results in {out_dir}/[/green]")


def _export_latex_table(
    rc_path: str | None,
    kv_summary: dict | None,
    comp_path: str | None,
    output_path: str,
) -> None:
    """Export comparison as LaTeX table for the paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{SemBlend: Response-Level vs KV-Tensor Caching}",
        r"\label{tab:semblend-comparison}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Baseline & Response Cache & KV Cache \\",
        r"\midrule",
    ]

    # Extract data
    rc_data = {}
    if rc_path and os.path.exists(rc_path):
        with open(rc_path) as f:
            rc_data = json.load(f)

    rc_latency = rc_data.get("summary", {}).get("latency", {})
    rc_bl = rc_latency.get("baseline", {})
    bl_p50 = rc_bl.get("p50_ms", 0)

    rc_tiers = rc_latency.get("tiers", {})
    rc_speedups = rc_latency.get("speedups", {})
    rc_p50 = 0
    rc_sp = 0
    for tier_data in rc_tiers.values():
        if tier_data.get("count", 0) > 0:
            rc_p50 = tier_data.get("p50_ms", 0)
            break
    for sp in rc_speedups.values():
        rc_sp = sp
        break

    kv_bl = kv_summary.get("baseline", {}) if kv_summary else {}
    kv_bl_p50 = kv_bl.get("p50", bl_p50)
    kv_sem = kv_summary.get("semantic", {}) if kv_summary else {}
    kv_p50 = kv_sem.get("p50", 0)
    kv_sp = kv_summary.get("speedup_semantic", 0) if kv_summary else 0

    baseline_p50 = kv_bl_p50 if kv_bl_p50 > 0 else bl_p50

    lines.append(
        f"TTFT P50 (ms) & {baseline_p50:.1f} & {rc_p50:.1f} & {kv_p50:.1f} \\\\"
    )
    lines.append(f"Speedup & 1.0$\\times$ & {rc_sp:.1f}$\\times$ & {kv_sp:.1f}$\\times$ \\\\")

    if kv_summary:
        comp_ratio = kv_summary.get("avg_computation_ratio", 1.0)
        lines.append(
            f"Computation Ratio & 1.0 & 0.0 & {comp_ratio:.2f} \\\\"
        )

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])

    with open(output_path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    mode = getattr(args, "mode", "response-cache")
    handlers = {
        "response-cache": cmd_response_cache,
        "kv-cache": cmd_kv_cache,
        "component": cmd_component,
        "full": cmd_full,
    }

    handler = handlers.get(mode)
    if handler:
        handler(args)
    else:
        console.print(f"[red]Unknown mode: {mode}[/red]")


if __name__ == "__main__":
    main()
