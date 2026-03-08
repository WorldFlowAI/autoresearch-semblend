"""Prometheus-based component latency breakdown for SemBlend paper.

Scrapes synapse_latency_breakdown_seconds histograms from the proxy's /metrics
endpoint before and after a query batch, computing per-stage average latency
from histogram sum/count deltas.

Usage (with port-forward to proxy on 8081):
    python -m benchmarks.e2e.prometheus_breakdown \
        --proxy-metrics http://localhost:8081/metrics \
        --proxy-query   http://localhost:8081/api/v1/query \
        --queries 100 \
        --model Qwen/Qwen2.5-1.5B-Instruct

Produces a component breakdown table suitable for the paper.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)
console = Console()

# Prometheus histogram metrics we scrape for the breakdown.
# Maps human-readable stage name -> (metric_name, label_filter)
BREAKDOWN_STAGES: dict[str, tuple[str, dict[str, str]]] = {
    "query_hash": ("synapse_latency_breakdown_seconds", {"stage": "query_hash"}),
    "bloom": ("synapse_latency_breakdown_seconds", {"stage": "bloom"}),
    "embedding": ("synapse_latency_breakdown_seconds", {"stage": "embedding"}),
    "l1_semantic": ("synapse_latency_breakdown_seconds", {"stage": "l1_semantic"}),
    "l1_exact": ("synapse_latency_breakdown_seconds", {"stage": "l1_exact"}),
    "l2": ("synapse_latency_breakdown_seconds", {"stage": "l2"}),
    "llm": ("synapse_latency_breakdown_seconds", {"stage": "llm"}),
}

# Fallback: coarser-grained metrics if latency_breakdown not populated
FALLBACK_STAGES: dict[str, tuple[str, dict[str, str]]] = {
    "embedding": ("synapse_query_latency_seconds", {"operation": "embedding"}),
    "l1_lookup": ("synapse_query_latency_seconds", {"operation": "l1_lookup"}),
    "l2_search": ("synapse_query_latency_seconds", {"operation": "l2_search"}),
    "llm": ("synapse_query_latency_seconds", {"operation": "llm"}),
    "total": ("synapse_query_latency_seconds", {"operation": "total"}),
}

# Additional fine-grained metrics
EXTRA_METRICS: dict[str, tuple[str, dict[str, str]]] = {
    "query_hash_latency": ("synapse_query_hash_latency_seconds", {}),
    "embedding_provider": ("synapse_embedding_latency_seconds", {}),
    "l0_search": ("synapse_l0_search_latency_seconds", {}),
    "gpu_embedder": ("synapse_gpu_embedder_latency_seconds", {}),
}

# Bitext seed/test queries for generating cache traffic
SEED_QUERIES = [
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
]

TEST_QUERIES = [
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
    "I need to recover my login credentials",
    "How do I end my membership?",
    "What time do you close?",
    "There's an error on my invoice",
    "Can you tell me where my shipment is?",
    "I want my money back",
    "I need to edit my profile information",
    "Do you take credit cards?",
    "My order was damaged in transit",
    "Is there a phone number for support?",
]


@dataclass
class HistogramSnapshot:
    """Sum and count from a Prometheus histogram."""

    sum_value: float = 0.0
    count_value: float = 0.0

    @property
    def avg_seconds(self) -> float:
        if self.count_value == 0:
            return 0.0
        return self.sum_value / self.count_value

    @property
    def avg_ms(self) -> float:
        return self.avg_seconds * 1000


@dataclass
class MetricsSnapshot:
    """All scraped histogram snapshots at a point in time."""

    stages: dict[str, HistogramSnapshot] = field(default_factory=dict)


def _matches_labels(line: str, labels: dict[str, str]) -> bool:
    """Check that all required key=value labels appear in a metric line."""
    return all(f'{k}="{v}"' in line for k, v in labels.items())


def _parse_metric_value(line: str) -> float:
    """Extract the numeric value from a Prometheus metric line."""
    return float(line.strip().rsplit(None, 1)[-1])


def scrape_metrics(metrics_url: str) -> str:
    """Fetch raw Prometheus metrics text with retry."""
    for attempt in range(3):
        try:
            resp = requests.get(metrics_url, timeout=10)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            if attempt == 2:
                raise
            logger.warning(
                "Metrics scrape attempt %d failed: %s", attempt + 1, exc,
            )
            time.sleep(1.0)
    raise RuntimeError("unreachable")


def parse_snapshot(
    raw: str,
    stage_defs: dict[str, tuple[str, dict[str, str]]],
) -> MetricsSnapshot:
    """Parse histogram sum/count values from raw Prometheus text.

    Uses simple string matching (not regex) to avoid label-ordering issues.
    """
    snapshot = MetricsSnapshot()

    for stage_name, (metric_name, labels) in stage_defs.items():
        total_sum = 0.0
        total_count = 0.0

        sum_prefix = f"{metric_name}_sum"
        count_prefix = f"{metric_name}_count"

        for line in raw.splitlines():
            if line.startswith("#"):
                continue
            if line.startswith(sum_prefix) and _matches_labels(line, labels):
                total_sum += _parse_metric_value(line)
            elif line.startswith(count_prefix) and _matches_labels(line, labels):
                total_count += _parse_metric_value(line)

        snapshot.stages[stage_name] = HistogramSnapshot(
            sum_value=total_sum,
            count_value=total_count,
        )

    return snapshot


def compute_deltas(
    before: MetricsSnapshot,
    after: MetricsSnapshot,
) -> dict[str, HistogramSnapshot]:
    """Compute per-stage sum/count deltas between two snapshots."""
    deltas: dict[str, HistogramSnapshot] = {}
    for stage_name in after.stages:
        a = after.stages[stage_name]
        b = before.stages.get(stage_name, HistogramSnapshot())
        delta_sum = a.sum_value - b.sum_value
        delta_count = a.count_value - b.count_value
        if delta_count < 0 or delta_sum < 0:
            logger.warning(
                "Negative delta for stage %r (possible counter reset). "
                "Re-run the benchmark.",
                stage_name,
            )
        deltas[stage_name] = HistogramSnapshot(
            sum_value=max(0.0, delta_sum),
            count_value=max(0.0, delta_count),
        )
    return deltas


def send_queries(
    proxy_url: str,
    model: str,
    queries: list[str],
    skip_cache: bool = False,
    max_tokens: int = 32,
) -> list[dict]:
    """Send queries sequentially and collect timing data."""
    results = []
    for query_text in queries:
        headers = {"Content-Type": "application/json"}
        if skip_cache:
            headers["X-Synapse-Skip-Cache"] = "true"

        body = {
            "query": query_text,
            "model": model,
            "max_tokens": max_tokens,
        }

        url = f"{proxy_url.rstrip('/')}/api/v1/query"
        t_start = time.monotonic()
        try:
            resp = requests.post(
                url,
                json=body,
                headers=headers,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            logger.warning("Query failed: %s", exc)
            data = {"error": str(exc)}

        e2e_ms = (time.monotonic() - t_start) * 1000

        results.append({
            "query": query_text,
            "e2e_ms": e2e_ms,
            "cache_hit": data.get("cache_hit", False),
            "cache_tier": data.get("cache_tier", "miss"),
            "latency_ms": data.get("latency_ms", e2e_ms),
            "similarity": data.get("similarity", 0.0),
        })

    return results


def print_breakdown_table(
    deltas: dict[str, HistogramSnapshot],
    e2e_avg_ms: float,
    title: str = "Latency Decomposition",
) -> None:
    """Print a formatted breakdown table."""
    table = Table(title=title)
    table.add_column("Component", style="bold")
    table.add_column("Avg (ms)", justify="right")
    table.add_column("Count", justify="right")
    table.add_column("% of Total", justify="right")

    computed_total = 0.0
    rows = []

    for stage_name, delta in sorted(deltas.items()):
        if delta.count_value > 0:
            avg = delta.avg_ms
            computed_total += avg
            rows.append((stage_name, avg, int(delta.count_value)))

    for stage_name, avg, count in rows:
        pct = (avg / e2e_avg_ms * 100) if e2e_avg_ms > 0 else 0
        table.add_row(stage_name, f"{avg:.1f}", str(count), f"{pct:.1f}%")

    # Network / overhead row
    overhead = e2e_avg_ms - computed_total
    if overhead > 0:
        pct = overhead / e2e_avg_ms * 100 if e2e_avg_ms > 0 else 0
        table.add_row(
            "network + overhead",
            f"{overhead:.1f}",
            "—",
            f"{pct:.1f}%",
            style="dim",
        )

    table.add_row("", "", "", "", end_section=True)
    table.add_row(
        "Total (E2E)", f"{e2e_avg_ms:.1f}", "—", "100.0%", style="bold"
    )

    console.print(table)


def export_json(
    deltas: dict[str, HistogramSnapshot],
    query_results: list[dict],
    e2e_avg_ms: float,
    output_path: Path,
) -> None:
    """Export breakdown results to JSON."""
    breakdown = {}
    for stage, delta in deltas.items():
        breakdown[stage] = {
            "avg_ms": round(delta.avg_ms, 2),
            "count": int(delta.count_value),
            "total_seconds": round(delta.sum_value, 6),
        }

    computed = sum(d.avg_ms for d in deltas.values() if d.count_value > 0)

    data = {
        "breakdown": breakdown,
        "e2e_avg_ms": round(e2e_avg_ms, 2),
        "computed_component_total_ms": round(computed, 2),
        "network_overhead_ms": round(e2e_avg_ms - computed, 2),
        "num_queries": len(query_results),
        "queries": query_results,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    console.print(f"[green]Results saved to {output_path}[/green]")


def print_latex_table(
    deltas: dict[str, HistogramSnapshot],
    e2e_avg_ms: float,
) -> None:
    """Print LaTeX table rows for the paper."""
    console.print("\n[bold]LaTeX table rows:[/bold]")
    print(r"% Component breakdown for SemBlend paper Section 5.3")
    print(r"\begin{tabular}{@{}lrr@{}}")
    print(r"\toprule")
    print(r"\textbf{Component} & \textbf{Avg (ms)} & \textbf{\% of Total} \\")
    print(r"\midrule")

    computed_total = 0.0
    for stage, delta in sorted(deltas.items()):
        if delta.count_value > 0:
            avg = delta.avg_ms
            computed_total += avg
            pct = avg / e2e_avg_ms * 100 if e2e_avg_ms > 0 else 0
            label = stage.replace("_", r"\_")
            print(rf"{label} & {avg:.1f} & {pct:.1f}\% \\")

    overhead = e2e_avg_ms - computed_total
    if overhead > 0:
        pct = overhead / e2e_avg_ms * 100 if e2e_avg_ms > 0 else 0
        print(rf"Network + overhead & {overhead:.1f} & {pct:.1f}\% \\")

    print(r"\midrule")
    print(rf"\textbf{{Total (E2E)}} & \textbf{{{e2e_avg_ms:.1f}}} & \textbf{{100.0\%}} \\")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prometheus-based latency breakdown for SemBlend paper",
    )
    parser.add_argument(
        "--proxy-metrics",
        default="http://localhost:8081/metrics",
        help="Proxy Prometheus metrics endpoint",
    )
    parser.add_argument(
        "--proxy-query",
        default="http://localhost:8081",
        help="Proxy query endpoint base URL",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name for queries",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of test queries to send (cycles through test set)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens per response (keep low to isolate cache latency)",
    )
    parser.add_argument(
        "--output",
        default="benchmarks/e2e/results/breakdown.json",
        help="Output JSON path",
    )
    parser.add_argument(
        "--skip-seed",
        action="store_true",
        help="Skip sending seed queries (assume cache is already warm)",
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

    # Step 1: Seed the cache (unless skipped)
    if not args.skip_seed:
        console.print("[bold]Phase 1: Seeding cache...[/bold]")
        send_queries(
            args.proxy_query,
            args.model,
            SEED_QUERIES,
            max_tokens=args.max_tokens,
        )
        console.print(f"  Sent {len(SEED_QUERIES)} seed queries")

    # Step 2: Scrape metrics BEFORE test queries
    console.print("[bold]Phase 2: Scraping pre-test metrics...[/bold]")
    raw_before = scrape_metrics(args.proxy_metrics)

    # Try fine-grained breakdown first, fall back to coarser metrics
    snap_before = parse_snapshot(raw_before, BREAKDOWN_STAGES)
    has_breakdown = any(
        s.count_value > 0 for s in snap_before.stages.values()
    )

    if not has_breakdown:
        console.print("  [yellow]No latency_breakdown data, using fallback metrics[/yellow]")
        snap_before = parse_snapshot(raw_before, FALLBACK_STAGES)

    # Also grab extra metrics
    extra_before = parse_snapshot(raw_before, EXTRA_METRICS)

    # Step 3: Send test queries
    console.print(f"[bold]Phase 3: Sending {args.queries} test queries...[/bold]")

    # Build query list by cycling through test queries
    full_query_list = []
    for i in range(args.queries):
        full_query_list.append(TEST_QUERIES[i % len(TEST_QUERIES)])

    query_results = send_queries(
        args.proxy_query,
        args.model,
        full_query_list,
        max_tokens=args.max_tokens,
    )

    hits = sum(1 for q in query_results if q["cache_hit"])
    e2e_avg_ms = sum(q["e2e_ms"] for q in query_results) / len(query_results)
    console.print(f"  Sent {len(query_results)} queries, {hits} hits, avg E2E: {e2e_avg_ms:.1f}ms")

    # Step 4: Scrape metrics AFTER test queries
    console.print("[bold]Phase 4: Scraping post-test metrics...[/bold]")
    raw_after = scrape_metrics(args.proxy_metrics)

    stage_defs = BREAKDOWN_STAGES if has_breakdown else FALLBACK_STAGES
    snap_after = parse_snapshot(raw_after, stage_defs)
    extra_after = parse_snapshot(raw_after, EXTRA_METRICS)

    # Step 5: Compute deltas
    deltas = compute_deltas(snap_before, snap_after)
    extra_deltas = compute_deltas(extra_before, extra_after)

    # Step 6: Print results
    console.print()
    print_breakdown_table(deltas, e2e_avg_ms)

    # Print extra metrics if available
    extra_with_data = {
        k: v for k, v in extra_deltas.items() if v.count_value > 0
    }
    if extra_with_data:
        console.print()
        print_breakdown_table(
            extra_with_data,
            e2e_avg_ms,
            title="Additional Fine-Grained Metrics",
        )

    # Step 7: Export
    output_path = Path(args.output)
    all_deltas = {**deltas, **extra_deltas}
    export_json(all_deltas, query_results, e2e_avg_ms, output_path)

    if args.latex:
        print_latex_table(deltas, e2e_avg_ms)

    # Step 8: Print paper-ready summary
    console.print("\n[bold]Paper-ready summary:[/bold]")
    console.print(f"  E2E avg: {e2e_avg_ms:.1f} ms")
    computed = sum(d.avg_ms for d in deltas.values() if d.count_value > 0)
    overhead = e2e_avg_ms - computed
    console.print(f"  Computed components: {computed:.1f} ms")
    pct = overhead / e2e_avg_ms * 100 if e2e_avg_ms > 0 else 0
    console.print(f"  Network overhead: {overhead:.1f} ms ({pct:.1f}%)")

    # Inference-engine estimate (components minus network)
    console.print(f"  Inference-engine estimate: {computed:.1f} ms (no proxy overhead)")


if __name__ == "__main__":
    main()
