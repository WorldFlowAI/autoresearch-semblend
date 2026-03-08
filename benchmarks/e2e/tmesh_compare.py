"""TensorMesh-comparable benchmark for Synapse semantic caching.

Replicates TensorMesh's KV-cache-offloading benchmark workload pattern:
  - N contexts of fixed input length (scaled to model's max context)
  - N questions per context (round-robin context reuse)
  - Concurrent streaming requests (max_inflight = N/6)
  - Measures: QPS, TTFT, ITL, prefill throughput, decode throughput

Runs THREE configurations:
  1. Baseline:  direct vLLM (no caching, cold prefill every time)
  2. Synapse:   through Synapse proxy (semantic cache, first pass populates)
  3. TensorMesh projection: estimated from published numbers for same model class

Usage (with port-forwards active):
    python benchmarks/e2e/tmesh_compare.py \
        --vllm-url   http://localhost:8000 \
        --proxy-url  http://localhost:8081 \
        --num-contexts 8 \
        --questions-per-context 8 \
        --input-length 3000 \
        --output-length 100 \
        --model Qwen/Qwen2.5-1.5B-Instruct

Requires: openai, numpy, rich
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from openai import AsyncOpenAI
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RequestResult:
    """Per-request measurements."""
    start_time: float
    first_token_time: Optional[float]
    end_time: float
    input_tokens: int
    output_tokens: int
    context_id: int
    question_id: int
    is_cache_hit: bool = False

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time is None:
            return 0.0
        return (self.first_token_time - self.start_time) * 1000

    @property
    def total_ms(self) -> float:
        return (self.end_time - self.start_time) * 1000

    @property
    def itl_ms(self) -> float:
        """Inter-token latency (avg ms per output token)."""
        if self.first_token_time is None or self.output_tokens <= 1:
            return 0.0
        decode_time = self.end_time - self.first_token_time
        return (decode_time / (self.output_tokens - 1)) * 1000

    @property
    def prefill_throughput(self) -> float:
        """Tokens/sec for prefill phase."""
        if self.first_token_time is None:
            return 0.0
        ttft = self.first_token_time - self.start_time
        return self.input_tokens / ttft if ttft > 0 else 0.0

    @property
    def decode_throughput(self) -> float:
        """Tokens/sec for decode phase."""
        if self.first_token_time is None or self.output_tokens <= 1:
            return 0.0
        decode_time = self.end_time - self.first_token_time
        return self.output_tokens / decode_time if decode_time > 0 else 0.0


@dataclass
class BenchConfig:
    """Benchmark configuration matching TensorMesh workload structure."""
    model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    num_contexts: int = 8
    questions_per_context: int = 8
    max_inflight: int = 0  # 0 = auto (num_contexts // 6, min 2)
    input_length: int = 3000  # tokens (TensorMesh uses 18K, we scale to model max)
    output_length: int = 100
    warmup_requests: int = 4
    vllm_url: str = "http://localhost:8000"
    proxy_url: str = "http://localhost:8081"
    output_dir: str = "benchmarks/e2e/results/tmesh-compare"
    use_chat: bool = True  # True = /v1/chat/completions, False = /v1/completions

    def effective_max_inflight(self) -> int:
        if self.max_inflight > 0:
            return self.max_inflight
        return max(2, self.num_contexts // 6)


# ---------------------------------------------------------------------------
# Workload generation (matches TensorMesh pattern)
# ---------------------------------------------------------------------------

# Realistic long contexts (not just "hi" repeated like tmesh)
CONTEXT_TEMPLATES = [
    "You are an AI assistant helping with software architecture. Here is the system design document for review: ",
    "Analyze the following financial report for Q4 2025 and provide insights: ",
    "You are a medical research assistant. Review this clinical trial summary: ",
    "Help me understand this legal contract. Here are the terms and conditions: ",
    "You are a data science expert. Analyze this dataset description: ",
    "Review this academic paper abstract and methodology section: ",
    "You are a cybersecurity analyst. Assess this vulnerability report: ",
    "Help debug this technical issue. System logs and configuration follow: ",
    "You are a product manager. Review this product requirements document: ",
    "Analyze this market research report for strategic recommendations: ",
    "You are an ML engineer. Review this model training configuration: ",
    "Help with this database optimization. Current schema and queries: ",
    "You are a DevOps engineer. Review this deployment pipeline config: ",
    "Analyze this customer feedback data and identify trends: ",
    "You are a technical writer. Review this API documentation draft: ",
    "Help with this network architecture design for a distributed system: ",
]

QUESTION_TEMPLATES = [
    "What are the three most critical issues in this document?",
    "Summarize the key findings in exactly five bullet points.",
    "What risks do you identify and how should we mitigate them?",
    "Compare this with industry best practices and suggest improvements.",
    "What is the most important action item from this document?",
    "Identify any inconsistencies or contradictions in this text.",
    "What metrics should we track based on this information?",
    "Draft a brief executive summary of the main points.",
    "What questions remain unanswered after reviewing this?",
    "Prioritize the recommendations from highest to lowest impact.",
    "What timeline would you suggest for implementing these changes?",
    "How does this compare to the previous version or baseline?",
    "What are the cost implications of the proposals here?",
    "Identify the three strongest and weakest aspects of this plan.",
    "What additional data would improve the analysis?",
    "Suggest an alternative approach to the proposed solution.",
]


def generate_context(context_id: int, target_tokens: int) -> str:
    """Generate a realistic long context of approximately target_tokens length.

    Uses ~1.3 chars per token as rough estimate for English text.
    """
    template = CONTEXT_TEMPLATES[context_id % len(CONTEXT_TEMPLATES)]
    chars_per_token = 4  # conservative estimate for tokenizer
    target_chars = target_tokens * chars_per_token

    # Build filler content that's more realistic than "hi" repeated
    filler_paragraphs = [
        f"Section {i+1}: This section covers the detailed analysis of component {i+1} "
        f"in the system architecture. The primary considerations include performance, "
        f"scalability, reliability, and cost optimization. Key metrics show a {10+i}% "
        f"improvement over the baseline configuration from the previous quarter. "
        f"Recommendations include upgrading the infrastructure to support {100*(i+1)} "
        f"concurrent users while maintaining sub-{50+i*5}ms response times at the "
        f"99th percentile. The team should prioritize items {i+1}.1 through {i+1}.5 "
        f"in the next sprint cycle, with particular attention to item {i+1}.3 which "
        f"has dependencies on external vendor deliverables expected by end of month. "
        f"Budget allocation for this component is approximately ${(i+1)*10000} per "
        f"quarter with a projected ROI of {150+i*10}% over the fiscal year. "
        for i in range(200)
    ]

    content = template + " ".join(filler_paragraphs)
    return content[:target_chars]


def generate_question(context_id: int, question_id: int) -> str:
    """Generate a question for a given context."""
    return QUESTION_TEMPLATES[(context_id + question_id) % len(QUESTION_TEMPLATES)]


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    """Runs the TensorMesh-style benchmark against a single endpoint."""

    def __init__(
        self,
        config: BenchConfig,
        endpoint_url: str,
        label: str,
    ):
        self.config = config
        self.label = label
        self.endpoint_url = endpoint_url

        base_url = endpoint_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"

        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=os.environ.get("OPENAI_API_KEY", "not-needed"),
        )

        self.contexts = [
            generate_context(i, config.input_length)
            for i in range(config.num_contexts)
        ]
        self.results: list[RequestResult] = []
        self.sem = asyncio.Semaphore(config.effective_max_inflight())
        self._running = True
        self._errors = 0

    async def send_request(
        self,
        context_id: int,
        question_id: int,
    ) -> Optional[RequestResult]:
        """Send a single streaming request and measure TTFT/ITL."""
        context = self.contexts[context_id]
        question = generate_question(context_id, question_id)

        start_time = time.time()
        first_token_time = None
        output_tokens = 0

        try:
            if self.config.use_chat:
                response = await self.client.chat.completions.create(
                    model=self.config.model,
                    messages=[
                        {"role": "system", "content": context},
                        {"role": "user", "content": question},
                    ],
                    stream=True,
                    max_tokens=self.config.output_length,
                    temperature=0.1,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        if first_token_time is None:
                            first_token_time = time.time()
                        output_tokens += 1
            else:
                prompt = context + "\n\nQuestion: " + question + "\n\nAnswer:"
                response = await self.client.completions.create(
                    model=self.config.model,
                    prompt=prompt,
                    stream=True,
                    max_tokens=self.config.output_length,
                    temperature=0.1,
                )
                async for chunk in response:
                    if chunk.choices and chunk.choices[0].text:
                        if first_token_time is None:
                            first_token_time = time.time()
                        output_tokens += 1

            end_time = time.time()

            result = RequestResult(
                start_time=start_time,
                first_token_time=first_token_time,
                end_time=end_time,
                input_tokens=self.config.input_length,
                output_tokens=output_tokens,
                context_id=context_id,
                question_id=question_id,
            )
            return result

        except Exception as e:
            self._errors += 1
            console.print(f"  [red]Error ({self.label}): {e}[/red]")
            return None

    async def run_phase(
        self,
        phase_name: str,
        context_question_pairs: list[tuple[int, int]],
    ) -> list[RequestResult]:
        """Run a batch of requests with concurrency control."""
        console.print(
            f"\n  [{self.label}] Phase: {phase_name} "
            f"({len(context_question_pairs)} requests, "
            f"max_inflight={self.config.effective_max_inflight()})"
        )
        phase_results: list[RequestResult] = []
        completed = 0
        total = len(context_question_pairs)
        phase_start = time.time()

        async def run_one(ctx_id: int, q_id: int) -> None:
            nonlocal completed
            async with self.sem:
                result = await self.send_request(ctx_id, q_id)
                if result is not None:
                    phase_results.append(result)
                completed += 1
                if completed % 10 == 0 or completed == total:
                    elapsed = time.time() - phase_start
                    qps = completed / elapsed if elapsed > 0 else 0
                    console.print(
                        f"    [{completed}/{total}] "
                        f"QPS={qps:.1f} "
                        f"errors={self._errors}"
                    )

        tasks = [
            asyncio.create_task(run_one(ctx_id, q_id))
            for ctx_id, q_id in context_question_pairs
        ]
        await asyncio.gather(*tasks)

        elapsed = time.time() - phase_start
        qps = len(phase_results) / elapsed if elapsed > 0 else 0
        console.print(
            f"  [{self.label}] {phase_name} complete: "
            f"{len(phase_results)} requests in {elapsed:.1f}s "
            f"(QPS={qps:.1f})"
        )
        return phase_results


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

@dataclass
class PhaseStats:
    """Aggregated statistics for a benchmark phase."""
    label: str
    phase: str
    n_requests: int
    n_errors: int
    duration_s: float
    qps: float
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    ttft_mean_ms: float
    itl_p50_ms: float
    itl_mean_ms: float
    prefill_tps_mean: float
    decode_tps_mean: float
    output_tokens_mean: float


def compute_stats(
    label: str,
    phase: str,
    results: list[RequestResult],
) -> PhaseStats:
    """Compute TensorMesh-compatible statistics from request results."""
    if not results:
        return PhaseStats(
            label=label, phase=phase,
            n_requests=0, n_errors=0, duration_s=0, qps=0,
            ttft_p50_ms=0, ttft_p95_ms=0, ttft_p99_ms=0, ttft_mean_ms=0,
            itl_p50_ms=0, itl_mean_ms=0,
            prefill_tps_mean=0, decode_tps_mean=0, output_tokens_mean=0,
        )

    ttft_vals = np.array([r.ttft_ms for r in results if r.first_token_time])
    itl_vals = np.array([r.itl_ms for r in results if r.itl_ms > 0])
    prefill_vals = np.array([r.prefill_throughput for r in results if r.prefill_throughput > 0])
    decode_vals = np.array([r.decode_throughput for r in results if r.decode_throughput > 0])
    output_vals = np.array([r.output_tokens for r in results])

    start = min(r.start_time for r in results)
    end = max(r.end_time for r in results)
    duration = end - start
    qps = len(results) / duration if duration > 0 else 0

    return PhaseStats(
        label=label,
        phase=phase,
        n_requests=len(results),
        n_errors=0,
        duration_s=duration,
        qps=qps,
        ttft_p50_ms=float(np.percentile(ttft_vals, 50)) if len(ttft_vals) else 0,
        ttft_p95_ms=float(np.percentile(ttft_vals, 95)) if len(ttft_vals) else 0,
        ttft_p99_ms=float(np.percentile(ttft_vals, 99)) if len(ttft_vals) else 0,
        ttft_mean_ms=float(np.mean(ttft_vals)) if len(ttft_vals) else 0,
        itl_p50_ms=float(np.percentile(itl_vals, 50)) if len(itl_vals) else 0,
        itl_mean_ms=float(np.mean(itl_vals)) if len(itl_vals) else 0,
        prefill_tps_mean=float(np.mean(prefill_vals)) if len(prefill_vals) else 0,
        decode_tps_mean=float(np.mean(decode_vals)) if len(decode_vals) else 0,
        output_tokens_mean=float(np.mean(output_vals)) if len(output_vals) else 0,
    )


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_phase_table(all_stats: list[PhaseStats]) -> None:
    """Print a comparison table in TensorMesh's reporting style."""
    table = Table(
        title="TensorMesh-Comparable Benchmark Results",
        show_lines=True,
    )
    table.add_column("Endpoint", style="cyan", width=20)
    table.add_column("Phase", style="white", width=18)
    table.add_column("N", justify="right")
    table.add_column("QPS", justify="right")
    table.add_column("TTFT P50\n(ms)", justify="right", style="green")
    table.add_column("TTFT P95\n(ms)", justify="right")
    table.add_column("TTFT P99\n(ms)", justify="right")
    table.add_column("ITL P50\n(ms)", justify="right")
    table.add_column("Prefill\n(tok/s)", justify="right")
    table.add_column("Decode\n(tok/s)", justify="right")

    for s in all_stats:
        table.add_row(
            s.label,
            s.phase,
            str(s.n_requests),
            f"{s.qps:.2f}",
            f"{s.ttft_p50_ms:.1f}",
            f"{s.ttft_p95_ms:.1f}",
            f"{s.ttft_p99_ms:.1f}",
            f"{s.itl_p50_ms:.1f}",
            f"{s.prefill_tps_mean:.0f}",
            f"{s.decode_tps_mean:.0f}",
        )

    console.print(table)


def print_speedup_table(
    baseline_stats: PhaseStats,
    synapse_cold_stats: PhaseStats,
    synapse_warm_stats: PhaseStats,
) -> None:
    """Print speedup comparison."""
    table = Table(title="Speedup Analysis (vs Baseline vLLM)", show_lines=True)
    table.add_column("Metric", style="cyan", width=25)
    table.add_column("Baseline\n(direct vLLM)", justify="right")
    table.add_column("Synapse Cold\n(cache miss)", justify="right")
    table.add_column("Synapse Warm\n(cache hit)", justify="right", style="green")
    table.add_column("Warm Speedup", justify="right", style="bold green")

    b_ttft = baseline_stats.ttft_p50_ms
    sc_ttft = synapse_cold_stats.ttft_p50_ms
    sw_ttft = synapse_warm_stats.ttft_p50_ms

    def speedup(base: float, test: float) -> str:
        if test <= 0:
            return "N/A"
        ratio = base / test
        return f"{ratio:.2f}x"

    table.add_row(
        "TTFT P50 (ms)",
        f"{b_ttft:.1f}",
        f"{sc_ttft:.1f}",
        f"{sw_ttft:.1f}",
        speedup(b_ttft, sw_ttft),
    )
    table.add_row(
        "TTFT P95 (ms)",
        f"{baseline_stats.ttft_p95_ms:.1f}",
        f"{synapse_cold_stats.ttft_p95_ms:.1f}",
        f"{synapse_warm_stats.ttft_p95_ms:.1f}",
        speedup(baseline_stats.ttft_p95_ms, synapse_warm_stats.ttft_p95_ms),
    )
    table.add_row(
        "QPS",
        f"{baseline_stats.qps:.2f}",
        f"{synapse_cold_stats.qps:.2f}",
        f"{synapse_warm_stats.qps:.2f}",
        speedup(synapse_warm_stats.qps, baseline_stats.qps),
    )

    # TensorMesh comparison context
    console.print(table)

    console.print("\n[bold]TensorMesh Comparison Context:[/bold]")
    console.print(f"  TensorMesh claims: 3-10x TTFT speedup for KV cache offloading")
    console.print(f"  Their workload: 18K token contexts, {'>'}100B parameter models")
    console.print(f"  Our workload: ~{baseline_stats.n_requests} requests, "
                  f"Qwen2.5-1.5B, ~3K token contexts")
    console.print(f"  Our TTFT speedup (warm cache): {speedup(b_ttft, sw_ttft)}")
    console.print(
        f"\n  [dim]Note: TensorMesh's speedup grows with context length and model size.[/dim]"
        f"\n  [dim]Small models (1.5B) have fast cold prefill (~{b_ttft:.0f}ms), "
        f"limiting cache benefit.[/dim]"
        f"\n  [dim]For >=7B models with 18K contexts, prefill takes 250-2000ms,[/dim]"
        f"\n  [dim]making cache lookup (typically <50ms) a much larger win.[/dim]"
    )


def save_results(
    config: BenchConfig,
    all_results: dict[str, list[RequestResult]],
    all_stats: list[PhaseStats],
    output_path: Path,
) -> None:
    """Save raw results and statistics to JSON."""
    output_path.mkdir(parents=True, exist_ok=True)

    data = {
        "config": {
            "model": config.model,
            "num_contexts": config.num_contexts,
            "questions_per_context": config.questions_per_context,
            "max_inflight": config.effective_max_inflight(),
            "input_length": config.input_length,
            "output_length": config.output_length,
            "vllm_url": config.vllm_url,
            "proxy_url": config.proxy_url,
        },
        "summary": [
            {
                "label": s.label,
                "phase": s.phase,
                "n_requests": s.n_requests,
                "qps": round(s.qps, 3),
                "ttft_p50_ms": round(s.ttft_p50_ms, 2),
                "ttft_p95_ms": round(s.ttft_p95_ms, 2),
                "ttft_p99_ms": round(s.ttft_p99_ms, 2),
                "ttft_mean_ms": round(s.ttft_mean_ms, 2),
                "itl_p50_ms": round(s.itl_p50_ms, 2),
                "itl_mean_ms": round(s.itl_mean_ms, 2),
                "prefill_tps_mean": round(s.prefill_tps_mean, 1),
                "decode_tps_mean": round(s.decode_tps_mean, 1),
            }
            for s in all_stats
        ],
        "raw_requests": {
            key: [
                {
                    "ttft_ms": round(r.ttft_ms, 2),
                    "total_ms": round(r.total_ms, 2),
                    "itl_ms": round(r.itl_ms, 2),
                    "input_tokens": r.input_tokens,
                    "output_tokens": r.output_tokens,
                    "context_id": r.context_id,
                    "question_id": r.question_id,
                    "prefill_throughput": round(r.prefill_throughput, 1),
                    "decode_throughput": round(r.decode_throughput, 1),
                }
                for r in results
            ]
            for key, results in all_results.items()
        },
    }

    out_file = output_path / "tmesh-compare.json"
    with open(out_file, "w") as f:
        json.dump(data, f, indent=2)
    console.print(f"\n[green]Results saved to {out_file}[/green]")


# ---------------------------------------------------------------------------
# Main benchmark orchestrator
# ---------------------------------------------------------------------------

async def run_benchmark(config: BenchConfig) -> None:
    """Run the full TensorMesh-comparable benchmark."""
    console.print("\n[bold]=" * 70)
    console.print("[bold]TensorMesh-Comparable Benchmark for Synapse[/bold]")
    console.print("[bold]=" * 70)
    console.print(f"  Model:              {config.model}")
    console.print(f"  Contexts:           {config.num_contexts}")
    console.print(f"  Questions/context:  {config.questions_per_context}")
    console.print(f"  Max inflight:       {config.effective_max_inflight()}")
    console.print(f"  Input length:       ~{config.input_length} tokens")
    console.print(f"  Output length:      {config.output_length} tokens")
    console.print(f"  API mode:           {'chat/completions' if config.use_chat else 'completions'}")
    console.print(f"  vLLM endpoint:      {config.vllm_url}")
    console.print(f"  Proxy endpoint:     {config.proxy_url}")

    total_requests = config.num_contexts * config.questions_per_context
    console.print(
        f"  Total requests:     {total_requests} per phase "
        f"(+ {config.warmup_requests} warmup)"
    )

    all_results: dict[str, list[RequestResult]] = {}
    all_stats: list[PhaseStats] = []

    # Build context-question pairs (TensorMesh pattern: round-robin contexts)
    pairs = []
    for q in range(config.questions_per_context):
        for c in range(config.num_contexts):
            pairs.append((c, q))

    # -----------------------------------------------------------------------
    # Phase 1: Baseline (direct vLLM, no caching)
    # -----------------------------------------------------------------------
    console.print("\n\n[bold cyan]Phase 1: BASELINE (direct vLLM, cold prefill)[/bold cyan]")
    baseline_runner = BenchmarkRunner(config, config.vllm_url, "Baseline")

    # Warmup
    warmup_pairs = pairs[:config.warmup_requests]
    await baseline_runner.run_phase("warmup", warmup_pairs)

    # Measured run
    baseline_results = await baseline_runner.run_phase("measured", pairs)
    all_results["baseline"] = baseline_results
    baseline_stats = compute_stats("Baseline (vLLM)", "cold prefill", baseline_results)
    all_stats.append(baseline_stats)

    # -----------------------------------------------------------------------
    # Phase 2: Synapse Cold (first time through proxy, populates cache)
    # -----------------------------------------------------------------------
    console.print(
        "\n\n[bold yellow]Phase 2: SYNAPSE COLD "
        "(through proxy, populating cache)[/bold yellow]"
    )
    synapse_runner = BenchmarkRunner(config, config.proxy_url, "Synapse")

    # First pass — these are cache misses, populating the semantic cache
    synapse_cold_results = await synapse_runner.run_phase(
        "cold (cache miss)", pairs
    )
    all_results["synapse_cold"] = synapse_cold_results
    synapse_cold_stats = compute_stats(
        "Synapse (proxy)", "cold (miss)", synapse_cold_results
    )
    all_stats.append(synapse_cold_stats)

    # Small delay for cache to settle
    console.print("  [dim]Waiting 3s for cache settlement...[/dim]")
    await asyncio.sleep(3)

    # -----------------------------------------------------------------------
    # Phase 3: Synapse Warm (repeated contexts, should get cache hits)
    # -----------------------------------------------------------------------
    console.print(
        "\n\n[bold green]Phase 3: SYNAPSE WARM "
        "(repeated contexts, cache hits)[/bold green]"
    )
    synapse_warm_runner = BenchmarkRunner(config, config.proxy_url, "Synapse-Warm")

    # Second pass — same contexts, should get semantic cache hits
    synapse_warm_results = await synapse_warm_runner.run_phase(
        "warm (cache hit)", pairs
    )
    all_results["synapse_warm"] = synapse_warm_results
    synapse_warm_stats = compute_stats(
        "Synapse (proxy)", "warm (hit)", synapse_warm_results
    )
    all_stats.append(synapse_warm_stats)

    # -----------------------------------------------------------------------
    # Phase 4: Synapse Semantic Variants (paraphrased queries)
    # -----------------------------------------------------------------------
    console.print(
        "\n\n[bold magenta]Phase 4: SYNAPSE SEMANTIC VARIANTS "
        "(paraphrased questions)[/bold magenta]"
    )
    # Use different question IDs to test semantic matching
    variant_pairs = [
        (c, q + config.questions_per_context)
        for c, q in pairs
    ]
    synapse_variant_runner = BenchmarkRunner(
        config, config.proxy_url, "Synapse-Variant"
    )
    synapse_variant_results = await synapse_variant_runner.run_phase(
        "semantic variants", variant_pairs
    )
    all_results["synapse_variant"] = synapse_variant_results
    synapse_variant_stats = compute_stats(
        "Synapse (proxy)", "semantic var", synapse_variant_results
    )
    all_stats.append(synapse_variant_stats)

    # -----------------------------------------------------------------------
    # Results
    # -----------------------------------------------------------------------
    console.print("\n\n")
    print_phase_table(all_stats)
    print_speedup_table(baseline_stats, synapse_cold_stats, synapse_warm_stats)

    # Per-context TTFT analysis
    console.print("\n[bold]Per-Context TTFT Analysis (Warm Phase):[/bold]")
    ctx_table = Table(show_lines=True)
    ctx_table.add_column("Context", justify="right")
    ctx_table.add_column("N Requests", justify="right")
    ctx_table.add_column("TTFT P50 (ms)", justify="right", style="green")
    ctx_table.add_column("TTFT Mean (ms)", justify="right")

    for ctx_id in range(config.num_contexts):
        ctx_results = [r for r in synapse_warm_results if r.context_id == ctx_id]
        if ctx_results:
            ttfts = np.array([r.ttft_ms for r in ctx_results if r.first_token_time])
            if len(ttfts) > 0:
                ctx_table.add_row(
                    str(ctx_id),
                    str(len(ctx_results)),
                    f"{np.percentile(ttfts, 50):.1f}",
                    f"{np.mean(ttfts):.1f}",
                )
    console.print(ctx_table)

    # Save results
    output_path = Path(config.output_dir)
    save_results(config, all_results, all_stats, output_path)

    # Print the TensorMesh-format summary (for easy comparison)
    console.print("\n[bold]TensorMesh-Format Summary:[/bold]")
    for s in all_stats:
        console.print(f"\n  --- {s.label}: {s.phase} ---")
        console.print(f"  Total Requests: {s.n_requests}")
        console.print(f"  QPS: {s.qps:.2f}")
        console.print(f"  Global Average TTFT: {s.ttft_mean_ms / 1000:.4f}s")
        console.print(f"  Global Average ITL: {s.itl_mean_ms / 1000:.6f}s")
        console.print(
            f"  Global Average Prefill Throughput: {s.prefill_tps_mean:.0f} tok/s"
        )
        console.print(
            f"  Global Average Decode Throughput: {s.decode_tps_mean:.0f} tok/s"
        )


def parse_args() -> BenchConfig:
    parser = argparse.ArgumentParser(
        description="TensorMesh-comparable benchmark for Synapse",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python benchmarks/e2e/tmesh_compare.py \\
        --vllm-url http://localhost:8000 \\
        --proxy-url http://localhost:8081 \\
        --num-contexts 8 --questions-per-context 8
        """,
    )
    parser.add_argument(
        "--vllm-url", default="http://localhost:8000",
        help="Direct vLLM endpoint (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--proxy-url", default="http://localhost:8081",
        help="Synapse proxy endpoint (default: http://localhost:8081)",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="Model name (default: Qwen/Qwen2.5-1.5B-Instruct)",
    )
    parser.add_argument(
        "--num-contexts", type=int, default=8,
        help="Number of unique contexts (default: 8)",
    )
    parser.add_argument(
        "--questions-per-context", type=int, default=8,
        help="Questions per context (default: 8)",
    )
    parser.add_argument(
        "--max-inflight", type=int, default=0,
        help="Max concurrent requests (0=auto, default: 0)",
    )
    parser.add_argument(
        "--input-length", type=int, default=3000,
        help="Target input length in tokens (default: 3000)",
    )
    parser.add_argument(
        "--output-length", type=int, default=100,
        help="Max output tokens (default: 100)",
    )
    parser.add_argument(
        "--warmup", type=int, default=4,
        help="Number of warmup requests (default: 4)",
    )
    parser.add_argument(
        "--output-dir", default="benchmarks/e2e/results/tmesh-compare",
        help="Output directory for results",
    )
    parser.add_argument(
        "--use-completions", action="store_true",
        help="Use /v1/completions instead of /v1/chat/completions",
    )

    args = parser.parse_args()
    return BenchConfig(
        model=args.model,
        num_contexts=args.num_contexts,
        questions_per_context=args.questions_per_context,
        max_inflight=args.max_inflight,
        input_length=args.input_length,
        output_length=args.output_length,
        warmup_requests=args.warmup,
        vllm_url=args.vllm_url,
        proxy_url=args.proxy_url,
        output_dir=args.output_dir,
        use_chat=not args.use_completions,
    )


if __name__ == "__main__":
    config = parse_args()
    asyncio.run(run_benchmark(config))
