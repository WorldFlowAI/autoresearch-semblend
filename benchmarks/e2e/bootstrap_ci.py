"""Bootstrap 95% confidence intervals for benchmark results.

Computes CIs for P50 TTFT, hit rate, and speedup from raw per-query JSON data.
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_BOOTSTRAP = 10_000
CI_ALPHA = 0.05  # 95% CI


@dataclass(frozen=True)
class CIResult:
    """A point estimate with 95% CI bounds."""

    estimate: float
    ci_lower: float
    ci_upper: float

    def __str__(self) -> str:
        return f"{self.estimate:.1f} [{self.ci_lower:.1f}, {self.ci_upper:.1f}]"

    def latex(self, fmt: str = ".1f") -> str:
        e = format(self.estimate, fmt)
        lo = format(self.ci_lower, fmt)
        hi = format(self.ci_upper, fmt)
        return f"${e}$ \\tiny{{[{lo}, {hi}]}}"


def bootstrap_percentile(
    values: np.ndarray,
    pct: float = 50.0,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> CIResult:
    """Bootstrap CI for a percentile statistic."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return CIResult(0.0, 0.0, 0.0)

    point = float(np.percentile(values, pct))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = np.percentile(sample, pct)

    lo = float(np.percentile(boot_stats, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - CI_ALPHA / 2)))
    return CIResult(point, lo, hi)


def bootstrap_mean(
    values: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> CIResult:
    """Bootstrap CI for the mean."""
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return CIResult(0.0, 0.0, 0.0)

    point = float(np.mean(values))
    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        boot_stats[i] = np.mean(sample)

    lo = float(np.percentile(boot_stats, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - CI_ALPHA / 2)))
    return CIResult(point, lo, hi)


def bootstrap_proportion(
    hits: int,
    total: int,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> CIResult:
    """Bootstrap CI for a proportion (hit rate)."""
    rng = np.random.default_rng(seed)
    if total == 0:
        return CIResult(0.0, 0.0, 0.0)

    point = hits / total
    outcomes = np.zeros(total)
    outcomes[:hits] = 1.0

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        sample = rng.choice(outcomes, size=total, replace=True)
        boot_stats[i] = np.mean(sample)

    lo = float(np.percentile(boot_stats, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - CI_ALPHA / 2)))
    return CIResult(100 * point, 100 * lo, 100 * hi)


def bootstrap_speedup(
    cached_ttft: np.ndarray,
    baseline_ttft: np.ndarray,
    n_boot: int = N_BOOTSTRAP,
    seed: int = 42,
) -> CIResult:
    """Bootstrap CI for speedup = median(baseline) / median(cached)."""
    rng = np.random.default_rng(seed)
    if len(cached_ttft) == 0 or len(baseline_ttft) == 0:
        return CIResult(0.0, 0.0, 0.0)

    base_p50 = float(np.median(baseline_ttft))
    cache_p50 = float(np.median(cached_ttft))
    point = base_p50 / cache_p50 if cache_p50 > 0 else 0.0

    boot_stats = np.empty(n_boot)
    for i in range(n_boot):
        b = rng.choice(baseline_ttft, size=len(baseline_ttft), replace=True)
        c = rng.choice(cached_ttft, size=len(cached_ttft), replace=True)
        c_med = float(np.median(c))
        boot_stats[i] = float(np.median(b)) / c_med if c_med > 0 else 0.0

    lo = float(np.percentile(boot_stats, 100 * CI_ALPHA / 2))
    hi = float(np.percentile(boot_stats, 100 * (1 - CI_ALPHA / 2)))
    return CIResult(point, lo, hi)


@dataclass
class DatasetCI:
    """All confidence intervals for one dataset."""

    name: str
    n_queries: int
    hit_rate: CIResult
    l0_p50: CIResult
    baseline_p50: CIResult
    speedup: CIResult
    bleu: CIResult
    rouge_l: CIResult


def compute_dataset_ci(name: str, queries: list[dict]) -> DatasetCI:
    """Compute all CIs for a single dataset's raw query results."""
    n = len(queries)
    hits = sum(1 for q in queries if q.get("cache_hit", False))

    # L0 GPU TTFT values
    l0_ttft = np.array([
        q["ttft_ms"]
        for q in queries
        if q.get("cache_tier") == "l0_gpu"
    ])

    # Baseline TTFT values
    baseline_ttft = np.array([
        q["baseline_ttft_ms"]
        for q in queries
        if q.get("baseline_ttft_ms") is not None
    ])

    # Quality metrics (cache hits only)
    bleu_vals = np.array([
        q["bleu"]
        for q in queries
        if q.get("bleu") is not None and q.get("cache_hit", False)
    ])
    rouge_vals = np.array([
        q["rouge_l"]
        for q in queries
        if q.get("rouge_l") is not None and q.get("cache_hit", False)
    ])

    return DatasetCI(
        name=name,
        n_queries=n,
        hit_rate=bootstrap_proportion(hits, n),
        l0_p50=bootstrap_percentile(l0_ttft, 50.0),
        baseline_p50=bootstrap_percentile(baseline_ttft, 50.0),
        speedup=bootstrap_speedup(l0_ttft, baseline_ttft),
        bleu=bootstrap_mean(bleu_vals),
        rouge_l=bootstrap_mean(rouge_vals),
    )


def load_and_compute(result_path: Path) -> list[DatasetCI]:
    """Load a benchmark JSON file and compute CIs for all datasets in it."""
    with open(result_path) as f:
        data = json.load(f)

    results = []
    for ds_name, queries in data.get("queries", {}).items():
        if queries:
            results.append(compute_dataset_ci(ds_name, queries))
    return results


def print_report(all_results: list[DatasetCI]) -> None:
    """Print a formatted report of all CIs."""
    for ds in all_results:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds.name} (n={ds.n_queries})")
        print(f"{'=' * 60}")
        print(f"  Hit Rate:     {ds.hit_rate}%")
        print(f"  L0 GPU P50:   {ds.l0_p50} ms")
        print(f"  Baseline P50: {ds.baseline_p50} ms")
        print(f"  Speedup:      {ds.speedup}x")
        print(f"  BLEU:         {ds.bleu}")
        print(f"  ROUGE-L:      {ds.rouge_l}")


def print_latex_table(all_results: list[DatasetCI]) -> None:
    """Print a LaTeX table row for each dataset."""
    print("\n% LaTeX table rows (for paper)")
    print("% Dataset & $n$ & Hit Rate & L0 P50 & Base P50 & Speedup \\\\")
    for ds in all_results:
        hr = ds.hit_rate
        l0 = ds.l0_p50
        bp = ds.baseline_p50
        sp = ds.speedup
        print(
            f"{ds.name} & {ds.n_queries} & "
            f"${hr.estimate:.1f}$\\% \\tiny{{[{hr.ci_lower:.1f}, {hr.ci_upper:.1f}]}} & "
            f"${l0.estimate:.0f}$\\,ms \\tiny{{[{l0.ci_lower:.0f}, {l0.ci_upper:.0f}]}} & "
            f"${bp.estimate:.0f}$\\,ms \\tiny{{[{bp.ci_lower:.0f}, {bp.ci_upper:.0f}]}} & "
            f"${sp.estimate:.1f}\\times$ \\tiny{{[{sp.ci_lower:.1f}, {sp.ci_upper:.1f}]}} \\\\"
        )


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m benchmarks.e2e.bootstrap_ci <result.json> [...]")
        print("  Computes bootstrap 95% CIs from raw query-level benchmark data.")
        sys.exit(1)

    all_results: list[DatasetCI] = []
    for path_str in sys.argv[1:]:
        path = Path(path_str)
        if not path.exists():
            print(f"Warning: {path} not found, skipping")
            continue
        all_results.extend(load_and_compute(path))

    print_report(all_results)
    print_latex_table(all_results)


if __name__ == "__main__":
    main()
