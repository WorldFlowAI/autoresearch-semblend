"""Latency metrics and percentile computation."""

from __future__ import annotations

from dataclasses import dataclass, field


def percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a sorted list.

    Args:
        values: List of numeric values.
        p: Percentile (0-100).

    Returns:
        The p-th percentile value, or 0.0 if list is empty.
    """
    if not values:
        return 0.0
    sorted_vals = sorted(values)
    idx = (p / 100.0) * (len(sorted_vals) - 1)
    lower = int(idx)
    upper = min(lower + 1, len(sorted_vals) - 1)
    frac = idx - lower
    return sorted_vals[lower] * (1 - frac) + sorted_vals[upper] * frac


@dataclass
class TierStats:
    """Latency statistics for a single cache tier."""

    tier: str
    count: int = 0
    ttft_values: list[float] = field(default_factory=list)

    @property
    def p50(self) -> float:
        return percentile(self.ttft_values, 50)

    @property
    def p95(self) -> float:
        return percentile(self.ttft_values, 95)

    @property
    def p99(self) -> float:
        return percentile(self.ttft_values, 99)

    @property
    def mean(self) -> float:
        if not self.ttft_values:
            return 0.0
        return sum(self.ttft_values) / len(self.ttft_values)

    def add(self, ttft_ms: float) -> None:
        self.ttft_values.append(ttft_ms)
        self.count += 1

    def summary(self) -> dict[str, float]:
        return {
            "count": self.count,
            "p50_ms": self.p50,
            "p95_ms": self.p95,
            "p99_ms": self.p99,
            "mean_ms": self.mean,
        }


@dataclass
class LatencyReport:
    """Aggregate latency report across all tiers."""

    tiers: dict[str, TierStats] = field(default_factory=dict)
    baseline_ttft: list[float] = field(default_factory=list)

    def add_result(self, tier: str, ttft_ms: float) -> None:
        if tier not in self.tiers:
            self.tiers[tier] = TierStats(tier=tier)
        self.tiers[tier].add(ttft_ms)

    def add_baseline(self, ttft_ms: float) -> None:
        self.baseline_ttft.append(ttft_ms)

    @property
    def baseline_p50(self) -> float:
        return percentile(self.baseline_ttft, 50)

    @property
    def baseline_p99(self) -> float:
        return percentile(self.baseline_ttft, 99)

    def speedup(self, tier: str) -> float:
        """Compute TTFT speedup for a tier vs baseline."""
        if tier not in self.tiers or not self.baseline_ttft:
            return 0.0
        tier_p50 = self.tiers[tier].p50
        base_p50 = self.baseline_p50
        if tier_p50 <= 0:
            return 0.0
        return base_p50 / tier_p50

    def summary(self) -> dict[str, object]:
        return {
            "tiers": {
                name: stats.summary()
                for name, stats in self.tiers.items()
            },
            "baseline": {
                "count": len(self.baseline_ttft),
                "p50_ms": self.baseline_p50,
                "p99_ms": self.baseline_p99,
            },
            "speedups": {
                name: self.speedup(name) for name in self.tiers
            },
        }
