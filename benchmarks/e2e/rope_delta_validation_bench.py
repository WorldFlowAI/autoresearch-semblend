#!/usr/bin/env python3
"""RoPE delta correction validation at Δ≠0.

Directly tests that rope_correct_k produces exact results when Δ > 0,
addressing the gap where E2E experiments always have Δ=0 (due to LMCache
chunk-boundary alignment).

Methodology:
  1. Generate K tensors at SOURCE positions using standard RoPE.
  2. Generate K tensors at TARGET positions (ground truth) using standard RoPE.
  3. Apply rope_correct_k to SOURCE K with (donor=source, target=target) positions.
  4. Compare corrected SOURCE against TARGET — L2 error should be ~0.
  5. Also report L2 error WITHOUT correction (shows what happens if Δ is ignored).

This confirms: the correction is mathematically exact, not an approximation.
Without correction, quality degrades proportionally to |Δ| and head_dim.

Usage:
    python -m benchmarks.e2e.rope_delta_validation_bench

Output:
    Per-Δ table: L2(corrected), L2(uncorrected), ratio, timing
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from synapse_kv_connector.rope_correction import rope_correct_k
except ImportError:
    print("ERROR: Cannot import rope_correct_k. Run from project root.", file=sys.stderr)
    sys.exit(1)


def apply_rope_at_positions(
    positions: torch.Tensor,
    head_dim: int,
    num_heads: int,
    rope_base: float = 10000.0,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate K tensors with RoPE applied at given positions.

    Creates random unit-norm K vectors and applies RoPE at the specified
    positions. This simulates what the model produces during prefill.

    Returns:
        K tensor [num_heads, num_positions, head_dim]
    """
    num_positions = positions.shape[0]
    torch.manual_seed(42)
    k_base = torch.randn(num_heads, num_positions, head_dim, dtype=dtype)

    inv_freq = 1.0 / (
        rope_base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
    )

    result = k_base.clone()
    for i, pos in enumerate(positions):
        theta = pos.float() * inv_freq  # [half_dim]
        cos_t = torch.cos(theta)  # [half_dim]
        sin_t = torch.sin(theta)

        k_i = k_base[:, i, :].float()  # [H, D]
        k_even = k_i[:, 0::2].clone()
        k_odd = k_i[:, 1::2].clone()
        k_i[:, 0::2] = k_even * cos_t - k_odd * sin_t
        k_i[:, 1::2] = k_even * sin_t + k_odd * cos_t
        result[:, i, :] = k_i.to(dtype)

    return result


def run_delta_validation(
    deltas: list[int],
    seq_len: int = 512,
    num_heads: int = 8,
    head_dim: int = 128,
    rope_base: float = 10000.0,
) -> list[dict]:
    """Run RoPE correction validation across a range of Δ values.

    For each Δ:
    - Source positions: [0, 1, ..., seq_len-1]
    - Target positions: [Δ, Δ+1, ..., Δ+seq_len-1]
    - Computes L2 error (corrected vs ground truth) and L2 error (uncorrected vs GT)
    """
    source_positions = torch.arange(seq_len, dtype=torch.int32)

    results = []
    for delta in deltas:
        target_positions = source_positions + delta

        # Ground truth: K at target positions
        k_target_gt = apply_rope_at_positions(
            target_positions, head_dim, num_heads, rope_base
        )

        # K at source positions (what would be stored in donor cache)
        k_source = apply_rope_at_positions(
            source_positions, head_dim, num_heads, rope_base
        )

        # Uncorrected: just use donor K as-is (wrong positions)
        l2_uncorrected = torch.norm(k_source - k_target_gt).item()

        # Apply RoPE delta correction
        t0 = time.perf_counter()
        k_corrected = rope_correct_k(
            k_tensor=k_source,
            donor_positions=source_positions,
            target_positions=target_positions,
            head_dim=head_dim,
            rope_base=rope_base,
        )
        correction_ms = (time.perf_counter() - t0) * 1000.0

        l2_corrected = torch.norm(k_corrected - k_target_gt).item()

        # Relative error (normalized by GT norm)
        gt_norm = torch.norm(k_target_gt).item()
        rel_corrected = l2_corrected / gt_norm
        rel_uncorrected = l2_uncorrected / gt_norm

        results.append({
            "delta": delta,
            "l2_corrected": l2_corrected,
            "l2_uncorrected": l2_uncorrected,
            "rel_corrected": rel_corrected,
            "rel_uncorrected": rel_uncorrected,
            "error_ratio": l2_uncorrected / max(l2_corrected, 1e-9),
            "correction_ms": correction_ms,
        })

    return results


def main() -> None:
    print("=" * 72)
    print("RoPE Delta Correction Validation (Δ≠0 micro-benchmark)")
    print(f"seq_len=512, num_heads=8, head_dim=128, rope_base=10000.0")
    print("=" * 72)

    deltas = [0, 1, 16, 32, 64, 128, 256, 512, 1024, 2048]

    results = run_delta_validation(
        deltas=deltas,
        seq_len=512,
        num_heads=8,
        head_dim=128,
    )

    # Header
    print(f"\n{'Δ (offset)':>12} {'L2(corrected)':>15} {'L2(uncorr)':>12} "
          f"{'Error reduction':>16} {'Rel error (corr)':>18} {'Correction (ms)':>16}")
    print("-" * 93)

    all_pass = True
    for r in results:
        flag = ""
        if r["rel_corrected"] > 1e-4:
            flag = " ⚠"
            all_pass = False
        print(
            f"{r['delta']:>12} {r['l2_corrected']:>15.6f} {r['l2_uncorrected']:>12.4f} "
            f"{r['error_ratio']:>15.1f}x {r['rel_corrected']:>17.2e} "
            f"{r['correction_ms']:>15.3f}{flag}"
        )

    print()
    if all_pass:
        print("✓ PASS: Relative L2 error < 1e-4 at all Δ values.")
        print("  RoPE delta correction is mathematically exact (float32 precision).")
    else:
        print("✗ FAIL: Some Δ values exceed the 1e-4 relative error threshold.")

    # Summary stats
    zero_row = next(r for r in results if r["delta"] == 0)
    max_delta_row = max(results, key=lambda r: r["delta"])
    print()
    print(f"Summary:")
    print(f"  Δ=0:    L2(corrected)={zero_row['l2_corrected']:.2e} (identity)")
    print(f"  Δ={max_delta_row['delta']}: L2(corrected)={max_delta_row['l2_corrected']:.2e}, "
          f"L2(uncorr)={max_delta_row['l2_uncorrected']:.4f} "
          f"({max_delta_row['error_ratio']:.0f}× worse without correction)")
    print(f"  Average correction time: "
          f"{sum(r['correction_ms'] for r in results) / len(results):.3f} ms "
          f"(seq_len=512, 8 heads, 128 dim)")

    # Also test larger sequence
    print()
    print("-" * 72)
    print("8K token validation (matching paper's primary benchmark scenario):")
    results_8k = run_delta_validation(
        deltas=[0, 256, 512, 1024, 2048],
        seq_len=8192,
        num_heads=8,
        head_dim=128,
    )
    print(f"{'Δ':>8} {'L2(corrected)':>15} {'L2(uncorr)':>12} {'Reduction':>12} {'ms':>8}")
    for r in results_8k:
        print(f"{r['delta']:>8} {r['l2_corrected']:>15.6f} "
              f"{r['l2_uncorrected']:>12.4f} {r['error_ratio']:>11.0f}x "
              f"{r['correction_ms']:>7.2f}")

    # Return non-zero if validation fails
    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
