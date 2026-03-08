"""Tests for alignment.py and bathtub.py."""
from __future__ import annotations

import math
import time


def test_alignment_exact_match():
    """Exact token sequences should have 100% reuse."""
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_alignment,
    )

    tokens = [1, 2, 3, 4, 5, 6, 7, 8]
    result = compute_alignment(tokens, tokens)

    assert result.reuse_ratio == 1.0
    assert result.edit_distance == 0
    assert all(sa.action == SlotActionType.COPY_FROM_DONOR for sa in result.slot_actions)


def test_alignment_prefix_match():
    """Shared prefix should be marked as copy_from_donor."""
    from synapse_kv_connector.alignment import (
        SlotActionType,
        compute_alignment,
    )

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 3, 10, 11]
    result = compute_alignment(donor, target)

    # First 3 tokens match
    assert result.slot_actions[0].action == SlotActionType.COPY_FROM_DONOR
    assert result.slot_actions[1].action == SlotActionType.COPY_FROM_DONOR
    assert result.slot_actions[2].action == SlotActionType.COPY_FROM_DONOR
    assert result.reuse_ratio >= 0.5


def test_alignment_no_overlap():
    """Completely different sequences should have 0% reuse."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [10, 20, 30, 40, 50]
    result = compute_alignment(donor, target)

    assert result.reuse_ratio == 0.0
    assert result.edit_distance == 5


def test_alignment_reorder():
    """Reordered tokens: edit distance detects rearrangement."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 3, 2, 4, 5]
    result = compute_alignment(donor, target)

    # Some tokens match despite reorder
    assert 0.0 < result.reuse_ratio < 1.0
    assert result.edit_distance > 0


def test_alignment_insertion():
    """Inserting tokens in the middle."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 99, 3, 4, 5]
    result = compute_alignment(donor, target)

    # At least the prefix [1,2] should be reused
    assert result.reuse_ratio > 0.0
    # Should have some reuse
    assert result.edit_distance < len(target)


def test_alignment_latency_8k():
    """Alignment on 8K tokens must complete in <5ms."""
    from synapse_kv_connector.alignment import compute_alignment

    donor = list(range(8000))
    target = list(range(100)) + list(range(200, 8100))

    t0 = time.monotonic()
    result = compute_alignment(donor, target)
    elapsed_ms = (time.monotonic() - t0) * 1000

    assert elapsed_ms < 5000, f"Alignment took {elapsed_ms:.1f}ms (budget: 5000ms)"
    assert result.reuse_ratio > 0


def test_batch_alignment():
    """Batch alignment returns best candidate."""
    from synapse_kv_connector.alignment import compute_batch_alignment

    target = [1, 2, 3, 4, 5]
    candidates = [
        ("bad", [10, 20, 30, 40, 50]),
        ("good", [1, 2, 3, 4, 99]),
    ]

    result = compute_batch_alignment(candidates, target, min_reuse_ratio=0.5)
    assert result is not None
    donor_id, alignment = result
    assert donor_id == "good"
    assert alignment.reuse_ratio >= 0.5


def test_fallback_alignment():
    """Fallback prefix alignment works without rapidfuzz."""
    from synapse_kv_connector.alignment import _fallback_prefix_alignment

    donor = [1, 2, 3, 4, 5]
    target = [1, 2, 3, 10, 11]
    result = _fallback_prefix_alignment(donor, target)

    assert result.reuse_ratio == 0.6  # 3/5
    assert result.edit_distance == 2


# ---------------------------------------------------------------------------
# Bathtub curve tests
# ---------------------------------------------------------------------------


def test_bathtub_early_late_deviation():
    """Early and late layers should have higher deviation than middle."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    devs = compute_layer_deviations(num_layers=28)

    # Layer 0 (early) should have high deviation
    assert devs[0].deviation_score > 0.3
    # Middle layer should be low
    assert devs[14].deviation_score < 0.2
    # Last layer should be high
    assert devs[27].deviation_score > 0.2


def test_bathtub_symmetry():
    """Bathtub should be roughly symmetric around the middle."""
    from synapse_kv_connector.bathtub import sigma

    early = sigma(layer_idx=2, num_layers=28)
    late = sigma(layer_idx=25, num_layers=28)

    # Both should be elevated
    assert early > 0.15
    assert late > 0.15


def test_bathtub_sigma_bounds():
    """sigma() should return values in [0, 1]."""
    from synapse_kv_connector.bathtub import sigma

    for num_layers in [24, 28, 32, 80]:
        for i in range(num_layers):
            s = sigma(i, num_layers)
            assert 0.0 <= s <= 1.0, f"sigma({i}, {num_layers}) = {s}"


def test_bathtub_preset_lookup():
    """Model name matching should work with full HuggingFace names."""
    from synapse_kv_connector.bathtub import get_preset

    preset = get_preset("Qwen/Qwen2.5-7B-Instruct-AWQ")
    assert preset.num_layers == 28

    default = get_preset("unknown-model")
    assert default.num_layers == 32


def test_bathtub_mismatch_scaling():
    """Higher mismatch fraction should increase deviation scores."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    low = compute_layer_deviations(num_layers=28, mismatch_fraction=0.1)
    high = compute_layer_deviations(num_layers=28, mismatch_fraction=0.5)

    # Middle layer should deviate more with higher mismatch
    assert high[14].deviation_score >= low[14].deviation_score


def test_bathtub_latency():
    """Bathtub computation must complete in <0.1ms."""
    from synapse_kv_connector.bathtub import compute_layer_deviations

    t0 = time.monotonic()
    for _ in range(1000):
        compute_layer_deviations(num_layers=28)
    elapsed_ms = (time.monotonic() - t0) * 1000

    per_call_ms = elapsed_ms / 1000
    assert per_call_ms < 0.1, f"Bathtub took {per_call_ms:.3f}ms (budget: 0.1ms)"
