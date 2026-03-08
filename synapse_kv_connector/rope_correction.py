"""RoPE delta correction for non-contiguous KV cache reuse.

Core innovation: when reusing donor KV at a different position, apply a
relative RoPE rotation to correct the position encoding baked into K:

    K_corrected = RoPE(target_pos - donor_pos) × K_donor

This is mathematically exact — RoPE is a rotation matrix, and rotations
compose: RoPE(a) × RoPE⁻¹(b) = RoPE(a - b). The correction costs O(d)
per token per head per layer, ~7μs total for 8K tokens on A10G.

V cache has NO position encoding and can be rearranged freely.

References:
    - Su et al., 2021 "RoFormer: Enhanced Transformer with Rotary Position
      Embedding" — defines RoPE as paired 2D rotations
    - SemShareKV (Zhao & Mastorakis, AACL 2025) — demonstrated cross-prompt
      KV reuse with position awareness; uses approximate correction (first-layer
      full recompute). We improve with exact delta correction.

Usage:
    # Correct K tensors when moving from donor_pos to target_pos
    rope_correct_k(
        k_tensor,              # [num_heads, seq_len, head_dim]
        donor_positions,       # [num_pairs] int32
        target_positions,      # [num_pairs] int32
        head_dim=128,
        rope_base=10000.0,
    )
"""
from __future__ import annotations

import logging
import math

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False


# ---------------------------------------------------------------------------
# Triton kernel: fused RoPE correction + scatter to paged cache
# ---------------------------------------------------------------------------

if HAS_TRITON:

    @triton.jit
    def _rope_correct_scatter_paged_kernel(
        # Pointers
        kv_cache_ptr,
        donor_kv_ptr,
        block_table_ptr,
        donor_pos_ptr,
        target_pos_ptr,
        inv_freq_ptr,
        # Dimensions
        num_pairs: tl.constexpr,
        num_heads: tl.constexpr,
        block_size: tl.constexpr,
        head_dim: tl.constexpr,
        half_dim: tl.constexpr,
        donor_seq_len: tl.constexpr,
        num_blocks_total: tl.constexpr,
        num_blocks_per_seq: tl.constexpr,
        # Strides for kv_cache: [num_blocks, 2, num_heads, block_size, head_dim]
        kv_stride_block,
        kv_stride_kv,
        kv_stride_head,
        kv_stride_pos,
        kv_stride_dim,
        # Strides for donor: [2, num_heads, donor_seq_len, head_dim]
        d_stride_kv,
        d_stride_head,
        d_stride_seq,
        d_stride_dim,
        # Block
        BLOCK_HALF: tl.constexpr,
    ):
        """Fused kernel: load donor KV, apply RoPE correction to K, scatter
        to paged cache. V is copied without correction.

        Grid: (num_pairs, 2 * num_heads)
        - kv_idx=0 (K): apply RoPE(delta) correction
        - kv_idx=1 (V): direct copy (no position encoding)
        """
        pair_idx = tl.program_id(0)
        compound_idx = tl.program_id(1)

        kv_idx = compound_idx // num_heads  # 0=K, 1=V
        head_idx = compound_idx % num_heads

        # Load positions
        d_pos = tl.load(donor_pos_ptr + pair_idx)
        t_pos = tl.load(target_pos_ptr + pair_idx)

        # Bounds check
        if d_pos >= donor_seq_len:
            return

        logical_block = t_pos // block_size
        block_offset = t_pos % block_size

        if logical_block >= num_blocks_per_seq:
            return

        physical_block = tl.load(block_table_ptr + logical_block)
        if physical_block >= num_blocks_total:
            return

        # Compute base offsets
        donor_base = (
            kv_idx * d_stride_kv
            + head_idx * d_stride_head
            + d_pos * d_stride_seq
        )
        target_base = (
            physical_block * kv_stride_block
            + kv_idx * kv_stride_kv
            + head_idx * kv_stride_head
            + block_offset * kv_stride_pos
        )

        if kv_idx == 1:
            # V: direct copy (no position encoding)
            offsets = tl.arange(0, BLOCK_HALF * 2)
            mask = offsets < head_dim
            vals = tl.load(
                donor_kv_ptr + donor_base + offsets * d_stride_dim, mask=mask
            )
            tl.store(
                kv_cache_ptr + target_base + offsets * kv_stride_dim,
                vals,
                mask=mask,
            )
        else:
            # K: apply RoPE(target_pos - donor_pos) correction
            delta = (t_pos - d_pos).to(tl.float32)

            half_offsets = tl.arange(0, BLOCK_HALF)
            half_mask = half_offsets < half_dim

            # Load inv_freq for this pair of dimensions
            freq = tl.load(inv_freq_ptr + half_offsets, mask=half_mask)
            theta = delta * freq
            cos_val = tl.cos(theta)
            sin_val = tl.sin(theta)

            # Load K[2i] and K[2i+1] (interleaved real/imaginary)
            even_offsets = half_offsets * 2
            odd_offsets = half_offsets * 2 + 1

            k_even = tl.load(
                donor_kv_ptr + donor_base + even_offsets * d_stride_dim,
                mask=half_mask,
            ).to(tl.float32)
            k_odd = tl.load(
                donor_kv_ptr + donor_base + odd_offsets * d_stride_dim,
                mask=half_mask,
            ).to(tl.float32)

            # RoPE rotation: [cos -sin; sin cos] × [k_even; k_odd]
            k_even_new = k_even * cos_val - k_odd * sin_val
            k_odd_new = k_even * sin_val + k_odd * cos_val

            # Store corrected K
            tl.store(
                kv_cache_ptr + target_base + even_offsets * kv_stride_dim,
                k_even_new.to(tl.float16),
                mask=half_mask,
            )
            tl.store(
                kv_cache_ptr + target_base + odd_offsets * kv_stride_dim,
                k_odd_new.to(tl.float16),
                mask=half_mask,
            )


def rope_correct_scatter_paged(
    kv_cache: torch.Tensor,
    donor_kv: torch.Tensor,
    block_table: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    rope_base: float = 10000.0,
    head_dim: int | None = None,
) -> torch.Tensor:
    """Scatter donor KV into paged cache with RoPE correction on K.

    For each (donor_pos, target_pos) pair:
    - K is corrected: K_new = RoPE(target - donor) × K_donor
    - V is copied directly (no position encoding)

    This enables non-contiguous KV reuse: donor KV from any position can be
    injected at any target position with mathematically exact correction.

    Args:
        kv_cache: Paged KV cache [num_blocks, 2, num_heads, block_size, head_dim].
        donor_kv: Donor KV tensor [2, num_heads, donor_seq_len, head_dim].
        block_table: Block table [num_blocks_per_seq] int32.
        donor_positions: Source positions in donor sequence [num_pairs] int32.
        target_positions: Target positions in target sequence [num_pairs] int32.
        rope_base: RoPE frequency base (10000.0 for most models).
        head_dim: Head dimension (inferred from kv_cache if None).

    Returns:
        Updated kv_cache (modified in-place).
    """
    num_pairs = donor_positions.shape[0]
    if num_pairs == 0:
        return kv_cache

    num_blocks_total, _, num_heads, block_size, hd = kv_cache.shape
    if head_dim is None:
        head_dim = hd
    half_dim = head_dim // 2
    _, _, donor_seq_len, _ = donor_kv.shape
    num_blocks_per_seq = block_table.shape[0]

    # Compute inv_freq: 1 / (base ^ (2i / dim)) for i in [0, dim/2)
    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=kv_cache.device) / head_dim)
    )

    if HAS_TRITON and kv_cache.is_cuda:
        block_half = triton.next_power_of_2(half_dim)
        grid = (num_pairs, 2 * num_heads)

        _rope_correct_scatter_paged_kernel[grid](
            kv_cache,
            donor_kv,
            block_table,
            donor_positions,
            target_positions,
            inv_freq,
            num_pairs=num_pairs,
            num_heads=num_heads,
            block_size=block_size,
            head_dim=head_dim,
            half_dim=half_dim,
            donor_seq_len=donor_seq_len,
            num_blocks_total=num_blocks_total,
            num_blocks_per_seq=num_blocks_per_seq,
            kv_stride_block=kv_cache.stride(0),
            kv_stride_kv=kv_cache.stride(1),
            kv_stride_head=kv_cache.stride(2),
            kv_stride_pos=kv_cache.stride(3),
            kv_stride_dim=kv_cache.stride(4),
            d_stride_kv=donor_kv.stride(0),
            d_stride_head=donor_kv.stride(1),
            d_stride_seq=donor_kv.stride(2),
            d_stride_dim=donor_kv.stride(3),
            BLOCK_HALF=block_half,
        )
    else:
        _rope_correct_scatter_paged_cpu(
            kv_cache, donor_kv, block_table,
            donor_positions, target_positions, inv_freq,
            block_size, head_dim,
        )

    return kv_cache


def _rope_correct_scatter_paged_cpu(
    kv_cache: torch.Tensor,
    donor_kv: torch.Tensor,
    block_table: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    inv_freq: torch.Tensor,
    block_size: int,
    head_dim: int,
) -> None:
    """CPU fallback for RoPE-corrected scatter."""
    for i in range(donor_positions.shape[0]):
        d_pos = int(donor_positions[i])
        t_pos = int(target_positions[i])

        logical_block = t_pos // block_size
        offset = t_pos % block_size
        physical_block = int(block_table[logical_block])

        delta = float(t_pos - d_pos)

        # V: direct copy
        kv_cache[physical_block, 1, :, offset, :] = donor_kv[1, :, d_pos, :]

        # K: RoPE correction
        k = donor_kv[0, :, d_pos, :].float()  # [num_heads, head_dim]

        # Apply RoPE(delta) to each dimension pair
        theta = delta * inv_freq  # [head_dim // 2]
        cos_vals = torch.cos(theta)
        sin_vals = torch.sin(theta)

        k_even = k[:, 0::2]  # [num_heads, head_dim//2]
        k_odd = k[:, 1::2]

        k_corrected_even = k_even * cos_vals - k_odd * sin_vals
        k_corrected_odd = k_even * sin_vals + k_odd * cos_vals

        # Interleave back
        k_corrected = torch.zeros_like(k)
        k_corrected[:, 0::2] = k_corrected_even
        k_corrected[:, 1::2] = k_corrected_odd

        kv_cache[physical_block, 0, :, offset, :] = k_corrected.to(
            kv_cache.dtype
        )


# ---------------------------------------------------------------------------
# Standalone RoPE correction (for flat KV cache or testing)
# ---------------------------------------------------------------------------


def rope_correct_k(
    k_tensor: torch.Tensor,
    donor_positions: torch.Tensor,
    target_positions: torch.Tensor,
    head_dim: int = 128,
    rope_base: float = 10000.0,
) -> torch.Tensor:
    """Apply RoPE delta correction to K tensor.

    For each position pair (donor_pos, target_pos), applies:
        K_corrected = RoPE(target - donor) × K_donor

    This corrects the position encoding baked into cached K tensors,
    enabling non-contiguous KV reuse with mathematical exactness.

    Args:
        k_tensor: K cache [num_heads, seq_len, head_dim] or
                  [num_layers, num_heads, seq_len, head_dim].
        donor_positions: Donor positions [num_pairs] int32.
        target_positions: Target positions [num_pairs] int32.
        head_dim: Dimension per head (default 128 for Qwen2.5).
        rope_base: RoPE base frequency (default 10000.0).

    Returns:
        Corrected K tensor (new tensor, original not modified).
    """
    device = k_tensor.device
    dtype = k_tensor.dtype

    inv_freq = 1.0 / (
        rope_base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    deltas = (target_positions - donor_positions).float()  # [num_pairs]

    # theta[i, d] = delta[i] * inv_freq[d]
    theta = deltas.unsqueeze(-1) * inv_freq.unsqueeze(0)  # [num_pairs, half_dim]
    cos_vals = torch.cos(theta)  # [num_pairs, half_dim]
    sin_vals = torch.sin(theta)

    result = k_tensor.clone()
    has_layer_dim = k_tensor.dim() == 4

    for i in range(donor_positions.shape[0]):
        d_pos = int(donor_positions[i])
        cos_i = cos_vals[i]  # [half_dim]
        sin_i = sin_vals[i]

        if has_layer_dim:
            k = result[:, :, d_pos, :].float()  # [L, H, D]
            k_even = k[:, :, 0::2].clone()
            k_odd = k[:, :, 1::2].clone()
            k[:, :, 0::2] = k_even * cos_i - k_odd * sin_i
            k[:, :, 1::2] = k_even * sin_i + k_odd * cos_i
            result[:, :, d_pos, :] = k.to(dtype)
        else:
            k = result[:, d_pos, :].float()  # [H, D]
            k_even = k[:, 0::2].clone()
            k_odd = k[:, 1::2].clone()
            k[:, 0::2] = k_even * cos_i - k_odd * sin_i
            k[:, 1::2] = k_even * sin_i + k_odd * cos_i
            result[:, d_pos, :] = k.to(dtype)

    return result


# ---------------------------------------------------------------------------
# In-place paged cache permutation with RoPE correction
# ---------------------------------------------------------------------------


def permute_paged_kv_with_rope(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    permutation: list[tuple[int, int]],
    rope_base: float = 10000.0,
) -> None:
    """Permute KV entries within a paged cache, applying RoPE correction to K.

    After LMCache loads donor KV in donor order, this function rearranges
    the KV to target order with exact position correction.

    Args:
        kv_cache: Paged cache [num_blocks, 2, num_heads, block_size, head_dim].
        block_table: Block table [num_blocks_per_seq] int32.
        permutation: List of (source_logical_pos, target_logical_pos) pairs.
        rope_base: RoPE base frequency.
    """
    if not permutation:
        return

    _, _, num_heads, block_size, head_dim = kv_cache.shape
    inv_freq = 1.0 / (
        rope_base
        ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=kv_cache.device)
            / head_dim
        )
    )

    # Read all source values first (avoid overwrite-before-read)
    src_kv = {}
    for src_pos, _ in permutation:
        if src_pos not in src_kv:
            sb = int(block_table[src_pos // block_size])
            so = src_pos % block_size
            src_kv[src_pos] = kv_cache[sb, :, :, so, :].clone()

    # Write to target positions with RoPE correction on K
    for src_pos, tgt_pos in permutation:
        tb = int(block_table[tgt_pos // block_size])
        to_ = tgt_pos % block_size
        kv = src_kv[src_pos]  # [2, num_heads, head_dim]

        # V (index 1): direct copy
        kv_cache[tb, 1, :, to_, :] = kv[1]

        # K (index 0): RoPE correction
        delta = float(tgt_pos - src_pos)
        if abs(delta) < 0.5:
            # Same position, no correction needed
            kv_cache[tb, 0, :, to_, :] = kv[0]
        else:
            k = kv[0].float()  # [num_heads, head_dim]
            theta = delta * inv_freq
            cos_v = torch.cos(theta)
            sin_v = torch.sin(theta)

            k_even = k[:, 0::2].clone()
            k_odd = k[:, 1::2].clone()
            k[:, 0::2] = k_even * cos_v - k_odd * sin_v
            k[:, 1::2] = k_even * sin_v + k_odd * cos_v

            kv_cache[tb, 0, :, to_, :] = k.to(kv_cache.dtype)
