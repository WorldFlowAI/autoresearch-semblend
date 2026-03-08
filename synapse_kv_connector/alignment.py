"""In-process token alignment for SemBlend donor KV reuse.

Two alignment strategies:
  1. Chunk-level alignment (primary): Splits sequences into LMCache-sized chunks
     (256 tokens) and matches by content hash. Handles REORDER scenarios where
     identical chunks appear in different positions. O(n) time.
  2. Levenshtein alignment (fallback): Uses rapidfuzz edit distance for fine-grained
     token-level alignment when chunk matching is insufficient.

LMCache stores KV in fixed-size chunks. Chunk-level matching is the correct
abstraction because we can only load/swap entire chunks, not individual tokens.
"""
from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# LMCache chunk size — KV is stored in blocks of this many tokens
LMCACHE_CHUNK_SIZE = 256

try:
    from rapidfuzz.distance import Opcodes
    from rapidfuzz.distance.Levenshtein import opcodes as lev_opcodes
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not available - alignment disabled")


class SlotActionType(Enum):
    COPY_FROM_DONOR = "copy_from_donor"
    RECOMPUTE = "recompute"


@dataclass(frozen=True)
class SlotAction:
    action: SlotActionType
    target_pos: int
    donor_pos: int | None = None


@dataclass(frozen=True)
class AlignmentResult:
    reuse_ratio: float
    slot_actions: list[SlotAction]
    edit_distance: int
    donor_len: int
    target_len: int


def _chunk_hash(tokens: list[int]) -> str:
    """Hash a chunk of token IDs for matching."""
    return hashlib.md5(
        bytes(t & 0xFF for t in tokens) + len(tokens).to_bytes(4, "little")
    ).hexdigest()


def compute_chunk_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = LMCACHE_CHUNK_SIZE,
) -> AlignmentResult:
    """Chunk-level alignment for LMCache KV reuse.

    Splits both sequences into fixed-size chunks (matching LMCache's storage
    granularity) and matches by content. Handles REORDER scenarios where
    identical chunks appear at different positions.

    For each target chunk:
      - If an identical donor chunk exists: COPY (reuse donor KV)
      - Otherwise: RECOMPUTE

    Args:
        donor_tokens: Donor token sequence.
        target_tokens: Target token sequence.
        chunk_size: Chunk size (must match LMCache config, default 256).

    Returns:
        AlignmentResult with chunk-aligned slot actions.
    """
    # Split into chunks
    donor_chunks = []
    for i in range(0, len(donor_tokens), chunk_size):
        donor_chunks.append(donor_tokens[i:i + chunk_size])

    target_chunks = []
    for i in range(0, len(target_tokens), chunk_size):
        target_chunks.append(target_tokens[i:i + chunk_size])

    # Build donor chunk hash → (chunk_index, used) map
    # Only full-size chunks can match (partial trailing chunks can't)
    donor_hash_map: dict[str, list[int]] = {}
    for idx, chunk in enumerate(donor_chunks):
        if len(chunk) == chunk_size:
            h = _chunk_hash(chunk)
            donor_hash_map.setdefault(h, []).append(idx)

    slot_actions: list[SlotAction] = []
    num_reused = 0
    used_donor_chunks: set[int] = set()

    for t_idx, t_chunk in enumerate(target_chunks):
        t_start = t_idx * chunk_size

        if len(t_chunk) == chunk_size:
            h = _chunk_hash(t_chunk)
            candidates = donor_hash_map.get(h, [])
            # Find first unused donor chunk with this hash
            matched_d_idx = None
            for d_idx in candidates:
                if d_idx not in used_donor_chunks:
                    matched_d_idx = d_idx
                    break

            if matched_d_idx is not None:
                used_donor_chunks.add(matched_d_idx)
                d_start = matched_d_idx * chunk_size
                for i in range(chunk_size):
                    slot_actions.append(SlotAction(
                        action=SlotActionType.COPY_FROM_DONOR,
                        target_pos=t_start + i,
                        donor_pos=d_start + i,
                    ))
                    num_reused += 1
                continue

        # No match — recompute this chunk
        for i in range(len(t_chunk)):
            slot_actions.append(SlotAction(
                action=SlotActionType.RECOMPUTE,
                target_pos=t_start + i,
            ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    matched_chunks = len(used_donor_chunks)
    total_target_chunks = sum(1 for c in target_chunks if len(c) == chunk_size)
    logger.info(
        "chunk_alignment: %d/%d chunks matched (reuse=%.2f), "
        "donor_chunks=%d, target_chunks=%d",
        matched_chunks, total_target_chunks, reuse_ratio,
        len(donor_chunks), len(target_chunks),
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
    )


def compute_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """Compute alignment between donor and target sequences.

    Uses chunk-level alignment (primary) which correctly handles REORDER
    scenarios. Falls back to Levenshtein for fine-grained alignment when
    chunk matching finds no reusable chunks.

    Args:
        donor_tokens: Token IDs of the cached donor sequence.
        target_tokens: Token IDs of the new target sequence.

    Returns:
        AlignmentResult with slot actions and reuse ratio.
    """
    # Primary: chunk-level alignment (handles REORDER correctly)
    chunk_result = compute_chunk_alignment(donor_tokens, target_tokens)

    # If chunk alignment found reusable chunks, use it
    if chunk_result.reuse_ratio > 0:
        return chunk_result

    # Fallback: Levenshtein for fine-grained alignment (PARTIAL scenarios
    # where changes are within chunks, not between them)
    return _levenshtein_alignment(donor_tokens, target_tokens)


def _levenshtein_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """Fine-grained Levenshtein alignment using rapidfuzz.

    Used when chunk-level matching finds no reusable chunks (e.g., PARTIAL
    overlap where changes are distributed within chunks).
    """
    if not HAS_RAPIDFUZZ:
        return _fallback_prefix_alignment(donor_tokens, target_tokens)

    ops = lev_opcodes(donor_tokens, target_tokens)

    slot_actions: list[SlotAction] = []
    num_reused = 0

    for op in ops:
        tag = op.tag
        src_start, src_end = op.src_start, op.src_end
        dest_start, dest_end = op.dest_start, op.dest_end

        if tag == "equal":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=dest_start + i,
                    donor_pos=src_start + i,
                ))
                num_reused += 1
        elif tag == "replace":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=dest_start + i,
                ))
        elif tag == "insert":
            for i in range(dest_end - dest_start):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=dest_start + i,
                ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = sum(
        1 for sa in slot_actions if sa.action == SlotActionType.RECOMPUTE
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
    )


def compute_batch_alignment(
    candidates: list[tuple[str, list[int]]],
    target_tokens: list[int],
    min_reuse_ratio: float = 0.5,
) -> tuple[str, AlignmentResult] | None:
    """Run alignment on multiple candidates, return best.

    Args:
        candidates: List of (donor_id, donor_tokens) tuples.
        target_tokens: Target token sequence.
        min_reuse_ratio: Minimum acceptable reuse ratio.

    Returns:
        (donor_id, AlignmentResult) for best candidate, or None.
    """
    best: tuple[str, AlignmentResult] | None = None
    best_ratio = min_reuse_ratio

    for donor_id, donor_tokens in candidates:
        result = compute_alignment(donor_tokens, target_tokens)
        if result.reuse_ratio >= best_ratio:
            best_ratio = result.reuse_ratio
            best = (donor_id, result)

    return best


def _fallback_prefix_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """Fallback: simple prefix matching when rapidfuzz is unavailable."""
    prefix_len = 0
    for i in range(min(len(donor_tokens), len(target_tokens))):
        if donor_tokens[i] == target_tokens[i]:
            prefix_len += 1
        else:
            break

    slot_actions: list[SlotAction] = []
    for i in range(len(target_tokens)):
        if i < prefix_len:
            slot_actions.append(SlotAction(
                action=SlotActionType.COPY_FROM_DONOR,
                target_pos=i,
                donor_pos=i,
            ))
        else:
            slot_actions.append(SlotAction(
                action=SlotActionType.RECOMPUTE,
                target_pos=i,
            ))

    reuse_ratio = prefix_len / max(len(target_tokens), 1)
    edit_dist = len(target_tokens) - prefix_len

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=len(target_tokens),
    )
