"""In-process token alignment for SemBlend donor KV reuse.

Two alignment strategies:
  1. Chunk-level alignment (primary): Splits sequences into engine-sized chunks
     and matches by content hash. Handles REORDER scenarios where identical
     chunks appear in different positions. O(n) time.
  2. Levenshtein alignment (fallback): Uses rapidfuzz edit distance for
     fine-grained token-level alignment when chunk matching is insufficient.

The chunk size is parameterized to support different backends:
  - LMCache (vLLM): 256 tokens per chunk
  - TRT-LLM: 128 tokens per chunk (configurable power-of-2)

Context gate (enabled by default): rejects isolated chunk matches where
no adjacent chunk also matches. Prevents semantic staleness from token-
identical chunks at wrong semantic positions.
"""
from __future__ import annotations

import hashlib
import logging
import os
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

# Default chunk size — overridden by backend's get_kv_block_size()
DEFAULT_CHUNK_SIZE = 256

# Context gate: reject isolated chunk matches where no adjacent chunk also
# matches. Prevents semantic staleness (PPL=1.27 outlier) from token-identical
# chunks appearing at wrong semantic positions in the document.
# Disable with SEMBLEND_CONTEXT_GATE=0 for backward compatibility.
_CONTEXT_GATE_ENABLED = os.environ.get("SEMBLEND_CONTEXT_GATE", "1") != "0"

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
    """Hash a chunk of token IDs for matching.

    Uses full 4-byte representation of each token ID to avoid collisions
    with large vocabularies (e.g., Qwen 152K vocab).
    """
    import struct
    data = struct.pack(f"<{len(tokens)}I", *tokens)
    return hashlib.md5(data).hexdigest()


def compute_chunk_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    context_gate: bool | None = None,
) -> AlignmentResult:
    """Chunk-level alignment for KV reuse.

    Splits both sequences into fixed-size chunks (matching the engine's KV
    storage granularity) and matches by content hash. Handles REORDER scenarios
    where identical chunks appear at different positions.

    Two-phase approach:
      Phase 1: Identify all hash matches between donor and target chunks.
      Phase 2 (context gate): Reject isolated matches — a chunk match is
        accepted only if at least one adjacent target chunk also matches.
        This prevents semantic staleness where a token-identical chunk from
        a different document position gets incorrectly reused.

    Args:
        donor_tokens: Donor token sequence.
        target_tokens: Target token sequence.
        chunk_size: Chunk size in tokens (must match engine block size).
        context_gate: Override for context gate. None uses env var default.

    Returns:
        AlignmentResult with chunk-aligned slot actions.
    """
    use_context_gate = (
        context_gate if context_gate is not None else _CONTEXT_GATE_ENABLED
    )

    # Split into chunks
    donor_chunks = []
    for i in range(0, len(donor_tokens), chunk_size):
        donor_chunks.append(donor_tokens[i:i + chunk_size])

    target_chunks = []
    for i in range(0, len(target_tokens), chunk_size):
        target_chunks.append(target_tokens[i:i + chunk_size])

    # Build donor chunk hash → list of chunk indices
    # Only full-size chunks can match (partial trailing chunks can't)
    donor_hash_map: dict[str, list[int]] = {}
    for idx, chunk in enumerate(donor_chunks):
        if len(chunk) == chunk_size:
            h = _chunk_hash(chunk)
            donor_hash_map.setdefault(h, []).append(idx)

    # Phase 1: identify all hash matches
    chunk_matches: dict[int, int] = {}  # target_idx -> donor_idx
    used_donor_chunks: set[int] = set()

    for t_idx, t_chunk in enumerate(target_chunks):
        if len(t_chunk) == chunk_size:
            h = _chunk_hash(t_chunk)
            candidates = donor_hash_map.get(h, [])
            for d_idx in candidates:
                if d_idx not in used_donor_chunks:
                    chunk_matches[t_idx] = d_idx
                    used_donor_chunks.add(d_idx)
                    break

    # Phase 2: context gate — reject isolated matches
    if use_context_gate and chunk_matches:
        matched_set = set(chunk_matches.keys())
        validated: dict[int, int] = {}
        rejected_count = 0
        for t_idx, d_idx in chunk_matches.items():
            has_neighbor = (
                (t_idx - 1) in matched_set
                or (t_idx + 1) in matched_set
            )
            if has_neighbor:
                validated[t_idx] = d_idx
            else:
                rejected_count += 1
                logger.info(
                    "context_gate: rejected isolated chunk match "
                    "target[%d] -> donor[%d]",
                    t_idx, d_idx,
                )
        if rejected_count > 0:
            logger.info(
                "context_gate: rejected %d/%d isolated chunk matches",
                rejected_count, len(chunk_matches),
            )
        chunk_matches = validated

    # Phase 3: build slot actions from validated matches
    slot_actions: list[SlotAction] = []
    num_reused = 0

    for t_idx, t_chunk in enumerate(target_chunks):
        t_start = t_idx * chunk_size
        d_idx = chunk_matches.get(t_idx)

        if d_idx is not None:
            d_start = d_idx * chunk_size
            for i in range(chunk_size):
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=t_start + i,
                    donor_pos=d_start + i,
                ))
                num_reused += 1
        else:
            for i in range(len(t_chunk)):
                slot_actions.append(SlotAction(
                    action=SlotActionType.RECOMPUTE,
                    target_pos=t_start + i,
                ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    matched_chunks = len(chunk_matches)
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


def estimate_reuse_ratio(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> float:
    """Fast O(n) reuse ratio estimation without building slot_actions.

    Used by donor_store candidate scoring to avoid creating thousands of
    SlotAction objects per candidate. Only the winning candidate needs
    the full alignment with slot_actions.

    Strategy: chunk alignment count, then token-set count as fallback.
    """
    import struct

    # Try chunk-level matching first
    donor_hashes: dict[str, int] = {}
    for i in range(0, len(donor_tokens), chunk_size):
        chunk = donor_tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:
            data = struct.pack(f"<{len(chunk)}I", *chunk)
            h = hashlib.md5(data).hexdigest()
            donor_hashes[h] = donor_hashes.get(h, 0) + 1

    matched_tokens = 0
    for i in range(0, len(target_tokens), chunk_size):
        chunk = target_tokens[i:i + chunk_size]
        if len(chunk) == chunk_size:
            data = struct.pack(f"<{len(chunk)}I", *chunk)
            h = hashlib.md5(data).hexdigest()
            if donor_hashes.get(h, 0) > 0:
                donor_hashes[h] -= 1
                matched_tokens += chunk_size

    if matched_tokens > 0:
        return matched_tokens / max(len(target_tokens), 1)

    # Fallback: token-set overlap count (O(n), no SlotAction creation)
    from collections import Counter
    donor_counts = Counter(donor_tokens)
    for tok in target_tokens:
        if donor_counts.get(tok, 0) > 0:
            donor_counts[tok] -= 1
            matched_tokens += 1

    return matched_tokens / max(len(target_tokens), 1)


def compute_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> AlignmentResult:
    """Compute alignment between donor and target sequences.

    Uses chunk-level alignment (primary) which correctly handles REORDER
    scenarios. Falls back to Levenshtein for fine-grained alignment when
    chunk matching finds no reusable chunks.

    Args:
        donor_tokens: Token IDs of the cached donor sequence.
        target_tokens: Token IDs of the new target sequence.
        chunk_size: Chunk size in tokens (engine block size).

    Returns:
        AlignmentResult with slot actions and reuse ratio.
    """
    # Primary: chunk-level alignment (handles REORDER correctly)
    chunk_result = compute_chunk_alignment(
        donor_tokens, target_tokens, chunk_size=chunk_size,
    )

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

    For sequences > 4096 tokens, falls back to O(n) token-set alignment
    to avoid O(n*m) blowup (~3s at 16K tokens).
    """
    if not HAS_RAPIDFUZZ:
        return _token_set_alignment(donor_tokens, target_tokens)

    # Cap Levenshtein to avoid O(n*m) blowup on long sequences.
    # At 16K tokens, lev_opcodes takes ~3 seconds — unacceptable for the
    # hot path. Use token-set alignment (O(n)) for long sequences.
    MAX_LEVENSHTEIN_TOKENS = 4096
    if (
        len(donor_tokens) > MAX_LEVENSHTEIN_TOKENS
        or len(target_tokens) > MAX_LEVENSHTEIN_TOKENS
    ):
        return _token_set_alignment(donor_tokens, target_tokens)

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
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> tuple[str, AlignmentResult] | None:
    """Run alignment on multiple candidates, return best.

    Args:
        candidates: List of (donor_id, donor_tokens) tuples.
        target_tokens: Target token sequence.
        min_reuse_ratio: Minimum acceptable reuse ratio.
        chunk_size: Chunk size in tokens (engine block size).

    Returns:
        (donor_id, AlignmentResult) for best candidate, or None.
    """
    best: tuple[str, AlignmentResult] | None = None
    best_ratio = min_reuse_ratio

    for donor_id, donor_tokens in candidates:
        result = compute_alignment(
            donor_tokens, target_tokens, chunk_size=chunk_size,
        )
        if result.reuse_ratio >= best_ratio:
            best_ratio = result.reuse_ratio
            best = (donor_id, result)

    return best


def _token_set_alignment(
    donor_tokens: list[int],
    target_tokens: list[int],
) -> AlignmentResult:
    """O(n) token-set alignment for long sequences.

    For sequences too long for Levenshtein (>4096 tokens), matches each
    target token to a donor token with the same ID using a hash map.

    Handles REORDER naturally: same tokens at different positions produce
    COPY_FROM_DONOR with donor_pos != target_pos, feeding into RoPE
    correction. For PARTIAL overlap, correctly identifies which target
    tokens have matching donor tokens and which need recomputation.

    Less precise than Levenshtein (ignores order/context) but O(n)
    vs O(n^2) and sufficient for the KV reuse decision gate.
    """
    from collections import defaultdict

    # Build donor token → list of available positions (FIFO order)
    donor_positions: dict[int, list[int]] = defaultdict(list)
    for pos, tok in enumerate(donor_tokens):
        donor_positions[tok].append(pos)

    # Index into each token's position list (avoids set removal overhead)
    donor_pos_idx: dict[int, int] = defaultdict(int)

    slot_actions: list[SlotAction] = []
    num_reused = 0

    for t_pos, t_tok in enumerate(target_tokens):
        candidates = donor_positions.get(t_tok)
        if candidates is not None:
            idx = donor_pos_idx[t_tok]
            if idx < len(candidates):
                d_pos = candidates[idx]
                donor_pos_idx[t_tok] = idx + 1
                slot_actions.append(SlotAction(
                    action=SlotActionType.COPY_FROM_DONOR,
                    target_pos=t_pos,
                    donor_pos=d_pos,
                ))
                num_reused += 1
                continue

        slot_actions.append(SlotAction(
            action=SlotActionType.RECOMPUTE,
            target_pos=t_pos,
        ))

    target_len = len(target_tokens)
    reuse_ratio = num_reused / max(target_len, 1)
    edit_dist = target_len - num_reused

    logger.info(
        "token_set_alignment: reuse=%.2f (%d/%d tokens), "
        "donor_len=%d, target_len=%d",
        reuse_ratio, num_reused, target_len,
        len(donor_tokens), target_len,
    )

    return AlignmentResult(
        reuse_ratio=reuse_ratio,
        slot_actions=slot_actions,
        edit_distance=edit_dist,
        donor_len=len(donor_tokens),
        target_len=target_len,
    )


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
