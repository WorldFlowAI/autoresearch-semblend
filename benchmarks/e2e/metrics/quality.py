"""Quality metrics: BLEU and ROUGE-L for response comparison."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

_nltk_initialized = False


def _ensure_nltk() -> None:
    global _nltk_initialized
    if not _nltk_initialized:
        import nltk
        nltk.download("punkt_tab", quiet=True)
        _nltk_initialized = True


def bleu_score(hypothesis: str, reference: str) -> float:
    """Compute smoothed sentence-level BLEU score.

    Uses NLTK's smoothing method 1 to handle short sentences.

    Args:
        hypothesis: Generated (cache-assisted) response.
        reference: Baseline (no-cache) response.

    Returns:
        BLEU score between 0.0 and 1.0.
    """
    if not hypothesis or not reference:
        return 0.0

    _ensure_nltk()
    from nltk.translate.bleu_score import (
        SmoothingFunction,
        sentence_bleu,
    )
    from nltk.tokenize import word_tokenize

    try:
        ref_tokens = word_tokenize(reference.lower())
        hyp_tokens = word_tokenize(hypothesis.lower())

        if not ref_tokens or not hyp_tokens:
            return 0.0

        smoothie = SmoothingFunction().method1
        return sentence_bleu(
            [ref_tokens],
            hyp_tokens,
            smoothing_function=smoothie,
        )
    except Exception:
        logger.warning("BLEU computation failed", exc_info=True)
        return 0.0


def rouge_l_score(hypothesis: str, reference: str) -> float:
    """Compute ROUGE-L F1 score.

    Args:
        hypothesis: Generated (cache-assisted) response.
        reference: Baseline (no-cache) response.

    Returns:
        ROUGE-L F1 score between 0.0 and 1.0.
    """
    if not hypothesis or not reference:
        return 0.0

    try:
        from rouge_score.rouge_scorer import RougeScorer

        scorer = RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(reference, hypothesis)
        return scores["rougeL"].fmeasure
    except Exception:
        logger.warning("ROUGE-L computation failed", exc_info=True)
        return 0.0


def compute_quality_metrics(
    hypothesis: str, reference: str
) -> dict[str, float]:
    """Compute all quality metrics for a response pair.

    Args:
        hypothesis: Cache-assisted response.
        reference: No-cache baseline response.

    Returns:
        Dict with "bleu" and "rouge_l" scores.
    """
    return {
        "bleu": bleu_score(hypothesis, reference),
        "rouge_l": rouge_l_score(hypothesis, reference),
    }
