"""Real-world dataset builder for SemBlend benchmarks.

Downloads and processes production datasets to create semantically
similar prompt clusters at scale. Modeled after SemShareKV's evaluation
methodology (CNN/DailyMail, XSum) but extended with:
  - Multiple summarization datasets (CNN/DailyMail, XSum, MultiNews, WikiHow, SAMSum)
  - RAG-style prompt construction (system + context + query)
  - Controlled semantic variation: REORDER, PARTIAL, PARAPHRASE, DIVERSE
  - Token-level length control (not character-level)
  - Scale: 100-100K prompt clusters

Requires: datasets, transformers (for tokenizer)

Usage:
    python -m benchmarks.e2e.real_dataset_builder \
        --output benchmarks/data/semblend_clusters.json \
        --num-clusters 1000 \
        --target-lengths 1024,4096,8192 \
        --seed 42
"""
from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import urllib.request
import urllib.error


@dataclass(frozen=True)
class PromptVariation:
    """A prompt variation with known semantic relationship to seed."""
    text: str
    overlap_type: str  # exact, reorder, partial_N, paraphrase, diverse
    expected_token_overlap: float
    description: str


@dataclass
class PromptCluster:
    """A cluster of semantically related prompts from real data."""
    cluster_id: str
    source_dataset: str
    seed_text: str
    seed_token_count: int
    target_token_length: int
    variations: list[PromptVariation] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Dataset loaders (HuggingFace datasets)
# ---------------------------------------------------------------------------

def _try_import_datasets():
    try:
        import datasets
        return datasets
    except ImportError:
        raise ImportError(
            "Install datasets: pip install datasets\n"
            "Required for real-world benchmark data."
        )


def _try_import_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(model_name)
    except ImportError:
        raise ImportError(
            "Install transformers: pip install transformers\n"
            "Required for token-level prompt length control."
        )


def load_cnn_dailymail(
    max_articles: int = 5000,
    split: str = "test",
) -> list[dict]:
    """Load CNN/DailyMail articles (same as SemShareKV)."""
    ds = _try_import_datasets()
    dataset = ds.load_dataset("cnn_dailymail", "3.0.0", split=split)
    articles = []
    for item in dataset:
        if len(articles) >= max_articles:
            break
        articles.append({
            "text": item["article"],
            "summary": item["highlights"],
            "source": "cnn_dailymail",
        })
    return articles


def load_xsum(
    max_articles: int = 5000,
    split: str = "test",
) -> list[dict]:
    """Load XSum articles (same as SemShareKV)."""
    ds = _try_import_datasets()
    dataset = ds.load_dataset("EdinburghNLP/xsum", split=split)
    articles = []
    for item in dataset:
        if len(articles) >= max_articles:
            break
        articles.append({
            "text": item["document"],
            "summary": item["summary"],
            "source": "xsum",
        })
    return articles


def load_multinews(
    max_articles: int = 5000,
    split: str = "validation",
) -> list[dict]:
    """Load Multi-News for long-form multi-document context.

    Uses kothasuhas/multi_news_long_context (Parquet format, no loading
    scripts) as the canonical multi_news dataset no longer supports the
    HuggingFace datasets library's current API.
    """
    ds = _try_import_datasets()
    dataset = ds.load_dataset(
        "kothasuhas/multi_news_long_context", split=split,
    )
    articles = []
    for item in dataset:
        if len(articles) >= max_articles:
            break
        text = item.get("text", item.get("document", ""))
        if not text or len(text) < 200:
            continue
        articles.append({
            "text": text,
            "summary": "",
            "source": "multinews",
        })
    return articles


def load_wikihow(
    max_articles: int = 5000,
    split: str = "train",
) -> list[dict]:
    """Load WikiHow instruction articles (used by SemShareKV).

    WikiHow contains long-form procedural text with natural semantic
    overlap between articles covering similar topics/categories.
    Uses dim/wikihow_en (Parquet format) which provides instruction-
    response pairs from WikiHow.
    """
    ds = _try_import_datasets()
    dataset = ds.load_dataset("dim/wikihow_en", split=split)
    articles = []
    for item in dataset:
        instruction = item.get("INSTRUCTION", "")
        response = item.get("RESPONSE", "")
        text = f"# {instruction}\n\n{response}" if instruction else response
        if not text or len(text) < 200:
            continue
        articles.append({
            "text": text,
            "summary": instruction,
            "source": "wikihow",
        })
        if len(articles) >= max_articles:
            break
    return articles


def load_samsum(
    max_articles: int = 5000,
    split: str = "test",
) -> list[dict]:
    """Load SAMSum dialogue-summarization dataset (used by SemShareKV).

    SAMSum contains short-to-medium length messenger-style dialogues
    with human-written summaries. Good for testing conversation
    summarization workloads.
    Fields: dialogue + summary.
    """
    ds = _try_import_datasets()
    dataset = ds.load_dataset("knkarthick/samsum", split=split)
    articles = []
    for item in dataset:
        dialogue = item.get("dialogue", "")
        summary = item.get("summary", "")
        if not dialogue or len(dialogue) < 50:
            continue
        articles.append({
            "text": dialogue,
            "summary": summary,
            "source": "samsum",
        })
        if len(articles) >= max_articles:
            break
    return articles


def load_sharegpt_real(
    path: str | None = None,
    max_convs: int = 5000,
) -> list[dict]:
    """Load ShareGPT conversations from file or HuggingFace."""
    if path and Path(path).exists():
        with open(path) as f:
            data = json.load(f)
    else:
        ds = _try_import_datasets()
        dataset = ds.load_dataset(
            "anon8231489123/ShareGPT_Vicuna_unfiltered",
            split="train",
        )
        data = list(dataset)

    articles = []
    for conv in data[:max_convs]:
        turns = conv.get("conversations", [])
        if not turns:
            continue
        # Concatenate conversation turns as context
        text = "\n".join(
            f"{t.get('from', 'user')}: {t.get('value', '')}"
            for t in turns
        )
        if len(text) < 200:
            continue
        articles.append({
            "text": text,
            "summary": turns[0].get("value", ""),
            "source": "sharegpt",
        })
        if len(articles) >= max_convs:
            break
    return articles


# ---------------------------------------------------------------------------
# Prompt construction (RAG-style)
# ---------------------------------------------------------------------------

SYSTEM_PROMPTS_REAL = [
    "You are a helpful assistant that summarizes documents accurately.",
    "You are an expert analyst. Provide detailed, factual responses.",
    "You are a research assistant. Answer based only on the provided context.",
    "You are a knowledgeable tutor. Explain concepts clearly and thoroughly.",
]

QUERY_TEMPLATES = [
    "Summarize the key points from the above context.",
    "What are the main findings discussed in this passage?",
    "Provide a detailed summary of the information presented.",
    "What are the most important takeaways from this text?",
    "Analyze the main arguments presented in the context above.",
]


def build_rag_prompt(
    system: str,
    context: str,
    query: str,
) -> str:
    """Build a RAG-style prompt."""
    return (
        f"<|im_start|>system\n{system}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Semantic variation generators
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def make_reorder_variation(
    context: str,
    rng: random.Random,
) -> tuple[str, float]:
    """Reorder sentences/chunks — same content, different order.

    This is the critical test for RoPE correction: same tokens at
    different positions.
    """
    sentences = _split_sentences(context)
    if len(sentences) < 4:
        return context, 1.0

    # Shuffle chunks (groups of 2-3 sentences)
    chunk_size = max(2, len(sentences) // 4)
    chunks = [
        " ".join(sentences[i:i + chunk_size])
        for i in range(0, len(sentences), chunk_size)
    ]
    rng.shuffle(chunks)
    reordered = " ".join(chunks)
    # Token overlap ~1.0 (same tokens), but position overlap ~0.0
    return reordered, 0.95


def make_partial_variation(
    context: str,
    replace_ratio: float,
    replacement_pool: list[str],
    rng: random.Random,
) -> tuple[str, float]:
    """Replace a fraction of sentences with different content.

    Models SemShareKV's sentence replacement ablation at scale.
    """
    sentences = _split_sentences(context)
    if len(sentences) < 3:
        return context, 1.0

    n_replace = max(1, int(len(sentences) * replace_ratio))
    indices = list(range(len(sentences)))
    rng.shuffle(indices)
    replace_indices = set(indices[:n_replace])

    result_sentences = []
    for i, sent in enumerate(sentences):
        if i in replace_indices and replacement_pool:
            # Replace with a sentence from another article
            replacement = rng.choice(replacement_pool)
            rep_sents = _split_sentences(replacement)
            if rep_sents:
                result_sentences.append(rng.choice(rep_sents))
            else:
                result_sentences.append(sent)
        else:
            result_sentences.append(sent)

    return " ".join(result_sentences), 1.0 - replace_ratio


def make_paraphrase_variation(
    context: str,
    rng: random.Random,
) -> tuple[str, float]:
    """Paraphrase by swapping word order within sentences and
    replacing common words. Approximates semantic equivalence
    without requiring an LLM.
    """
    sentences = _split_sentences(context)
    paraphrased = []

    synonyms = {
        "important": "significant",
        "significant": "important",
        "show": "demonstrate",
        "demonstrate": "show",
        "use": "utilize",
        "utilize": "use",
        "help": "assist",
        "assist": "help",
        "big": "large",
        "large": "big",
        "small": "minor",
        "minor": "small",
        "good": "effective",
        "effective": "good",
        "new": "novel",
        "novel": "new",
        "make": "create",
        "create": "make",
        "find": "discover",
        "discover": "find",
        "get": "obtain",
        "obtain": "get",
        "give": "provide",
        "provide": "give",
        "said": "stated",
        "stated": "said",
        "also": "additionally",
        "additionally": "also",
        "however": "nevertheless",
        "nevertheless": "however",
    }

    for sent in sentences:
        words = sent.split()
        # Swap some synonyms
        new_words = []
        for w in words:
            w_lower = w.lower().strip(".,!?;:")
            if w_lower in synonyms and rng.random() < 0.5:
                replacement = synonyms[w_lower]
                # Preserve capitalization
                if w[0].isupper():
                    replacement = replacement.capitalize()
                # Preserve punctuation
                trail = ""
                for c in reversed(w):
                    if c in ".,!?;:":
                        trail = c + trail
                    else:
                        break
                new_words.append(replacement + trail)
            else:
                new_words.append(w)
        paraphrased.append(" ".join(new_words))

    return " ".join(paraphrased), 0.80


def make_diverse_variation(
    replacement_pool: list[str],
    rng: random.Random,
) -> tuple[str, float]:
    """Completely different context — should NOT match."""
    if replacement_pool:
        return rng.choice(replacement_pool), 0.0
    return "This is a completely unrelated document about a different topic.", 0.0


def _llm_complete(
    endpoint: str,
    prompt: str,
    model: str | None = None,
    max_tokens: int = 512,
    temperature: float = 0.7,
    timeout_s: float = 60.0,
) -> str | None:
    """Call a vLLM /v1/completions endpoint and return generated text.

    Returns None on any failure (network, timeout, bad response).
    """
    try:
        import requests as _requests
    except ImportError:
        # Fall back to urllib if requests not available
        payload = json.dumps({
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": 0.9,
            **({"model": model} if model else {}),
        }).encode("utf-8")
        url = endpoint.rstrip("/") + "/v1/completions"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            choices = body.get("choices", [])
            if choices:
                return choices[0].get("text", "").strip()
        except Exception as exc:
            logging.debug("LLM call failed (urllib): %s", exc)
        return None

    url = endpoint.rstrip("/") + "/v1/completions"
    body = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
    }
    if model:
        body["model"] = model
    try:
        resp = _requests.post(url, json=body, timeout=timeout_s)
        resp.raise_for_status()
        data = resp.json()
        choices = data.get("choices", [])
        if choices:
            return choices[0].get("text", "").strip()
    except Exception as exc:
        logging.debug("LLM call failed (requests): %s", exc)
    return None


def _chunk_text(text: str, max_chunk_chars: int = 2000) -> list[str]:
    """Split text into paragraph-sized chunks for per-chunk LLM rewriting.

    Splits on double-newlines first, then on sentence boundaries if
    paragraphs are still too long. Each chunk stays under max_chunk_chars.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    chunks: list[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        if len(para) <= max_chunk_chars:
            chunks.append(para)
        else:
            # Split long paragraphs on sentence boundaries
            sentences = _split_sentences(para)
            current: list[str] = []
            current_len = 0
            for sent in sentences:
                if current_len + len(sent) > max_chunk_chars and current:
                    chunks.append(" ".join(current))
                    current = []
                    current_len = 0
                current.append(sent)
                current_len += len(sent) + 1
            if current:
                chunks.append(" ".join(current))
    return chunks if chunks else [text[:max_chunk_chars]]


# ---- LLM rewrite prompts matching SemShareKV methodology ----

_PARAPHRASE_PROMPT = (
    "Rewrite the following passage using different words and sentence "
    "structures while preserving the exact same meaning and factual "
    "content. Keep approximately the same length (±10%). Do not add "
    "new information or omit any facts.\n\n"
    "Original:\n{text}\n\n"
    "Rewritten version:\n"
)

_LIGHT_REWRITE_PROMPT = (
    "Lightly rephrase the following text: change a few words and "
    "rearrange some clauses, but keep most of the original phrasing "
    "intact. Preserve all facts and the same length.\n\n"
    "Original:\n{text}\n\n"
    "Rephrased:\n"
)

_HEAVY_REWRITE_PROMPT = (
    "Completely rewrite the following passage in your own words. Use "
    "entirely different sentence structures and vocabulary while "
    "conveying the same information. The rewrite should be similar "
    "in length (±15%) but share very few exact phrases with the "
    "original.\n\n"
    "Original:\n{text}\n\n"
    "Complete rewrite:\n"
)


def _llm_rewrite_chunks(
    text: str,
    endpoint: str,
    prompt_template: str,
    model: str | None = None,
    rng: random.Random | None = None,
    max_chunk_chars: int = 2000,
    timeout_s: float = 60.0,
) -> str | None:
    """Rewrite text chunk-by-chunk using an LLM.

    Splits text into manageable chunks, rewrites each with the given
    prompt template, and reassembles. Returns None if >30% of chunks
    fail (falls back to non-LLM method in caller).
    """
    chunks = _chunk_text(text, max_chunk_chars=max_chunk_chars)
    rewritten: list[str] = []
    failures = 0

    for chunk in chunks:
        prompt = prompt_template.format(text=chunk)
        # Estimate max_tokens: ~1.3x the word count of the chunk
        est_tokens = max(128, int(len(chunk.split()) * 1.3))
        result = _llm_complete(
            endpoint, prompt,
            model=model,
            max_tokens=est_tokens,
            temperature=0.7,
            timeout_s=timeout_s,
        )
        if result and len(result) > len(chunk) * 0.3:
            # Trim if LLM output much longer than input
            if len(result) > len(chunk) * 1.5:
                result = result[:int(len(chunk) * 1.15)]
            rewritten.append(result)
        else:
            failures += 1
            rewritten.append(chunk)  # keep original on failure

    # If too many chunks failed, signal overall failure
    if failures > len(chunks) * 0.3:
        return None

    return "\n\n".join(rewritten)


def make_llm_paraphrase_variation(
    context: str,
    vllm_endpoint: str,
    rng: random.Random,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> tuple[str, float]:
    """Paraphrase text using chunk-wise LLM rewriting.

    Matches SemShareKV's methodology: each paragraph is independently
    rewritten by the LLM to produce a semantically equivalent but
    lexically different version. This creates realistic paraphrase
    patterns that test embedding-based donor matching.

    Falls back to synonym-based paraphrase if the LLM is unavailable.

    Returns (paraphrased_text, expected_overlap=0.65).
    """
    result = _llm_rewrite_chunks(
        context, vllm_endpoint, _PARAPHRASE_PROMPT,
        model=model, rng=rng, timeout_s=timeout_s,
    )
    if result is not None:
        return result, 0.65

    logging.warning("LLM paraphrase failed, falling back to synonym method")
    return make_paraphrase_variation(context, rng)


def make_llm_light_rewrite_variation(
    context: str,
    vllm_endpoint: str,
    rng: random.Random,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> tuple[str, float]:
    """Light LLM rewrite — keeps most original phrasing, changes a few words.

    Produces variants with high token overlap (~85%) but different enough
    to miss exact prefix matching. Tests the boundary between prefix cache
    hits and semantic matching.

    Returns (rewritten_text, expected_overlap=0.85).
    """
    result = _llm_rewrite_chunks(
        context, vllm_endpoint, _LIGHT_REWRITE_PROMPT,
        model=model, rng=rng, timeout_s=timeout_s,
    )
    if result is not None:
        return result, 0.85

    # Fallback: synonym substitution produces similar light changes
    return make_paraphrase_variation(context, rng)


def make_llm_heavy_rewrite_variation(
    context: str,
    vllm_endpoint: str,
    rng: random.Random,
    model: str | None = None,
    timeout_s: float = 60.0,
) -> tuple[str, float]:
    """Heavy LLM rewrite — completely different wording, same meaning.

    Produces variants with low token overlap (~40-50%) but high semantic
    similarity. This is the hardest test for donor matching: Jaccard
    will miss these entirely, only embedding-based search catches them.

    Returns (rewritten_text, expected_overlap=0.45).
    """
    result = _llm_rewrite_chunks(
        context, vllm_endpoint, _HEAVY_REWRITE_PROMPT,
        model=model, rng=rng, timeout_s=timeout_s,
    )
    if result is not None:
        return result, 0.45

    # Fallback: no good non-LLM equivalent for heavy rewrite
    return make_paraphrase_variation(context, rng)


# ---------------------------------------------------------------------------
# Cluster builder
# ---------------------------------------------------------------------------

def _truncate_variation(
    context: str,
    system: str,
    query: str,
    target_len: int,
    tokenizer,
) -> str:
    """Build RAG prompt from variation context, truncated to target_len tokens."""
    prompt = build_rag_prompt(system, context, query)
    tokens = tokenizer.encode(prompt)[:target_len]
    return tokenizer.decode(tokens, skip_special_tokens=False)


def build_clusters(
    articles: list[dict],
    tokenizer,
    target_lengths: list[int],
    variations_per_cluster: int = 8,
    rng: random.Random | None = None,
    vllm_endpoint: str | None = None,
    vllm_model: str | None = None,
) -> list[PromptCluster]:
    """Build prompt clusters from real articles.

    For each article and target length, creates variation types modeled
    after SemShareKV's evaluation methodology:

    Without LLM endpoint (8 variations):
      EXACT, REORDER, PARTIAL_80/60/40/20, PARAPHRASE (synonym), DIVERSE

    With LLM endpoint (10 variations — adds LLM-based rewrites):
      EXACT, REORDER, PARTIAL_80/60/40/20,
      PARAPHRASE (LLM chunk-wise rewrite, ~65% token overlap),
      LLM_LIGHT_REWRITE (~85% token overlap),
      LLM_HEAVY_REWRITE (~45% token overlap),
      DIVERSE

    The LLM-based variations match SemShareKV's methodology of using
    LLM-rewritten prompts at controlled similarity levels to test
    semantic KV cache matching across the full similarity spectrum.
    """
    if rng is None:
        rng = random.Random(42)

    replacement_pool = [a["text"] for a in articles]
    clusters = []
    llm_ok = vllm_endpoint is not None

    if llm_ok:
        logging.info(
            "LLM rewriting enabled: endpoint=%s, model=%s",
            vllm_endpoint, vllm_model or "(default)",
        )

    for target_len in target_lengths:
        for idx, article in enumerate(articles):
            system = SYSTEM_PROMPTS_REAL[idx % len(SYSTEM_PROMPTS_REAL)]
            query = QUERY_TEMPLATES[idx % len(QUERY_TEMPLATES)]

            # Build seed prompt and truncate to target token length
            full_prompt = build_rag_prompt(system, article["text"], query)
            tokens = tokenizer.encode(full_prompt)

            if len(tokens) < target_len:
                padded_context = article["text"]
                while True:
                    extended_prompt = build_rag_prompt(
                        system, padded_context, query
                    )
                    extended_tokens = tokenizer.encode(extended_prompt)
                    if len(extended_tokens) >= target_len:
                        tokens = extended_tokens[:target_len]
                        break
                    padded_context = padded_context + "\n\n" + article["text"]
                seed_text = tokenizer.decode(tokens, skip_special_tokens=False)
            elif len(tokens) > target_len:
                tokens = tokens[:target_len]
                seed_text = tokenizer.decode(tokens, skip_special_tokens=False)
            else:
                seed_text = tokenizer.decode(tokens, skip_special_tokens=False)

            cluster_id = hashlib.md5(
                f"{article['source']}:{idx}:{target_len}".encode()
            ).hexdigest()[:12]

            cluster = PromptCluster(
                cluster_id=cluster_id,
                source_dataset=article["source"],
                seed_text=seed_text,
                seed_token_count=len(tokens),
                target_token_length=target_len,
            )

            # Prepare context (padded to target length if needed)
            context = article["text"]
            if len(tokenizer.encode(build_rag_prompt(system, context, query))) < target_len:
                padded = context
                while len(tokenizer.encode(build_rag_prompt(system, padded, query))) < target_len:
                    padded = padded + "\n\n" + context
                context = padded

            # ---- Generate variations ----

            # 1. EXACT
            cluster.variations.append(PromptVariation(
                text=seed_text,
                overlap_type="exact",
                expected_token_overlap=1.0,
                description="Identical prompt",
            ))

            # 2. REORDER (same tokens, different positions — tests RoPE)
            reordered_ctx, overlap = make_reorder_variation(context, rng)
            cluster.variations.append(PromptVariation(
                text=_truncate_variation(reordered_ctx, system, query, target_len, tokenizer),
                overlap_type="reorder",
                expected_token_overlap=overlap,
                description="Same sentences, shuffled order",
            ))

            # 3-6. PARTIAL at 20%, 40%, 60%, 80% replacement
            other_texts = [
                a["text"] for a in articles
                if a["text"] != article["text"]
            ]
            for pct in [20, 40, 60, 80]:
                partial_ctx, overlap = make_partial_variation(
                    context, pct / 100.0, other_texts, rng
                )
                cluster.variations.append(PromptVariation(
                    text=_truncate_variation(partial_ctx, system, query, target_len, tokenizer),
                    overlap_type=f"partial_{100 - pct}",
                    expected_token_overlap=overlap,
                    description=f"{pct}% of sentences replaced",
                ))

            # 7. PARAPHRASE — LLM chunk-wise rewrite or synonym fallback
            if llm_ok:
                para_ctx, overlap = make_llm_paraphrase_variation(
                    context, vllm_endpoint, rng,
                    model=vllm_model,
                )
                para_desc = "LLM chunk-wise paraphrase (SemShareKV-style)"
            else:
                para_ctx, overlap = make_paraphrase_variation(context, rng)
                para_desc = "Synonym-substituted paraphrase"
            cluster.variations.append(PromptVariation(
                text=_truncate_variation(para_ctx, system, query, target_len, tokenizer),
                overlap_type="paraphrase",
                expected_token_overlap=overlap,
                description=para_desc,
            ))

            # 8+9. LLM rewrites at different intensity (only with endpoint)
            if llm_ok:
                # Light rewrite (~85% overlap — boundary case)
                light_ctx, light_overlap = make_llm_light_rewrite_variation(
                    context, vllm_endpoint, rng,
                    model=vllm_model,
                )
                cluster.variations.append(PromptVariation(
                    text=_truncate_variation(light_ctx, system, query, target_len, tokenizer),
                    overlap_type="llm_light_rewrite",
                    expected_token_overlap=light_overlap,
                    description="LLM light rewrite (~85% token overlap)",
                ))

                # Heavy rewrite (~45% overlap — hardest for matching)
                heavy_ctx, heavy_overlap = make_llm_heavy_rewrite_variation(
                    context, vllm_endpoint, rng,
                    model=vllm_model,
                )
                cluster.variations.append(PromptVariation(
                    text=_truncate_variation(heavy_ctx, system, query, target_len, tokenizer),
                    overlap_type="llm_heavy_rewrite",
                    expected_token_overlap=heavy_overlap,
                    description="LLM heavy rewrite (~45% token overlap)",
                ))

            # Last. DIVERSE (completely different article — negative control)
            diverse_ctx, overlap = make_diverse_variation(other_texts, rng)
            cluster.variations.append(PromptVariation(
                text=_truncate_variation(diverse_ctx, system, query, target_len, tokenizer),
                overlap_type="diverse",
                expected_token_overlap=overlap,
                description="Completely different article",
            ))

            clusters.append(cluster)

    return clusters


def build_and_save(
    output_path: str,
    datasets: list[str],
    target_lengths: list[int],
    max_articles_per_dataset: int = 1000,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-AWQ",
    seed: int = 42,
    vllm_endpoint: str | None = None,
    vllm_model: str | None = None,
) -> list[PromptCluster]:
    """Build clusters from real datasets and save to JSON."""
    rng = random.Random(seed)
    tokenizer = _try_import_tokenizer(model_name)

    all_articles = []
    for ds_name in datasets:
        print(f"Loading {ds_name}...")
        if ds_name == "cnn_dailymail":
            all_articles.extend(load_cnn_dailymail(max_articles_per_dataset))
        elif ds_name == "xsum":
            all_articles.extend(load_xsum(max_articles_per_dataset))
        elif ds_name == "multinews":
            all_articles.extend(load_multinews(max_articles_per_dataset))
        elif ds_name == "wikihow":
            all_articles.extend(load_wikihow(max_articles_per_dataset))
        elif ds_name == "samsum":
            all_articles.extend(load_samsum(max_articles_per_dataset))
        elif ds_name == "sharegpt":
            all_articles.extend(load_sharegpt_real(
                max_convs=max_articles_per_dataset
            ))

    rng.shuffle(all_articles)
    print(f"Loaded {len(all_articles)} articles total")

    print(f"Building clusters for lengths {target_lengths}...")
    clusters = build_clusters(
        all_articles, tokenizer, target_lengths, rng=rng,
        vllm_endpoint=vllm_endpoint, vllm_model=vllm_model,
    )
    print(f"Built {len(clusters)} clusters")

    # Save
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(
            [asdict(c) for c in clusters],
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Saved to {out}")
    return clusters


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        description="Build real-world datasets for SemBlend benchmarks"
    )
    parser.add_argument(
        "--output", default="benchmarks/data/semblend_clusters.json",
    )
    parser.add_argument(
        "--datasets", nargs="+",
        default=["cnn_dailymail", "xsum"],
        help="Datasets to use (cnn_dailymail, xsum, multinews, wikihow, samsum, sharegpt)",
    )
    parser.add_argument(
        "--target-lengths", nargs="+", type=int,
        default=[1024, 2048, 4096, 8192, 16384],
    )
    parser.add_argument(
        "--max-articles", type=int, default=1000,
        help="Max articles per dataset",
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-7B-Instruct-AWQ",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    build_and_save(
        output_path=args.output,
        datasets=args.datasets,
        target_lengths=args.target_lengths,
        max_articles_per_dataset=args.max_articles,
        model_name=args.model,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
