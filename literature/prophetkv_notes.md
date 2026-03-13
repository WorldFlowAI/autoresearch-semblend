# ProphetKV Notes

**arXiv**: 2602.02579
**Full title**: "ProphetKV: Selective KV Cache Recomputation by Predicting Future Attention"
**Relation to primer**: FM2 fix (semantic staleness) — the paper referenced in `primer_kvreuse_notes.md`

---

## What It Does

ProphetKV addresses a specific sub-problem: RAG prefill where multiple retrieved documents have
pre-computed KV caches. Those caches were computed without seeing each other or the user query,
so cross-attention between chunks is missing. Existing methods (CacheBlend, EPIC, KVShare) recompute
a budget of tokens to recover that cross-attention. ProphetKV argues they pick the wrong tokens.

**The crowding-out effect**: Globally salient but query-irrelevant tokens (structural tokens, topic
headings, named entities that happen to be frequent) saturate the recomputation budget. This
displaces the tokens that actually matter for the user's specific question.

**ProphetKV's fix**: Use the user query as a "prophet." The query appears at the prompt end and can
be run against the cached keys in a lightweight pass before any recomputation budget is spent.

---

## Methodology: Query-Attention-Weighted Token Selection

**Stage I — lightweight relevance pass** (no recomputation, uses cached keys only):

For each context token t, compute its average attention weight across all query tokens Q_s:

    α(t) = (1/|Q_s|) Σ_{q ∈ Q_s} Softmax(q · K'_{1:C} / √d_k)

This is the practical proxy for the semantic loss bound:

    L_semantic ≤ Σ_t ||Φ̂(Q_s,t)V_t − Φ̂'(Q_s,t)V'_t||_2

derived via triangle inequality. α(t) is cheap: it uses only pre-computed K tensors and the query
token representations, not a full attention layer.

**Stage II — multi-layer fusion**:

    ᾱ(t) = (1/L) Σ_{l=1}^{L} α_l(t)

Top-p tokens by ᾱ(t) are selected for recomputation.

**Stage III**: Full KV recomputation for selected tokens across all layers; remainder reuse cache.

Note: ProphetKV's overhead is extremely small. The ideal oracle converges at ~2% budget; ProphetKV
at ~4%; CacheBlend at ~6%.

---

## Key Empirical Results

**TTFT (H100 GPU, 20% recompute budget)**:

| Model | Context | Full Recompute | ProphetKV | CacheBlend |
|-------|---------|---------------|-----------|-----------|
| Llama-3.1-8B | 16K | 5.23s | 1.13s (4.6x) | 1.29s |
| Qwen2.5-14B | 16K | 9.94s | 2.12s (4.7x) | 2.31s |

**Accuracy (RULER, Llama-3.1-8B, 20% budget)**:

| Task | NaiveReuse | CacheBlend | EPIC | KVShare | ProphetKV |
|------|-----------|-----------|------|---------|-----------|
| Multi-Query | 54.75% | 82.25% | 67.00% | 79.50% | **98.50%** |
| Multi-Value | 32.00% | 73.50% | 50.50% | 67.25% | **100.00%** |
| LongBench avg | 29.84 | 39.82 | 35.41 | 38.67 | **50.80** |

Improvement over CacheBlend: 8.8–24.9% on RULER, 18.6–50.9% on LongBench.
ProphetKV retains **96–101% of full-prefill accuracy** at 20% recomputation ratio.

**Chunk size dependency**: accuracy degrades for chunks < 256 tokens — matches LMCache's constraint.

---

## What ProphetKV Does NOT Address

- No concept of **cross-request** KV reuse (all caches come from the same request's documents)
- No concept of **cache staleness** across time or across users
- No discussion of RoPE positional correction (chunks are injected at fixed offsets within one request)
- The "semantic staleness" FM2 framing in `primer_kvreuse_notes.md` was the primer's interpretation;
  ProphetKV itself frames the problem as intra-request query-relevance, not cross-request drift

---

## Relation to SemBlend

**Different problem layer — complementary, not competing.**

| Aspect | ProphetKV | SemBlend |
|--------|-----------|----------|
| Problem | Intra-request, RAG multi-doc prefill | Cross-request KV injection |
| Cache source | Same request's retrieved documents | Prior requests from other users |
| Token handling | Recomputes 20% of tokens (query-relevant) | Recomputes 7–9% of layers (architecture-sensitive) |
| Hit detection | Per-token query-attention score | LMCache 256-token block hash match |
| Positional correction | Not addressed | Exact RoPE delta correction |
| Context length tested | 16K (H100) | 8K–32K (A10G) |
| Quality metric | Task accuracy (RULER/LongBench) | PPL ratio (1.000–1.065) |

**Shared insight**: Both identify that naive full-cache reuse degrades quality and that selective
recomputation of the right subset recovers fidelity. ProphetKV selects by token-level query relevance;
SemBlend selects by architecture-specific layer sensitivity (bathtub). The selection axes are
orthogonal.

**ProphetKV-lite as future work**: ProphetKV's Stage I (lightweight query-attention pass against
cached keys) could be applied on top of SemBlend's injected caches to score each donor chunk's
query relevance before injection. This would provide a per-chunk quality gate to catch borderline-
threshold hits (e.g., cosine τ=0.60–0.65 where some chunks may be semantically stale). Low overhead
since it uses cached K tensors already present in LMCache.

---

## Paper Actions

**Related Work**: Distinguish from ProphetKV as intra-request vs. cross-request. ProphetKV is the
strongest 2025/2026 baseline for RAG KV reuse and shows what targeted recomputation can achieve.
Frame SemBlend as "operating one layer above ProphetKV's problem": SemBlend handles the donor
selection and injection; ProphetKV-style refinement could improve borderline hits.

**Key differentiator**: SemBlend's block-level exact matching (LMCache hash) is a conservative
semantic gate that already prevents stale injection within matched blocks — the problem ProphetKV
solves for intra-request is pre-empted by SemBlend's matching criterion for cross-request.
