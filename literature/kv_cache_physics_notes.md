# KV Cache Physics Notes

**arXiv**: 2603.01426
**Full title**: "Understanding the Physics of Key-Value Cache Compression for LLMs through Attention Dynamics"
**Relation to primer**: FM3 fix (wrong layer selection) — architecture-specific criticality profiles

---

## Core Argument

Standard KV compression evaluations are structurally misleading. They report 80–90% memory savings
with "minimal degradation" on surface benchmarks, but those benchmarks do not detect when attention
routing has been fatally disrupted. The paper introduces a physics-inspired framework treating KV
compression as a controlled perturbation of token-level attention routing, and distinguishes three
independent concepts:

- **Retention**: token survives in cache (the standard metric)
- **Accessibility**: token is reachable via attention routing (often assumed, rarely verified)
- **Utilization**: token actually influences generation output

Key finding: **retention ≠ utilization**. High probe accuracy can coexist with generation failure
when attention has over-consolidated (representational rigidity).

---

## The Safety Cliff

**Threshold**: α ≈ 0.90 (90% KV compression ratio)

At α < 0.90, accuracy degrades gradually. At α ≥ 0.90, hallucination rates spike sharply in a
pattern consistent with a phase transition. The paper characterizes this as:

    compression susceptibility χ = ∂H/∂α peaks sharply near α ≈ 0.90

where H is the hallucination rate.

**Mechanism — Global Eviction Ratio (GER)**:

    GER(α) = (1/|T_ans|) × Σ_t 1[Σ_h S_h(t,α) = 0]

A token is "globally evicted" when every attention head has simultaneously dropped it. At that
point no redundant routing path survives to recover the information. GER correlates strongly with
hallucination rate (Pearson r = 0.86–0.93 across datasets), with the correlation steepening near
the cliff.

**Why it's a phase transition**: Below the cliff, redundant head-wise routes allow reconstruction
even when some heads have evicted a token ("lottery ticket hypothesis" for attention heads). Above
the cliff, the winning tickets are all destroyed simultaneously.

**Second failure mode — Representational Rigidity**: Even when tokens survive, excessive head-level
consensus (all heads attending the same small token set) collapses routing flexibility. Formally:
Consensus(l) → 1.0. Generation fails not because tokens are missing but because the attention map
has over-consolidated and cannot route flexibly. Qwen is particularly susceptible due to its
late-stage funnel convergence.

---

## Architecture-Specific Findings

| Property | LLaMA family | Qwen family |
|----------|-------------|-------------|
| Depth pattern | Inverted funnel: early consensus, late diversification | Funnel: early exploration, late consolidation |
| Critical layers | Early layers (1–8): information consolidation point | Late layers (20–28): consolidation point |
| Safe to evict | Middle and late layers (high redundancy) | Early layers (relatively redundant) |
| Failure mode | Late-layer specialization disrupted at high compression | Representational rigidity from late over-consolidation |
| Compression behavior | Stabilizes early; better on multi-entity tracking | More gradual degradation; better on knowledge manipulation |

This is the empirical grounding for why different layer sets must be recomputed per architecture when
injecting cross-request KV caches.

---

## Relation to SemBlend

### The safety cliff does not apply to SemBlend's regime

SemBlend does not compress KV caches — it substitutes a full pre-computed KV cache from a
semantically similar prior request. No tokens are evicted; the full token count is preserved.
GER is structurally 0 for matched chunks because no answer-critical tokens are globally evicted.

The phase transition at α ≈ 0.90 applies to methods that discard 90% of KV entries. SemBlend's
failure mode is semantic drift between donor and current request — a different axis from token
erasure. This is why SemBlend's PPL ratios cluster near 1.0 on high-hit-rate datasets: the
donor cache provides sufficient routing structure for the current request.

### Architecture-specific bathtub — confirmed correct

SemBlend's implemented bathtub recomputation directly targets the layers this paper identifies as
sensitive:

| Architecture | Paper's critical layers | SemBlend's bathtub |
|-------------|------------------------|--------------------|
| Qwen (funnel) | Late layers (20–28) | Recomputes layers [26, 27] = 7.1% |
| LLaMA (inverted funnel) | Early layers (1–8) | Recomputes layers [0, 1, 2] = 9.4% |

The physics paper provides independent theoretical grounding for why these layer selections are
correct. This is not a coincidence — the bathtub targets the exact consolidation points where
injecting a donor's KV state (which was computed from a different prompt) introduces the most
distributional mismatch.

### Representational rigidity warning for Qwen

The paper flags late over-consolidation in Qwen as the primary compression failure mode. SemBlend's
late-layer recomputation (layers 26–27) directly addresses this: by recomputing rather than reusing
the donor's consolidated attention representation at the funnel layers, SemBlend avoids imposing
the donor's over-consolidated routing pattern on the current request. This is the mechanism behind
Qwen's clean PPL ratios (1.006–1.025) on high-hit-rate datasets.

### Safety cliff reframed for SemBlend's operating regime

From `primer_kvreuse_notes.md`, the GER-based safety analysis translates to hit rate:
- Hit rate 75% (GER = 0.25) → safe zone (below cliff)
- Within hits: SemBlend's LMCache block exact matching means per-hit GER is always 0 (full block
  match) or 1 (miss, not injected) — no partial injection
- This binary gate is why PPL ratios are clean on hits: there are no "partial" injections near the
  GER cliff

**Threshold implication**: With RoPE correction (FM1 fixed), safe cosine similarity range is
approximately τ ≈ 0.72–0.80 per primer analysis. SemBlend uses τ = 0.60 with LMCache block matching
as the real gate — the cosine threshold is a coarse pre-filter, not the quality gate.

---

## Key Numbers for Paper

- Safety cliff: α ≈ 0.90 (90% compression), χ = ∂H/∂α peaks here
- GER-hallucination correlation: Pearson r = 0.86–0.93
- LLaMA critical layers: 1–8 (early; inverted funnel)
- Qwen critical layers: 20–28 (late; funnel)
- SemBlend bathtub: Qwen [26,27] = 7.1%, LLaMA [0,1,2] = 9.4%
- SemBlend PPL ratios at ≥88% hit rate: 1.000–1.065 (structurally safe; not in compression regime)

---

## Paper Actions

**Related Work / Architecture-specific bathtub section**: Cite arXiv:2603.01426 as independent
empirical support for the layer selection rationale. The paper's LLaMA/Qwen depth patterns directly
predict which layers experience the most disruption from cross-request KV substitution.

**Safety analysis paragraph**: Frame SemBlend's regime as qualitatively distinct from the
compression regime studied here. The cliff at α ≈ 0.90 is irrelevant when GER = 0 (no eviction);
SemBlend's quality is governed by semantic similarity thresholds (τ = 0.60, hit rate ≥ 88%), not
by token survival rates.

**Quality section**: PPL ratios of 1.000–1.065 (≥88% hit) and near-1.0 hit-only PPL on miss-
inflated datasets (MultiNews: hit-only 1.007 Qwen, 0.992 LLaMA) confirm that matched-block
injection with architecture-specific layer recomputation stays well below any quality cliff.
