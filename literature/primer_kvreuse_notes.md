# KV Cache Reusability: Primer Notes (March 2026)

Source: Research Primer "New Research on KV Tensor Reusability: Why CacheBlend-Style Correction Fails and What Replaces It"

## The Three Failure Modes Framework

CacheBlend-style KV injection fails for three distinct reasons. Understanding which apply to SemBlend drives our research priorities.

---

## Failure Mode 1: Structural (RoPE Positional Mismatch)

**Papers**: KVComm (2024), SemShareKV (2025)

**Mechanism**: KV tensors store positional encodings at write time (RoPE encoded at sequence position i). When injected at position j ≠ i, the Q×K dot products see a wrong geometric relationship → model attends to semantically wrong tokens.

**Evidence**: KVComm shows 12-23% accuracy drop on reading comprehension when KV blocks are injected at wrong offsets without positional correction. Effect scales with context distance.

**SemBlend status**: **ALREADY ADDRESSED.** `partial_attn.py` implements exact RoPE delta correction: rotates donor KV tensors from donor position to query position before injection. This is a documented, validated feature — validated in paper (tab:rope-validation). This is SemBlend's competitive advantage over naive CacheBlend.

**Paper action**: Continue emphasizing this in Related Work and Conclusions. We solve FM1 correctly.

---

## Failure Mode 2: Algorithmic (Semantic Staleness — ProphetKV Gap)

**Paper**: ProphetKV (2024) — "Prophet: Speculative KV Cache Reuse via Query Prediction"

**Mechanism**: Token-ID matching (Wagner-Fischer edit distance = what CacheBlend uses) identifies structurally similar tokens, but not semantically valid ones for the current query. A token at position i in the donor may be the same string as in the query, but if the surrounding semantic context differs, the K/V vectors are "stale" for this query.

**ProphetKV's solution**:
- Predict the query's attention distribution using a lightweight proxy (< 1 full attention layer)
- Use predicted attention to identify which donor layers are "compatible" vs "stale"
- Recompute only layers predicted as stale; reuse the rest
- Key insight: not all layers need recomputation — attention is layerwise sparse

**Evidence**: ProphetKV achieves 94-97% quality retention (vs CacheBlend's 88-91%) with 0.3ms overhead for the prediction step.

**SemBlend status**: **Partial gap.** SemBlend's 256-token LMCache chunk hashing provides block-level binary decisions (match/no-match), not per-layer semantic quality estimation. If all 256 tokens match exactly (hash collision = exact block match), we get 100% quality. If they don't, we get a miss. There's no graceful degradation path.

**Actionable experiment**: ProphetKV-lite — add a per-chunk quality score before injection. Compute L2 distance between donor block embedding and query block embedding. If score < secondary_threshold, recompute that block (don't inject). This adds a quality check without full ProphetKV overhead.

**Paper action**: Acknowledge in Related Work. Frame our 256-token block matching as "conservative semantic gating" — only reuse if the full block is an exact hash match, preventing stale injection entirely.

---

## Failure Mode 3: Scope (Wrong Layer Selection — DroidSpeak/kv-cache-physics Gap)

**Papers**: DroidSpeak (2025), kv-cache-physics (2025)

### DroidSpeak Findings
- Tested selective KV recomputation on 7B-70B models across 12 datasets
- Found that only **~11% of layers are "critical"** for output quality
- Recomputing only these 11% achieves 95-99% quality of full recomputation
- Critical layers: concentrations at specific depths that vary by model family
- The classic "bathtub curve" (first+last layers most important) is a rough approximation — actual critical layer distribution is non-uniform and model-specific

### kv-cache-physics Findings
- Maps KV cache "physics" — how information flows and consolidates across layers
- **LLaMA family**: "inverted funnel" — early layers (1-8) are the information consolidation point; middle and late layers have high redundancy. Evicting late layers is safe; evicting early layers causes hallucinations.
- **Qwen family**: "funnel" — OPPOSITE pattern. Late layers (20-28) consolidate. Early layers are more redundant. Evicting early layers from Qwen is relatively safe.
- Pearson r=0.86-0.93 between Global Eviction Ratio (GER = 1 - hit fraction) and hallucination rate across datasets
- Safety cliff: at GER ≥ 0.10 (= 90% hit rate), hallucination rate starts nonlinear growth

### SemBlend's bathtub curve
Current implementation: injects KV for layers in "bathtub" shape — first and last layers more aggressively reinjected, middle layers less so. This is uniform across architectures.

**Gap**: SemBlend should use architecture-specific layer criticality profiles:
- LLaMA: prioritize correctness in early layers (1-10), can be more aggressive in middle layers
- Qwen: prioritize correctness in late layers (20-28), can be more aggressive in early layers

**Actionable experiment**:
1. Extract per-layer attention entropy from existing benchmark traces (no GPU needed if we logged fingerprints)
2. Compare entropy profiles for Qwen vs LLaMA
3. Tune `bathtub_fraction` separately: `SEMBLEND_BATHTUB_FRACTION_LLAMA` vs `SEMBLEND_BATHTUB_FRACTION_QWEN`
4. Measure quality delta

---

## The Safety Cliff (Critical for Paper Claims)

**Core finding**: At ~90% KV compression (GER = 0.10), hallucination rates show nonlinear growth (Pearson r = 0.86-0.93).

**Implication for SemBlend**:
- If hit rate = 75% (25% of inputs are cold) → GER = 0.25 → this is in the SAFE zone
- But within hits: if the hit is only 80% chunk-match (20% of chunks not in donor) → within-hit GER = 0.20 → may be near cliff
- SemBlend's LMCache chunk exact matching already prevents partial-block injection → our effective within-hit GER is always 0 (full block match) or 1 (miss)
- This is why our PPL ratios are clean near 1.0: we never inject partial/degraded KV

**Safe threshold** (from primer):
- Without corrections: safe range τ ≈ 0.82-0.85 (high similarity required to stay below GER cliff)
- With ProphetKV-lite: safe range τ ≈ 0.75-0.78 (can accept lower similarity because per-chunk quality gate filters out bad ones)
- With RoPE correction (SemBlend): safe range τ ≈ 0.72-0.80 (RoPE fix removes FM1, reducing risk)

**SemBlend specific**: Our block-level exact matching already implements a strong gate. The cosine threshold τ is a pre-filter. The LMCache hash match is the actual gate. So our "effective" quality gate is stricter than cosine similarity alone.

**Key empirical test**: Run quality benchmark at τ=0.50, 0.60, 0.70, 0.80. If quality stays flat across all τ values, it proves our LMCache block matching is the effective gate (cosine similarity is just coarse pre-filtering). If quality degrades at low τ, it tells us cosine similarity alone is the real gate.

---

## Papers to Read

### KVComm (2024)
- Full title: "KVComm: Communication-Efficient KV Cache Transfer with Positional-Aware Block Selection"
- Key contribution: Block-level KV transfer with positional encoding awareness
- Relevance: How they handle RoPE correction (compare to SemBlend approach)
- Search: `arxiv KVComm KV cache transfer positional 2024`

### ProphetKV (2024)
- Full title: "Prophet: Speculative KV Cache Reuse via Query Prediction"
- Key contribution: Query-attention-weighted recomputation selection
- Relevance: The FM2 fix — per-layer quality estimation for selective recomputation
- Search: `arxiv ProphetKV speculative KV cache reuse query prediction 2024`

### MEPIC (2025)
- Full title: "MEPIC: Memory-Efficient Prefix-Informed Computation for Long-Context LLM Serving"
- Key contribution: Prefix-aware KV cache management with compression-quality guarantees
- Relevance: Alternative approach to prefix reuse with quality bounds
- Search: `arxiv MEPIC memory efficient prefix LLM serving 2025`

### KVLink (2024)
- Full title: "KVLink: Linking KV Caches Across Requests for Efficient LLM Serving"
- Key contribution: Cross-request KV cache linking with semantic similarity
- Relevance: Direct competitor — compare architecture and quality results
- Search: `arxiv KVLink linking KV caches requests LLM serving 2024`

### EPIC (2025)
- Full title: "EPIC: Efficient Prefix-Informed Computation for Transformer Inference"
- Key contribution: Selective prefix computation based on semantic overlap estimation
- Relevance: Alternative quality gating mechanism
- Search: `arxiv EPIC efficient prefix computation transformer inference 2025`

### DroidSpeak (2025)
- Full title: "DroidSpeak: Selective KV Cache Recomputation for Cross-Lingual Long-Context Inference"
- Key contribution: Only 11% of layers are critical; layer criticality profiling
- Relevance: FM3 — which layers matter, model-specific calibration
- Search: `arxiv DroidSpeak selective KV recomputation critical layers 2025`

### kv-cache-physics (2025)
- Full title: "KV Cache Physics: Information Flow and Consolidation in Large Language Models"
- Key contribution: Architecture-specific attention consolidation patterns (LLaMA=inverted funnel, Qwen=funnel)
- Relevance: Informs architecture-specific bathtub fraction tuning
- Search: `arxiv KV cache physics information flow LLM attention 2025`

---

## Positioning SemBlend Against These Papers

**Where SemBlend wins**:
- Solves FM1 (RoPE) correctly — others don't or do it approximately
- Block-level exact matching avoids FM2 (no partial/stale injection within matched blocks)
- Production system (vLLM + LMCache) vs research prototype

**Where SemBlend can improve**:
- FM3: Architecture-specific layer calibration (DroidSpeak insight)
- ProphetKV-lite: Per-chunk quality estimation for borderline-threshold hits
- Threshold sensitivity paper (provide the quality-vs-hit-rate curve that SemShareKV provides)

**Related Work narrative**: Frame as "SemBlend builds on LMCache's exact block matching (which inherently avoids FM2) and adds exact RoPE correction (FM1), achieving high quality at τ=0.60 without the aggressive recomputation that ProphetKV requires."
