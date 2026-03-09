# Reading List

## Priority 1: Direct Competitors
- [ ] **SemShareKV** — Semantic KV Cache Sharing for Large Language Models (2025)
  - Token-level LSH matching, RoPE-aware E-cache, 3 models x 9 datasets
  - Key comparison target for SemBlend paper
- [ ] **CacheBlend** — Fast Large Language Model Serving with Cached Knowledge Fusion (EuroSys'25)
  - Selective recomputation for KV cache blending
  - Potential baseline comparison

## Priority 2: Related Systems
- [ ] **LMCache** — KV cache management for vLLM
  - SemBlend's underlying KV transport layer
  - Understand chunk storage, retrieval, connector API
- [ ] **vLLM** — PagedAttention and prefix caching
  - Production baseline for prefix-match KV reuse

## Priority 2.5: Primer Papers (March 2026 — KV Reusability Analysis)

These papers were cited in a research primer analyzing failure modes of CacheBlend-style correction. Notes in `primer_kvreuse_notes.md`.

- [ ] **KVComm** (2024) — KV cache transfer with positional-aware block selection; how RoPE correction is done in other systems
- [ ] **ProphetKV** (2024) — "Prophet: Speculative KV Cache Reuse via Query Prediction"; query-attention-weighted recomputation selection (FM2 fix)
- [ ] **MEPIC** (2025) — Memory-Efficient Prefix-Informed Computation; prefix reuse with quality bounds
- [ ] **KVLink** (2024) — Cross-request KV cache linking with semantic similarity; direct competitor
- [ ] **EPIC** (2025) — Efficient Prefix-Informed Computation; selective prefix computation with semantic overlap
- [ ] **DroidSpeak** (2025) — Only 11% of layers are critical; per-model layer criticality profiling (FM3 fix)
- [ ] **kv-cache-physics** (2025) — LLaMA=inverted funnel (early layers critical), Qwen=funnel (late layers critical); architecture-specific KV importance profiles

## Priority 3: Theoretical Foundations
- [ ] **RoFormer** — Enhanced Transformer with Rotary Position Embedding (Su et al., 2024)
  - Theoretical basis for RoPE delta correction
  - Exact position correction proof
- [ ] **SnapKV** — LLM Knows What You Are Looking For Before Generation
  - KV cache compression baseline (SemShareKV compares against this)
- [ ] **PyramidKV** — Dynamic KV Cache Compression
  - Another baseline from SemShareKV evaluation
