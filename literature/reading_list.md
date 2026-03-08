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

## Priority 3: Theoretical Foundations
- [ ] **RoFormer** — Enhanced Transformer with Rotary Position Embedding (Su et al., 2024)
  - Theoretical basis for RoPE delta correction
  - Exact position correction proof
- [ ] **SnapKV** — LLM Knows What You Are Looking For Before Generation
  - KV cache compression baseline (SemShareKV compares against this)
- [ ] **PyramidKV** — Dynamic KV Cache Compression
  - Another baseline from SemShareKV evaluation
