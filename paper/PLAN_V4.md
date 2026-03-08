# SemBlend Implementation Plan V4 — Local-First Architecture

## Architectural Shift

**V3 Architecture (gateway-dependent):**
```
vLLM → SemBlendConnectorV1 → HTTP → Synapse Gateway → embedding + ANN search → transfer plan
                                        ~100-150ms round-trip
```

**V4 Architecture (local-first):**
```
vLLM → SemBlendConnectorV2 → in-process donor discovery → LMCache chunk-swap
                                   ~5ms total, zero network hops

Optional scaled tier (async, non-blocking):
  SemBlendConnectorV2 → Synapse Gateway (CAGRA, 100K+ donors)
```

## Key Insight

The gateway dependency in V3 added ~150ms of network overhead to every donor lookup — embedding via TEI HTTP (150ms) + gateway round-trip (~10ms). This makes SemBlend viable only for prompts >2K tokens on 7B models where cold prefill TTFT (>3s) dwarfs the overhead.

The local-first approach eliminates all network calls from the hot path:
- **MiniLM-L6 CPU embedder**: ~5ms, 80MB model, no GPU needed
- **Numpy cosine similarity**: ~1ms for <10K donors, in-process
- **rapidfuzz alignment**: ~3ms compiled C++ for 8K token alignment
- **LMCache chunk-swap**: existing injection path, zero additional overhead

Total: **~5ms** overhead vs **~150ms** with gateway. This makes SemBlend viable even for 1.5B models at 256 tokens.

## Phase 0: Quality Validation (1 week)

**Goal:** Prove chunk-swap injection doesn't degrade output quality before building anything else.

### 0.1 Ground Truth Collection
- Run Qwen2.5-7B-AWQ on A10G with 100 RAG prompts (8K tokens each)
- Record full output text + per-layer KV fingerprints (L2 norm, mean K vector)
- Create test sets: REORDER (same chunks different order), PARTIAL (75% overlap), PARAPHRASE (rewritten chunks)

### 0.2 Injection Quality Measurement
- For each test: find donor, inject KV via LMCache chunk-swap, record output
- Compute: ROUGE-L, BERTScore, exact-match accuracy vs ground truth
- **Pass criterion:** ROUGE-L ≥ 0.90 on REORDER, ≥ 0.85 on PARTIAL

### 0.3 Bathtub Curve Validation
- Compare predicted layer deviations (Eq. 4 from paper) against measured per-layer KV fingerprint deltas
- Identify if first/last 12.5% recomputation heuristic holds for Qwen2.5 architecture
- Document per-model calibration requirements

### Deliverables
- Quality validation report with ROUGE-L/BERTScore numbers
- Per-model bathtub curve parameters (if validated) or alternative heuristic
- Go/no-go decision for Phase 1

---

## Phase 1: SemBlend Local — In-Process Fast Path (2 weeks)

**Goal:** Fully self-contained donor discovery in the vLLM process. Zero external dependencies.

### 1.1 Local Embedder (`semblend_local/embedder.py`)
```python
class LocalEmbedder:
    """In-process sentence embedding using sentence-transformers."""

    def __init__(self, model_name="all-MiniLM-L6-v2", device="cpu"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        # 80MB model, loads in ~2s, embeds in ~5ms

    def embed(self, text: str) -> np.ndarray:
        return self.model.encode(text, normalize_embeddings=True)

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(texts, normalize_embeddings=True)
```

**Why MiniLM-L6 over jina-v4:**
- 80MB vs 2.4GB — fits comfortably in CPU memory alongside vLLM
- 384-dim vs 1024-dim — sufficient for donor discovery (not fine-grained retrieval)
- ~5ms CPU vs ~150ms GPU+network — 30x faster for the use case
- Quality is adequate: paraphrase detection doesn't need state-of-art retrieval embeddings

**Fallback:** SimHash (4 hash functions on token trigrams) for sub-millisecond pre-filtering when embedding is unavailable. SimHash catches high-overlap donors (>80% token overlap) without any model.

### 1.2 In-Process Donor Store (`semblend_local/donor_store.py`)
```python
class NumpyDonorStore:
    """In-memory donor store with numpy cosine similarity."""

    def __init__(self, max_entries=10_000, dim=384):
        self.embeddings = np.zeros((max_entries, dim), dtype=np.float32)
        self.entries: OrderedDict[str, DonorEntry] = OrderedDict()
        self.count = 0

    def add(self, entry: DonorEntry) -> None:
        # LRU eviction when full
        if self.count >= self.max_entries:
            oldest_key = next(iter(self.entries))
            self.remove(oldest_key)
        self.embeddings[self.count] = entry.embedding
        self.entries[entry.request_id] = entry
        self.count += 1

    def search(self, query_emb: np.ndarray, top_k=5, threshold=0.60):
        if self.count == 0:
            return []
        # Numpy cosine similarity: ~1ms for 10K entries
        scores = self.embeddings[:self.count] @ query_emb
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        results = [(idx, scores[idx]) for idx in top_indices if scores[idx] >= threshold]
        return sorted(results, key=lambda x: -x[1])
```

**Why numpy over faiss:**
- For <10K entries, numpy cosine is ~1ms — faiss adds complexity without speed benefit
- No additional dependency
- Scales linearly: 10K entries × 384 dims = 15MB — trivial memory

### 1.3 Token Alignment (`semblend_local/alignment.py`)
```python
from rapidfuzz.distance import Opcodes

def compute_transfer_plan(donor_tokens, target_tokens, chunk_size=256):
    """Compute which chunks to copy vs recompute."""
    opcodes = Opcodes.get(donor_tokens, target_tokens)
    # opcodes gives aligned regions: equal, replace, insert, delete
    copy_positions = []
    placeholder_positions = []
    for op, d_start, d_end, t_start, t_end in opcodes:
        if op == "equal":
            copy_positions.extend(range(t_start, t_end))
        else:
            placeholder_positions.extend(range(t_start, t_end))

    # Align to LMCache chunk boundaries (256 tokens)
    chunk_aligned_copy = align_to_chunks(copy_positions, chunk_size)
    return TransferPlan(
        copy_positions=chunk_aligned_copy,
        placeholder_positions=placeholder_positions,
        reuse_ratio=len(chunk_aligned_copy) / max(len(target_tokens), 1),
    )
```

**Why rapidfuzz over Wagner-Fischer:**
- rapidfuzz uses C++ Levenshtein implementation — ~3ms for 8K tokens vs ~2-4s pure Python
- Same edit-distance semantics, compiled execution
- Falls back cleanly: `pip install rapidfuzz` (pure C++ wheel, no CUDA dependency)

### 1.4 Updated Connector (`semblend_connector_v2.py`)
- Replaces `SemBlendConnectorV1` in vLLM config
- In `get_num_new_matched_tokens()`:
  1. LMCache exact-prefix check (unchanged)
  2. On miss: `LocalEmbedder.embed(prompt_text)` — 5ms
  3. `NumpyDonorStore.search(embedding)` — 1ms
  4. `compute_transfer_plan(donor_tokens, target_tokens)` — 3ms
  5. Monkeypatch LMCache lookup to return donor's chunk hit count
  6. Return matched token count → vLLM loads donor KV via LMCache
- In `save_kv_layer()`:
  - Save to LMCache (CPU offload) — unchanged
  - Compute embedding, add to NumpyDonorStore — async, non-blocking

### 1.5 Smoke Test
- Integration test: seed 10 prompts → query REORDER variant → verify donor found + KV injected
- Latency target: <10ms total SemBlend overhead (embed + search + align)
- Correctness: ROUGE-L ≥ 0.90 on REORDER

### Deliverables
- `services/vllm/synapse_kv_connector/semblend_local/` package
- `semblend_connector_v2.py` replacing V1
- Unit tests: embedder, donor store, alignment, connector
- Integration test: full pipeline on Qwen2.5-7B-AWQ

---

## Phase 2: Evaluation & Paper Numbers (1 week)

**Goal:** Collect empirical numbers for the paper comparing Local vs Gateway paths.

### 2.1 Benchmark Matrix
| Scenario | Tokens | Model | GPU | Metric |
|----------|--------|-------|-----|--------|
| REORDER | 8K | Qwen2.5-7B-AWQ | A10G | TTFT, ROUGE-L |
| PARTIAL (75%) | 8K | Qwen2.5-7B-AWQ | A10G | TTFT, ROUGE-L |
| PARAPHRASE | 8K | Qwen2.5-7B-AWQ | A10G | TTFT, ROUGE-L |
| COLD (different domain) | 8K | Qwen2.5-7B-AWQ | A10G | TTFT (verify no injection) |
| ShareGPT diverse | mixed | Qwen2.5-7B-AWQ | A10G | Effective TTFT |

### 2.2 Latency Decomposition
Compare overhead breakdown:

| Component | V3 (Gateway) | V4 (Local) |
|-----------|-------------|------------|
| Embedding | 150ms (jina-v4 HTTP) | 5ms (MiniLM CPU) |
| ANN search | <1ms (CAGRA) | 1ms (numpy cosine) |
| Alignment | 3ms (rapidfuzz) | 3ms (rapidfuzz) |
| Network | ~10ms round-trip | 0ms |
| **Total** | **~165ms** | **~9ms** |

### 2.3 Quality Comparison
- Same test prompts, Local embedder (MiniLM-384) vs Gateway embedder (jina-v4-1024)
- Measure: donor match rate, cosine similarity distribution, ROUGE-L post-injection
- Hypothesis: MiniLM-384 is sufficient for donor discovery (not a retrieval task)

### Deliverables
- Benchmark results JSON files
- Paper tables with bootstrap 95% CIs
- Latency decomposition figure

---

## Phase 3: CAGRA Scaled Tier — Multi-Instance (2 weeks, deferred)

**Goal:** Enable cross-instance donor discovery for deployments with >10K unique prompts.

### Architecture Decision: Option A vs Option B

**Option A: cuVS Python bindings**
```python
import cuvs.cagra as cagra
from pylibraft.common import DeviceResources
# Risk: RMM/PyTorch allocator conflict on shared GPU
```

**Option B: PyO3 on existing Rust code**
```python
import synapse_cagra  # PyO3 module wrapping crates/synapse-l0-gpu
# Benefit: reuses existing CAGRA pool pattern, cudarc allocation, no RMM
```

**Recommendation: Start with Option A, fall back to B if allocator conflicts arise.**

Rationale:
- Phase 3 is for donors >10K — this is a background/async operation, not hot path
- cuVS Python bindings are simpler and maintained by NVIDIA
- Allocator conflict only matters if CAGRA search runs on the same GPU as vLLM inference
- Solution: run CAGRA on the proxy/gateway GPU (separate pod), not on vLLM GPU
- Option B (PyO3) is the right choice if/when we want in-process CAGRA on the vLLM GPU

### 3.1 Two-Tier Architecture
```
Tier 1 (Local, <10K): NumpyDonorStore → in-process, ~1ms
Tier 2 (Remote, 100K+): CAGRA via Gateway → async, ~50ms

Routing:
  1. Always check Tier 1 first (in-process, no latency)
  2. On Tier 1 miss: optionally query Tier 2 via async HTTP
  3. Tier 2 results pre-warm Tier 1 for future queries
```

### 3.2 Cross-Instance Donor Sharing
- Each vLLM instance publishes donor metadata to gateway on `save_kv_layer()`
- Gateway aggregates donors from all instances in CAGRA index
- On Tier 2 hit: transfer plan points to the source instance's LMCache
- P2P KV transfer via LMCache's existing distributed mechanism

### 3.3 Implementation
- Gateway: `POST /api/v1/kv-cache/semblend/register-donor` (metadata only, no KV data)
- Gateway: `POST /api/v1/kv-cache/semblend/search` (returns donor ID + source instance)
- Connector: async background task calls gateway on local miss
- Connector: pre-warms local NumpyDonorStore with gateway results

### Deliverables
- Gateway semblend endpoints (Rust)
- Connector Tier 2 integration
- Cross-instance integration test

---

## Phase 4: PartialAttention Production Integration (3 weeks, deferred)

**Goal:** Enable non-contiguous KV injection for higher-quality semantic donor reuse.

### Current State
- Triton kernels implemented and unit-tested (17 tests passing)
- Integration with vLLM model runner requires per-model hooks
- LMCache chunk-swap provides contiguous-prefix injection today

### 4.1 When PartialAttention Is Needed
- Current chunk-swap: aligns donor KV to 256-token chunk boundaries → some tokens wasted
- PartialAttention: per-position, per-layer control → no wasted tokens
- Value: matters when donor overlap is sparse (e.g., 3 matching paragraphs scattered across a 10-paragraph prompt)
- Not needed for REORDER (high contiguous overlap) or PARTIAL (chunk-aligned overlap)

### 4.2 Implementation Path
- Hook `PartialAttentionHook` into vLLM's `ModelRunner` via `model_runner_hook.py`
- Use `attention_patch.py` to intercept per-layer attention computation
- Triton kernels execute on the existing CUDA stream (no additional sync)

### 4.3 Quality Budget
- With PartialAttention + bathtub recomputation: 95%+ quality at 35% computation
- Without PartialAttention (current): 100% of donor tokens injected, no recomputation → quality depends on donor similarity

---

## Execution Timeline

| Week | Phase | Key Deliverable | Blocked By |
|------|-------|----------------|------------|
| 1 | 0: Quality Validation | Go/no-go report | Nothing |
| 2-3 | 1: SemBlend Local | `semblend_connector_v2.py` working | Phase 0 pass |
| 4 | 2: Evaluation | Paper numbers, tables | Phase 1 |
| 5-6 | 3: CAGRA Scaled | Cross-instance discovery | Phase 1 |
| 7-9 | 4: PartialAttention | Production integration | Phase 1 |

---

## Paper Strategy

### Section Structure (updated)
1. **Introduction** — motivate semantic KV donor reuse, position vs LMCache
2. **Background** — existing systems taxonomy (keep existing table)
3. **Design: SemBlend Local** (NEW) — in-process architecture, ~5ms overhead
   - Local embedder (MiniLM-L6, CPU)
   - Numpy donor store
   - rapidfuzz alignment
   - LMCache chunk-swap injection
4. **Design: SemBlend Scaled** — CAGRA-backed tier for 100K+ donors
   - Two-tier routing (local → gateway)
   - Cross-instance donor sharing
5. **Quality Model** — bathtub curve, calibration (keep existing)
6. **PartialAttention** — Triton kernels (keep existing)
7. **Evaluation** — both local and gateway paths
   - Local: ~5ms overhead, viable for small models
   - Gateway: ~150ms overhead, validated 4.8-5.5x speedup
   - Quality: ROUGE-L, BERTScore
8. **Discussion** — operating envelope, when each tier helps
9. **Conclusion**

### Key Claims for Paper
1. **~5ms in-process overhead** makes semantic KV reuse viable for any model size
2. **4.8-5.5x TTFT reduction** on reordered/partial RAG contexts (measured)
3. **Two-tier architecture** scales from single-instance (<10K donors) to multi-instance (100K+)
4. **Embedding-based discovery essential** — Jaccard fails for partial overlap (measured)
5. **Complementary to LMCache** — activates only on exact-prefix miss
