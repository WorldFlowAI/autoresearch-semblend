# autoresearch-semblend

Autonomous research loop for SemBlend — a semantic KV-cache reuse system for LLM inference. You are a fully autonomous research agent. Your job is to fix logic holes in SemBlend, run experiments with large datasets, improve the paper, and iterate until every claim is empirically validated.

**Goal**: Resolve all identified logic and research holes in the SemBlend paper. Every claimed contribution must have end-to-end empirical validation. No quality regressions. No hand-waving.

## Setup

1. **Branch**: Work on `autoresearch/<tag>` from current master.
2. **Read all source files** before any changes:
   - `synapse_kv_connector/*.py` — SemBlend vLLM connector (you modify these)
   - `benchmarks/e2e/*.py` — benchmark scripts (you modify these)
   - `paper/semblend.tex` — the paper (you modify this)
   - `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/*.py` — production SemBlend (you modify these, then copy to working copy)
3. **Verify infrastructure**: `bash infra/ensure_gpu.sh`
4. **Confirm and go**: Start the experiment loop.

---

## What You Can Modify

- `synapse_kv_connector/*.py` — working copy of SemBlend connector
- `synapse_kv_connector/tests/*.py` — tests
- `benchmarks/e2e/*.py` — benchmark scripts
- `paper/semblend.tex` — the paper
- `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/*.py` — production SemBlend core (apply fixes here FIRST, then copy to working copy)
- `infra/values-autoresearch.yaml` — Helm values
- `results.tsv` — experiment log

## What You Cannot Modify

- `prepare.py`, `run_experiment.py` — read-only
- `infra/ensure_gpu.sh`, `infra/deploy.sh`, `infra/teardown.sh` — infrastructure scripts

---

## Critical Research Gaps (In Priority Order)

These are **blocking** issues that must be resolved. Each gap has a specific diagnosis, required fix, and validation criterion.

---

### Gap 1: Semantic Staleness — PPL=1.27 Outlier (MUST FIX)

**Problem**: The paper claims a "quality floor guarantee: no input can experience worse quality than cold baseline." But a persistent outlier at PPL ratio 1.270 directly contradicts this. The outlier occurs when a donor's 256-token chunk is token-identical to a target chunk but comes from a **different semantic position** in the article (incident description vs. institutional response). Chunk-hash identity does not guarantee semantic position alignment.

**Root cause in code**: `alignment.py:compute_chunk_alignment()` matches chunks purely by MD5 hash of token IDs (line 55-59). There is no check for whether the matched chunk is at a semantically appropriate position.

**Required fix — implement per-chunk semantic position gate**:

The fix lives in the alignment and pipeline code. Before accepting a chunk match, validate that the donor chunk's position context is compatible with the target chunk's position context.

**Implementation approach (in `alignment.py` and `pipeline.py`)**:

1. **K-norm fingerprint gate** (lightweight, ~0.1ms): When registering a donor, capture per-chunk K-norm fingerprints (sum of L2 norms across a few key layers). At match time, compare the donor chunk's K-norm fingerprint with a freshly computed estimate from the target's surrounding context. If the K-norm diverges beyond a threshold (e.g., >2σ), reject the chunk match even though the hash matches.

2. **Surrounding-context hash gate** (simpler alternative): For each chunk, also hash the preceding and following chunks. A chunk match is accepted only if at least one adjacent chunk also matches (confirming the chunk is at a similar structural position in the document). This catches the "same paragraph, different article position" failure mode.

3. **Per-chunk cosine similarity** (most robust): Embed each 256-token chunk independently (not the full prompt). Accept chunk match only if the chunk's local embedding similarity exceeds a sub-threshold (e.g., ≥0.85). This adds ~1ms for MiniLM on 5-10 chunks but catches semantic staleness.

**Validation**:
- Reproduce the PPL=1.27 outlier scenario (CNN/DM same-story pairs)
- Show the fix rejects the problematic chunk match
- Re-run the full threshold sweep and confirm PPL=1.27 is gone
- Re-run cross-dataset quality benchmark and confirm no new regressions
- **Success criterion**: No single sample exceeds PPL ratio 1.10 across all datasets

**Apply fix in BOTH**:
- `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/alignment.py`
- `synapse_kv_connector/alignment.py` (working copy)

---

### Gap 2: RoPE Delta Correction Never Activated (Δ=0 for All Benchmarks)

**Problem**: RoPE delta correction is listed as contribution #4 but is never activated in any benchmark. The paper admits: "In the current 256-token LMCache chunk deployment, paragraph-level reordering aligns KV reuse to chunk boundaries so Δ=0 for matched pairs, making correction a no-op."

**Root cause**: The chunk-swap injection path (`semblend_connector.py`) replaces the target's entire token sequence with the donor's tokens, so LMCache loads donor KV at the donor's original positions (all Δ=0 by construction). RoPE correction only matters for the **PartialAttention path** (non-contiguous injection where donor chunks land at different target positions).

**Why PartialAttention matters — this is not academic**: The chunk-swap path is inherently limited. It can only reuse donor KV when the ENTIRE donor token sequence is loaded as a block. This means:
- **No partial document reuse**: If a target prompt contains document A + document B, but the donor only has document A, chunk-swap cannot inject just document A's KV. It's all-or-nothing.
- **No sub-chunk matching**: If two prompts share 200 tokens in the middle but differ at the start and end, chunk-swap has zero reuse. PartialAttention can inject those 200 tokens at their correct positions.
- **No cross-position reuse**: RAG systems frequently prepend different context chunks before the same user query. The query tokens appear at different absolute positions in different requests. Chunk-swap fails (different token sequences). PartialAttention with RoPE correction handles this by placing shared tokens at their new positions and correcting the baked-in RoPE rotation.
- **Enables sentence-level KV reuse**: Instead of requiring entire 256-token chunk matches, PartialAttention enables fine-grained token-level matching via Levenshtein alignment — dramatically increasing the set of reusable KV in real workloads where exact chunk matches are rare.

Without PartialAttention + RoPE, SemBlend is fundamentally limited to the same coverage as LMCache (exact chunk matches). PartialAttention is what makes semantic KV reuse truly semantic rather than just a fancier lookup key for the same exact-match mechanism.

**Required fix — activate PartialAttention path with sub-chunk alignment for E2E benchmarks**:

1. **Enable PartialAttention E2E**: Set `SEMBLEND_USE_PARTIAL_ATTN=1` in deployment. This path uses the Triton scatter kernels to place donor KV at target positions with explicit RoPE correction.

2. **Create sub-chunk injection scenarios**: Build test cases where donor content appears at different absolute positions in the target:
   - **Shifted-prefix**: Same document body with different-length instruction prefixes (100 vs 500 tokens). Body chunks start at different offsets → Δ=100-400 for each chunk.
   - **Interleaved-documents**: Target has 2 documents A+B, donor has just document A. Document A's chunks appear at different positions.
   - **Sentence-level injection**: Use Levenshtein alignment (not chunk hash) to inject individual matching sentences at their target positions.

3. **Benchmark with PartialAttention on vs off**:
   - `SEMBLEND_USE_PARTIAL_ATTN=1` + `SEMBLEND_DISABLE_ROPE_CORRECTION=0` (full system)
   - `SEMBLEND_USE_PARTIAL_ATTN=1` + `SEMBLEND_DISABLE_ROPE_CORRECTION=1` (no RoPE)
   - `SEMBLEND_USE_PARTIAL_ATTN=0` (chunk-swap baseline, current default)

**Validation**:
- Show that with PartialAttention + RoPE correction, PPL ratio ≈1.0 on shifted-prefix scenarios
- Show that WITHOUT RoPE correction, PPL ratio degrades proportionally to Δ
- Show TTFT speedup for the PartialAttention path vs cold prefill
- **Success criterion**: At Δ≥256 tokens, uncorrected PPL > 1.10 while corrected PPL < 1.05

**Create benchmark**: `benchmarks/e2e/partial_attention_bench.py`
- Tests all three scenarios (shifted-prefix, interleaved, sentence-level)
- Measures TTFT, PPL ratio, hit rate for each
- Compares PartialAttention with/without RoPE correction

---

### Gap 3: Bathtub Recomputation Shown Unnecessary by Own Data

**Problem**: The paper's own data shows: "With RoPE correction enabled, all per-layer deviations are below 0.2%" and "the deviation profile is essentially flat rather than bathtub-shaped." The ablation shows 2.6% speedup difference (5.51× vs 5.66×). Yet bathtub is contribution #3.

**Root cause**: Same as Gap 2 — the chunk-swap path has Δ=0, so there's no positional drift and no attention pattern drift. All layers receive correctly-positioned, token-identical KV. The bathtub curve only matters when injecting **semantically similar but not identical** KV at non-contiguous positions.

**Required fix**: The bathtub recomputation becomes meaningful ONLY in the PartialAttention path (Gap 2). Once PartialAttention is benchmarked with real Δ≠0 scenarios:

1. **Run bathtub ablation on PartialAttention path**:
   - PartialAttention + RoPE + bathtub recomputation → measure PPL
   - PartialAttention + RoPE + NO bathtub (all layers reused) → measure PPL
   - PartialAttention + NO RoPE + NO bathtub → measure PPL

2. **Show the contribution hierarchy**:
   - RoPE correction handles position encoding drift (exact)
   - Bathtub recomputation handles attention pattern drift (heuristic)
   - Together they should produce lower PPL than either alone

3. **If bathtub still shows negligible impact**: Reframe it honestly as an insurance mechanism for edge cases rather than a core contribution. Downgrade from contribution #3 to a "robustness enhancement."

**Validation**:
- On shifted-prefix scenarios with Δ≥256: show bathtub recomputation reduces PPL vs no recomputation
- On sentence-level injection (high mismatch): show bathtub makes a meaningful (>3%) quality difference
- **Success criterion**: Identify at least one realistic scenario where bathtub recomputation reduces PPL by ≥5%

---

### Gap 4: Embedding Model Ablation (Only MiniLM-L6-v2 Tested)

**Problem**: MiniLM-L6-v2 has a 512-token input limit, producing only 384-dim embeddings. The sliding window sampling (40/30/30 head/mid/tail) was added to compensate, but no alternative embedders have been tested.

**Required experiments**:

1. **Test these embedders**:
   - `all-MiniLM-L6-v2` (current, 384-dim, 512 tokens) — baseline
   - `all-MiniLM-L12-v2` (384-dim, 512 tokens, deeper) — slightly more accurate
   - `nomic-embed-text-v1.5` (768-dim, 8192 tokens) — long-context native
   - `BAAI/bge-small-en-v1.5` (384-dim, 512 tokens) — alternative small
   - `Alibaba-NLP/gte-base-en-v1.5` (768-dim, 8192 tokens) — long-context

2. **For each embedder, measure**:
   - Embedding latency (CPU and ONNX GPU)
   - Hit rate on 5 datasets at 8K tokens (n=8 per dataset)
   - TTFT speedup (same datasets)
   - PPL ratio (same datasets)
   - Memory footprint of donor store at N=10K

3. **Key hypothesis**: Long-context embedders (nomic, gte) should eliminate the need for sliding window sampling and improve hit rate on partial-overlap scenarios. They may also improve the semantic staleness gate (Gap 1).

**Validation**:
- Show hit rate comparison across all embedders
- Identify whether a larger embedder improves partial-overlap hit rate
- **Success criterion**: Find at least one embedder that improves hit rate by ≥10% on partial scenarios while maintaining <20ms latency

**Create benchmark**: `benchmarks/e2e/embedder_ablation_bench.py`

---

### Gap 5: LMCache Chunk Size Treated as Fixed (256 Tokens)

**Problem**: The paper repeatedly identifies 256-token chunk boundaries as a source of problems (boundary shifts from instruction prefix changes, etc.) but never experiments with other chunk sizes.

**Required experiments**:

1. **Test chunk sizes**: 64, 128, 256 (current), 512 tokens
2. **For each chunk size, measure**:
   - Hit rate on cross-instruction benchmark (where boundary shifts cause 0% hits at 256)
   - TTFT speedup
   - PPL ratio
   - KV transfer overhead (smaller chunks = more chunks = more overhead?)
   - Cross-dataset hit rate

3. **Key hypothesis**: Smaller chunks (128 or 64) should improve hit rate for cross-instruction scenarios because boundary shifts affect fewer tokens. But smaller chunks may increase KV transfer overhead and reduce quality.

**Implementation**: `LMCACHE_CHUNK_SIZE` is in `alignment.py:23`. Also need to configure LMCache itself to use the new chunk size (LMCache env var: `LMCACHE_CHUNK_SIZE`).

**Validation**:
- Show hit rate vs chunk size tradeoff
- **Success criterion**: Determine optimal chunk size or demonstrate why 256 is correct

**Create benchmark**: `benchmarks/e2e/chunk_size_ablation_bench.py`

---

### Gap 6: No Evaluation Beyond Summarization

**Problem**: All 5 datasets are NLP summarization tasks. No code generation, multi-turn dialogue, or diverse workloads. WildChat validation uses only n=30 pairs (10 hits).

**Required experiments**:

1. **Code generation workload**:
   - Use HumanEval or MBPP prompts as donors/targets
   - Create variations: same coding problem, different natural language description
   - Measure hit rate, TTFT speedup, and functional correctness (pass@1)
   - n≥1000 pairs

2. **Multi-turn dialogue**:
   - Use ShareGPT dataset: consecutive turns from same conversation
   - Donor = turn N context, target = turn N+1 context (growing prefix)
   - Measure hit rate, speedup, quality across 200+ conversation pairs

3. **WildChat at scale**:
   - Sample 500+ long-context pairs (≥6K chars) instead of 30
   - Stratify by similarity bucket (0.5-0.6, 0.6-0.7, 0.7-0.8, 0.8-0.9, 0.9-1.0)
   - Report per-bucket hit rate and speedup
   - n≥500 pairs total

4. **Customer support / search**:
   - Use MSMarco or NQ (Natural Questions) passages
   - Same passage retrieved for different user queries
   - Measure SemBlend hit rate across query reformulations

**Validation**:
- Demonstrate SemBlend works beyond summarization
- Show realistic hit rates on production-like workloads
- **Success criterion**: ≥25% hit rate on at least 2 non-summarization workloads with ≥1.5× hit-only speedup

**Create benchmarks**:
- `benchmarks/e2e/code_gen_bench.py`
- `benchmarks/e2e/multiturn_dialogue_bench.py`
- `benchmarks/e2e/wildchat_large_bench.py`

---

### Gap 7: Break-Even Hit Rate and 0.59× Regression

**Problem**: LLaMA CNN/DM 8K shows 0.59× overall speedup (SemBlend makes things WORSE). At 25% hit rate, the ~8ms pipeline overhead on misses outweighs the savings from hits. The paper never models the break-even $P_h$.

**Required analysis and experiments**:

1. **Analytical break-even model**:
   - SemBlend adds overhead_miss ms on every miss (embedding + search + alignment, ~8ms)
   - SemBlend adds overhead_hit ms on every hit (same pipeline + KV transfer, ~40ms)
   - Cold TTFT = C ms
   - Hit TTFT = H ms (C >> H when context is long)
   - Break-even: P_h × (C - H - overhead_hit) > (1 - P_h) × overhead_miss
   - Solve for P_h at each context length

2. **Empirical validation**:
   - Run controlled hit-rate sweep at 2K, 4K, 8K, 16K tokens
   - At each length, artificially control hit fraction: 0%, 10%, 25%, 50%, 75%, 100%
   - Measure actual blended TTFT vs cold baseline
   - Plot: blended TTFT / cold TTFT vs P_h at each length
   - Find the crossover point where SemBlend becomes net-positive

3. **Reproduce and explain the 0.59× regression**:
   - Run LLaMA CNN/DM 8K with n=16 (double current sample size)
   - Analyze per-sample TTFT: which specific samples show overhead?
   - Is the overhead from pipeline processing on misses, or from LMCache contention?

**Validation**:
- Derive analytical break-even formula
- Show empirical break-even at each context length
- **Success criterion**: Determine the break-even P_h (e.g., "SemBlend is net-positive when P_h ≥ X% at 8K tokens")

**Create benchmark**: `benchmarks/e2e/breakeven_analysis_bench.py`

---

### Gap 8: Statistical Rigor (n=5-8 Per Condition)

**Problem**: Most experiments use n=5-8 samples per condition with no statistical significance tests. Bootstrap CIs are shown for one configuration only.

**Required improvements**:

1. **Increase sample sizes**: All key experiments should use n≥32 (preferably n=32) per condition
2. **Report 95% bootstrap CIs** for all TTFT and PPL measurements (not just one table)
3. **Compute effect sizes** (Cohen's d) for SemBlend vs cold comparisons
4. **Add Wilcoxon signed-rank tests** for paired comparisons (same prompt, cold vs SemBlend)

**Implementation**: Add `--runs` flag to all benchmarks defaulting to 16. Add `scipy.stats.bootstrap` CI computation to all result reporting.

---

### Gap 9: No Baseline Comparisons Against Prior Systems

**Problem**: The paper compares only against cold prefill — the weakest possible baseline. SemShareKV claims 6.25× at 5K tokens. ProphetKV claims 4.6-4.7× at 16K. CacheBlend reports 2.2-3.3× TTFT reduction. No head-to-head comparison exists.

**Required approach**:

Since reproducing competitor systems end-to-end is impractical (different codebases, models, hardware), use a multi-pronged comparison strategy:

1. **Normalized comparison table**: All competitor numbers are on A100 (80GB, ~3-6× more expensive than A10G). Normalize by:
   - Hardware cost per speedup-hour
   - Speedup at equivalent prompt lengths (5K for SemShareKV, 16K for ProphetKV)
   - Coverage: what fraction of workload each system can cache (exact-only vs semantic)

2. **LMCache-only baseline** (we CAN run this): Deploy vLLM with LMCache enabled but SemBlend disabled. Run the SAME benchmarks on the SAME datasets. This directly quantifies what SemBlend adds over exact-prefix caching.
   - Configure: `SEMBLEND_ENABLED=0` in deployment
   - Run all 5 datasets at 2K/4K/8K
   - Measure hit rate and TTFT for LMCache-only
   - Expected: 0% hits on REORDER, PARAPHRASE, cross-instruction (LMCache only does exact prefix)
   - This proves the "semantic gap" that SemBlend fills

3. **vLLM prefix cache baseline**: Already partially done (Table 2). Expand to all datasets.

4. **Published numbers comparison table**: Create a structured table:
   | System | Hardware | Prompt Len | Speedup | Quality | Coverage |
   - Use exact numbers from published papers
   - Note hardware differences explicitly
   - Highlight SemBlend's unique coverage of semantic variations

**Validation**:
- LMCache-only E2E benchmark on all 5 datasets
- Structured comparison table in paper
- **Success criterion**: Show clear additive value of SemBlend over LMCache-only

---

### Gap 10: PPL Discrepancy — 1.000±0.001 vs 1.171

**Problem**: The variation sensitivity experiment shows PPL=1.000±0.001 across all 7 types. But Table 8 shows PPL up to 1.171. These contradict each other and the paper doesn't explain why.

**Root cause**: The experiments measure different things:
- Variation sensitivity uses `max_tokens=32` (TTFT benchmark, minimal generation)
- Table 8 uses `max_tokens=256` (extended generation)
- Variation sensitivity measures cold-fallback PPL on low-hit scenarios (PPL≡1.0 by construction when miss → cold)
- Table 8 measures actual SemBlend-injected outputs including miss-run outliers

**Required fix**:

1. **Re-run variation sensitivity with max_tokens=256** to get directly comparable PPL
2. **Clearly label in paper** what each PPL measures:
   - "Per-sample PPL on successful KV injection" vs "mean PPL including miss-run outliers"
3. **Decompose Table 8**: For each cell, report both mean PPL AND hit-only PPL separately
4. **Remove or qualify the PPL=1.000±0.001 claim** — if it only applies to cold-fallback outputs, it's tautological (cold PPL ≡ 1.0)

**Validation**:
- Consistent PPL measurement methodology across all experiments
- Clear labeling preventing reader confusion
- **Success criterion**: All PPL comparisons in the paper use the same generation length and methodology

---

### Gap 11: Cross-Instruction Benchmark Is Self-Fulfilling

**Problem**: Table 3 shows 100% hit rate on cross-instruction RAG. But instruction variants change only 9-25 characters out of ~32,000 characters of article body. The embeddings are dominated by the identical body text. 100% hit rate is trivially expected by construction.

**100% hit rates should not appear in the paper** — they are artifacts of controlled benchmarks, not indicative of real workloads.

**Required fix**:

1. **Create realistic cross-instruction scenarios** where instructions differ substantially:
   - Different instruction lengths (50 vs 500 tokens, not 50 vs 55)
   - Different task types: "summarize this article" vs "extract key facts" vs "generate questions about this article" vs "translate this article"
   - Different system prompts: ChatGPT-style vs Claude-style vs custom enterprise
   - These should produce genuinely different embeddings — not trivially similar ones

2. **Use real user instruction diversity** from WildChat or ShareGPT:
   - Sample 100+ user instructions that reference the same retrieved document
   - Measure hit rate, TTFT, quality

3. **Replace Table 3** with a version showing realistic hit rates (likely 50-80%, not 100%)

4. **Remove all "100% hit rate" claims** from the paper unless they come from exact-match scenarios where 100% is mathematically expected

**Validation**:
- No table shows 100% hit rate on any semantic (non-exact) benchmark
- Cross-instruction benchmark uses instructions with ≥100 token length difference
- **Success criterion**: Cross-instruction hit rate is 50-85% with realistic instruction diversity

**Create benchmark**: `benchmarks/e2e/realistic_cross_instruction_bench.py`

---

## Completed Work (Do Not Repeat)

The following are done with results in `results.tsv` and paper updated:

- Multi-model: Qwen2.5-7B-AWQ + LLaMA-3.1-8B-AWQ-INT4 ✓
- Real datasets: XSum, CNN/DM, MultiNews, WikiHow, SAMSum ✓
- TTFT scaling 2K→32K (both models) ✓
- Quality (PPL ratio) across 5 datasets × 2 models ✓
- Donor store scaling up to 32 donors ✓
- Throughput (QPS) at c=1/4/8/32 ✓
- WildChat-1M similarity analysis ✓ (but E2E benchmark needs larger n)
- Long-output quality (max_tokens up to 2048) ✓
- Threshold sweep (τ=0.30-0.80) ✓
- Jaccard vs MiniLM comparison ✓
- Bathtub fraction ablation ✓
- Variation sensitivity ✓
- NoPE vs delta validation ✓
- Cross-instruction RAG ✓
- CAGRA/cuVS latency scaling ✓

---

## Execution Order (MANDATORY)

**Phase 1 — ALL CODE CHANGES FIRST (no benchmarks yet)**:
1. Implement Gap 1 fix (per-chunk semantic position gate) in alignment.py
2. Enable PartialAttention path for E2E use (Gap 2) — wire up Triton scatter + RoPE correction in the connector so it can be activated via env var
3. Create all new benchmark scripts (partial_attention_bench, embedder_ablation_bench, chunk_size_ablation_bench, realistic_cross_instruction_bench, code_gen_bench, multiturn_dialogue_bench, wildchat_large_bench, breakeven_analysis_bench)
4. Add bootstrap CI computation and n≥32 default to all benchmark scripts
5. Run `pytest synapse_kv_connector/tests/ -v` — everything must pass
6. Copy all changes from semblend-demo to working copy

**Phase 2 — DEPLOY AND BENCHMARK (only after Phase 1 is complete)**:
7. Build Docker image with all code changes
8. Deploy to autoresearch namespace
9. Run benchmarks in gap priority order (1 through 11)
10. Update paper after each gap is resolved

## The Experiment Loop

LOOP FOREVER:

1. **Think** — Review `results.tsv` and the gaps above. Choose the next gap to work on. Work them in priority order unless blocked.

2. **Implement** — Modify SemBlend core code in `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/` FIRST. Then copy changes to `synapse_kv_connector/` working copy. This ensures production code stays in sync.

3. **Test locally** — Run `pytest synapse_kv_connector/tests/ -v` to verify no breakage.

4. **Deploy & benchmark** — Build Docker image, deploy to `autoresearch` namespace, run the benchmark. Use n≥32 samples per condition. Use n≥1000 for dataset-level experiments.

5. **Evaluate** — Check all metrics: TTFT, PPL, hit rate, statistical significance.

6. **Log** — Append to `results.tsv`:
   ```
   commit	tier	benchmark	primary_metric	primary_value	secondary_metrics	status	description
   ```

7. **Update paper** — After each successful experiment, update `paper/semblend.tex` with new data. Remove or qualify any claims that are no longer supported.

8. **Repeat** — Go to step 1. Never stop.

---

## Infrastructure

Before every Tier 3 run, ensure GPU is available:

```bash
bash infra/ensure_gpu.sh
```

**Port-forward**:
```bash
POD=$(kubectl get pods -n autoresearch -l app=vllm --no-headers | awk '{print $1}' | head -1)
kubectl port-forward -n autoresearch $POD 8100:8000 &
```

**Helm deploy**:
```bash
helm upgrade --install autoresearch \
  /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse \
  -n autoresearch \
  -f /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse/values/staging.yaml \
  -f infra/values-autoresearch.yaml
```

**Always restart vLLM between benchmark runs** for clean state.

**vLLM pod label**: `app=vllm`
**Port**: 8000 (internal), port-forward to 8100

---

## Key Codebase Paths

| Component | Production Path | Working Copy |
|-----------|----------------|--------------|
| Alignment | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/alignment.py` | `synapse_kv_connector/alignment.py` |
| Pipeline | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/pipeline.py` | `synapse_kv_connector/pipeline.py` |
| Connector | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/semblend_connector.py` | `synapse_kv_connector/semblend_connector.py` |
| RoPE | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/rope_correction.py` | `synapse_kv_connector/rope_correction.py` |
| Bathtub | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/bathtub.py` | `synapse_kv_connector/bathtub.py` |
| Embedder | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/embedder.py` | `synapse_kv_connector/embedder.py` |
| Donor Store | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/donor_store.py` | `synapse_kv_connector/donor_store.py` |
| Triton Kernels | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/triton_kernels.py` | `synapse_kv_connector/triton_kernels.py` |
| Model Hook | `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/model_runner_hook.py` | `synapse_kv_connector/model_runner_hook.py` |
| Chunk Size Constant | `alignment.py:23` → `LMCACHE_CHUNK_SIZE = 256` | Same |

---

## Success Criteria (All Must Be Met)

Before this research cycle is complete, ALL of the following must hold:

1. **No quality outlier exceeds PPL 1.10** across any dataset, any sample, any model
2. **RoPE delta correction produces measurable quality improvement** on at least one Δ≠0 scenario (corrected PPL < 1.05 vs uncorrected PPL > 1.10)
3. **Bathtub recomputation either demonstrated necessary OR honestly reframed** in the paper
4. **At least 3 embedding models benchmarked** with hit rate, speedup, and quality comparisons
5. **At least 2 non-summarization workloads evaluated** with n≥1000 samples each
6. **WildChat E2E validation uses n≥1000** long-context pairs (sample from the full 1M dataset)
7. **Break-even hit rate computed** analytically and validated empirically
8. **All key results have 95% bootstrap CIs** with n≥32 per condition
9. **LMCache-only baseline run** on all datasets, proving SemBlend's additive value
10. **PPL methodology consistent** across all experiments (same max_tokens, same reporting)
11. **No table shows 100% hit rate** on semantic (non-exact) benchmarks
12. **At least one chunk size ablation** (128 vs 256 tokens) with hit rate comparison
13. **Paper updated** to reflect all new findings — no unsupported claims remain

---

## NEVER STOP

The loop runs until interrupted. If infrastructure is down, provision it. If a deploy fails, diagnose and fix. If a benchmark crashes, debug and re-run. You are fully autonomous.
