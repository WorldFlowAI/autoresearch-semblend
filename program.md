# autoresearch-semblend

Autonomous research loop for SemBlend — a semantic KV-cache reuse system for LLM inference. You are a fully autonomous research agent. Your job is to prove (or disprove) that SemBlend delivers real inference speedups on enterprise workloads, run large-scale experiments, update the paper with honest results, and iterate until every claim is empirically validated.

**Goal**: Validate SemBlend's enterprise viability through real-world workload benchmarks. Every claim must have end-to-end empirical validation on datasets that represent real production traffic, not just academic summarization benchmarks.

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

## Enterprise Viability Review (Critical Assessment)

The paper has strong controlled-benchmark results (2.2×–12× TTFT speedup on synthetic variations). However, these do NOT demonstrate enterprise viability. An honest reviewer would flag:

1. **All 5 "real" datasets are NLP summarization benchmarks** (CNN/DM, XSum, WikiHow, SAMSum, MultiNews) — none represent enterprise workloads
2. **All semantic variations are researcher-created** (REORDER, PARAPHRASE, PARTIAL) — real traffic variation patterns are different
3. **WildChat validation uses n=30** — statistically meaningless (10 hits)
4. **Multi-turn 5.1× likely duplicates LMCache** — extending same prefix is what LMCache already does; no comparison showing SemBlend adds value
5. **Cross-instruction benchmark uses 9-25 char instruction differences** on 32K prompts — trivially similar; not representative of real diverse queries
6. **Chunk-swap brittleness** — any single-token change within a 256-token chunk invalidates it; real enterprise prompts have timestamps, user IDs, entity names that shift boundaries everywhere

### Bottom line:
SemBlend's semantic routing is genuine value-add, but the evidence that it helps on REAL enterprise workloads is thin. We need to either prove it works on realistic traffic or honestly characterize the operating envelope.

---

## Critical Enterprise Gaps (In Priority Order)

These gaps are ordered by impact on the paper's credibility and enterprise viability argument.

---

### E-Gap 1: WildChat at Enterprise Scale (n≥500)

**Problem**: The only real-world validation uses n=30 pairs (10 hits). This is the paper's weakest point. A reviewer will dismiss the entire enterprise argument based on this.

**Existing benchmark**: `benchmarks/e2e/wildchat_large_bench.py` — already built for this purpose but never run at scale.

**Required experiment**:
1. Run `wildchat_large_bench.py` with `--n-per-bucket 100` across 5 similarity buckets
2. Use `--min-chars 6000` for long-context pairs (the only ones where SemBlend helps)
3. Report per-bucket: hit rate, TTFT speedup (P50), sample size
4. Total n≥500 real user conversation pairs

**Success criterion**:
- Clear per-bucket speedup curve (higher similarity → higher speedup)
- Statistically significant result (95% CI excludes 1.0× for at least 2 buckets)
- Honest reporting of which similarity levels are net-positive vs net-negative

**Paper update**: Replace the n=30 WildChat paragraph with n≥500 results and per-bucket table.

---

### E-Gap 2: Enterprise RAG with Diverse Queries

**Problem**: The cross-instruction benchmark (Table 3) uses instructions differing by only 9-25 characters. In real enterprise RAG, users ask genuinely different questions about the same retrieved document — "What are the safety risks?" vs "How much does it cost?" vs "Compare this to competitor X". These differ by hundreds of tokens and test a completely different embedding similarity regime.

**Required experiment**: Create `benchmarks/e2e/enterprise_rag_bench.py`:
1. Use 50 long documents (8K tokens each) from diverse sources (news, technical docs, legal, medical)
2. For each document, create 5 genuinely different queries:
   - Summarization: "Summarize the key points of this document"
   - Extraction: "Extract all dates, names, and monetary amounts"
   - Analysis: "What are the main risks and limitations discussed?"
   - Comparison: "How does this compare to standard industry practice?"
   - Q&A: "Based on this document, answer: [specific factual question]"
3. Queries should differ by 100-500 tokens (realistic enterprise diversity)
4. Measure: hit rate, TTFT speedup, PPL ratio for each query type vs each donor type
5. Compare SemBlend vs LMCache-only (LMCache should get 0% on all of these)

**Success criterion**:
- Hit rate ≥30% across query types (SemBlend finds the same document)
- TTFT speedup ≥1.5× hit-only at 8K tokens
- LMCache-only gets 0% (proving SemBlend's additive value)

---

### E-Gap 3: Multi-Tenant API Pattern

**Problem**: Enterprise LLM APIs serve multiple tenants with similar but not identical system prompts, retrieved documents, and user queries. Template-heavy prompts have variable-length fields (timestamps, user IDs, entity names) that shift chunk boundaries. No benchmark tests this pattern.

**Required experiment**: Create `benchmarks/e2e/multitenant_api_bench.py`:
1. Simulate 10 tenants, each with a unique system prompt (50-200 tokens, varying length)
2. Each tenant sends 20 queries about the same 5 shared knowledge base documents
3. Documents are 4K-8K tokens, identical across tenants
4. Variable-length tenant-specific fields: `"Request from {tenant_name} at {timestamp} for user {user_id}..."`
5. Measure: cross-tenant hit rate (can tenant B reuse tenant A's KV for the same document?)
6. Compare SemBlend vs LMCache (LMCache should get 0% cross-tenant due to different system prompts)

**Success criterion**:
- Cross-tenant hit rate ≥20% for same-document queries
- TTFT speedup ≥1.5× hit-only
- Paper-worthy evidence that SemBlend enables cross-tenant KV sharing

---

### E-Gap 4: LMCache-Only Baseline on All Workloads

**Problem**: The paper never runs LMCache-only (SemBlend disabled) on the same benchmarks. Multi-turn dialogue (5.1×) may be entirely LMCache. Without this baseline, the paper cannot claim SemBlend adds value — it might be measuring LMCache's contribution.

**Required experiment**:
1. Deploy with SemBlend disabled (`SEMBLEND_ENABLED=0` or equivalent)
2. Run the SAME benchmarks: multi-turn dialogue, cross-instruction, WildChat
3. Report LMCache-only hit rate and speedup alongside SemBlend
4. The gap (SemBlend − LMCache) = SemBlend's actual contribution

**Success criterion**:
- Multi-turn: LMCache-only should get similar speedup (proving it's prefix caching, not SemBlend)
- Cross-instruction: LMCache-only should get 0% (proving SemBlend fills the gap)
- WildChat: LMCache-only should get lower hit rate than SemBlend (proving additive value)

---

### E-Gap 5: Customer Support / Ticket Pattern

**Problem**: Customer support is a high-value enterprise use case where users ask similar questions about the same products. No benchmark tests this pattern.

**Required experiment**: Create `benchmarks/e2e/customer_support_bench.py`:
1. Generate 20 product descriptions (4K-8K tokens each)
2. For each product, create 10 customer questions in different styles:
   - "How do I return this product?"
   - "Is this compatible with [other product]?"
   - "What's the warranty on this?"
   - "Can you explain feature X in more detail?"
   - etc.
3. Sequential processing (simulating ticket queue): register first question as donor, measure subsequent questions
4. Measure: cross-question hit rate, TTFT speedup, quality

**Success criterion**:
- Hit rate ≥40% across question types (same product = high semantic overlap)
- TTFT speedup ≥2× hit-only at 8K tokens

---

### E-Gap 6: Template-Heavy Workload Robustness

**Problem**: Enterprise prompts are assembled from templates with variable-length fields. A user ID like "user_12345" vs "user_987654321" shifts chunk boundaries by 5 tokens, invalidating ALL LMCache chunks. SemBlend should handle this via semantic routing to the donor's exact chunks, but this has never been tested explicitly.

**Required experiment**: Create `benchmarks/e2e/template_robustness_bench.py`:
1. Use a fixed document (8K tokens)
2. Wrap in templates with variable-length fields:
   - `"[Request ID: {4-8 digit number}] [User: {variable name}] [Timestamp: {ISO datetime}]"`
   - Each field varies by 3-20 characters
3. Total prefix variation: 10-60 characters
4. Measure SemBlend hit rate across different template fill patterns
5. This is the same mechanism as cross-instruction but with realistic enterprise variation

**Success criterion**:
- SemBlend routes to the correct donor despite template variation
- Hit rate ≥80% (semantic similarity should be very high for same document)
- TTFT speedup ≥2× at 8K tokens

---

### E-Gap 7: Increase Statistical Rigor on Key Results

**Problem**: Main tables use n=5-8. Bootstrap CIs exist for one table only.

**Required**: For any NEW benchmark results added to the paper:
- Use n≥20 per condition minimum
- Report 95% bootstrap CIs
- Report effect size (Cohen's d) for hit-only speedup vs cold

---

## Completed Work (Do Not Repeat)

The following are done with results in `results.tsv` and paper updated:

- Multi-model: Qwen2.5-7B-AWQ + LLaMA-3.1-8B-AWQ-INT4 ✓
- Real datasets: XSum, CNN/DM, MultiNews, WikiHow, SAMSum ✓
- TTFT scaling 2K→32K (both models) ✓
- Quality (PPL ratio) across 5 datasets × 2 models ✓
- Donor store scaling up to 32 donors ✓
- Throughput (QPS) at c=1/4/8/32 ✓
- WildChat-1M similarity analysis (33K pairs) ✓
- WildChat E2E n=30 ✓ (needs scale-up → E-Gap 1)
- Long-output quality (max_tokens up to 2048) ✓
- Threshold sweep (τ=0.30-0.80) ✓
- Break-even analysis (4K/8K/16K) ✓
- Cross-instruction RAG n=8 ✓ (needs realism → E-Gap 2)
- Multi-turn dialogue n=200 ✓ (needs LMCache baseline → E-Gap 4)
- Code gen negative control n=1000 ✓
- Chunk size ablation (64/128/256/512) ✓
- RoPE FORCE_DELTA E2E validation ✓
- NoPE vs delta validation ✓
- CAGRA/cuVS latency scaling ✓
- Bootstrap CIs for XSum 8K ✓

---

## Execution Order (MANDATORY)

**Phase 1 — Infrastructure & Benchmarks (create scripts)**:
1. Create `enterprise_rag_bench.py` — diverse queries on same documents
2. Create `multitenant_api_bench.py` — cross-tenant KV sharing
3. Create `customer_support_bench.py` — sequential ticket pattern
4. Create `template_robustness_bench.py` — variable-length template fields
5. Verify `wildchat_large_bench.py` works with existing data
6. Run `pytest synapse_kv_connector/tests/ -v` — everything must pass

**Phase 2 — Deploy & Run (in priority order)**:
7. Ensure GPU: `bash infra/ensure_gpu.sh`
8. Deploy latest image to autoresearch namespace
9. **E-Gap 1**: Run WildChat large-scale (n≥500)
10. **E-Gap 2**: Run enterprise RAG benchmark
11. **E-Gap 3**: Run multi-tenant API benchmark
12. **E-Gap 4**: Run LMCache-only baseline on multi-turn + cross-instruction + WildChat
13. **E-Gap 5**: Run customer support benchmark
14. **E-Gap 6**: Run template robustness benchmark

**Phase 3 — Paper Update**:
15. Add enterprise workload results section to paper
16. Update WildChat section with n≥500 results
17. Add LMCache-only comparison column to relevant tables
18. Update abstract/conclusion to reflect enterprise findings
19. Recompile PDF

## The Experiment Loop

LOOP FOREVER:

1. **Think** — Review `results.tsv` and the gaps above. Choose the next gap to work on. Work them in priority order unless blocked.

2. **Implement** — Create or modify benchmark scripts. If modifying SemBlend core code, apply changes in `~/dev/worldflowai/semblend-demo/src/connector/synapse_kv_connector/` FIRST, then copy to working copy.

3. **Test locally** — Run `pytest synapse_kv_connector/tests/ -v` to verify no breakage.

4. **Deploy & benchmark** — Build Docker image, deploy to `autoresearch` namespace, run the benchmark. Use n≥20 samples per condition. Use n≥100 for dataset-level experiments.

5. **Evaluate** — Check all metrics: TTFT, PPL, hit rate, statistical significance.

6. **Log** — Append to `results.tsv`:
   ```
   commit	tier	benchmark	primary_metric	primary_value	secondary_metrics	status	description
   ```

7. **Update paper** — After each successful experiment, update `paper/semblend.tex` with new data. Remove or qualify any claims that are no longer supported. Be honest about what works and what doesn't.

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

**LMCache-only mode**: To disable SemBlend and test LMCache baseline, write config:
```bash
kubectl exec -n autoresearch $POD -- bash -c 'echo "{\"semblend_enabled\": false}" > /tmp/semblend_config.json'
```

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

---

## Success Criteria (All Must Be Met)

Before this research cycle is complete, ALL of the following must hold:

1. **WildChat E2E uses n≥500** long-context pairs with per-bucket reporting
2. **Enterprise RAG benchmark** with genuinely diverse queries (100+ token length differences) shows hit rate ≥30%
3. **Multi-tenant API benchmark** demonstrates cross-tenant KV sharing
4. **LMCache-only baseline** run on multi-turn, cross-instruction, and WildChat — proving SemBlend's additive value
5. **Customer support pattern** benchmark with sequential tickets shows ≥40% hit rate
6. **Template robustness** benchmark proves SemBlend handles variable-length prefix fields
7. **All new results have 95% bootstrap CIs** with n≥20 per condition
8. **Paper updated** with enterprise workload section — honest about what works and what doesn't
9. **PDF recompiled** with all updates

---

---

## Paper Improvement Plan (From Full Analysis — March 2026)

The paper is 29 pages, two-column, with 18 tables, 1 figure, 2 algorithms, 31 references. Below are the issues identified and the execution plan.

### Structural Fixes (P0)

1. **Add speedup-vs-length figure**: The paper's central finding (sub-linear TTFT growth → increasing speedup) is described textually 4+ times but never plotted. Create a pgfplots figure showing speedup vs prompt length for both Qwen and LLaMA (data from tab:results-kv).

2. **Add PPL-vs-hit-rate scatter**: Show the relationship between hit rate and mean PPL ratio across all datasets/models. This visualizes the quality floor guarantee.

3. **Add WildChat similarity histogram**: Show the distribution of consecutive-pair cosine similarities (data from wildchat_similarity.json). Annotate τ=0.60 threshold.

4. **Deduplicate repeated text (~2 pages)**:
   - "Sub-linear TTFT growth" stated in tab:results-kv caption, body text after table, §4.1.1, §4.3, conclusion — consolidate to ONE explanation + forward references
   - "15.4s cold → 1.3s SemBlend at 32K" appears 3+ times — state once prominently
   - Miss-run outlier explanation for elevated PPL — state once in quality section, reference elsewhere
   - "Cascading divergence" for ROUGE-L — state once

5. **Merge duplicate Qwen data**: tab:cross-dataset and tab:multi-model-cross share all 12 Qwen rows identically. Remove Qwen from tab:multi-model-cross, reference tab:cross-dataset.

6. **Promote Analysis to §5**: Move threshold sweep, chunk ablation, RoPE validation, shifted-prefix analysis, and variation sensitivity out of §4 into a top-level "Analysis" section.

7. **Add Future Work subsection** to Discussion: PartialAttention integration, sub-chunk alignment, multi-GPU, production deployment.

### Content Fixes (P1)

8. **Fix speedup range "2.1×–7.5×" (line 1152)**: This mixes synthetic single-donor controlled experiments with cross-dataset results. Clarify the context or split the range.

9. **Resolve chunk size ablation narrative**: §4.4.2 opens by motivating smaller chunks but then shows identical hit rates. Restructure: lead with the finding (chunk size is irrelevant due to binary alignment), then explain why.

10. **Contextualize WildChat 30% hit rate**: The blended 1.17× is modest. Be upfront: "In practice, 70% of requests see ~8ms overhead with no speedup; the remaining 30% see 1.6–1.9× speedup, for a blended 1.17× improvement."

11. **Clarify enterprise workloads use synthetic documents**: Rename section or add prominent note that documents are generated, not from real enterprise systems.

12. **Justify sliding window 40/30/30 split**: Add brief rationale or sensitivity note.

13. **Explore PPL < 1.0 values**: Multiple measurements show SemBlend is *better* than cold (0.944, 0.960, 0.984, 0.992). Add a sentence noting this potential implicit regularization effect.

14. **Break-even 2K "0ms" overhead**: Clarify this is *wall-clock overlap* (pipeline runs within prefill compute window), not zero computation.

### Formatting Fixes (P2) — Conference-Ready

15. **Adopt proper CS paper formatting**:
    - Switch to a standard template (ACM `acmart` or NeurIPS/ICML style)
    - Single-column for readability if targeting arXiv, or proper two-column with correct margins
    - Proper author block with affiliations and emails
    - Running headers
    - Page numbers

16. **Fix notation conflict**: Eq. 6 uses P_partial/P_full (layer counts) vs P_h (probability). Rename to N_partial/N_full.

17. **Standardize bibliography**: Consistent arXiv ID format, consistent venue+year formatting.

18. **Clean up font/spacing**: Ensure consistent use of \textbf, \emph, table formatting. Remove any overfull hboxes.

### Execution Order

Execute in this order:
1. Structural fixes 4-5 (dedup + merge tables) — reduces page count
2. Structural fix 6 (promote Analysis)
3. Content fixes 8-14
4. Structural fixes 1-3 (add figures) — requires pgfplots data
5. Structural fix 7 (Future Work)
6. Formatting fixes 15-18 (final pass)

---

## NEVER STOP

The loop runs until interrupted. If infrastructure is down, provision it. If a deploy fails, diagnose and fix. If a benchmark crashes, debug and re-run. You are fully autonomous. Report results honestly — if SemBlend doesn't help on a workload, say so in the paper. Negative results are valuable.
