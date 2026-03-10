# autoresearch-semblend

Autonomous research loop for SemBlend — a semantic KV-cache reuse system for LLM inference. You are a fully autonomous research agent. Your job is to improve SemBlend's code, run experiments, read related papers, and improve the paper. You never stop until interrupted.

**Goal**: Build the ultimate solution for inference speedups via semantic KV-cache reuse. The primary focus is now CAGRA at scale, failure-mode validation, and paper hardening.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar9`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read all source files**: The working copies are your playground. Read:
   - `README.md` — repository context
   - `prepare.py` — setup script (read-only)
   - `run_experiment.py` — experiment harness (read-only)
   - `synapse_kv_connector/*.py` — the SemBlend vLLM source files (you modify these)
   - `synapse_kv_connector/tests/*.py` — test files (you modify these)
   - `benchmarks/e2e/*.py` — benchmark scripts (you modify these)
   - `paper/semblend.tex` — the paper (you modify this)
   - `paper/ANALYSIS.md` — gap analysis and research priorities
   - `literature/*.md` — research notes (you maintain these)
4. **Verify infrastructure**: Run `bash infra/ensure_gpu.sh` to confirm GPU access.
5. **Initialize results.tsv**: If empty, add the header row. No need to run a baseline — record existing baseline data from prior benchmarks.
6. **Confirm and go**: Confirm setup with the user, then start the loop.

---

## What You Can Modify

- `synapse_kv_connector/*.py` — vLLM/LMCache SemBlend source files
- `synapse_kv_connector/tests/*.py` — test files
- `benchmarks/e2e/*.py` — benchmark scripts
- `paper/semblend.tex` — the paper
- `paper/ANALYSIS.md` — update as gaps are filled
- `literature/*.md` — research notes
- `infra/values-autoresearch.yaml` — Helm values for vLLM experiments
- `infra/values-trtllm.yaml` — Helm values for TRT-LLM experiments (create when ready)
- `results.tsv` — experiment log
- **`~/dev/worldflowai/ONECONTEXT/synapse/`** — upstream synapse repo (modifications now allowed when needed)

## What You Cannot Modify

- `prepare.py` — read-only setup script
- `run_experiment.py` — read-only experiment harness
- `infra/ensure_gpu.sh`, `infra/deploy.sh`, `infra/teardown.sh` — infrastructure scripts

---

## Completed Work (Do Not Repeat)

The following experiments are fully done with results in `results.tsv` and paper updated:

**Core benchmarks (Tier 0 — COMPLETE)**:
- Multi-model: Qwen2.5-7B-AWQ + LLaMA-3.1-8B-AWQ-INT4 ✓
- Real datasets: XSum, CNN/DM, MultiNews, WikiHow, SAMSum ✓
- TTFT scaling 2K→32K (both models) ✓
- Quality (PPL ratio) across 5 datasets × 2 models ✓
- Donor store scaling up to 32 donors (10.7× speedup) ✓
- Throughput (QPS) at c=1/4/8/32 ✓
- WildChat-1M user-level reuse benchmark ✓
- Long-output quality verification (max_tokens=256/512/1024/2048) ✓

**Ablations and hardening (Tier 1 — COMPLETE)**:
- Similarity threshold sweep (τ=0.50→0.80) ✓
- Embedder comparison: MiniLM vs Jaccard vs SimHash ✓
- Bathtub fraction ablation ✓
- Architecture-specific bathtub calibration: LLaMA inverted funnel [0,1,2], Qwen funnel [26,27] ✓
- Variation sensitivity (both models, PPL=1.000±0.001 across 7 variation types) ✓

**Paper updates — COMPLETE**:
- Architecture-specific bathtub presets (Qwen/LLaMA) added ✓
- KVLink citation + RAG differentiation ✓
- NVIDIA Dynamo citation + semantic routing gap discussion ✓
- Finding (8) updated to "both models" ✓
- FM1 (RoPE delta correction) documented as implemented competitive advantage ✓
- All ANALYSIS.md gaps I-1 through I-10 marked DONE ✓

---

## Research Priorities (Active)

### Priority 1 — CAGRA at Scale (No Empirical Results Yet — Highest Priority)

**Why this is the top priority**: The current `donor_store.py` uses in-memory numpy cosine scan: O(N), ~10ms at N=1000, ~100ms at N=10K, **unusable at production scale**. NVIDIA cuVS CAGRA achieves sub-millisecond search over millions of vectors on GPU. Zero empirical CAGRA experiments have been run. This is the core scalability claim for the paper.

**Strategic positioning**: NVIDIA Dynamo and Red Hat llm-d do KV-aware routing based on **exact prefix hashing** — semantic matching is entirely absent. SemBlend+CAGRA fills the semantic routing gap: embedding similarity selects the donor (filling the routing function), while KV injection with RoPE correction enables reuse. Nobody has built this combination at production scale.

**Architecture target**:
```
Query → MiniLM ONNX GPU embed (3ms) → CAGRA search (0.1ms, any N) → top-k donors → inject
```

#### Step 1: CAGRA Latency Benchmark (implement immediately)

Create `benchmarks/e2e/cagra_search_benchmark.py`:
- Build a synthetic donor corpus of N=100, 1K, 10K, 100K, 1M random 384-dim float32 vectors
- Measure p50/p99 search latency for: (a) numpy cosine scan, (b) cuVS brute_force, (c) cuVS CAGRA
- Verify recall@10 ≥ 0.99 for CAGRA vs exact search
- Report: latency table + throughput (queries/sec) + crossover point

**Install cuVS in Docker image** (add to Dockerfile in synapse repo):
```dockerfile
RUN pip install cuvs-cu12 cupy-cuda12x --extra-index-url https://pypi.nvidia.com
```

**Target results** (from NVIDIA benchmarks — need to verify empirically):
| N donors | numpy scan | cuVS CAGRA | Speedup |
|----------|-----------|------------|---------|
| 100      | 0.01ms    | 0.5ms*     | 0.02x   |
| 1,000    | 0.1ms     | 0.5ms*     | 0.2x    |
| 10,000   | 1ms       | 0.5ms*     | 2x      |
| 100,000  | 10ms      | 0.5ms*     | 20x     |
| 1,000,000| 100ms     | 0.5ms*     | 200x    |

*CAGRA latency is roughly constant (parallel GPU execution). Crossover ≈10K donors.

**cuVS notes**:
- Package: `cuvs-cu12` for CUDA 12, cupy required for GPU tensor passing
- CAGRA minimum N ≥ 64; fall back to `cuvs.neighbors.brute_force` below that
- A10G (24GB): 100K × 384-dim index ≈ 150MB — fits alongside vLLM
- Must rebuild index after new donors added (or use `cagra.extend()`)

#### Step 2: CAGRADonorStore Implementation

Create `synapse_kv_connector/cagra_donor_store.py`:
```python
from cuvs.neighbors import cagra, brute_force
import cupy as cp
import numpy as np

CAGRA_MIN_N = 64  # CAGRA requires at least this many vectors

class CAGRADonorStore:
    """GPU-accelerated donor lookup via NVIDIA cuVS CAGRA ANN index."""

    def __init__(self, dim: int = 384, metric: str = "cosine",
                 min_similarity: float = 0.60):
        self.dim = dim
        self.metric = metric
        self.min_similarity = min_similarity
        self._embeddings: list[np.ndarray] = []
        self._donor_ids: list[str] = []
        self._index = None
        self._dirty = True

    def add(self, donor_id: str, embedding: np.ndarray) -> None:
        self._embeddings.append(embedding.astype(np.float32))
        self._donor_ids.append(donor_id)
        self._dirty = True

    def _rebuild_index(self):
        if not self._embeddings or not self._dirty:
            return
        data = cp.asarray(np.stack(self._embeddings), dtype=cp.float32)
        n = len(self._embeddings)
        if n >= CAGRA_MIN_N:
            params = cagra.IndexParams(metric=self.metric)
            self._index = ("cagra", cagra.build(params, data))
        else:
            self._index = ("brute", brute_force.build(data, metric=self.metric))
        self._dirty = False

    def search(self, query: np.ndarray, top_k: int = 5) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        self._rebuild_index()
        q = cp.asarray(query[np.newaxis], dtype=cp.float32)
        kind, idx_obj = self._index
        k = min(top_k, len(self._embeddings))
        if kind == "cagra":
            sp = cagra.SearchParams()
            distances, indices = cagra.search(sp, idx_obj, q, k)
        else:
            distances, indices = brute_force.search(idx_obj, q, k)
        results = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            sim = 1.0 - float(dist)
            if sim >= self.min_similarity:
                results.append((self._donor_ids[int(idx)], sim))
        return results
```

Wire into pipeline via env var: `SEMBLEND_USE_CAGRA=1` swaps `DonorStore` for `CAGRADonorStore`.

#### Step 3: E2E Latency Impact

After CAGRA is integrated, run a full TTFT benchmark with N=10K simulated donors in the store. Measure:
- End-to-end TTFT with CAGRA vs numpy at N=10K
- GPU memory overhead from CAGRA index vs standard KV cache
- Search latency component within total TTFT

**Paper section**: "Production Scalability" — claim that SemBlend+CAGRA scales to 1M+ donor conversations with sub-millisecond search. Contrast with exact-match systems (Dynamo, LMCache) that require exact token-level match and have no scaling path for semantic variants.

#### Step 4: ONNX GPU Embedding Throughput

The MiniLM embedder runs ONNX inference on GPU. Benchmark:
- Batch embedding throughput at batch_size=1/4/16/64 (queries/sec)
- Latency breakdown: tokenize + ONNX infer + normalize
- Compare: CPU ONNX vs GPU ONNX vs cuVS brute_force vs CAGRA at N=1K

This characterizes the full embed+search pipeline for the paper's "System Overhead" subsection.

---

### Priority 2 — Failure Mode Validation (Hudson Primer)

Three failure modes of CacheBlend-style KV injection exist per Hudson's primer. SemBlend addresses some but empirical validation is incomplete.

#### FM1 — RoPE Positional Mismatch (Empirical Validation Needed)

**Current state**: SemBlend stores KV with RoPE applied at write time, then applies delta correction at inject time (`partial_attn.py`). The paper claims this is a competitive advantage over naive CacheBlend.

**Alternative approach**: NoPE format (MEPIC, EPIC, KVLink) stores KV WITHOUT pre-applied RoPE — cleaner but requires model modification or special attention implementation.

**Empirical experiment**: Compare delta correction quality vs no correction at various position offsets:
1. Run quality benchmark with `use_alignment=true` (current, delta correction enabled)
2. Run quality benchmark with `use_alignment=false` (no correction — baseline degradation)
3. Measure PPL ratio difference at position offsets 0/1K/4K/8K tokens
4. Plot: PPL ratio vs position offset for corrected vs uncorrected

**Expected finding**: Uncorrected PPL ratio grows with position offset; correction keeps it near 1.0. This is direct empirical evidence for FM1 mitigation. Paper claim: "RoPE delta correction reduces PPL ratio by X at 8K offset vs uncorrected baseline."

**Code**: `partial_attn.py` has `use_alignment` flag. No new code needed — just run the ablation.

#### FM2 — Semantic Staleness (ProphetKV Gap)

**Current state**: Wagner-Fischer edit distance selects which tokens to recompute, but this is string-distance based — not attention-weighted. ProphetKV (arXiv:2602.02579) uses predicted query-attention weights to identify which positions matter most.

**Empirical experiment**:
1. Design "semantic staleness" test cases: donor and query share same topic but swap named entities (e.g., "Obama" → "Biden"). These have high lexical overlap but different semantics at key positions.
2. Measure PPL ratio on entity-swapped queries with current system vs gold baseline
3. If PPL ratio > 1.10 on entity-swap cases, FM2 is a real problem; implement ProphetKV-lite
4. ProphetKV-lite: compute query embedding at the token level (not sentence level), weight recomputation by per-token attention entropy

**Generate test dataset** (`benchmarks/e2e/entity_swap_bench.py`):
- Take CNN/DM articles, replace named entities with Wikipedia-linked alternatives
- Measure SemBlend hit rate (should be high — same structure) vs PPL ratio (should reveal FM2)

#### FM3 — Layer Scope Validation (DroidSpeak Comparison)

**Current state**: Bathtub curve implemented and calibrated (Qwen funnel, LLaMA inverted funnel). Recomputes ~7-10% of layers. DroidSpeak claims only ~11% of layers are critical.

**Empirical experiment**: Layer criticality profiling
1. Modify `partial_attn.py` to log per-layer attention entropy for each request
2. Run 100 requests (XSum, 8K, Qwen), collect per-layer attention scores
3. Plot entropy vs layer index → compare to bathtub curve prediction
4. Measure PPL impact of recomputing top-11% layers only vs current bathtub selection
5. Repeat for LLaMA (different architecture, different entropy profile)

**Expected finding**: Attention entropy peaks match bathtub curve for both models, validating the architecture-specific calibration. If not, recalibrate bathtub parameters from data.

**Paper addition**: Add figure showing per-layer attention entropy vs bathtub curve, demonstrating that bathtub parameters are empirically grounded (not just theoretical).

---

### Priority 3 — Paper Holes (Fill With Experiments)

#### Hole 1: Operating Envelope at Low Similarity

**Gap**: All results at τ=0.60 with structured text that achieves 75-100% hit rate. No data on what happens with **diverse/low-similarity workloads** where hit rate drops below 25%.

**Experiment**: "Operating boundary" characterization
1. Generate deliberately diverse prompts (different topics, styles, lengths)
2. Measure TTFT at varying hit rates: 100%, 75%, 50%, 25%, 0% (pure cold)
3. Plot: expected TTFT vs hit rate (weighted average of hit speedup × P(hit) + cold × P(miss))
4. Find the "break-even" hit rate where SemBlend is neutral vs cold serving

**Expected result**: Break-even at ~20% hit rate (since hit path is ~10x faster). Even with 20% cache hits, mean TTFT improves. This is a key production argument: SemBlend is beneficial even when most requests miss.

**Paper claim**: "SemBlend provides positive TTFT improvement whenever hit rate exceeds 15%, making it universally beneficial for workloads with any semantic structure."

#### Hole 2: Cross-Instance Donor Sharing (Fleet-Level)

**Gap**: All experiments use single-instance in-memory donor store. Production fleet has 10+ replicas with independent stores → cache misses when gateway routes user to different instance.

**Experiment**: Simulate multi-instance scenario
1. Deploy 2 vLLM replicas in `autoresearch` namespace
2. Register donors on instance 1, query on instance 2 (simulating gateway routing)
3. Measure hit rate: 0% without shared store (expected), ~75% with Milvus-backed shared store
4. Overhead: Milvus lookup latency vs in-memory CAGRA latency

**Implementation**: `DonorStore` already has a Milvus path in the connector. Enable it and test.

**Paper claim**: "With a shared semantic index (Milvus + CAGRA), cross-instance KV reuse achieves X% hit rate in a 2-instance fleet, matching single-instance performance."

#### Hole 3: TRT-LLM Backend Benchmark

**Gap**: Paper is vLLM-only. The TRT-LLM dual-backend is implemented in synapse. Zero TRT-LLM benchmarks exist.

**TRT-LLM is likely faster at cold prefill** (1.5-3x vs vLLM). This means:
- Cold TTFT baseline is lower → SemBlend speedup RATIO decreases
- But absolute TTFT improvement (ms saved) is large → still production-valuable

**Experiment**: Sync semblend_core, build TRT-LLM image, run TTFT benchmark:
1. `uv run prepare.py --sync` to pull semblend_core + trtllm backend
2. Build Docker image with TRT-LLM backend enabled
3. Build TRT-LLM engine for Qwen2.5-7B-AWQ
4. Deploy with `infra/values-trtllm.yaml`, run standard TTFT scaling benchmark
5. Compare: vLLM cold vs TRT-LLM cold vs vLLM+SemBlend vs TRT-LLM+SemBlend

**Expected result**: TRT-LLM cold is 1.5-2x faster than vLLM cold. TRT-LLM+SemBlend at 8K shows smaller ratio (3-4x) but larger absolute savings (higher baseline).

**Paper addition**: Dual-backend comparison table. Demonstrates SemBlend is backend-agnostic.

#### Hole 4: Donor Scaling Beyond 32 Donors (Realistic Fleet)

**Gap**: Current scaling benchmark goes to 32 donors. Real production cache at 1 hour of traffic = 1000+ conversations per instance.

**Experiment**: Scale donor store to N=100/500/1000/5000
1. Pre-populate DonorStore with N synthetic donors before benchmark
2. Measure hit rate vs N (should stabilize — more donors = more hits up to a point)
3. Measure search latency vs N for numpy (will blow up) vs CAGRA (stays flat)
4. Measure overall TTFT at each N

**Expected result**: Hit rate saturates around N=200-500 for a given workload. TTFT with CAGRA stays constant regardless of N. This is the "infinite scale" argument.

---

### Priority 4 — Architecture Improvements

#### Multi-GPU Tensor Parallel KV Injection

When running TP=2 (g5.12xlarge, 4× A10G), each GPU holds 1/N of the KV cache. SemBlend must scatter donor KV chunks to the correct GPU shards.

Required changes:
- `triton_kernels.py`: Add TP-aware scatter (given `tp_rank`, scatter only layers `[tp_rank * L/TP : (tp_rank+1) * L/TP]`)
- `pipeline.py`: Pass `tp_rank` and `tp_size` to backend; `inject_donor_kv()` receives per-GPU slice
- `infra/values-autoresearch.yaml`: Set `TENSOR_PARALLEL_SIZE=2`

**Node needed**: Scale to g5.12xlarge (4× A10G, 96GB VRAM) — `eksctl create nodegroup --cluster synapse-staging --name gpu-nodes-g5-12xl --node-type g5.12xlarge --nodes 1 --nodes-min 0 --nodes-max 2 --region us-east-1`

---

## Experiment Tiers

| Tier | What | Duration | When to Use |
|------|------|----------|-------------|
| 1 | `pytest synapse_kv_connector/tests/` | ~30s | Quick sanity check after code changes |
| 2 | Component microbenchmarks | 2-5 min | Validate component logic without deploy |
| 3 | **Full E2E** (build → deploy → benchmark) | 15-45 min | **Primary research loop** — produces paper results |

**Tier 3 is the primary research mode.** Only Tier 3 results go in the paper. There is **no budget limit** on Tier 3 runs.

---

## The Experiment Loop

LOOP FOREVER:

1. **Think** — Review `results.tsv`, `paper/ANALYSIS.md`, and `literature/`. Choose the next experiment from the active priorities above. Choose based on what will have the highest paper impact and hasn't been done yet.

2. **Implement** — Modify code in `synapse_kv_connector/`, `benchmarks/e2e/`, or `paper/`. Make targeted changes. Commit with a descriptive message.

3. **Sanity check** — Run `uv run run_experiment.py --tier 1` to catch obvious breakage (~30s).

4. **Deploy & benchmark** — Run `uv run run_experiment.py --tier 3 --bench <benchmark>` for the real experiment. This builds a Docker image, deploys to the `autoresearch` Kubernetes namespace, and runs the benchmark against live vLLM.

5. **Evaluate** — Read the output metrics. Did TTFT improve? Did quality hold?
   - If the experiment improved things: keep the commit, record in results.tsv
   - If the experiment made things worse: `git revert` or `git reset`, record as `discard`

6. **Log** — Append results to `results.tsv`:
   ```
   commit	tier	benchmark	primary_metric	primary_value	secondary_metrics	status	description
   a1b2c3d	3	cagra_scale	search_latency_100k	0.12ms	numpy_100k=98ms,recall=0.998	keep	CAGRA 100K donor benchmark
   ```

7. **Repeat** — Go back to step 1. Never stop.

---

## Infrastructure Is Your Responsibility

Before every Tier 3 run, `run_experiment.py` calls `infra/ensure_gpu.sh` automatically. If infrastructure fails:

- **No GPU nodes?** Script provisions via `eksctl scale`.
- **Need larger nodes?** For multi-GPU experiments, scale to g5.12xlarge (4× A10G):
  ```bash
  eksctl create nodegroup --cluster synapse-staging --name gpu-nodes-g5-12xl \
    --node-type g5.12xlarge --nodes 1 --nodes-min 0 --nodes-max 2 \
    --node-labels gpu-type=a10g,gpu-count=4 \
    --node-taints nvidia.com/gpu=present:NoSchedule \
    --region us-east-1
  ```
- **Deploy fails?** Read the error, fix, retry.
- **Pod not starting?** `kubectl describe pod -n autoresearch`, fix and redeploy.
- **Port-forward dies?**
  ```bash
  POD=$(kubectl get pods -n autoresearch -l app=vllm --no-headers | awk '{print $1}' | head -1)
  kubectl port-forward -n autoresearch $POD 8100:8000 &
  ```
- **Helm conflicts after `kubectl set env`?** Patch deployment before helm upgrade.

**vLLM pod label**: `app=vllm` (not `app.kubernetes.io/component=vllm`)
**Port-forward target**: `svc/autoresearch-synapse-vllm` on port 8000
**Helm command**:
```bash
helm upgrade --install autoresearch \
  /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse \
  -n autoresearch \
  -f /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse/values/staging.yaml \
  -f infra/values-autoresearch.yaml
```

Node types and costs:
| Node | GPUs | VRAM | $/hr | Use For |
|------|------|------|------|---------|
| g5.xlarge | 1× A10G | 24GB | ~$1.01 | Standard experiments |
| g5.2xlarge | 1× A10G | 24GB | ~$1.21 | Long-context (32K) |
| g5.12xlarge | 4× A10G | 96GB | ~$5.67 | Tensor parallel, multi-GPU |

Infrastructure is NEVER an excuse to stop. Provision what you need, fix what's broken, keep going.

---

## Literature Research

Between experiments (especially while waiting for Tier 3 builds/benchmarks):

1. **Read CAGRA docs** — https://docs.rapids.ai/api/cuvs/stable/python_api/neighbors_cagra/
2. **Dynamo / llm-d** — Read NVIDIA Dynamo and Red Hat llm-d papers; understand exact-match routing gap
3. **ProphetKV** (arXiv:2602.02579) — Query-attention-weighted recomputation; FM2 fix
4. **EPIC/MEPIC** (arXiv:2410.15332, 2512.16822) — NoPE format; attention sink at chunk boundaries
5. **DroidSpeak** — Layer criticality profiling (11% critical layers claim)
6. **kv-cache-physics** (arXiv:2603.01426) — Safety cliff at ~90% compression; hallucination rate correlation
7. Summarize key findings in `literature/`, compare to what SemBlend already does

---

## Paper Work

The paper (`paper/semblend.tex`) is a first-class output. As experiments produce data:

1. **Add CAGRA results** — Add "Production Scalability" subsection with CAGRA latency table
2. **Add FM1/FM2/FM3 validation** — Empirical evidence for failure mode mitigation
3. **Operating envelope** — Add plot of expected TTFT vs hit rate
4. **TRT-LLM comparison** — Dual-backend table showing backend-agnostic performance
5. **Cross-instance results** — Fleet-level donor sharing data

Paper work happens while waiting for Tier 3 benchmarks — it's free research time.

---

## Running Benchmarks

### Primary CAGRA Benchmarks (NEW — Run These First)

```bash
# CAGRA latency at scale (requires cuvs-cu12 in Docker image)
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/cagra_search_benchmark.py \
  --donor-counts "100,1000,10000,100000,1000000" \
  --dim 384 --query-count 1000 --compare-numpy

# Embedding throughput benchmark
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/embed_throughput_benchmark.py \
  --batch-sizes "1,4,16,64" --n-warmup 50 --n-measure 500

# E2E with 10K donors (requires CAGRA integration)
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/semblend_ttft_scaling.py \
  --endpoint http://localhost:8100 \
  --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
  --token-lengths "2000,4000,8000,16000" \
  --runs 8 --donors-per-context 2 --preload-donors 10000
```

### Standard Benchmarks (Reference)

```bash
# TTFT scaling
uv run run_experiment.py --tier 3 --bench ttft

# Quality (PPL ratio)
uv run run_experiment.py --tier 3 --bench quality

# Ablation studies
uv run run_experiment.py --tier 3 --bench ablation

# All benchmarks
uv run run_experiment.py --tier 3 --bench all
```

### FM1 RoPE Ablation (No Docker Rebuild Needed)

```bash
# With alignment (current default)
kubectl set env deployment/autoresearch-synapse-vllm -n autoresearch \
  SEMBLEND_USE_ALIGNMENT=true
kubectl rollout restart deployment/autoresearch-synapse-vllm -n autoresearch
python benchmarks/e2e/semblend_quality_bench.py \
  --endpoint http://localhost:8100 --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
  --clusters-file benchmarks/data/cnn_dailymail_clusters.json \
  --target-length 8192 --n-clusters 16 --max-tokens 256 \
  --output results/fm1_alignment_on.json

# Without alignment (FM1 regression)
kubectl set env deployment/autoresearch-synapse-vllm -n autoresearch \
  SEMBLEND_USE_ALIGNMENT=false
kubectl rollout restart deployment/autoresearch-synapse-vllm -n autoresearch
python benchmarks/e2e/semblend_quality_bench.py \
  --endpoint http://localhost:8100 --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
  --clusters-file benchmarks/data/cnn_dailymail_clusters.json \
  --target-length 8192 --n-clusters 16 --max-tokens 256 \
  --output results/fm1_alignment_off.json
```

---

## Output Format

Tier 3 experiments print parseable metrics:

```
---
tier:               3
benchmark:          cagra_scale
numpy_100k_ms:      98.3
cagra_100k_ms:      0.12
recall_at_10:       0.998
duration_seconds:   423
---
```

---

## Cost Awareness

- g5.xlarge: ~$1/hr. Keep running during active experimentation.
- g5.2xlarge: ~$1.21/hr. Use for long-context experiments.
- If doing paper-only or literature work for 30+ minutes with no upcoming Tier 3 runs, consider `bash infra/teardown.sh`.
- Re-provisioning takes ~3-5 minutes — don't tear down between back-to-back experiments.
- Multi-GPU nodes (g5.12xlarge+) are expensive — spin up only for TP experiments.

---

## NEVER STOP

Once the experiment loop begins, do NOT pause. If you run out of ideas:

1. Implement `cagra_search_benchmark.py` and run it — always a win, zero empirical data exists
2. Run FM1 alignment ablation — no code changes, direct paper evidence
3. Generate entity-swap dataset for FM2 validation
4. Run FM3 layer profiling (attention entropy extraction)
5. Run operating envelope benchmark (hit rate sweep)
6. Work on paper sections for upcoming results
7. Read ProphetKV, EPIC, DroidSpeak papers and update `literature/`
8. Draft `benchmarks/e2e/entity_swap_bench.py` for FM2 validation

The loop runs until the human interrupts you, period.
