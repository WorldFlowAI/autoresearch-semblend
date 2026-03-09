# autoresearch-semblend

Autonomous research loop for SemBlend — a semantic KV-cache reuse system for LLM inference. You are a fully autonomous research agent. Your job is to improve SemBlend's code, run experiments, read related papers, and improve the paper. You never stop until interrupted.

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

## Research Priorities

Work on these roughly in order of impact. The first two tiers are already substantially complete; the focus is now on the next-generation architecture.

### Tier 0 — COMPLETE (Foundational Results)
- Multi-model support: Qwen2.5-7B-AWQ + LLaMA-3.1-8B-AWQ-INT4 ✓
- Real-world datasets: XSum, CNN/DM, MultiNews, WikiHow, SAMSum ✓
- Long-context scaling: 2K→32K TTFT (11.97x@32K Qwen, 10.85x@32K LLaMA hit-only) ✓
- Quality metrics: PPL ratio across 5 datasets × 2 models ✓
- Donor store scaling: 10.7x with 32 donors ✓
- Throughput: QPS scaling at c=1/4/8/32 ✓

### Tier 1 — IN PROGRESS (Paper Quality Gaps)
- WildChat-1M benchmark: user-level semantic overlap validation (highest priority)
- **Long-output quality verification** (HIGH PRIORITY): Rerun quality benchmarks at max_tokens=512/1024/2048 to verify PPL ratio stability at realistic generation lengths. Current 256-token cap is not representative of production. See "Long-Output Quality Verification" section under Running Benchmarks.
- Ablation: similarity threshold sweep (0.40–0.80)
- Ablation: embedder comparison (MiniLM vs Jaccard vs SimHash)
- Ablation: bathtub fraction sweep
- PartialAttention + RoPE correction E2E validation

### Tier 2 — NEXT ARCHITECTURE (High Impact)
The TRT-LLM dual-backend is **COMPLETE** in `~/dev/worldflowai/ONECONTEXT/synapse/`. The new structure:
- `services/shared/semblend_core/` — shared backend-agnostic core (pipeline, alignment, donor_store, embedder, bathtub, rope_correction, triton_kernels, simhash)
- `services/trtllm/semblend_kv_cache_manager.py` — TRT-LLM KVCacheManager subclass
- `services/trtllm/trtllm_backend.py` — SemBlendBackend impl for TRT-LLM
- `synapse_kv_connector/*.py` — now thin re-export stubs for backward compat (all 34 existing tests still pass)

Sync via `prepare.py --sync` to pull the new files into this repo. Then build on them:

1. **CAGRA/cuVS GPU-Accelerated Donor Search** — replace numpy cosine scan with GPU ANN index
2. **TRT-LLM Backend Benchmarks** — dual-backend is ready; sync, build image, compare vLLM vs TRT-LLM TTFT
3. **NVIDIA Dynamo Integration** — SemBlend in disaggregated prefill/decode architecture
4. **Multi-GPU SemBlend** — tensor parallel KV injection, cross-replica donor sharing

---

## Priority Dataset: WildChat-1M

**WildChat-1M** (Allen AI, ODC-BY license) is the highest-value dataset for validating SemBlend's production thesis:

- **1M conversations from ~204K unique users**, with `hashed_ip` field enabling per-user grouping
- This is the only large public dataset that lets you measure **intra-user semantic overlap** — i.e., "the same tenant asks similar questions over time"
- Directly models the real deployment scenario SemBlend is built for: KV reuse across a user's session history

**Benchmark design**:
1. Group conversations by `hashed_ip` → get user conversation histories
2. For each user with ≥4 conversations: use conversation N as donor, conversation N+1/N+2 as queries
3. Measure hit rate, TTFT speedup, and quality against cold baseline
4. Report: % of user pairs with similarity ≥τ, speedup on hit pairs, per-user speedup distribution

**HuggingFace**: `allenai/WildChat-1M` (download ~2GB). Priority over additional synthetic benchmarks.

---

## CAGRA/cuVS GPU-Accelerated Donor Search

### Why This Matters
The current `donor_store.py` uses in-memory numpy cosine scan: O(N) per query, ~1-2ms for small N, but **unusable at production scale** (10K+ donors → 50ms+). GPU-accelerated ANN via [NVIDIA cuVS CAGRA](https://github.com/rapidsai/cuvs) achieves sub-millisecond search over millions of vectors.

### Target Architecture
```
Query → MiniLM embed (3ms GPU) → CAGRA search (0.1ms, any scale) → top-k donors → pipeline
```

### Implementation Plan

**1. Install cuVS in Docker image**:
```dockerfile
# In synapse-vllm Dockerfile:
RUN pip install cuvs-cu12 --extra-index-url https://pypi.nvidia.com
```

**2. Create `synapse_kv_connector/cagra_donor_store.py`** — subclass or replace `DonorStore`:
```python
from cuvs.neighbors import cagra
import cupy as cp

class CAGRADonorStore:
    """GPU-accelerated donor lookup via NVIDIA cuVS CAGRA ANN index."""

    def __init__(self, dim: int = 384, metric: str = "cosine",
                 index_params=None, search_params=None):
        self.dim = dim
        self.metric = metric
        self._embeddings: list[np.ndarray] = []
        self._donor_ids: list[str] = []
        self._index = None  # Rebuilt on demand after adds
        self._dirty = True

    def add(self, donor_id: str, embedding: np.ndarray) -> None:
        self._embeddings.append(embedding)
        self._donor_ids.append(donor_id)
        self._dirty = True

    def _rebuild_index(self):
        if not self._embeddings or not self._dirty:
            return
        data = cp.asarray(np.stack(self._embeddings), dtype=cp.float32)
        params = cagra.IndexParams(metric=self.metric)
        self._index = cagra.build(params, data)
        self._dirty = False

    def search(self, query: np.ndarray, top_k: int = 5,
               min_similarity: float = 0.60) -> list[tuple[str, float]]:
        if len(self._embeddings) == 0:
            return []
        self._rebuild_index()
        q = cp.asarray(query[np.newaxis], dtype=cp.float32)
        sp = cagra.SearchParams()
        distances, indices = cagra.search(sp, self._index, q, top_k)
        results = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            sim = 1.0 - dist  # cosine distance → similarity
            if sim >= min_similarity:
                results.append((self._donor_ids[idx], sim))
        return results
```

**3. Wire into pipeline** — swap `DonorStore` for `CAGRADonorStore` when `SEMBLEND_EMBEDDER=cagra` or `SEMBLEND_USE_CAGRA=1`.

**4. Benchmark** — measure search latency vs numpy at N=10/100/1K/10K donors. Report in paper as "production scalability" result.

### Key cuVS Notes
- cuVS CAGRA requires CUDA 11.8+. The A10G g5 nodes are CUDA 12.x — compatible.
- Package: `cuvs-cu12` for CUDA 12, `cuvs-cu11` for CUDA 11.
- For N < 64: CAGRA minimum dataset size requirement — fall back to brute-force (`cuvs.neighbors.brute_force`) below that threshold.
- cupy is required for GPU tensor passing to cuVS APIs.
- CAGRA index must be rebuilt after new donors added (or use dynamic index with `cagra.extend()`).
- For the autoresearch node (g5.2xlarge, 24GB VRAM): CAGRA index for 100K × 384-dim vectors ≈ 150MB — fits easily alongside vLLM.

---

## TRT-LLM Backend (Implementation Complete — Ready to Benchmark)

The TRT-LLM dual-backend implementation is complete in `~/dev/worldflowai/ONECONTEXT/synapse/`. **Do not modify that repo.** Sync and benchmark:

### Sync & Build
1. Run `uv run prepare.py --sync` to copy:
   - Updated `synapse_kv_connector/` (now re-export stubs) → `synapse_kv_connector/`
   - New `services/shared/semblend_core/` → `semblend_core/`
   - New `services/trtllm/` → `trtllm/`
2. Run Tier 1 tests to verify backward compat (98 tests should pass)
3. Build a new Docker image (`semblend-autoresearch-<commit>`) with both backends

**Important**: The `semblend_core` package uses `chunk_size` as a constructor arg (default 256 for vLLM, 128 for TRT-LLM). The vLLM `synapse_kv_connector` stubs re-export everything; `LMCACHE_CHUNK_SIZE` alias is preserved.

### TRT-LLM Benchmark Flow
TRT-LLM requires an engine build step (not just deploy):
```bash
# Build TRT-LLM engine for Qwen2.5-7B-AWQ
docker run --gpus all nvcr.io/nvidia/tensorrt-llm/release:latest \
  trtllm-build --model_dir ... --output_dir /engines/qwen-7b-awq \
  --max_seq_len 32768 --kv_cache_type paged --tp_size 1

# Deploy with SemBlendKVCacheManager
helm upgrade autoresearch ... -f infra/values-trtllm.yaml \
  --set providers.trtllm.engineDir=/engines/qwen-7b-awq \
  --set providers.trtllm.semblend=true
```

### What to Benchmark
1. **TTFT speedup comparison**: vLLM/LMCache path vs TRT-LLM/SemBlendKVCacheManager at 2K/8K/16K/32K
2. **Baseline latency**: TRT-LLM cold TTFT vs vLLM cold TTFT (TRT-LLM typically 1.5-3x faster cold → but SemBlend speedup is multiplicative on top)
3. **Hit rate parity**: Verify same donor store produces same hit rates
4. **Block size impact**: TRT-LLM default 128-token blocks vs LMCache 256-token blocks — measure hit rate sensitivity

### New Helm Values
Create `infra/values-trtllm.yaml` mirroring `values-autoresearch.yaml` but with:
```yaml
providers:
  defaultProvider: "trtllm"
  trtllm:
    enabled: true
    deploy: true
    semblend: true
    tokensPerBlock: 128   # TRT-LLM default (vs LMCache 256)
```

---

## NVIDIA Dynamo Integration

**Dynamo** (https://github.com/ai-dynamo/dynamo) is NVIDIA's open-source disaggregated LLM serving framework. It separates prefill workers (heavy compute) from decode workers (memory-bound), routing requests intelligently.

### Why SemBlend + Dynamo is Powerful
In standard serving, prefill is the bottleneck that SemBlend attacks. In Dynamo's disaggregated architecture:
- **Prefill workers** handle the full KV computation — this is exactly where SemBlend injects cached KV
- **SemBlend makes prefill workers nearly instantaneous** for cache hits → decode workers get results 10x faster
- **Cross-prefill-worker donor sharing**: If Dynamo routes semantically similar requests to the same prefill worker, SemBlend hit rate approaches 100%
- **Cluster-scale KV reuse**: Donor store can be backed by a distributed store (Redis, shared memory) across all prefill workers

### Integration Points
1. **Prefill worker integration**: SemBlend runs inside the TRT-LLM prefill worker. When a request arrives, run `SemBlendPipeline.run(token_ids)` — if hit, inject KV and skip prefill computation entirely.
2. **Router integration**: Modify Dynamo's request router to route semantically similar requests to the same prefill worker (improves hit rate). Use MiniLM embeddings for routing decisions.
3. **Shared donor store**: Back `CAGRADonorStore` with a shared Redis/Milvus instance so all prefill workers share the same donor pool.

### Dynamo Architecture Diagram (for paper)
```
Client Requests
      │
      ▼
 Dynamo Router ──(semantic affinity routing)──►
      │
      ├── Prefill Worker 1 [SemBlend+TRT-LLM] ──►┐
      ├── Prefill Worker 2 [SemBlend+TRT-LLM] ──►├── KV Transfer ──► Decode Workers
      └── Prefill Worker N [SemBlend+TRT-LLM] ──►┘
                │
         Shared Donor Store
         (cuVS CAGRA index)
```

### Implementation Steps
1. Set up Dynamo on 2× g5.2xlarge: 1 prefill + 1 decode (separate pods, same cluster)
2. Wire SemBlendKVCacheManager into Dynamo's TRT-LLM executor
3. Implement shared donor store (Redis-backed) so both prefill workers share donors
4. Add semantic affinity routing: use embedding similarity to sticky-route user sessions
5. Benchmark: cold TTFT vs SemBlend TTFT in disaggregated setting

---

## Multi-GPU SemBlend

### Tensor Parallel (TP) — Single-Node Multi-GPU
When running with TP=2 (e.g., g5.12xlarge with 4× A10G), each GPU holds 1/N of the KV cache. SemBlend must scatter donor KV chunks to the correct GPU shards.

**Changes required**:
- `triton_kernels.py`: The scatter kernel currently assumes single-GPU block table. Add TP-aware scatter: given `tp_rank`, scatter only layers `[tp_rank * layers_per_rank : (tp_rank+1) * layers_per_rank]`.
- `pipeline.py`: Pass `tp_rank` and `tp_size` to backend; `inject_donor_kv()` receives a per-GPU tensor slice.
- `infra/values-autoresearch.yaml`: Set `TENSOR_PARALLEL_SIZE=2` for TP experiments.

**Node for TP experiments**: Scale up to a `g5.12xlarge` (4× A10G, 96GB total VRAM) or `g5.48xlarge` (8× A10G). Much better memory headroom for large models + SemBlend.

### Data Parallel (DP) — Cross-Replica Donor Sharing
Multiple vLLM replicas behind a gateway. Each replica has an independent donor store → cache misses occur when the gateway routes the same user to different replicas.

**Solution**: Centralize the donor store via Milvus (already deployed in autoresearch namespace) + Redis (for KV tensor bytes). The connector already interfaces with Milvus for semantic lookup — extend it to store KV tensors there too.

**Architecture**:
```
vLLM Replica 1 ──► Milvus (embeddings) + Redis (KV bytes) ◄── vLLM Replica 2
                           ▲
                    vLLM Replica N
```

**Benchmark**: Scale from 1 → 4 replicas, measure hit rate with/without shared store. Expect hit rate to hold at ~90% with shared store vs ~25% without.

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

## What You Cannot Modify

- `prepare.py` — read-only setup script
- `run_experiment.py` — read-only experiment harness
- Anything in `~/dev/worldflowai/ONECONTEXT/synapse/` — the upstream synapse repo is read-only (another Claude process works there)
- `infra/ensure_gpu.sh`, `infra/deploy.sh`, `infra/teardown.sh` — infrastructure scripts

---

## Experiment Tiers

| Tier | What | Duration | When to Use |
|------|------|----------|-------------|
| 1 | `pytest synapse_kv_connector/tests/` | ~30s | Quick sanity check after code changes |
| 2 | Component microbenchmarks | 2-5 min | Validate component logic without deploy |
| 3 | **Full E2E** (build → deploy → benchmark) | 15-45 min | **Primary research loop** — produces paper results |

**Tier 3 is the primary research mode.** Only Tier 3 results go in the paper. Tiers 1 and 2 are sanity checks. There is **no budget limit** on Tier 3 runs.

---

## The Experiment Loop

LOOP FOREVER:

1. **Think** — Review `results.tsv`, `paper/ANALYSIS.md`, and `literature/`. Choose the next experiment based on research priorities. Consider what gaps remain and what will have the highest impact.

2. **Implement** — Modify code in `synapse_kv_connector/`, `benchmarks/e2e/`, or `paper/`. Make targeted changes. Commit with a descriptive message.

3. **Sanity check** — Run `uv run run_experiment.py --tier 1` to catch obvious breakage (~30s).

4. **Deploy & benchmark** — Run `uv run run_experiment.py --tier 3 --bench <benchmark>` for the real experiment. This builds a Docker image, deploys to the `autoresearch` Kubernetes namespace, and runs the benchmark against live vLLM.

5. **Evaluate** — Read the output metrics. Did TTFT improve? Did quality hold? Is hit rate acceptable?
   - If the experiment improved things: keep the commit, record in results.tsv
   - If the experiment made things worse: `git revert` or `git reset`, record as `discard` in results.tsv

6. **Log** — Append results to `results.tsv`:
   ```
   commit	tier	benchmark	primary_metric	primary_value	secondary_metrics	status	description
   a1b2c3d	3	ttft_scaling	ttft_speedup_8k	5.10	rouge_l=0.985,hit_rate=0.95	keep	increase bathtub fraction to 0.15
   ```

7. **Repeat** — Go back to step 1. Never stop.

---

## Infrastructure Is Your Responsibility

Before every Tier 3 run, `run_experiment.py` calls `infra/ensure_gpu.sh` automatically. But if infrastructure fails:

- **No GPU nodes?** The script provisions them automatically via `eksctl scale`.
- **Need larger nodes?** For multi-GPU experiments, scale up to `g5.12xlarge` (4× A10G):
  ```bash
  eksctl create nodegroup --cluster synapse-staging --name gpu-nodes-g5-12xl \
    --node-type g5.12xlarge --nodes 1 --nodes-min 0 --nodes-max 2 \
    --node-labels gpu-type=a10g,gpu-count=4 --region us-east-1
  ```
- **Deploy fails?** Read the error, fix the issue (Helm values, Dockerfile, etc.), retry.
- **Pod not starting?** Check `kubectl describe pod -n autoresearch`, fix and redeploy.
- **Port-forward dies?** Get the current pod name and restart:
  ```bash
  POD=$(kubectl get pods -n autoresearch -l app.kubernetes.io/component=vllm --no-headers | awk '{print $1}' | head -1)
  kubectl port-forward -n autoresearch $POD 8100:8000 &
  ```
- **Helm conflicts after `kubectl set env`?** Patch the deployment before helm upgrade:
  ```bash
  kubectl patch deployment autoresearch-synapse-vllm -n autoresearch --type=json \
    -p='[{"op": "remove", "path": "/spec/template/spec/volumes"}, ...]'
  ```

Node types and costs:
| Node | GPUs | VRAM | RAM | $/hr | Use For |
|------|------|------|-----|------|---------|
| g5.xlarge | 1× A10G | 24GB | 16GB | ~$1.01 | Standard experiments |
| g5.2xlarge | 1× A10G | 24GB | 32GB | ~$1.21 | Long-context (32K), large LMCache buffer |
| g5.12xlarge | 4× A10G | 96GB | 192GB | ~$5.67 | Tensor parallel, multi-GPU |
| g5.48xlarge | 8× A10G | 192GB | 768GB | ~$16.29 | Large models, TP=8 |
| p4d.24xlarge | 8× A100 | 320GB | 1.1TB | ~$32.77 | Maximum scale |

Infrastructure is NEVER an excuse to stop. Provision what you need, fix what's broken, keep going.

---

## Literature Research

Between experiments (especially while waiting for Tier 3 deploys/benchmarks):

1. **Read papers** — Use web search to find and read SemShareKV, CacheBlend, LMCache, RoFormer, Dynamo papers. Summarize key findings in `literature/`.
2. **cuVS/CAGRA docs** — Read https://docs.rapids.ai/api/cuvs/stable/python_api/neighbors_cagra/ for integration details.
3. **Dynamo docs** — Read https://github.com/ai-dynamo/dynamo for disaggregated serving architecture.
4. **Compare methodologies** — Note what competitors test that we don't, and vice versa.
5. **Update reading list** — Check off papers as you read them, add new ones you discover.
6. **Inform experiments** — Let literature findings guide your next experiment choice.

---

## Paper Work

The paper (`paper/semblend.tex`) is a first-class output. Improve it when you can:

1. **Fix inconsistencies** — See `paper/ANALYSIS.md` Part III for known issues.
2. **Add results** — As experiments produce data, update tables and figures.
3. **Improve framing** — Position SemBlend relative to SemShareKV, CacheBlend, LMCache, Dynamo.
4. **Add citations** — Fix undefined `\cite{}` references.
5. **Future work section** — Add TRT-LLM, CAGRA, Dynamo, multi-GPU as future/concurrent work.

Paper work can happen while waiting for Tier 3 benchmarks to complete — it's "free" research time.

---

## Running Benchmarks

### Core Benchmarks

```bash
# TTFT scaling (primary metric)
uv run run_experiment.py --tier 3 --bench ttft

# Long-context TTFT (16K/24K/32K) — LLaMA on g5.2xlarge with 15GB buffer
uv run python benchmarks/e2e/semblend_ttft_scaling.py \
  --endpoint http://localhost:8100 \
  --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
  --token-lengths "16000,24000,32000" \
  --runs 8 --donors-per-context 2

# Quality (ROUGE-L, PPL ratio)
uv run run_experiment.py --tier 3 --bench quality

# Ablation studies
uv run run_experiment.py --tier 3 --bench ablation

# Donor store scaling
uv run run_experiment.py --tier 3 --bench scale

# All benchmarks
uv run run_experiment.py --tier 3 --bench all
```

### WildChat-1M User-Level Reuse Benchmark

```bash
# Download 100K rows and run benchmark (requires HF_TOKEN)
HF_TOKEN=<token> PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/wildchat_semblend_bench.py \
  --download \
  --download-rows 100000 \
  --data-path benchmarks/data/wildchat/wildchat_100k.jsonl \
  --endpoint http://localhost:8100 \
  --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
  --target-tokens 4096 \
  --n-pairs 60 \
  --settle-time 5.0 \
  --output benchmarks/data/wildchat/results_llama_4k.json

# Run-only (data already downloaded)
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/wildchat_semblend_bench.py \
  --data-path benchmarks/data/wildchat/wildchat_100k.jsonl \
  --endpoint http://localhost:8100 \
  --model "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
  --target-tokens 8192 \
  --n-pairs 60
```

**Expected metrics to report**:
- `hit_rate`: % of consecutive user conversation pairs with cosine sim ≥ 0.60
- `hit_speedup_p50`: TTFT speedup on hit pairs (expect 3-8x at 4K)
- `cold_p50_ms`: baseline cold TTFT for this token length
- `n_unique_users`: number of distinct hashed_ip values in test

**Paper narrative**: "In a sample of {N} consecutive conversation pairs from WildChat-1M,
{hit_rate*100:.0f}% share sufficient semantic overlap for KV reuse (sim ≥ 0.60),
achieving {hit_speedup_p50:.1f}× TTFT speedup on matched pairs vs. cold baseline."

### Long-Output Quality Verification (PRIORITY: HIGH)

**Motivation**: Current quality benchmarks use `max_tokens=256`, which is unrealistically short for production LLM responses. Real-world outputs are 500–2000+ tokens. We need to verify that SemBlend's PPL ratio remains stable (≈1.0) when generating much longer responses — if KV injection is correct at the prompt level, output quality should not degrade regardless of generation length.

**Hypothesis**: PPL ratio stability is independent of output length because donor KV only affects prompt prefill, not autoregressive generation. If hypothesis holds, PPL ratio at max_tokens=2048 ≈ PPL ratio at max_tokens=256. If it degrades, the KV injection has positional encoding or attention artifacts that compound over long outputs.

**Experiment sequence** (run in this order, one Tier 3 run each):

1. **Baseline sweep** — Rerun quality benchmark at max_tokens=512, 1024, 2048 on XSum synthetic (Qwen, 8K context):
   ```bash
   for MAX_TOK in 512 1024 2048; do
     python benchmarks/e2e/semblend_quality_bench.py \
       --endpoint http://localhost:8100 \
       --model "Qwen/Qwen2.5-7B-Instruct-AWQ" \
       --clusters-file benchmarks/data/cnn_dailymail_clusters.json \
       --target-length 8192 --n-clusters 25 \
       --max-tokens $MAX_TOK \
       --output results/quality_qwen_8k_maxtok${MAX_TOK}.json
   done
   ```

2. **Cross-model check** — Repeat with LLaMA at max_tokens=1024 and max_tokens=2048 to verify the pattern holds across architectures.

3. **Cross-dataset check** — Run max_tokens=1024 on WikiHow and SAMSum (already have 256-token baselines to compare against).

4. **Degradation curve** — If PPL ratio is stable up to 2048, try max_tokens=4096. Plot PPL ratio vs. output length.

**Key metrics to capture**:
- `ppl_ratio_2k / 4k / 8k` at each max_tokens setting (should stay ~stable)
- `rouge_l_avg` — may improve at longer max_tokens since ROUGE-L has more text to match
- `hit_rate` — verify unchanged (hit detection is independent of max_tokens)
- Absolute generation time (longer outputs = more GPU time, document for cost awareness)

**Paper impact**: Adds a "quality at scale" subsection demonstrating production readiness. Target claim: "PPL ratio remains below {X} even at 2048 output tokens, confirming that KV donor injection does not compound quality degradation during long-form generation."

**Expected result**: PPL ratio at max_tokens=2048 within ±0.02 of max_tokens=256 value. If we see PPL ratio growing with output length, investigate PartialAttention RoPE correction (check `use_alignment` flag and position ID handling in `synapse_kv_connector/partial_attn.py`).

**Results to record in results.tsv**:
```
<commit>	3	quality_longout_512	ppl_ratio_8k	<val>	ppl_2k=<v>,ppl_4k=<v>,hit_rate=<v>	keep	quality at max_tokens=512
<commit>	3	quality_longout_1024	ppl_ratio_8k	<val>	ppl_2k=<v>,ppl_4k=<v>,hit_rate=<v>	keep	quality at max_tokens=1024
<commit>	3	quality_longout_2048	ppl_ratio_8k	<val>	ppl_2k=<v>,ppl_4k=<v>,hit_rate=<v>	keep	quality at max_tokens=2048
```

### CAGRA/cuVS GPU Search Benchmark

Measures ANN search latency at scale: numpy cosine scan vs CAGRA on GPU.

```bash
# First install cuVS in the Docker image (add to Dockerfile):
# RUN pip install cuvs-cu12 --extra-index-url https://pypi.nvidia.com

# Run CAGRA latency benchmark (after installing cuvs):
PYTHONUNBUFFERED=1 .venv/bin/python -u \
  benchmarks/e2e/cagra_search_benchmark.py \
  --donor-counts "10,100,1000,10000,100000" \
  --dim 384 \
  --query-count 1000 \
  --compare-numpy

# Create the benchmark script (benchmarks/e2e/cagra_search_benchmark.py):
# - Build CAGRA index with N random 384-dim float32 vectors
# - Run 1000 cosine queries, measure p50/p99 latency
# - Compare: numpy scan vs cuVS brute_force vs cuVS CAGRA
# - Report: latency table + throughput (queries/sec)
# - Verify: top-1 recall@10 ≥ 0.99 for CAGRA vs exact
```

**Expected results** (from NVIDIA benchmarks):
| N donors | numpy scan | cuVS CAGRA | Speedup |
|----------|-----------|------------|---------|
| 100      | 0.01ms    | 0.5ms*     | 0.02x   |
| 1,000    | 0.1ms     | 0.5ms*     | 0.2x    |
| 10,000   | 1ms       | 0.5ms*     | 2x      |
| 100,000  | 10ms      | 0.5ms*     | 20x     |
| 1,000,000| 100ms     | 0.5ms*     | 200x    |

*CAGRA GPU latency is roughly constant due to parallel GPU execution. Numpy degrades linearly. Crossover ~10K donors. At 100K+, CAGRA is 20-200x faster.

**Paper impact**: Demonstrates production scalability. Current numpy scan works for lab N<1000; production needs CAGRA at N=100K+ cached conversations.

### Multi-GPU SemBlend Benchmark (g5.12xlarge — 4× A10G)

Tests tensor parallel (TP=2/4) to show horizontal scalability of KV injection.

```bash
# Scale up to g5.12xlarge (4× A10G, 96GB VRAM)
eksctl create nodegroup --cluster synapse-staging --name gpu-nodes-g5-12xl \
  --node-type g5.12xlarge --nodes 1 --nodes-min 0 --nodes-max 2 \
  --node-labels gpu-type=a10g,gpu-count=4 \
  --node-taints nvidia.com/gpu=present:NoSchedule \
  --region us-east-1

# Deploy with TP=2 (2 GPUs, model sharded across them)
# Update values-autoresearch.yaml:
#   providers.vllm.tensorParallelSize: 2
#   resources.limits.nvidia.com/gpu: 2
#   nodeSelector: eks.amazonaws.com/nodegroup: gpu-nodes-g5-12xl
helm upgrade autoresearch ... -f infra/values-autoresearch.yaml

# Then run standard TTFT benchmark to compare TP=1 vs TP=2 vs TP=4
# Expected: cold TTFT improves (more VRAM = larger model fits + better compute)
# SemBlend TTFT should remain ~constant (KV transfer is the bottleneck, not GPU)
# → Speedup ratio increases with TP because cold gets faster but SemBlend stays constant
```

**Metrics to collect**:
- `cold_ttft_tp{1,2,4}`: cold TTFT baseline at each TP level
- `semblend_ttft_tp{1,2,4}`: SemBlend TTFT at each TP level
- `speedup_tp{1,2,4}`: speedup at each TP level
- `hit_rate_tp{1,2,4}`: verify hit rate doesn't degrade with TP

**Expected insight**: With TP=2, cold prefill gets faster (2 GPUs compute in parallel), but SemBlend TTFT stays ~constant → speedup ratio DECREASES vs TP=1 because cold is now faster. But at very long contexts (32K+), cold is SO slow that SemBlend still wins even at TP=2.

**Alternative TP insight**: With TP=2 and 48GB total VRAM per model shard, we can run larger models (LLaMA-70B) or longer contexts (64K+), where SemBlend's constant TTFT becomes even more dominant.

### Cross-Replica Donor Sharing Benchmark (2× replicas)

```bash
# Deploy 2 vLLM replicas with shared Milvus donor store
helm upgrade autoresearch ... \
  --set providers.vllm.replicaCount=2 \
  --set providers.vllm.kvConnector.sharedDonorStore=true

# Register donor via replica 1, query via replica 2
# Verify: hit rate with shared store ≈ single-replica hit rate
# Without shared store: hit rate drops to ~1/N (random routing)
```

---

## Output Format

Tier 3 experiments print parseable metrics:

```
---
tier:               3
benchmark:          ttft_scaling
ttft_speedup_2k:    2.15
ttft_speedup_5k:    3.20
ttft_speedup_8k:    5.10
ttft_speedup_16k:   7.50
ttft_speedup_32k:   12.0
rouge_l_avg:        0.985
ppl_ratio_avg:      1.002
hit_rate_avg:       0.95
duration_seconds:   1847
---
```

---

## Cost Awareness

- g5.xlarge: ~$1/hr. Keep running during active experimentation.
- g5.2xlarge: ~$1.21/hr. Use for long-context experiments (better LMCache buffer headroom).
- If doing paper-only or literature work for 30+ minutes with no upcoming Tier 3 runs, consider `bash infra/teardown.sh`.
- Re-provisioning takes ~3-5 minutes, so don't tear down between back-to-back experiments.
- Multi-GPU nodes (g5.12xlarge+) are expensive — spin up only when actively benchmarking TP experiments.

---

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. The human may be asleep. You are autonomous. If you run out of ideas:

- Re-read `paper/ANALYSIS.md` for unfilled gaps
- Re-read `literature/` for inspiration from related work
- Start on CAGRA donor store implementation — it's always a win
- Try WildChat-1M benchmark if not yet done
- Try ablation studies (threshold sweep, bathtub sweep)
- Work on the paper while thinking
- Start drafting the Dynamo integration design in `literature/dynamo_notes.md`

The loop runs until the human interrupts you, period.
