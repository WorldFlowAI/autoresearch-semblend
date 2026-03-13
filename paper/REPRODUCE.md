# Reproducing SemBlend Results

End-to-end instructions for reproducing the SemBlend paper evaluation.
All benchmarks run locally via `kubectl port-forward` — no GitHub Actions.

## Prerequisites

| Requirement | Details |
|---|---|
| AWS Account | ECR push access + EKS cluster admin |
| EKS Cluster | `synapse-staging` in us-east-1 (or any cluster with GPU nodes) |
| GPU Nodes | g5.xlarge (A10G, 24GB) for 7B-AWQ; g4dn.xlarge (T4, 16GB) for 1.5B |
| CLI Tools | `aws`, `eksctl`, `kubectl`, `helm`, `docker buildx`, `python3` |
| Python | 3.10+ with `pip install requests datasets transformers numpy` |
| ECR Registry | `198333343348.dkr.ecr.us-east-1.amazonaws.com` (replace with yours) |

## Architecture

SemBlend Local (primary evaluation target, zero network hops):
- SimHash pre-filter (~0.1ms, 64-bit token n-gram fingerprint)
- MiniLM-L6-v2 CPU embedding (~5ms, 384-dim, sentence-transformers)
- Numpy cosine donor store (~1-2ms for 1K entries, ~2ms for 10K)
- rapidfuzz token alignment (<1ms, compiled C++ Levenshtein opcodes)
- Multi-candidate KV fallback (tries ranked donors until KV found in LMCache)
- Recency bias in donor scoring (prefers recently-stored donors)
- Calibrated bathtub recomputation (early/late 12.5% of layers)
- LMCache KV offload + chunk-swap injection (~20-35ms CPU→GPU transfer)
- LMCache CPU buffer: 10GB (holds ~23 donors at 8K, ~11 at 16K)
- Total pipeline decision overhead: ~8ms (excluding KV load)

## 1. Build Real-World Datasets

Build prompt clusters from real-world summarization datasets. We evaluate
on the same datasets as SemShareKV (AACL 2025) plus additional sources
for broader coverage.

### Supported Datasets

| Dataset | Source | SemShareKV? | Prompt Length | Notes |
|---------|--------|-------------|---------------|-------|
| CNN/DailyMail | `cnn_dailymail` | Yes | Medium-Long | News articles, SemShareKV primary |
| XSum | `EdinburghNLP/xsum` | Yes | Short-Medium | BBC article summaries |
| WikiHow | `xinghan/wikihow-en` | Yes | Long | Procedural instruction articles |
| SAMSum | `Samsung/samsum` | Yes | Short | Messenger-style dialogues |
| MultiNews | `multi_news` | Yes | Very Long | Multi-document summarization |
| ShareGPT | `anon8231489123/ShareGPT_Vicuna_unfiltered` | No | Variable | Production conversation traces |

Each cluster contains a seed prompt and 8 variations:
EXACT, REORDER, PARTIAL (80/60/40/20%), PARAPHRASE, DIVERSE.

```bash
cd /path/to/synapse

# Install dataset dependencies
pip install datasets transformers

# Build clusters from all SemShareKV-comparable datasets
python -m benchmarks.e2e.real_dataset_builder \
  --output benchmarks/data/semblend_clusters.json \
  --datasets cnn_dailymail xsum wikihow samsum multinews \
  --target-lengths 1024 2048 4096 8192 16384 \
  --max-articles 200 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --seed 42

# For LLaMA-3.1-8B evaluation (different tokenizer → different clusters)
python -m benchmarks.e2e.real_dataset_builder \
  --output benchmarks/data/semblend_clusters_llama.json \
  --datasets cnn_dailymail xsum wikihow samsum multinews \
  --target-lengths 1024 2048 4096 8192 16384 \
  --max-articles 200 \
  --model meta-llama/Llama-3.1-8B-Instruct \
  --seed 42
```

This produces ~1000 clusters per model (200 articles x 5 lengths), each
with 8 variations = 8,000 total prompt pairs per model. Takes ~5 min
(HuggingFace download + tokenization).

## 2. Build and Push Docker Image

```bash
ECR=198333343348.dkr.ecr.us-east-1.amazonaws.com
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin $ECR

# Build from project root — Dockerfile copies synapse_kv_connector/
docker buildx build --platform linux/amd64 \
  -t $ECR/synapse-vllm:semblend-local-v11 \
  -f services/vllm/Dockerfile --push .
```

The Dockerfile installs: `lmcache`, `sentence-transformers`,
`rapidfuzz`, `onnxruntime-gpu`, and bundles the full
`synapse_kv_connector/` package (pipeline, donor store, alignment,
bathtub, RoPE correction, PartialAttention kernels).

## 3. Deploy

```bash
# Scale up GPU nodes
eksctl scale nodegroup --cluster synapse-staging --name gpu-nodes-g5 --nodes 1
kubectl get nodes -l nvidia.com/gpu=true -w  # Wait for Ready

# Update image tag in values/staging.yaml, then deploy
helm upgrade --install synapse-staging ./helm/synapse \
  -n synapse-staging \
  -f helm/synapse/values/staging.yaml

# Watch pod (model download + load ~3-5 min first time)
kubectl get pods -n synapse-staging -l app=vllm -w

# Verify health
kubectl exec -n synapse-staging deploy/synapse-staging-vllm -- \
  curl -s http://localhost:8000/health

# Port-forward for benchmarks
kubectl port-forward -n synapse-staging svc/synapse-staging-vllm 8001:8000 &
```

Key environment variables (set by Helm):

| Variable | Value | Purpose |
|---|---|---|
| `SEMBLEND_ENABLED` | `1` | Enable semantic donor discovery |
| `SEMBLEND_USE_PIPELINE` | `1` | In-process SemBlend Local pipeline |
| `SEMBLEND_EMBEDDER` | `minilm` | ONNX GPU MiniLM-L6-v2 embedding |
| `SEMBLEND_USE_ALIGNMENT` | `1` | rapidfuzz token alignment |
| `SEMBLEND_MIN_SIMILARITY` | `0.60` | Cosine similarity threshold |
| `LMCACHE_CONFIG_FILE` | config path | LMCache config (10GB CPU buffer) |

## 4. Benchmark Suite Overview

The evaluation consists of 7 benchmarks that together provide the
empirical evidence for the paper. Run them in order — each builds on
the previous.

| # | Benchmark | What It Measures | GPU Required? |
|---|-----------|-----------------|---------------|
| 4a | TTFT Scaling | Primary speedup claim (2.1x→7.5x) | Yes |
| 4b | Prefix Cache Baseline | vLLM native prefix cache comparison | Yes |
| 4c | Quality (Long Generation) | ROUGE-L, EM, PPL with max_tokens=256 | Yes |
| 4d | Scale Benchmark | Multi-dataset, multi-scenario evaluation | Yes |
| 4e | RoPE Correction Ablation | Ablation: with vs without RoPE correction | Yes |
| 4f | Memory Savings | GPU memory utilization reduction | Yes |
| 4g | Bathtub Curve Validation | Per-layer deviation measurement | Yes |

### 4a. TTFT Scaling Benchmark (Per-Length Pod Restart)

This is the primary benchmark — measures TTFT speedup at multiple
prompt lengths with per-length pod restarts to prevent cross-length
contamination.

**Why per-length restarts?** Cold runs register as donors via
`request_finished()`. Without restarting, cold runs at 2K create
donors that inflate hit rates for subsequent 5K/8K/16K cold baselines,
corrupting the measurements.

**Prompt length verification:** All prompts are built using
`AutoTokenizer` from Hugging Face Transformers to ensure exact token
counts. Early benchmarks used a character-based approximation (4
chars/token) that produced prompts ~60% shorter than claimed due to
Qwen's BPE ratio of ~6.5 chars/token.

```bash
# Per-length benchmark with pod restarts:
#   For each token length:
#     1. Restart vLLM pod (clear donor store + LMCache)
#     2. Cold baselines (10 unique contexts, fresh pod)
#     3. Seed donors (same contexts)
#     4. SemBlend measurements (same contexts, different ref_id)
python -m benchmarks.e2e.semblend_per_length_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --token-lengths "2048,5120,8192,16000" \
  --runs 10 \
  --output results/per_length_qwen7b.json
```

Note: Use 16000 instead of 16384 to leave room for generation tokens
within max_model_len=16384.

#### Expected Results (v10, A10G, per-length clean pod)

| Tokens | Cold P50 | SemBlend P50 | Speedup | Hit Rate | ROUGE-L | PPL Ratio |
|--------|----------|-------------|---------|----------|---------|-----------|
| 2K | ~1,369ms | ~653ms | ~2.10x | 100% | --- | --- |
| 5K | ~2,246ms | ~714ms | ~3.15x | 100% | 1.000 | 1.000 |
| 8K | ~4,048ms | ~785ms | ~5.16x | 90% | 0.991 | 1.005 |
| 16K | ~6,781ms | ~909ms | ~7.46x | 100% | 1.000 | 1.001 |

SemBlend TTFT is near-constant (~650-910ms) while cold prefill scales
linearly with prompt length. Peak individual speedup: 9.12x at 16K
(744ms vs 6,781ms).

#### Quality Preservation

| Length | ROUGE-L | Exact Match | PPL Ratio |
|--------|---------|-------------|-----------|
| 5K | 1.000 | 100% | 1.000 |
| 8K | 0.991 | 83% | 1.005 |
| 16K | 1.000 | 100% | 1.001 |

Quality improves with prompt length: the 12-character reference ID
(~3 tokens) is a smaller fraction at longer lengths.

### 4b. Prefix Cache Baseline Comparison

Measure vLLM's native prefix cache to quantify SemBlend's advantage
over exact-match-only caching.

```bash
python -m benchmarks.e2e.prefix_cache_baseline_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --token-lengths "2048,5120,8192,16000" \
  --runs 10 \
  --output-dir results/prefix_cache_baseline
```

**Key insight:** vLLM prefix cache only helps with exact prefix matches.
SemBlend activates on semantic similarity (reorder, partial, paraphrase)
— scenarios where prefix cache misses entirely.

| Scenario | Prefix Cache | SemBlend |
|----------|-------------|----------|
| EXACT_PREFIX (same context, diff question) | Fast | Fast |
| REORDER (shuffled sentences) | Full cold | Fast (RoPE correction) |
| PARTIAL (overlapping content) | Partial | Fast |
| PARAPHRASE (rephrased) | Full cold | Fast |

### 4c. Quality Benchmark (Extended Generation)

Measure output quality with longer generation (max_tokens=256) to
validate that SemBlend doesn't degrade output quality even for
extended responses.

```bash
python -m benchmarks.e2e.semblend_quality_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --token-lengths "2048,5120,8192,16000" \
  --runs 10 \
  --max-tokens 256 \
  --output-dir results/quality_extended
```

Metrics:
- **ROUGE-L**: LCS F1 between SemBlend output and cold baseline output
- **Exact Match**: Fraction of identical outputs (at temperature=0)
- **Perplexity Ratio**: exp(-logprob_semblend) / exp(-logprob_cold) via vLLM logprobs
- PPL ratio near 1.0 indicates no quality degradation

### 4d. Scale Benchmark Suite

Full benchmark with real datasets, multiple scenarios, and quality:

```bash
# Ensure clusters are built (Step 1)
python -m benchmarks.e2e.semblend_scale_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --clusters benchmarks/data/semblend_clusters.json \
  --donor-scales 100,1000 \
  --scenarios cold exact reorder partial_80 partial_60 partial_40 partial_20 paraphrase diverse \
  --num-runs 10 \
  --hardware A10G \
  --output-dir results/semblend-scale-7b-a10g
```

#### Metrics

| Metric | Method | SemShareKV Comparison |
|---|---|---|
| **TTFT** | p50/p95/p99 + bootstrap 95% CI (n=10) | SemShareKV: latency (no CIs) |
| **Speedup** | TTFT_cold / TTFT_semblend per scenario | SemShareKV: up to 6.25x |
| **ROUGE-L** | LCS F1 vs cold baseline output | SemShareKV: ROUGE-1/2/L |
| **Exact Match** | Fraction identical output | Not measured by SemShareKV |
| **Perplexity Ratio** | exp(-logprob) ratio via vLLM logprobs | Not measured by SemShareKV |
| **Donor Store Scale** | 100, 1K entries | SemShareKV: single reference |
| **Token Lengths** | 1K, 2K, 4K, 8K, 16K tokens | SemShareKV: up to 5K |

#### Scenarios

| Scenario | Description | RoPE Correction? |
|---|---|---|
| **cold** | No donor available (baseline) | N/A |
| **exact** | Identical prompt | No (d=t for all) |
| **reorder** | Same sentences, shuffled order | Yes (d != t) |
| **partial_80** | 20% sentences replaced | Partial |
| **partial_60** | 40% sentences replaced | Partial |
| **partial_40** | 60% sentences replaced | Partial |
| **partial_20** | 80% sentences replaced | Partial |
| **paraphrase** | Synonym substitution | Yes |
| **diverse** | Completely different article | N/A (no match) |

### 4e. RoPE Correction Ablation

Validates the key contribution: exact RoPE delta correction enables
non-contiguous KV reuse. Measures REORDER quality WITH vs WITHOUT
correction.

```bash
python -m benchmarks.e2e.rope_ablation_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --clusters benchmarks/data/semblend_clusters.json \
  --num-runs 20 \
  --hardware A10G \
  --output-dir results/rope-ablation
```

Expected: Without RoPE correction, REORDER outputs should show
significant quality degradation (ROUGE-L < 0.5) because donor K
tensors encode wrong positions.

### 4f. Memory Savings Measurement

Measure GPU memory utilization reduction from KV cache reuse.

```bash
python -m benchmarks.e2e.semblend_memory_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --num-donors 50 \
  --output-dir results/memory_savings
```

### 4g. Bathtub Curve Validation

Measure per-layer KV deviation to validate the bathtub recomputation
heuristic (paper Eq. 4).

```bash
python -m benchmarks.e2e.semblend_comprehensive_bench \
  --endpoint http://localhost:8001 \
  --model Qwen/Qwen2.5-7B-Instruct-AWQ \
  --output-dir results/bathtub_validation
```

## 5. Multi-Model Evaluation

To match SemShareKV's multi-model evaluation (Mistral-7B, LLaMA-3.1-8B,
MPT-7B), run benchmarks on additional models.

### LLaMA-3.1-8B-Instruct-AWQ

Requires redeploying with the LLaMA AWQ model (INT4, fits on A10G 24GB):

```bash
# Deploy with LLaMA model (override via Helm):
helm upgrade --install synapse-staging ./helm/synapse \
  -n synapse-staging \
  -f helm/synapse/values/staging.yaml \
  --set providers.vllm.model="hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4" \
  --set providers.vllm.quantization="awq"

# Wait for model load (~5-10 min), then rebuild clusters with LLaMA tokenizer:
python -m benchmarks.e2e.real_dataset_builder \
  --dataset cnn_dailymail xsum \
  --target-lengths "2048,5120,8192,16384" \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --output benchmarks/data/semblend_clusters_llama.json

# Run per-length TTFT benchmark:
python -m benchmarks.e2e.semblend_per_length_bench \
  --endpoint http://localhost:8001 \
  --model hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 \
  --token-lengths "2048,5120,8192,16000" \
  --runs 10 \
  --output results/per_length_llama8b.json
```

### Model Comparison Table (Target)

| Model | Hardware | 2K | 5K | 8K | 16K |
|-------|----------|----|----|----|----|
| Qwen2.5-7B-AWQ | A10G | 2.10x | 3.15x | 5.16x | 7.46x |
| LLaMA-3.1-8B | A10G | 2.33x | — | 5.75x | — |

## 6. SemShareKV Comparison Summary

| Metric | SemBlend | SemShareKV |
|--------|----------|------------|
| Peak speedup | 7.46x (16K P50), 9.12x peak | 6.25x (5K) |
| Hardware | A10G ($1.01/hr) | A100 ($3-6/hr) |
| Models tested | Qwen2.5-7B-AWQ + LLaMA-3.1-8B | Mistral-7B, LLaMA-3.1-8B, MPT-7B |
| Datasets | CNN/DM, XSum, WikiHow, SAMSum, MultiNews | MultiNews, WikiHow, Qasper, SAMSum, PubMed, BookSum, BigPatent, LCC, MMLU |
| Tested lengths | 2K-16K (4 lengths) | 5K only |
| Scaling curve | Yes (linear increase with length) | No |
| Quality metrics | ROUGE-L, EM, PPL ratio | ROUGE-L only |
| Ablations | RoPE correction, bathtub curve | E-cache/R-cache, layer-level recomp |
| Baselines | vLLM prefix cache | Full KV, prompt cache |
| Donor store scale | 100, 1K, 10K entries | Single reference |
| Below 700 tokens | Overhead-limited (~8ms) | "Limited" (no data) |
| Statistical rigor | P50/P95/P99 + bootstrap 95% CI (n=10) | Mean latency (no CIs) |

## 7. Full Suite (All Benchmarks)

```bash
./benchmarks/e2e/run_full_semblend_bench.sh http://localhost:8001 Qwen/Qwen2.5-7B-Instruct-AWQ
```

## 8. Output Files

Results are saved per run:
- `per_length_*.json` — TTFT data with raw arrays and hit rates
- `prefix_cache_baseline/` — Prefix cache comparison data
- `quality_extended/` — Quality metrics with max_tokens=256
- `scale_report.json` — Full data with raw TTFT arrays and CIs
- `scale_results.csv` — For plotting (pandas/matplotlib)
- `scale_table.tex` — LaTeX table ready for paper inclusion
- `rope_ablation.json` — RoPE correction ablation data
- `memory_savings/` — GPU memory utilization data

## 9. Scale Down (Cost Saving)

```bash
eksctl scale nodegroup --cluster synapse-staging --name gpu-nodes-g5 --nodes 0
eksctl scale nodegroup --cluster synapse-staging --name gpu-nodes --nodes 0
```

GPU costs: ~$0.53/hr (T4), ~$1.01/hr (A10G). Always scale down.

## Troubleshooting

### SemBlend not finding donors
Check `[SemBlend] get_matched` logs. If prefix cache hit covers >50%
of prompt, SemBlend skips (proportional threshold).

### "donor KV NOT in cache" / all candidates evicted
LMCache CPU buffer is full. Increase `max_local_cpu_size` in
`lmcache_config.yaml` (default: 10GB). Each 8K donor uses ~440MB.
The multi-candidate fallback (v10+) tries multiple donors ranked by
recency, but if ALL are evicted, this won't help.

### Port-forward dying during benchmarks
Use `./benchmarks/e2e/port_forward_keepalive.sh` — auto-restarts on failure.

### vLLM OOM
Reduce `gpuMemoryUtilization` from `0.85` to `0.80` or
`maxModelLen` from `16384` to `8192`.

### 16K prompts returning 400 Bad Request
One of the 8 contexts exceeds `max_model_len=16384` after tokenization.
Context [4] (cloud GPU economics) with padding exceeds the limit.
The benchmark handles this gracefully — failed runs are excluded.

### No REORDER speedup
Verify RoPE correction is enabled in the pipeline. Without RoPE
correction, donor K tensors have wrong position encoding.

### Quality degradation on PARTIAL scenarios
Expected for high replacement ratios (60-80% replaced). The bathtub
curve catches layers with high attention drift and recomputes them.
If quality is still low, increase `SEMBLEND_MIN_SIMILARITY` threshold.

### Different model tokenizers
Each model has a different tokenizer with different BPE vocabularies.
Prompt clusters must be rebuilt per model using `real_dataset_builder.py`
with the correct `--model` argument to ensure accurate token counts.
