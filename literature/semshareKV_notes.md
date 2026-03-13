# SemShareKV Notes

## Key Architecture Differences vs SemBlend

| Aspect | SemShareKV | SemBlend |
|--------|-----------|----------|
| Matching | Token-level LSH, O(T) | Embedding-level, O(1) |
| Position | First-layer recompute (heuristic) | Exact RoPE delta correction |
| Framework | HuggingFace Transformers (custom) | vLLM + LMCache (production) |
| Donors | Single reference per target | Multi-donor store (10K+) |
| Max sequence | 5K tokens | 16K+ tokens |

## Evaluation Methodology
- 3 models: Mistral-7B, LLaMA-3.1-8B, MPT-7B
- 9 datasets: MultiNews, WikiHow, Qasper, SAMSum, PubMed, BookSum, BigPatent, LCC, MMLU
- 4 baselines: Full Recompute, SnapKV, PyramidKV, H2O
- 3 ablation studies: fuzzy+full cache, zero ablation, random ablation
- Quality metric: ROUGE-L on full generation outputs
- 42% KV cache memory reduction reported

## What We Must Match
1. Multi-model testing (at least 2 models)
2. Real-world datasets (at least 3)
3. Baseline comparisons (at least prefix caching + cold)
4. Ablation studies (at least threshold sweep + embedder comparison)
5. Quality metrics on meaningful output (max_tokens=256)

## Where We're Already Better
1. Production vLLM integration (vs HuggingFace hack)
2. Exact RoPE correction (vs brute-force first-layer recompute)
3. Longer sequences (16K vs 5K)
4. Honest negative results (0.72x on diverse workloads)
5. Statistical methodology (n=10, per-length restarts)
