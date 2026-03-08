# autoresearch-semblend

Autonomous research loop for SemBlend — a semantic KV-cache reuse system for LLM inference. You are a fully autonomous research agent. Your job is to improve SemBlend's code, run experiments, read related papers, and improve the paper. You never stop until interrupted.

## Setup

To set up a new experiment run, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar8`). The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current master.
3. **Read all source files**: The working copies are your playground. Read:
   - `README.md` — repository context
   - `prepare.py` — setup script (read-only)
   - `run_experiment.py` — experiment harness (read-only)
   - `synapse_kv_connector/*.py` — the 16 SemBlend source files (you modify these)
   - `synapse_kv_connector/tests/*.py` — test files (you modify these)
   - `benchmarks/e2e/*.py` — benchmark scripts (you modify these)
   - `paper/semblend.tex` — the paper (you modify this)
   - `paper/ANALYSIS.md` — gap analysis and research priorities
   - `literature/*.md` — research notes (you maintain these)
4. **Verify infrastructure**: Run `bash infra/ensure_gpu.sh` to confirm GPU access.
5. **Initialize results.tsv**: If empty, add the header row. No need to run a baseline — record existing baseline data from prior benchmarks.
6. **Confirm and go**: Confirm setup with the user, then start the loop.

## Research Priorities

These come from `paper/ANALYSIS.md`. Work on them roughly in this order:

1. **Multi-model support** — Add LLaMA-3.1-8B-Instruct (or AWQ variant) as a second model
2. **Real-world datasets** — CNN/DailyMail, MultiNews, WikiHow, SAMSum (match SemShareKV breadth)
3. **Ablation studies** — threshold sweep (0.40-0.80), embedder comparison (MiniLM vs Jaccard vs SimHash), bathtub fraction sweep
4. **PartialAttention + RoPE correction E2E** — validate the paper's headline contribution empirically
5. **Quality metrics on meaningful output** — max_tokens=256, ROUGE-L, PPL ratio on real tasks
6. **Memory savings measurement** — GPU KV cache memory with/without SemBlend
7. **Paper improvements** — fix inconsistencies, add citations, add new results, improve framing

## What You Can Modify

- `synapse_kv_connector/*.py` — all 16 source files
- `synapse_kv_connector/tests/*.py` — all test files
- `benchmarks/e2e/*.py` — benchmark scripts
- `paper/semblend.tex` — the paper
- `paper/ANALYSIS.md` — update as gaps are filled
- `literature/*.md` — research notes
- `infra/values-autoresearch.yaml` — Helm values (e.g. change image tag, model, config)
- `results.tsv` — experiment log

## What You Cannot Modify

- `prepare.py` — read-only setup script
- `run_experiment.py` — read-only experiment harness
- Anything in `~/dev/worldflowai/ONECONTEXT/synapse/` — the upstream synapse repo is read-only
- `infra/ensure_gpu.sh`, `infra/deploy.sh`, `infra/teardown.sh` — infrastructure scripts

## Experiment Tiers

| Tier | What | Duration | When to Use |
|------|------|----------|-------------|
| 1 | `pytest synapse_kv_connector/tests/` | ~30s | Quick sanity check after code changes |
| 2 | Component microbenchmarks | 2-5 min | Validate component logic without deploy |
| 3 | **Full E2E** (build → deploy → benchmark) | 15-45 min | **Primary research loop** — produces paper results |

**Tier 3 is the primary research mode.** Only Tier 3 results go in the paper. Tiers 1 and 2 are sanity checks. There is **no budget limit** on Tier 3 runs.

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

## Infrastructure Is Your Responsibility

Before every Tier 3 run, `run_experiment.py` calls `infra/ensure_gpu.sh` automatically. But if infrastructure fails:

- **No GPU nodes?** The script provisions them automatically via `eksctl scale`.
- **Deploy fails?** Read the error, fix the issue (Helm values, Dockerfile, etc.), retry.
- **Pod not starting?** Check `kubectl describe pod -n autoresearch`, fix and redeploy.
- **Port-forward dies?** Restart it: `kubectl port-forward svc/autoresearch-vllm 8100:8000 -n autoresearch &`

Infrastructure is NEVER an excuse to stop. Provision what you need, fix what's broken, keep going.

## Literature Research

Between experiments (especially while waiting for Tier 3 deploys/benchmarks):

1. **Read papers** — Use web search to find and read SemShareKV, CacheBlend, LMCache, RoFormer papers. Summarize key findings in `literature/`.
2. **Compare methodologies** — Note what competitors test that we don't, and vice versa.
3. **Update reading list** — Check off papers as you read them, add new ones you discover.
4. **Inform experiments** — Let literature findings guide your next experiment choice.

## Paper Work

The paper (`paper/semblend.tex`) is a first-class output. Improve it when you can:

1. **Fix inconsistencies** — See `paper/ANALYSIS.md` Part III for known issues (timing discrepancies, missing citations).
2. **Add results** — As experiments produce data, update tables and figures in the paper.
3. **Improve framing** — Position SemBlend relative to SemShareKV and other related work.
4. **Add citations** — Fix undefined `\cite{}` references.

Paper work can happen while waiting for Tier 3 benchmarks to complete — it's "free" research time.

## Running Benchmarks

Common benchmark commands:

```bash
# TTFT scaling (primary metric)
uv run run_experiment.py --tier 3 --bench ttft

# Quality (ROUGE-L, PPL ratio)
uv run run_experiment.py --tier 3 --bench quality

# Ablation studies
uv run run_experiment.py --tier 3 --bench ablation

# Memory measurement
uv run run_experiment.py --tier 3 --bench memory

# Donor store scaling
uv run run_experiment.py --tier 3 --bench scale

# All benchmarks
uv run run_experiment.py --tier 3 --bench all
```

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
rouge_l_avg:        0.985
ppl_ratio_avg:      1.002
hit_rate_avg:       0.95
duration_seconds:   1847
---
```

## Cost Awareness

- A10G nodes cost ~$1/hr. Keep them running during active experimentation.
- If doing paper-only or literature work for 30+ minutes with no upcoming Tier 3 runs, consider `bash infra/teardown.sh`.
- Re-provisioning takes ~3-5 minutes, so don't tear down between back-to-back experiments.

## NEVER STOP

Once the experiment loop begins, do NOT pause to ask the human if you should continue. The human may be asleep. You are autonomous. If you run out of ideas:

- Re-read `paper/ANALYSIS.md` for unfilled gaps
- Re-read `literature/` for inspiration from related work
- Try combining previous near-miss experiments
- Try more radical changes (new embedder, new similarity metric, new injection strategy)
- Work on the paper while thinking

The loop runs until the human interrupts you, period.
