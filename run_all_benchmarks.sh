#!/bin/bash
# Run all live benchmarks sequentially with n>=1000 samples.
# Each benchmark restarts vLLM between runs for clean state.
set -euo pipefail

ENDPOINT="http://localhost:8100"
MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
NAMESPACE="autoresearch"
RESULTS_DIR="results"
mkdir -p "$RESULTS_DIR"

restart_vllm() {
    echo "=== Restarting vLLM for clean state ==="
    kubectl rollout restart deployment/autoresearch-synapse-vllm -n "$NAMESPACE"
    kubectl rollout status deployment/autoresearch-synapse-vllm -n "$NAMESPACE" --timeout=300s
    # Kill old port-forward and set up new one
    pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true
    sleep 5
    POD=$(kubectl get pods -n "$NAMESPACE" -l app=vllm --no-headers | awk '{print $1}' | head -1)
    kubectl port-forward -n "$NAMESPACE" "pod/$POD" 8100:8000 &
    sleep 10
    # Verify health
    curl -s --max-time 10 "$ENDPOINT/health" || { echo "Health check failed"; exit 1; }
    echo "=== vLLM ready ==="
}

timestamp() {
    date +"%Y-%m-%d %H:%M:%S"
}

echo "=== SemBlend Full Benchmark Suite ==="
echo "=== Started: $(timestamp) ==="
echo ""

# 1. Break-even analysis (Gap 7) — n=500 per condition × 2 lengths = 3000 requests
echo "=== [1/6] Break-even Analysis (Gap 7) ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.breakeven_analysis_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-samples 500 \
    --token-lengths 4096,8192 \
    --output "$RESULTS_DIR/breakeven_n500.json" 2>&1 | tee "$RESULTS_DIR/breakeven_n500.log"
echo "    Finished: $(timestamp)"
echo ""

# 2. Realistic cross-instruction (Gap 11) — 250 articles × 4 variants = 1000 pairs
echo "=== [2/6] Realistic Cross-Instruction (Gap 11) ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.realistic_cross_instruction_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-articles 250 \
    --token-length 8192 \
    --output "$RESULTS_DIR/realistic_cross_instr_n250.json" 2>&1 | tee "$RESULTS_DIR/realistic_cross_instr_n250.log"
echo "    Finished: $(timestamp)"
echo ""

# 3. Multi-turn dialogue (Gap 6) — 200 conversations × 6 turns = 1200 measurements
echo "=== [3/6] Multi-turn Dialogue (Gap 6) ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.multiturn_dialogue_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-conversations 200 \
    --max-turns 6 \
    --token-length 4096 \
    --output "$RESULTS_DIR/multiturn_n200.json" 2>&1 | tee "$RESULTS_DIR/multiturn_n200.log"
echo "    Finished: $(timestamp)"
echo ""

# 4. WildChat large-scale (Gap 6) — 200 per bucket × 5 buckets = 1000 pairs
echo "=== [4/6] WildChat Large-Scale (Gap 6) ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.wildchat_large_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-per-bucket 200 \
    --min-chars 6000 \
    --output "$RESULTS_DIR/wildchat_large_n200.json" 2>&1 | tee "$RESULTS_DIR/wildchat_large_n200.log"
echo "    Finished: $(timestamp)"
echo ""

# 5. Code generation (Gap 8) — n=1000 problems
echo "=== [5/6] Code Generation (Gap 8) ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.code_gen_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-problems 1000 \
    --token-length 4096 \
    --output "$RESULTS_DIR/code_gen_n1000.json" 2>&1 | tee "$RESULTS_DIR/code_gen_n1000.log"
echo "    Finished: $(timestamp)"
echo ""

# 6. Chunk size E2E ablation (Gap 5) — 4 chunk sizes × 250 articles × 4 variants
#    This benchmark handles its own deploys (changes ConfigMap + env var per chunk size)
echo "=== [6/6] Chunk Size E2E Ablation (Gap 5) ==="
echo "    Started: $(timestamp)"
PYTHONPATH=. uv run python -m benchmarks.e2e.chunk_size_e2e_ablation \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-articles 250 \
    --token-length 8192 \
    --chunk-sizes 64,128,256,512 \
    --output "$RESULTS_DIR/chunk_size_e2e_ablation.json" 2>&1 | tee "$RESULTS_DIR/chunk_size_e2e_ablation.log"
echo "    Finished: $(timestamp)"
echo ""

echo "=== All benchmarks complete: $(timestamp) ==="
echo "Results in $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json
