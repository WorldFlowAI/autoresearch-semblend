#!/bin/bash
# Run all remaining live benchmarks after code_gen_bench completes.
# This script: builds, deploys, then runs 5 benchmarks sequentially.
# Expected total runtime: ~8-10 hours.
set -euo pipefail

ENDPOINT="http://localhost:8100"
MODEL="Qwen/Qwen2.5-7B-Instruct-AWQ"
NAMESPACE="autoresearch"
RESULTS_DIR="results"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$RESULTS_DIR"

timestamp() { date +"%Y-%m-%d %H:%M:%S"; }

restart_vllm() {
    echo "=== Restarting vLLM for clean state ==="
    kubectl rollout restart deployment/autoresearch-synapse-vllm -n "$NAMESPACE"
    kubectl rollout status deployment/autoresearch-synapse-vllm -n "$NAMESPACE" --timeout=300s
    pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true
    sleep 5
    POD=$(kubectl get pods -n "$NAMESPACE" -l app=vllm --no-headers | awk '{print $1}' | head -1)
    kubectl port-forward -n "$NAMESPACE" "pod/$POD" 8100:8000 &
    sleep 15
    # Health check with retries
    for i in 1 2 3 4 5; do
        curl -s --max-time 10 "$ENDPOINT/health" && break
        echo "  Health check attempt $i failed, waiting..."
        sleep 10
    done
    echo "=== vLLM ready ==="
}

echo "=========================================="
echo "=== SemBlend Remaining Benchmarks      ==="
echo "=== Started: $(timestamp)              ==="
echo "=========================================="
echo ""

# Step 1: Build and deploy new Docker image
echo "=== Step 1: Build & Deploy ==="
echo "    Started: $(timestamp)"
bash "$SCRIPT_DIR/infra/deploy.sh"
echo "    Deploy complete: $(timestamp)"
echo ""

# Wait for vLLM to be ready
sleep 10
POD=$(kubectl get pods -n "$NAMESPACE" -l app=vllm --no-headers | awk '{print $1}' | head -1)
pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true
sleep 3
kubectl port-forward -n "$NAMESPACE" "pod/$POD" 8100:8000 &
sleep 15
curl -s --max-time 30 "$ENDPOINT/health" || { echo "FATAL: Health check failed after deploy"; exit 1; }
echo "vLLM healthy at $ENDPOINT"

# Step 2: Break-even analysis
echo ""
echo "=== [1/5] Break-even Analysis ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.breakeven_analysis_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-samples 500 \
    --token-lengths 4096,8192 \
    --output "$RESULTS_DIR/breakeven_n500.json" 2>&1 | tee "$RESULTS_DIR/breakeven_n500.log"
echo "    Finished: $(timestamp)"

# Step 3: Realistic cross-instruction
echo ""
echo "=== [2/5] Realistic Cross-Instruction ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.realistic_cross_instruction_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-articles 250 \
    --token-length 8192 \
    --output "$RESULTS_DIR/realistic_cross_instr_n250.json" 2>&1 | tee "$RESULTS_DIR/realistic_cross_instr_n250.log"
echo "    Finished: $(timestamp)"

# Step 4: Multi-turn dialogue
echo ""
echo "=== [3/5] Multi-turn Dialogue ==="
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

# Step 5: WildChat large-scale
echo ""
echo "=== [4/5] WildChat Large-Scale ==="
echo "    Started: $(timestamp)"
restart_vllm
PYTHONPATH=. uv run python -m benchmarks.e2e.wildchat_large_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --n-per-bucket 200 \
    --min-chars 6000 \
    --output "$RESULTS_DIR/wildchat_large_n200.json" 2>&1 | tee "$RESULTS_DIR/wildchat_large_n200.log"
echo "    Finished: $(timestamp)"

# Step 6: Chunk size E2E ablation (handles its own deploys)
echo ""
echo "=== [5/5] Chunk Size E2E Ablation ==="
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
echo "=========================================="
echo "=== All benchmarks complete             ==="
echo "=== Finished: $(timestamp)              ==="
echo "=========================================="
echo ""
echo "Results:"
ls -la "$RESULTS_DIR/"*.json "$RESULTS_DIR/"*.log 2>/dev/null
