#!/usr/bin/env bash
set -euo pipefail

# SemBlend Full Benchmark Suite
# Covers all 6 requirements for paper-quality results:
#  1. Real datasets (CNN/DailyMail, XSum)
#  2. Semantically similar clusters from real data
#  3. Scale (100, 1K, 10K donors)
#  4. Quality (ROUGE-L, exact match, perplexity ratio)
#  5. Production lengths (1K, 2K, 4K, 8K, 16K tokens)
#  6. RoPE correction ablation (REORDER with/without)
#
# Prerequisites:
#   - vLLM pod running with SemBlend enabled
#   - kubectl port-forward to vLLM on localhost:8001
#   - Clusters built: python -m benchmarks.e2e.build_clusters
#
# Usage:
#   ./benchmarks/e2e/run_full_semblend_bench.sh [endpoint] [model]

ENDPOINT="${1:-http://localhost:8001}"
MODEL="${2:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
CLUSTERS="benchmarks/data/semblend_clusters.json"
RESULTS_DIR="results/semblend-full-$(date +%Y%m%d_%H%M%S)"
HARDWARE="A10G"

echo "=============================================="
echo "SemBlend Full Benchmark Suite"
echo "=============================================="
echo "Endpoint: $ENDPOINT"
echo "Model:    $MODEL"
echo "Clusters: $CLUSTERS"
echo "Results:  $RESULTS_DIR"
echo "Hardware: $HARDWARE"
echo ""

# Verify prerequisites
if [ ! -f "$CLUSTERS" ]; then
    echo "Clusters file not found. Building..."
    python -m benchmarks.e2e.build_clusters
fi

# Verify endpoint is reachable
echo "Checking endpoint..."
if ! curl -s -f "$ENDPOINT/health" > /dev/null 2>&1; then
    echo "WARNING: $ENDPOINT/health not reachable."
    echo "Ensure vLLM is running and port-forwarded."
    echo "Continue anyway? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        exit 1
    fi
fi

mkdir -p "$RESULTS_DIR"

# ---- Phase 1: Scale Benchmark (requirements 1-5) ----
echo ""
echo "=============================================="
echo "Phase 1: Scale Benchmark (100, 1K donors)"
echo "=============================================="
python -m benchmarks.e2e.semblend_scale_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --clusters "$CLUSTERS" \
    --donor-scales "100,1000" \
    --num-runs 10 \
    --output-dir "$RESULTS_DIR/scale" \
    --hardware "$HARDWARE" \
    2>&1 | tee "$RESULTS_DIR/scale_bench.log"

# ---- Phase 2: RoPE Ablation (requirement 6) ----
echo ""
echo "=============================================="
echo "Phase 2: RoPE Correction Ablation"
echo "=============================================="
python -m benchmarks.e2e.rope_ablation_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --clusters "$CLUSTERS" \
    --num-runs 10 \
    --output-dir "$RESULTS_DIR/rope_ablation" \
    2>&1 | tee "$RESULTS_DIR/rope_ablation.log"

# ---- Phase 3: MS MARCO Real-Data (additional real dataset) ----
echo ""
echo "=============================================="
echo "Phase 3: MS MARCO RAG Benchmark"
echo "=============================================="
python -m benchmarks.e2e.semblend_real_bench \
    --endpoint "$ENDPOINT" \
    --model "$MODEL" \
    --runs 7 \
    --chunks 8 16 24 32 \
    --output "$RESULTS_DIR/msmarco" \
    2>&1 | tee "$RESULTS_DIR/msmarco_bench.log"

echo ""
echo "=============================================="
echo "All benchmarks complete!"
echo "Results saved to: $RESULTS_DIR"
echo "=============================================="
echo ""
echo "Files:"
ls -la "$RESULTS_DIR"/ "$RESULTS_DIR"/scale/ "$RESULTS_DIR"/rope_ablation/ 2>/dev/null
