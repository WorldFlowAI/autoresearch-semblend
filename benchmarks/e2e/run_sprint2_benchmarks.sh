#!/usr/bin/env bash
# ============================================================================
# Sprint 2: Publication-Quality SemBlend Benchmark Suite
# ============================================================================
# Runs all benchmarks needed for the SemBlend paper evaluation section:
#   1. Build real-dataset clusters (CNN/DailyMail, XSum, MultiNews, WikiHow)
#   2. TTFT scaling per dataset
#   3. Quality preservation per dataset
#   4. Prefix cache baseline (gap analysis)
#   5. Threshold ablation (post-hoc)
#   6. Embedding ablation (MiniLM vs Jaccard)
#
# Prerequisites:
#   - vLLM running with SemBlend enabled (port-forward to localhost:8001)
#   - Python deps: datasets, transformers, rouge-score, requests
#
# Usage:
#   ./benchmarks/e2e/run_sprint2_benchmarks.sh [--skip-clusters] [--skip-ttft]
#   ./benchmarks/e2e/run_sprint2_benchmarks.sh --datasets cnn_dailymail xsum
# ============================================================================

set -euo pipefail

VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8001}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen2.5-7B-Instruct-AWQ}"
RESULTS_DIR="results/sprint2"
DATA_DIR="benchmarks/data"
TIMESTAMP=$(date +%Y%m%d-%H%M%S)
DATASETS="${DATASETS:-cnn_dailymail xsum multinews wikihow}"
RUNS_PER_LENGTH="${RUNS_PER_LENGTH:-10}"
SKIP_CLUSTERS=false
SKIP_TTFT=false
SKIP_QUALITY=false
SKIP_PREFIX=false

# Parse args
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-clusters) SKIP_CLUSTERS=true; shift ;;
        --skip-ttft) SKIP_TTFT=true; shift ;;
        --skip-quality) SKIP_QUALITY=true; shift ;;
        --skip-prefix) SKIP_PREFIX=true; shift ;;
        --datasets) shift; DATASETS="$*"; break ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

mkdir -p "$RESULTS_DIR"

echo "============================================="
echo "SemBlend Sprint 2 Benchmark Suite"
echo "============================================="
echo "Endpoint:   $VLLM_ENDPOINT"
echo "Model:      $VLLM_MODEL"
echo "Datasets:   $DATASETS"
echo "Results:    $RESULTS_DIR"
echo "Timestamp:  $TIMESTAMP"
echo "Runs/len:   $RUNS_PER_LENGTH"
echo "============================================="

# Verify vLLM is responding
echo ""
echo "[0/6] Verifying vLLM endpoint..."
if ! curl -sf "$VLLM_ENDPOINT/v1/models" > /dev/null 2>&1; then
    echo "ERROR: vLLM not responding at $VLLM_ENDPOINT"
    echo "Start port-forward: kubectl port-forward -n synapse-staging svc/synapse-staging-vllm 8001:8000"
    exit 1
fi
echo "  vLLM OK"

# ============================================================================
# Phase 1: Build Real-Dataset Clusters
# ============================================================================
if [ "$SKIP_CLUSTERS" = false ]; then
    echo ""
    echo "[1/6] Building real-dataset clusters with LLM rewriting..."
    echo "  This may take 20-60 minutes (LLM-based paraphrase per cluster)"
    python3 -m benchmarks.e2e.build_clusters \
        --datasets $DATASETS \
        --vllm-endpoint "$VLLM_ENDPOINT" \
        --model "$VLLM_MODEL" \
        2>&1 | tee "$RESULTS_DIR/cluster_build_${TIMESTAMP}.log"
    echo "  Clusters built successfully"
else
    echo ""
    echo "[1/6] Skipping cluster build (--skip-clusters)"
fi

# ============================================================================
# Phase 2: TTFT Scaling per Dataset
# ============================================================================
if [ "$SKIP_TTFT" = false ]; then
    echo ""
    echo "[2/6] Running TTFT scaling benchmarks per dataset..."
    for DS in $DATASETS; do
        CLUSTER_FILE="${DATA_DIR}/${DS}_clusters.json"
        if [ ! -f "$CLUSTER_FILE" ]; then
            echo "  WARNING: $CLUSTER_FILE not found, skipping $DS"
            continue
        fi
        echo "  Running TTFT for $DS..."
        python3 -m benchmarks.e2e.semblend_ttft_scaling \
            --endpoint "$VLLM_ENDPOINT" \
            --runs "$RUNS_PER_LENGTH" \
            --clusters-file "$CLUSTER_FILE" \
            --dataset-name "$DS" \
            --output "$RESULTS_DIR/${DS}_ttft_${TIMESTAMP}.json" \
            2>&1 | tee -a "$RESULTS_DIR/ttft_${TIMESTAMP}.log"
        echo "  $DS TTFT complete"
    done
else
    echo ""
    echo "[2/6] Skipping TTFT benchmarks (--skip-ttft)"
fi

# ============================================================================
# Phase 3: Quality Preservation per Dataset
# ============================================================================
if [ "$SKIP_QUALITY" = false ]; then
    echo ""
    echo "[3/6] Running quality benchmarks per dataset..."
    for DS in $DATASETS; do
        CLUSTER_FILE="${DATA_DIR}/${DS}_clusters.json"
        if [ ! -f "$CLUSTER_FILE" ]; then
            echo "  WARNING: $CLUSTER_FILE not found, skipping $DS"
            continue
        fi
        echo "  Running quality for $DS..."
        python3 -m benchmarks.e2e.semblend_quality_bench \
            --endpoint "$VLLM_ENDPOINT" \
            --runs 5 \
            --max-tokens 256 \
            --clusters-file "$CLUSTER_FILE" \
            --output "$RESULTS_DIR/${DS}_quality_${TIMESTAMP}.json" \
            2>&1 | tee -a "$RESULTS_DIR/quality_${TIMESTAMP}.log"
        echo "  $DS quality complete"
    done
else
    echo ""
    echo "[3/6] Skipping quality benchmarks (--skip-quality)"
fi

# ============================================================================
# Phase 4: Prefix Cache Baseline
# ============================================================================
if [ "$SKIP_PREFIX" = false ]; then
    echo ""
    echo "[4/6] Running prefix cache baseline..."
    # Use first dataset's clusters for prefix cache comparison
    FIRST_DS=$(echo $DATASETS | awk '{print $1}')
    CLUSTER_FILE="${DATA_DIR}/${FIRST_DS}_clusters.json"
    if [ -f "$CLUSTER_FILE" ]; then
        python3 -m benchmarks.e2e.prefix_cache_baseline_bench \
            --endpoint "$VLLM_ENDPOINT" \
            --runs 5 \
            --clusters-file "$CLUSTER_FILE" \
            --output "$RESULTS_DIR/prefix_cache_baseline_${TIMESTAMP}.json" \
            2>&1 | tee "$RESULTS_DIR/prefix_cache_${TIMESTAMP}.log"
    else
        echo "  WARNING: $CLUSTER_FILE not found, using built-in scenarios"
        python3 -m benchmarks.e2e.prefix_cache_baseline_bench \
            --endpoint "$VLLM_ENDPOINT" \
            --runs 5 \
            --output "$RESULTS_DIR/prefix_cache_baseline_${TIMESTAMP}.json" \
            2>&1 | tee "$RESULTS_DIR/prefix_cache_${TIMESTAMP}.log"
    fi
    echo "  Prefix cache baseline complete"
else
    echo ""
    echo "[4/6] Skipping prefix cache baseline (--skip-prefix)"
fi

# ============================================================================
# Phase 5: Threshold Ablation (Post-Hoc)
# ============================================================================
echo ""
echo "[5/6] Running threshold ablation..."
# Collect all TTFT result files for threshold analysis
TTFT_FILES=""
for DS in $DATASETS; do
    F="$RESULTS_DIR/${DS}_ttft_${TIMESTAMP}.json"
    if [ -f "$F" ]; then
        TTFT_FILES="$TTFT_FILES $F"
    fi
done
if [ -n "$TTFT_FILES" ]; then
    python3 -m benchmarks.e2e.threshold_ablation_bench \
        --results-files $TTFT_FILES \
        --output "$RESULTS_DIR/threshold_ablation_${TIMESTAMP}.json" \
        2>&1 | tee "$RESULTS_DIR/threshold_ablation_${TIMESTAMP}.log"
    echo "  Threshold ablation complete"
else
    echo "  No TTFT results found, skipping threshold ablation"
fi

# ============================================================================
# Phase 6: Embedding Ablation (MiniLM vs Jaccard)
# ============================================================================
echo ""
echo "[6/6] Running embedding ablation..."
FIRST_DS=$(echo $DATASETS | awk '{print $1}')
CLUSTER_FILE="${DATA_DIR}/${FIRST_DS}_clusters.json"
if [ -f "$CLUSTER_FILE" ]; then
    python3 -m benchmarks.e2e.embedding_ablation_bench \
        --endpoint "$VLLM_ENDPOINT" \
        --runs 5 \
        --clusters-file "$CLUSTER_FILE" \
        --output "$RESULTS_DIR/embedding_ablation_${TIMESTAMP}.json" \
        2>&1 | tee "$RESULTS_DIR/embedding_ablation_${TIMESTAMP}.log"
    echo "  Embedding ablation complete"
else
    echo "  No cluster file found, skipping embedding ablation"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "============================================="
echo "Sprint 2 Benchmark Suite Complete"
echo "============================================="
echo "Results saved to: $RESULTS_DIR/"
ls -la "$RESULTS_DIR/"*.json 2>/dev/null || echo "  (no JSON results found)"
echo ""
echo "Next: Run results integration to generate LaTeX tables"
echo "  python3 benchmarks/e2e/integrate_results.py $RESULTS_DIR/"
