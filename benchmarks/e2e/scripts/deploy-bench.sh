#!/usr/bin/env bash
# Deploy/teardown isolated benchmark namespace on EKS.
#
# Usage:
#   ./deploy-bench.sh up    # Scale GPU nodes + deploy benchmark stack
#   ./deploy-bench.sh down  # Tear down stack + scale down GPU nodes
set -euo pipefail

CLUSTER="${CLUSTER:-synapse-staging}"
NAMESPACE="synapse-bench"
GPU_NODEGROUP="${GPU_NODEGROUP:-gpu-nodes}"
GPU_NODES="${GPU_NODES:-3}"
HELM_RELEASE="synapse-bench"
HELM_CHART="./helm/synapse"
VALUES_FILE="helm/synapse/values/benchmark.yaml"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

cd "${REPO_ROOT}"

up() {
    echo "=== Scaling GPU nodes to ${GPU_NODES} ==="
    eksctl scale nodegroup \
        --cluster "${CLUSTER}" \
        --name "${GPU_NODEGROUP}" \
        --nodes "${GPU_NODES}" \
        --nodes-min 0 \
        --nodes-max "${GPU_NODES}" \
        || echo "Warning: GPU scale failed (may already be scaled)"

    echo "=== Waiting for GPU nodes ==="
    kubectl wait --for=condition=Ready nodes \
        -l nvidia.com/gpu.present=true \
        --timeout=300s 2>/dev/null || true

    echo "=== Creating namespace ==="
    kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml \
        | kubectl apply -f -

    echo "=== Deploying benchmark stack ==="
    helm upgrade --install "${HELM_RELEASE}" "${HELM_CHART}" \
        -n "${NAMESPACE}" \
        -f "${VALUES_FILE}" \
        --wait \
        --timeout 20m

    echo "=== Waiting for vLLM pod ==="
    kubectl wait -n "${NAMESPACE}" \
        --for=condition=Ready pod \
        -l app.kubernetes.io/component=vllm \
        --timeout=600s 2>/dev/null || echo "Warning: vLLM pod not ready yet"

    echo "=== Benchmark stack deployed ==="
    kubectl get pods -n "${NAMESPACE}"

    echo ""
    echo "Port-forward gateway for local benchmark runs:"
    echo "  kubectl port-forward -n ${NAMESPACE} svc/synapse-gateway 8080:8080"
}

down() {
    echo "=== Uninstalling benchmark stack ==="
    helm uninstall "${HELM_RELEASE}" -n "${NAMESPACE}" 2>/dev/null \
        || echo "Release not found"

    echo "=== Deleting namespace ==="
    kubectl delete namespace "${NAMESPACE}" --ignore-not-found

    echo "=== Scaling down GPU nodes ==="
    eksctl scale nodegroup \
        --cluster "${CLUSTER}" \
        --name "${GPU_NODEGROUP}" \
        --nodes 0 \
        --nodes-min 0 \
        || echo "Warning: GPU scale-down failed"

    echo "=== Benchmark stack torn down ==="
}

case "${1:-}" in
    up)   up ;;
    down) down ;;
    *)
        echo "Usage: $0 {up|down}"
        exit 1
        ;;
esac
