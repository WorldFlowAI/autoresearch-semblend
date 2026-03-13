#!/bin/bash
# Ensure at least one A10G GPU node is running and Ready.
# Provisions one if needed. ALWAYS exits 0 with a running GPU node, or exits 1 after timeout.
set -euo pipefail

CLUSTER="synapse-staging"
NODEGROUP="gpu-nodes-g5"
REGION="us-east-1"
TIMEOUT=300

echo "=== Checking GPU nodes ==="

# Count Ready A10G nodes
GPU_READY=$(kubectl get nodes -l gpu-type=a10g --no-headers 2>/dev/null | grep -c "Ready" || true)

if [ "$GPU_READY" -gt 0 ]; then
  echo "GPU node(s) already running: $GPU_READY Ready"
  kubectl get nodes -l gpu-type=a10g --no-headers
  exit 0
fi

echo "No GPU nodes running. Scaling up $NODEGROUP..."

# Check if staging pods are running on GPU nodes — if so, request 2 nodes
STAGING_GPU_PODS=$(kubectl get pods -n synapse-staging -o wide --no-headers 2>/dev/null | grep -c "a10g" || true)
DESIRED_NODES=1
if [ "$STAGING_GPU_PODS" -gt 0 ]; then
  echo "Staging has GPU pods — requesting 2 nodes"
  DESIRED_NODES=2
fi

eksctl scale nodegroup \
  --cluster "$CLUSTER" \
  --name "$NODEGROUP" \
  --nodes "$DESIRED_NODES" \
  --nodes-min 0 \
  --nodes-max 4 \
  --region "$REGION"

echo "Waiting for GPU node(s) to be Ready (timeout: ${TIMEOUT}s)..."
kubectl wait --for=condition=Ready node \
  -l "eks.amazonaws.com/nodegroup=$NODEGROUP" \
  --timeout="${TIMEOUT}s"

echo "=== GPU node(s) ready ==="
kubectl get nodes -l gpu-type=a10g --no-headers
