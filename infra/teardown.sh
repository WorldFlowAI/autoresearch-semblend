#!/bin/bash
# Tear down autoresearch infrastructure. Optionally scale down GPU nodes.
set -euo pipefail

NAMESPACE="autoresearch"
CLUSTER="synapse-staging"
NODEGROUP="gpu-nodes-g5"
REGION="us-east-1"

echo "=== Teardown: autoresearch ==="

# Step 1: Kill port-forward processes
echo "--- Killing port-forwards ---"
pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true

# Step 2: Helm uninstall (keep history for debugging)
echo "--- Helm uninstall ---"
helm uninstall autoresearch -n "$NAMESPACE" --keep-history 2>/dev/null || true

# Step 3: Scale down GPU nodes (only if staging isn't using them)
STAGING_GPU_PODS=$(kubectl get pods -n synapse-staging -o wide --no-headers 2>/dev/null | grep -c "gpu" || true)

if [ "$STAGING_GPU_PODS" -eq 0 ]; then
  echo "--- Scaling GPU nodes to 0 (staging has no GPU pods) ---"
  eksctl scale nodegroup \
    --cluster "$CLUSTER" \
    --name "$NODEGROUP" \
    --nodes 0 \
    --nodes-min 0 \
    --nodes-max 4 \
    --region "$REGION"
  echo "GPU nodes scaling down"
else
  echo "--- Keeping GPU nodes (staging has $STAGING_GPU_PODS GPU pods) ---"
fi

echo "=== Teardown complete ==="
