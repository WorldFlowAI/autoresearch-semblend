#!/bin/bash
# Build Docker image with working synapse_kv_connector, push to ECR, deploy to autoresearch namespace.
# Uses a temp build context with symlinks to avoid modifying the synapse repo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SYNAPSE_DIR="$HOME/dev/worldflowai/ONECONTEXT/synapse"
NAMESPACE="autoresearch"
ECR_REPO="198333343348.dkr.ecr.us-east-1.amazonaws.com/synapse-vllm"
REGION="us-east-1"

# Tag: semblend-autoresearch-<short commit hash>
COMMIT=$(cd "$REPO_DIR" && git rev-parse --short HEAD)
TAG="semblend-autoresearch-${COMMIT}"

echo "=== Deploy: tag=$TAG ==="

# Step 1: Ensure GPU is available
echo "--- Step 1: Ensure GPU ---"
bash "$SCRIPT_DIR/ensure_gpu.sh"

# Step 2: Build Docker image using temp build context
echo "--- Step 2: Docker build ---"
BUILD_CTX="$REPO_DIR/.build_context"
rm -rf "$BUILD_CTX"
mkdir -p "$BUILD_CTX"

# Mirror synapse repo structure via symlinks (except synapse_kv_connector)
for item in "$SYNAPSE_DIR"/*; do
  basename=$(basename "$item")
  ln -sf "$item" "$BUILD_CTX/$basename"
done

# Symlink services directory, but override synapse_kv_connector
mkdir -p "$BUILD_CTX/services/vllm"
for item in "$SYNAPSE_DIR/services/vllm"/*; do
  basename=$(basename "$item")
  if [ "$basename" != "synapse_kv_connector" ]; then
    ln -sf "$item" "$BUILD_CTX/services/vllm/$basename"
  fi
done

# Copy working synapse_kv_connector (not symlink — this is the modified version)
cp -r "$REPO_DIR/synapse_kv_connector" "$BUILD_CTX/services/vllm/synapse_kv_connector"

# Remove top-level services symlink (we created real dir)
rm -f "$BUILD_CTX/services"
mkdir -p "$BUILD_CTX/services"
for item in "$SYNAPSE_DIR/services"/*; do
  svc_name=$(basename "$item")
  if [ "$svc_name" != "vllm" ]; then
    ln -sf "$item" "$BUILD_CTX/services/$svc_name"
  fi
done
# Move our vllm dir into services
mv "$BUILD_CTX/services/vllm" "$BUILD_CTX/services/vllm_tmp" 2>/dev/null || true
rm -rf "$BUILD_CTX/services/vllm"
# Reconstruct properly
mkdir -p "$BUILD_CTX/services/vllm"
for item in "$SYNAPSE_DIR/services/vllm"/*; do
  basename=$(basename "$item")
  if [ "$basename" != "synapse_kv_connector" ]; then
    ln -sf "$item" "$BUILD_CTX/services/vllm/$basename"
  fi
done
cp -r "$REPO_DIR/synapse_kv_connector" "$BUILD_CTX/services/vllm/synapse_kv_connector"

# Build from the vLLM Dockerfile
docker build \
  -t "$ECR_REPO:$TAG" \
  -f "$SYNAPSE_DIR/services/vllm/Dockerfile" \
  "$BUILD_CTX" \
  --build-arg BUILDKIT_INLINE_CACHE=1

# Clean up build context
rm -rf "$BUILD_CTX"

# Step 3: Push to ECR
echo "--- Step 3: Push to ECR ---"
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REPO"
docker push "$ECR_REPO:$TAG"

# Also tag as latest for the autoresearch namespace
docker tag "$ECR_REPO:$TAG" "$ECR_REPO:semblend-autoresearch-latest"
docker push "$ECR_REPO:semblend-autoresearch-latest"

# Step 4: Update Helm values with new tag and upgrade
echo "--- Step 4: Helm upgrade ---"
sed -i.bak "s|tag: semblend-autoresearch-.*|tag: $TAG|" "$REPO_DIR/infra/values-autoresearch.yaml"
rm -f "$REPO_DIR/infra/values-autoresearch.yaml.bak"

helm upgrade --install autoresearch \
  "$SYNAPSE_DIR/helm/synapse" \
  -n "$NAMESPACE" \
  -f "$SYNAPSE_DIR/helm/synapse/values/staging.yaml" \
  -f "$REPO_DIR/infra/values-autoresearch.yaml"

# Step 5: Wait for vLLM pod readiness
echo "--- Step 5: Wait for vLLM pod ---"
kubectl rollout status deployment/autoresearch-vllm -n "$NAMESPACE" --timeout=600s 2>/dev/null || \
  kubectl wait --for=condition=Ready pod -l "app.kubernetes.io/component=vllm" -n "$NAMESPACE" --timeout=600s

# Step 6: Port-forward
echo "--- Step 6: Port-forward ---"
# Kill any existing port-forward for autoresearch
pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true

VLLM_SVC=$(kubectl get svc -n "$NAMESPACE" -l "app.kubernetes.io/component=vllm" -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "autoresearch-vllm")
kubectl port-forward "svc/$VLLM_SVC" 8100:8000 -n "$NAMESPACE" &
PORT_FWD_PID=$!
echo "Port-forward PID: $PORT_FWD_PID (localhost:8100 -> vLLM in $NAMESPACE)"

echo "=== Deploy complete: $TAG ==="
echo "vLLM available at http://localhost:8100/v1"
