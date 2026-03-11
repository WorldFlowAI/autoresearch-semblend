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

# Ensure we're targeting the right cluster
STAGING_CONTEXT="arn:aws:eks:us-east-1:198333343348:cluster/synapse-staging"
kubectl config use-context "$STAGING_CONTEXT" || {
  echo "ERROR: Cannot switch to synapse-staging context"; exit 1
}

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
cp -rL "$REPO_DIR/synapse_kv_connector" "$BUILD_CTX/services/vllm/synapse_kv_connector"

# Remove top-level services symlink (we created real dir)
rm -f "$BUILD_CTX/services"
mkdir -p "$BUILD_CTX/services"

# Build services/vllm with real files for COPY'd items
# Note: Several files in synapse are circular symlinks — extract from git instead
mkdir -p "$BUILD_CTX/services/vllm"
cd "$SYNAPSE_DIR"
git show HEAD:services/vllm/export_onnx.py > "$BUILD_CTX/services/vllm/export_onnx.py"
git show HEAD:services/vllm/lmcache_config.yaml > "$BUILD_CTX/services/vllm/lmcache_config.yaml"
cd "$REPO_DIR"

# Copy our modified synapse_kv_connector (not symlink)
cp -rL "$REPO_DIR/synapse_kv_connector" "$BUILD_CTX/services/vllm/synapse_kv_connector"

# Build services/lmcache with real copy from git
mkdir -p "$BUILD_CTX/services/lmcache"
cd "$SYNAPSE_DIR"
git show HEAD:services/lmcache/synapse_lmcache/__init__.py > /dev/null 2>&1 && \
  git archive HEAD services/lmcache/ | tar -x -C "$BUILD_CTX/" 2>/dev/null || \
  cp -rL "$SYNAPSE_DIR/services/lmcache" "$BUILD_CTX/services/" 2>/dev/null || true
cd "$REPO_DIR"

# Extract Dockerfile from git (services/vllm/Dockerfile is a broken symlink on disk)
DOCKERFILE_PATH="$BUILD_CTX/Dockerfile.vllm"
cd "$SYNAPSE_DIR" && git show HEAD:services/vllm/Dockerfile > "$DOCKERFILE_PATH"
cd "$REPO_DIR"

# Build from the vLLM Dockerfile
docker build \
  --platform linux/amd64 \
  -t "$ECR_REPO:$TAG" \
  -f "$DOCKERFILE_PATH" \
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
sed -i.bak "s|tag: .*|tag: $TAG|" "$REPO_DIR/infra/values-autoresearch.yaml"
rm -f "$REPO_DIR/infra/values-autoresearch.yaml.bak"

kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -

helm upgrade --install autoresearch \
  /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse \
  -n "$NAMESPACE" \
  -f /Users/zach/dev/worldflowai/ONECONTEXT/synapse/helm/synapse/values/staging.yaml \
  -f "$REPO_DIR/infra/values-autoresearch.yaml"

# Step 5: Wait for vLLM pod readiness
echo "--- Step 5: Wait for vLLM pod ---"
kubectl rollout status deployment/autoresearch-synapse-vllm -n "$NAMESPACE" --timeout=600s 2>/dev/null || \
  kubectl wait --for=condition=Ready pod -l "app=vllm" -n "$NAMESPACE" --timeout=600s

# Step 6: Port-forward
echo "--- Step 6: Port-forward ---"
# Kill any existing port-forward for autoresearch
pkill -f "kubectl port-forward.*-n $NAMESPACE" 2>/dev/null || true

POD=$(kubectl get pods -n "$NAMESPACE" -l app=vllm --no-headers | awk '{print $1}' | head -1)
kubectl port-forward -n "$NAMESPACE" "pod/$POD" 8100:8000 &
PORT_FWD_PID=$!
echo "Port-forward PID: $PORT_FWD_PID (localhost:8100 -> vLLM in $NAMESPACE)"

echo "=== Deploy complete: $TAG ==="
echo "vLLM available at http://localhost:8100/v1"
