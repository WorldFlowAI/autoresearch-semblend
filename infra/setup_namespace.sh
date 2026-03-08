#!/bin/bash
# One-time: create autoresearch namespace and install Helm chart
# Safe to re-run (idempotent)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
SYNAPSE_DIR="$HOME/dev/worldflowai/ONECONTEXT/synapse"
NAMESPACE="autoresearch"

echo "=== Setting up autoresearch namespace ==="

# Create namespace (idempotent)
kubectl create namespace "$NAMESPACE" --dry-run=client -o yaml | kubectl apply -f -
echo "Namespace '$NAMESPACE' ready"

# Install/upgrade Helm chart
echo "Installing Helm chart..."
helm upgrade --install autoresearch \
  "$SYNAPSE_DIR/helm/synapse" \
  -n "$NAMESPACE" \
  -f "$SYNAPSE_DIR/helm/synapse/values/staging.yaml" \
  -f "$REPO_DIR/infra/values-autoresearch.yaml" \
  --wait --timeout 10m

echo "=== Setup complete ==="
kubectl get pods -n "$NAMESPACE"
