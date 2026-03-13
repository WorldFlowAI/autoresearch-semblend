#!/usr/bin/env bash
# Keeps kubectl port-forward alive by restarting on failure.
# Usage: ./port_forward_keepalive.sh [namespace] [service] [local_port] [remote_port]
set -euo pipefail

NS="${1:-synapse-staging}"
SVC="${2:-svc/synapse-staging-vllm}"
LOCAL="${3:-8001}"
REMOTE="${4:-8000}"

echo "Port-forward keepalive: $NS/$SVC $LOCAL:$REMOTE"
echo "Press Ctrl+C to stop"

while true; do
    echo "[$(date)] Starting port-forward..."
    kubectl port-forward -n "$NS" "$SVC" "$LOCAL:$REMOTE" 2>&1 || true
    echo "[$(date)] Port-forward died, restarting in 2s..."
    sleep 2
done
