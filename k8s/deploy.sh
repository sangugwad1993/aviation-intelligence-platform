#!/bin/bash
# ============================================================================
# Aviation Intelligence Platform — Kubernetes Deployment Script
# Usage:
#   ./k8s/deploy.sh                     → Deploy to current K8s context
#   ./k8s/deploy.sh --delete            → Remove all resources
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [[ "${1:-}" == "--delete" ]]; then
    echo "🗑  Removing Aviation Intelligence Platform from Kubernetes..."
    kubectl delete -f "$SCRIPT_DIR/ingress.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/service.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/deployment.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/secret.yaml" --ignore-not-found
    kubectl delete -f "$SCRIPT_DIR/namespace.yaml" --ignore-not-found
    echo "✅ All resources removed."
    exit 0
fi

echo "🚀 Deploying Aviation Intelligence Platform to Kubernetes..."

echo "  [1/5] Creating namespace..."
kubectl apply -f "$SCRIPT_DIR/namespace.yaml"

echo "  [2/5] Creating secrets..."
kubectl apply -f "$SCRIPT_DIR/secret.yaml"

echo "  [3/5] Creating deployment (2 replicas)..."
kubectl apply -f "$SCRIPT_DIR/deployment.yaml"

echo "  [4/5] Creating service..."
kubectl apply -f "$SCRIPT_DIR/service.yaml"

echo "  [5/5] Creating ingress..."
kubectl apply -f "$SCRIPT_DIR/ingress.yaml"

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📋 Status:"
kubectl -n aviation-intelligence get pods,svc,ingress
echo ""
echo "🔗 Access:"
echo "  • Internal:  kubectl -n aviation-intelligence port-forward svc/aviation-dashboard-svc 8501:80"
echo "  • External:  https://aviation.example.com (update host in ingress.yaml)"
echo ""
echo "🔑 Default credentials:"
echo "  • admin / aviation2024"
echo "  • viewer / readonly2024"
echo "  ⚠️  Change these in k8s/secret.yaml before production!"
