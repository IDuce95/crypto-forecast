#!/bin/bash
# Kubernetes deployment script for crypto-forecasting project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="crypto-forecasting"
CONTEXT="${KUBE_CONTEXT:-$(kubectl config current-context)}"
MANIFESTS_DIR="$SCRIPT_DIR"

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}=== $1 ===${NC}"
}

# Check prerequisites
check_prerequisites() {
    print_header "Checking Prerequisites"
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    if ! command -v helm &> /dev/null; then
        print_warning "helm is not installed. Helm charts won't be available."
    fi
    
    # Check if kubectl can connect to cluster
    if ! kubectl cluster-info &> /dev/null; then
        print_error "Cannot connect to Kubernetes cluster. Please check your kubeconfig."
        exit 1
    fi
    
    print_status "Using context: $CONTEXT"
    print_status "Target namespace: $NAMESPACE"
}

# Deploy Kubernetes manifests
deploy_manifests() {
    print_header "Deploying Kubernetes Manifests"
    
    local manifests=(
        "00-namespace-config.yaml"
        "01-database-redis.yaml"
        "02-ml-backend.yaml"
        "03-frontend.yaml"
        "05-monitoring.yaml"
        "06-ingress-networking.yaml"
    )
    
    for manifest in "${manifests[@]}"; do
        local file_path="$MANIFESTS_DIR/$manifest"
        if [[ -f "$file_path" ]]; then
            print_status "Applying $manifest..."
            kubectl apply -f "$file_path" --context="$CONTEXT"
        else
            print_warning "Manifest $manifest not found, skipping..."
        fi
    done
}

# Deploy using Helm
deploy_helm() {
    print_header "Deploying with Helm"
    
    local helm_dir="$PROJECT_ROOT/helm/crypto-forecasting"
    
    if [[ ! -d "$helm_dir" ]]; then
        print_error "Helm chart directory not found: $helm_dir"
        return 1
    fi
    
    # Add dependencies
    print_status "Adding Helm repositories..."
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    # Install or upgrade
    if helm list -n "$NAMESPACE" | grep -q crypto-forecasting; then
        print_status "Upgrading Helm release..."
        helm upgrade crypto-forecasting "$helm_dir" \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --wait \
            --timeout 10m
    else
        print_status "Installing Helm release..."
        helm install crypto-forecasting "$helm_dir" \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --wait \
            --timeout 10m
    fi
}

# Wait for deployments to be ready
wait_for_deployments() {
    print_header "Waiting for Deployments"
    
    local deployments=("postgres" "redis" "backend" "frontend")
    
    for deployment in "${deployments[@]}"; do
        print_status "Waiting for deployment: $deployment"
        kubectl wait --for=condition=available \
            --timeout=300s \
            deployment/$deployment \
            -n "$NAMESPACE" \
            --context="$CONTEXT" || true
    done
}

# Check deployment status
check_status() {
    print_header "Deployment Status"
    
    print_status "Namespace status:"
    kubectl get namespace "$NAMESPACE" --context="$CONTEXT" || true
    
    print_status "Pods status:"
    kubectl get pods -n "$NAMESPACE" --context="$CONTEXT" || true
    
    print_status "Services status:"
    kubectl get services -n "$NAMESPACE" --context="$CONTEXT" || true
    
    print_status "Ingress status:"
    kubectl get ingress -n "$NAMESPACE" --context="$CONTEXT" || true
    
    print_status "PVC status:"
    kubectl get pvc -n "$NAMESPACE" --context="$CONTEXT" || true
}

# Get access URLs
get_access_info() {
    print_header "Access Information"
    
    # Try to get ingress host
    local ingress_host
    ingress_host=$(kubectl get ingress crypto-forecasting-ingress -n "$NAMESPACE" -o jsonpath='{.spec.rules[0].host}' --context="$CONTEXT" 2>/dev/null || echo "")
    
    if [[ -n "$ingress_host" ]]; then
        print_status "Application URLs (via Ingress):"
        echo "  Frontend: https://$ingress_host"
        echo "  Backend API: https://$ingress_host/api"
    fi
    
    # Get NodePort services
    print_status "NodePort access:"
    kubectl get services -n "$NAMESPACE" --context="$CONTEXT" -o wide | grep NodePort || true
    
    # Port forward commands
    print_status "Port forwarding commands:"
    echo "  kubectl port-forward -n $NAMESPACE svc/frontend 3000:3000"
    echo "  kubectl port-forward -n $NAMESPACE svc/backend 5000:5000"
    echo "  kubectl port-forward -n $NAMESPACE svc/grafana 3001:3001"
    echo "  kubectl port-forward -n $NAMESPACE svc/prometheus 9090:9090"
}

# Clean up deployment
cleanup() {
    print_header "Cleaning up deployment"
    
    if [[ "$1" == "helm" ]]; then
        print_status "Uninstalling Helm release..."
        helm uninstall crypto-forecasting -n "$NAMESPACE" || true
    else
        print_status "Deleting Kubernetes manifests..."
        kubectl delete namespace "$NAMESPACE" --context="$CONTEXT" || true
    fi
    
    print_status "Cleanup completed"
}

# Show help
show_help() {
    cat << EOF
Crypto Forecasting Kubernetes Deployment Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Deploy using Kubernetes manifests (default)
    helm            Deploy using Helm charts
    status          Show deployment status
    logs [POD]      Show logs for a pod
    cleanup         Remove all resources
    cleanup-helm    Remove Helm release
    port-forward    Start port forwarding for development
    help            Show this help message

Options:
    -n NAMESPACE    Target namespace (default: crypto-forecasting)
    -c CONTEXT      Kubernetes context to use
    --dry-run       Show what would be deployed without applying

Examples:
    $0 deploy                    # Deploy using manifests
    $0 helm                      # Deploy using Helm
    $0 status                    # Check deployment status
    $0 logs backend-xxx          # Show logs for backend pod
    $0 cleanup                   # Remove all resources

EOF
}

# Start port forwarding for development
start_port_forward() {
    print_header "Starting Port Forwarding"
    
    print_status "Port forwarding services for local development..."
    print_status "Access URLs will be:"
    echo "  Frontend: http://localhost:3000"
    echo "  Backend API: http://localhost:5000"
    echo "  Grafana: http://localhost:3001"
    echo "  Prometheus: http://localhost:9090"
    echo ""
    print_status "Press Ctrl+C to stop port forwarding"
    
    # Start port forwarding in background
    kubectl port-forward -n "$NAMESPACE" svc/frontend 3000:3000 --context="$CONTEXT" &
    kubectl port-forward -n "$NAMESPACE" svc/backend 5000:5000 --context="$CONTEXT" &
    kubectl port-forward -n "$NAMESPACE" svc/grafana 3001:3001 --context="$CONTEXT" &
    kubectl port-forward -n "$NAMESPACE" svc/prometheus 9090:9090 --context="$CONTEXT" &
    
    # Wait for interrupt
    trap 'kill $(jobs -p); exit' INT TERM
    wait
}

# Main command dispatcher
main() {
    # Parse options
    while [[ $# -gt 0 ]]; do
        case $1 in
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -c|--context)
                CONTEXT="$2"
                shift 2
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            -*|--*)
                print_error "Unknown option $1"
                show_help
                exit 1
                ;;
            *)
                break
                ;;
        esac
    done
    
    local command="${1:-deploy}"
    
    case "$command" in
        deploy)
            check_prerequisites
            deploy_manifests
            wait_for_deployments
            check_status
            get_access_info
            ;;
        helm)
            check_prerequisites
            deploy_helm
            wait_for_deployments
            check_status
            get_access_info
            ;;
        status)
            check_status
            get_access_info
            ;;
        logs)
            local pod_name="${2}"
            if [[ -z "$pod_name" ]]; then
                print_error "Please specify pod name"
                exit 1
            fi
            kubectl logs -f "$pod_name" -n "$NAMESPACE" --context="$CONTEXT"
            ;;
        cleanup)
            cleanup
            ;;
        cleanup-helm)
            cleanup helm
            ;;
        port-forward)
            start_port_forward
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $command"
            show_help
            exit 1
            ;;
    esac
}

# Run main function with all arguments
main "$@"