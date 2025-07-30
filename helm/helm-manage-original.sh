#!/bin/bash
# Helm management script for crypto-forecasting project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
HELM_CHART_DIR="$PROJECT_ROOT/helm/crypto-forecasting"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
NAMESPACE="crypto-forecasting"
RELEASE_NAME="crypto-forecasting"
CONTEXT="${KUBE_CONTEXT:-$(kubectl config current-context)}"

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
    if ! command -v helm &> /dev/null; then
        print_error "Helm is not installed. Please install it first:"
        echo "  curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        print_error "kubectl is not installed. Please install it first."
        exit 1
    fi
    
    if [[ ! -d "$HELM_CHART_DIR" ]]; then
        print_error "Helm chart directory not found: $HELM_CHART_DIR"
        exit 1
    fi
    
    print_status "Helm version: $(helm version --short)"
    print_status "Using context: $CONTEXT"
    print_status "Target namespace: $NAMESPACE"
}

# Add required Helm repositories
setup_repositories() {
    print_header "Setting up Helm Repositories"
    
    helm repo add bitnami https://charts.bitnami.com/bitnami
    helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
    helm repo add grafana https://grafana.github.io/helm-charts
    helm repo update
    
    print_status "Repositories added and updated"
}

# Update dependencies
update_dependencies() {
    print_header "Updating Helm Dependencies"
    
    cd "$HELM_CHART_DIR"
    helm dependency update
    
    print_status "Dependencies updated"
}

# Install or upgrade the release
deploy() {
    print_header "Deploying Helm Release"
    
    local values_file="${VALUES_FILE:-$HELM_CHART_DIR/values.yaml}"
    local extra_args=()
    
    # Add extra arguments
    if [[ -n "$ENVIRONMENT" ]]; then
        extra_args+=(--set global.environment="$ENVIRONMENT")
    fi
    
    if [[ -n "$IMAGE_TAG" ]]; then
        extra_args+=(--set app.image.tag="$IMAGE_TAG")
        extra_args+=(--set frontend.image.tag="$IMAGE_TAG")
    fi
    
    # Check if release exists
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        print_status "Upgrading existing release..."
        helm upgrade "$RELEASE_NAME" "$HELM_CHART_DIR" \
            --namespace "$NAMESPACE" \
            --kube-context="$CONTEXT" \
            --values "$values_file" \
            --wait \
            --timeout 10m \
            "${extra_args[@]}" \
            "${HELM_EXTRA_ARGS[@]}"
    else
        print_status "Installing new release..."
        helm install "$RELEASE_NAME" "$HELM_CHART_DIR" \
            --namespace "$NAMESPACE" \
            --create-namespace \
            --kube-context="$CONTEXT" \
            --values "$values_file" \
            --wait \
            --timeout 10m \
            "${extra_args[@]}" \
            "${HELM_EXTRA_ARGS[@]}"
    fi
    
    print_status "Deployment completed successfully"
}

# Show release status
status() {
    print_header "Helm Release Status"
    
    helm status "$RELEASE_NAME" -n "$NAMESPACE" --kube-context="$CONTEXT"
    
    print_header "Pod Status"
    kubectl get pods -n "$NAMESPACE" --context="$CONTEXT"
    
    print_header "Service Status"
    kubectl get services -n "$NAMESPACE" --context="$CONTEXT"
}

# Uninstall the release
uninstall() {
    print_header "Uninstalling Helm Release"
    
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        helm uninstall "$RELEASE_NAME" -n "$NAMESPACE" --kube-context="$CONTEXT"
        print_status "Release uninstalled successfully"
    else
        print_warning "Release $RELEASE_NAME not found"
    fi
}

# Test the release
test() {
    print_header "Testing Helm Release"
    
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        helm test "$RELEASE_NAME" -n "$NAMESPACE" --kube-context="$CONTEXT"
    else
        print_error "Release $RELEASE_NAME not found"
        exit 1
    fi
}

# Lint the chart
lint() {
    print_header "Linting Helm Chart"
    
    cd "$HELM_CHART_DIR"
    helm lint .
    
    print_status "Lint completed"
}

# Template the chart
template() {
    print_header "Templating Helm Chart"
    
    local output_dir="${OUTPUT_DIR:-$PROJECT_ROOT/k8s/generated}"
    local values_file="${VALUES_FILE:-$HELM_CHART_DIR/values.yaml}"
    
    mkdir -p "$output_dir"
    
    helm template "$RELEASE_NAME" "$HELM_CHART_DIR" \
        --namespace "$NAMESPACE" \
        --values "$values_file" \
        --output-dir "$output_dir"
    
    print_status "Templates generated in: $output_dir"
}

# Package the chart
package() {
    print_header "Packaging Helm Chart"
    
    local output_dir="${OUTPUT_DIR:-$PROJECT_ROOT/helm/packages}"
    mkdir -p "$output_dir"
    
    cd "$HELM_CHART_DIR"
    helm package . --destination "$output_dir"
    
    print_status "Chart packaged in: $output_dir"
}

# Show values
show_values() {
    print_header "Helm Values"
    
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        helm get values "$RELEASE_NAME" -n "$NAMESPACE" --kube-context="$CONTEXT"
    else
        print_warning "Release $RELEASE_NAME not found. Showing default values:"
        cat "$HELM_CHART_DIR/values.yaml"
    fi
}

# Show history
history() {
    print_header "Helm Release History"
    
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        helm history "$RELEASE_NAME" -n "$NAMESPACE" --kube-context="$CONTEXT"
    else
        print_error "Release $RELEASE_NAME not found"
        exit 1
    fi
}

# Rollback to previous version
rollback() {
    print_header "Rolling back Helm Release"
    
    local revision="${1:-0}"  # 0 means previous revision
    
    if helm list -n "$NAMESPACE" --kube-context="$CONTEXT" | grep -q "$RELEASE_NAME"; then
        helm rollback "$RELEASE_NAME" "$revision" -n "$NAMESPACE" --kube-context="$CONTEXT"
        print_status "Rollback completed"
    else
        print_error "Release $RELEASE_NAME not found"
        exit 1
    fi
}

# Show help
show_help() {
    cat << EOF
Crypto Forecasting Helm Management Script

Usage: $0 [COMMAND] [OPTIONS]

Commands:
    deploy          Install or upgrade the Helm release
    status          Show release and resource status
    uninstall       Uninstall the Helm release
    test            Run Helm tests
    lint            Lint the Helm chart
    template        Generate Kubernetes manifests from templates
    package         Package the Helm chart
    values          Show current values
    history         Show release history
    rollback [REV]  Rollback to previous or specific revision
    deps            Update chart dependencies
    repos           Setup required Helm repositories

Options:
    -n NAMESPACE           Target namespace (default: crypto-forecasting)
    -c CONTEXT            Kubernetes context to use
    -f VALUES_FILE        Values file to use
    -e ENVIRONMENT        Environment (development, staging, production)
    -t IMAGE_TAG          Docker image tag to use
    --output-dir DIR      Output directory for templates/packages
    --extra-args "ARGS"   Additional Helm arguments

Environment Variables:
    KUBE_CONTEXT          Kubernetes context
    VALUES_FILE           Values file path
    ENVIRONMENT           Target environment
    IMAGE_TAG             Docker image tag
    OUTPUT_DIR            Output directory

Examples:
    $0 deploy                                    # Deploy with default values
    $0 deploy -e production -t v1.2.0           # Deploy production with specific tag
    $0 deploy -f values-staging.yaml            # Deploy with custom values file
    $0 status                                    # Check deployment status
    $0 template --output-dir ./generated        # Generate manifests
    $0 rollback 2                               # Rollback to revision 2
    $0 uninstall                                # Remove deployment

EOF
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
            -f|--values-file)
                VALUES_FILE="$2"
                shift 2
                ;;
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--image-tag)
                IMAGE_TAG="$2"
                shift 2
                ;;
            --output-dir)
                OUTPUT_DIR="$2"
                shift 2
                ;;
            --extra-args)
                IFS=' ' read -ra HELM_EXTRA_ARGS <<< "$2"
                shift 2
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
            setup_repositories
            update_dependencies
            deploy
            status
            ;;
        status)
            check_prerequisites
            status
            ;;
        uninstall)
            check_prerequisites
            uninstall
            ;;
        test)
            check_prerequisites
            test
            ;;
        lint)
            check_prerequisites
            lint
            ;;
        template)
            check_prerequisites
            template
            ;;
        package)
            check_prerequisites
            package
            ;;
        values)
            check_prerequisites
            show_values
            ;;
        history)
            check_prerequisites
            history
            ;;
        rollback)
            check_prerequisites
            rollback "${2}"
            ;;
        deps)
            check_prerequisites
            update_dependencies
            ;;
        repos)
            setup_repositories
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
