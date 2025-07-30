#!/bin/bash

# Helm management script for Crypto Forecasting Platform - LOCAL DEVELOPMENT
# Simplified for single-environment local deployment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration for local development
CHART_NAME="crypto-forecasting"
CHART_PATH="./crypto-forecasting"
NAMESPACE="crypto-forecasting"
RELEASE_NAME="crypto-forecasting"
VALUES_FILE="values.yaml"  # Use main values file optimized for local

# Function to print usage
print_usage() {
    echo -e "${BLUE}Helm Management Script - Local Development${NC}"
    echo "Usage: $0 {deploy|upgrade|uninstall|status|logs|test|deps|lint|template|values}"
    echo ""
    echo "Commands:"
    echo "  deploy     - Deploy the application to local cluster"
    echo "  upgrade    - Upgrade existing deployment"
    echo "  uninstall  - Remove the application"
    echo "  status     - Show deployment status"
    echo "  logs       - Show application logs"
    echo "  test       - Run Helm tests"
    echo "  deps       - Update chart dependencies"
    echo "  lint       - Lint the Helm chart"
    echo "  template   - Render templates locally"
    echo "  values     - Show effective values"
    echo ""
    echo "Examples:"
    echo "  $0 deploy              # Deploy with default local values"
    echo "  $0 status              # Check deployment status"
    echo "  $0 logs backend        # Show backend logs"
    echo "  $0 uninstall           # Clean remove"
}

# Function to check prerequisites
check_prerequisites() {
    echo -e "${YELLOW}🔍 Checking prerequisites...${NC}"
    
    # Check Helm
    if ! command -v helm &> /dev/null; then
        echo -e "${RED}❌ Helm is not installed${NC}"
        exit 1
    fi
    
    # Check kubectl
    if ! command -v kubectl &> /dev/null; then
        echo -e "${RED}❌ kubectl is not installed${NC}"
        exit 1
    fi
    
    # Check cluster connectivity
    if ! kubectl cluster-info &> /dev/null; then
        echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
        echo "Make sure you have a local cluster running (minikube, Docker Desktop, etc.)"
        exit 1
    fi
    
    echo -e "${GREEN}✅ All prerequisites met${NC}"
}

# Function to deploy application
deploy() {
    echo -e "${BLUE}🚀 Deploying Crypto Forecasting Platform locally...${NC}"
    
    check_prerequisites
    
    # Create namespace if it doesn't exist
    kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
    
    # Update dependencies
    echo -e "${YELLOW}📦 Updating chart dependencies...${NC}"
    helm dependency update $CHART_PATH
    
    # Deploy
    echo -e "${YELLOW}🚀 Installing/upgrading release...${NC}"
    helm upgrade --install $RELEASE_NAME $CHART_PATH \
        --namespace $NAMESPACE \
        --values $CHART_PATH/$VALUES_FILE \
        --timeout 10m \
        --wait \
        --create-namespace
    
    echo -e "${GREEN}✅ Deployment completed!${NC}"
    show_access_info
}

# Function to upgrade application
upgrade() {
    echo -e "${BLUE}⬆️  Upgrading Crypto Forecasting Platform...${NC}"
    
    check_prerequisites
    
    helm upgrade $RELEASE_NAME $CHART_PATH \
        --namespace $NAMESPACE \
        --values $CHART_PATH/$VALUES_FILE \
        --timeout 10m \
        --wait
    
    echo -e "${GREEN}✅ Upgrade completed!${NC}"
    show_access_info
}

# Function to uninstall application
uninstall() {
    echo -e "${YELLOW}🗑️  Uninstalling Crypto Forecasting Platform...${NC}"
    
    read -p "Are you sure you want to uninstall? This will delete all data! (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        helm uninstall $RELEASE_NAME --namespace $NAMESPACE
        echo -e "${YELLOW}🧹 Cleaning up namespace...${NC}"
        kubectl delete namespace $NAMESPACE --ignore-not-found=true
        echo -e "${GREEN}✅ Uninstall completed!${NC}"
    else
        echo "Uninstall cancelled."
    fi
}

# Function to show status
status() {
    echo -e "${BLUE}📊 Deployment Status${NC}"
    echo "=================================="
    
    # Helm status
    echo -e "\n${YELLOW}📦 Helm Release Status:${NC}"
    helm status $RELEASE_NAME --namespace $NAMESPACE 2>/dev/null || echo "Release not found"
    
    # Kubernetes resources
    echo -e "\n${YELLOW}🔍 Kubernetes Resources:${NC}"
    kubectl get all -n $NAMESPACE 2>/dev/null || echo "Namespace not found"
    
    # Pod status with details
    echo -e "\n${YELLOW}🏠 Pod Details:${NC}"
    kubectl get pods -n $NAMESPACE -o wide 2>/dev/null || echo "No pods found"
    
    # Service endpoints
    echo -e "\n${YELLOW}🌐 Service Endpoints:${NC}"
    kubectl get svc -n $NAMESPACE 2>/dev/null || echo "No services found"
    
    show_access_info
}

# Function to show access information
show_access_info() {
    echo -e "\n${BLUE}🌐 Local Access Information${NC}"
    echo "=================================="
    echo -e "${GREEN}Frontend:${NC}          http://localhost:30030"
    echo -e "${GREEN}Backend API:${NC}       http://localhost:30050"
    echo -e "${GREEN}API Documentation:${NC}  http://localhost:30050/docs"
    echo -e "${GREEN}PostgreSQL:${NC}        localhost:30432 (user: crypto, db: crypto_forecasting)"
    echo -e "${GREEN}Redis:${NC}             localhost:30379"
    
    # Check if monitoring is enabled
    if kubectl get svc -n $NAMESPACE | grep -q prometheus 2>/dev/null; then
        echo -e "${GREEN}Prometheus:${NC}        http://localhost:30090"
    fi
    
    if kubectl get svc -n $NAMESPACE | grep -q grafana 2>/dev/null; then
        echo -e "${GREEN}Grafana:${NC}           http://localhost:30003"
    fi
}

# Function to show logs
logs() {
    local component=${1:-all}
    
    echo -e "${BLUE}📋 Application Logs${NC}"
    echo "=================================="
    
    case $component in
        "backend"|"api")
            echo -e "${YELLOW}🔍 Backend logs:${NC}"
            kubectl logs -f deployment/backend -n $NAMESPACE
            ;;
        "frontend"|"ui")
            echo -e "${YELLOW}🔍 Frontend logs:${NC}"
            kubectl logs -f deployment/frontend -n $NAMESPACE
            ;;
        "postgres"|"db")
            echo -e "${YELLOW}🔍 PostgreSQL logs:${NC}"
            kubectl logs -f deployment/postgres -n $NAMESPACE
            ;;
        "redis")
            echo -e "${YELLOW}🔍 Redis logs:${NC}"
            kubectl logs -f deployment/redis -n $NAMESPACE
            ;;
        "all"|*)
            echo -e "${YELLOW}🔍 All application logs:${NC}"
            kubectl logs -f deployment/backend -n $NAMESPACE --tail=20 &
            kubectl logs -f deployment/frontend -n $NAMESPACE --tail=20 &
            wait
            ;;
    esac
}

# Function to run tests
test() {
    echo -e "${BLUE}🧪 Running Helm Tests${NC}"
    echo "=================================="
    
    helm test $RELEASE_NAME --namespace $NAMESPACE
}

# Function to update dependencies
deps() {
    echo -e "${BLUE}📦 Updating Chart Dependencies${NC}"
    echo "=================================="
    
    cd $CHART_PATH
    helm dependency update
    cd -
    
    echo -e "${GREEN}✅ Dependencies updated${NC}"
}

# Function to lint chart
lint() {
    echo -e "${BLUE}🔍 Linting Helm Chart${NC}"
    echo "=================================="
    
    helm lint $CHART_PATH --values $CHART_PATH/$VALUES_FILE
    
    echo -e "${GREEN}✅ Linting completed${NC}"
}

# Function to render templates
template() {
    echo -e "${BLUE}📝 Rendering Helm Templates${NC}"
    echo "=================================="
    
    helm template $RELEASE_NAME $CHART_PATH \
        --values $CHART_PATH/$VALUES_FILE \
        --namespace $NAMESPACE
}

# Function to show effective values
values() {
    echo -e "${BLUE}⚙️  Effective Values${NC}"
    echo "=================================="
    
    helm get values $RELEASE_NAME --namespace $NAMESPACE 2>/dev/null || {
        echo "Release not deployed. Showing default values:"
        cat $CHART_PATH/$VALUES_FILE
    }
}

# Main script logic
case ${1:-help} in
    "deploy")
        deploy
        ;;
    "upgrade")
        upgrade
        ;;
    "uninstall"|"remove"|"delete")
        uninstall
        ;;
    "status"|"info")
        status
        ;;
    "logs")
        logs $2
        ;;
    "test")
        test
        ;;
    "deps"|"dependencies")
        deps
        ;;
    "lint")
        lint
        ;;
    "template"|"render")
        template
        ;;
    "values")
        values
        ;;
    "help"|*)
        print_usage
        ;;
esac
