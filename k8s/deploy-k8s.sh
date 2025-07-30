#!/bin/bash

# Kubernetes deployment script for DEV ENVIRONMENT
# Optimized for single-node clusters like minikube or Docker Desktop

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Deploying Crypto Forecasting Platform - DEV ENVIRONMENT${NC}"
echo "=================================================================="

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo -e "${RED}❌ kubectl is not installed or not in PATH${NC}"
    exit 1
fi

# Check if cluster is accessible
if ! kubectl cluster-info &> /dev/null; then
    echo -e "${RED}❌ Cannot connect to Kubernetes cluster${NC}"
    echo "Make sure you have a dev cluster running (minikube, Docker Desktop, etc.)"
    exit 1
fi

echo -e "${GREEN}✅ Kubernetes cluster is accessible${NC}"

# Function to wait for deployment
wait_for_deployment() {
    local deployment=$1
    local namespace=$2
    echo -e "${YELLOW}⏳ Waiting for deployment $deployment to be ready...${NC}"
    kubectl wait --for=condition=available --timeout=300s deployment/$deployment -n $namespace || {
        echo -e "${RED}❌ Deployment $deployment failed to become ready${NC}"
        kubectl describe deployment $deployment -n $namespace
        return 1
    }
    echo -e "${GREEN}✅ Deployment $deployment is ready${NC}"
}

# Function to check if pod is running
check_pod_status() {
    local app=$1
    local namespace=$2
    echo -e "${YELLOW}🔍 Checking $app pod status...${NC}"
    kubectl get pods -n $namespace -l app=$app
}

echo -e "\n${YELLOW}📋 Phase 1: Namespace and RBAC Setup${NC}"
echo "--------------------------------------------"
kubectl apply -f 00-namespace-config.yaml
echo -e "${GREEN}✅ Namespace and RBAC configured${NC}"

echo -e "\n${YELLOW}📋 Phase 2: Database and Redis Deployment${NC}"
echo "--------------------------------------------"
kubectl apply -f 01-database-redis.yaml
echo -e "${GREEN}✅ Database manifests applied${NC}"

# Wait for databases to be ready
echo -e "${YELLOW}⏳ Waiting for databases to initialize...${NC}"
sleep 10

wait_for_deployment "postgres" "crypto-forecasting"
wait_for_deployment "redis" "crypto-forecasting"

echo -e "\n${YELLOW}📋 Phase 3: Backend Application Deployment${NC}"
echo "--------------------------------------------"
kubectl apply -f 02-ml-backend.yaml
echo -e "${GREEN}✅ Backend manifests applied${NC}"

wait_for_deployment "backend" "crypto-forecasting"

echo -e "\n${YELLOW}📋 Phase 4: Frontend Application Deployment${NC}"
echo "--------------------------------------------"
kubectl apply -f 03-frontend.yaml
echo -e "${GREEN}✅ Frontend manifests applied${NC}"

wait_for_deployment "frontend" "crypto-forecasting"

echo -e "\n${YELLOW}📋 Phase 5: Monitoring Stack Deployment (Optional)${NC}"
echo "--------------------------------------------"
if [ -f "05-monitoring.yaml" ]; then
    kubectl apply -f 05-monitoring.yaml
    echo -e "${GREEN}✅ Monitoring manifests applied${NC}"
    
    # Wait for monitoring components (non-blocking)
    echo -e "${YELLOW}⏳ Starting monitoring components...${NC}"
    sleep 5
else
    echo -e "${YELLOW}⚠️  Monitoring manifests not found, skipping...${NC}"
fi

echo -e "\n${BLUE}📊 Deployment Summary${NC}"
echo "=================================================================="

# Show deployment status
echo -e "\n${YELLOW}🔍 Deployment Status:${NC}"
kubectl get deployments -n crypto-forecasting

echo -e "\n${YELLOW}🔍 Service Status:${NC}"
kubectl get services -n crypto-forecasting

echo -e "\n${YELLOW}🔍 Pod Status:${NC}"
kubectl get pods -n crypto-forecasting

echo -e "\n${BLUE}🌐 Dev Environment Access Information${NC}"
echo "=================================================================="
echo -e "${GREEN}Frontend Application:${NC}     http://localhost:30030"
echo -e "${GREEN}Backend API:${NC}              http://localhost:30050"
echo -e "${GREEN}Backend API Docs:${NC}         http://localhost:30050/docs"
echo -e "${GREEN}PostgreSQL Database:${NC}      localhost:30432 (user: crypto, db: crypto_forecasting)"
echo -e "${GREEN}Redis Cache:${NC}              localhost:30379"

if kubectl get deployment prometheus -n crypto-forecasting &>/dev/null; then
    echo -e "${GREEN}Prometheus Monitoring:${NC}    http://localhost:30090"
fi

if kubectl get deployment grafana -n crypto-forecasting &>/dev/null; then
    echo -e "${GREEN}Grafana Dashboard:${NC}        http://localhost:30003"
fi

echo -e "\n${BLUE}📝 Quick Commands${NC}"
echo "=================================================================="
echo "# View logs:"
echo "kubectl logs -f deployment/backend -n crypto-forecasting"
echo "kubectl logs -f deployment/frontend -n crypto-forecasting"
echo ""
echo "# Scale services:"
echo "kubectl scale deployment backend --replicas=2 -n crypto-forecasting"
echo ""
echo "# Connect to database:"
echo "psql -h localhost -p 30432 -U crypto -d crypto_forecasting"
echo ""
echo "# Connect to Redis:"
echo "redis-cli -h localhost -p 30379"
echo ""
echo "# Delete everything:"
echo "kubectl delete namespace crypto-forecasting"

echo -e "\n${GREEN}🎉 Dev deployment completed successfully!${NC}"
echo -e "${YELLOW}💡 TIP: Use 'kubectl get pods -n crypto-forecasting -w' to monitor pod status${NC}"
