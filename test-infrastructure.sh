#!/bin/bash

# Test script for the entire crypto-forecasting infrastructure
# Usage: ./test-infrastructure.sh [env]

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Environment (default: dev)
ENV=${1:-dev}

echo -e "${BLUE}🚀 Testing Crypto Forecasting Infrastructure - Environment: $ENV${NC}"
echo "=============================================================="

# Function to print test results
print_result() {
    if [ $1 -eq 0 ]; then
        echo -e "${GREEN}✅ $2${NC}"
    else
        echo -e "${RED}❌ $2${NC}"
        return 1
    fi
}

# Function to run test with timeout
run_test() {
    timeout 30s bash -c "$1" >/dev/null 2>&1
    return $?
}

echo -e "\n${YELLOW}📋 Phase 1: Prerequisites Check${NC}"
echo "----------------------------------------"

# Check if required tools are installed
echo "Checking required tools..."

# Docker
if command -v docker &> /dev/null; then
    print_result 0 "Docker is installed"
    DOCKER_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Docker is not installed (optional for structure validation)${NC}"
    DOCKER_AVAILABLE=false
fi

# Kubectl
if command -v kubectl &> /dev/null; then
    print_result 0 "kubectl is installed"
    KUBECTL_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  kubectl is not installed (optional for structure validation)${NC}"
    KUBECTL_AVAILABLE=false
fi

# Helm
if command -v helm &> /dev/null; then
    print_result 0 "Helm is installed"
    HELM_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Helm is not installed (optional for structure validation)${NC}"
    HELM_AVAILABLE=false
fi

# Ansible
if command -v ansible &> /dev/null; then
    print_result 0 "Ansible is installed"
    ANSIBLE_AVAILABLE=true
else
    echo -e "${YELLOW}⚠️  Ansible is not installed (optional for structure validation)${NC}"
    ANSIBLE_AVAILABLE=false
fi

echo -e "\n${YELLOW}📋 Phase 2: Project Structure Validation${NC}"
echo "----------------------------------------"

# Check project structure
echo "Validating project structure..."

# Data directory
if [ -d "data/raw" ] && [ -d "data/processed" ]; then
    print_result 0 "Data directories are properly structured"
else
    print_result 1 "Data directories are missing or incorrectly structured"
fi

# Ansible structure
if [ -f "ansible/ansible-manage.sh" ] && [ -f "ansible/inventory/hosts.yml" ]; then
    print_result 0 "Ansible structure is valid"
else
    print_result 1 "Ansible structure is invalid"
fi

# Kubernetes manifests
if [ -d "k8s" ] && [ -f "k8s/deploy-k8s.sh" ]; then
    print_result 0 "Kubernetes manifests exist"
else
    print_result 1 "Kubernetes manifests are missing"
fi

# Helm charts
if [ -f "helm/crypto-forecasting/Chart.yaml" ] && [ -f "helm/crypto-forecasting/values.yaml" ]; then
    print_result 0 "Helm charts exist"
else
    print_result 1 "Helm charts are missing"
fi

echo -e "\n${YELLOW}📋 Phase 3: Ansible Tests${NC}"
echo "----------------------------------------"

if [ "$ANSIBLE_AVAILABLE" = true ]; then
    cd ansible

    # Ansible syntax check
    echo "Testing Ansible syntax..."
    if ansible-playbook --syntax-check playbooks/main-deployment.yml >/dev/null 2>&1; then
        print_result 0 "Ansible playbook syntax is valid"
    else
        print_result 1 "Ansible playbook syntax errors found"
    fi

    # Ansible inventory check
    echo "Testing Ansible inventory..."
    if ansible-inventory --list >/dev/null 2>&1; then
        print_result 0 "Ansible inventory is valid"
    else
        print_result 1 "Ansible inventory has errors"
    fi

    cd ..
else
    echo -e "${YELLOW}⚠️  Skipping Ansible tests - Ansible not installed${NC}"
fi

echo -e "\n${YELLOW}📋 Phase 4: Helm Chart Tests${NC}"
echo "----------------------------------------"

if [ "$HELM_AVAILABLE" = true ]; then
    cd helm/crypto-forecasting

    # Helm chart linting
    echo "Linting Helm charts..."
    if helm lint . >/dev/null 2>&1; then
        print_result 0 "Helm chart passes linting"
    else
        print_result 1 "Helm chart linting failed"
    fi

    # Helm template rendering
    echo "Testing Helm template rendering..."
    if helm template crypto-forecasting . --values values-${ENV}.yaml >/dev/null 2>&1; then
        print_result 0 "Helm templates render successfully"
    else
        print_result 1 "Helm template rendering failed"
    fi

    # Helm dependencies
    echo "Checking Helm dependencies..."
    if helm dependency list >/dev/null 2>&1; then
        print_result 0 "Helm dependencies are valid"
    else
        print_result 1 "Helm dependencies check failed"
    fi

    cd ../..
else
    echo -e "${YELLOW}⚠️  Skipping Helm tests - Helm not installed${NC}"
fi

echo -e "\n${YELLOW}📋 Phase 5: Kubernetes Manifest Tests${NC}"
echo "----------------------------------------"

if [ "$KUBECTL_AVAILABLE" = true ]; then
    cd k8s

    # Kubernetes manifest validation
    echo "Validating Kubernetes manifests..."
    VALIDATION_PASSED=true

    for manifest in *.yaml; do
        if kubectl apply --dry-run=client -f "$manifest" >/dev/null 2>&1; then
            echo -e "${GREEN}  ✅ $manifest${NC}"
        else
            echo -e "${RED}  ❌ $manifest${NC}"
            VALIDATION_PASSED=false
        fi
    done

    if [ "$VALIDATION_PASSED" = true ]; then
        print_result 0 "All Kubernetes manifests are valid"
    else
        print_result 1 "Some Kubernetes manifests have validation errors"
    fi

    cd ..
else
    echo -e "${YELLOW}⚠️  Skipping Kubernetes manifest validation - kubectl not available${NC}"
    # Just check if manifests exist
    if [ -d "k8s" ] && [ "$(ls k8s/*.yaml 2>/dev/null | wc -l)" -gt 0 ]; then
        print_result 0 "Kubernetes manifest files exist"
    else
        print_result 1 "Kubernetes manifest files are missing"
    fi
fi

echo -e "\n${YELLOW}📋 Phase 6: Docker Tests${NC}"
echo "----------------------------------------"

# Check if Dockerfiles exist
echo "Checking Dockerfiles..."
DOCKERFILES_FOUND=0

if [ -f "app/Dockerfile.backend" ]; then
    print_result 0 "Backend Dockerfile exists"
    DOCKERFILES_FOUND=$((DOCKERFILES_FOUND + 1))
else
    print_result 1 "Backend Dockerfile is missing"
fi

if [ -f "app/Dockerfile.frontend" ]; then
    print_result 0 "Frontend Dockerfile exists"
    DOCKERFILES_FOUND=$((DOCKERFILES_FOUND + 1))
else
    print_result 1 "Frontend Dockerfile is missing"
fi

if [ -f "airflow/Dockerfile.airflow" ]; then
    print_result 0 "Airflow Dockerfile exists"
    DOCKERFILES_FOUND=$((DOCKERFILES_FOUND + 1))
else
    print_result 1 "Airflow Dockerfile is missing"
fi

echo -e "\n${YELLOW}📋 Phase 7: Integration Tests${NC}"
echo "----------------------------------------"

# Test script permissions
echo "Checking script permissions..."
SCRIPTS_EXECUTABLE=0

for script in ansible/ansible-manage.sh helm/helm-manage.sh k8s/deploy-k8s.sh; do
    if [ -x "$script" ]; then
        SCRIPTS_EXECUTABLE=$((SCRIPTS_EXECUTABLE + 1))
    fi
done

if [ $SCRIPTS_EXECUTABLE -eq 3 ]; then
    print_result 0 "All management scripts are executable"
else
    print_result 1 "Some management scripts are not executable"
fi

# Test configuration files
echo "Checking configuration files..."
CONFIG_FILES_FOUND=0

if [ -f "app/config.py" ] || [ -f "app/settings.toml" ]; then
    print_result 0 "Application configuration files exist"
    CONFIG_FILES_FOUND=$((CONFIG_FILES_FOUND + 1))
else
    print_result 1 "Application configuration files are missing"
fi

if [ -f "docker-compose.yml" ]; then
    print_result 0 "Docker Compose configuration exists"
    CONFIG_FILES_FOUND=$((CONFIG_FILES_FOUND + 1))
else
    print_result 1 "Docker Compose configuration is missing"
fi

echo -e "\n${BLUE}📊 Test Summary${NC}"
echo "=============================================================="

# Calculate overall score
TOTAL_TESTS=20
PASSED_TESTS=0

# This is a simplified scoring - in a real scenario, you'd track each test result
if [ -d "data/raw" ]; then ((PASSED_TESTS++)); fi
if [ -f "ansible/ansible-manage.sh" ]; then ((PASSED_TESTS++)); fi
if [ -f "helm/crypto-forecasting/Chart.yaml" ]; then ((PASSED_TESTS++)); fi
if [ -d "k8s" ]; then ((PASSED_TESTS++)); fi
if command -v docker &> /dev/null; then ((PASSED_TESTS++)); fi
if command -v kubectl &> /dev/null; then ((PASSED_TESTS++)); fi
if command -v helm &> /dev/null; then ((PASSED_TESTS++)); fi
if command -v ansible &> /dev/null; then ((PASSED_TESTS++)); fi

SCORE=$((PASSED_TESTS * 100 / 8))  # Simplified calculation

echo "Environment: $ENV"
echo "Tests Passed: $PASSED_TESTS/8 (simplified count)"
echo "Score: $SCORE%"

if [ $SCORE -ge 80 ]; then
    echo -e "${GREEN}🎉 Infrastructure is ready for deployment!${NC}"
    exit 0
elif [ $SCORE -ge 60 ]; then
    echo -e "${YELLOW}⚠️  Infrastructure needs some improvements${NC}"
    exit 1
else
    echo -e "${RED}🚨 Infrastructure has critical issues${NC}"
    exit 2
fi
