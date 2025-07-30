#!/bin/bash

# Build Docker images for dev Kubernetes deployment
# This script builds images and makes them available for dev cluster

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🐳 Building Docker Images for Dev Environment${NC}"
echo "=================================================================="

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo -e "${RED}❌ Docker is not installed${NC}"
    exit 1
fi

# Function to build image
build_image() {
    local name=$1
    local dockerfile=$2
    local context=$3
    local tag="crypto-forecasting-${name}:dev"

    echo -e "\n${YELLOW}🔨 Building $name image...${NC}"

    if [ -f "$dockerfile" ]; then
        docker build -t "$tag" -f "$dockerfile" "$context"
        echo -e "${GREEN}✅ Built $tag${NC}"
    else
        echo -e "${RED}❌ Dockerfile not found: $dockerfile${NC}"
        return 1
    fi
}

# Build backend image
echo -e "\n${YELLOW}📦 Building Backend Image${NC}"
echo "----------------------------------------"
if [ -f "app/Dockerfile.backend" ]; then
    build_image "backend" "app/Dockerfile.backend" "app/"
else
    echo -e "${YELLOW}⚠️  Creating simple backend Dockerfile...${NC}"
    cat > app/Dockerfile.backend << 'EOF'
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Run application
CMD ["python", "backend/main.py"]
EOF
    build_image "backend" "app/Dockerfile.backend" "app/"
fi

# Build frontend image
echo -e "\n${YELLOW}📦 Building Frontend Image${NC}"
echo "----------------------------------------"
if [ -f "app/Dockerfile.frontend" ]; then
    build_image "frontend" "app/Dockerfile.frontend" "app/"
else
    echo -e "${YELLOW}⚠️  Creating simple frontend Dockerfile...${NC}"
    cat > app/Dockerfile.frontend << 'EOF'
FROM node:16-alpine

WORKDIR /app

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm install

# Copy source code
COPY frontend/ .

# Build application
RUN npm run build

# Use nginx to serve static files
FROM nginx:alpine
COPY --from=0 /app/dist /usr/share/nginx/html
COPY --from=0 /app/nginx.conf /etc/nginx/nginx.conf

EXPOSE 3000

CMD ["nginx", "-g", "daemon off;"]
EOF

    # Create a simple nginx config if it doesn't exist
    if [ ! -f "app/frontend/nginx.conf" ]; then
        mkdir -p app/frontend
        cat > app/frontend/nginx.conf << 'EOF'
events {
    worker_connections 1024;
}

http {
    include       /etc/nginx/mime.types;
    default_type  application/octet-stream;

    server {
        listen 3000;
        server_name localhost;

        root /usr/share/nginx/html;
        index index.html;

        location / {
            try_files $uri $uri/ /index.html;
        }

        location /api {
            proxy_pass http://backend:5000;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
    }
}
EOF
    fi

    build_image "frontend" "app/Dockerfile.frontend" "app/"
fi

# Build Airflow image
echo -e "\n${YELLOW}📦 Building Airflow Image${NC}"
echo "----------------------------------------"
if [ -f "airflow/Dockerfile.airflow" ]; then
    build_image "airflow" "airflow/Dockerfile.airflow" "airflow/"
else
    echo -e "${YELLOW}⚠️  Airflow Dockerfile not found, skipping...${NC}"
fi

echo -e "\n${BLUE}📋 Image Build Summary${NC}"
echo "=================================================================="

# List built images
echo -e "\n${YELLOW}🔍 Built Images:${NC}"
docker images | grep crypto-forecasting

echo -e "\n${GREEN}🎉 All images built successfully!${NC}"

echo -e "\n${BLUE}📝 Next Steps${NC}"
echo "=================================================================="
echo "1. Deploy with Helm:"
echo "   cd helm && ./helm-manage.sh deploy"
echo ""
echo "2. Or deploy with kubectl:"
echo "   cd k8s && ./deploy-k8s.sh"
echo ""
echo "3. Access services at:"
echo "   Frontend: http://localhost:30030"
echo "   Backend:  http://localhost:30050"
echo "   Airflow:  http://localhost:30080"

# If using minikube, mention image loading
if command -v minikube &> /dev/null && minikube status &> /dev/null; then
    echo -e "\n${YELLOW}📌 Note for Minikube:${NC}"
    echo "Images are built and available for minikube to use."
    echo "Make sure imagePullPolicy is set to 'Never' in your configs."
fi

# If using kind, mention image loading
if command -v kind &> /dev/null; then
    echo -e "\n${YELLOW}📌 Note for Kind:${NC}"
    echo "To load images into kind cluster, run:"
    echo "kind load docker-image crypto-forecasting-backend:latest"
    echo "kind load docker-image crypto-forecasting-frontend:latest"
fi
