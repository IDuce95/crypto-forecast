# Quick Start Guide

Get the Crypto Forecasting Platform up and running in minutes!

## Prerequisites

Before you begin, ensure you have the following installed:

- Docker & Docker Compose
- Python 3.12+
- Git
- At least 8GB RAM available
- 20GB free disk space

## Installation Options

### Option 1: Docker Compose (Recommended for Development)

```bash
# Clone the repository
git clone https://github.com/crypto-forecasting/platform.git
cd crypto-forecasting

# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### Option 2: Kubernetes with Helm

```bash
# Navigate to helm directory
cd helm

# Deploy the platform
./helm-manage.sh deploy

# Check deployment status
./helm-manage.sh status

# Access services
kubectl get svc -n crypto-forecasting
```

### Option 3: Full Automation with Ansible

```bash
# Navigate to ansible directory
cd ansible

# Setup infrastructure
./ansible-manage.sh setup

# Deploy application
./ansible-manage.sh deploy

# Verify health
./ansible-manage.sh health
```

## Accessing the Platform

Once deployed, you can access the following services:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Frontend** | http://localhost:8501 | No auth required |
| **API Documentation** | http://localhost:5000/docs | No auth required |
| **Airflow** | http://localhost:8080 | admin / admin123 |
| **Grafana** | http://localhost:3000 | admin / admin123 |
| **Prometheus** | http://localhost:9090 | No auth required |

## Your First Prediction

### 1. Train a Model

Using the API:

```bash
curl -X POST "http://localhost:5000/train" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "Bitcoin",
    "model_type": "random_forest",
    "prediction_horizon": 7,
    "test_size": 0.2,
    "validation_size": 0.2,
    "feature_window": 7,
    "enable_hyperparameter_tuning": false
  }'
```

### 2. Generate Predictions

```bash
curl -X POST "http://localhost:5000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "Bitcoin",
    "model_type": "random_forest",
    "prediction_steps": 7,
    "use_latest_model": true
  }'
```

### 3. Using the Web Interface

1. Open http://localhost:8501 in your browser
2. Select "Bitcoin" from the dropdown
3. Choose "Random Forest" as the model type
4. Click "Train Model"
5. Once training completes, click "Generate Predictions"

## Verify Installation

Run the health check to ensure everything is working:

```bash
# API Health Check
curl http://localhost:5000/health

# Expected response:
{
  "status": "healthy",
  "services": {
    "api": "active",
    "cache": "active",
    "ml_pipeline": "active"
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

## Sample Workflow

### Complete ML Pipeline

```python
import requests
import json

API_URL = "http://localhost:5000"

# Step 1: Train multiple models
models = ["random_forest", "xgboost", "decision_tree"]
for model in models:
    response = requests.post(
        f"{API_URL}/train",
        json={
            "symbol": "Bitcoin",
            "model_type": model,
            "prediction_horizon": 7,
            "test_size": 0.2,
            "validation_size": 0.2,
            "feature_window": 7,
            "enable_hyperparameter_tuning": False
        }
    )
    print(f"Training {model}: {response.json()}")

# Step 2: Compare models
comparison = requests.post(
    f"{API_URL}/compare-models",
    json={
        "symbol": "Bitcoin",
        "model_types": models,
        "prediction_horizon": 7,
        "enable_tuning": False
    }
)
best_model = comparison.json()["best_model"]
print(f"Best model: {best_model}")

# Step 3: Generate predictions with best model
predictions = requests.post(
    f"{API_URL}/predict",
    json={
        "symbol": "Bitcoin",
        "model_type": best_model,
        "prediction_steps": 7,
        "use_latest_model": True
    }
)
print(f"Predictions: {predictions.json()}")
```

## Monitoring

### View Grafana Dashboard

1. Open http://localhost:3000
2. Login with admin/admin123
3. Navigate to Dashboards > Crypto Forecasting Platform
4. Monitor real-time metrics

### Check Prometheus Metrics

1. Open http://localhost:9090
2. Execute queries:
   - `up` - Service health status
   - `http_requests_total` - API request count
   - `model_accuracy` - Model performance

## Troubleshooting

### Services Not Starting

```bash
# Check Docker resources
docker system df
docker system prune -a

# Restart services
docker-compose down
docker-compose up -d
```

### Port Conflicts

If you encounter port conflicts:

```bash
# Check what's using the ports
sudo lsof -i :5000
sudo lsof -i :8501

# Modify docker-compose.yml to use different ports
```

### Database Connection Issues

```bash
# Check PostgreSQL logs
docker-compose logs postgres

# Reset database
docker-compose exec postgres psql -U postgres -c "DROP DATABASE IF EXISTS crypto_forecasting;"
docker-compose exec postgres psql -U postgres -c "CREATE DATABASE crypto_forecasting;"
```

## Next Steps

- [Configuration Guide](configuration.md) - Customize your deployment
- [API Reference](../api/rest-api.md) - Explore all API endpoints
- [ML Models](../ml/models/random-forest.md) - Understand the models
- [Production Deployment](../deployment/kubernetes.md) - Deploy to production

## Support

Need help? Check our:
- [FAQ](../troubleshooting/faq.md)
- [Common Issues](../troubleshooting/common-issues.md)
- [GitHub Issues](https://github.com/crypto-forecasting/platform/issues)
