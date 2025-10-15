"""
Integration tests for the v2 API endpoints
"""
import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from datetime import datetime
import json

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'app'))

from backend.main_v2 import app
from backend.pydantic_models import (
    TrainingRequest, PredictionRequest, HyperparameterTuningRequest,
    ModelComparisonRequest, CryptocurrencyEnum, ModelTypeEnum
)


class TestHealthEndpoints:
    """Test health and status endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_root_endpoint(self, client):
        """Test root endpoint"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Crypto Forecasting API"
        assert data["version"] == "2.0.0"
        assert "status" in data
        assert "timestamp" in data
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert data["services"]["api"] == "active"
    
    def test_metrics_endpoint(self, client):
        """Test metrics endpoint"""
        response = client.get("/metrics")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert "timestamp" in data


class TestTrainingEndpoints:
    """Test training-related endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    @pytest.fixture
    def training_request(self):
        """Create sample training request"""
        return {
            "symbol": "Bitcoin",
            "model_type": "random_forest",
            "prediction_horizon": 7,
            "test_size": 0.2,
            "validation_size": 0.2,
            "feature_window": 7,
            "enable_hyperparameter_tuning": False
        }
    
    def test_train_endpoint(self, client, training_request):
        """Test model training endpoint"""
        response = client.post("/train", json=training_request)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "Bitcoin"
        assert data["model_type"] == "random_forest"
        assert "training_metrics" in data
        assert "model_path" in data
    
    def test_hyperparameter_tuning_endpoint(self, client):
        """Test hyperparameter tuning endpoint"""
        request = {
            "symbol": "Bitcoin",
            "model_types": ["random_forest", "xgboost"],
            "tuning_iterations": 10,
            "cv_folds": 5,
            "test_size": 0.2,
            "validation_size": 0.2
        }
        response = client.post("/hyperparameter-tuning", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert data["symbol"] == "Bitcoin"
        assert len(data["models"]) == 2


class TestPredictionEndpoints:
    """Test prediction endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_predict_endpoint(self, client):
        """Test prediction endpoint"""
        request = {
            "symbol": "Bitcoin",
            "model_type": "random_forest",
            "prediction_steps": 7,
            "use_latest_model": True
        }
        response = client.post("/predict", json=request)
        assert response.status_code == 200
        data = response.json()
        assert data["symbol"] == "Bitcoin"
        assert len(data["predictions"]) == 7
        assert "confidence_intervals" in data
        assert "prediction_dates" in data
    
    def test_realtime_prediction_endpoint(self, client):
        """Test real-time prediction endpoint"""
        response = client.post(
            "/predict/realtime",
            json={
                "symbol": "BTC",
                "data": {"close": 45000, "volume": 1000000},
                "include_confidence": True
            }
        )
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert "timestamp" in data


class TestModelManagementEndpoints:
    """Test model management endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_compare_models_endpoint(self, client):
        """Test model comparison endpoint"""
        request = {
            "symbol": "Bitcoin",
            "model_types": ["random_forest", "xgboost", "decision_tree"],
            "prediction_horizon": 1,
            "enable_tuning": False
        }
        response = client.post("/compare-models", json=request)
        assert response.status_code == 200
        data = response.json()
        assert "comparison_results" in data
        assert len(data["comparison_results"]) == 3
        assert "best_model" in data
    
    def test_list_models_endpoint(self, client):
        """Test list models endpoint"""
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "total" in data
        assert isinstance(data["models"], list)
    
    def test_list_models_with_filters(self, client):
        """Test list models with filters"""
        response = client.get("/models?symbol=Bitcoin&model_type=random_forest")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
    
    def test_delete_model_endpoint(self, client):
        """Test delete model endpoint"""
        response = client.delete("/models/model_001")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "model_001" in data["message"]


class TestCacheEndpoints:
    """Test cache management endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_cache_invalidate_endpoint(self, client):
        """Test cache invalidation endpoint"""
        response = client.delete("/cache/invalidate?pattern=crypto:*")
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "pattern" in data
        assert data["pattern"] == "crypto:*"
    
    def test_cache_stats_endpoint(self, client):
        """Test cache stats endpoint"""
        response = client.get("/cache/stats")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "timestamp" in data


class TestDeepLearningEndpoints:
    """Test deep learning endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_train_deep_learning_endpoint(self, client):
        """Test deep learning training endpoint"""
        response = client.post(
            "/deep-learning/train?symbol=BTC&model_type=lstm&epochs=10"
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "training_started"
        assert data["config"]["model_type"] == "lstm"
        assert data["config"]["epochs"] == 10
    
    def test_list_deep_learning_models_endpoint(self, client):
        """Test list deep learning models endpoint"""
        response = client.get("/deep-learning/models")
        assert response.status_code == 200
        data = response.json()
        assert "deep_learning_models" in data
        assert "total" in data
        assert isinstance(data["deep_learning_models"], list)


class TestMonitoringEndpoints:
    """Test monitoring endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_system_health_endpoint(self, client):
        """Test system health monitoring endpoint"""
        response = client.get("/monitoring/health?hours=2")
        assert response.status_code == 200
        data = response.json()
        assert "metrics" in data
        assert data["period_hours"] == 2
        assert "timestamp" in data
    
    def test_serving_status_endpoint(self, client):
        """Test model serving status endpoint"""
        response = client.get("/serving/status")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data


class TestErrorHandling:
    """Test error handling in API endpoints"""
    
    @pytest.fixture
    def client(self):
        """Create test client"""
        return TestClient(app)
    
    def test_invalid_training_request(self, client):
        """Test handling of invalid training request"""
        invalid_request = {
            "symbol": "InvalidCrypto",  # Not in enum
            "model_type": "invalid_model",  # Not in enum
            "prediction_horizon": -1  # Invalid value
        }
        response = client.post("/train", json=invalid_request)
        assert response.status_code == 422  # Validation error
    
    def test_missing_required_fields(self, client):
        """Test handling of missing required fields"""
        incomplete_request = {
            "symbol": "Bitcoin"
            # Missing other required fields
        }
        response = client.post("/predict", json=incomplete_request)
        assert response.status_code == 422
    
    def test_invalid_query_parameters(self, client):
        """Test handling of invalid query parameters"""
        response = client.get("/monitoring/health?hours=0")  # hours must be >= 1
        assert response.status_code == 422
    
    def test_nonexistent_endpoint(self, client):
        """Test 404 for nonexistent endpoint"""
        response = client.get("/nonexistent")
        assert response.status_code == 404
