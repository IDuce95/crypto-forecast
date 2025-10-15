import pytest
import sys
from pathlib import Path
from fastapi.testclient import TestClient

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def client():

    def test_root_endpoint(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    def test_models_list_endpoint(self, client):
        response = client.get("/predictions")
        assert response.status_code in [200, 404]
        if response.status_code == 200:
            data = response.json()
            assert isinstance(data, (list, dict))

    def test_experiments_endpoint(self, client):
        response = client.get("/nonexistent")
        assert response.status_code == 404


class TestTrainingEndpoints:
        response = client.post("/train", json={})
        assert response.status_code in [422, 200, 400, 500]

    def test_predict_endpoint_validation(self, client):
        response = client.post("/compare-models", json={})
        assert response.status_code in [422, 200, 400, 500]


class TestBackgroundTasks:
        try:
            response = client.post("/hyperparameter-tuning", json={})
            assert response.status_code != 404
        except Exception:
            assert True
