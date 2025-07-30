

from fastapi import FastAPI
from typing import Dict, Any
import uvicorn
import sys
import os

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from backend import pydantic_models
    from config import config
    from logger import logger
    from backend.services.model_trainer import ModelTrainer
    from backend.services.model_optimizer import ModelOptimizer
    from backend.services.endpoint_handlers import EndpointHandlers
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    raise

app = FastAPI(
    title="Crypto Forecasting API",
    description="API for training ML models for cryptocurrency forecasting",
    version="1.0.0"
)

logger.info(f"Starting Crypto Forecasting API in {config.current_env} environment")

model_trainer = ModelTrainer()
model_optimizer = ModelOptimizer()
handlers = EndpointHandlers(model_trainer, model_optimizer)

@app.get("/")
def read_root() -> Dict[str, str]:

    return {
        "message": "Crypto Forecasting API",
        "version": "1.0.0",
        "environment": config.current_env
    }

@app.post(config.api_endpoints.train_dt_endpoint)
def train_decision_tree_endpoint(
    request: pydantic_models.TrainingRequest
) -> Dict[str, Any]:

    return handlers.handle_training_request(model_trainer.train_decision_tree, "decision_tree", request)

@app.post(config.api_endpoints.train_rf_endpoint)
def train_random_forest_endpoint(
    request: pydantic_models.TrainingRequest
) -> Dict[str, Any]:

    return handlers.handle_training_request(model_trainer.train_random_forest, "random_forest", request)

@app.post(config.api_endpoints.train_xgb_endpoint)
def train_xgboost_endpoint(
    request: pydantic_models.TrainingRequest
) -> Dict[str, Any]:

    return handlers.handle_training_request(model_trainer.train_xgboost, "xgboost", request)

@app.post(config.api_endpoints.train_lasso_endpoint)
def train_lasso_endpoint(
    request: pydantic_models.TrainingRequest
) -> Dict[str, Any]:

    return handlers.handle_training_request(model_trainer.train_lasso, "lasso", request)

@app.post(config.api_endpoints.optimize_dt_endpoint)
def optimize_decision_tree_endpoint(
    request: pydantic_models.OptimizationRequest
) -> Dict[str, Any]:

    return handlers.handle_decision_tree_optimization(request)

@app.post(config.api_endpoints.optimize_rf_endpoint)
def optimize_random_forest_endpoint(
    request: pydantic_models.OptimizationRequest
) -> Dict[str, Any]:

    return handlers.handle_random_forest_optimization(request)

@app.post(config.api_endpoints.optimize_xgb_endpoint)
def optimize_xgboost_endpoint(
    request: pydantic_models.OptimizationRequest
) -> Dict[str, Any]:

    return handlers.handle_xgboost_optimization(request)

@app.post(config.api_endpoints.optimize_lasso_endpoint)
def optimize_lasso_endpoint(
    request: pydantic_models.OptimizationRequest
) -> Dict[str, Any]:

    return handlers.handle_lasso_optimization(request)

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=5000)
