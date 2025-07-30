from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from typing import Dict, Any, List
import uvicorn
import sys
import os
from pathlib import Path

app_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

try:
    from backend.pydantic_models import (
        TrainingRequest, PredictionRequest, HyperparameterTuningRequest,
        ModelComparisonRequest, ModelResponse, PredictionResponse, HyperparameterTuningResponse,
        MetricsResponse
    )
    from config import config
    from logger_manager import LoggerManager
    from ml_preprocessor import DataPreprocessor, PreprocessingConfig
    from ml_model_trainer import ModelTrainer, ModelConfig
    from ml_prediction_generator import PredictionGenerator
    from ml_pipeline import MLPipeline, PipelineConfig
    from mlflow_manager import MLflowManager
    from cache_manager import CacheManager
    from deep_learning_models import DeepLearningTrainer, DeepLearningConfig
    from real_time_serving import ModelServing, ServingConfig
    from real_time_serving import PredictionRequest as RTPredictionRequest, PredictionType
    from performance_monitoring import PerformanceMonitor, MonitoringConfig
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current working directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    raise

app = FastAPI(
    title="Crypto Forecasting API",
    description="Advanced API for cryptocurrency price forecasting with ML models",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

LoggerManager().logger.info(f"Starting Crypto Forecasting API v2.0 in {config.current_env} environment")

def convert_metrics(metrics):
    try:
        from cache_manager import cache_manager
        stats = cache_manager.get_cache_stats()
        health = cache_manager.health_check()
        
        return {
            "cache_stats": stats,
            "health": health,
            "timestamp": LoggerManager().logger.get_timestamp()
        }
    except Exception as e:
        LoggerManager().logger.error(f"Error getting cache stats: {e}")
        return {
            "cache_stats": {"enabled": False, "error": str(e)},
            "health": False,
            "timestamp": LoggerManager().logger.get_timestamp()
        }


@app.delete("/cache/invalidate")
def invalidate_cache(pattern: str = "crypto_forecast:*") -> Dict[str, Any]:
    try:
        config_dl = DeepLearningConfig(
            model_type=model_type,
            sequence_length=sequence_length,
            hidden_size=hidden_size,
            num_layers=num_layers,
            num_epochs=epochs
        )
        
        trainer = DeepLearningTrainer(config_dl)
        
        LoggerManager().logger.info(f"Deep learning {model_type.upper()} training started for {symbol}")
        
        return {
            "message": f"Deep learning {model_type.upper()} training started for {symbol}",
            "config": {
                "model_type": model_type,
                "sequence_length": sequence_length,
                "hidden_size": hidden_size,
                "num_layers": num_layers,
                "epochs": epochs
            },
            "status": "training_started"
        }
        
    except Exception as e:
        LoggerManager().logger.error(f"Error training deep learning model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/deep-learning/models")
async def list_deep_learning_models():
    try:
        if not hasattr(app.state, 'model_serving'):
            app.state.model_serving = ModelServing(ServingConfig())
        
        request = RTPredictionRequest(
            symbol=symbol,
            data=data,
            model_id=model_id,
            prediction_type=PredictionType.SINGLE,
            include_confidence=include_confidence
        )
        
        response = await app.state.model_serving.predict(request)
        
        return {
            "prediction": response.prediction,
            "confidence": response.confidence,
            "model_id": response.model_id,
            "timestamp": response.timestamp.isoformat(),
            "processing_time_ms": response.processing_time_ms
        }
        
    except Exception as e:
        LoggerManager().logger.error(f"Error in real-time prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/serving/status")
async def get_serving_status():
    try:
        if not hasattr(app.state, 'model_serving'):
            return {"message": "Model serving not initialized"}
        
        metrics = app.state.model_serving.get_performance_metrics()
        return metrics
        
    except Exception as e:
        LoggerManager().logger.error(f"Error getting serving metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/monitoring/health")
async def get_system_health():
    try:
        if not hasattr(app.state, 'performance_monitor'):
            app.state.performance_monitor = PerformanceMonitor(MonitoringConfig())
        
        metrics = app.state.performance_monitor.get_metrics_summary(hours)
        return metrics
        
    except Exception as e:
        LoggerManager().logger.error(f"Error getting monitoring metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=config.api_settings.host,
        port=config.api_settings.port,
        reload=config.api_settings.debug
    )
