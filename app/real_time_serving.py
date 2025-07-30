
import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from enum import Enum
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
import queue

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logger_manager import LoggerManager
from cache_manager import CacheManager
from ml_model_trainer import ModelTrainer
from deep_learning_models import DeepLearningTrainer
from advanced_feature_engineering import AdvancedFeatureEngineer


class ModelStatus(Enum):
    SINGLE = "single"
    BATCH = "batch"
    STREAMING = "streaming"


@dataclass
class ModelMetadata:
    symbol: str
    data: Union[Dict[str, Any], List[Dict[str, Any]]]
    model_id: Optional[str] = None
    prediction_type: PredictionType = PredictionType.SINGLE
    include_confidence: bool = True
    include_explanation: bool = False


@dataclass
class PredictionResponse:
    max_models_in_memory: int = 5
    model_cache_ttl: int = 3600  # seconds
    auto_reload_models: bool = True
    
    batch_size: int = 32
    max_workers: int = 4
    prediction_timeout: float = 5.0  # seconds
    
    enable_ab_testing: bool = False
    traffic_split: Dict[str, float] = None
    
    enable_performance_monitoring: bool = True
    metrics_collection_interval: int = 60  # seconds
    
    enable_real_time_features: bool = True
    feature_cache_ttl: int = 300  # seconds
    
    def __post_init__(self):
        if self.traffic_split is None:
            self.traffic_split = {"model_a": 0.5, "model_b": 0.5}


class ModelServing:
        Initialize model serving system
        
        Args:
            config: Serving configuration
        if self.config.enable_performance_monitoring:
            self.monitoring_thread = threading.Thread(
                target=self._performance_monitoring_loop,
                daemon=True
            )
            self.monitoring_thread.start()
            
        if self.config.auto_reload_models:
            self.model_update_thread = threading.Thread(
                target=self._model_update_loop,
                daemon=True
            )
            self.model_update_thread.start()
            
        self.logger.info("Background tasks started")
    
    def stop_background_tasks(self):
        Load model for serving
        
        Args:
            model_path: Path to model file
            model_id: Unique model identifier
            model_type: Type of model (ml, deep_learning)
            
        Returns:
            Success status
        Unload model from memory
        
        Args:
            model_id: Model identifier
            
        Returns:
            Success status
        if not self.model_metadata:
            return
            
        oldest_model = min(
            self.model_metadata.items(),
            key=lambda x: x[1].last_updated
        )[0]
        
        self.unload_model(oldest_model)
        self.logger.info(f"Evicted oldest model: {oldest_model}")
    
    def _select_model_for_request(self, request: PredictionRequest) -> str:
        if self.config.enable_real_time_features:
            cache_key = f"features_{request.symbol}_{hash(str(request.data))}"
            cached_features = self.cache_manager.get_cached_processed_data(cache_key)
            if cached_features is not None:
                return cached_features
        
        if isinstance(request.data, dict):
            df = pd.DataFrame([request.data])
        else:
            df = pd.DataFrame(request.data)
        
        if model_id in self.feature_engineers:
            engineer = self.feature_engineers[model_id]
            features, _ = engineer.transform_features(df)
        else:
            features = df
        
        if self.config.enable_real_time_features:
            self.cache_manager.cache_processed_data(
                cache_key, features, ttl=self.config.feature_cache_ttl
            )
        
        return features
    
    async def predict(self, request: PredictionRequest) -> PredictionResponse:
        start_time = time.time()
        
        try:
            model_id = self._select_model_for_request(request)
            model = self.models[model_id]
            
            features = self._prepare_features(request, model_id)
            
            if request.prediction_type == PredictionType.SINGLE:
                prediction = model.predict(features.iloc[-1:].values)[0]
                
                confidence = self._calculate_confidence(model, features, model_id)
                
            elif request.prediction_type == PredictionType.BATCH:
                prediction = model.predict(features.values).tolist()
                confidence = None
            else:
                raise ValueError(f"Unsupported prediction type: {request.prediction_type}")
            
            explanation = None
            if request.include_explanation:
                explanation = self._generate_explanation(model, features, model_id)
            
            processing_time = (time.time() - start_time) * 1000
            
            self.prediction_counts[model_id] += 1
            self.prediction_times[model_id].append(processing_time)
            
            return PredictionResponse(
                prediction=prediction,
                confidence=confidence if request.include_confidence else None,
                model_id=model_id,
                timestamp=datetime.now(),
                explanation=explanation,
                processing_time_ms=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"Prediction error: {e}")
            if 'model_id' in locals():
                self.error_counts[model_id] = self.error_counts.get(model_id, 0) + 1
            
            return PredictionResponse(
                prediction=None,
                model_id=getattr(locals().get('model_id'), 'model_id', 'unknown'),
                timestamp=datetime.now(),
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _calculate_confidence(self, model, features: pd.DataFrame, model_id: str) -> float:
        try:
            explanation = {
                "model_type": self.model_metadata[model_id].model_type,
                "feature_count": features.shape[1],
                "top_features": []
            }
            
            if hasattr(model, 'feature_importances_'):
                feature_names = self.model_metadata[model_id].feature_names
                if len(feature_names) == len(model.feature_importances_):
                    importance_pairs = list(zip(feature_names, model.feature_importances_))
                    top_features = sorted(importance_pairs, key=lambda x: x[1], reverse=True)[:5]
                    explanation["top_features"] = [
                        {"feature": name, "importance": float(importance)}
                        for name, importance in top_features
                    ]
            
            return explanation
            
        except Exception as e:
            self.logger.warning(f"Error generating explanation: {e}")
            return {"error": "Could not generate explanation"}
    
    def get_model_status(self) -> Dict[str, Any]:
        total_predictions = sum(self.prediction_counts.values())
        total_errors = sum(self.error_counts.values())
        
        all_times = []
        for times in self.prediction_times.values():
            all_times.extend(times)
        
        return {
            "total_predictions": total_predictions,
            "total_errors": total_errors,
            "error_rate": total_errors / max(total_predictions, 1),
            "avg_prediction_time_ms": np.mean(all_times) if all_times else 0,
            "p95_prediction_time_ms": np.percentile(all_times, 95) if all_times else 0,
            "throughput_per_minute": total_predictions / max(1, (time.time() - getattr(self, '_start_time', time.time())) / 60),
            "models_in_memory": len(self.models)
        }
    
    def _performance_monitoring_loop(self):
        while not self._shutdown_event.is_set():
            try:
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Error in model update loop: {e}")
                time.sleep(60)


model_serving = ModelServing(ServingConfig())
