

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.metrics import mean_squared_error, mean_absolute_error
from dataclasses import dataclass
from pathlib import Path
import pickle
import warnings
from datetime import datetime, timedelta
import hashlib
import json

from app.config import config
from app.logger import logger
from app.ml_preprocessor import DataPreprocessor, PreprocessedData
from app.ml_model_trainer import ModelTrainer, ModelResults, MultiModelTrainer

try:
    from app.cache_manager import cache_manager
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False
    cache_manager = None

@dataclass
class PredictionConfig:

    prediction_horizon: int = 1
    confidence_level: float = 0.95
    enable_ensemble: bool = True
    ensemble_method: str = "weighted_average"
    include_confidence_intervals: bool = True
    output_format: str = "dataframe"

@dataclass
class SinglePrediction:

    symbol: str
    timestamp: datetime
    predicted_value: float
    actual_value: Optional[float]
    confidence_lower: Optional[float]
    confidence_upper: Optional[float]
    model_used: str
    prediction_horizon: int
    metadata: Dict[str, Any]

@dataclass
class BatchPredictions:

    predictions: List[SinglePrediction]
    summary_stats: Dict[str, float]
    ensemble_weights: Optional[Dict[str, float]]
    metadata: Dict[str, Any]

@dataclass
class EnsemblePrediction:

    individual_predictions: Dict[str, float]
    ensemble_prediction: float
    weights: Dict[str, float]
    confidence_interval: Tuple[float, float]
    metadata: Dict[str, Any]

class PredictionGenerator:

    def __init__(self, prediction_config: PredictionConfig = None):

        if prediction_config is None:
            prediction_config = PredictionConfig(
                prediction_horizon=config.model_settings.prediction_horizon
            )
        
        self.config = prediction_config
        self.trained_models = {}
        self.ensemble_weights = {}
        
        logger.info(f"PredictionGenerator initialized")
    
    def load_trained_models(self, model_paths: Dict[str, Path]) -> None:

        logger.info("Loading trained models...")
        
        for model_name, model_path in model_paths.items():
            try:
                trainer = ModelTrainer()
                results = trainer.load_model(model_path)
                self.trained_models[model_name] = trainer
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
                continue
        
        logger.info(f"Loaded {len(self.trained_models)} models")
    
    def add_trained_model(self, model_name: str, trainer: ModelTrainer) -> None:

        self.trained_models[model_name] = trainer
        logger.info(f"Added model: {model_name}")
    
    def calculate_ensemble_weights(self, validation_data: PreprocessedData, 
                                 method: str = "performance_based") -> Dict[str, float]:

        if not self.trained_models:
            raise ValueError("No trained models available")
        
        logger.info(f"Calculating ensemble weights using {method} method")
        
        if method == "performance_based":
            model_scores = {}
            
            for model_name, trainer in self.trained_models.items():
                try:
                    val_pred = trainer.predict(validation_data.X_val)
                    mse = mean_squared_error(validation_data.y_val, val_pred)
                    model_scores[model_name] = 1.0 / (mse + 1e-10)
                    
                except Exception as e:
                    logger.warning(f"Error calculating score for {model_name}: {e}")
                    model_scores[model_name] = 0.0
            
            total_score = sum(model_scores.values())
            weights = {name: score / total_score for name, score in model_scores.items()}
            
        elif method == "equal":
            num_models = len(self.trained_models)
            weights = {name: 1.0 / num_models for name in self.trained_models.keys()}
            
        else:
            raise ValueError(f"Unknown ensemble method: {method}")
        
        self.ensemble_weights = weights
        logger.info(f"Ensemble weights calculated: {weights}")
        
        return weights
    
    def predict_single(self, features: pd.DataFrame, 
                      model_name: str = None) -> Dict[str, float]:

        if not self.trained_models:
            raise ValueError("No trained models available")
        
        predictions = {}
        
        if model_name:
            if model_name not in self.trained_models:
                raise ValueError(f"Model {model_name} not found")
            
            trainer = self.trained_models[model_name]
            pred = trainer.predict(features)
            predictions[model_name] = float(pred[0])
            
        else:
            for name, trainer in self.trained_models.items():
                try:
                    pred = trainer.predict(features)
                    predictions[name] = float(pred[0])
                except Exception as e:
                    logger.warning(f"Error predicting with {name}: {e}")
                    continue
        
        return predictions
    
    def predict_ensemble(self, features: pd.DataFrame, 
                        weights: Dict[str, float] = None) -> EnsemblePrediction:

        if not self.trained_models:
            raise ValueError("No trained models available")
        
        individual_preds = self.predict_single(features)
        
        if not individual_preds:
            raise ValueError("No valid predictions generated")
        
        if weights is None:
            weights = self.ensemble_weights
        
        if not weights:
            weights = {name: 1.0 / len(individual_preds) for name in individual_preds.keys()}
        
        if self.config.ensemble_method == "simple_average":
            ensemble_pred = np.mean(list(individual_preds.values()))
        elif self.config.ensemble_method == "weighted_average":
            weighted_sum = sum(weights.get(name, 0) * pred 
                             for name, pred in individual_preds.items())
            total_weight = sum(weights.get(name, 0) for name in individual_preds.keys())
            ensemble_pred = weighted_sum / total_weight if total_weight > 0 else 0
        else:
            raise ValueError(f"Unknown ensemble method: {self.config.ensemble_method}")
        
        pred_values = list(individual_preds.values())
        pred_std = np.std(pred_values)
        
        confidence_alpha = 1 - self.config.confidence_level
        z_score = 1.96
        
        confidence_lower = ensemble_pred - z_score * pred_std
        confidence_upper = ensemble_pred + z_score * pred_std
        
        metadata = {
            'ensemble_method': self.config.ensemble_method,
            'confidence_level': self.config.confidence_level,
            'num_models': len(individual_preds),
            'prediction_std': pred_std
        }
        
        return EnsemblePrediction(
            individual_predictions=individual_preds,
            ensemble_prediction=float(ensemble_pred),
            weights=weights,
            confidence_interval=(float(confidence_lower), float(confidence_upper)),
            metadata=metadata
        )
    
    def predict_time_series(self, symbol: str, preprocessor: DataPreprocessor,
                          start_date: str, end_date: str) -> BatchPredictions:

        logger.info(f"Generating time series predictions for {symbol}")
        
        raw_data = preprocessor.load_data(symbol)
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        prediction_dates = pd.date_range(start=start_dt, end=end_dt, freq='D')
        predictions = []
        
        for pred_date in prediction_dates:
            try:
                feature_data = raw_data[raw_data.index < pred_date]
                
                if len(feature_data) < preprocessor.config.feature_window:
                    logger.warning(f"Insufficient data for prediction on {pred_date}")
                    continue
                
                feature_set = preprocessor.engineer_features(feature_data.tail(100))
                
                if feature_set.engineered_features.empty:
                    continue
                
                latest_features = feature_set.engineered_features.tail(1)
                
                if preprocessor.scaler:
                    latest_features_scaled = pd.DataFrame(
                        preprocessor.scaler.transform(latest_features),
                        columns=latest_features.columns,
                        index=latest_features.index
                    )
                else:
                    latest_features_scaled = latest_features
                
                if preprocessor.feature_selector:
                    latest_features_final = pd.DataFrame(
                        preprocessor.feature_selector.transform(latest_features_scaled),
                        columns=[col for i, col in enumerate(latest_features_scaled.columns) 
                               if preprocessor.feature_selector.get_support()[i]],
                        index=latest_features_scaled.index
                    )
                else:
                    latest_features_final = latest_features_scaled
                
                if self.config.enable_ensemble and len(self.trained_models) > 1:
                    ensemble_pred = self.predict_ensemble(latest_features_final)
                    predicted_value = ensemble_pred.ensemble_prediction
                    confidence_lower, confidence_upper = ensemble_pred.confidence_interval
                    model_used = "ensemble"
                else:
                    model_name = list(self.trained_models.keys())[0]
                    individual_preds = self.predict_single(latest_features_final, model_name)
                    predicted_value = individual_preds[model_name]
                    confidence_lower = confidence_upper = None
                    model_used = model_name
                
                actual_value = None
                future_date = pred_date + timedelta(days=self.config.prediction_horizon)
                if future_date in raw_data.index:
                    actual_value = float(raw_data.loc[future_date, 'Close'])
                
                prediction = SinglePrediction(
                    symbol=symbol,
                    timestamp=pred_date,
                    predicted_value=predicted_value,
                    actual_value=actual_value,
                    confidence_lower=confidence_lower,
                    confidence_upper=confidence_upper,
                    model_used=model_used,
                    prediction_horizon=self.config.prediction_horizon,
                    metadata={}
                )
                
                predictions.append(prediction)
                
            except Exception as e:
                logger.warning(f"Error generating prediction for {pred_date}: {e}")
                continue
        
        pred_values = [p.predicted_value for p in predictions]
        actual_values = [p.actual_value for p in predictions if p.actual_value is not None]
        
        summary_stats = {
            'total_predictions': len(predictions),
            'predictions_with_actuals': len(actual_values),
            'mean_prediction': np.mean(pred_values) if pred_values else 0,
            'std_prediction': np.std(pred_values) if pred_values else 0,
            'min_prediction': np.min(pred_values) if pred_values else 0,
            'max_prediction': np.max(pred_values) if pred_values else 0
        }
        
        if len(actual_values) >= len(pred_values[:len(actual_values)]):
            pred_subset = pred_values[:len(actual_values)]
            summary_stats['mse'] = mean_squared_error(actual_values, pred_subset)
            summary_stats['mae'] = mean_absolute_error(actual_values, pred_subset)
        
        metadata = {
            'symbol': symbol,
            'start_date': start_date,
            'end_date': end_date,
            'prediction_horizon': self.config.prediction_horizon,
            'ensemble_enabled': self.config.enable_ensemble,
            'generation_timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Generated {len(predictions)} predictions for {symbol}")
        
        return BatchPredictions(
            predictions=predictions,
            summary_stats=summary_stats,
            ensemble_weights=self.ensemble_weights,
            metadata=metadata
        )
    
    def predict_next_value(self, symbol: str, preprocessor: DataPreprocessor) -> SinglePrediction:

        logger.info(f"Predicting next value for {symbol}")
        
        raw_data = preprocessor.load_data(symbol)
        
        feature_set = preprocessor.engineer_features(raw_data)
        
        latest_features = feature_set.engineered_features.tail(1)
        
        if preprocessor.scaler:
            latest_features_scaled = pd.DataFrame(
                preprocessor.scaler.transform(latest_features),
                columns=latest_features.columns,
                index=latest_features.index
            )
        else:
            latest_features_scaled = latest_features
        
        if preprocessor.feature_selector:
            latest_features_final = pd.DataFrame(
                preprocessor.feature_selector.transform(latest_features_scaled),
                columns=[col for i, col in enumerate(latest_features_scaled.columns) 
                       if preprocessor.feature_selector.get_support()[i]],
                index=latest_features_scaled.index
            )
        else:
            latest_features_final = latest_features_scaled
        
        if self.config.enable_ensemble and len(self.trained_models) > 1:
            ensemble_pred = self.predict_ensemble(latest_features_final)
            predicted_value = ensemble_pred.ensemble_prediction
            confidence_lower, confidence_upper = ensemble_pred.confidence_interval
            model_used = "ensemble"
        else:
            model_name = list(self.trained_models.keys())[0]
            individual_preds = self.predict_single(latest_features_final, model_name)
            predicted_value = individual_preds[model_name]
            confidence_lower = confidence_upper = None
            model_used = model_name
        
        prediction_timestamp = raw_data.index[-1] + timedelta(days=self.config.prediction_horizon)
        
        return SinglePrediction(
            symbol=symbol,
            timestamp=prediction_timestamp,
            predicted_value=predicted_value,
            actual_value=None,
            confidence_lower=confidence_lower,
            confidence_upper=confidence_upper,
            model_used=model_used,
            prediction_horizon=self.config.prediction_horizon,
            metadata={'latest_data_date': raw_data.index[-1].isoformat()}
        )
    
    def save_predictions(self, predictions: BatchPredictions, file_path: Path) -> None:

        pred_data = []
        for pred in predictions.predictions:
            pred_data.append({
                'symbol': pred.symbol,
                'timestamp': pred.timestamp,
                'predicted_value': pred.predicted_value,
                'actual_value': pred.actual_value,
                'confidence_lower': pred.confidence_lower,
                'confidence_upper': pred.confidence_upper,
                'model_used': pred.model_used,
                'prediction_horizon': pred.prediction_horizon
            })
        
        pred_df = pd.DataFrame(pred_data)
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        pred_df.to_csv(file_path, index=False)
        
        metadata_path = file_path.with_suffix('.metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump({
                'summary_stats': predictions.summary_stats,
                'ensemble_weights': predictions.ensemble_weights,
                'metadata': predictions.metadata
            }, f, indent=2)
        
        logger.info(f"Predictions saved to {file_path}")

if __name__ == "__main__":
