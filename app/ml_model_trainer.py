

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any, Union
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
from bayes_opt import BayesianOptimization
from dataclasses import dataclass
from pathlib import Path
import pickle
import joblib
from datetime import datetime

from app.config import config
from app.logger import logger
from app.ml_preprocessor import PreprocessedData

@dataclass
class ModelConfig:

    model_type: str = "random_forest"
    cv_folds: int = 5
    enable_hyperparameter_tuning: bool = True
    tuning_iterations: int = 20
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

@dataclass
class ModelResults:

    model: Any
    best_params: Dict[str, Any]
    cv_scores: List[float]
    test_score: float
    validation_score: float
    training_score: float
    feature_importance: Optional[pd.Series]
    metadata: Dict[str, Any]

@dataclass
class TrainingMetrics:

    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float

class ModelTrainer:

    def __init__(self, model_config: ModelConfig = None):

        if model_config is None:
            model_config = ModelConfig(
                cv_folds=config.ml_pipeline.cross_validation_folds,
                enable_hyperparameter_tuning=config.ml_pipeline.enable_hyperparameter_tuning,
                random_state=config.model_settings.random_state
            )
        
        self.config = model_config
        self.model = None
        self.best_params = {}
        self.feature_importance = None
        
        logger.info(f"ModelTrainer initialized for {self.config.model_type}")
    
    def get_default_params(self, model_type: str) -> Dict[str, Any]:

        default_params = {
            "decision_tree": {
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.config.random_state
            },
            "random_forest": {
                "n_estimators": 100,
                "max_depth": 10,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": self.config.random_state,
                "n_jobs": -1
            },
            "xgboost": {
                "n_estimators": 100,
                "max_depth": 6,
                "learning_rate": 0.1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "random_state": self.config.random_state,
                "n_jobs": -1
            },
            "lasso": {
                "alpha": 1.0,
                "max_iter": 1000,
                "random_state": self.config.random_state
            }
        }
        
        return default_params.get(model_type, {})
    
    def get_param_bounds(self, model_type: str) -> Dict[str, Tuple[float, float]]:

        param_bounds = {
            "decision_tree": {
                "max_depth": (3, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            },
            "random_forest": {
                "n_estimators": (50, 200),
                "max_depth": (3, 20),
                "min_samples_split": (2, 20),
                "min_samples_leaf": (1, 10)
            },
            "xgboost": {
                "n_estimators": (50, 200),
                "max_depth": (3, 10),
                "learning_rate": (0.01, 0.3),
                "subsample": (0.6, 1.0),
                "colsample_bytree": (0.6, 1.0)
            },
            "lasso": {
                "alpha": (0.001, 10.0)
            }
        }
        
        return param_bounds.get(model_type, {})
    
    def create_model(self, model_type: str, **params) -> Any:

        default_params = self.get_default_params(model_type)
        model_params = {**default_params, **params}
        
        if model_type == "decision_tree":
            for key in ["max_depth", "min_samples_split", "min_samples_leaf"]:
                if key in model_params:
                    model_params[key] = int(model_params[key])
            return DecisionTreeRegressor(**model_params)
        
        elif model_type == "random_forest":
            for key in ["n_estimators", "max_depth", "min_samples_split", "min_samples_leaf"]:
                if key in model_params:
                    model_params[key] = int(model_params[key])
            return RandomForestRegressor(**model_params)
        
        elif model_type == "xgboost":
            for key in ["n_estimators", "max_depth"]:
                if key in model_params:
                    model_params[key] = int(model_params[key])
            return xgb.XGBRegressor(**model_params)
        
        elif model_type == "lasso":
            return Lasso(**model_params)
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> TrainingMetrics:

        mse = mean_squared_error(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        return TrainingMetrics(
            mse=mse,
            mae=mae,
            rmse=rmse,
            r2=r2,
            mape=mape
        )
    
    def objective_function(self, X: pd.DataFrame, y: pd.Series, 
                          model_type: str, **params) -> float:

        try:
            model = self.create_model(model_type, **params)
            
            tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
            cv_scores = cross_val_score(
                model, X, y, 
                cv=tscv, 
                scoring='neg_mean_squared_error',
                n_jobs=-1
            )
            
            return cv_scores.mean()
            
        except Exception as e:
            logger.warning(f"Error in objective function: {e}")
            return -np.inf
    
    def optimize_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:

        if not self.config.enable_hyperparameter_tuning:
            return self.get_default_params(self.config.model_type)
        
        logger.info(f"Starting hyperparameter optimization for {self.config.model_type}")
        
        param_bounds = self.get_param_bounds(self.config.model_type)
        
        if not param_bounds:
            logger.warning(f"No parameter bounds defined for {self.config.model_type}")
            return self.get_default_params(self.config.model_type)
        
        def objective(**params):
            return self.objective_function(X, y, self.config.model_type, **params)
        
        optimizer = BayesianOptimization(
            f=objective,
            pbounds=param_bounds,
            random_state=self.config.random_state,
            verbose=1
        )
        
        optimizer.maximize(
            init_points=5,
            n_iter=self.config.tuning_iterations
        )
        
        best_params = optimizer.max['params']
        
        logger.info(f"Best parameters found: {best_params}")
        logger.info(f"Best CV score: {optimizer.max['target']:.4f}")
        
        return best_params
    
    def train_model(self, preprocessed_data: PreprocessedData) -> ModelResults:

        logger.info(f"Training {self.config.model_type} model")
        
        X_train = preprocessed_data.X_train
        y_train = preprocessed_data.y_train
        X_val = preprocessed_data.X_val
        y_val = preprocessed_data.y_val
        X_test = preprocessed_data.X_test
        y_test = preprocessed_data.y_test
        
        self.best_params = self.optimize_hyperparameters(X_train, y_train)
        
        self.model = self.create_model(self.config.model_type, **self.best_params)
        self.model.fit(X_train, y_train)
        
        tscv = TimeSeriesSplit(n_splits=self.config.cv_folds)
        cv_scores = cross_val_score(
            self.model, X_train, y_train,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        test_pred = self.model.predict(X_test)
        
        training_score = -mean_squared_error(y_train, train_pred)
        validation_score = -mean_squared_error(y_val, val_pred)
        test_score = -mean_squared_error(y_test, test_pred)
        
        train_metrics = self.calculate_metrics(y_train.values, train_pred)
        val_metrics = self.calculate_metrics(y_val.values, val_pred)
        test_metrics = self.calculate_metrics(y_test.values, test_pred)
        
        feature_importance = None
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = pd.Series(
                self.model.feature_importances_,
                index=X_train.columns,
                name='feature_importance'
            ).sort_values(ascending=False)
            self.feature_importance = feature_importance
        
        metadata = {
            'model_type': self.config.model_type,
            'training_timestamp': datetime.now().isoformat(),
            'cv_folds': self.config.cv_folds,
            'hyperparameter_tuning': self.config.enable_hyperparameter_tuning,
            'train_metrics': train_metrics,
            'validation_metrics': val_metrics,
            'test_metrics': test_metrics,
            'data_shape': {
                'train': X_train.shape,
                'val': X_val.shape,
                'test': X_test.shape
            },
            'best_params': self.best_params
        }
        
        logger.info(f"Model training completed")
        logger.info(f"Train MSE: {train_metrics.mse:.4f}")
        logger.info(f"Validation MSE: {val_metrics.mse:.4f}")
        logger.info(f"Test MSE: {test_metrics.mse:.4f}")
        
        return ModelResults(
            model=self.model,
            best_params=self.best_params,
            cv_scores=cv_scores,
            test_score=test_score,
            validation_score=validation_score,
            training_score=training_score,
            feature_importance=feature_importance,
            metadata=metadata
        )
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:

        if self.model is None:
            raise ValueError("Model not trained yet. Call train_model() first.")
        
        return self.model.predict(X)
    
    def save_model(self, file_path: Path, model_results: ModelResults) -> None:

        model_data = {
            'model': model_results.model,
            'best_params': model_results.best_params,
            'feature_importance': model_results.feature_importance,
            'metadata': model_results.metadata,
            'config': self.config
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {file_path}")
    
    def load_model(self, file_path: Path) -> ModelResults:

        with open(file_path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.best_params = model_data['best_params']
        self.feature_importance = model_data['feature_importance']
        self.config = model_data['config']
        
        logger.info(f"Model loaded from {file_path}")
        
        return ModelResults(
            model=self.model,
            best_params=self.best_params,
            cv_scores=[],
            test_score=0,
            validation_score=0,
            training_score=0,
            feature_importance=self.feature_importance,
            metadata=model_data['metadata']
        )

class MultiModelTrainer:

    def __init__(self, model_types: List[str] = None):

        if model_types is None:
            model_types = ["decision_tree", "random_forest", "xgboost", "lasso"]
        
        self.model_types = model_types
        self.trainers = {}
        self.results = {}
        
        logger.info(f"MultiModelTrainer initialized with models: {model_types}")
    
    def train_all_models(self, preprocessed_data: PreprocessedData) -> Dict[str, ModelResults]:

        logger.info("Training all models...")
        
        for model_type in self.model_types:
            logger.info(f"Training {model_type}...")
            
            config = ModelConfig(model_type=model_type)
            trainer = ModelTrainer(config)
            
            try:
                results = trainer.train_model(preprocessed_data)
                self.trainers[model_type] = trainer
                self.results[model_type] = results
                
                logger.info(f"{model_type} training completed successfully")
                
            except Exception as e:
                logger.error(f"Error training {model_type}: {e}")
                continue
        
        logger.info("All models training completed")
        return self.results
    
    def compare_models(self) -> pd.DataFrame:

        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        comparison_data = []
        
        for model_type, results in self.results.items():
            test_metrics = results.metadata['test_metrics']
            val_metrics = results.metadata['validation_metrics']
            
            comparison_data.append({
                'model': model_type,
                'test_mse': test_metrics.mse,
                'test_mae': test_metrics.mae,
                'test_rmse': test_metrics.rmse,
                'test_r2': test_metrics.r2,
                'test_mape': test_metrics.mape,
                'val_mse': val_metrics.mse,
                'val_mae': val_metrics.mae,
                'val_rmse': val_metrics.rmse,
                'val_r2': val_metrics.r2,
                'val_mape': val_metrics.mape,
                'cv_score_mean': results.cv_scores.mean(),
                'cv_score_std': results.cv_scores.std()
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('test_mse')
        
        return comparison_df
    
    def get_best_model(self, metric: str = 'test_mse') -> Tuple[str, ModelResults]:

        if not self.results:
            raise ValueError("No models trained yet. Call train_all_models() first.")
        
        comparison_df = self.compare_models()
        best_model_name = comparison_df.iloc[0]['model']
        
        return best_model_name, self.results[best_model_name]

if __name__ == "__main__":
