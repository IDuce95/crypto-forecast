
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import HyperbandPruner
import numpy as np
import pandas as pd
from typing import Dict, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logger_manager import LoggerManager
from cache_manager import CacheManager
from deep_learning_models import DeepLearningTrainer, DeepLearningConfig
from ml_model_trainer import ModelTrainer, ModelConfig
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')


@dataclass
class OptimizationConfig:
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.logger = LoggerManager().logger
        self.cache_manager = CacheManager()
        
        if config.sampler == "tpe":
            self.sampler = TPESampler()
        elif config.sampler == "random":
            self.sampler = optuna.samplers.RandomSampler()
        elif config.sampler == "cmaes":
            self.sampler = optuna.samplers.CmaEsSampler()
        else:
            self.sampler = TPESampler()
            
        if config.enable_pruning:
            self.pruner = HyperbandPruner(
                min_resource=config.min_resource,
                max_resource=config.max_resource
            )
        else:
            self.pruner = optuna.pruners.NopPruner()
            
        self.logger.info(f"Hyperparameter optimizer initialized - {config.n_trials} trials")
    
    def create_study(self) -> optuna.Study:
        params = {}
        
        if model_type == "random_forest":
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 10, 200),
                "max_depth": trial.suggest_int("max_depth", 3, 20),
                "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "max_features": trial.suggest_categorical("max_features", ["auto", "sqrt", "log2"])
            })
        elif model_type == "gradient_boosting":
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0)
            })
        elif model_type == "xgboost":
            params.update({
                "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0)
            })
        elif model_type == "linear_regression":
            params.update({
                "fit_intercept": trial.suggest_categorical("fit_intercept", [True, False])
            })
        elif model_type == "svr":
            params.update({
                "kernel": trial.suggest_categorical("kernel", ["linear", "poly", "rbf"]),
                "C": trial.suggest_float("C", 0.1, 100.0, log=True),
                "gamma": trial.suggest_categorical("gamma", ["scale", "auto"])
            })
            if params["kernel"] == "poly":
                params["degree"] = trial.suggest_int("degree", 2, 5)
        
        return params
    
    def suggest_deep_learning_parameters(self, trial: optuna.Trial, model_type: str) -> Dict[str, Any]:
        return {
            "technical_indicators": trial.suggest_categorical("technical_indicators", [True, False]),
            "feature_scaling": trial.suggest_categorical("feature_scaling", ["standard", "minmax", "robust"]),
            "outlier_removal": trial.suggest_categorical("outlier_removal", [True, False]),
            "lag_features": trial.suggest_int("lag_features", 1, 10),
            "rolling_window": trial.suggest_int("rolling_window", 5, 30)
        }
    
    def evaluate_ml_model(self, params: Dict[str, Any], X: pd.DataFrame, y: pd.Series) -> float:
        try:
            dl_config = DeepLearningConfig(**params)
            
            trainer = DeepLearningTrainer(dl_config)
            
            split_idx = int(len(train_data) * (1 - self.config.test_size))
            train_split = train_data.iloc[:split_idx]
            val_split = train_data.iloc[split_idx:]
            
            results = trainer.train(train_split, val_split)
            
            if results["final_val_loss"] is not None:
                return results["final_val_loss"]
            else:
                return results["final_train_loss"]
                
        except Exception as e:
            self.logger.error(f"Error evaluating deep learning model: {e}")
            return float('inf')
    
    def optimize_ml_model(self, X: pd.DataFrame, y: pd.Series, model_type: str) -> optuna.Study:
        def objective(trial):
            params = self.suggest_deep_learning_parameters(trial, model_type)
            
            score = self.evaluate_deep_learning_model(params, train_data)
            
            return score
        
        study = self.create_study()
        study.optimize(
            objective,
            n_trials=self.config.n_trials,
            timeout=self.config.timeout,
            n_jobs=self.config.n_jobs
        )
        
        self.logger.info(f"Deep learning optimization completed - Best value: {study.best_value:.6f}")
        return study
    
    def get_optimization_results(self, study: optuna.Study) -> Dict[str, Any]:
        results = self.get_optimization_results(study)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
            
        self.logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: str) -> Dict[str, Any]:
