

from typing import Dict, Any
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from logger import logger
from app.backend.services.service_registry import registry

class ModelTrainer:

    def __init__(self):
        self.data_manager = registry.get_data_manager()
        self.metrics_calculator = registry.get_metrics_calculator()

    def train_and_evaluate_model(
        self,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray
    ) -> Dict[str, float]:

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return self.metrics_calculator.calculate_metrics(y_test, y_pred)

    def train_decision_tree(
        self,
        max_depth: int = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        dataset_name: str = "Bitcoin",
        prediction_horizon: int = 1
    ) -> Dict[str, float]:

        logger.debug(
            f"Training decision tree with max_depth={max_depth}, "
            f"min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf} "
            f"on dataset={dataset_name}, prediction_horizon={prediction_horizon}"
        )

        X_train, X_test, y_train, y_test = self.data_manager.prepare_training_data(
            dataset_name, prediction_horizon
        )

        model = DecisionTreeRegressor(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=42
        )

        metrics = self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        logger.info(
            f"Decision tree trained successfully with MAPE: {metrics['mape']:.4f}, "
            f"R2: {metrics['r2']:.4f}"
        )
        return metrics

    def train_random_forest(
        self,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        dataset_name: str = "Bitcoin",
        prediction_horizon: int = 1
    ) -> Dict[str, float]:

        logger.debug(
            f"Training random forest with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, min_samples_split={min_samples_split} "
            f"on dataset={dataset_name}, prediction_horizon={prediction_horizon}"
        )

        X_train, X_test, y_train, y_test = self.data_manager.prepare_training_data(
            dataset_name, prediction_horizon
        )

        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )

        metrics = self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        logger.info(
            f"Random forest trained successfully with MAPE: {metrics['mape']:.4f}, "
            f"R2: {metrics['r2']:.4f}"
        )
        return metrics

    def train_xgboost(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        dataset_name: str = "Bitcoin",
        prediction_horizon: int = 1
    ) -> Dict[str, float]:

        logger.debug(
            f"Training XGBoost with n_estimators={n_estimators}, "
            f"max_depth={max_depth}, learning_rate={learning_rate} "
            f"on dataset={dataset_name}, prediction_horizon={prediction_horizon}"
        )

        X_train, X_test, y_train, y_test = self.data_manager.prepare_training_data(
            dataset_name, prediction_horizon
        )

        model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=42
        )

        metrics = self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        logger.info(
            f"XGBoost trained successfully with MAPE: {metrics['mape']:.4f}, "
            f"R2: {metrics['r2']:.4f}"
        )
        return metrics

    def train_lasso(
        self,
        alpha: float = 1.0,
        max_iter: int = 1000,
        tol: float = 0.0001,
        dataset_name: str = "Bitcoin",
        prediction_horizon: int = 1
    ) -> Dict[str, float]:

        logger.debug(
            f"Training Lasso with alpha={alpha}, max_iter={max_iter}, tol={tol} "
            f"on dataset={dataset_name}, prediction_horizon={prediction_horizon}"
        )

        X_train, X_test, y_train, y_test = self.data_manager.prepare_training_data(
            dataset_name, prediction_horizon
        )

        model = Lasso(
            alpha=alpha,
            max_iter=max_iter,
            tol=tol,
            random_state=42
        )

        metrics = self.train_and_evaluate_model(model, X_train, X_test, y_train, y_test)
        logger.info(
            f"Lasso trained successfully with MAPE: {metrics['mape']:.4f}, "
            f"R2: {metrics['r2']:.4f}"
        )
        return metrics
