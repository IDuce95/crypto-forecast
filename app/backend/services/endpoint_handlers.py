

from fastapi import HTTPException
from typing import Dict, Any
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    from logger import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class EndpointHandlers:

    def __init__(self, model_trainer, model_optimizer):

        self.model_trainer = model_trainer
        self.model_optimizer = model_optimizer

    def create_endpoint_response(
        self,
        model_name: str,
        metrics: Dict[str, float],
        dataset_name: str,
        prediction_horizon: int
    ) -> Dict[str, Any]:

        return {
            "model": model_name,
            "metrics": metrics,
            "dataset": dataset_name,
            "prediction_horizon": prediction_horizon
        }

    def handle_training_request(
        self,
        trainer_func: Any,
        model_name: str,
        request: Any
    ) -> Dict[str, Any]:

        try:
            logger.info(f"Received {model_name} training request")
            if model_name == "decision_tree":
                metrics = trainer_func(
                    request.max_depth, request.min_samples_split, request.min_samples_leaf,
                    request.dataset_name, request.prediction_horizon
                )
            elif model_name == "random_forest":
                metrics = trainer_func(
                    request.n_estimators, request.max_depth, request.min_samples_split,
                    request.dataset_name, request.prediction_horizon
                )
            elif model_name == "xgboost":
                metrics = trainer_func(
                    request.n_estimators, request.max_depth, request.learning_rate,
                    request.dataset_name, request.prediction_horizon
                )
            elif model_name == "lasso":
                metrics = trainer_func(
                    request.alpha, request.max_iter, request.tol,
                    request.dataset_name, request.prediction_horizon
                )

            logger.info(f"{model_name} endpoint completed successfully")
            return self.create_endpoint_response(model_name, metrics, request.dataset_name, request.prediction_horizon)
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def handle_decision_tree_optimization(self, request) -> Dict[str, Any]:

        try:
            logger.info(
                f"Starting decision tree optimization for dataset={request.dataset_name}, "
                f"prediction_horizon={request.prediction_horizon}"
            )
            result = self.model_optimizer.optimize_decision_tree(
                request.dataset_name,
                request.prediction_horizon,
                request.dt_max_depth_min,
                request.dt_max_depth_max,
                request.dt_min_samples_split_min,
                request.dt_min_samples_split_max,
                request.dt_min_samples_leaf_min,
                request.dt_min_samples_leaf_max,
                init_points=request.init_points,
                n_iter=request.n_iter,
                split_ratios=request.split_ratios
            )
            logger.info("Decision tree optimization completed successfully")
            return {
                "model": "decision_tree",
                "dataset": request.dataset_name,
                "prediction_horizon": request.prediction_horizon,
                **result
            }
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def handle_random_forest_optimization(self, request) -> Dict[str, Any]:

        try:
            logger.info(
                f"Starting random forest optimization for dataset={request.dataset_name}, "
                f"prediction_horizon={request.prediction_horizon}"
            )
            result = self.model_optimizer.optimize_random_forest(
                request.dataset_name,
                request.prediction_horizon,
                request.rf_n_estimators_min,
                request.rf_n_estimators_max,
                request.rf_max_depth_min,
                request.rf_max_depth_max,
                request.rf_min_samples_split_min,
                request.rf_min_samples_split_max,
                init_points=request.init_points,
                n_iter=request.n_iter,
                split_ratios=request.split_ratios
            )
            logger.info("Random forest optimization completed successfully")
            return {
                "model": "random_forest",
                "dataset": request.dataset_name,
                "prediction_horizon": request.prediction_horizon,
                **result
            }
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def handle_xgboost_optimization(self, request) -> Dict[str, Any]:

        try:
            logger.info(
                f"Starting XGBoost optimization for dataset={request.dataset_name}, "
                f"prediction_horizon={request.prediction_horizon}"
            )
            result = self.model_optimizer.optimize_xgboost(
                request.dataset_name,
                request.prediction_horizon,
                request.xgb_n_estimators_min,
                request.xgb_n_estimators_max,
                request.xgb_max_depth_min,
                request.xgb_max_depth_max,
                request.xgb_learning_rate_min,
                request.xgb_learning_rate_max,
                init_points=request.init_points,
                n_iter=request.n_iter,
                split_ratios=request.split_ratios
            )
            logger.info("XGBoost optimization completed successfully")
            return {
                "model": "xgboost",
                "dataset": request.dataset_name,
                "prediction_horizon": request.prediction_horizon,
                **result
            }
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))

    def handle_lasso_optimization(self, request) -> Dict[str, Any]:

        try:
            logger.info(
                f"Starting Lasso optimization for dataset={request.dataset_name}, "
                f"prediction_horizon={request.prediction_horizon}"
            )
            result = self.model_optimizer.optimize_lasso(
                request.dataset_name,
                request.prediction_horizon,
                request.lasso_alpha_min,
                request.lasso_alpha_max,
                request.lasso_max_iter_min,
                request.lasso_max_iter_max,
                request.lasso_tol_min,
                request.lasso_tol_max,
                init_points=request.init_points,
                n_iter=request.n_iter,
                split_ratios=request.split_ratios
            )
            logger.info("Lasso optimization completed successfully")
            return {
                "model": "lasso",
                "dataset": request.dataset_name,
                "prediction_horizon": request.prediction_horizon,
                **result
            }
        except Exception as e:
            logger.error(e)
            raise HTTPException(status_code=500, detail=str(e))
