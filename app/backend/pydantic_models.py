from pydantic import BaseModel
from typing import Optional, Dict

class DecisionTreeRequest(BaseModel):
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    dataset_name: str = "Bitcoin"
    prediction_horizon: int = 1

class RandomForestRequest(BaseModel):
    n_estimators: int = 100
    max_depth: Optional[int] = None
    min_samples_split: int = 2
    dataset_name: str = "Bitcoin"
    prediction_horizon: int = 1

class XGBoostRequest(BaseModel):
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    dataset_name: str = "Bitcoin"
    prediction_horizon: int = 1

class LassoRequest(BaseModel):
    alpha: float = 1.0
    max_iter: int = 1000
    tol: float = 0.0001
    dataset_name: str = "Bitcoin"
    prediction_horizon: int = 1

class OptimizationRequest(BaseModel):
    dataset_name: str = "Bitcoin"
    prediction_horizon: int = 1
    split_ratios: Optional[Dict[str, float]] = {"train": 0.6, "validation": 0.2, "test": 0.2}
    init_points: Optional[int] = 5
    n_iter: Optional[int] = 5
    dt_max_depth_min: Optional[int] = 2
    dt_max_depth_max: Optional[int] = 20
    dt_min_samples_split_min: Optional[int] = 2
    dt_min_samples_split_max: Optional[int] = 20
    dt_min_samples_leaf_min: Optional[int] = 1
    dt_min_samples_leaf_max: Optional[int] = 20
    rf_n_estimators_min: Optional[int] = 10
    rf_n_estimators_max: Optional[int] = 200
    rf_max_depth_min: Optional[int] = 2
    rf_max_depth_max: Optional[int] = 20
    rf_min_samples_split_min: Optional[int] = 2
    rf_min_samples_split_max: Optional[int] = 20
    xgb_n_estimators_min: Optional[int] = 10
    xgb_n_estimators_max: Optional[int] = 200
    xgb_max_depth_min: Optional[int] = 2
    xgb_max_depth_max: Optional[int] = 10
    xgb_learning_rate_min: Optional[float] = 0.01
    xgb_learning_rate_max: Optional[float] = 0.3
    lasso_alpha_min: Optional[float] = 0.001
    lasso_alpha_max: Optional[float] = 10.0
    lasso_max_iter_min: Optional[int] = 100
    lasso_max_iter_max: Optional[int] = 5000
    lasso_tol_min: Optional[float] = 0.0001
    lasso_tol_max: Optional[float] = 0.01
