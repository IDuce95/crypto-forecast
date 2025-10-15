from pydantic import BaseModel, Field
from typing import Optional, Dict, List, Union
from datetime import datetime
from enum import Enum

class CryptocurrencyEnum(str, Enum):
    BITCOIN = "Bitcoin"
    ETHEREUM = "Ethereum"
    LITECOIN = "Litecoin"
    XRP = "XRP"
    DOGECOIN = "Dogecoin"
    MONERO = "Monero"
    STELLAR = "Stellar"
    NEM = "NEM"
    TETHER = "Tether"

class ModelTypeEnum(str, Enum):
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LASSO = "lasso"

class SplitMethodEnum(str, Enum):
    PERCENTAGE = "percentage"
    TEMPORAL = "temporal"

class TrainingRequest(BaseModel):
    symbol: CryptocurrencyEnum = Field(default=CryptocurrencyEnum.BITCOIN)
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.RANDOM_FOREST)
    prediction_horizon: int = Field(default=1, ge=1, le=30)
    split_method: SplitMethodEnum = Field(default=SplitMethodEnum.PERCENTAGE)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    validation_size: float = Field(default=0.2, ge=0.1, le=0.4)
    split_date: Optional[str] = Field(default=None)
    feature_window: int = Field(default=7, ge=1, le=30)
    enable_hyperparameter_tuning: bool = Field(default=False)
    hyperparameters: Optional[Dict] = Field(default=None)

class PredictionRequest(BaseModel):
    symbol: CryptocurrencyEnum = Field(default=CryptocurrencyEnum.BITCOIN)
    model_type: ModelTypeEnum = Field(default=ModelTypeEnum.RANDOM_FOREST)
    prediction_steps: int = Field(default=7, ge=1, le=90)
    use_latest_model: bool = Field(default=True)
    model_path: Optional[str] = Field(default=None)

class HyperparameterTuningRequest(BaseModel):
    symbol: CryptocurrencyEnum = Field(default=CryptocurrencyEnum.BITCOIN)
    model_types: List[ModelTypeEnum] = Field(default=[ModelTypeEnum.RANDOM_FOREST])
    tuning_iterations: int = Field(default=20, ge=5, le=100)
    cv_folds: int = Field(default=5, ge=3, le=10)
    test_size: float = Field(default=0.2, ge=0.1, le=0.4)
    validation_size: float = Field(default=0.2, ge=0.1, le=0.4)

class ModelComparisonRequest(BaseModel):
    symbol: CryptocurrencyEnum = Field(default=CryptocurrencyEnum.BITCOIN)
    model_types: List[ModelTypeEnum] = Field(default=[
        ModelTypeEnum.RANDOM_FOREST, 
        ModelTypeEnum.XGBOOST, 
        ModelTypeEnum.DECISION_TREE
    ])
    prediction_horizon: int = Field(default=1, ge=1, le=30)
    enable_tuning: bool = Field(default=False)

class MetricsResponse(BaseModel):
    mse: float
    mae: float
    rmse: float
    r2: float
    mape: float

class ModelResponse(BaseModel):
    symbol: str
    model_type: str
    training_metrics: MetricsResponse
    validation_metrics: MetricsResponse
    test_metrics: MetricsResponse
    model_path: str
    training_time: float
    feature_count: int
    prediction_horizon: int

class PredictionResponse(BaseModel):
    symbol: str
    model_type: str
    predictions: List[float]
    confidence_intervals: Optional[List[Dict[str, float]]] = None
    prediction_dates: List[str]
    model_metrics: MetricsResponse

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


class HyperparameterTuningResponse(BaseModel):
    message: str
    symbol: str
    models: List[str]
    iterations: int
