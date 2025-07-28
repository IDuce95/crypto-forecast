

import pandas as pd
import numpy as np
from typing import Dict
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .metrics_adapter import MetricsCalculatorAdapter
from app.logger import logger

class MetricsCalculator:

    _shared_adapter = None

    def __init__(self):

        if MetricsCalculator._shared_adapter is None:
            MetricsCalculator._shared_adapter = MetricsCalculatorAdapter()
            if not os.environ.get('METRICS_CALCULATOR_LOGGED'):
                logger.info("MetricsCalculator initialized with submodule adapter")
                os.environ['METRICS_CALCULATOR_LOGGED'] = 'true'
        self.adapter = MetricsCalculator._shared_adapter

    @classmethod
    def _get_adapter(cls):

        if cls._shared_adapter is None:
            cls._shared_adapter = MetricsCalculatorAdapter()
        return cls._shared_adapter

    @staticmethod
    def calculate_metrics(
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> Dict[str, float]:

        y_true_series = pd.Series(y_true)
        y_pred_series = pd.Series(y_pred)

        adapter = MetricsCalculator._get_adapter()

        return adapter.calculate_metrics_for_single_split(
            y_true=y_true_series,
            y_pred=y_pred_series,
            split_name="single",
            save_results=False
        )

    @staticmethod
    def calculate_all_metrics(
        y_train: np.ndarray,
        y_val: np.ndarray,
        y_test: np.ndarray,
        train_pred: np.ndarray,
        val_pred: np.ndarray,
        test_pred: np.ndarray
    ) -> Dict[str, Dict[str, float]]:

        y_train_series = pd.Series(y_train)
        y_val_series = pd.Series(y_val)
        y_test_series = pd.Series(y_test)
        train_pred_series = pd.Series(train_pred)
        val_pred_series = pd.Series(val_pred)
        test_pred_series = pd.Series(test_pred)

        adapter = MetricsCalculator._get_adapter()

        return adapter.calculate_metrics_for_splits(
            y_true_train=y_train_series,
            y_pred_train=train_pred_series,
            y_true_val=y_val_series,
            y_pred_val=val_pred_series,
            y_true_test=y_test_series,
            y_pred_test=test_pred_series,
            save_results=False
        )

    def calculate_metrics_with_save(
        self,
        y_true_train: pd.Series,
        y_pred_train: pd.Series,
        y_true_val: pd.Series,
        y_pred_val: pd.Series,
        y_true_test: pd.Series,
        y_pred_test: pd.Series,
        model_name: str = "model",
        save_results: bool = True
    ) -> Dict[str, Dict[str, float]]:

        logger.info(f"Calculating metrics for all splits using {model_name}")

        return self.adapter.calculate_metrics_for_splits(
            y_true_train=y_true_train,
            y_pred_train=y_pred_train,
            y_true_val=y_true_val,
            y_pred_val=y_pred_val,
            y_true_test=y_true_test,
            y_pred_test=y_pred_test,
            save_results=save_results,
            results_file_name=f"{model_name}_metrics",
            comment=f"Metrics for {model_name} model"
        )
