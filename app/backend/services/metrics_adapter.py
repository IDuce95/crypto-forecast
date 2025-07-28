

import pandas as pd
from typing import Dict
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from logger import logger

class MetricsCalculatorAdapter:

    def __init__(self):

        if not os.environ.get('METRICS_ADAPTER_LOGGED'):
            logger.info("Initializing MetricsCalculatorAdapter")
            os.environ['METRICS_ADAPTER_LOGGED'] = 'true'
        self._setup_submodule_path()

    def _setup_submodule_path(self):

        current_dir = os.path.dirname(__file__)
        metrics_calc_path = os.path.join(current_dir, '../../metrics-calculator')
        metrics_calc_abs_path = os.path.abspath(metrics_calc_path)

        if metrics_calc_abs_path not in sys.path:
            sys.path.insert(0, metrics_calc_abs_path)
            logger.info(f"Added metrics-calculator path: {metrics_calc_abs_path}")

    def _import_calculate_metrics(self):

        try:
            from metrics_calculator.app import calculate_metrics
            return calculate_metrics
        except ImportError as e:
            logger.error(f"Failed to import metrics-calculator submodule: {e}")
            logger.warning("Submodule import failed - will use fallback implementation")
            return None

    def calculate_metrics_for_splits(
        self,
        y_true_train: pd.Series,
        y_pred_train: pd.Series,
        y_true_val: pd.Series,
        y_pred_val: pd.Series,
        y_true_test: pd.Series,
        y_pred_test: pd.Series,
        save_results: bool = False,
        results_file_name: str = "crypto_metrics",
        comment: str = "Crypto forecasting metrics"
    ) -> Dict[str, Dict[str, float]]:

        try:
            data = {
                'train': {
                    'y_real': y_true_train,
                    'y_pred': y_pred_train
                },
                'val': {
                    'y_real': y_true_val,
                    'y_pred': y_pred_val
                },
                'test': {
                    'y_real': y_true_test,
                    'y_pred': y_pred_test
                }
            }

            metadata = {
                'problem_type': 'regression',
                'metrics': ('mape', 'r2'),
                'results_file_name': results_file_name,
                'results_structure': 'multiple_rows',
                'comment': comment,
                'save': save_results
            }

            logger.info("Calculating metrics using metrics-calculator submodule")

            calculate_metrics = self._import_calculate_metrics()
            if calculate_metrics is None:
                logger.warning("Falling back to current metrics implementation")
                return self._fallback_metrics_calculation(
                    y_true_train, y_pred_train,
                    y_true_val, y_pred_val,
                    y_true_test, y_pred_test
                )

            results_df = calculate_metrics(data=data, metadata=metadata)

            metrics_dict = {
                'train': {
                    'mape': float(results_df.loc['train', 'mape']),
                    'r2': float(results_df.loc['train', 'r2'])
                },
                'validation': {
                    'mape': float(results_df.loc['val', 'mape']),
                    'r2': float(results_df.loc['val', 'r2'])
                },
                'test': {
                    'mape': float(results_df.loc['test', 'mape']),
                    'r2': float(results_df.loc['test', 'r2'])
                }
            }

            logger.info("Metrics calculated successfully using submodule")
            return metrics_dict

        except Exception as e:
            logger.error(f"Error calculating metrics with submodule: {str(e)}")
            logger.warning("Falling back to current metrics implementation")
            return self._fallback_metrics_calculation(
                y_true_train, y_pred_train,
                y_true_val, y_pred_val,
                y_true_test, y_pred_test
            )

    def calculate_metrics_for_single_split(
        self,
        y_true: pd.Series,
        y_pred: pd.Series,
        split_name: str = "train",
        save_results: bool = False,
        results_file_name: str = "crypto_metrics_single",
        comment: str = "Single split metrics"
    ) -> Dict[str, float]:

        try:
            split_mapping = {
                'train': 'train',
                'validation': 'val',
                'val': 'val',
                'test': 'test'
            }

            mapped_split = split_mapping.get(split_name, 'train')

            data = {
                mapped_split: {
                    'y_real': y_true,
                    'y_pred': y_pred
                }
            }

            metadata = {
                'problem_type': 'regression',
                'metrics': ('mape', 'r2'),
                'results_file_name': results_file_name,
                'results_structure': 'multiple_rows',
                'comment': comment,
                'save': save_results
            }

            logger.info(f"Calculating metrics for {split_name} using metrics-calculator submodule")

            calculate_metrics = self._import_calculate_metrics()
            if calculate_metrics is None:
                logger.warning("Falling back to current metrics implementation")
                return self._fallback_single_metrics_calculation(y_true, y_pred)

            results_df = calculate_metrics(data=data, metadata=metadata)

            metrics_dict = {
                'mape': float(results_df.loc[mapped_split, 'mape']),
                'r2': float(results_df.loc[mapped_split, 'r2'])
            }

            logger.info(f"Metrics calculated successfully for {split_name}")
            return metrics_dict

        except Exception as e:
            logger.error(f"Error calculating metrics with submodule for {split_name}: {str(e)}")
            logger.warning("Falling back to current metrics implementation")
            return self._fallback_single_metrics_calculation(y_true, y_pred)

    def _fallback_metrics_calculation(
        self,
        y_true_train: pd.Series,
        y_pred_train: pd.Series,
        y_true_val: pd.Series,
        y_pred_val: pd.Series,
        y_true_test: pd.Series,
        y_pred_test: pd.Series
    ) -> Dict[str, Dict[str, float]]:

        from sklearn.metrics import mean_absolute_percentage_error, r2_score

        return {
            'train': {
                'mape': mean_absolute_percentage_error(y_true_train, y_pred_train),
                'r2': r2_score(y_true_train, y_pred_train)
            },
            'validation': {
                'mape': mean_absolute_percentage_error(y_true_val, y_pred_val),
                'r2': r2_score(y_true_val, y_pred_val)
            },
            'test': {
                'mape': mean_absolute_percentage_error(y_true_test, y_pred_test),
                'r2': r2_score(y_true_test, y_pred_test)
            }
        }

    def _fallback_single_metrics_calculation(
        self,
        y_true: pd.Series,
        y_pred: pd.Series
    ) -> Dict[str, float]:

        from sklearn.metrics import mean_absolute_percentage_error, r2_score

        return {
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
