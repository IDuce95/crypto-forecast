

from typing import Tuple, Dict
import numpy as np
from sklearn.model_selection import train_test_split
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import config
from tools import load_and_validate_dataset, prepare_ml_data

class DataManager:

    def __init__(self):
        self.config = config

    def prepare_training_data(
        self,
        dataset_name: str,
        prediction_horizon: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        df = load_and_validate_dataset(dataset_name, self.config)
        X, y = prepare_ml_data(df, prediction_horizon=prediction_horizon)
        return train_test_split(X.values, y.values, test_size=0.2, random_state=42)

    def prepare_training_data_three_splits(
        self,
        dataset_name: str,
        prediction_horizon: int,
        split_ratios: Dict[str, float] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}

        df = load_and_validate_dataset(dataset_name, self.config)
        X, y = prepare_ml_data(df, prediction_horizon=prediction_horizon)

        test_size = split_ratios["test"]
        val_size_relative = split_ratios["validation"] / (split_ratios["train"] + split_ratios["validation"])

        X_temp, X_test, y_temp, y_test = train_test_split(
            X.values, y.values, test_size=test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_relative, random_state=42
        )

        return X_train, X_val, X_test, y_train, y_val, y_test

    def prepare_data_with_visualization_info(
        self,
        dataset_name: str,
        prediction_horizon: int,
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, any]:

        if split_ratios is None:
            split_ratios = {"train": 0.6, "validation": 0.2, "test": 0.2}
        df = load_and_validate_dataset(dataset_name, self.config)
        original_close_values = df['Close'].values
        original_dates = df.index

        X, y = prepare_ml_data(df, prediction_horizon=prediction_horizon)

        ml_data_length = len(X)
        original_close_for_viz = original_close_values[:ml_data_length]
        original_dates_for_viz = original_dates[:ml_data_length]

        total_len = len(X)
        train_end = int(total_len * split_ratios["train"])
        val_end = int(total_len * (split_ratios["train"] + split_ratios["validation"]))

        X_train = X.values[:train_end]
        X_val = X.values[train_end:val_end]
        X_test = X.values[val_end:]

        y_train = y.values[:train_end]
        y_val = y.values[train_end:val_end]
        y_test = y.values[val_end:]

        train_indices = list(range(train_end))
        val_indices = list(range(train_end, val_end))
        test_indices = list(range(val_end, total_len))

        train_end_idx = len(train_indices)
        val_end_idx = train_end_idx + len(val_indices)

        return {
            'X_train': X_train, 'X_val': X_val, 'X_test': X_test,
            'y_train': y_train, 'y_val': y_val, 'y_test': y_test,
            'idx_train': train_indices, 'idx_val': val_indices, 'idx_test': test_indices,
            'original_values': original_close_for_viz,
            'original_dates': original_dates_for_viz.strftime('%Y-%m-%d').tolist(),
            'train_end_idx': train_end_idx,
            'val_end_idx': val_end_idx,
            'split_ratios': split_ratios
        }
