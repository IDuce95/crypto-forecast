import os
import pandas as pd
from typing import List, Tuple
from dynaconf import Dynaconf

from app.validation import DataValidator
from app.logger import logger

def get_available_datasets(
    config: Dynaconf
) -> List[str]:
    datasets_dir = config.data_settings.processed_data_path
    validator = DataValidator()
    available_datasets = []

    for file in os.listdir(datasets_dir):
        if file != config.other.gitkeep_filename and file.endswith('.csv'):
            dataset_name = file.split(".")[0]
            file_path = os.path.join(datasets_dir, file)

            try:
                validator.validate_file_exists(file_path)
                validator.validate_file_format(file_path)

                df = pd.read_csv(file_path)
                validator.validate_dataframe_structure(df, dataset_name)

                available_datasets.append(dataset_name)
                logger.debug(f"Dataset {dataset_name} passed validation")

            except Exception as e:
                logger.warning(f"Dataset {dataset_name} failed validation: {e}")
                continue

    logger.info(f"Found {len(available_datasets)} valid datasets")
    return available_datasets

def load_and_validate_dataset(
    dataset_name: str,
    config: Dynaconf
) -> pd.DataFrame:
    datasets_dir = config.data_settings.processed_data_path
    file_path = os.path.join(datasets_dir, f"{dataset_name}.csv")

    validator = DataValidator()

    validator.validate_file_exists(file_path)
    validator.validate_file_format(file_path)

    logger.info(f"Loading dataset: {dataset_name}")
    df = pd.read_csv(file_path)

    validator.validate_dataframe_structure(df, dataset_name)
    validator.validate_data_quality(df, dataset_name)
    validator.validate_data_length(df, dataset_name)

    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    df.sort_index(inplace=True)

    logger.info(f"Dataset {dataset_name} loaded and validated successfully")
    return df

def prepare_ml_data(
    df: pd.DataFrame,
    target_column: str = 'Close',
    prediction_horizon: int = 1
) -> Tuple[pd.DataFrame, pd.Series]:
    target = df[target_column].shift(-prediction_horizon)
    valid_indices = target.dropna().index
    features = df.drop(columns=[target_column]).loc[valid_indices]
    target = target.loc[valid_indices]

    logger.debug(f"Prepared ML data with {features.shape[0]} samples, {features.shape[1]} features, and prediction horizon of {prediction_horizon} days")
    return features, target
