from typing import Dict
import pandas as pd
from pathlib import Path

from app.config import config
from app.logger import logger

class DataValidationError(Exception):
    pass

class DataValidator:

    def __init__(self):
        self.required_columns = ['Date', 'High', 'Low', 'Open', 'Close']
        self.numeric_columns = ['High', 'Low', 'Open', 'Close']
        self.min_data_length_years = config.data_settings.min_data_length_years

    def validate_file_exists(
        self,
        file_path: str
    ) -> bool:
        path = Path(file_path)
        if not path.exists():
            raise DataValidationError(f"Data file does not exist: {file_path}")
        if not path.is_file():
            raise DataValidationError(f"Path is not a file: {file_path}")
        return True

    def validate_file_format(
        self,
        file_path: str
    ) -> bool:
        if not file_path.endswith('.csv'):
            raise DataValidationError(f"Only CSV files are supported, got: {file_path}")
        return True

    def validate_dataframe_structure(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating dataframe structure for dataset: {dataset_name}")

        if df.empty:
            raise DataValidationError(f"Dataset {dataset_name} is empty")

        missing_columns = [col for col in self.required_columns if col not in df.columns]
        if missing_columns:
            raise DataValidationError(
                f"Dataset {dataset_name} missing required columns: {missing_columns}"
            )

        logger.info(f"Dataframe structure validation passed for dataset: {dataset_name}")
        return True

    def validate_data_types(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating data types for dataset: {dataset_name}")

        if 'Date' in df.columns:
            try:
                pd.to_datetime(df['Date'])
            except Exception as e:
                raise DataValidationError(
                    f"Dataset {dataset_name}: Date column cannot be converted to datetime: {e}"
                )

        for col in self.numeric_columns:
            if col in df.columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        pd.to_numeric(df[col])
                    except Exception as e:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Column {col} cannot be converted to numeric: {e}"
                        )

        logger.info(f"Data types validation passed for dataset: {dataset_name}")
        return True

    def validate_data_quality(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating data quality for dataset: {dataset_name}")

        for col in self.numeric_columns:
            if col in df.columns:
                negative_values = df[df[col] < 0]
                if not negative_values.empty:
                    raise DataValidationError(
                        f"Dataset {dataset_name}: Column {col} contains negative values. "
                        f"Found {len(negative_values)} negative values."
                    )

                null_values = df[col].isnull().sum()
                if null_values > 0:
                    raise DataValidationError(
                        f"Dataset {dataset_name}: Column {col} contains {null_values} null values"
                    )

                zero_values = (df[col] == 0).sum()
                if zero_values > 0:
                    logger.warning(
                        f"Dataset {dataset_name}: Column {col} contains {zero_values} zero values"
                    )

        logger.info(f"Data quality validation passed for dataset: {dataset_name}")
        return True

    def validate_data_length(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating data length for dataset: {dataset_name}")

        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['Date'] = pd.to_datetime(df_copy['Date'])

            min_date = df_copy['Date'].min()
            max_date = df_copy['Date'].max()
            data_span = max_date - min_date

            min_required_days = self.min_data_length_years * 365

            if data_span.days < min_required_days:
                raise DataValidationError(
                    f"Dataset {dataset_name}: Data span is {data_span.days} days, "
                    f"but minimum required is {min_required_days} days ({self.min_data_length_years} years)"
                )

            logger.info(f"Data length validation passed for dataset: {dataset_name}. "
                        f"Data span: {data_span.days} days")

        return True

    def validate_data_completeness(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating data completeness for dataset: {dataset_name}")

        total_rows = len(df)

        for col in df.columns:
            missing_count = df[col].isnull().sum()
            missing_percentage = (missing_count / total_rows) * 100

            if missing_percentage > 5:
                raise DataValidationError(
                    f"Dataset {dataset_name}: Column {col} has {missing_percentage:.2f}% missing values. "
                    f"Maximum allowed is 5%"
                )
            elif missing_percentage > 0:
                logger.warning(
                    f"Dataset {dataset_name}: Column {col} has {missing_percentage:.2f}% missing values"
                )

        logger.info(f"Data completeness validation passed for dataset: {dataset_name}")
        return True

    def validate_price_relationships(
        self,
        df: pd.DataFrame,
        dataset_name: str = None
    ) -> bool:
        logger.debug(f"Validating price relationships for dataset: {dataset_name}")

        required_price_cols = ['High', 'Low', 'Open', 'Close']
        available_cols = [col for col in required_price_cols if col in df.columns]

        if len(available_cols) >= 2:
            for index, row in df.iterrows():
                if 'High' in available_cols and 'Low' in available_cols:
                    if row['High'] < row['Low']:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Row {index} has High ({row['High']}) < Low ({row['Low']})"
                        )

                if 'High' in available_cols and 'Open' in available_cols:
                    if row['Open'] > row['High']:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Row {index} has Open ({row['Open']}) > High ({row['High']})"
                        )

                if 'Low' in available_cols and 'Open' in available_cols:
                    if row['Open'] < row['Low']:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Row {index} has Open ({row['Open']}) < Low ({row['Low']})"
                        )

                if 'High' in available_cols and 'Close' in available_cols:
                    if row['Close'] > row['High']:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Row {index} has Close ({row['Close']}) > High ({row['High']})"
                        )

                if 'Low' in available_cols and 'Close' in available_cols:
                    if row['Close'] < row['Low']:
                        raise DataValidationError(
                            f"Dataset {dataset_name}: Row {index} has Close ({row['Close']}) < Low ({row['Low']})"
                        )

        logger.info(f"Price relationships validation passed for dataset: {dataset_name}")
        return True

    def validate_dataset_file(
        self,
        file_path: str,
        dataset_name: str = None
    ) -> bool:
        if dataset_name is None:
            dataset_name = Path(file_path).stem

        logger.info(f"Starting validation for dataset: {dataset_name}")

        try:
            self.validate_file_exists(file_path)
            self.validate_file_format(file_path)

            df = pd.read_csv(file_path)

            self.validate_dataframe_structure(df, dataset_name)
            self.validate_data_types(df, dataset_name)
            self.validate_data_quality(df, dataset_name)
            self.validate_data_length(df, dataset_name)
            self.validate_data_completeness(df, dataset_name)
            self.validate_price_relationships(df, dataset_name)

            logger.info(f"All validations passed for dataset: {dataset_name}")
            return True

        except DataValidationError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error during validation of dataset {dataset_name}: {e}")
            raise DataValidationError(f"Validation failed for dataset {dataset_name}: {e}")

    def validate_all_datasets(
        self,
        data_dir: str = None
    ) -> Dict[str, bool]:
        if data_dir is None:
            data_dir = config.data_settings.processed_data_path

        logger.info(f"Starting validation for all datasets in: {data_dir}")

        data_path = Path(data_dir)
        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return {}

        results = {}

        for csv_file in csv_files:
            dataset_name = csv_file.stem
            try:
                self.validate_dataset_file(str(csv_file), dataset_name)
                results[dataset_name] = True
                logger.info(f"Dataset {dataset_name}: PASSED")
            except DataValidationError as e:
                results[dataset_name] = False
                logger.error(f"Dataset {dataset_name}: FAILED - {e}")

        passed_count = sum(results.values())
        total_count = len(results)

        logger.info(f"Validation summary: {passed_count}/{total_count} datasets passed")

        return results

validator = DataValidator()
