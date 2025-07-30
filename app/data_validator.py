

import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, field

from app.config import config
from app.logger import logger

@dataclass
class ValidationResult:

    is_valid: bool = True
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    data_info: Dict[str, Any] = field(default_factory=dict)

    def add_error(self, message: str) -> None:

        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:

        self.warnings.append(message)

    def __str__(self) -> str:

        status = "PASSED" if self.is_valid else "FAILED"
        result = f"Validation {status}\n"

        if self.errors:
            result += f"Errors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  - {error}\n"

        if self.warnings:
            result += f"Warnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  - {warning}\n"

        return result

class DataValidator:

    def __init__(self, min_data_length_years: Optional[int] = None):

        self.min_data_length_years = (min_data_length_years if min_data_length_years is not None
                                      else config.data_settings.min_data_length_years)
        self.required_columns = {
            config.data_settings.date_column: 'datetime',
            'Open': 'numeric',
            'High': 'numeric',
            'Low': 'numeric',
            'Close': 'numeric'
        }

        self.optional_columns = {
            'Volume': 'numeric',
            'Market Cap': 'numeric'
        }

        logger.info(f"DataValidator initialized with {self.min_data_length_years} years minimum data requirement")

    def validate_dataframe(self, df: pd.DataFrame, dataset_name: str = "Unknown") -> ValidationResult:

        logger.info(f"Starting validation for dataset: {dataset_name}")
        result = ValidationResult()

        self._validate_basic_structure(df, result)

        if not result.is_valid:
            logger.warning(f"Basic structure validation failed for {dataset_name}")
            return result

        self._validate_columns(df, result)

        self._validate_data_types(df, result)

        self._validate_missing_values(df, result)

        self._validate_datetime(df, result)

        self._validate_data_length(df, result)

        self._validate_statistics(df, result)

        self._collect_data_info(df, result, dataset_name)

        logger.info(f"Validation completed for {dataset_name}. Valid: {result.is_valid}")
        return result

    def validate_csv_file(self, file_path: Path, dataset_name: Optional[str] = None) -> ValidationResult:

        if dataset_name is None:
            dataset_name = file_path.stem

        logger.info(f"Validating CSV file: {file_path}")

        result = ValidationResult()

        if not file_path.exists():
            result.add_error(f"File does not exist: {file_path}")
            return result

        try:
            df = pd.read_csv(file_path)
            logger.debug(f"Successfully loaded CSV file with {len(df)} rows")
        except Exception as e:
            result.add_error(f"Failed to read CSV file: {str(e)}")
            return result

        return self.validate_dataframe(df, dataset_name)

    def _validate_basic_structure(self, df: pd.DataFrame, result: ValidationResult) -> None:

        if df.empty:
            result.add_error("DataFrame is empty")
            return

        if len(df.columns) == 0:
            result.add_error("DataFrame has no columns")
            return

        logger.debug(f"DataFrame has {len(df)} rows and {len(df.columns)} columns")

    def _validate_columns(self, df: pd.DataFrame, result: ValidationResult) -> None:

        missing_required = []

        for col_name in self.required_columns.keys():
            if col_name not in df.columns:
                missing_required.append(col_name)

        if missing_required:
            result.add_error(f"Missing required columns: {missing_required}")

        all_expected = set(self.required_columns.keys()) | set(self.optional_columns.keys())
        unexpected = set(df.columns) - all_expected

        if unexpected:
            result.add_warning(f"Unexpected columns found: {list(unexpected)}")

        logger.debug(f"Column validation completed. Required missing: {missing_required}")

    def _validate_data_types(self, df: pd.DataFrame, result: ValidationResult) -> None:

        for col_name, expected_type in self.required_columns.items():
            if col_name not in df.columns:
                continue

            if expected_type == 'numeric':
                try:
                    pd.to_numeric(df[col_name], errors='raise')
                except (ValueError, TypeError):
                    result.add_error(f"Column '{col_name}' contains non-numeric values")

            elif expected_type == 'datetime':
                try:
                    pd.to_datetime(df[col_name], errors='raise')
                except (ValueError, TypeError):
                    result.add_error(f"Column '{col_name}' contains invalid datetime values")

        for col_name, expected_type in self.optional_columns.items():
            if col_name in df.columns and expected_type == 'numeric':
                try:
                    pd.to_numeric(df[col_name], errors='raise')
                except (ValueError, TypeError):
                    result.add_warning(f"Optional column '{col_name}' contains non-numeric values")

    def _validate_missing_values(self, df: pd.DataFrame, result: ValidationResult) -> None:

        critical_columns = list(self.required_columns.keys())

        for col_name in critical_columns:
            if col_name not in df.columns:
                continue

            missing_count = df[col_name].isnull().sum()
            missing_pct = (missing_count / len(df)) * 100

            if missing_count > 0:
                if missing_pct > 10:
                    result.add_error(f"Column '{col_name}' has {missing_pct:.1f}% missing values ({missing_count} out of {len(df)})")
                elif missing_pct > 5:
                    result.add_warning(f"Column '{col_name}' has {missing_pct:.1f}% missing values ({missing_count} out of {len(df)})")
                else:
                    result.add_warning(f"Column '{col_name}' has {missing_count} missing values")

    def _validate_datetime(self, df: pd.DataFrame, result: ValidationResult) -> None:

        date_col = config.data_settings.date_column

        if date_col not in df.columns:
            return

        try:
            dates = pd.to_datetime(df[date_col])

            duplicates = dates.duplicated().sum()
            if duplicates > 0:
                result.add_error(f"Found {duplicates} duplicate dates")

            date_range = dates.max() - dates.min()
            result.add_warning(f"Date range: {dates.min()} to {dates.max()} ({date_range.days} days)")

            if len(dates) > 1:
                date_diff = dates.diff().dropna()
                most_common_freq = date_diff.mode()

                if len(most_common_freq) > 0:
                    expected_freq = most_common_freq.iloc[0]
                    large_gaps = date_diff[date_diff > expected_freq * 2]

                    if len(large_gaps) > 0:
                        result.add_warning(f"Found {len(large_gaps)} potential gaps in time series")

        except Exception as e:
            result.add_error(f"Failed to validate datetime column: {str(e)}")

    def _validate_data_length(self, df: pd.DataFrame, result: ValidationResult) -> None:

        date_col = config.data_settings.date_column

        if date_col not in df.columns:
            return

        try:
            dates = pd.to_datetime(df[date_col])
            data_span = dates.max() - dates.min()
            data_years = data_span.days / 365.25

            if data_years < self.min_data_length_years:
                result.add_error(
                    f"Data length ({data_years:.1f} years) is less than minimum requirement "
                    f"({self.min_data_length_years} years)"
                )
            else:
                result.add_warning(f"Data span: {data_years:.1f} years (meets requirement)")

        except Exception as e:
            result.add_error(f"Failed to validate data length: {str(e)}")

    def _validate_statistics(self, df: pd.DataFrame, result: ValidationResult) -> None:

        numeric_columns = ['Open', 'High', 'Low', 'Close']

        for col_name in numeric_columns:
            if col_name not in df.columns:
                continue

            try:
                col_data = pd.to_numeric(df[col_name], errors='coerce')

                negative_count = (col_data < 0).sum()
                if negative_count > 0:
                    result.add_error(f"Column '{col_name}' has {negative_count} negative values")

                zero_count = (col_data == 0).sum()
                if zero_count > 0:
                    result.add_warning(f"Column '{col_name}' has {zero_count} zero values")

                if not col_data.empty:
                    q99 = col_data.quantile(0.99)
                    q01 = col_data.quantile(0.01)
                    iqr = col_data.quantile(0.75) - col_data.quantile(0.25)

                    outliers = col_data[(col_data > q99 + 3 * iqr) | (col_data < q01 - 3 * iqr)]

                    if len(outliers) > 0:
                        result.add_warning(f"Column '{col_name}' has {len(outliers)} potential outliers")

            except Exception as e:
                result.add_warning(f"Failed to validate statistics for '{col_name}': {str(e)}")

        if all(col in df.columns for col in ['Open', 'High', 'Low', 'Close']):
            try:
                high = pd.to_numeric(df['High'], errors='coerce')
                low = pd.to_numeric(df['Low'], errors='coerce')
                open_price = pd.to_numeric(df['Open'], errors='coerce')
                close = pd.to_numeric(df['Close'], errors='coerce')

                invalid_high_low = (high < low).sum()
                if invalid_high_low > 0:
                    result.add_error(f"Found {invalid_high_low} records where High < Low")

                invalid_high_open = (high < open_price).sum()
                invalid_high_close = (high < close).sum()

                if invalid_high_open > 0:
                    result.add_error(f"Found {invalid_high_open} records where High < Open")
                if invalid_high_close > 0:
                    result.add_error(f"Found {invalid_high_close} records where High < Close")

                invalid_low_open = (low > open_price).sum()
                invalid_low_close = (low > close).sum()

                if invalid_low_open > 0:
                    result.add_error(f"Found {invalid_low_open} records where Low > Open")
                if invalid_low_close > 0:
                    result.add_error(f"Found {invalid_low_close} records where Low > Close")

            except Exception as e:
                result.add_warning(f"Failed to validate price relationships: {str(e)}")

    def _collect_data_info(self, df: pd.DataFrame, result: ValidationResult, dataset_name: str) -> None:

        result.data_info = {
            'dataset_name': dataset_name,
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'columns': list(df.columns),
            'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
        }

        date_col = config.data_settings.date_column
        if date_col in df.columns:
            try:
                dates = pd.to_datetime(df[date_col])
                result.data_info.update({
                    'date_range_start': dates.min().isoformat(),
                    'date_range_end': dates.max().isoformat(),
                    'date_span_days': (dates.max() - dates.min()).days,
                    'date_span_years': (dates.max() - dates.min()).days / 365.25
                })
            except Exception:
                pass

def validate_all_datasets(data_path: Path = None) -> Dict[str, ValidationResult]:

    if data_path is None:
        data_path = Path(config.data_settings.processed_data_path)

    logger.info(f"Validating all datasets in: {data_path}")

    validator = DataValidator()
    results = {}

    csv_files = list(data_path.glob("*.csv"))

    if not csv_files:
        logger.warning(f"No CSV files found in {data_path}")
        return results

    logger.info(f"Found {len(csv_files)} CSV files to validate")

    for csv_file in csv_files:
        dataset_name = csv_file.stem
        logger.info(f"Validating dataset: {dataset_name}")

        try:
            result = validator.validate_csv_file(csv_file, dataset_name)
            results[dataset_name] = result

            if result.is_valid:
                logger.info(f"✓ {dataset_name}: Validation PASSED")
            else:
                logger.warning(f"✗ {dataset_name}: Validation FAILED ({len(result.errors)} errors)")

        except Exception as e:
            print(f"Failed to validate {dataset_name}: {str(e)}")
            error_result = ValidationResult()
            error_result.add_error(f"Validation failed with exception: {str(e)}")
            results[dataset_name] = error_result

    valid_count = sum(1 for result in results.values() if result.is_valid)
    total_count = len(results)

    logger.info(f"Validation summary: {valid_count}/{total_count} datasets passed validation")

    return results

if __name__ == "__main__":
