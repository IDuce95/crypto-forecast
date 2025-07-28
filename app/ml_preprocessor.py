import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from dataclasses import dataclass
from pathlib import Path
import pickle
import joblib

from app.config import config
from app.logger import logger
from app.database.dal import CryptocurrencyDataDAL

@dataclass
class PreprocessingConfig:
    test_size: float = 0.2
    validation_size: float = 0.2
    feature_window: int = 7
    target_column: str = "Close"
    prediction_horizon: int = 1
    scaling_method: str = "standard"
    imputation_method: str = "simple"
    enable_feature_selection: bool = True
    max_features: int = 50
    outlier_threshold: float = 3.0
    random_state: int = 42

@dataclass
class FeatureSet:
    raw_features: pd.DataFrame
    engineered_features: pd.DataFrame
    target: pd.Series
    feature_names: List[str]
    target_name: str
    metadata: Dict[str, Any]

@dataclass
class PreprocessedData:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series
    scaler: Any
    feature_selector: Any
    metadata: Dict[str, Any]

class TechnicalIndicators:

    @staticmethod
    def moving_averages(data: pd.Series, windows: List[int] = [5, 10, 20, 50]) -> pd.DataFrame:
        ma_features = pd.DataFrame(index=data.index)

        for window in windows:
            ma_features[f'MA_{window}'] = data.rolling(window=window).mean()
            ma_features[f'MA_{window}_ratio'] = data / ma_features[f'MA_{window}']

        return ma_features

    @staticmethod
    def exponential_moving_averages(data: pd.Series, spans: List[int] = [12, 26]) -> pd.DataFrame:
        ema_features = pd.DataFrame(index=data.index)

        for span in spans:
            ema_features[f'EMA_{span}'] = data.ewm(span=span).mean()
            ema_features[f'EMA_{span}_ratio'] = data / ema_features[f'EMA_{span}']

        return ema_features

    @staticmethod
    def rsi(data: pd.Series, window: int = 14) -> pd.Series:
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def macd(data: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
        ema_fast = data.ewm(span=fast).mean()
        ema_slow = data.ewm(span=slow).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line

        return pd.DataFrame({
            'MACD': macd_line,
            'MACD_signal': signal_line,
            'MACD_histogram': histogram
        }, index=data.index)

    @staticmethod
    def bollinger_bands(data: pd.Series, window: int = 20, num_std: float = 2) -> pd.DataFrame:
        ma = data.rolling(window=window).mean()
        std = data.rolling(window=window).std()

        upper_band = ma + (std * num_std)
        lower_band = ma - (std * num_std)

        return pd.DataFrame({
            'BB_upper': upper_band,
            'BB_middle': ma,
            'BB_lower': lower_band,
            'BB_width': upper_band - lower_band,
            'BB_position': (data - lower_band) / (upper_band - lower_band)
        }, index=data.index)

    @staticmethod
    def stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series,
                            k_window: int = 14, d_window: int = 3) -> pd.DataFrame:
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()

        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()

        return pd.DataFrame({
            'Stoch_K': k_percent,
            'Stoch_D': d_percent
        }, index=close.index)

class DataPreprocessor:

    def __init__(self, preprocessing_config: PreprocessingConfig = None):
        if preprocessing_config is None:
            preprocessing_config = PreprocessingConfig(
                test_size=config.model_settings.default_test_size,
                validation_size=config.model_settings.default_validation_size,
                feature_window=config.model_settings.feature_engineering_window,
                target_column=config.model_settings.target_column,
                prediction_horizon=config.model_settings.prediction_horizon,
                random_state=config.model_settings.random_state
            )

        self.config = preprocessing_config
        self.indicators = TechnicalIndicators()
        self.scaler = None
        self.feature_selector = None
        self.feature_names = []

        logger.info(f"DataPreprocessor initialized with config: {self.config}")

    def load_data(self, symbol: str) -> pd.DataFrame:

        dal = CryptocurrencyDataDAL()

        try:
            df = dal.get_data_by_symbol(symbol)

            if not df.empty:
                if 'Date' in df.columns:
                    df['Date'] = pd.to_datetime(df['Date'])
                    df.set_index('Date', inplace=True)

                df.sort_index(inplace=True)
                logger.info(f"Loaded {len(df)} records for {symbol} from database")
                return df

        except Exception as e:
            logger.warning(f"Could not load data from database for {symbol}: {e}")

        try:
            csv_path = Path(__file__).parent.parent / "data" / "processed" / f"{symbol}.csv"

            if not csv_path.exists():
                raise ValueError(f"No data found for symbol: {symbol} (neither in database nor CSV)")

            df = pd.read_csv(csv_path)
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            df.sort_index(inplace=True)

            required_columns = ['Open', 'High', 'Low', 'Close']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

            logger.info(f"Loaded {len(df)} records for {symbol} from CSV")
            return df

        except Exception as e:
            logger.error(f"Error loading data for {symbol}: {e}")
            raise

    def engineer_features(self, data: pd.DataFrame) -> FeatureSet:

        logger.info("Starting feature engineering...")

        features = pd.DataFrame(index=data.index)

        features['returns'] = data['Close'].pct_change()
        features['log_returns'] = np.log(data['Close'] / data['Close'].shift(1))
        features['price_range'] = (data['High'] - data['Low']) / data['Close']
        features['price_change'] = (data['Close'] - data['Open']) / data['Open']

        if 'Volume' in data.columns:
            features['volume_ma_ratio'] = data['Volume'] / data['Volume'].rolling(20).mean()
            features['price_volume'] = data['Close'] * data['Volume']

        windows = [5, 10] if len(data) < 1000 else [5, 10, 20, 50]
        ma_features = self.indicators.moving_averages(data['Close'], windows)
        features = pd.concat([features, ma_features], axis=1)

        ema_features = self.indicators.exponential_moving_averages(data['Close'])
        features = pd.concat([features, ema_features], axis=1)

        if len(data) > 20:
            features['RSI'] = self.indicators.rsi(data['Close'])

        if len(data) > 30:
            macd_features = self.indicators.macd(data['Close'])
            features = pd.concat([features, macd_features], axis=1)

        if len(data) > 25:
            bb_features = self.indicators.bollinger_bands(data['Close'])
            features = pd.concat([features, bb_features], axis=1)

        if all(col in data.columns for col in ['High', 'Low', 'Close']) and len(data) > 20:
            stoch_features = self.indicators.stochastic_oscillator(
                data['High'], data['Low'], data['Close']
            )
            features = pd.concat([features, stoch_features], axis=1)

        for lag in range(1, self.config.feature_window + 1):
            features[f'close_lag_{lag}'] = data['Close'].shift(lag)
            features[f'returns_lag_{lag}'] = features['returns'].shift(lag)
            if 'Volume' in data.columns:
                features[f'volume_lag_{lag}'] = data['Volume'].shift(lag)

        windows_stats = [5, 10] if len(data) < 1000 else [5, 10, 20]
        for window in windows_stats:
            features[f'close_std_{window}'] = data['Close'].rolling(window).std()
            features[f'returns_std_{window}'] = features['returns'].rolling(window).std()
            features[f'close_min_{window}'] = data['Close'].rolling(window).min()
            features[f'close_max_{window}'] = data['Close'].rolling(window).max()

        target = data[self.config.target_column].shift(-self.config.prediction_horizon)

        features_nan_count = features.isnull().sum().sum()
        target_nan_count = target.isnull().sum()
        logger.info(f"Before cleanup - Features NaN count: {features_nan_count}, Target NaN count: {target_nan_count}")
        logger.info(f"Features shape before cleanup: {features.shape}, Target shape: {target.shape}")

        target_valid = target.dropna()
        features_aligned = features.loc[target_valid.index]

        final_valid_idx = features_aligned.dropna(how='all').index
        features_clean = features_aligned.loc[final_valid_idx]
        target_clean = target_valid.loc[final_valid_idx]

        if len(features_clean) < 100:
            logger.warning(f"Only {len(features_clean)} valid samples, trying forward fill...")
            features_clean = features_aligned.ffill().dropna()
            target_clean = target_valid.loc[features_clean.index]

        feature_names = features_clean.columns.tolist()

        metadata = {
            'original_shape': data.shape,
            'engineered_shape': features_clean.shape,
            'feature_count': len(feature_names),
            'target_column': self.config.target_column,
            'prediction_horizon': self.config.prediction_horizon,
            'feature_window': self.config.feature_window,
            'nan_handling': {
                'original_features_nans': features_nan_count,
                'original_target_nans': target_nan_count,
                'final_valid_samples': len(features_clean)
            }
        }

        logger.info(f"Feature engineering completed: {len(feature_names)} features created, {len(features_clean)} valid samples")

        return FeatureSet(
            raw_features=data,
            engineered_features=features_clean,
            target=target_clean,
            feature_names=feature_names,
            target_name=self.config.target_column,
            metadata=metadata
        )

    def split_data_temporal(self, features: pd.DataFrame, target: pd.Series,
                           split_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:

        logger.info("Performing temporal data split")

        if split_date:
            split_timestamp = pd.to_datetime(split_date)
            test_mask = features.index >= split_timestamp
            train_val_mask = features.index < split_timestamp

            train_val_data = features.loc[train_val_mask]
            val_split_idx = int(len(train_val_data) * (1 - self.config.validation_size))

            X_train = train_val_data.iloc[:val_split_idx]
            X_val = train_val_data.iloc[val_split_idx:]
            X_test = features.loc[test_mask]

            y_train = target.loc[X_train.index]
            y_val = target.loc[X_val.index]
            y_test = target.loc[X_test.index]
        else:
            n_samples = len(features)

            test_size = int(n_samples * self.config.test_size)
            test_start_idx = n_samples - test_size

            val_size = int((n_samples - test_size) * self.config.validation_size)
            val_start_idx = test_start_idx - val_size

            train_end_idx = val_start_idx

            X_train = features.iloc[:train_end_idx]
            X_val = features.iloc[val_start_idx:test_start_idx]
            X_test = features.iloc[test_start_idx:]

            y_train = target.iloc[:train_end_idx]
            y_val = target.iloc[val_start_idx:test_start_idx]
            y_test = target.iloc[test_start_idx:]

        logger.info(f"Temporal split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def split_data_percentage(self, features: pd.DataFrame, target: pd.Series) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:

        return self.split_data_temporal(features, target, split_date=None)

    def handle_outliers(self, data: pd.DataFrame) -> pd.DataFrame:

        logger.info("Handling outliers...")

        numeric_columns = data.select_dtypes(include=[np.number]).columns

        for col in numeric_columns:
            z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
            outlier_mask = z_scores > self.config.outlier_threshold

            if outlier_mask.sum() > 0:
                median_value = data[col].median()
                data.loc[outlier_mask, col] = median_value
                logger.info(f"Replaced {outlier_mask.sum()} outliers in {col}")

        return data

    def scale_features(self, X_train: pd.DataFrame, X_val: pd.DataFrame,
                      X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        logger.info(f"Scaling features using {self.config.scaling_method} method")

        if self.config.scaling_method == "standard":
            self.scaler = StandardScaler()
        elif self.config.scaling_method == "minmax":
            self.scaler = MinMaxScaler()
        elif self.config.scaling_method == "robust":
            self.scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {self.config.scaling_method}")

        X_train_scaled = pd.DataFrame(
            self.scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index
        )

        X_val_scaled = pd.DataFrame(
            self.scaler.transform(X_val),
            columns=X_val.columns,
            index=X_val.index
        )

        X_test_scaled = pd.DataFrame(
            self.scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index
        )

        X_train_scaled = X_train_scaled.fillna(0)
        X_val_scaled = X_val_scaled.fillna(0)
        X_test_scaled = X_test_scaled.fillna(0)

        return X_train_scaled, X_val_scaled, X_test_scaled

    def select_features(self, X_train: pd.DataFrame, y_train: pd.Series,
                       X_val: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        if not self.config.enable_feature_selection:
            return X_train, X_val, X_test

        logger.info(f"Selecting top {self.config.max_features} features")

        self.feature_selector = SelectKBest(
            score_func=f_regression,
            k=min(self.config.max_features, X_train.shape[1])
        )

        X_train_selected = pd.DataFrame(
            self.feature_selector.fit_transform(X_train, y_train),
            columns=X_train.columns[self.feature_selector.get_support()],
            index=X_train.index
        )

        X_val_selected = pd.DataFrame(
            self.feature_selector.transform(X_val),
            columns=X_train.columns[self.feature_selector.get_support()],
            index=X_val.index
        )

        X_test_selected = pd.DataFrame(
            self.feature_selector.transform(X_test),
            columns=X_train.columns[self.feature_selector.get_support()],
            index=X_test.index
        )

        self.feature_names = X_train_selected.columns.tolist()

        logger.info(f"Selected {len(self.feature_names)} features")

        return X_train_selected, X_val_selected, X_test_selected

    def preprocess(self, symbol: str, split_method: str = "percentage",
                  split_date: Optional[str] = None) -> PreprocessedData:

        logger.info(f"Starting complete preprocessing pipeline for {symbol}")

        raw_data = self.load_data(symbol)

        feature_set = self.engineer_features(raw_data)

        clean_features = self.handle_outliers(feature_set.engineered_features)

        if split_method == "temporal" and split_date:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_temporal(
                clean_features, feature_set.target, split_date
            )
        else:
            X_train, X_val, X_test, y_train, y_val, y_test = self.split_data_percentage(
                clean_features, feature_set.target
            )

        X_train_scaled, X_val_scaled, X_test_scaled = self.scale_features(
            X_train, X_val, X_test
        )

        X_train_final, X_val_final, X_test_final = self.select_features(
            X_train_scaled, y_train, X_val_scaled, X_test_scaled
        )

        metadata = {
            'symbol': symbol,
            'preprocessing_config': self.config,
            'split_method': split_method,
            'split_date': split_date,
            'original_features': feature_set.feature_names,
            'selected_features': self.feature_names,
            'data_shape': {
                'train': X_train_final.shape,
                'val': X_val_final.shape,
                'test': X_test_final.shape
            },
            'feature_engineering_metadata': feature_set.metadata
        }

        logger.info(f"Preprocessing completed for {symbol}")

        return PreprocessedData(
            X_train=X_train_final,
            X_val=X_val_final,
            X_test=X_test_final,
            y_train=y_train,
            y_val=y_val,
            y_test=y_test,
            scaler=self.scaler,
            feature_selector=self.feature_selector,
            metadata=metadata
        )

    def save_preprocessor(self, file_path: Path) -> None:

        preprocessor_data = {
            'config': self.config,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'feature_names': self.feature_names
        }

        with open(file_path, 'wb') as f:
            pickle.dump(preprocessor_data, f)

        logger.info(f"Preprocessor saved to {file_path}")

    def load_preprocessor(self, file_path: Path) -> None:

        with open(file_path, 'rb') as f:
            preprocessor_data = pickle.load(f)

        self.config = preprocessor_data['config']
        self.scaler = preprocessor_data['scaler']
        self.feature_selector = preprocessor_data['feature_selector']
        self.feature_names = preprocessor_data['feature_names']

        logger.info(f"Preprocessor loaded from {file_path}")

if __name__ == "__main__":
    """Test the data preprocessor."""

    test_config = PreprocessingConfig(
        test_size=0.2,
        validation_size=0.2,
        feature_window=7,
        max_features=30
    )

    preprocessor = DataPreprocessor(test_config)

    try:
        preprocessed_data = preprocessor.preprocess("Bitcoin")

        print("\n" + "="*50)
        print("PREPROCESSING RESULTS")
        print("="*50)

        print(f"Symbol: {preprocessed_data.metadata['symbol']}")
        print(f"Train shape: {preprocessed_data.X_train.shape}")
        print(f"Validation shape: {preprocessed_data.X_val.shape}")
        print(f"Test shape: {preprocessed_data.X_test.shape}")
        print(f"Selected features: {len(preprocessed_data.metadata['selected_features'])}")

        print("\nTop 10 selected features:")
        for i, feature in enumerate(preprocessed_data.metadata['selected_features'][:10]):
            print(f"  {i+1}. {feature}")

    except Exception as e:
        logger.error(f"Preprocessing test failed: {e}")
        print(f"Error: {e}")
