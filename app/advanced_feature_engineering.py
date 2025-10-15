
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
import warnings
warnings.filterwarnings('ignore')
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logger_manager import LoggerManager
from cache_manager import CacheManager


@dataclass
class FeatureEngineeringConfig:
        if self.sma_periods is None:
            self.sma_periods = [5, 10, 20, 50, 100, 200]
        if self.ema_periods is None:
            self.ema_periods = [5, 10, 20, 50, 100]
        if self.rsi_periods is None:
            self.rsi_periods = [14, 21, 30]
        if self.macd_settings is None:
            self.macd_settings = {"fast": 12, "slow": 26, "signal": 9}
        if self.rolling_windows is None:
            self.rolling_windows = [5, 10, 20, 50]
        if self.lag_features is None:
            self.lag_features = [1, 2, 3, 5, 10]


class AdvancedFeatureEngineer:
        Initialize feature engineer
        
        Args:
            config: Feature engineering configuration
        if not self.config.enable_technical_indicators:
            return data
        
        features_df = data.copy()
        
        for period in self.config.sma_periods:
            features_df[f'SMA_{period}'] = data['Close'].rolling(window=period).mean()
            features_df[f'SMA_ratio_{period}'] = data['Close'] / features_df[f'SMA_{period}']
        
        for period in self.config.ema_periods:
            features_df[f'EMA_{period}'] = data['Close'].ewm(span=period).mean()
            features_df[f'EMA_ratio_{period}'] = data['Close'] / features_df[f'EMA_{period}']
        
        for period in self.config.rsi_periods:
            delta = data['Close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            features_df[f'RSI_{period}'] = 100 - (100 / (1 + rs))
        
        fast = self.config.macd_settings["fast"]
        slow = self.config.macd_settings["slow"]
        signal = self.config.macd_settings["signal"]
        
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        macd = ema_fast - ema_slow
        macd_signal = macd.ewm(span=signal).mean()
        macd_hist = macd - macd_signal
        
        features_df['MACD'] = macd
        features_df['MACD_signal'] = macd_signal
        features_df['MACD_histogram'] = macd_hist
        
        sma = data['Close'].rolling(window=self.config.bollinger_period).mean()
        std = data['Close'].rolling(window=self.config.bollinger_period).std()
        bb_upper = sma + (std * self.config.bollinger_std)
        bb_lower = sma - (std * self.config.bollinger_std)
        
        features_df['BB_upper'] = bb_upper
        features_df['BB_middle'] = sma
        features_df['BB_lower'] = bb_lower
        features_df['BB_position'] = (data['Close'] - bb_lower) / (bb_upper - bb_lower)
        features_df['BB_width'] = (bb_upper - bb_lower) / sma
        
        high_low = data['High'] - data['Low']
        high_close = abs(data['High'] - data['Close'].shift())
        low_close = abs(data['Low'] - data['Close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        features_df['ATR'] = true_range.rolling(window=14).mean()
        
        typical_price = (data['High'] + data['Low'] + data['Close']) / 3
        sma_tp = typical_price.rolling(window=20).mean()
        mad = typical_price.rolling(window=20).apply(lambda x: np.mean(np.abs(x - x.mean())))
        features_df['CCI'] = (typical_price - sma_tp) / (0.015 * mad)
        
        highest_high = data['High'].rolling(window=14).max()
        lowest_low = data['Low'].rolling(window=14).min()
        features_df['Williams_R'] = -100 * (highest_high - data['Close']) / (highest_high - lowest_low)
        
        features_df['Momentum'] = data['Close'] / data['Close'].shift(10) - 1
        
        features_df['ROC'] = data['Close'].pct_change(periods=10) * 100
        
        lowest_low_14 = data['Low'].rolling(window=14).min()
        highest_high_14 = data['High'].rolling(window=14).max()
        k_percent = 100 * (data['Close'] - lowest_low_14) / (highest_high_14 - lowest_low_14)
        features_df['STOCH_K'] = k_percent.rolling(window=3).mean()
        features_df['STOCH_D'] = features_df['STOCH_K'].rolling(window=3).mean()
        
        self.logger.info(f"Created {len([c for c in features_df.columns if c not in data.columns])} technical indicators")
        return features_df
    
    def create_statistical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enable_time_features:
            return data
        
        features_df = data.copy()
        
        if not isinstance(data.index, pd.DatetimeIndex):
            try:
                features_df.index = pd.to_datetime(features_df.index)
            except:
                self.logger.warning("Could not convert index to datetime")
                return features_df
        
        features_df['hour'] = features_df.index.hour
        features_df['day_of_week'] = features_df.index.dayofweek
        features_df['day_of_month'] = features_df.index.day
        features_df['month'] = features_df.index.month
        features_df['quarter'] = features_df.index.quarter
        features_df['year'] = features_df.index.year
        
        if self.config.cyclical_encoding:
            features_df['hour_sin'] = np.sin(2 * np.pi * features_df['hour'] / 24)
            features_df['hour_cos'] = np.cos(2 * np.pi * features_df['hour'] / 24)
            features_df['day_sin'] = np.sin(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['day_cos'] = np.cos(2 * np.pi * features_df['day_of_week'] / 7)
            features_df['month_sin'] = np.sin(2 * np.pi * features_df['month'] / 12)
            features_df['month_cos'] = np.cos(2 * np.pi * features_df['month'] / 12)
        
        features_df['is_weekend'] = features_df['day_of_week'].isin([5, 6]).astype(int)
        features_df['is_month_end'] = (features_df.index.day > 25).astype(int)
        features_df['is_quarter_end'] = (
            (features_df['month'].isin([3, 6, 9, 12])) & 
            (features_df.index.day > 25)
        ).astype(int)
        
        self.logger.info("Created time-based features")
        return features_df
    
    def create_microstructure_features(self, data: pd.DataFrame) -> pd.DataFrame:
        if not self.config.enable_fourier_features:
            return data
        
        features_df = data.copy()
        prices = data['Close'].dropna().values
        
        fft = np.fft.fft(prices)
        fft_freq = np.fft.fftfreq(len(prices))
        
        for i in range(1, min(self.config.fourier_order + 1, len(fft) // 2)):
            features_df[f'fourier_real_{i}'] = np.real(fft[i])
            features_df[f'fourier_imag_{i}'] = np.imag(fft[i])
            features_df[f'fourier_mag_{i}'] = np.abs(fft[i])
        
        self.logger.info("Created Fourier features")
        return features_df
    
    def select_features(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        self.logger.info("Starting feature engineering transformation")
        
        cache_key = f"features_{len(data)}_{hash(str(self.config))}"
        cached_features = self.cache_manager.get_cached_processed_data(cache_key)
        
        if cached_features is not None:
            self.logger.info("Using cached features")
            return cached_features['X'], cached_features['y']
        
        features_df = data.copy()
        
        features_df = self.create_technical_indicators(features_df)
        features_df = self.create_statistical_features(features_df)
        features_df = self.create_time_features(features_df)
        features_df = self.create_microstructure_features(features_df)
        features_df = self.create_fourier_features(features_df)
        
        if target_column in features_df.columns:
            y = features_df[target_column].shift(-1)  # Predict next period
            X = features_df.drop(columns=[target_column])
        else:
            self.logger.warning(f"Target column {target_column} not found")
            y = features_df.iloc[:, 0].shift(-1)
            X = features_df
        
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        X = X.drop(columns=[col for col in ohlcv_cols if col in X.columns], errors='ignore')
        
        X = self.select_features(X, y)
        
        if self.scaler is not None:
            numeric_cols = X.select_dtypes(include=[np.number]).columns
            X_numeric = X[numeric_cols]
            
            mask = ~X_numeric.isna().any(axis=1)
            if mask.sum() > 0:
                self.scaler.fit(X_numeric[mask])
                X_scaled = X_numeric.copy()
                X_scaled[mask] = self.scaler.transform(X_numeric[mask])
                X[numeric_cols] = X_scaled
        
        self.feature_names = X.columns.tolist()
        
        self.cache_manager.cache_processed_data(cache_key, {
            'X': X,
            'y': y,
            'feature_names': self.feature_names
        })
        
        self.logger.info(f"Feature engineering completed - {len(self.feature_names)} features created")
        return X, y
    
    def get_feature_importance_analysis(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
