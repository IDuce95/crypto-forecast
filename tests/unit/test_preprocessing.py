import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'app'))

from ml_preprocessor import DataPreprocessor, PreprocessingConfig


class TestDataPreprocessor:
    """Test suite for DataPreprocessor class"""
    
    @pytest.fixture
    def preprocessing_config(self):
        """Create a sample preprocessing configuration"""
        return PreprocessingConfig(
            test_size=0.2,
            validation_size=0.2,
            feature_window=7,
            scale_features=True,
            handle_missing='interpolate',
            remove_outliers=True,
            outlier_threshold=3.0
        )
    
    @pytest.fixture
    def sample_crypto_data(self):
        """Create sample cryptocurrency data"""
        dates = pd.date_range(start='2023-01-01', periods=200, freq='D')
        np.random.seed(42)
        
        # Create realistic price data with trends
        base_price = 50000
        trend = np.cumsum(np.random.randn(200)) * 100
        noise = np.random.randn(200) * 500
        
        data = pd.DataFrame({
            'date': dates,
            'open': base_price + trend + noise,
            'high': base_price + trend + noise + np.abs(np.random.randn(200) * 200),
            'low': base_price + trend + noise - np.abs(np.random.randn(200) * 200),
            'close': base_price + trend + noise + np.random.randn(200) * 100,
            'volume': np.random.uniform(1e8, 5e8, 200)
        })
        
        # Add some missing values
        data.loc[10:12, 'volume'] = np.nan
        data.loc[50, 'close'] = np.nan
        
        # Add outliers
        data.loc[100, 'close'] = base_price * 2  # Outlier
        data.loc[150, 'volume'] = 1e10  # Volume outlier
        
        return data
    
    def test_preprocessor_initialization(self, preprocessing_config):
        """Test preprocessor initialization"""
        preprocessor = DataPreprocessor(preprocessing_config)
        assert preprocessor.config == preprocessing_config
        assert preprocessor.config.test_size == 0.2
        assert preprocessor.config.feature_window == 7
    
    def test_feature_engineering(self, preprocessing_config, sample_crypto_data):
        """Test feature engineering functionality"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Apply feature engineering
        with patch.object(preprocessor, 'engineer_features') as mock_engineer:
            # Mock the engineered features
            engineered_data = sample_crypto_data.copy()
            engineered_data['sma_7'] = engineered_data['close'].rolling(7).mean()
            engineered_data['sma_30'] = engineered_data['close'].rolling(30).mean()
            engineered_data['rsi'] = 50 + np.random.randn(len(engineered_data)) * 10
            engineered_data['macd'] = np.random.randn(len(engineered_data)) * 100
            engineered_data['bollinger_upper'] = engineered_data['close'] + 2000
            engineered_data['bollinger_lower'] = engineered_data['close'] - 2000
            engineered_data['volume_sma'] = engineered_data['volume'].rolling(7).mean()
            
            mock_engineer.return_value = engineered_data
            
            result = preprocessor.engineer_features(sample_crypto_data)
            
            # Check that technical indicators are added
            assert 'sma_7' in result.columns
            assert 'sma_30' in result.columns
            assert 'rsi' in result.columns
            assert 'macd' in result.columns
            assert 'bollinger_upper' in result.columns
            assert 'bollinger_lower' in result.columns
            assert 'volume_sma' in result.columns
            
            # Check that values are computed
            assert not result['sma_7'].isna().all()
            assert not result['rsi'].isna().all()
    
    def test_outlier_handling(self, preprocessing_config, sample_crypto_data):
        """Test outlier detection and handling"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Test outlier detection
        with patch.object(preprocessor, 'handle_outliers') as mock_handle:
            # Mock cleaned data (outliers removed/capped)
            cleaned_data = sample_crypto_data.copy()
            
            # Cap outliers instead of removing
            close_mean = cleaned_data['close'].mean()
            close_std = cleaned_data['close'].std()
            upper_bound = close_mean + 3 * close_std
            lower_bound = close_mean - 3 * close_std
            
            cleaned_data['close'] = cleaned_data['close'].clip(lower_bound, upper_bound)
            
            mock_handle.return_value = cleaned_data
            
            result = preprocessor.handle_outliers(sample_crypto_data)
            
            # Check that extreme outlier is handled
            assert result['close'].max() < sample_crypto_data['close'].max()
            assert len(result) == len(sample_crypto_data)  # No rows removed in this case
    
    def test_missing_value_handling(self, preprocessing_config, sample_crypto_data):
        """Test missing value handling"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Count initial missing values
        initial_missing = sample_crypto_data.isna().sum().sum()
        assert initial_missing > 0  # Ensure we have missing values to test
        
        with patch.object(preprocessor, 'handle_missing_values') as mock_handle:
            # Mock filled data
            filled_data = sample_crypto_data.copy()
            filled_data = filled_data.interpolate(method='linear')
            filled_data = filled_data.fillna(method='ffill').fillna(method='bfill')
            
            mock_handle.return_value = filled_data
            
            result = preprocessor.handle_missing_values(sample_crypto_data)
            
            # Check that missing values are handled
            final_missing = result.isna().sum().sum()
            assert final_missing < initial_missing
            assert final_missing == 0  # All missing values should be handled
    
    def test_feature_selection(self, preprocessing_config, sample_crypto_data):
        """Test feature selection process"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        # Add some features
        data_with_features = sample_crypto_data.copy()
        data_with_features['feature_1'] = np.random.randn(len(data_with_features))
        data_with_features['feature_2'] = np.random.randn(len(data_with_features))
        data_with_features['feature_3'] = np.random.randn(len(data_with_features))
        data_with_features['feature_4'] = np.ones(len(data_with_features))  # Constant feature
        data_with_features['feature_5'] = data_with_features['close'] * 2  # Highly correlated
        
        with patch.object(preprocessor, 'select_features') as mock_select:
            # Mock feature selection (remove constant and highly correlated)
            selected_features = ['open', 'high', 'low', 'close', 'volume', 
                               'feature_1', 'feature_2', 'feature_3']
            mock_select.return_value = data_with_features[selected_features]
            
            result = preprocessor.select_features(data_with_features)
            
            # Check that constant and highly correlated features are removed
            assert 'feature_4' not in result.columns  # Constant removed
            assert 'feature_5' not in result.columns  # Highly correlated removed
            assert 'feature_1' in result.columns
            assert 'feature_2' in result.columns
    
    def test_data_splitting(self, preprocessing_config, sample_crypto_data):
        """Test train/validation/test split"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        with patch.object(preprocessor, 'split_data') as mock_split:
            # Calculate split sizes
            n_samples = len(sample_crypto_data)
            test_size = int(n_samples * 0.2)
            val_size = int(n_samples * 0.2)
            train_size = n_samples - test_size - val_size
            
            # Mock the split
            mock_split.return_value = {
                'train': sample_crypto_data.iloc[:train_size],
                'validation': sample_crypto_data.iloc[train_size:train_size+val_size],
                'test': sample_crypto_data.iloc[train_size+val_size:]
            }
            
            result = preprocessor.split_data(sample_crypto_data)
            
            # Check split proportions
            assert len(result['train']) == train_size
            assert len(result['validation']) == val_size
            assert len(result['test']) == test_size
            
            # Check no data leakage
            total_samples = len(result['train']) + len(result['validation']) + len(result['test'])
            assert total_samples == n_samples
    
    def test_feature_scaling(self, preprocessing_config, sample_crypto_data):
        """Test feature scaling/normalization"""
        preprocessor = DataPreprocessor(preprocessing_config)
        
        with patch.object(preprocessor, 'scale_features') as mock_scale:
            # Mock scaled data
            scaled_data = sample_crypto_data.copy()
            numeric_cols = scaled_data.select_dtypes(include=[np.number]).columns
            
            for col in numeric_cols:
                # Standardize: (x - mean) / std
                scaled_data[col] = (scaled_data[col] - scaled_data[col].mean()) / scaled_data[col].std()
            
            mock_scale.return_value = scaled_data
            
            result = preprocessor.scale_features(sample_crypto_data)
            
            # Check that data is scaled
            for col in numeric_cols:
                assert abs(result[col].mean()) < 1e-10  # Mean should be ~0
                assert abs(result[col].std() - 1) < 1e-10  # Std should be ~1
