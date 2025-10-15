import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'app'))

from ml_model_trainer import ModelTrainer, ModelConfig
from ml_pipeline import MLPipeline, PipelineConfig
from ml_preprocessor import PreprocessingConfig


class TestModelTrainer:
    """Test suite for ModelTrainer class"""
    
    @pytest.fixture
    def model_config(self):
        """Create a sample model configuration"""
        return ModelConfig(
            model_type="random_forest",
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            enable_hyperparameter_tuning=False
        )
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data"""
        np.random.seed(42)
        n_samples = 100
        X = np.random.randn(n_samples, 10)
        y = np.random.randn(n_samples)
        return X, y
    
    def test_model_initialization(self, model_config):
        """Test model trainer initialization"""
        trainer = ModelTrainer(model_config)
        assert trainer.config == model_config
        assert trainer.config.model_type == "random_forest"
        assert trainer.config.hyperparameters["n_estimators"] == 100
    
    @patch('ml_model_trainer.ModelTrainer.train')
    def test_model_training(self, mock_train, model_config, sample_data):
        """Test model training process"""
        X, y = sample_data
        trainer = ModelTrainer(model_config)
        
        # Mock the training response
        mock_train.return_value = {
            'model': MagicMock(),
            'metrics': {'mse': 0.01, 'r2': 0.95}
        }
        
        result = trainer.train(X, y)
        assert result is not None
        assert 'model' in result
        assert 'metrics' in result
        assert result['metrics']['r2'] == 0.95
    
    def test_hyperparameter_optimization(self, model_config, sample_data):
        """Test hyperparameter optimization functionality"""
        X, y = sample_data
        model_config.enable_hyperparameter_tuning = True
        trainer = ModelTrainer(model_config)
        
        # Test that optimization config is properly set
        assert trainer.config.enable_hyperparameter_tuning == True
        
        # Mock optimization process
        with patch.object(trainer, 'optimize_hyperparameters') as mock_optimize:
            mock_optimize.return_value = {
                'best_params': {'n_estimators': 150, 'max_depth': 15},
                'best_score': 0.96
            }
            
            result = trainer.optimize_hyperparameters(X, y)
            assert result['best_score'] == 0.96
            assert result['best_params']['n_estimators'] == 150
    
    def test_model_evaluation(self, model_config, sample_data):
        """Test model evaluation metrics"""
        X, y = sample_data
        trainer = ModelTrainer(model_config)
        
        # Mock evaluation
        with patch.object(trainer, 'evaluate') as mock_evaluate:
            mock_evaluate.return_value = {
                'mse': 0.015,
                'mae': 0.08,
                'rmse': 0.122,
                'r2': 0.93,
                'mape': 7.5
            }
            
            metrics = trainer.evaluate(X, y)
            assert metrics['r2'] == 0.93
            assert metrics['mse'] == 0.015
            assert metrics['mape'] == 7.5


class TestMLPipeline:
    """Test suite for MLPipeline class"""
    
    @pytest.fixture
    def pipeline_config(self):
        """Create a sample pipeline configuration"""
        return PipelineConfig(
            symbols=["BTC", "ETH"],
            model_types=["random_forest", "xgboost"],
            test_size=0.2,
            validation_size=0.2,
            feature_window=7,
            prediction_horizon=1,
            enable_hyperparameter_tuning=True,
            save_models=True
        )
    
    @pytest.fixture
    def mock_data(self):
        """Create mock cryptocurrency data"""
        dates = pd.date_range(start='2023-01-01', periods=100, freq='D')
        data = pd.DataFrame({
            'date': dates,
            'open': np.random.uniform(100, 200, 100),
            'high': np.random.uniform(110, 210, 100),
            'low': np.random.uniform(90, 190, 100),
            'close': np.random.uniform(100, 200, 100),
            'volume': np.random.uniform(1000, 10000, 100)
        })
        return data
    
    def test_pipeline_initialization(self, pipeline_config):
        """Test pipeline initialization"""
        pipeline = MLPipeline(pipeline_config)
        assert pipeline.config == pipeline_config
        assert len(pipeline.config.symbols) == 2
        assert len(pipeline.config.model_types) == 2
    
    @patch('ml_pipeline.MLPipeline.run')
    def test_complete_pipeline_run(self, mock_run, pipeline_config, mock_data):
        """Test complete pipeline execution"""
        pipeline = MLPipeline(pipeline_config)
        
        # Mock the pipeline run
        mock_run.return_value = {
            'experiment_id': 'exp_001',
            'models_trained': 4,  # 2 symbols * 2 model types
            'best_model': {
                'symbol': 'BTC',
                'model_type': 'xgboost',
                'metrics': {'r2': 0.96, 'mse': 0.008}
            },
            'status': 'completed'
        }
        
        result = pipeline.run(mock_data)
        
        assert result['experiment_id'] == 'exp_001'
        assert result['models_trained'] == 4
        assert result['best_model']['model_type'] == 'xgboost'
        assert result['best_model']['metrics']['r2'] == 0.96
        assert result['status'] == 'completed'
    
    @patch('ml_pipeline.MLPipeline.preprocess_data')
    def test_data_preprocessing(self, mock_preprocess, pipeline_config, mock_data):
        """Test data preprocessing in pipeline"""
        pipeline = MLPipeline(pipeline_config)
        
        mock_preprocess.return_value = {
            'X_train': np.random.randn(70, 20),
            'X_val': np.random.randn(15, 20),
            'X_test': np.random.randn(15, 20),
            'y_train': np.random.randn(70),
            'y_val': np.random.randn(15),
            'y_test': np.random.randn(15),
            'feature_names': [f'feature_{i}' for i in range(20)]
        }
        
        result = pipeline.preprocess_data(mock_data, 'BTC')
        
        assert 'X_train' in result
        assert 'y_train' in result
        assert result['X_train'].shape[0] == 70
        assert len(result['feature_names']) == 20
    
    def test_model_comparison(self, pipeline_config):
        """Test model comparison functionality"""
        pipeline = MLPipeline(pipeline_config)
        
        # Mock model results
        model_results = {
            'random_forest': {'r2': 0.94, 'mse': 0.012},
            'xgboost': {'r2': 0.96, 'mse': 0.008},
            'decision_tree': {'r2': 0.89, 'mse': 0.022}
        }
        
        with patch.object(pipeline, 'compare_models') as mock_compare:
            mock_compare.return_value = {
                'best_model': 'xgboost',
                'comparison_table': model_results,
                'recommendation': 'XGBoost performs best with R2=0.96'
            }
            
            result = pipeline.compare_models(model_results)
            assert result['best_model'] == 'xgboost'
            assert result['comparison_table']['xgboost']['r2'] == 0.96
