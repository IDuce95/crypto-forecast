

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta

from app.config import config
from app.logger import logger
from app.ml_preprocessor import DataPreprocessor, PreprocessingConfig, PreprocessedData
from app.ml_model_trainer import ModelTrainer, ModelConfig, MultiModelTrainer, ModelResults
from app.ml_prediction_generator import PredictionGenerator, PredictionConfig, BatchPredictions
from app.mlflow_manager import MLflowManager, setup_mlflow
from app.pyspark_manager import PySparkManager, PySparkDataProcessor

@dataclass
class PipelineConfig:

    symbols: List[str] = None
    test_size: float = 0.2
    validation_size: float = 0.2
    feature_window: int = 7
    prediction_horizon: int = 1
    
    model_types: List[str] = None
    enable_hyperparameter_tuning: bool = True
    cross_validation_folds: int = 5
    
    enable_ensemble: bool = True
    confidence_level: float = 0.95
    
    save_models: bool = True
    save_predictions: bool = True
    models_dir: str = "./models/"
    predictions_dir: str = "./predictions/"
    experiments_dir: str = "./experiments/"
    
    random_state: int = 42

@dataclass
class ExperimentResults:

    experiment_id: str
    config: PipelineConfig
    preprocessed_data: Dict[str, PreprocessedData]
    model_results: Dict[str, Dict[str, ModelResults]]
    model_comparison: Dict[str, pd.DataFrame]
    predictions: Dict[str, BatchPredictions]
    summary_metrics: Dict[str, Any]
    experiment_timestamp: str
    metadata: Dict[str, Any]

class MLPipeline:

    def __init__(self, pipeline_config: PipelineConfig = None, mlflow_enabled: bool = True, experiment_name: str = "crypto-forecasting"):

        if pipeline_config is None:
            pipeline_config = PipelineConfig(
                symbols=["Bitcoin", "Ethereum"],
                model_types=["decision_tree", "random_forest", "xgboost", "lasso"],
                test_size=config.model_settings.default_test_size,
                validation_size=config.model_settings.default_validation_size,
                feature_window=config.model_settings.feature_engineering_window,
                prediction_horizon=config.model_settings.prediction_horizon,
                enable_hyperparameter_tuning=config.ml_pipeline.enable_hyperparameter_tuning,
                cross_validation_folds=config.ml_pipeline.cross_validation_folds,
                models_dir=config.ml_pipeline.models_output_dir,
                predictions_dir=config.ml_pipeline.predictions_output_dir,
                random_state=config.model_settings.random_state
            )
        
        self.config = pipeline_config
        self.experiment_id = None
        self.preprocessors = {}
        self.model_trainers = {}
        self.prediction_generators = {}
        
        self.mlflow_enabled = mlflow_enabled
        self.mlflow_manager = None
        if mlflow_enabled:
            try:
                self.mlflow_manager = setup_mlflow(experiment_name)
                logger.info(f"MLflow tracking enabled - Experiment: {experiment_name}")
            except Exception as e:
                logger.warning(f"Failed to initialize MLflow: {e}. Continuing without MLflow.")
                self.mlflow_enabled = False
        
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.predictions_dir).mkdir(parents=True, exist_ok=True)
        Path(self.config.experiments_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("MLPipeline initialized")
    
    def generate_experiment_id(self) -> str:

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        symbols_str = "_".join(self.config.symbols[:3])
        models_str = "_".join(self.config.model_types[:2])
        
        return f"exp_{timestamp}_{symbols_str}_{models_str}"
    
    def preprocess_data(self) -> Dict[str, PreprocessedData]:

        logger.info("Starting data preprocessing for all symbols...")
        
        preprocessed_data = {}
        
        preprocessing_config = PreprocessingConfig(
            test_size=self.config.test_size,
            validation_size=self.config.validation_size,
            feature_window=self.config.feature_window,
            prediction_horizon=self.config.prediction_horizon,
            random_state=self.config.random_state
        )
        
        for symbol in self.config.symbols:
            logger.info(f"Preprocessing data for {symbol}...")
            
            try:
                preprocessor = DataPreprocessor(preprocessing_config)
                data = preprocessor.preprocess(symbol)
                
                preprocessed_data[symbol] = data
                self.preprocessors[symbol] = preprocessor
                
                logger.info(f"Successfully preprocessed {symbol}: "
                          f"Train: {data.X_train.shape}, "
                          f"Val: {data.X_val.shape}, "
                          f"Test: {data.X_test.shape}")
                
            except Exception as e:
                logger.error(f"Failed to preprocess {symbol}: {e}")
                continue
        
        logger.info(f"Data preprocessing completed for {len(preprocessed_data)} symbols")
        return preprocessed_data
    
    def train_models(self, preprocessed_data: Dict[str, PreprocessedData]) -> Dict[str, Dict[str, ModelResults]]:

        logger.info("Starting model training for all symbols and model types...")
        
        all_model_results = {}
        
        for symbol, data in preprocessed_data.items():
            logger.info(f"Training models for {symbol}...")
            
            multi_trainer = MultiModelTrainer(self.config.model_types)
            
            try:
                symbol_results = multi_trainer.train_all_models(data)
                all_model_results[symbol] = symbol_results
                self.model_trainers[symbol] = multi_trainer
                
                logger.info(f"Completed training {len(symbol_results)} models for {symbol}")
                
                if self.config.save_models:
                    self._save_models(symbol, symbol_results)
                
            except Exception as e:
                logger.error(f"Failed to train models for {symbol}: {e}")
                continue
        
        logger.info(f"Model training completed for {len(all_model_results)} symbols")
        return all_model_results
    
    def _save_models(self, symbol: str, model_results: Dict[str, ModelResults]) -> None:

        symbol_dir = Path(self.config.models_dir) / self.experiment_id / symbol
        symbol_dir.mkdir(parents=True, exist_ok=True)
        
        for model_type, results in model_results.items():
            model_path = symbol_dir / f"{model_type}.pkl"
            
            trainer = self.model_trainers[symbol].trainers[model_type]
            trainer.save_model(model_path, results)
            
            logger.info(f"Saved {model_type} model for {symbol}")
    
    def compare_models(self, model_results: Dict[str, Dict[str, ModelResults]]) -> Dict[str, pd.DataFrame]:

        logger.info("Comparing model performance...")
        
        comparisons = {}
        
        for symbol, symbol_results in model_results.items():
            if symbol not in self.model_trainers:
                continue
            
            try:
                comparison_df = self.model_trainers[symbol].compare_models()
                comparisons[symbol] = comparison_df
                
                best_model = comparison_df.iloc[0]['model']
                best_mse = comparison_df.iloc[0]['test_mse']
                
                logger.info(f"Best model for {symbol}: {best_model} (MSE: {best_mse:.4f})")
                
            except Exception as e:
                logger.error(f"Failed to compare models for {symbol}: {e}")
                continue
        
        return comparisons
    
    def generate_predictions(self, preprocessed_data: Dict[str, PreprocessedData],
                           model_results: Dict[str, Dict[str, ModelResults]],
                           prediction_period_days: int = 30) -> Dict[str, BatchPredictions]:

        logger.info("Generating predictions for all symbols...")
        
        all_predictions = {}
        
        pred_config = PredictionConfig(
            prediction_horizon=self.config.prediction_horizon,
            confidence_level=self.config.confidence_level,
            enable_ensemble=self.config.enable_ensemble
        )
        
        for symbol in self.config.symbols:
            if symbol not in model_results or symbol not in preprocessed_data:
                logger.warning(f"Skipping predictions for {symbol}: missing data or models")
                continue
            
            logger.info(f"Generating predictions for {symbol}...")
            
            try:
                generator = PredictionGenerator(pred_config)
                
                for model_type, results in model_results[symbol].items():
                    trainer = self.model_trainers[symbol].trainers[model_type]
                    generator.add_trained_model(model_type, trainer)
                
                if len(model_results[symbol]) > 1:
                    generator.calculate_ensemble_weights(preprocessed_data[symbol])
                
                end_date = datetime.now()
                start_date = end_date - timedelta(days=prediction_period_days)
                
                predictions = generator.predict_time_series(
                    symbol=symbol,
                    preprocessor=self.preprocessors[symbol],
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                
                all_predictions[symbol] = predictions
                self.prediction_generators[symbol] = generator
                
                if self.config.save_predictions:
                    self._save_predictions(symbol, predictions)
                
                logger.info(f"Generated {len(predictions.predictions)} predictions for {symbol}")
                
            except Exception as e:
                logger.error(f"Failed to generate predictions for {symbol}: {e}")
                continue
        
        logger.info(f"Prediction generation completed for {len(all_predictions)} symbols")
        return all_predictions
    
    def _save_predictions(self, symbol: str, predictions: BatchPredictions) -> None:

        pred_dir = Path(self.config.predictions_dir) / self.experiment_id
        pred_dir.mkdir(parents=True, exist_ok=True)
        
        pred_path = pred_dir / f"{symbol}_predictions.csv"
        
        if symbol in self.prediction_generators:
            self.prediction_generators[symbol].save_predictions(predictions, pred_path)
    
    def calculate_summary_metrics(self, model_results: Dict[str, Dict[str, ModelResults]],
                                predictions: Dict[str, BatchPredictions]) -> Dict[str, Any]:

        logger.info("Calculating summary metrics...")
        
        summary = {
            'experiment_overview': {
                'symbols_processed': len(model_results),
                'total_models_trained': sum(len(models) for models in model_results.values()),
                'total_predictions_generated': sum(len(preds.predictions) for preds in predictions.values()),
                'experiment_duration': None
            },
            'model_performance': {},
            'prediction_summary': {},
            'best_models': {}
        }
        
        for symbol, symbol_results in model_results.items():
            symbol_metrics = {}
            
            for model_type, results in symbol_results.items():
                test_metrics = results.metadata['test_metrics']
                symbol_metrics[model_type] = {
                    'test_mse': test_metrics.mse,
                    'test_mae': test_metrics.mae,
                    'test_r2': test_metrics.r2,
                    'test_mape': test_metrics.mape
                }
            
            summary['model_performance'][symbol] = symbol_metrics
            
            best_model_type = min(symbol_metrics.keys(), 
                                key=lambda x: symbol_metrics[x]['test_mse'])
            summary['best_models'][symbol] = {
                'model_type': best_model_type,
                'test_mse': symbol_metrics[best_model_type]['test_mse'],
                'test_r2': symbol_metrics[best_model_type]['test_r2']
            }
        
        for symbol, preds in predictions.items():
            summary['prediction_summary'][symbol] = {
                'total_predictions': preds.summary_stats['total_predictions'],
                'predictions_with_actuals': preds.summary_stats['predictions_with_actuals'],
                'mean_prediction': preds.summary_stats['mean_prediction'],
                'prediction_mse': preds.summary_stats.get('mse', None),
                'prediction_mae': preds.summary_stats.get('mae', None)
            }
        
        return summary
    
    def run_complete_pipeline(self, prediction_period_days: int = 30) -> ExperimentResults:

        start_time = datetime.now()
        self.experiment_id = self.generate_experiment_id()
        
        logger.info(f"Starting complete ML pipeline - Experiment ID: {self.experiment_id}")
        
        mlflow_run_id = None
        if self.mlflow_enabled and self.mlflow_manager:
            try:
                mlflow_run_id = self.mlflow_manager.start_run(f"pipeline_{self.experiment_id}")
                
                pipeline_params = {
                    "experiment_id": self.experiment_id,
                    "symbols": self.config.symbols,
                    "model_types": self.config.model_types,
                    "test_size": self.config.test_size,
                    "validation_size": self.config.validation_size,
                    "feature_window": self.config.feature_window,
                    "prediction_horizon": self.config.prediction_horizon,
                    "prediction_period_days": prediction_period_days,
                    "hyperparameter_tuning_enabled": self.config.enable_hyperparameter_tuning
                }
                self.mlflow_manager.log_params(pipeline_params)
                
            except Exception as e:
                logger.warning(f"Failed to start MLflow run: {e}")
        
        try:
            preprocessed_data = self.preprocess_data()
            
            if not preprocessed_data:
                raise ValueError("No data was successfully preprocessed")
            
            if self.mlflow_enabled and self.mlflow_manager:
                for symbol, data in preprocessed_data.items():
                    self.mlflow_manager.log_preprocessing_results(data, symbol)
            
            model_results = self.train_models(preprocessed_data)
            
            if not model_results:
                raise ValueError("No models were successfully trained")
            
            if self.mlflow_enabled and self.mlflow_manager:
                for symbol, symbol_results in model_results.items():
                    for model_type, results in symbol_results.items():
                        training_metrics = {
                            "train_mse": results.metadata['train_metrics'].mse,
                            "val_mse": results.metadata['val_metrics'].mse,
                            "test_mse": results.metadata['test_metrics'].mse,
                            "test_r2": results.metadata['test_metrics'].r2,
                            "test_mae": results.metadata['test_metrics'].mae,
                        }
                        self.mlflow_manager.log_model_training_results(
                            results.model, model_type, symbol, training_metrics, results.metadata
                        )
            
            model_comparisons = self.compare_models(model_results)
            
            predictions = self.generate_predictions(
                preprocessed_data, model_results, prediction_period_days
            )
            
            summary_metrics = self.calculate_summary_metrics(model_results, predictions)
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            summary_metrics['experiment_overview']['experiment_duration'] = duration
            
            results = ExperimentResults(
                experiment_id=self.experiment_id,
                config=self.config,
                preprocessed_data=preprocessed_data,
                model_results=model_results,
                model_comparison=model_comparisons,
                predictions=predictions,
                summary_metrics=summary_metrics,
                experiment_timestamp=start_time.isoformat(),
                metadata={
                    'start_time': start_time.isoformat(),
                    'end_time': end_time.isoformat(),
                    'duration_seconds': duration
                }
            )
            
            self._save_experiment_results(results)
            
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    summary_metrics = {
                        "total_symbols_processed": len(model_results),
                        "total_models_trained": sum(len(models) for models in model_results.values()),
                        "experiment_duration_seconds": duration,
                        "avg_test_mse": np.mean([results.metadata['test_metrics'].mse 
                                               for symbol_results in model_results.values() 
                                               for results in symbol_results.values()]),
                        "avg_test_r2": np.mean([results.metadata['test_metrics'].r2 
                                              for symbol_results in model_results.values() 
                                              for results in symbol_results.values()])
                    }
                    self.mlflow_manager.log_metrics(summary_metrics)
                    
                    experiment_summary = {
                        "config": results.config.__dict__,
                        "summary_metrics": results.summary_metrics,
                        "metadata": results.metadata
                    }
                    self.mlflow_manager.log_experiment_summary(experiment_summary)
                    
                except Exception as e:
                    logger.warning(f"Failed to log final results to MLflow: {e}")
            
            logger.info(f"Complete pipeline execution finished - Duration: {duration:.2f}s")
            return results
            
        except Exception as e:
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    self.mlflow_manager.end_run("FAILED")
                except:
                    pass
            
            logger.error(f"Pipeline execution failed: {e}")
            raise
        
        finally:
            if self.mlflow_enabled and self.mlflow_manager:
                try:
                    self.mlflow_manager.end_run("FINISHED")
                except:
                    pass
    
    def _save_experiment_results(self, results: ExperimentResults) -> None:

        exp_dir = Path(self.config.experiments_dir) / self.experiment_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        summary_path = exp_dir / "experiment_summary.json"
        
        summary_data = {
            'experiment_id': results.experiment_id,
            'config': results.config.__dict__,
            'summary_metrics': results.summary_metrics,
            'experiment_timestamp': results.experiment_timestamp,
            'metadata': results.metadata
        }
        
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        for symbol, comparison_df in results.model_comparison.items():
            comp_path = exp_dir / f"{symbol}_model_comparison.csv"
            comparison_df.to_csv(comp_path, index=False)
        
        logger.info(f"Experiment results saved to {exp_dir}")

def create_default_pipeline() -> MLPipeline:

    config = PipelineConfig(
        symbols=["Bitcoin", "Ethereum", "Litecoin"],
        model_types=["random_forest", "xgboost", "lasso"],
        enable_hyperparameter_tuning=True,
        enable_ensemble=True,
        save_models=True,
        save_predictions=True
    )
    
    return MLPipeline(config)

if __name__ == "__main__":
