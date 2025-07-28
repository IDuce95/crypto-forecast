

import mlflow
import mlflow.sklearn
import mlflow.pytorch
import mlflow.xgboost
import os
import tempfile
import json
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from datetime import datetime
import pickle

from app.config import config
from app.logger import logger

class MLflowManager:

    def __init__(self, experiment_name: str = "crypto-forecasting", tracking_uri: Optional[str] = None):

        self.experiment_name = experiment_name
        
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            mlruns_path = Path(__file__).parent.parent / "mlruns"
            mlruns_path.mkdir(exist_ok=True)
            mlflow.set_tracking_uri(f"file://{mlruns_path.absolute()}")
        
        try:
            self.experiment = mlflow.get_experiment_by_name(experiment_name)
            if self.experiment is None:
                experiment_id = mlflow.create_experiment(experiment_name)
                self.experiment = mlflow.get_experiment(experiment_id)
                logger.info(f"Created new MLflow experiment: {experiment_name}")
            else:
                logger.info(f"Using existing MLflow experiment: {experiment_name}")
        except Exception as e:
            logger.error(f"Error setting up MLflow experiment: {e}")
            raise

        mlflow.set_experiment(experiment_name)
        
        self.active_run = None
        logger.info(f"MLflowManager initialized - Experiment: {experiment_name}, URI: {mlflow.get_tracking_uri()}")

    def start_run(self, run_name: Optional[str] = None, nested: bool = False) -> str:

        try:
            if run_name is None:
                run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.active_run = mlflow.start_run(run_name=run_name, nested=nested)
            run_id = self.active_run.info.run_id
            
            logger.info(f"Started MLflow run: {run_name} (ID: {run_id})")
            return run_id
            
        except Exception as e:
            logger.error(f"Error starting MLflow run: {e}")
            raise

    def end_run(self, status: str = "FINISHED"):

        try:
            if self.active_run:
                mlflow.end_run(status=status)
                logger.info(f"Ended MLflow run: {self.active_run.info.run_id}")
                self.active_run = None
            else:
                logger.warning("No active MLflow run to end")
                
        except Exception as e:
            logger.error(f"Error ending MLflow run: {e}")

    def log_params(self, params: Dict[str, Any]):

        try:
            for key, value in params.items():
                if isinstance(value, (dict, list)):
                    value = json.dumps(value)
                elif not isinstance(value, (str, int, float, bool)):
                    value = str(value)
                
                mlflow.log_param(key, value)
            
            logger.info(f"Logged {len(params)} parameters to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging parameters to MLflow: {e}")

    def log_metrics(self, metrics: Dict[str, Union[float, int]], step: Optional[int] = None):

        try:
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    mlflow.log_metric(key, value, step=step)
                else:
                    logger.warning(f"Skipping invalid metric {key}: {value}")
            
            logger.info(f"Logged {len(metrics)} metrics to MLflow")
            
        except Exception as e:
            logger.error(f"Error logging metrics to MLflow: {e}")

    def log_model(self, model: Any, model_name: str, model_type: str = "sklearn",
                  signature: Optional[Any] = None, input_example: Optional[Any] = None,
                  conda_env: Optional[Dict] = None) -> str:

        try:
            model_info = None
            
            if model_type.lower() == "sklearn":
                model_info = mlflow.sklearn.log_model(
                    sk_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            elif model_type.lower() == "xgboost":
                model_info = mlflow.xgboost.log_model(
                    xgb_model=model,
                    artifact_path=model_name,
                    signature=signature,
                    input_example=input_example,
                    conda_env=conda_env
                )
            else:
                model_info = mlflow.log_artifact(
                    local_path=self._save_model_temporarily(model, model_name),
                    artifact_path=f"models/{model_name}"
                )
            
            model_uri = model_info.model_uri if model_info else f"runs:/{mlflow.active_run().info.run_id}/models/{model_name}"
            logger.info(f"Logged {model_type} model '{model_name}' to MLflow: {model_uri}")
            
            return model_uri
            
        except Exception as e:
            logger.error(f"Error logging model to MLflow: {e}")
            raise

    def log_artifacts(self, artifacts: Dict[str, Any], artifact_dir: str = "artifacts"):

        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = Path(tmp_dir)
                
                for name, data in artifacts.items():
                    file_path = tmp_path / f"{name}.json"
                    
                    if isinstance(data, (dict, list)):
                        with open(file_path, 'w') as f:
                            json.dump(data, f, indent=2, default=str)
                    elif isinstance(data, pd.DataFrame):
                        file_path = tmp_path / f"{name}.csv"
                        data.to_csv(file_path, index=False)
                    elif isinstance(data, np.ndarray):
                        file_path = tmp_path / f"{name}.npy"
                        np.save(file_path, data)
                    else:
                        file_path = tmp_path / f"{name}.txt"
                        with open(file_path, 'w') as f:
                            f.write(str(data))
                    
                    mlflow.log_artifact(str(file_path), artifact_dir)
                
                logger.info(f"Logged {len(artifacts)} artifacts to MLflow")
                
        except Exception as e:
            logger.error(f"Error logging artifacts to MLflow: {e}")

    def log_preprocessing_results(self, preprocessed_data, symbol: str):

        try:
            preprocessing_params = {
                f"preprocessing_{symbol}_train_shape": str(preprocessed_data.X_train.shape),
                f"preprocessing_{symbol}_val_shape": str(preprocessed_data.X_val.shape),
                f"preprocessing_{symbol}_test_shape": str(preprocessed_data.X_test.shape),
                f"preprocessing_{symbol}_features_count": len(preprocessed_data.metadata['selected_features']),
                f"preprocessing_{symbol}_scaling_method": preprocessed_data.metadata['preprocessing_config'].scaling_method,
            }
            self.log_params(preprocessing_params)
            
            artifacts = {
                f"selected_features_{symbol}": preprocessed_data.metadata['selected_features'],
                f"preprocessing_metadata_{symbol}": preprocessed_data.metadata
            }
            self.log_artifacts(artifacts, f"preprocessing/{symbol}")
            
            logger.info(f"Logged preprocessing results for {symbol}")
            
        except Exception as e:
            logger.error(f"Error logging preprocessing results: {e}")

    def log_model_training_results(self, model, model_name: str, symbol: str, 
                                 training_metrics: Dict[str, float], 
                                 model_metadata: Dict[str, Any]):

        try:
            model_uri = self.log_model(model, f"{model_name}_{symbol}", "sklearn")
            
            metrics_with_prefix = {f"{model_name}_{symbol}_{k}": v for k, v in training_metrics.items()}
            self.log_metrics(metrics_with_prefix)
            
            params_with_prefix = {f"{model_name}_{symbol}_{k}": v for k, v in model_metadata.items()}
            self.log_params(params_with_prefix)
            
            artifacts = {
                f"model_metadata_{model_name}_{symbol}": model_metadata
            }
            self.log_artifacts(artifacts, f"models/{symbol}")
            
            logger.info(f"Logged training results for {model_name} on {symbol}")
            
            return model_uri
            
        except Exception as e:
            logger.error(f"Error logging model training results: {e}")

    def log_prediction_results(self, predictions: Dict[str, Any], symbol: str):

        try:
            if 'metrics' in predictions:
                metrics_with_prefix = {f"prediction_{symbol}_{k}": v for k, v in predictions['metrics'].items()}
                self.log_metrics(metrics_with_prefix)
            
            artifacts = {
                f"predictions_{symbol}": predictions
            }
            self.log_artifacts(artifacts, f"predictions/{symbol}")
            
            logger.info(f"Logged prediction results for {symbol}")
            
        except Exception as e:
            logger.error(f"Error logging prediction results: {e}")

    def log_experiment_summary(self, experiment_results: Dict[str, Any]):

        try:
            if 'summary_metrics' in experiment_results:
                self.log_metrics(experiment_results['summary_metrics'])
            
            if 'config' in experiment_results:
                self.log_params(experiment_results['config'])
            
            self.log_artifacts({'experiment_summary': experiment_results}, 'experiment')
            
            logger.info("Logged complete experiment summary")
            
        except Exception as e:
            logger.error(f"Error logging experiment summary: {e}")

    def load_model(self, model_uri: str) -> Any:

        try:
            if "sklearn" in model_uri or "artifact_path" in model_uri:
                model = mlflow.sklearn.load_model(model_uri)
            else:
                model = mlflow.pyfunc.load_model(model_uri)
            
            logger.info(f"Loaded model from MLflow: {model_uri}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model from MLflow: {e}")
            raise

    def search_experiments(self, filter_string: Optional[str] = None, 
                          max_results: int = 100) -> List[Dict]:

        try:
            experiment = mlflow.get_experiment_by_name(self.experiment_name)
            runs = mlflow.search_runs(
                experiment_ids=[experiment.experiment_id],
                filter_string=filter_string,
                max_results=max_results,
                order_by=["start_time DESC"]
            )
            
            logger.info(f"Found {len(runs)} runs matching criteria")
            return runs.to_dict('records') if not runs.empty else []
            
        except Exception as e:
            logger.error(f"Error searching experiments: {e}")
            return []

    def get_best_models(self, metric_name: str = "test_mse", 
                       ascending: bool = True, top_k: int = 5) -> List[Dict]:

        try:
            runs = self.search_experiments()
            
            runs_with_metric = [run for run in runs if f"metrics.{metric_name}" in run and run[f"metrics.{metric_name}"] is not None]
            
            sorted_runs = sorted(runs_with_metric, 
                               key=lambda x: x[f"metrics.{metric_name}"], 
                               reverse=not ascending)
            
            best_models = sorted_runs[:top_k]
            
            logger.info(f"Found {len(best_models)} best models based on {metric_name}")
            return best_models
            
        except Exception as e:
            logger.error(f"Error getting best models: {e}")
            return []

    def _save_model_temporarily(self, model: Any, model_name: str) -> str:

        temp_dir = tempfile.gettempdir()
        model_path = Path(temp_dir) / f"{model_name}.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        return str(model_path)

    def cleanup(self):

        try:
            if self.active_run:
                self.end_run("FINISHED")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        if exc_type:
            self.end_run("FAILED")
        else:
            self.end_run("FINISHED")

def setup_mlflow(experiment_name: str = "crypto-forecasting") -> MLflowManager:

    return MLflowManager(experiment_name=experiment_name)

if __name__ == "__main__":
    """Test MLflow integration."""
    
    print("Testing MLflow integration...")
    
    try:
        with setup_mlflow("test-experiment") as mlflow_manager:
            run_id = mlflow_manager.start_run("test_run")
            print(f"Started test run: {run_id}")
            
            test_params = {
                "learning_rate": 0.01,
                "batch_size": 32,
                "model_type": "random_forest"
            }
            mlflow_manager.log_params(test_params)
            
            test_metrics = {
                "accuracy": 0.95,
                "loss": 0.05,
                "f1_score": 0.92
            }
            mlflow_manager.log_metrics(test_metrics)
            
            test_artifacts = {
                "config": {"test": True, "version": "1.0"},
                "results": [1, 2, 3, 4, 5]
            }
            mlflow_manager.log_artifacts(test_artifacts)
            
            print("✅ MLflow integration test completed successfully!")
            print(f"MLflow UI available at: {mlflow.get_tracking_uri()}")
            
    except Exception as e:
        print(f"❌ MLflow integration test failed: {e}")
        raise
