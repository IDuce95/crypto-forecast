
from datetime import datetime, timedelta
from typing import Dict, Any
import sys
import os

sys.path.append('/home/palianm/Desktop/crypto-forecasting')

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.email import EmailOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from app.datalake_manager import DataLakeManager
from app.ml_pipeline import MLPipeline
from app.database_optimizer import DatabaseOptimizer
from app.cache_manager import CacheManager
from app.logger_manager import LoggerManager

DAG_ID = 'crypto_data_pipeline'
SCHEDULE_INTERVAL = '0 4 * * *'  # Daily at 4 AM
START_DATE = days_ago(1)
CATCHUP = False
MAX_ACTIVE_RUNS = 1

default_args = {
    'owner': 'crypto-forecasting-team',
    'depends_on_past': False,
    'start_date': START_DATE,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Cryptocurrency data processing and ML pipeline',
    schedule_interval=SCHEDULE_INTERVAL,
    catchup=CATCHUP,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['crypto', 'ml', 'data-pipeline'],
)

def check_data_sources(**context) -> bool:
    logger = LoggerManager().get_logger('airflow.data_extraction')

    try:
        datalake = DataLakeManager()

        symbols = Variable.get('crypto_symbols', ['BTC', 'ETH', 'ADA'])
        if isinstance(symbols, str):
            symbols = symbols.split(',')

        start_date = Variable.get('data_start_date', '2023-01-01')
        end_date = datetime.now().strftime('%Y-%m-%d')

        extraction_results = {}

        for symbol in symbols:
            try:
                data_path = datalake.get_latest_data_path(symbol)

                if data_path:
                    validation_result = datalake.validate_data_quality(data_path)

                    extraction_results[symbol] = {
                        'status': 'success',
                        'data_path': data_path,
                        'validation': validation_result,
                        'extracted_at': datetime.now().isoformat()
                    }

                    logger.info(f"Successfully extracted and validated data for {symbol}")
                else:
                    extraction_results[symbol] = {
                        'status': 'failed',
                        'error': 'No data available',
                        'extracted_at': datetime.now().isoformat()
                    }
                    logger.warning(f"No data available for {symbol}")

            except Exception as e:
                extraction_results[symbol] = {
                    'status': 'error',
                    'error': str(e),
                    'extracted_at': datetime.now().isoformat()
                }
                logger.error(f"Failed to extract data for {symbol}: {e}")

        context['task_instance'].xcom_push(key='extraction_results', value=extraction_results)

        logger.info(f"Data extraction completed for {len(symbols)} symbols")
        return extraction_results

    except Exception as e:
        logger.error(f"Data extraction failed: {e}")
        raise

def optimize_database(**context) -> Dict[str, Any]:
    logger = LoggerManager().get_logger('airflow.feature_preprocessing')

    try:
        from app.pyspark_manager import PySparkManager, PySparkDataProcessor

        extraction_results = context['task_instance'].xcom_pull(
            task_ids='extract_and_validate_data',
            key='extraction_results'
        )

        spark_manager = PySparkManager(
            app_name="CryptoForecastingPreprocessing",
            executor_memory="2g",
            driver_memory="1g"
        )

        processor = PySparkDataProcessor(spark_manager)

        processing_results = {}

        for symbol, result in extraction_results.items():
            if result['status'] == 'success':
                try:
                    processed_result = processor.process_crypto_pipeline(
                        input_path=result['data_path'],
                        output_path=f"/opt/airflow/data/processed/{symbol}_processed.parquet",
                        symbols=[symbol],
                        start_date=Variable.get('data_start_date', '2023-01-01'),
                        end_date=datetime.now().strftime('%Y-%m-%d')
                    )

                    processing_results[symbol] = processed_result
                    logger.info(f"Successfully processed features for {symbol}")

                except Exception as e:
                    processing_results[symbol] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    logger.error(f"Failed to process features for {symbol}: {e}")
            else:
                processing_results[symbol] = {
                    'status': 'skipped',
                    'reason': 'Extraction failed'
                }

        spark_manager.stop_spark_session()

        logger.info("Feature preprocessing completed")
        return processing_results

    except Exception as e:
        logger.error(f"Feature preprocessing failed: {e}")
        raise

def train_ml_models(**context) -> Dict[str, Any]:
    logger = LoggerManager().get_logger('airflow.prediction_generation')

    try:
        from app.ml_prediction_generator import PredictionGenerator, PredictionConfig

        training_results = context['task_instance'].xcom_pull(
            task_ids='train_ml_models',
            key='training_results'
        )

        symbols = Variable.get('crypto_symbols', ['BTC', 'ETH', 'ADA'])
        if isinstance(symbols, str):
            symbols = symbols.split(',')

        prediction_config = PredictionConfig(
            prediction_horizon=1,
            confidence_level=0.95,
            save_predictions=True
        )

        predictor = PredictionGenerator(prediction_config)

        prediction_results = {}

        for symbol in symbols:
            try:
                predictions = predictor.generate_batch_predictions(
                    symbol=symbol,
                    start_date=datetime.now().strftime('%Y-%m-%d'),
                    end_date=(datetime.now() + timedelta(days=7)).strftime('%Y-%m-%d')
                )

                prediction_results[symbol] = {
                    'status': 'success',
                    'predictions_count': len(predictions.predictions),
                    'generated_at': datetime.now().isoformat()
                }

                logger.info(f"Generated predictions for {symbol}")

            except Exception as e:
                prediction_results[symbol] = {
                    'status': 'error',
                    'error': str(e)
                }
                logger.error(f"Failed to generate predictions for {symbol}: {e}")

        logger.info("Prediction generation completed")
        return prediction_results

    except Exception as e:
        logger.error(f"Prediction generation failed: {e}")
        raise

def update_cache(**context) -> Dict[str, Any]:
    logger = LoggerManager().get_logger('airflow.notifications')

    try:
        training_results = context['task_instance'].xcom_pull(
            task_ids='train_ml_models',
            key='training_results'
        )

        message = f"""
        Crypto ML Pipeline Completed Successfully!

        Experiment ID: {training_results.get('experiment_id', 'N/A')}
        Models Trained: {training_results.get('models_trained', 0)}
        Symbols Processed: {training_results.get('symbols_processed', [])}
        Completed At: {datetime.now().isoformat()}
    <h3>Crypto ML Pipeline Failed</h3>
    <p>The cryptocurrency machine learning pipeline has failed.</p>
    <p>Please check the Airflow logs for more details.</p>
    <p>Run ID: {{ run_id }}</p>
    <p>Failed At: {{ ts }}</p>
