
from datetime import datetime, timedelta
from typing import Dict, Any
import sys

sys.path.append('/home/palianm/Desktop/crypto-forecasting')

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.sensors.filesystem import FileSensor
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.trigger_rule import TriggerRule

from app.real_time_serving import RealTimePredictor
from app.performance_monitoring import PerformanceMonitor
from app.cache_manager import CacheManager
from app.logger_manager import LoggerManager

DAG_ID = 'crypto_realtime_monitoring'
SCHEDULE_INTERVAL = '*/15 * * * *'  # Every 15 minutes
START_DATE = days_ago(1)
CATCHUP = False
MAX_ACTIVE_RUNS = 3

default_args = {
    'owner': 'crypto-forecasting-team',
    'depends_on_past': False,
    'start_date': START_DATE,
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
    'execution_timeout': timedelta(minutes=10),
}

dag = DAG(
    DAG_ID,
    default_args=default_args,
    description='Real-time cryptocurrency monitoring and predictions',
    schedule_interval=SCHEDULE_INTERVAL,
    catchup=CATCHUP,
    max_active_runs=MAX_ACTIVE_RUNS,
    tags=['crypto', 'realtime', 'monitoring'],
)


def check_market_hours(**context) -> bool:
    logger = LoggerManager().get_logger('airflow.market_data')
    
    try:
        symbols = Variable.get('crypto_symbols', ['BTC', 'ETH', 'ADA'])
        if isinstance(symbols, str):
            symbols = symbols.split(',')
        
        market_data = {}
        
        for symbol in symbols:
            try:
                market_data[symbol] = {
                    'price': 50000.0,  # Placeholder
                    'volume': 1000000.0,  # Placeholder
                    'timestamp': datetime.now().isoformat(),
                    'status': 'active'
                }
                
                logger.info(f"Fetched market data for {symbol}")
                
            except Exception as e:
                market_data[symbol] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                logger.error(f"Failed to fetch data for {symbol}: {e}")
        
        context['task_instance'].xcom_push(key='market_data', value=market_data)
        
        logger.info(f"Market data fetched for {len(symbols)} symbols")
        return market_data
        
    except Exception as e:
        logger.error(f"Failed to fetch market data: {e}")
        raise


def generate_realtime_predictions(**context) -> Dict[str, Any]:
    logger = LoggerManager().get_logger('airflow.performance_monitoring')
    
    try:
        monitor = PerformanceMonitor()
        
        symbols = Variable.get('crypto_symbols', ['BTC', 'ETH', 'ADA'])
        if isinstance(symbols, str):
            symbols = symbols.split(',')
        
        performance_results = {}
        
        for symbol in symbols:
            try:
                metrics = monitor.calculate_prediction_accuracy(
                    symbol=symbol,
                    time_window_hours=24
                )
                
                performance_results[symbol] = {
                    'accuracy': metrics.get('accuracy', 0.0),
                    'mae': metrics.get('mae', 0.0),
                    'rmse': metrics.get('rmse', 0.0),
                    'evaluated_at': datetime.now().isoformat(),
                    'status': 'success'
                }
                
                accuracy_threshold = Variable.get('accuracy_threshold', 0.7)
                if metrics.get('accuracy', 0.0) < float(accuracy_threshold):
                    logger.warning(f"Model performance below threshold for {symbol}: {metrics.get('accuracy', 0.0)}")
                
                logger.info(f"Performance monitored for {symbol}")
                
            except Exception as e:
                performance_results[symbol] = {
                    'status': 'error',
                    'error': str(e),
                    'evaluated_at': datetime.now().isoformat()
                }
                logger.error(f"Failed to monitor performance for {symbol}: {e}")
        
        context['task_instance'].xcom_push(key='performance_metrics', value=performance_results)
        
        logger.info("Model performance monitoring completed")
        return performance_results
        
    except Exception as e:
        logger.error(f"Performance monitoring failed: {e}")
        raise


def update_realtime_cache(**context) -> Dict[str, Any]:
    logger = LoggerManager().get_logger('airflow.alerts')
    
    try:
        performance_metrics = context['task_instance'].xcom_pull(
            task_ids='monitor_model_performance',
            key='performance_metrics'
        )
        
        predictions = context['task_instance'].xcom_pull(
            task_ids='generate_realtime_predictions',
            key='predictions'
        )
        
        alerts = {}
        
        accuracy_threshold = float(Variable.get('accuracy_threshold', 0.7))
        prediction_confidence_threshold = float(Variable.get('confidence_threshold', 0.8))
        
        for symbol in performance_metrics.keys():
            symbol_alerts = []
            
            accuracy = performance_metrics.get(symbol, {}).get('accuracy', 1.0)
            if accuracy < accuracy_threshold:
                symbol_alerts.append({
                    'type': 'low_accuracy',
                    'message': f"Model accuracy below threshold: {accuracy:.3f} < {accuracy_threshold}",
                    'severity': 'warning'
                })
            
            confidence = predictions.get(symbol, {}).get('confidence', 1.0)
            if confidence < prediction_confidence_threshold:
                symbol_alerts.append({
                    'type': 'low_confidence',
                    'message': f"Prediction confidence below threshold: {confidence:.3f} < {prediction_confidence_threshold}",
                    'severity': 'info'
                })
            
            if predictions.get(symbol, {}).get('status') == 'error':
                symbol_alerts.append({
                    'type': 'prediction_error',
                    'message': f"Prediction generation failed: {predictions[symbol].get('error', 'Unknown error')}",
                    'severity': 'error'
                })
            
            alerts[symbol] = symbol_alerts
        
        total_alerts = sum(len(symbol_alerts) for symbol_alerts in alerts.values())
        
        alert_summary = {
            'total_alerts': total_alerts,
            'alerts_by_symbol': alerts,
            'checked_at': datetime.now().isoformat()
        }
        
        if total_alerts > 0:
            logger.warning(f"Found {total_alerts} alerts across all symbols")
        else:
            logger.info("No alerts found")
        
        return alert_summary
        
    except Exception as e:
        logger.error(f"Alert checking failed: {e}")
        raise


check_market_task = PythonOperator(
    task_id='check_market_hours',
    python_callable=check_market_hours,
    dag=dag,
    doc_md="Check if crypto markets are in high activity period"
)

fetch_data_task = PythonOperator(
    task_id='fetch_latest_market_data',
    python_callable=fetch_latest_market_data,
    dag=dag,
    doc_md="Fetch latest market data for all symbols"
)

predict_task = PythonOperator(
    task_id='generate_realtime_predictions',
    python_callable=generate_realtime_predictions,
    dag=dag,
    doc_md="Generate real-time predictions using cached models"
)

monitor_task = PythonOperator(
    task_id='monitor_model_performance',
    python_callable=monitor_model_performance,
    dag=dag,
    doc_md="Monitor model performance and accuracy"
)

cache_update_task = PythonOperator(
    task_id='update_realtime_cache',
    python_callable=update_realtime_cache,
    dag=dag,
    doc_md="Update cache with latest predictions and metrics"
)

alerts_task = PythonOperator(
    task_id='check_alerts',
    python_callable=check_alerts,
    dag=dag,
    doc_md="Check for alerts and anomalies"
)

health_check_task = BashOperator(
    task_id='system_health_check',
    bash_command="""
    echo "=== System Health Check ==="
    echo "Disk Usage:"
    df -h
    echo "Memory Usage:"
    free -h
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)"
    echo "Redis Status:"
    redis-cli ping || echo "Redis not available"
    echo "Health check completed"
