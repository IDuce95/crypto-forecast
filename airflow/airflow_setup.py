
import os
from pathlib import Path

PROJECT_ROOT = Path('/home/palianm/Desktop/crypto-forecasting')

AIRFLOW_HOME = PROJECT_ROOT / 'airflow'

AIRFLOW_ENV_VARS = {
    'AIRFLOW_HOME': str(AIRFLOW_HOME),
    'AIRFLOW__CORE__DAGS_FOLDER': str(AIRFLOW_HOME / 'dags'),
    'AIRFLOW__CORE__PLUGINS_FOLDER': str(AIRFLOW_HOME / 'plugins'),
    'AIRFLOW__CORE__BASE_LOG_FOLDER': str(AIRFLOW_HOME / 'logs'),
    'AIRFLOW__CORE__EXECUTOR': 'LocalExecutor',
    'AIRFLOW__CORE__LOAD_EXAMPLES': 'False',
    'AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION': 'False',
    'AIRFLOW__CORE__MAX_ACTIVE_RUNS_PER_DAG': '3',
    'AIRFLOW__CORE__MAX_ACTIVE_TASKS_PER_DAG': '10',
    
    'AIRFLOW__DATABASE__SQL_ALCHEMY_CONN': f'sqlite:///{AIRFLOW_HOME}/airflow.db',
    
    'AIRFLOW__WEBSERVER__WEB_SERVER_PORT': '8080',
    'AIRFLOW__WEBSERVER__BASE_URL': 'http://localhost:8080',
    'AIRFLOW__WEBSERVER__EXPOSE_CONFIG': 'True',
    'AIRFLOW__WEBSERVER__RBAC': 'True',
    
    'AIRFLOW__WEBSERVER__SECRET_KEY': 'crypto_forecasting_secret_key_2024',
    
    'AIRFLOW__SCHEDULER__DAG_DIR_LIST_INTERVAL': '30',
    'AIRFLOW__SCHEDULER__CATCHUP_BY_DEFAULT': 'False',
    'AIRFLOW__SCHEDULER__SCHEDULE_AFTER_TASK_EXECUTION': 'True',
    
    'AIRFLOW__LOGGING__LOGGING_LEVEL': 'INFO',
    'AIRFLOW__LOGGING__FAB_LOGGING_LEVEL': 'WARN',
    'AIRFLOW__LOGGING__COLORED_CONSOLE_LOG': 'True',
    
    'AIRFLOW__EMAIL__EMAIL_BACKEND': 'airflow.utils.email.send_email_smtp',
    'AIRFLOW__SMTP__SMTP_HOST': 'localhost',
    'AIRFLOW__SMTP__SMTP_PORT': '587',
    'AIRFLOW__SMTP__SMTP_STARTTLS': 'True',
    'AIRFLOW__SMTP__SMTP_SSL': 'False',
    'AIRFLOW__SMTP__SMTP_MAIL_FROM': 'crypto-forecasting@localhost',
    
    'AIRFLOW__REDIS__REDIS_HOST': 'localhost',
    'AIRFLOW__REDIS__REDIS_PORT': '6379',
    'AIRFLOW__REDIS__REDIS_DB': '1',
    
    'PYTHONPATH': str(PROJECT_ROOT),
}

AIRFLOW_VARIABLES = {
    'crypto_symbols': 'BTC,ETH,ADA,DOT,LINK',
    'data_start_date': '2023-01-01',
    'accuracy_threshold': '0.75',
    'confidence_threshold': '0.80',
    'ml_model_types': 'random_forest,xgboost,lgbm',
    'prediction_horizon_days': '7',
    'cache_ttl_seconds': '3600',
    'max_parallel_tasks': '5',
    'notification_email': 'admin@crypto-forecasting.com',
    'model_retrain_frequency_days': '7',
    'performance_monitoring_window_hours': '24',
}

AIRFLOW_CONNECTIONS = {
    'postgres_default': {
        'conn_type': 'postgres',
        'host': 'localhost',
        'schema': 'crypto_forecasting',
        'login': 'postgres',
        'password': 'password',
        'port': 5432,
    },
    'redis_default': {
        'conn_type': 'redis',
        'host': 'localhost',
        'port': 6379,
        'extra': '{"db": 1}',
    },
    'crypto_api': {
        'conn_type': 'http',
        'host': 'api.coingecko.com',
        'extra': '{"timeout": 30}',
    },
}

def setup_airflow_environment():
    directories = [
        AIRFLOW_HOME,
        AIRFLOW_HOME / 'dags',
        AIRFLOW_HOME / 'plugins',
        AIRFLOW_HOME / 'logs',
        AIRFLOW_HOME / 'config',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

def generate_airflow_cfg():
[core]
dags_folder = {AIRFLOW_HOME}/dags
hostname_callable = airflow.utils.net.get_host_ip_address
default_timezone = utc
executor = LocalExecutor
sql_alchemy_conn = sqlite:///{AIRFLOW_HOME}/airflow.db
sql_alchemy_pool_enabled = True
sql_alchemy_pool_size = 5
sql_alchemy_max_overflow = 10
parallelism = 32
dag_concurrency = 16
dags_are_paused_at_creation = False
max_active_runs_per_dag = 3
load_examples = False
plugins_folder = {AIRFLOW_HOME}/plugins
fernet_key = 

[webserver]
base_url = http://localhost:8080
web_server_host = 0.0.0.0
web_server_port = 8080
web_server_ssl_cert = 
web_server_ssl_key = 
web_server_master_timeout = 120
web_server_worker_timeout = 120
worker_refresh_batch_size = 1
worker_refresh_interval = 30
secret_key = crypto_forecasting_secret_key_2024
workers = 4
worker_class = sync
access_logfile = -
error_logfile = -
expose_config = True
authenticate = False
filter_by_owner = False

[email]
email_backend = airflow.utils.email.send_email_smtp

[smtp]
smtp_host = localhost
smtp_starttls = True
smtp_ssl = False
smtp_port = 587
smtp_mail_from = crypto-forecasting@localhost

[celery]
worker_concurrency = 16

[scheduler]
job_heartbeat_sec = 5
scheduler_heartbeat_sec = 5
run_duration = -1
min_file_process_interval = 0
dag_dir_list_interval = 300
print_stats_interval = 30
child_process_timeout = 60
scheduler_zombie_task_threshold = 300
catchup_by_default = False
max_threads = 2
authenticate = False

[logging]
base_log_folder = {AIRFLOW_HOME}/logs
remote_logging = False
remote_log_conn_id = 
remote_base_log_folder = 
encrypt_s3_logs = False
logging_level = INFO
fab_logging_level = WARN
logging_config_class = 
colored_console_log = True
colored_log_format = [%%(blue)s%%(asctime)s%%(reset)s] {{%%(blue)s%%(filename)s:%%(reset)s%%(lineno)d}} %%(log_color)s%%(levelname)s%%(reset)s - %%(log_color)s%%(message)s%%(reset)s
colored_formatter_class = airflow.utils.log.colored_log.CustomTTYColoredFormatter
log_format = [%%(asctime)s] {{%%(filename)s:%%(lineno)d}} %%(levelname)s - %%(message)s
simple_log_format = %%(asctime)s %%(levelname)s - %%(message)s

[metrics]
statsd_on = False
statsd_host = localhost
statsd_port = 8125
statsd_prefix = airflow

[lineage]
backend = 

[atlas]
sasl_enabled = False
host = 
port = 21000
username = 
password = 

[api]
auth_backend = airflow.api.auth.backend.default

[admin]
hide_sensitive_variable_fields = True
