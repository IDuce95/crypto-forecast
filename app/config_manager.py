

import os
from typing import Optional
from dynaconf import Dynaconf, Validator
from pathlib import Path

class ConfigManager:

    _instance: Optional['ConfigManager'] = None
    _config: Optional[Dynaconf] = None
    _initialized: bool = False

    def __new__(cls) -> 'ConfigManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._load_config()
            self._initialized = True

    def _create_validators(self) -> list:

        return [
            Validator("rest_api_settings.port", default=5000, is_type_of=int),
            Validator("rest_api_settings.backend_hostname", default="localhost", is_type_of=str),
            Validator("data_settings.processed_data_path", default="data/processed", is_type_of=str),
            Validator("data_settings.raw_data_path", default="data/raw", is_type_of=str),
            Validator("data_settings.date_column", default="Date", is_type_of=str),
            Validator("data_settings.min_data_length_years", default=2, is_type_of=int),
            Validator("model_settings.default_test_size", default=0.2, gte=0.1, lte=0.5),
            Validator("model_settings.default_validation_size", default=0.2, gte=0.1, lte=0.5),
            Validator("model_settings.random_state", default=42, is_type_of=int),
            Validator("logging.level", default="INFO", is_in=["DEBUG", "INFO", "WARNING", "ERROR"]),
            Validator("logging.format", default="%(asctime)s - %(levelname)s - %(message)s", is_type_of=str),
            Validator("logging.file_path", default="./logs/app.log", is_type_of=str),
            Validator("api_endpoints.train_dt_endpoint", default="/train/decision_tree", is_type_of=str),
            Validator("api_endpoints.train_rf_endpoint", default="/train/random_forest", is_type_of=str),
            Validator("api_endpoints.train_xgb_endpoint", default="/train/xgboost", is_type_of=str),
            Validator("api_endpoints.train_lasso_endpoint", default="/train/lasso", is_type_of=str),
            Validator("api_endpoints.optimize_dt_endpoint", default="/optimize/decision_tree", is_type_of=str),
            Validator("api_endpoints.optimize_rf_endpoint", default="/optimize/random_forest", is_type_of=str),
            Validator("api_endpoints.optimize_xgb_endpoint", default="/optimize/xgboost", is_type_of=str),
            Validator("api_endpoints.optimize_lasso_endpoint", default="/optimize/lasso", is_type_of=str),
            Validator("other.gitkeep_filename", default=".gitkeep", is_type_of=str),
        ]

    def _ensure_directory_exists(self, path: Path) -> None:

        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> None:

        project_root = Path(__file__).parent.parent.parent

        self._config = Dynaconf(
            envvar_prefix="CRYPTO_FORECASTING",
            settings_files=[
                str(project_root / "app" / "settings.toml"),
            ],
            environments=True,
            env_switcher="ENV_FOR_DYNACONF",
            load_dotenv=True,
            dotenv_path=str(project_root / ".env.dev"),
            merge_enabled=True,
            validators=self._create_validators()
        )

        self._validate_config()
        self._initialize_logging()

    def _validate_config(self) -> None:

        try:
            self._ensure_directory_exists(Path(self._config.data_settings.processed_data_path))
            self._ensure_directory_exists(Path(self._config.data_settings.raw_data_path))

            if hasattr(self._config, 'MODEL_STORAGE_PATH'):
                self._ensure_directory_exists(Path(self._config.MODEL_STORAGE_PATH))

            if hasattr(self._config.logging, 'file_path'):
                self._ensure_directory_exists(Path(self._config.logging.file_path).parent)
            else:
                self._ensure_directory_exists(Path("logs"))

        except Exception as e:
            raise ValueError(f"Configuration validation failed: {e}")

    def _initialize_logging(self) -> None:

        try:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logger = logging.getLogger(__name__)

            if not os.environ.get('CONFIG_LOGGED'):
                logger.info(f"Configuration loaded successfully for environment: {self._config.current_env}")
                os.environ['CONFIG_LOGGED'] = 'true'
        except Exception as e:
            print(f"Warning: Could not initialize basic logging: {e}")

    @property
    def config(self) -> Dynaconf:

        if self._config is None:
            self._load_config()
        return self._config

    @classmethod
    def get_config(cls) -> Dynaconf:

        return cls().config

    @classmethod
    def reset(cls) -> None:

        cls._instance = None
        cls._config = None
        cls._initialized = False

config_manager = ConfigManager()
config = config_manager.config

get_config = ConfigManager.get_config

def validate_config(config) -> bool:

    return True

def initialize_logging(config) -> None:

    pass
