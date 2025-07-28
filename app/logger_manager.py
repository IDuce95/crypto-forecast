

import os
import sys
import inspect
from typing import Optional, Any
from pathlib import Path

class LoggerManager:

    _instance: Optional['LoggerManager'] = None
    _app_logger: Optional[Any] = None
    _initialized: bool = False
    
    def __new__(cls) -> 'LoggerManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._setup_logger()
            self._initialized = True
    
    def _setup_logger(self):

        custom_logger_path = os.path.join(os.path.dirname(__file__), 'custom-logger')
        if custom_logger_path not in sys.path:
            sys.path.insert(0, custom_logger_path)

        try:
            import logger_utils.logger as logger_module
            from app.config import config
        except ImportError:
            import logging
            
            class FallbackLogger:
                @classmethod
                def info(cls, message: str, module_name: str = None):
                    logging.info(message)
                
                @classmethod
                def debug(cls, message: str, func_name: str = None, module_name: str = None):
                    logging.debug(f"{func_name}: {message}" if func_name else message)
                
                @classmethod
                def warning(cls, message: str, func_name: str = None, module_name: str = None):
                    logging.warning(f"{func_name}: {message}" if func_name else message)
                
                @classmethod
                def error(cls, error: Exception, error_type=None, func_name: str = None, module_name: str = None):
                    logging.error(f"{func_name}: {error}" if func_name else str(error))
                
                @classmethod
                def result(cls, dataset: str, horizon: str, metric: str, model_type: str,
                           hyperparams: str, module_name: str = None):
                    logging.info(f"RESULT: {dataset}, {horizon}, {metric}, {model_type}, {hyperparams}")
            
            self._app_logger = FallbackLogger
            return
        
        class AppLogger:
            _loggers = {}

            @classmethod
            def get_logger(cls, module_name: str, file_name: Optional[str] = None):
                if module_name not in cls._loggers:
                    if file_name is None:
                        frame = inspect.currentframe().f_back
                        file_name = os.path.basename(frame.f_code.co_filename)

                    log_path = Path(config.logging.file_path)
                    log_path.parent.mkdir(parents=True, exist_ok=True)

                    level_mapping = {
                        "DEBUG": 10,
                        "INFO": 20,
                        "WARNING": 30,
                        "ERROR": 40,
                        "CRITICAL": 50
                    }

                    level = level_mapping.get(config.logging.level, 20)

                    logger = logger_module.get_module_logger(
                        mod_name=module_name,
                        file_name=file_name,
                        log_path=str(log_path),
                        level=level
                    )

                    cls._loggers[module_name] = logger

                return cls._loggers[module_name]

            @classmethod
            def info(cls, message: str, module_name: str = None):
                if module_name is None:
                    frame = inspect.currentframe().f_back
                    module_name = frame.f_code.co_filename

                logger = cls.get_logger(module_name)
                logger_module.log_info(logger, message)

            @classmethod
            def debug(cls, message: str, func_name: str = None, module_name: str = None):
                if module_name is None:
                    frame = inspect.currentframe().f_back
                    module_name = frame.f_code.co_filename

                if func_name is None:
                    frame = inspect.currentframe().f_back
                    func_name = frame.f_code.co_name

                logger = cls.get_logger(module_name)
                logger_module.log_debug(logger, message, func_name)

            @classmethod
            def warning(cls, message: str, func_name: str = None, module_name: str = None):
                if module_name is None:
                    frame = inspect.currentframe().f_back
                    module_name = frame.f_code.co_filename

                if func_name is None:
                    frame = inspect.currentframe().f_back
                    func_name = frame.f_code.co_name

                logger = cls.get_logger(module_name)
                logger_module.log_warning(logger, message, func_name)

            @classmethod
            def error(cls, error: Exception, error_type=None, func_name: str = None, module_name: str = None):
                if module_name is None:
                    frame = inspect.currentframe().f_back
                    module_name = frame.f_code.co_filename

                if func_name is None:
                    frame = inspect.currentframe().f_back
                    func_name = frame.f_code.co_name

                logger = cls.get_logger(module_name)
                code_line = inspect.currentframe().f_back.f_lineno

                logger_module.log_error_no_raise(logger, str(error), func_name, code_line)

            @classmethod
            def result(cls, dataset: str, horizon: str, metric: str, model_type: str,
                       hyperparams: str, module_name: str = None):
                if module_name is None:
                    frame = inspect.currentframe().f_back
                    module_name = frame.f_code.co_filename

                logger = cls.get_logger(module_name)
                logger_module.log_result(logger, dataset, horizon, metric, model_type, hyperparams)
        
        if not os.environ.get('LOGGER_MANAGER_LOGGED'):
            print("LoggerManager: Custom logger initialized")
            os.environ['LOGGER_MANAGER_LOGGED'] = 'true'
        
        self._app_logger = AppLogger
    
    @property
    def logger(self):

        if self._app_logger is None:
            self._setup_logger()
        return self._app_logger()
    
    @classmethod
    def get_logger(cls):

        return cls().logger
    
    @classmethod
    def reset(cls) -> None:

        cls._instance = None
        cls._app_logger = None
        cls._initialized = False

logger_manager = LoggerManager()
logger = logger_manager.logger
