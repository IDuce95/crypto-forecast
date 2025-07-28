

import os
from typing import Optional
from app.backend.services.data_manager import DataManager
from app.backend.services.metrics_calculator import MetricsCalculator

class ServiceRegistry:

    _instance: Optional['ServiceRegistry'] = None
    _data_manager: Optional[DataManager] = None
    _metrics_calculator: Optional[MetricsCalculator] = None
    
    def __new__(cls) -> 'ServiceRegistry':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def get_data_manager(self) -> DataManager:

        if self._data_manager is None:
            self._data_manager = DataManager()
            if not os.environ.get('DATA_MANAGER_LOGGED'):
                from app.logger import logger
                logger.info("DataManager instance created in ServiceRegistry")
                os.environ['DATA_MANAGER_LOGGED'] = 'true'
        return self._data_manager
    
    def get_metrics_calculator(self) -> MetricsCalculator:

        if self._metrics_calculator is None:
            self._metrics_calculator = MetricsCalculator()
        return self._metrics_calculator
    
    @classmethod
    def reset(cls) -> None:

        cls._instance = None
        cls._data_manager = None
        cls._metrics_calculator = None

registry = ServiceRegistry()
