
import redis
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config_manager import ConfigManager
from logger_manager import LoggerManager


class CacheManager:
        Initialize Redis cache manager
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
        Generate cache key from parameters
        
        Args:
            prefix: Key prefix (e.g., 'prediction', 'model', 'data')
            **kwargs: Parameters to include in key
            
        Returns:
            Generated cache key
        Cache prediction results
        
        Args:
            predictions: Prediction DataFrame
            symbol: Cryptocurrency symbol
            model_type: ML model type
            config_params: Model configuration parameters
            ttl_hours: Cache TTL in hours
            
        Returns:
            True if cached successfully, False otherwise
        Retrieve cached prediction
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: ML model type
            config_params: Model configuration parameters
            
        Returns:
            Cached predictions DataFrame or None if not found
        Cache model metadata (metrics, parameters, etc.)
        
        Args:
            model_metadata: Model metadata dictionary
            symbol: Cryptocurrency symbol
            model_type: ML model type
            ttl_hours: Cache TTL in hours
            
        Returns:
            True if cached successfully, False otherwise
        Retrieve cached model metadata
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: ML model type
            
        Returns:
            Cached model metadata or None if not found
        Cache preprocessed data
        
        Args:
            data: Processed DataFrame
            symbol: Cryptocurrency symbol
            preprocessing_config: Preprocessing configuration
            ttl_hours: Cache TTL in hours
            
        Returns:
            True if cached successfully, False otherwise
        Retrieve cached processed data
        
        Args:
            symbol: Cryptocurrency symbol
            preprocessing_config: Preprocessing configuration
            
        Returns:
            Cached processed DataFrame or None if not found
        Invalidate cache entries matching pattern
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        Check if Redis cache is healthy
        
        Returns:
            True if Redis is accessible, False otherwise
