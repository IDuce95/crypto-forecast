
import redis
import json
import pickle
import hashlib
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta
import pandas as pd
from pathlib import Path
import sys
import logging

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from config_manager import ConfigManager
    from logger_manager import LoggerManager
except ImportError:
    # Fallback if logger_manager doesn't exist
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)


class CacheManager:
    """Redis cache manager for the crypto forecasting application"""
    
    def __init__(self, redis_host: str = None, redis_port: int = None, redis_db: int = None):
        """Initialize Redis cache manager
        
        Args:
            redis_host: Redis server host
            redis_port: Redis server port  
            redis_db: Redis database number
        """
        try:
            config = ConfigManager.get_config()
            self.redis_host = redis_host or config.cache.host
            self.redis_port = redis_port or config.cache.port
            self.redis_db = redis_db or config.cache.db
            self.default_ttl = config.cache.default_ttl_hours * 3600
        except:
            # Fallback defaults
            self.redis_host = redis_host or 'localhost'
            self.redis_port = redis_port or 6379
            self.redis_db = redis_db or 0
            self.default_ttl = 24 * 3600
        
        self.redis_client = None
        self.logger = logging.getLogger(__name__)
        self._connect()
    
    def _connect(self):
        """Establish connection to Redis"""
        try:
            self.redis_client = redis.Redis(
                host=self.redis_host,
                port=self.redis_port,
                db=self.redis_db,
                decode_responses=False
            )
            # Test connection
            self.redis_client.ping()
            self.logger.info(f"Connected to Redis at {self.redis_host}:{self.redis_port}")
        except Exception as e:
            self.logger.warning(f"Failed to connect to Redis: {e}. Cache disabled.")
            self.redis_client = None
    
    def _generate_cache_key(self, prefix: str, **kwargs) -> str:
        """Generate cache key from parameters
        
        Args:
            prefix: Key prefix (e.g., 'prediction', 'model', 'data')
            **kwargs: Parameters to include in key
            
        Returns:
            Generated cache key
        """
        params_str = json.dumps(kwargs, sort_keys=True)
        params_hash = hashlib.md5(params_str.encode()).hexdigest()
        return f"crypto_forecast:{prefix}:{params_hash}"
    
    def cache_predictions(self, predictions: pd.DataFrame, symbol: str, 
                         model_type: str, config_params: Dict, ttl_hours: int = None) -> bool:
        """Cache prediction results
        
        Args:
            predictions: Prediction DataFrame
            symbol: Cryptocurrency symbol
            model_type: ML model type
            config_params: Model configuration parameters
            ttl_hours: Cache TTL in hours
            
        Returns:
            True if cached successfully, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            key = self._generate_cache_key(
                'prediction',
                symbol=symbol,
                model_type=model_type,
                **config_params
            )
            
            ttl = (ttl_hours * 3600) if ttl_hours else self.default_ttl
            data = pickle.dumps(predictions)
            self.redis_client.setex(key, ttl, data)
            
            self.logger.info(f"Cached predictions for {symbol}/{model_type}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to cache predictions: {e}")
            return False
    
    def get_cached_predictions(self, symbol: str, model_type: str, 
                              config_params: Dict) -> Optional[pd.DataFrame]:
        """Retrieve cached prediction
        
        Args:
            symbol: Cryptocurrency symbol
            model_type: ML model type
            config_params: Model configuration parameters
            
        Returns:
            Cached predictions DataFrame or None if not found
        """
        if not self.redis_client:
            return None
        
        try:
            key = self._generate_cache_key(
                'prediction',
                symbol=symbol,
                model_type=model_type,
                **config_params
            )
            
            data = self.redis_client.get(key)
            if data:
                self.logger.info(f"Cache hit for {symbol}/{model_type} predictions")
                return pickle.loads(data)
            return None
        except Exception as e:
            self.logger.error(f"Failed to retrieve cached predictions: {e}")
            return None
    
    def invalidate_pattern(self, pattern: str) -> int:
        """Invalidate cache entries matching pattern
        
        Args:
            pattern: Redis key pattern to match
            
        Returns:
            Number of keys deleted
        """
        if not self.redis_client:
            return 0
        
        try:
            keys = self.redis_client.keys(pattern)
            if keys:
                deleted = self.redis_client.delete(*keys)
                self.logger.info(f"Invalidated {deleted} cache entries matching {pattern}")
                return deleted
            return 0
        except Exception as e:
            self.logger.error(f"Failed to invalidate cache: {e}")
            return 0
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics
        
        Returns:
            Dictionary with cache statistics
        """
        if not self.redis_client:
            return {"enabled": False}
        
        try:
            info = self.redis_client.info()
            return {
                "enabled": True,
                "hits": info.get('keyspace_hits', 0),
                "misses": info.get('keyspace_misses', 0),
                "memory_used_mb": info.get('used_memory', 0) / 1024 / 1024,
                "total_keys": self.redis_client.dbsize()
            }
        except Exception as e:
            self.logger.error(f"Failed to get cache stats: {e}")
            return {"enabled": False, "error": str(e)}
    
    def health_check(self) -> bool:
        """Check if Redis cache is healthy
        
        Returns:
            True if Redis is accessible, False otherwise
        """
        if not self.redis_client:
            return False
        
        try:
            self.redis_client.ping()
            return True
        except:
            return False

# Create singleton instance
cache_manager = CacheManager()
