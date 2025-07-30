
from typing import Dict, List, Optional, Any, Tuple
from sqlalchemy import create_engine, text, Index, inspect
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool
from sqlalchemy.engine import Engine
import time
import pandas as pd
from contextlib import contextmanager
import logging
from datetime import datetime, timedelta

from .logger_manager import LoggerManager
from .database.connection import get_database_connection
from .database.models import Base, CryptocurrencyData, ModelMetadata, PredictionResult
from .cache_manager import CacheManager

class DatabaseOptimizer:
    
    def __init__(self, engine: Optional[Engine] = None, cache_manager: Optional[CacheManager] = None):
        self.logger = LoggerManager().get_logger(self.__class__.__name__)
        self.engine = engine or self._create_optimized_engine()
        self.cache_manager = cache_manager or CacheManager()
        self.Session = sessionmaker(bind=self.engine)
        self.query_performance_log = []
        
    def _create_optimized_engine(self) -> Engine:
        session = self.Session()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            self.logger.error(f"Database session error: {e}")
            raise
        finally:
            session.close()
    
    def create_database_indexes(self) -> Dict[str, bool]:
        try:
            results = {}
            
            with self.get_session() as session:
                inspector = inspect(self.engine)
                existing_indexes = {}
                for table_name in inspector.get_table_names():
                    existing_indexes[table_name] = [idx['name'] for idx in inspector.get_indexes(table_name)]
                
                indexes_to_create = [
                    {
                        'table': 'cryptocurrency_data',
                        'name': 'idx_crypto_symbol_date',
                        'columns': ['symbol', 'date'],
                        'unique': False
                    },
                    {
                        'table': 'cryptocurrency_data',
                        'name': 'idx_crypto_date_desc',
                        'columns': ['date DESC'],
                        'unique': False
                    },
                    {
                        'table': 'cryptocurrency_data',
                        'name': 'idx_crypto_volume',
                        'columns': ['volume'],
                        'unique': False
                    },
                    {
                        'table': 'model_metadata',
                        'name': 'idx_model_type_symbol',
                        'columns': ['model_type', 'training_data_symbol'],
                        'unique': False
                    },
                    {
                        'table': 'model_metadata',
                        'name': 'idx_model_created_at',
                        'columns': ['created_at DESC'],
                        'unique': False
                    },
                    {
                        'table': 'prediction_results',
                        'name': 'idx_pred_model_symbol',
                        'columns': ['model_id', 'crypto_data_id'],
                        'unique': False
                    },
                    {
                        'table': 'prediction_results',
                        'name': 'idx_pred_created_at',
                        'columns': ['created_at DESC'],
                        'unique': False
                    }
                ]
                
                for idx_config in indexes_to_create:
                    table_name = idx_config['table']
                    idx_name = idx_config['name']
                    
                    if idx_name not in existing_indexes.get(table_name, []):
                        try:
                            columns_str = ', '.join(idx_config['columns'])
                            unique_str = 'UNIQUE' if idx_config['unique'] else ''
                            
                            sql = f"""
                            CREATE {unique_str} INDEX IF NOT EXISTS {idx_name} 
                            ON {table_name} ({columns_str})
        try:
            with self.get_session() as session:
                optimization_commands = [
                    "PRAGMA optimize",
                    "PRAGMA journal_mode = WAL",
                    "PRAGMA synchronous = NORMAL",
                    "PRAGMA cache_size = -64000",  # 64MB cache
                    "PRAGMA temp_store = MEMORY",
                    "PRAGMA mmap_size = 268435456",  # 256MB mmap
                    "PRAGMA page_size = 4096"
                ]
                
                for cmd in optimization_commands:
                    try:
                        session.execute(text(cmd))
                        self.logger.debug(f"Executed optimization: {cmd}")
                    except Exception as e:
                        self.logger.warning(f"Optimization command failed {cmd}: {e}")
                
                session.commit()
                
            self.logger.info("Database optimization commands executed")
            
        except Exception as e:
            self.logger.error(f"Failed to optimize database: {e}")
            raise
    
    def measure_query_performance(self, query: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        try:
            start_time = time.time()
            
            with self.get_session() as session:
                if params:
                    result = session.execute(text(query), params)
                else:
                    result = session.execute(text(query))
                
                rows = result.fetchall()
                
            end_time = time.time()
            execution_time = end_time - start_time
            
            performance_metrics = {
                "query": query[:100] + "..." if len(query) > 100 else query,
                "execution_time_seconds": execution_time,
                "rows_returned": len(rows),
                "timestamp": datetime.now().isoformat(),
                "params": params
            }
            
            self.query_performance_log.append(performance_metrics)
            
            self.logger.debug(f"Query performance: {execution_time:.3f}s, {len(rows)} rows")
            return performance_metrics
            
        except Exception as e:
            self.logger.error(f"Failed to measure query performance: {e}")
            raise
    
    def get_cached_crypto_data(self, symbol: str, start_date: str, end_date: str, 
                              cache_ttl: int = 3600) -> Optional[pd.DataFrame]:
        try:
            cache_key = f"crypto_data:{symbol}:{start_date}:{end_date}"
            
            cached_data = self.cache_manager.get_data(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Cache hit for crypto data: {symbol}")
                return pd.DataFrame(cached_data)
            
            query = """
            SELECT symbol, date, open, high, low, close, volume, market_cap
            FROM cryptocurrency_data 
            WHERE symbol = :symbol 
            AND date >= :start_date 
            AND date <= :end_date
            ORDER BY date ASC
        Optimized bulk insert for cryptocurrency data
        
        Args:
            data_records: List of data records to insert
            batch_size: Number of records per batch
            
        Returns:
            Insert statistics
        Generate database performance report
        
        Returns:
            Performance analysis report
                    SELECT name, 
                           COUNT(*) as row_count
                    FROM sqlite_master 
                    WHERE type='table' AND name NOT LIKE 'sqlite_%'
        Analyze query execution plan
        
        Args:
            query: SQL query to analyze
            params: Query parameters
            
        Returns:
            Query plan analysis
        Clean up old data to maintain database performance
        
        Args:
            days_to_keep: Number of days of data to keep
            
        Returns:
            Cleanup statistics
