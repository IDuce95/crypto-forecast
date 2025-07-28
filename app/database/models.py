

from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from datetime import datetime
from typing import Optional

Base = declarative_base()

class CryptocurrencyData(Base):

    __tablename__ = 'cryptocurrency_data'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String(20), nullable=False, index=True)
    name = Column(String(50), nullable=False)
    date = Column(DateTime, nullable=False, index=True)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    volume = Column(Float, nullable=True)
    market_cap = Column(Float, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    predictions = relationship("PredictionResult", back_populates="crypto_data")
    
    def __repr__(self):
        return f"<CryptocurrencyData(symbol={self.symbol}, date={self.date}, close={self.close})>"

class DataSource(Base):

    __tablename__ = 'data_sources'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False, unique=True)
    description = Column(Text, nullable=True)
    url = Column(String(500), nullable=True)
    is_active = Column(Boolean, default=True, nullable=False)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    import_jobs = relationship("DataImportJob", back_populates="source")
    
    def __repr__(self):
        return f"<DataSource(name={self.name}, active={self.is_active})>"

class DataImportJob(Base):

    __tablename__ = 'data_import_jobs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(Integer, ForeignKey('data_sources.id'), nullable=False)
    file_path = Column(String(500), nullable=True)
    symbol = Column(String(20), nullable=False)
    status = Column(String(20), nullable=False, default='pending')
    
    records_processed = Column(Integer, default=0)
    records_inserted = Column(Integer, default=0)
    records_updated = Column(Integer, default=0)
    records_skipped = Column(Integer, default=0)
    
    data_start_date = Column(DateTime, nullable=True)
    data_end_date = Column(DateTime, nullable=True)
    
    error_message = Column(Text, nullable=True)
    validation_errors = Column(Text, nullable=True)
    
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    source = relationship("DataSource", back_populates="import_jobs")
    
    def __repr__(self):
        return f"<DataImportJob(symbol={self.symbol}, status={self.status}, records={self.records_processed})>"

class ModelMetadata(Base):

    __tablename__ = 'model_metadata'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    model_type = Column(String(50), nullable=False)
    version = Column(String(20), nullable=False, default='1.0')
    
    hyperparameters = Column(Text, nullable=True)
    features = Column(Text, nullable=True)
    target_variable = Column(String(50), nullable=False)
    
    training_data_symbol = Column(String(20), nullable=False)
    training_start_date = Column(DateTime, nullable=False)
    training_end_date = Column(DateTime, nullable=False)
    validation_start_date = Column(DateTime, nullable=True)
    validation_end_date = Column(DateTime, nullable=True)
    test_start_date = Column(DateTime, nullable=True)
    test_end_date = Column(DateTime, nullable=True)
    
    training_score = Column(Float, nullable=True)
    validation_score = Column(Float, nullable=True)
    test_score = Column(Float, nullable=True)
    cross_val_score_mean = Column(Float, nullable=True)
    cross_val_score_std = Column(Float, nullable=True)
    
    model_file_path = Column(String(500), nullable=True)
    preprocessor_file_path = Column(String(500), nullable=True)
    
    is_active = Column(Boolean, default=True, nullable=False)
    is_production = Column(Boolean, default=False, nullable=False)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    
    predictions = relationship("PredictionResult", back_populates="model")
    
    def __repr__(self):
        return f"<ModelMetadata(name={self.name}, type={self.model_type}, symbol={self.training_data_symbol})>"

class PredictionResult(Base):

    __tablename__ = 'prediction_results'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('model_metadata.id'), nullable=False)
    crypto_data_id = Column(Integer, ForeignKey('cryptocurrency_data.id'), nullable=True)
    
    prediction_date = Column(DateTime, nullable=False)
    target_date = Column(DateTime, nullable=False)
    predicted_value = Column(Float, nullable=False)
    actual_value = Column(Float, nullable=True)
    
    confidence_interval_lower = Column(Float, nullable=True)
    confidence_interval_upper = Column(Float, nullable=True)
    prediction_probability = Column(Float, nullable=True)
    
    absolute_error = Column(Float, nullable=True)
    squared_error = Column(Float, nullable=True)
    percentage_error = Column(Float, nullable=True)
    
    prediction_horizon_days = Column(Integer, nullable=False)
    input_features = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    model = relationship("ModelMetadata", back_populates="predictions")
    crypto_data = relationship("CryptocurrencyData", back_populates="predictions")
    
    def __repr__(self):
        return f"<PredictionResult(target_date={self.target_date}, predicted={self.predicted_value}, actual={self.actual_value})>"

class ModelPerformanceMetric(Base):

    __tablename__ = 'model_performance_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    model_id = Column(Integer, ForeignKey('model_metadata.id'), nullable=False)
    
    metric_name = Column(String(50), nullable=False)
    metric_value = Column(Float, nullable=False)
    dataset_type = Column(String(20), nullable=False)
    
    evaluation_start_date = Column(DateTime, nullable=False)
    evaluation_end_date = Column(DateTime, nullable=False)
    
    calculated_at = Column(DateTime, default=func.now(), nullable=False)
    
    def __repr__(self):
        return f"<ModelPerformanceMetric(metric={self.metric_name}, value={self.metric_value}, dataset={self.dataset_type})>"
