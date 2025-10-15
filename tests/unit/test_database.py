import pytest
import sys
from pathlib import Path
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'app'))

from database.models import (
    Base, CryptocurrencyData, DataSource, DataImportJob, 
    ModelMetadata, PredictionResult, ModelPerformanceMetric
)
from database.dal import DataAccessLayer
from database.connection import DatabaseConnection


class TestDatabaseModels:
    """Test suite for database models"""
    
    @pytest.fixture
    def test_engine(self):
        """Create an in-memory SQLite database for testing"""
        engine = create_engine('sqlite:///:memory:')
        Base.metadata.create_all(engine)
        return engine
    
    @pytest.fixture
    def test_session(self, test_engine):
        """Create a test database session"""
        Session = sessionmaker(bind=test_engine)
        session = Session()
        yield session
        session.close()
    
    def test_cryptocurrency_data_model(self, test_session):
        """Test CryptocurrencyData model"""
        # Create a new cryptocurrency data record
        crypto_data = CryptocurrencyData(
            symbol='BTC',
            name='Bitcoin',
            date=datetime(2024, 1, 15),
            open=42000.0,
            high=43000.0,
            low=41000.0,
            close=42500.0,
            volume=1000000000.0,
            market_cap=800000000000.0
        )
        
        test_session.add(crypto_data)
        test_session.commit()
        
        # Query the record
        result = test_session.query(CryptocurrencyData).filter_by(symbol='BTC').first()
        
        assert result is not None
        assert result.symbol == 'BTC'
        assert result.name == 'Bitcoin'
        assert result.close == 42500.0
        assert result.created_at is not None
    
    def test_model_metadata(self, test_session):
        """Test ModelMetadata model"""
        # Create model metadata
        model_meta = ModelMetadata(
            name='BTC_RandomForest_v1',
            model_type='random_forest',
            version='1.0',
            hyperparameters='{"n_estimators": 100, "max_depth": 10}',
            features='open,high,low,close,volume,sma_7,sma_30',
            target_variable='close',
            training_data_symbol='BTC',
            training_start_date=datetime(2023, 1, 1),
            training_end_date=datetime(2023, 12, 31),
            training_score=0.95,
            validation_score=0.93,
            test_score=0.92,
            model_file_path='/models/btc_rf_v1.pkl',
            is_active=True,
            is_production=False
        )
        
        test_session.add(model_meta)
        test_session.commit()
        
        # Query the record
        result = test_session.query(ModelMetadata).filter_by(name='BTC_RandomForest_v1').first()
        
        assert result is not None
        assert result.model_type == 'random_forest'
        assert result.training_score == 0.95
        assert result.is_active == True
    
    def test_prediction_result(self, test_session):
        """Test PredictionResult model"""
        # First, create a model metadata
        model_meta = ModelMetadata(
            name='Test_Model',
            model_type='xgboost',
            version='1.0',
            target_variable='close',
            training_data_symbol='ETH',
            training_start_date=datetime(2023, 1, 1),
            training_end_date=datetime(2023, 12, 31)
        )
        test_session.add(model_meta)
        test_session.commit()
        
        # Create prediction result
        prediction = PredictionResult(
            model_id=model_meta.id,
            prediction_date=datetime(2024, 1, 15),
            target_date=datetime(2024, 1, 16),
            predicted_value=3500.0,
            actual_value=3480.0,
            confidence_interval_lower=3400.0,
            confidence_interval_upper=3600.0,
            prediction_probability=0.85,
            absolute_error=20.0,
            squared_error=400.0,
            percentage_error=0.57,
            prediction_horizon_days=1
        )
        
        test_session.add(prediction)
        test_session.commit()
        
        # Query the record
        result = test_session.query(PredictionResult).filter_by(model_id=model_meta.id).first()
        
        assert result is not None
        assert result.predicted_value == 3500.0
        assert result.actual_value == 3480.0
        assert result.percentage_error == 0.57
    
    def test_model_validation(self, test_session):
        """Test model validation and constraints"""
        # Test required fields
        with pytest.raises(Exception):  # Should raise integrity error
            invalid_crypto = CryptocurrencyData(
                symbol=None,  # Required field
                name='Invalid',
                date=datetime.now()
            )
            test_session.add(invalid_crypto)
            test_session.commit()
        
        test_session.rollback()
        
        # Test unique constraints if any
        source1 = DataSource(name='Binance', description='Binance API')
        source2 = DataSource(name='Binance', description='Duplicate name')
        
        test_session.add(source1)
        test_session.commit()
        
        # This should raise an integrity error due to unique constraint
        with pytest.raises(Exception):
            test_session.add(source2)
            test_session.commit()
    
    def test_relationships(self, test_session):
        """Test model relationships"""
        # Create related models
        source = DataSource(name='CoinGecko', description='CoinGecko API')
        test_session.add(source)
        test_session.commit()
        
        import_job = DataImportJob(
            source_id=source.id,
            symbol='BTC',
            status='completed',
            records_processed=1000,
            records_inserted=950,
            records_updated=50
        )
        test_session.add(import_job)
        test_session.commit()
        
        # Test relationship
        assert import_job.source == source
        assert import_job in source.import_jobs


class TestDAL:
    """Test suite for Data Access Layer"""
    
    @pytest.fixture
    def mock_dal(self):
        """Create a mock DAL instance"""
        with patch('database.dal.DatabaseConnection') as mock_conn:
            mock_session = MagicMock()
            mock_conn.return_value.get_session.return_value = mock_session
            dal = DataAccessLayer()
            dal.session = mock_session
            return dal
    
    def test_dal_initialization(self, mock_dal):
        """Test DAL initialization"""
        assert mock_dal is not None
        assert mock_dal.session is not None
    
    def test_crud_operations(self, mock_dal):
        """Test CRUD operations in DAL"""
        # Test Create
        mock_crypto_data = CryptocurrencyData(
            symbol='LTC',
            name='Litecoin',
            date=datetime(2024, 1, 15),
            open=100.0,
            high=110.0,
            low=95.0,
            close=105.0
        )
        
        with patch.object(mock_dal, 'add_cryptocurrency_data') as mock_add:
            mock_add.return_value = mock_crypto_data
            result = mock_dal.add_cryptocurrency_data(mock_crypto_data)
            assert result.symbol == 'LTC'
        
        # Test Read
        with patch.object(mock_dal, 'get_cryptocurrency_data') as mock_get:
            mock_get.return_value = [mock_crypto_data]
            result = mock_dal.get_cryptocurrency_data('LTC')
            assert len(result) == 1
            assert result[0].symbol == 'LTC'
        
        # Test Update
        with patch.object(mock_dal, 'update_cryptocurrency_data') as mock_update:
            mock_crypto_data.close = 108.0
            mock_update.return_value = mock_crypto_data
            result = mock_dal.update_cryptocurrency_data(mock_crypto_data)
            assert result.close == 108.0
        
        # Test Delete
        with patch.object(mock_dal, 'delete_cryptocurrency_data') as mock_delete:
            mock_delete.return_value = True
            result = mock_dal.delete_cryptocurrency_data(mock_crypto_data.id)
            assert result == True
    
    def test_batch_operations(self, mock_dal):
        """Test batch database operations"""
        # Test batch insert
        mock_data_list = [
            CryptocurrencyData(symbol='BTC', name='Bitcoin', date=datetime(2024, 1, i), close=40000 + i*100)
            for i in range(1, 11)
        ]
        
        with patch.object(mock_dal, 'bulk_insert_cryptocurrency_data') as mock_bulk:
            mock_bulk.return_value = len(mock_data_list)
            result = mock_dal.bulk_insert_cryptocurrency_data(mock_data_list)
            assert result == 10
    
    def test_query_operations(self, mock_dal):
        """Test complex query operations"""
        # Test date range query
        with patch.object(mock_dal, 'get_data_by_date_range') as mock_query:
            mock_query.return_value = []
            result = mock_dal.get_data_by_date_range(
                'BTC',
                datetime(2024, 1, 1),
                datetime(2024, 1, 31)
            )
            assert isinstance(result, list)
        
        # Test aggregation query
        with patch.object(mock_dal, 'get_model_performance_stats') as mock_stats:
            mock_stats.return_value = {
                'avg_accuracy': 0.92,
                'best_model': 'xgboost',
                'total_predictions': 1000
            }
            result = mock_dal.get_model_performance_stats()
            assert result['avg_accuracy'] == 0.92
            assert result['best_model'] == 'xgboost'
    
    def test_transaction_handling(self, mock_dal):
        """Test database transaction handling"""
        with patch.object(mock_dal, 'execute_transaction') as mock_transaction:
            mock_transaction.return_value = True
            
            def transaction_operations():
                # Multiple operations that should be atomic
                pass
            
            result = mock_dal.execute_transaction(transaction_operations)
            assert result == True
    
    def test_error_handling(self, mock_dal):
        """Test error handling in DAL"""
        # Test handling of database errors
        with patch.object(mock_dal.session, 'add') as mock_add:
            mock_add.side_effect = Exception("Database error")
            
            with pytest.raises(Exception) as exc_info:
                mock_dal.session.add(CryptocurrencyData())
            
            assert "Database error" in str(exc_info.value)
