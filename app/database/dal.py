

import pandas as pd
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, or_, desc, asc, func
from pathlib import Path
import json

from app.database.connection import db_manager
from app.database.models import (
    CryptocurrencyData, DataSource, DataImportJob,
    ModelMetadata, PredictionResult, ModelPerformanceMetric
)
from app.logger import logger
from app.data_validator import DataValidator, ValidationResult

class CryptocurrencyDataDAL:

    @staticmethod
    def get_symbols() -> List[str]:

        with db_manager.session_scope() as session:
            symbols = session.query(CryptocurrencyData.symbol).distinct().all()
            return [symbol[0] for symbol in symbols]

    @staticmethod
    def get_data_by_symbol(
        symbol: str,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> pd.DataFrame:

        with db_manager.session_scope() as session:
            query = session.query(CryptocurrencyData).filter(
                CryptocurrencyData.symbol == symbol.upper()
            )

            if start_date:
                query = query.filter(CryptocurrencyData.date >= start_date)

            if end_date:
                query = query.filter(CryptocurrencyData.date <= end_date)

            query = query.order_by(CryptocurrencyData.date)

            if limit:
                query = query.limit(limit)

            results = query.all()

            if not results:
                logger.warning(f"No data found for symbol: {symbol}")
                return pd.DataFrame()

            data = []
            for record in results:
                data.append({
                    'Date': record.date,
                    'Open': record.open,
                    'High': record.high,
                    'Low': record.low,
                    'Close': record.close,
                    'Volume': record.volume,
                    'Market Cap': record.market_cap
                })

            df = pd.DataFrame(data)
            logger.info(f"Retrieved {len(df)} records for {symbol}")
            return df

    @staticmethod
    def get_date_range(symbol: str) -> Tuple[Optional[datetime], Optional[datetime]]:

        with db_manager.session_scope() as session:
            result = session.query(
                func.min(CryptocurrencyData.date),
                func.max(CryptocurrencyData.date)
            ).filter(CryptocurrencyData.symbol == symbol.upper()).first()

            return result if result else (None, None)

    @staticmethod
    def insert_data_from_dataframe(
        df: pd.DataFrame,
        symbol: str,
        name: str,
        source_id: Optional[int] = None
    ) -> int:

        logger.info(f"Inserting data for {symbol} - {len(df)} records")

        inserted_count = 0
        with db_manager.session_scope() as session:
            for _, row in df.iterrows():
                existing = session.query(CryptocurrencyData).filter(
                    and_(
                        CryptocurrencyData.symbol == symbol.upper(),
                        CryptocurrencyData.date == pd.to_datetime(row['Date'])
                    )
                ).first()

                if existing:
                    existing.open = float(row['Open'])
                    existing.high = float(row['High'])
                    existing.low = float(row['Low'])
                    existing.close = float(row['Close'])
                    existing.volume = float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None
                    existing.market_cap = float(row.get('Market Cap', 0)) if pd.notna(row.get('Market Cap')) else None
                    existing.updated_at = datetime.utcnow()
                else:
                    record = CryptocurrencyData(
                        symbol=symbol.upper(),
                        name=name,
                        date=pd.to_datetime(row['Date']),
                        open=float(row['Open']),
                        high=float(row['High']),
                        low=float(row['Low']),
                        close=float(row['Close']),
                        volume=float(row.get('Volume', 0)) if pd.notna(row.get('Volume')) else None,
                        market_cap=float(row.get('Market Cap', 0)) if pd.notna(row.get('Market Cap')) else None
                    )
                    session.add(record)
                    inserted_count += 1

        logger.info(f"Inserted {inserted_count} new records for {symbol}")
        return inserted_count

    @staticmethod
    def delete_data_by_symbol(symbol: str) -> int:

        with db_manager.session_scope() as session:
            deleted_count = session.query(CryptocurrencyData).filter(
                CryptocurrencyData.symbol == symbol.upper()
            ).delete()

            logger.info(f"Deleted {deleted_count} records for {symbol}")
            return deleted_count

class DataImportDAL:

    @staticmethod
    def create_data_source(name: str, description: str = None, url: str = None) -> int:

        with db_manager.session_scope() as session:
            source = DataSource(
                name=name,
                description=description,
                url=url
            )
            session.add(source)
            session.flush()
            source_id = source.id
            logger.info(f"Created data source: {name} (ID: {source_id})")
            return source_id

    @staticmethod
    def get_or_create_data_source(name: str, description: str = None, url: str = None) -> int:

        with db_manager.session_scope() as session:
            source = session.query(DataSource).filter(DataSource.name == name).first()

            if source:
                return source.id

            source = DataSource(
                name=name,
                description=description,
                url=url
            )
            session.add(source)
            session.flush()
            return source.id

    @staticmethod
    def create_import_job(
        source_id: int,
        symbol: str,
        file_path: str = None
    ) -> int:

        with db_manager.session_scope() as session:
            job = DataImportJob(
                source_id=source_id,
                symbol=symbol.upper(),
                file_path=file_path,
                status='pending',
                started_at=datetime.utcnow()
            )
            session.add(job)
            session.flush()
            job_id = job.id
            logger.info(f"Created import job for {symbol} (ID: {job_id})")
            return job_id

    @staticmethod
    def update_import_job_status(
        job_id: int,
        status: str,
        records_processed: int = 0,
        records_inserted: int = 0,
        error_message: str = None,
        validation_result: ValidationResult = None
    ) -> None:

        with db_manager.session_scope() as session:
            job = session.query(DataImportJob).filter(DataImportJob.id == job_id).first()

            if job:
                job.status = status
                job.records_processed = records_processed
                job.records_inserted = records_inserted
                job.error_message = error_message

                if validation_result:
                    job.validation_errors = json.dumps({
                        'errors': validation_result.errors,
                        'warnings': validation_result.warnings
                    })

                if status == 'completed':
                    job.completed_at = datetime.utcnow()

                logger.info(f"Updated import job {job_id}: {status}")

class DataMigrationService:

    def __init__(self):
        self.validator = DataValidator()
        self.crypto_dal = CryptocurrencyDataDAL()
        self.import_dal = DataImportDAL()

    def migrate_csv_file(self, csv_path: Path, symbol: str = None, name: str = None) -> Dict[str, Any]:

        if symbol is None:
            symbol = csv_path.stem.upper()
        if name is None:
            name = csv_path.stem

        logger.info(f"Starting migration for {csv_path}")

        source_id = self.import_dal.get_or_create_data_source(
            name="csv_import",
            description="Data imported from CSV files"
        )

        job_id = self.import_dal.create_import_job(
            source_id=source_id,
            symbol=symbol,
            file_path=str(csv_path)
        )

        try:
            validation_result = self.validator.validate_csv_file(csv_path, symbol)

            if not validation_result.is_valid:
                error_msg = f"Validation failed: {'; '.join(validation_result.errors)}"
                self.import_dal.update_import_job_status(
                    job_id, 'failed', error_message=error_msg,
                    validation_result=validation_result
                )
                return {
                    'success': False,
                    'error': error_msg,
                    'validation_result': validation_result
                }

            df = pd.read_csv(csv_path)

            inserted_count = self.crypto_dal.insert_data_from_dataframe(
                df, symbol, name, source_id
            )

            self.import_dal.update_import_job_status(
                job_id, 'completed',
                records_processed=len(df),
                records_inserted=inserted_count,
                validation_result=validation_result
            )

            return {
                'success': True,
                'symbol': symbol,
                'records_processed': len(df),
                'records_inserted': inserted_count,
                'validation_result': validation_result
            }

        except Exception as e:
            error_msg = str(e)
            logger.error(f"Migration failed for {csv_path}: {error_msg}")

            self.import_dal.update_import_job_status(
                job_id, 'failed', error_message=error_msg
            )

            return {
                'success': False,
                'error': error_msg
            }

    def migrate_all_csv_files(self, data_dir: Path = None) -> Dict[str, Any]:

        if data_dir is None:
            from app.config import config
            data_dir = Path(config.data_settings.processed_data_path)

        logger.info(f"Starting migration of all CSV files from {data_dir}")

        csv_files = list(data_dir.glob("*.csv"))
        if not csv_files:
            logger.warning(f"No CSV files found in {data_dir}")
            return {'success': False, 'error': 'No CSV files found'}

        results = {
            'success': True,
            'total_files': len(csv_files),
            'successful_migrations': 0,
            'failed_migrations': 0,
            'details': {}
        }

        for csv_file in csv_files:
            symbol = csv_file.stem
            result = self.migrate_csv_file(csv_file, symbol=symbol)

            results['details'][symbol] = result

            if result['success']:
                results['successful_migrations'] += 1
            else:
                results['failed_migrations'] += 1

        logger.info(
            f"Migration completed: {results['successful_migrations']}/{results['total_files']} successful"
        )

        return results

def migrate_csv_data_to_database() -> None:

    migration_service = DataMigrationService()
    results = migration_service.migrate_all_csv_files()

    print(f"Migration Results:")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful_migrations']}")
    print(f"Failed: {results['failed_migrations']}")

    for symbol, details in results['details'].items():
        if details['success']:
            print(f"✓ {symbol}: {details['records_inserted']} records inserted")
        else:
            print(f"✗ {symbol}: {details.get('error', 'Unknown error')}")

if __name__ == "__main__":
    """Test DAL operations."""
    from app.database.connection import init_database

    init_database(create_tables=True)

    migrate_csv_data_to_database()
