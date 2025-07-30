

import os
from typing import Optional, Generator
from sqlalchemy import create_engine, Engine, text
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError
from contextlib import contextmanager

from app.config import config
from app.logger import logger
from app.database.models import Base

class DatabaseManager:

    _instance: Optional['DatabaseManager'] = None
    _engine: Optional[Engine] = None
    _session_factory: Optional[sessionmaker] = None
    _initialized: bool = False

    def __new__(cls) -> 'DatabaseManager':
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._setup_database()
            self._initialized = True

    def _get_database_url(self) -> str:

        db_url = os.getenv('DATABASE_URL')

        if db_url:
            return db_url

        if hasattr(config, 'database'):
            db_config = config.database
            if db_config.get('url'):
                return db_config.url

            db_type = db_config.get('type', 'sqlite')
            if db_type == 'sqlite':
                db_path = db_config.get('path', './crypto_forecasting.db')
                return f"sqlite:///{db_path}"
            elif db_type == 'postgresql':
                user = db_config.get('user', 'postgres')
                password = db_config.get('password', '')
                host = db_config.get('host', 'localhost')
                port = db_config.get('port', 5432)
                name = db_config.get('name', 'crypto_forecasting')
                return f"postgresql://{user}:{password}@{host}:{port}/{name}"

        return "sqlite:///./crypto_forecasting.db"

    def _setup_database(self) -> None:

        try:
            database_url = self._get_database_url()
            logger.info(f"Connecting to database: {database_url.split('://')[0]}://...")

            if database_url.startswith('sqlite'):
                self._engine = create_engine(
                    database_url,
                    echo=False,
                    connect_args={"check_same_thread": False}
                )
            else:
                self._engine = create_engine(
                    database_url,
                    echo=False,
                    pool_size=5,
                    max_overflow=10,
                    pool_timeout=30,
                    pool_recycle=3600
                )

            self._session_factory = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self._engine
            )

            logger.info("Database connection established successfully")

        except Exception as e:
            logger.error(f"Failed to setup database: {str(e)}")
            raise

    def create_tables(self) -> None:

        try:
            logger.info("Creating database tables...")
            Base.metadata.create_all(bind=self._engine)
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Failed to create tables: {str(e)}")
            raise

    def drop_tables(self) -> None:

        try:
            logger.warning("Dropping all database tables...")
            Base.metadata.drop_all(bind=self._engine)
            logger.info("Database tables dropped successfully")
        except Exception as e:
            logger.error(f"Failed to drop tables: {str(e)}")
            raise

    def get_session(self) -> Session:

        if self._session_factory is None:
            raise RuntimeError("Database not initialized. Call _setup_database() first.")
        return self._session_factory()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:

        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:

        try:
            with self.session_scope() as session:
                from sqlalchemy import text
                session.execute(text("SELECT 1"))
            logger.info("Database connection test successful")
            return True
        except Exception as e:
            logger.error(f"Database connection test failed: {str(e)}")
            return False

    def get_engine(self) -> Engine:

        if self._engine is None:
            raise RuntimeError("Database not initialized.")
        return self._engine

    @classmethod
    def reset(cls) -> None:

        if cls._instance and cls._instance._engine:
            cls._instance._engine.dispose()
        cls._instance = None
        cls._engine = None
        cls._session_factory = None
        cls._initialized = False

db_manager = DatabaseManager()

def get_db_session() -> Session:

    return db_manager.get_session()

def init_database(create_tables: bool = True) -> None:

    logger.info("Initializing database...")

    if not db_manager.test_connection():
        raise RuntimeError("Failed to connect to database")

    if create_tables:
        db_manager.create_tables()

    logger.info("Database initialization completed")

def close_database() -> None:

    if db_manager._engine:
        db_manager._engine.dispose()
        logger.info("Database connections closed")

if __name__ == "__main__":
