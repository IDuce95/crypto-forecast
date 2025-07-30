import pytest
import tempfile
import os
from pathlib import Path
from unittest.mock import Mock


@pytest.fixture(scope="session")
def temp_dir():
    return Path(__file__).parent / "data"


@pytest.fixture
def mock_config():
    import pandas as pd
    import numpy as np
    from datetime import datetime, timedelta
    
    dates = [datetime.now() - timedelta(days=i) for i in range(100, 0, -1)]
    np.random.seed(42)  # For reproducible tests
    
    data = {
        'Date': dates,
        'Open': np.random.uniform(30000, 50000, 100),
        'High': np.random.uniform(30000, 50000, 100),
        'Low': np.random.uniform(30000, 50000, 100),
        'Close': np.random.uniform(30000, 50000, 100),
        'Volume': np.random.uniform(1e9, 5e9, 100)
    }
    
    df = pd.DataFrame(data)
    df = df.sort_values('Date').reset_index(drop=True)
    
    df['High'] = df[['Open', 'High', 'Close']].max(axis=1)
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1)
    
    return df


@pytest.fixture
def mock_model():
    original_env = os.environ.copy()
    
    os.environ["ENVIRONMENT"] = "test"
    os.environ["DATABASE_URL"] = "sqlite:///:memory:"
    os.environ["LOG_LEVEL"] = "INFO"
    
    yield
    
    os.environ.clear()
    os.environ.update(original_env)
