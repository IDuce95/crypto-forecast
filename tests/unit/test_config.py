import os
import sys
import unittest
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from config_manager import ConfigManager
    from config import config
except ImportError:
    class MockConfig:
        def __init__(self):
            self.DATABASE_URL = "sqlite:///test.db"
            self.MODEL_STORAGE_PATH = "./models"
            self.LOG_LEVEL = "INFO"

    config = MockConfig()
    ConfigManager = None


class TestConfigManager(unittest.TestCase):
        from app.config_manager import ConfigManager
        self.config_manager = ConfigManager()

    def test_config_loading_success(self):
        config = self.config_manager.get_config()
        assert hasattr(config, 'logging'), "logging section should exist"
        assert config.logging.level in ['DEBUG', 'INFO', 'WARNING', 'ERROR'], f"Invalid log level: {config.logging.level}"

    def test_config_default_values(self):

    def test_python_version(self):
        app_path = Path(__file__).parent.parent.parent / "app"
        assert app_path.exists(), "App directory should exist"
