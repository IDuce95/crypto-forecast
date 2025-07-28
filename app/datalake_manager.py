

import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import json
import pandas as pd
from enum import Enum

from app.config import config
from app.logger import logger

class DataStage(Enum):

    RAW = "raw"
    PROCESSED = "processed"
    TRANSFORMED = "transformed"
    AGGREGATED = "aggregated"
    ARCHIVED = "archived"

class DataFormat(Enum):

    CSV = "csv"
    JSON = "json"
    PARQUET = "parquet"
    LOG = "log"

class DatalakeManager:

    def __init__(self, datalake_root: Path = None):

        if datalake_root is None:
            datalake_root = Path("./datalake")

        self.datalake_root = datalake_root
        self.structure = {
            DataStage.RAW: self.datalake_root / "raw",
            DataStage.PROCESSED: self.datalake_root / "processed",
            DataStage.ARCHIVED: self.datalake_root / "archive"
        }

        self.raw_subdirs = {
            DataFormat.CSV: self.structure[DataStage.RAW] / "csv",
            DataFormat.JSON: self.structure[DataStage.RAW] / "json",
            DataFormat.LOG: self.structure[DataStage.RAW] / "logs"
        }

        self.processed_subdirs = {
            "transformed": self.structure[DataStage.PROCESSED] / "transformed",
            "aggregated": self.structure[DataStage.PROCESSED] / "aggregated"
        }

        logger.info(f"DatalakeManager initialized with root: {self.datalake_root}")

    def ensure_structure(self) -> None:

        for stage_dir in self.structure.values():
            stage_dir.mkdir(parents=True, exist_ok=True)

        for subdir in self.raw_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

        for subdir in self.processed_subdirs.values():
            subdir.mkdir(parents=True, exist_ok=True)

        logger.info("Datalake directory structure ensured")

    def get_path(self, stage: DataStage, data_format: DataFormat = None,
                 subdir: str = None) -> Path:

        if stage == DataStage.RAW and data_format:
            return self.raw_subdirs.get(data_format, self.structure[stage])
        elif stage == DataStage.PROCESSED and subdir:
            return self.processed_subdirs.get(subdir, self.structure[stage])
        else:
            return self.structure.get(stage, self.datalake_root)

    def copy_to_raw(self, source_path: Path, data_format: DataFormat,
                    preserve_structure: bool = True) -> Path:

        target_dir = self.get_path(DataStage.RAW, data_format)
        target_dir.mkdir(parents=True, exist_ok=True)

        if source_path.is_file():
            target_path = target_dir / source_path.name
            shutil.copy2(source_path, target_path)
            logger.info(f"Copied file {source_path} to {target_path}")
            return target_path
        elif source_path.is_dir():
            if preserve_structure:
                target_path = target_dir / source_path.name
                shutil.copytree(source_path, target_path, dirs_exist_ok=True)
            else:
                target_path = target_dir
                for file_path in source_path.glob("*"):
                    if file_path.is_file():
                        shutil.copy2(file_path, target_path / file_path.name)
            logger.info(f"Copied directory {source_path} to {target_path}")
            return target_path
        else:
            raise ValueError(f"Source path {source_path} does not exist")

    def move_to_archive(self, source_path: Path,
                       archive_subdir: str = None) -> Path:

        archive_dir = self.structure[DataStage.ARCHIVED]

        if archive_subdir:
            archive_dir = archive_dir / archive_subdir

        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if source_path.is_file():
            stem = source_path.stem
            suffix = source_path.suffix
            target_name = f"{stem}_{timestamp}{suffix}"
        else:
            target_name = f"{source_path.name}_{timestamp}"

        target_path = archive_dir / target_name
        shutil.move(str(source_path), str(target_path))

        logger.info(f"Archived {source_path} to {target_path}")
        return target_path

    def cleanup_old_data(self, stage: DataStage, days_old: int = 30,
                        dry_run: bool = True) -> List[Path]:

        stage_dir = self.structure.get(stage)
        if not stage_dir or not stage_dir.exists():
            return []

        cutoff_date = datetime.now() - timedelta(days=days_old)
        old_files = []

        for file_path in stage_dir.rglob("*"):
            if file_path.is_file():
                file_mtime = datetime.fromtimestamp(file_path.stat().st_mtime)
                if file_mtime < cutoff_date:
                    old_files.append(file_path)

        if not dry_run:
            for file_path in old_files:
                try:
                    file_path.unlink()
                    logger.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logger.warning(f"Failed to delete {file_path}: {e}")
        else:
            logger.info(f"Dry run: would delete {len(old_files)} old files from {stage.value}")

        return old_files

    def get_inventory(self) -> Dict[str, Any]:

        inventory = {
            "timestamp": datetime.now().isoformat(),
            "stages": {}
        }

        for stage, stage_dir in self.structure.items():
            if stage_dir.exists():
                stage_info = {
                    "path": str(stage_dir),
                    "total_files": 0,
                    "total_size_mb": 0,
                    "file_types": {},
                    "subdirectories": {}
                }

                for file_path in stage_dir.rglob("*"):
                    if file_path.is_file():
                        stage_info["total_files"] += 1
                        file_size = file_path.stat().st_size / (1024 * 1024)
                        stage_info["total_size_mb"] += file_size

                        ext = file_path.suffix.lower()
                        stage_info["file_types"][ext] = stage_info["file_types"].get(ext, 0) + 1

                        rel_path = file_path.relative_to(stage_dir)
                        if len(rel_path.parts) > 1:
                            subdir = rel_path.parts[0]
                            if subdir not in stage_info["subdirectories"]:
                                stage_info["subdirectories"][subdir] = {"files": 0, "size_mb": 0}
                            stage_info["subdirectories"][subdir]["files"] += 1
                            stage_info["subdirectories"][subdir]["size_mb"] += file_size

                stage_info["total_size_mb"] = round(stage_info["total_size_mb"], 2)
                inventory["stages"][stage.value] = stage_info

        return inventory

    def save_inventory(self, output_path: Path = None) -> Path:

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.datalake_root / f"inventory_{timestamp}.json"

        inventory = self.get_inventory()

        with open(output_path, 'w') as f:
            json.dump(inventory, f, indent=2)

        logger.info(f"Saved datalake inventory to {output_path}")
        return output_path

    def migrate_existing_data(self) -> Dict[str, Any]:

        results = {
            "migrated_files": 0,
            "errors": [],
            "details": {}
        }

        processed_dir = Path(config.data_settings.processed_data_path)
        if processed_dir.exists():
            csv_files = list(processed_dir.glob("*.csv"))
            target_dir = self.get_path(DataStage.PROCESSED, subdir="transformed")

            for csv_file in csv_files:
                try:
                    target_path = target_dir / csv_file.name
                    shutil.copy2(csv_file, target_path)
                    results["migrated_files"] += 1
                    logger.info(f"Migrated {csv_file} to datalake")
                except Exception as e:
                    error_msg = f"Failed to migrate {csv_file}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

        raw_dir = Path(config.data_settings.raw_data_path)
        if raw_dir.exists():
            csv_files = list(raw_dir.glob("*.csv"))
            target_dir = self.get_path(DataStage.RAW, DataFormat.CSV)

            for csv_file in csv_files:
                try:
                    target_path = target_dir / csv_file.name
                    shutil.copy2(csv_file, target_path)
                    results["migrated_files"] += 1
                    logger.info(f"Migrated {csv_file} to datalake")
                except Exception as e:
                    error_msg = f"Failed to migrate {csv_file}: {e}"
                    results["errors"].append(error_msg)
                    logger.error(error_msg)

        results["details"]["total_migrated"] = results["migrated_files"]
        results["details"]["total_errors"] = len(results["errors"])

        logger.info(f"Data migration completed: {results['migrated_files']} files migrated")
        return results

def setup_datalake() -> DatalakeManager:

    logger.info("Setting up datalake structure...")

    manager = DatalakeManager()

    manager.ensure_structure()

    migration_results = manager.migrate_existing_data()

    inventory_path = manager.save_inventory()

    logger.info(f"Datalake setup completed!")
    logger.info(f"Migrated {migration_results['migrated_files']} files")
    logger.info(f"Inventory saved to {inventory_path}")

    return manager

if __name__ == "__main__":
    """Setup datalake and show inventory."""

    datalake = setup_datalake()

    inventory = datalake.get_inventory()

    print("\n" + "="*50)
    print("DATALAKE INVENTORY")
    print("="*50)

    for stage_name, stage_info in inventory["stages"].items():
        print(f"\n📁 {stage_name.upper()}:")
        print(f"  Files: {stage_info['total_files']}")
        print(f"  Size: {stage_info['total_size_mb']} MB")

        if stage_info["subdirectories"]:
            print("  Subdirectories:")
            for subdir, subdir_info in stage_info["subdirectories"].items():
                print(f"    📂 {subdir}: {subdir_info['files']} files ({subdir_info['size_mb']:.1f} MB)")

    print(f"\n📊 Generated at: {inventory['timestamp']}")
