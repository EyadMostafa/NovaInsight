"""
cache_manager.py

This module contains the CacheManager class, which is responsible for all
persistent storage and caching operations for the Sleuth agent.
"""
import hashlib
import json
import shutil
from pathlib import Path

from pydantic import ValidationError

from sleuth.config.config import CacheSettings
from sleuth.exceptions import CacheError
from sleuth.schemas.analysis_report import AnalysisReport
from sleuth.utils.logger import get_logger
from sleuth.utils.validators import validate_directory

logger = get_logger("sleuth.utils.cache_manager")

class CacheManager:
    """
    Manages the creation, retrieval, and deletion of cached analysis reports.
    Implements self-healing for corrupt or outdated cache entries.
    """
    def __init__(self, config: CacheSettings):
        """Initializes the CacheManager with the cache configuration."""
        self.enabled = config.enabled
        self.base_cache_dir: Path | None = None

        if self.enabled:
            is_valid, message, resolved_path = validate_directory(config.directory_path)
            if is_valid:
                self.base_cache_dir = resolved_path
                logger.info(f"Cache enabled. Workspace location: {self.base_cache_dir}")
            else:
                logger.error(f"Failed to initialize cache directory: {message}. Caching will be disabled.")
                self.enabled = False
        else:
            logger.info("Caching is disabled in the configuration.")

    def hash_file(self, file_path: Path) -> str:
        """Computes the SHA256 hash of a file's content."""
        hasher = hashlib.sha256()
        chunk_size = 65536  # 64KB
        try:
            with open(file_path, 'rb') as f:
                while chunk := f.read(chunk_size):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except (OSError, PermissionError) as e:
            raise CacheError(f"Could not read file for hashing: {file_path}. Reason: {e}") from e

    def load_report(self, file_hash: str) -> AnalysisReport | None:
        """
        Loads an AnalysisReport from the cache. Returns None and clears the
        workspace if the cache is corrupt or outdated.
        """
        if not self.enabled:
            return None

        report_path = self.base_cache_dir / file_hash / "report.json"

        if not report_path.is_file():
            logger.debug(f"No cache found for hash: {file_hash}")
            return None

        try:
            logger.info(f"Found cached report for hash {file_hash}. Attempting to load.")
            json_data = report_path.read_text(encoding="utf-8")

            return AnalysisReport.model_validate_json(json_data)

        except (ValidationError, TypeError, json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load or validate cached report from {report_path}. Reason: {e}")
            logger.warning("Cache may be corrupt or from an older version. Clearing workspace and rerunning.")
            self.clear_workspace(file_hash)
            return None

    def save_report(self, report: AnalysisReport):
        """Serializes and saves the current AnalysisReport to the cache."""
        if not self.enabled:
            return

        file_hash = report.metadata.file_hash
        workspace_path = self._get_workspace_path(file_hash)
        report_path = workspace_path / "report.json"

        try:
            workspace_path.mkdir(parents=True, exist_ok=True)
            report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
            logger.info(f"Successfully saved analysis state to cache: {workspace_path}")
        except (OSError, PermissionError) as e:
            logger.error(f"Failed to save report to cache at {workspace_path}. Caching may not work correctly. Reason: {e}")

    def clear_workspace(self, file_hash: str):
        """Deletes the entire workspace directory for a given file hash."""
        if not self.enabled or not self.base_cache_dir:
            return

        workspace_path = self._get_workspace_path(file_hash)
        if workspace_path.exists():
            try:
                shutil.rmtree(workspace_path)
                logger.info(f"Successfully cleared workspace for hash: {file_hash}")
            except Exception as e:
                logger.error(f"Failed to clear workspace at {workspace_path}. Reason: {e}")

    def _get_workspace_path(self, file_hash: str) -> Path:
        """Constructs the full path to a specific analysis workspace directory."""
        return self.base_cache_dir / file_hash

