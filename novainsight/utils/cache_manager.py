"""
cache_manager.py

This module contains the CacheManager class, which is responsible for all
persistent storage and caching operations for the NovaInsight agent. It
handles hashing datasets to detect changes and manages the analysis
workspaces on the file system.
"""
import shutil
from hashlib import sha256
from pathlib import Path
from typing import Optional
from logging import getLogger
from novainsight.schemas.analysis_report import AnalysisReport
from novainsight.utils.validators import (validate_file_path, validate_directory)

logger = logging.getLogger(__name__)

class CacheManager:
    """
    Manages the creation, retrieval, and deletion of cached analysis reports.
    """

    def __init__(self, config: 'CacheSettings'):
        """
        Initializes the CacheManager with the cache configuration.
        """
        self.enabled = config.enabled
        self.base_cache_dir = config.directory_path
        if self.enabled:
            is_valid, message, self.base_cache_dir = validate_directory(self.base_cache_dir)
            if not is_valid:
                logger.error(f"Error: Failed to resolve caching directory, will proceed without caching: {message}")
                return
        else:
            logger.info("Caching is disabled in the configuration.")

    def hash_file(self, file_path: Path) -> str:
        """
        Computes the SHA256 hash of a file's content to uniquely identify it.
        """
        hasher = sha256()
        chunk_size = 8192
        try:
            with open(file_path, 'rb') as f:
                while True:
                    chunk = f.read(chunk_size)
                    if not chunk: break
                    hasher.update(chunk)

                return hasher.hexdigest()
        except Exception:
            logger.error(f"An error occurred while hashing the file {file_path}: {e}")


    def load_report(self, file_hash: str) -> Optional[AnalysisReport]:
        """
        Loads and deserializes an AnalysisReport from the cache if it exists.
        """
        pass

    def save_report(self, file_hash: str, report: AnalysisReport):
        """
        Serializes and saves the current AnalysisReport to the appropriate
        workspace in the cache.
        """

        pass

    def clear_workspace(self, file_hash: str):
        """
        Deletes the entire workspace directory for a given file hash.
        """
        pass

    def _get_workspace_path(self, file_hash: str) -> Path:
        """
        Constructs the full path to a specific analysis workspace directory.
        """
        return self.base_cache_dir / file_hash