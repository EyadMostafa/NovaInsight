"""
validators.py

Utility functions for validating file-system paths.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
from logging import getLogger

from novainsight.exceptions import DataLoadError

logger = getLogger(__name__)


def validate_file_path(file_path: str | Path) -> Path:
    """Validates a file path for existence, type, and read permissions."""
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise DataLoadError(f"Input file not found: {p}")
    if not p.is_file():
        raise DataLoadError(f"Path points to a directory, not a file: {p}")
    if not os.access(p, os.R_OK):
        raise DataLoadError(f"Read access denied for file: {p}")
    return p


def validate_directory(path_str: str | Path) -> Tuple[bool, str, Path | str]:
    """
    Validates a path as a writable directory, creating it if absent.
    Returns (is_valid, error_message, resolved_path).
    """
    try:
        p = Path(path_str).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory is valid and ready at: {p}")
        return True, "", p
    except FileExistsError:
        return False, f"A file already exists at the directory path: {path_str}", ""
    except PermissionError:
        return False, f"Write access denied for output directory: {path_str}", ""
    except Exception as e:
        return False, f"Unexpected OS error creating directory {path_str}: {e}", ""
