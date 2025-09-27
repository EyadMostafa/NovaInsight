"""
validators.py

A collection of standalone utility functions for validating file system paths.

This module provides reusable functions for checking input files and ensuring
output directories are valid and writable.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple
from logging import getLogger

logger = getLogger(__name__)

def validate_file_path(file_path: str) -> Path:
    """
    Validates a file path based on existence, type, permissions, and extension.
    """
    p = Path(file_path).expanduser().resolve()
    if not p.exists():
        raise FileNotFoundError(f"Input file not found at the specified path: {p}")

    if not p.is_file():
        raise ValueError(f"The provided path points to a directory, not a file: {p}")

    if not os.access(p, os.R_OK):
        raise PermissionError(f"Read access denied for the file: {p}")
    
    return p

def validate_directory(path_str: str) -> Tuple[bool, str, Path | str]:
    """
    Validates a path intended to be a directory, ensuring it can be written to.
    If the directory does not exist, it attempts to create it.
    """
    try:
        p = Path(path_str).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Directory is valid and ready at: {p}")
        return True, "", p
    except FileExistsError:
        return False, f"Directory '{path_str}' cannot be created because a file with the same name exists: {path_str}", ""
    except PermissionError:
        return False, f"Write access denied for the output directory: {path_str}", ""
    except Exception as e:
        return False, f"An unexpected OS error occurred while creating the directory: {path_str}: {e}", ""
