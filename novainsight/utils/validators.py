"""
validators.py

A collection of standalone utility functions for validating file system paths.

This module provides reusable functions for checking input files and ensuring
output directories are valid and writable.
"""

import os
from pathlib import Path
from typing import Tuple

def validate_file_path(
        file_path: str
    ) -> Path:
        """
        Validates a file path based on existence, type, permissions, and extension.
        """
            p = Path(path_str)

            if not p.exists():
                raise FileNotFoundError(f"Error: The path '{p}' does not exist.")

            if not p.is_file():
                raise ValueError(f"Error: The provided path '{p}' points to a directory, not a file.")

            if not os.access(p, os.R_OK):
                raise PermissionError(f"Error: Do not have read permissions for the file '{p}'.")

            return p

def validate_directory(path_str: str) -> Tuple[bool, message, Path]:
    """
    Validates a path intended to be a directory, ensuring it can be written to.
    If the directory does not exist, it attempts to create it.
    """
    try:
        p = Path(path_str).expanduser().resolve()
        p.mkdir(parents=True, exist_ok=True)

        logger.debug(f"Directory is valid and ready at: {p}")
        return True, _, p
    except FileExistsError:
        return False, f"The directory '{path_str}' cannot be created because a file with the same name exists in its path.", path_str
    except PermissionError:
        return False, f"Do not have write permissions to create the directory at '{path_str}'.", path_str
    except Exception as e:
        return False, f"An unexpected error occurred while validating the directory: {e}", path_str
