"""Unit tests for filesystem validators."""

from __future__ import annotations

from pathlib import Path

import pytest

from sleuth.exceptions import DataLoadError
from sleuth.utils.validators import validate_directory, validate_file_path


class TestValidateFilePath:
    def test_valid_file_returns_resolved_path(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("a,b\n1,2\n")
        result = validate_file_path(f)
        assert result == f.resolve()

    def test_nonexistent_raises_data_load_error(self, tmp_path):
        with pytest.raises(DataLoadError, match="not found"):
            validate_file_path(tmp_path / "missing.csv")

    def test_directory_raises_data_load_error(self, tmp_path):
        with pytest.raises(DataLoadError):
            validate_file_path(tmp_path)

    def test_accepts_string_path(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_bytes(b"data")
        result = validate_file_path(str(f))
        assert isinstance(result, Path)


class TestValidateDirectory:
    def test_creates_missing_directory(self, tmp_path):
        new_dir = tmp_path / "new" / "nested"
        is_valid, msg, resolved = validate_directory(new_dir)
        assert is_valid
        assert new_dir.exists()

    def test_existing_directory_is_valid(self, tmp_path):
        is_valid, msg, resolved = validate_directory(tmp_path)
        assert is_valid
        assert msg == ""

    def test_file_at_path_returns_invalid(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("content")
        is_valid, msg, _ = validate_directory(f)
        assert not is_valid

    def test_resolved_path_is_absolute(self, tmp_path):
        _, _, resolved = validate_directory(tmp_path / "sub")
        assert Path(resolved).is_absolute()
