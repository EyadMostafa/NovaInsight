"""Unit tests for CacheManager."""
from __future__ import annotations

from pathlib import Path

import pytest

from sleuth.config.config import CacheSettings
from sleuth.exceptions import CacheError
from sleuth.utils.cache_manager import CacheManager


def _make_cache(tmp_path: Path, enabled: bool = True) -> CacheManager:
    return CacheManager(CacheSettings(enabled=enabled, directory_path=tmp_path / "cache"))


class TestHashFile:
    def test_stable_across_calls(self, tmp_path):
        f = tmp_path / "data.csv"
        f.write_text("col1,col2\n1,2\n3,4\n")
        cm = _make_cache(tmp_path)
        h1 = cm.hash_file(f)
        h2 = cm.hash_file(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.csv"
        f2 = tmp_path / "b.csv"
        f1.write_text("hello")
        f2.write_text("world")
        cm = _make_cache(tmp_path)
        assert cm.hash_file(f1) != cm.hash_file(f2)

    def test_missing_file_raises_cache_error(self, tmp_path):
        cm = _make_cache(tmp_path)
        with pytest.raises(CacheError):
            cm.hash_file(tmp_path / "nonexistent.csv")

    def test_hash_is_64_char_hex(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_bytes(b"data")
        cm = _make_cache(tmp_path)
        h = cm.hash_file(f)
        assert len(h) == 64
        int(h, 16)  # must be valid hex


class TestSaveLoad:
    def test_roundtrip_preserves_report(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path)
        cm.save_report(bare_report)
        loaded = cm.load_report(bare_report.metadata.file_hash)
        assert loaded is not None
        assert loaded.metadata.report_title == bare_report.metadata.report_title
        assert loaded.metadata.file_hash == bare_report.metadata.file_hash

    def test_load_returns_none_when_no_cache(self, tmp_path):
        cm = _make_cache(tmp_path)
        result = cm.load_report("nonexistentdeadbeef")
        assert result is None

    def test_corrupt_cache_returns_none_and_clears(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path)
        cm.save_report(bare_report)
        # Corrupt the file
        cache_path = cm.base_cache_dir / bare_report.metadata.file_hash / "report.json"
        cache_path.write_text("{invalid json{{{{")
        result = cm.load_report(bare_report.metadata.file_hash)
        assert result is None
        # Workspace should be cleaned up
        assert not cache_path.exists()

    def test_truncated_json_returns_none(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path)
        cm.save_report(bare_report)
        cache_path = cm.base_cache_dir / bare_report.metadata.file_hash / "report.json"
        content = cache_path.read_text()
        cache_path.write_text(content[:50])  # truncate
        result = cm.load_report(bare_report.metadata.file_hash)
        assert result is None


class TestClearWorkspace:
    def test_clear_removes_directory(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path)
        cm.save_report(bare_report)
        workspace = cm.base_cache_dir / bare_report.metadata.file_hash
        assert workspace.exists()
        cm.clear_workspace(bare_report.metadata.file_hash)
        assert not workspace.exists()

    def test_clear_nonexistent_is_noop(self, tmp_path):
        cm = _make_cache(tmp_path)
        # Should not raise
        cm.clear_workspace("neverexisted123")


class TestDisabledCache:
    def test_disabled_save_is_noop(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path, enabled=False)
        cm.save_report(bare_report)
        # Nothing should be written
        cache_dir = tmp_path / "cache"
        assert not cache_dir.exists() or not any(cache_dir.iterdir())

    def test_disabled_load_returns_none(self, tmp_path, bare_report):
        cm = _make_cache(tmp_path, enabled=False)
        result = cm.load_report(bare_report.metadata.file_hash)
        assert result is None

    def test_disabled_hash_still_works(self, tmp_path):
        cm = _make_cache(tmp_path, enabled=False)
        f = tmp_path / "data.csv"
        f.write_bytes(b"test content")
        # hash_file does not depend on cache being enabled
        h = cm.hash_file(f)
        assert len(h) == 64
