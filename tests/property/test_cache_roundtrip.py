"""Property-based tests for CacheManager serialization round-trip."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sleuth.config.config import CacheSettings
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    DatasetProfile,
    DatasetStats,
    RunMetadata,
)
from sleuth.utils.cache_manager import CacheManager


def _make_cache(tmp_path: Path) -> CacheManager:
    return CacheManager(CacheSettings(enabled=True, directory_path=tmp_path / "cache"))


def _build_report(
    title: str,
    file_hash: str,
    original_rows: int,
    output_dir: Path,
) -> AnalysisReport:
    metadata = RunMetadata(
        input_file=output_dir / "data.csv",
        output_dir=output_dir,
        file_hash=file_hash,
        report_title=title,
        run_timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc),
        analysis_mode="full",
        task="supervised",
    )
    stats = DatasetStats(
        original_row_count=max(1, original_rows),
        analyzed_row_count=max(1, original_rows),
        column_count=5,
        total_memory_usage_mb=1.0,
        duplicate_rows_count=0,
        column_pct=0.1,
        duplicates_pct=0.0,
    )
    profile = DatasetProfile(dataset_stats=stats, column_details=[], findings=[])
    return AnalysisReport(metadata=metadata, profile=profile, findings=[])


# ---------------------------------------------------------------------------
# Round-trip: save → load must preserve key fields
# ---------------------------------------------------------------------------


@given(
    title=st.text(
        min_size=1,
        max_size=40,
        alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd", "Zs")),
    ),
    original_rows=st.integers(min_value=1, max_value=10000),
)
@settings(
    max_examples=30,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=5000,
)
def test_roundtrip_preserves_title_and_hash(title, original_rows, tmp_path):
    file_hash = "a" * 64
    output_dir = tmp_path / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _build_report(title, file_hash, original_rows, output_dir)
    cm = _make_cache(tmp_path)
    cm.save_report(report)

    loaded = cm.load_report(file_hash)
    assert loaded is not None
    assert loaded.metadata.report_title == title
    assert loaded.metadata.file_hash == file_hash
    assert loaded.profile.dataset_stats.original_row_count == max(1, original_rows)


# ---------------------------------------------------------------------------
# Disabled cache: load always returns None regardless of content
# ---------------------------------------------------------------------------


@given(original_rows=st.integers(min_value=1, max_value=100))
@settings(
    max_examples=10,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=5000,
)
def test_disabled_cache_always_returns_none(original_rows, tmp_path):
    cm_enabled = _make_cache(tmp_path)
    output_dir = tmp_path / "out2"
    output_dir.mkdir(parents=True, exist_ok=True)

    report = _build_report("title", "b" * 64, original_rows, output_dir)
    cm_enabled.save_report(report)

    cm_disabled = CacheManager(CacheSettings(enabled=False, directory_path=tmp_path / "cache"))
    assert cm_disabled.load_report("b" * 64) is None
