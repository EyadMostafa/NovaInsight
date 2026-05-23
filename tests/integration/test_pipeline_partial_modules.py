"""Integration: running a subset of modules stops at the right boundary."""

from __future__ import annotations

import pytest

from tests.integration.conftest import build_pipeline


@pytest.mark.slow
@pytest.mark.integration
def test_profiler_only(tmp_path):
    pipeline = build_pipeline(tmp_path, requested_modules=["profiler"])
    pipeline.run()
    report = pipeline.report
    assert report.profile is not None
    assert len(report.profile.column_details) > 0
    assert report.statistical_analysis is None
    assert report.dimensionality_analysis is None
    assert report.visualizations is None


@pytest.mark.slow
@pytest.mark.integration
def test_stats_includes_transitive_deps(tmp_path):
    """Requesting 'stats' should automatically pull in profiler and target."""
    pipeline = build_pipeline(tmp_path, requested_modules=["stats"])
    pipeline.run()
    report = pipeline.report
    # Profiler and target were run as deps
    assert report.profile is not None
    assert report.target_analysis is not None
    assert report.statistical_analysis is not None
    # VIZ was NOT requested
    assert report.visualizations is None


@pytest.mark.slow
@pytest.mark.integration
def test_unknown_module_name_is_ignored(tmp_path):
    """Unknown module names are silently ignored; valid ones still run."""
    pipeline = build_pipeline(tmp_path, requested_modules=["profiler", "does_not_exist"])
    pipeline.run()
    assert pipeline.report.profile is not None


@pytest.mark.slow
@pytest.mark.integration
def test_dim_reduction_with_profiler_dep(tmp_path):
    pipeline = build_pipeline(tmp_path, requested_modules=["dim_reduction"])
    pipeline.run()
    assert pipeline.report.profile is not None
    assert pipeline.report.dimensionality_analysis is not None
