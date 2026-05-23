"""Integration: module failure skips dependents, pipeline still completes."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from sleuth.exceptions import StatisticalAnalyzerError
from tests.integration.conftest import build_pipeline, MODULES_NO_LLM


@pytest.mark.slow
@pytest.mark.integration
def test_stats_failure_skips_downstream(tmp_path):
    """When StatisticalAnalyzer raises ModuleError, VIZ is skipped."""
    pipeline = build_pipeline(tmp_path)

    with patch(
        "sleuth.modules.statistical_analyzer.StatisticalAnalyzer.run",
        side_effect=StatisticalAnalyzerError("injected failure"),
    ):
        pipeline.run()

    report = pipeline.report
    # STATS failed → no statistical_analysis
    assert report.statistical_analysis is None
    # VIZ depends on STATS → also skipped
    assert report.visualizations is None


@pytest.mark.slow
@pytest.mark.integration
def test_stats_failure_produces_error_finding(tmp_path):
    pipeline = build_pipeline(tmp_path)

    with patch(
        "sleuth.modules.statistical_analyzer.StatisticalAnalyzer.run",
        side_effect=StatisticalAnalyzerError("injected failure"),
    ):
        pipeline.run()

    error_findings = [f for f in pipeline.report.findings if f.level == "ERROR"]
    assert len(error_findings) >= 1
    assert any("stats" in f.message.lower() for f in error_findings)


@pytest.mark.slow
@pytest.mark.integration
def test_html_still_produced_after_module_failure(tmp_path):
    """Pipeline always generates an HTML report even if modules fail."""
    pipeline = build_pipeline(tmp_path)

    with patch(
        "sleuth.modules.statistical_analyzer.StatisticalAnalyzer.run",
        side_effect=StatisticalAnalyzerError("injected failure"),
    ):
        pipeline.run()

    output_dir = pipeline.report.metadata.output_dir
    html_files = list(output_dir.glob("*.html"))
    assert len(html_files) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_profiler_failure_skips_all_dependents(tmp_path):
    from sleuth.exceptions import ProfilerError

    pipeline = build_pipeline(tmp_path)

    with patch(
        "sleuth.modules.data_profiler.DataProfiler.run",
        side_effect=ProfilerError("profiler broken"),
    ):
        pipeline.run()

    report = pipeline.report
    # Everything depends on profiler → all should be None
    assert report.statistical_analysis is None
    assert report.dimensionality_analysis is None
    assert report.visualizations is None
