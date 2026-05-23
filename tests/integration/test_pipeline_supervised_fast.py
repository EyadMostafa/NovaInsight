"""Integration: full supervised pipeline (no LLM) on student_performance_small.csv."""

from __future__ import annotations

import pytest

from tests.integration.conftest import MODULES_NO_LLM, build_pipeline


@pytest.mark.slow
@pytest.mark.integration
def test_report_profile_populated(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    report = pipeline.report
    assert report.profile is not None
    assert len(report.profile.column_details) == 32  # student_performance has 32 columns


@pytest.mark.slow
@pytest.mark.integration
def test_target_analysis_populated(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    assert pipeline.report.target_analysis is not None
    assert pipeline.report.target_analysis.identified_target is not None


@pytest.mark.slow
@pytest.mark.integration
def test_statistical_analysis_populated(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    sa = pipeline.report.statistical_analysis
    assert sa is not None
    assert sa.outlier_report is not None
    assert sa.correlation_report is not None


@pytest.mark.slow
@pytest.mark.integration
def test_dimensionality_analysis_populated(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    dim = pipeline.report.dimensionality_analysis
    assert dim is not None
    assert dim.pca_explained_variance_ratios  # 10 numerical cols → PCA runs


@pytest.mark.slow
@pytest.mark.integration
def test_visualizations_populated(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    viz = pipeline.report.visualizations
    assert viz is not None
    assert len(viz.univariate_plots) > 0


@pytest.mark.slow
@pytest.mark.integration
def test_html_report_created(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    output_dir = pipeline.report.metadata.output_dir
    html_files = list(output_dir.glob("*.html"))
    assert len(html_files) == 1


@pytest.mark.slow
@pytest.mark.integration
def test_json_report_created(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    output_dir = pipeline.report.metadata.output_dir
    json_file = output_dir / "report.json"
    assert json_file.exists()


@pytest.mark.slow
@pytest.mark.integration
def test_timing_populated_for_all_modules(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    timing = pipeline.report.timing
    assert timing is not None
    assert timing.total_seconds > 0
    for module in MODULES_NO_LLM:
        assert module in timing.modules


@pytest.mark.slow
@pytest.mark.integration
def test_no_error_level_findings(tmp_path):
    pipeline = build_pipeline(tmp_path)
    pipeline.run()
    error_findings = [f for f in pipeline.report.findings if f.level == "ERROR"]
    assert len(error_findings) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_user_specified_target_respected(tmp_path):
    pipeline = build_pipeline(tmp_path, user_target="final_exam_score")
    pipeline.run()
    assert pipeline.report.target_analysis.identified_target == "final_exam_score"
    assert pipeline.report.target_analysis.detection_method == "user_specified"
