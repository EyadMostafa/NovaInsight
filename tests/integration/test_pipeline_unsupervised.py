"""Integration: unsupervised pipeline skips target detection."""

from __future__ import annotations

import pytest

from tests.integration.conftest import MODULES_NO_LLM_UNSUPERVISED, build_pipeline


@pytest.mark.slow
@pytest.mark.integration
def test_target_analysis_is_none(tmp_path):
    pipeline = build_pipeline(
        tmp_path,
        task="unsupervised",
        requested_modules=MODULES_NO_LLM_UNSUPERVISED,
    )
    pipeline.run()
    assert pipeline.report.target_analysis is None


@pytest.mark.slow
@pytest.mark.integration
def test_stats_still_runs_unsupervised(tmp_path):
    pipeline = build_pipeline(
        tmp_path,
        task="unsupervised",
        requested_modules=MODULES_NO_LLM_UNSUPERVISED,
    )
    pipeline.run()
    assert pipeline.report.statistical_analysis is not None


@pytest.mark.slow
@pytest.mark.integration
def test_visualizations_produced_unsupervised(tmp_path):
    pipeline = build_pipeline(
        tmp_path,
        task="unsupervised",
        requested_modules=MODULES_NO_LLM_UNSUPERVISED,
    )
    pipeline.run()
    assert pipeline.report.visualizations is not None
    assert len(pipeline.report.visualizations.univariate_plots) > 0


@pytest.mark.slow
@pytest.mark.integration
def test_no_bivariate_target_plots_unsupervised(tmp_path):
    pipeline = build_pipeline(
        tmp_path,
        task="unsupervised",
        requested_modules=MODULES_NO_LLM_UNSUPERVISED,
    )
    pipeline.run()
    bivariate = pipeline.report.visualizations.bivariate_plots
    # Supervised target-vs-feature plots should not exist
    target_plots = [k for k in bivariate if "target" in k.lower()]
    assert len(target_plots) == 0


@pytest.mark.slow
@pytest.mark.integration
def test_class_imbalance_not_computed_unsupervised(tmp_path):
    pipeline = build_pipeline(
        tmp_path,
        task="unsupervised",
        requested_modules=MODULES_NO_LLM_UNSUPERVISED,
    )
    pipeline.run()
    sa = pipeline.report.statistical_analysis
    assert sa.class_imbalance_report is None
