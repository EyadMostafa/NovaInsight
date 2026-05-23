"""Unit tests for Visualizer."""
from __future__ import annotations

import pandas as pd
import pytest

from sleuth.modules.statistical_analyzer import StatisticalAnalyzer
from sleuth.modules.target_detector import TargetDetector
from sleuth.modules.visualizer import Visualizer
from sleuth.schemas.analysis_report import (
    CorrelationReport,
    DimensionalityAnalysis,
    StatisticalAnalysis,
    OutlierReport,
)


def _run_target(config, df, report, user_target=None):
    if user_target:
        report.metadata = report.metadata.model_copy(update={"user_target": user_target})
    TargetDetector(config).run(df, report)
    return report


def _run_stats(config, df, report):
    StatisticalAnalyzer(config).run(df, report)
    return report


def _add_dim_analysis(report):
    report.dimensionality_analysis = DimensionalityAnalysis(
        pca_explained_variance_ratios=[0.5, 0.3],
        pca_embeddings_path=None,
        tsne_embeddings_path=None,
        umap_embeddings_path=None,
    )
    return report


class TestVisualizerRun:
    def test_run_populates_visualizations(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        report = _run_stats(test_config, titanic_small_df, report)
        report = _add_dim_analysis(report)
        result = Visualizer(test_config).run(titanic_small_df, report)
        assert result.visualizations is not None

    def test_univariate_plots_produced(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        report = _run_stats(test_config, titanic_small_df, report)
        report = _add_dim_analysis(report)
        result = Visualizer(test_config).run(titanic_small_df, report)
        assert len(result.visualizations.univariate_plots) > 0

    def test_plots_dir_created(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        report = _run_stats(test_config, titanic_small_df, report)
        report = _add_dim_analysis(report)
        Visualizer(test_config).run(titanic_small_df, report)
        plots_dir = report.metadata.output_dir / "plots"
        assert plots_dir.exists()

    def test_plot_files_are_png(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        report = _run_stats(test_config, titanic_small_df, report)
        report = _add_dim_analysis(report)
        result = Visualizer(test_config).run(titanic_small_df, report)
        for path in result.visualizations.univariate_plots.values():
            assert str(path).endswith(".png"), f"Expected .png, got {path}"

    def test_unsupervised_task_still_produces_univariate(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.metadata = report.metadata.model_copy(update={"task": "unsupervised"})
        report.target_analysis = None
        # minimal stats
        report.statistical_analysis = StatisticalAnalysis(
            outlier_report={},
            multicollinearity_report={},
            correlation_report=CorrelationReport(),
            class_imbalance_report=None,
            findings=[],
        )
        report = _add_dim_analysis(report)
        result = Visualizer(test_config).run(titanic_small_df, report)
        assert result.visualizations is not None
        assert len(result.visualizations.univariate_plots) > 0
