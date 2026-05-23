"""Unit tests for ReportGenerator."""

from __future__ import annotations

from sleuth.modules.report_generator import ReportGenerator
from sleuth.schemas.analysis_report import VisualizationOutput


class TestReportGeneratorRun:
    def test_run_creates_html_file(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)
        html_files = list(report.metadata.output_dir.glob("*.html"))
        assert len(html_files) == 1

    def test_run_sets_html_report_path(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)
        assert report.html_report_path is not None
        from pathlib import Path

        assert Path(report.html_report_path).exists()

    def test_html_contains_report_title(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)
        html_file = next(report.metadata.output_dir.glob("*.html"))
        content = html_file.read_text(encoding="utf-8")
        assert report.metadata.report_title in content

    def test_html_contains_table(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)
        html_file = next(report.metadata.output_dir.glob("*.html"))
        content = html_file.read_text(encoding="utf-8")
        assert "<table" in content.lower()

    def test_empty_plots_renders_without_crash(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.visualizations = VisualizationOutput(
            univariate_plots={},
            bivariate_plots={},
            correlation_heatmap_plots={},
            dim_reduction_scatter_plots={},
            findings=[],
        )
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)  # must not raise
        html_files = list(report.metadata.output_dir.glob("*.html"))
        assert len(html_files) == 1

    def test_is_completed_false_before_run(self, bare_report):
        """is_completed predicate must return False on a fresh report."""
        assert bare_report.html_report_path is None

    def test_is_completed_true_after_run(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        """is_completed predicate must return True after run() sets html_report_path."""
        report = run_profiler(test_config, titanic_small_df, bare_report)
        rg = ReportGenerator(test_config)
        rg.run(titanic_small_df, report)
        assert report.html_report_path is not None
