"""Unit tests for StatisticalAnalyzer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleuth.modules.statistical_analyzer import StatisticalAnalyzer
from sleuth.modules.target_detector import TargetDetector
from sleuth.schemas.analysis_report import ColumnDetails, Finding


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_target(config, df, report, user_target=None):
    if user_target:
        report.metadata = report.metadata.model_copy(update={"user_target": user_target})
    TargetDetector(config).run(df, report)
    return report


def _analyzer(test_config):
    return StatisticalAnalyzer(test_config)


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

class TestDetectOutliers:
    def test_z_score_finds_extreme_values(self, test_config):
        a = _analyzer(test_config)
        # 100 normal values + one extreme: z-score of 1000 ≈ 9.9
        col = pd.Series([0.0] * 100 + [1000.0])
        scores = a._compute_z_scores(col)
        assert scores is not None
        outlier_mask = np.abs(scores) > 3.0
        assert outlier_mask.any()

    def test_modified_z_score_finds_extreme_values(self, test_config):
        a = _analyzer(test_config)
        col = pd.Series([1.0, 2.0, 3.0, 2.0, 1.5, 1000.0])
        scores = a._compute_modified_z_scores(col)
        assert scores is not None
        outlier_mask = np.abs(scores) > 3.5
        assert outlier_mask.any()

    def test_zero_mad_returns_none(self, test_config):
        a = _analyzer(test_config)
        # Constant column → MAD = 0
        col = pd.Series([5.0, 5.0, 5.0, 5.0])
        result = a._compute_modified_z_scores(col)
        assert result is None

    def test_zero_std_returns_none(self, test_config):
        a = _analyzer(test_config)
        col = pd.Series([7.0, 7.0, 7.0])
        result = a._compute_z_scores(col)
        assert result is None

    def test_outlier_report_has_correct_keys(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report)
        report.metadata = report.metadata.model_copy(update={"task": "unsupervised"})
        report.target_analysis = None
        result = _analyzer(test_config).run(titanic_small_df, report)
        assert result.statistical_analysis is not None
        # Fare and Age should have outlier reports
        assert "Fare" in result.statistical_analysis.outlier_report


# ---------------------------------------------------------------------------
# Correlation helpers
# ---------------------------------------------------------------------------

class TestCorrelationHelpers:
    def test_dict_to_corr_matrix_symmetric(self, test_config):
        a = _analyzer(test_config)
        corr_dict = {("a", "b"): 0.7, ("a", "c"): 0.3}
        matrix = a._dict_to_corr_matrix(corr_dict)
        assert matrix.loc["a", "b"] == matrix.loc["b", "a"]
        assert matrix.loc["a", "a"] == 1.0

    def test_dict_to_corr_matrix_empty_returns_empty_df(self, test_config):
        a = _analyzer(test_config)
        result = a._dict_to_corr_matrix({})
        assert result.empty

    def test_cramers_v_perfect_association(self, test_config):
        a = _analyzer(test_config)
        # 3 categories avoid Yates' 2x2 continuity correction; V must be 1.0
        s1 = pd.Series(["a", "b", "c"] * 20)
        s2 = s1.copy()  # identical → perfect association
        v, pval = a._compute_cramers_v(s1, s2)
        assert v == pytest.approx(1.0, abs=0.01)

    def test_cramers_v_no_association(self, test_config):
        a = _analyzer(test_config)
        rng = np.random.default_rng(0)
        s1 = pd.Series(rng.choice(["a", "b"], size=200))
        s2 = pd.Series(rng.choice(["x", "y"], size=200))
        v, _ = a._compute_cramers_v(s1, s2)
        assert v < 0.3

    def test_correlation_ratio_perfect(self, test_config):
        a = _analyzer(test_config)
        cat = pd.Series(["A"] * 50 + ["B"] * 50)
        num = pd.Series([1.0] * 50 + [10.0] * 50)
        eta, _ = a._compute_correlation_ratio(cat, num)
        assert eta > 0.9

    def test_correlation_ratio_empty_returns_nan(self, test_config):
        a = _analyzer(test_config)
        cat = pd.Series([], dtype=object)
        num = pd.Series([], dtype=float)
        eta, pval = a._compute_correlation_ratio(cat, num)
        assert np.isnan(eta)


# ---------------------------------------------------------------------------
# Class imbalance
# ---------------------------------------------------------------------------

class TestClassImbalance:
    def test_imbalanced_target_triggers_finding(self, test_config, bare_report):
        a = _analyzer(test_config)
        target = pd.Series(["majority"] * 90 + ["minority"] * 10)
        target.name = "y"
        report = a._analyze_class_imbalance(target)
        assert report is not None
        assert "majority" in report.class_counts
        assert any("imbalance" in f.message.lower() for f in a.findings)

    def test_balanced_target_no_warning(self, test_config):
        a = _analyzer(test_config)
        target = pd.Series([0, 1] * 50)
        target.name = "y"
        a._analyze_class_imbalance(target)
        imbalance_findings = [f for f in a.findings if "imbalance" in f.message.lower()]
        assert len(imbalance_findings) == 0


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

class TestStatisticalAnalyzerRun:
    def test_run_populates_statistical_analysis(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        result = _analyzer(test_config).run(titanic_small_df, report)
        assert result.statistical_analysis is not None
        assert result.statistical_analysis.outlier_report is not None
        assert result.statistical_analysis.correlation_report is not None

    def test_correlation_csvs_written(self, test_config, bare_report, titanic_small_df, tmp_path, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report = _run_target(test_config, titanic_small_df, report, user_target="Survived")
        _analyzer(test_config).run(titanic_small_df, report)
        corr_dir = report.metadata.output_dir / "correlations"
        assert corr_dir.exists()
        csv_files = list(corr_dir.glob("*.csv"))
        assert len(csv_files) >= 1

    def test_unsupervised_skips_leakage_check(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.metadata = report.metadata.model_copy(update={"task": "unsupervised"})
        report.target_analysis = None
        result = _analyzer(test_config).run(titanic_small_df, report)
        assert result.statistical_analysis is not None
        # No leakage findings for unsupervised
        leakage = [f for f in result.statistical_analysis.findings if "leakage" in f.message.lower()]
        assert len(leakage) == 0
