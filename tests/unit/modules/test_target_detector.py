"""Unit tests for TargetDetector."""

from __future__ import annotations

import pandas as pd
import pytest

from sleuth.exceptions import TargetDetectorError
from sleuth.modules.target_detector import TargetDetector
from sleuth.schemas.analysis_report import ColumnDetails

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _detector(test_config):
    return TargetDetector(test_config)


# ---------------------------------------------------------------------------
# User-specified target: validation
# ---------------------------------------------------------------------------


class TestUserSpecifiedTarget:
    def test_valid_target_sets_identified_target(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.metadata = report.metadata.model_copy(update={"user_target": "Survived"})
        result = _detector(test_config).run(titanic_small_df, report)
        assert result.target_analysis is not None
        assert result.target_analysis.identified_target == "Survived"
        assert result.target_analysis.detection_method == "user_specified"

    def test_missing_column_raises_target_detector_error(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.metadata = report.metadata.model_copy(update={"user_target": "NonExistentColumn"})
        with pytest.raises(TargetDetectorError, match="does not exist"):
            _detector(test_config).run(titanic_small_df, report)

    def test_text_type_target_raises(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        # 'Name' should be inferred as text/id
        report.metadata = report.metadata.model_copy(update={"user_target": "Name"})
        with pytest.raises(TargetDetectorError):
            _detector(test_config).run(titanic_small_df, report)

    def test_confidence_score_is_1_for_user_specified(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        report.metadata = report.metadata.model_copy(update={"user_target": "Survived"})
        result = _detector(test_config).run(titanic_small_df, report)
        assert result.target_analysis.candidate_targets[0].confidence_score == 1.0


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------


class TestAutoTargetDetection:
    def test_finds_candidate_targets(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        result = _detector(test_config).run(titanic_small_df, report)
        assert result.target_analysis is not None
        assert len(result.target_analysis.candidate_targets) > 0

    def test_identified_target_is_not_id_column(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        result = _detector(test_config).run(titanic_small_df, report)
        assert result.target_analysis.identified_target != "PassengerId"

    def test_detection_method_is_auto(
        self, test_config, bare_report, titanic_small_df, run_profiler
    ):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        result = _detector(test_config).run(titanic_small_df, report)
        assert result.target_analysis.detection_method == "auto"

    def test_classification_task_inferred_for_binary_column(
        self, test_config, bare_report, run_profiler
    ):
        df = pd.DataFrame(
            {
                "feature": list(range(60)) * 2,
                "target": [0, 1] * 60,
            }
        )
        report = run_profiler(test_config, df, bare_report)
        result = _detector(test_config).run(df, report)
        assert result.target_analysis.identified_target == "target"
        assert result.target_analysis.candidate_targets[0].ml_task == "classification"

    def test_regression_task_for_continuous_column(self, test_config, bare_report, run_profiler):
        # Use bounded integer ranges so unique_pct < 0.99 (avoids 'id' inference)
        # and unique_count > target_detection.max_categorical_cardinality=50 (→ regression)
        df = pd.DataFrame(
            {
                "feature_a": list(range(60)) * 2,  # 60 unique / 120 rows
                "feature_b": list(range(60)) * 2,  # same
                "price": list(range(100, 160)) * 2,  # 60 unique / 120 rows
            }
        )
        report = run_profiler(test_config, df, bare_report)
        result = _detector(test_config).run(df, report)
        found_regression = any(
            c.ml_task == "regression"
            for c in result.target_analysis.candidate_targets
            if c.column_name == "price"
        )
        assert found_regression

    def test_no_valid_candidates_raises(self, test_config, bare_report, run_profiler):
        # DataFrame with only id-type columns
        df = pd.DataFrame({"id": range(200), "name": [f"name_{i}" for i in range(200)]})
        report = run_profiler(test_config, df, bare_report)
        with pytest.raises(TargetDetectorError):
            _detector(test_config).run(df, report)


# ---------------------------------------------------------------------------
# Task detection helper
# ---------------------------------------------------------------------------


class TestTaskDetection:
    def test_categorical_col_gives_classification(self, test_config):
        d = _detector(test_config)
        col = ColumnDetails(
            column_name="cat",
            inferred_type="categorical",
            dtype="object",
            missing_values_count=0,
            missing_values_pct=0.0,
            unique_values_count=3,
            unique_values_pct=0.01,
            stats={},
            memory_usage_mb=0.01,
        )
        assert d._task_detection(col) == "classification"

    def test_boolean_col_gives_classification(self, test_config):
        d = _detector(test_config)
        col = ColumnDetails(
            column_name="bool_col",
            inferred_type="boolean",
            dtype="int64",
            missing_values_count=0,
            missing_values_pct=0.0,
            unique_values_count=2,
            unique_values_pct=0.01,
            stats={},
            memory_usage_mb=0.01,
        )
        assert d._task_detection(col) == "classification"

    def test_continuous_numerical_gives_regression(self, test_config):
        d = _detector(test_config)
        col = ColumnDetails(
            column_name="price",
            inferred_type="numerical",
            dtype="float64",
            missing_values_count=0,
            missing_values_pct=0.0,
            unique_values_count=200,
            unique_values_pct=0.5,
            stats={},
            memory_usage_mb=0.01,
        )
        assert d._task_detection(col) == "regression"
