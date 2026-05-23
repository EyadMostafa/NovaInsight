"""Unit tests for DataProfiler."""
from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from sleuth.exceptions import ProfilerError
from sleuth.modules.data_profiler import DataProfiler

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _profiler(test_config):
    return DataProfiler(test_config)


def _run(profiler, df, bare_report):
    """Run profiler and return the enriched report."""
    bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": len(df)})
    return profiler.run(df, bare_report)


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------

class TestInferLogicalType:
    def test_bool_dtype(self, test_config):
        p = _profiler(test_config)
        s = pd.Series([True, False, True, False] * 20)
        assert p._infer_logical_type(s) == "boolean"

    def test_numeric_two_unique_is_boolean(self, test_config):
        p = _profiler(test_config)
        # 2 unique, not >99% unique → boolean
        s = pd.Series([0, 1] * 100)
        assert p._infer_logical_type(s) == "boolean"

    def test_id_type_via_high_uniqueness(self, test_config):
        p = _profiler(test_config)
        # All unique integers → uniqueness = 1.0 > 0.99
        s = pd.Series(range(200))
        assert p._infer_logical_type(s) == "id"

    def test_categorical_from_few_unique_numeric(self, test_config):
        p = _profiler(test_config)
        # 5 unique, 300 total, pct=0.017 ≤ 0.99, count=5 ≤ 50 → categorical
        s = pd.Series([i % 5 for i in range(300)])
        assert p._infer_logical_type(s) == "categorical"

    def test_numerical_from_many_unique_numeric(self, test_config):
        p = _profiler(test_config)
        # 60 unique, 120 total, pct=0.5, count=60 > 50 → numerical
        s = pd.Series(list(range(60)) * 2)
        assert p._infer_logical_type(s) == "numerical"

    def test_datetime_native_dtype(self, test_config):
        p = _profiler(test_config)
        s = pd.to_datetime(pd.Series(["2021-01-01", "2021-06-15", "2022-12-31"] * 20))
        assert p._infer_logical_type(s) == "datetime"

    def test_datetime_convertible_string(self, test_config):
        p = _profiler(test_config)
        # Strings that parse as dates — sample-based check
        s = pd.Series(["2021-01-01", "2021-01-02", "2021-01-03"] * 100)
        assert p._infer_logical_type(s) == "datetime"

    def test_categorical_from_few_unique_strings(self, test_config):
        p = _profiler(test_config)
        s = pd.Series(["apple", "banana", "cherry"] * 100)
        assert p._infer_logical_type(s) == "categorical"

    def test_text_from_many_unique_strings(self, test_config):
        p = _profiler(test_config)
        # 60 unique, each repeated 3× → pct=0.33, count=60 > 50 → text
        words = [f"unique_word_{i}" for i in range(60)] * 3
        s = pd.Series(words)
        assert p._infer_logical_type(s) == "text"

    def test_all_null_column(self, test_config):
        p = _profiler(test_config)
        s = pd.Series([None] * 50, dtype=object)
        # No assertion on value — just must not raise
        result = p._infer_logical_type(s)
        assert isinstance(result, str)

    def test_single_row_column(self, test_config):
        p = _profiler(test_config)
        s = pd.Series([42])
        result = p._infer_logical_type(s)
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# Column stats shape
# ---------------------------------------------------------------------------

class TestGetColumnStats:
    def test_numerical_returns_describe_keys(self, test_config):
        p = _profiler(test_config)
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        stats = p._get_column_stats(s, "numerical")
        assert "mean" in stats
        assert "std" in stats
        assert all(isinstance(v, float) for v in stats.values())

    def test_categorical_returns_top_and_freq(self, test_config):
        p = _profiler(test_config)
        s = pd.Series(["a", "a", "b", "c"])
        stats = p._get_column_stats(s, "categorical")
        assert "top" in stats
        assert "frequency" in stats
        assert "value_counts" in stats
        assert stats["top"] == "a"

    def test_categorical_empty_series(self, test_config):
        p = _profiler(test_config)
        s = pd.Series([], dtype=object)
        stats = p._get_column_stats(s, "categorical")
        assert stats["top"] == "N/A"
        assert stats["frequency"] == 0

    def test_datetime_returns_first_last(self, test_config):
        p = _profiler(test_config)
        s = pd.to_datetime(pd.Series(["2021-01-01", "2022-12-31"]))
        stats = p._get_column_stats(s, "datetime")
        assert "first" in stats
        assert "last" in stats

    def test_unknown_type_returns_empty_dict(self, test_config):
        p = _profiler(test_config)
        s = pd.Series(["x", "y"])
        stats = p._get_column_stats(s, "id")
        assert stats == {}


# ---------------------------------------------------------------------------
# Threshold-driven Findings
# ---------------------------------------------------------------------------

class TestProfilerFindings:
    def test_duplicate_rows_finding(self, test_config, bare_report):
        # 100% duplicates → triggers duplicate_threshold (0.10)
        df = pd.DataFrame({"a": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]})
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": 12})
        p = _profiler(test_config)
        result = p.run(df, bare_report)
        messages = [f.message for f in result.profile.findings]
        assert any("duplicate" in m.lower() for m in messages)

    def test_missing_value_finding(self, test_config, bare_report):
        # Column with 50% missing → triggers missing_value_threshold (0.20)
        data = [None] * 25 + [1.0] * 25
        df = pd.DataFrame({"x": data})
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": 50})
        p = _profiler(test_config)
        result = p.run(df, bare_report)
        messages = [f.message for f in result.profile.findings]
        assert any("missing" in m.lower() for m in messages)

    def test_skewness_finding(self, test_config, bare_report):
        # Extremely skewed distribution
        values = [1.0] * 90 + [1000.0, 5000.0, 10000.0, 50000.0, 100000.0,
                                200000.0, 300000.0, 400000.0, 500000.0, 1000000.0]
        df = pd.DataFrame({"skewed": values})
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": 100})
        p = _profiler(test_config)
        result = p.run(df, bare_report)
        messages = [f.message for f in result.profile.findings]
        assert any("skew" in m.lower() for m in messages)

    def test_zero_variance_finding(self, test_config, bare_report):
        df = pd.DataFrame({"constant": [42] * 50})
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": 50})
        p = _profiler(test_config)
        result = p.run(df, bare_report)
        messages = [f.message for f in result.profile.findings]
        assert any("one unique" in m.lower() or "zero variance" in m.lower() for m in messages)

    def test_high_cardinality_finding(self, test_config, bare_report):
        # 48 unique values in 50 rows: unique_pct=0.96 > high_cardinality_threshold=0.90
        # but < 0.99 id threshold and 48 > max_categorical_cardinality=15 → 'text' type
        values = [f"cat_{i}" for i in range(48)] + ["cat_0", "cat_1"]
        df = pd.DataFrame({"high_card": values})
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": 50})
        p = _profiler(test_config)
        result = p.run(df, bare_report)
        messages = [f.message for f in result.profile.findings]
        assert any("cardinality" in m.lower() for m in messages)


# ---------------------------------------------------------------------------
# Full run on DataFrame
# ---------------------------------------------------------------------------

class TestProfilerRun:
    def test_run_populates_column_details(self, test_config, bare_report, titanic_small_df):
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": len(titanic_small_df)})
        p = _profiler(test_config)
        result = p.run(titanic_small_df, bare_report)
        assert len(result.profile.column_details) == len(titanic_small_df.columns)

    def test_run_populates_dataset_stats(self, test_config, bare_report, titanic_small_df):
        bare_report.metadata = bare_report.metadata.model_copy(update={"original_row_count": len(titanic_small_df)})
        p = _profiler(test_config)
        result = p.run(titanic_small_df, bare_report)
        stats = result.profile.dataset_stats
        assert stats.analyzed_row_count == len(titanic_small_df)
        assert stats.column_count == len(titanic_small_df.columns)

    def test_profiler_error_wraps_exception(self, test_config, bare_report):
        p = _profiler(test_config)
        with patch.object(pd.DataFrame, "memory_usage", side_effect=RuntimeError("boom")):
            with pytest.raises(ProfilerError):
                p.run(pd.DataFrame({"x": [1, 2, 3]}), bare_report)
