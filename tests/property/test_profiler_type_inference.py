"""Property-based tests for DataProfiler._infer_logical_type."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sleuth.exceptions import ProfilerError
from sleuth.modules.data_profiler import DataProfiler

VALID_INFERRED_TYPES = {"id", "boolean", "numerical", "categorical", "datetime", "text"}

# Pydantic defaults (no YAML loaded — matches ProfilerSettings defaults)
_MIN_ROWS = 5


def _profiler(test_config):
    return DataProfiler(test_config)


# ---------------------------------------------------------------------------
# Numeric series — must return a known type and never raise
# ---------------------------------------------------------------------------


@given(
    values=st.lists(
        st.one_of(st.integers(min_value=-1000, max_value=1000), st.none()),
        min_size=_MIN_ROWS,
        max_size=200,
    )
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_numeric_series_never_raises(values, test_config):
    s = pd.Series(values, dtype="Int64").astype(float)
    p = _profiler(test_config)
    result = p._infer_logical_type(s)
    assert result in VALID_INFERRED_TYPES


# ---------------------------------------------------------------------------
# String series — must return a known type and never raise
# ---------------------------------------------------------------------------


@given(
    values=st.lists(
        st.one_of(st.text(max_size=20), st.none()),
        min_size=_MIN_ROWS,
        max_size=100,
    )
)
@settings(max_examples=80, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_string_series_never_raises(values, test_config):
    s = pd.Series(values, dtype=object)
    p = _profiler(test_config)
    result = p._infer_logical_type(s)
    assert result in VALID_INFERRED_TYPES


# ---------------------------------------------------------------------------
# All-null series — must return a known type and never raise
# ---------------------------------------------------------------------------


@given(n=st.integers(min_value=_MIN_ROWS, max_value=50))
@settings(max_examples=20, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_all_null_series_never_raises(n, test_config):
    s = pd.Series([None] * n, dtype=object)
    p = _profiler(test_config)
    result = p._infer_logical_type(s)
    assert result in VALID_INFERRED_TYPES


# ---------------------------------------------------------------------------
# Boolean series — always returns 'boolean'
# ---------------------------------------------------------------------------


@given(values=st.lists(st.booleans(), min_size=_MIN_ROWS, max_size=100))
@settings(max_examples=40, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_bool_dtype_always_boolean(values, test_config):
    s = pd.Series(values, dtype=bool)
    p = _profiler(test_config)
    assert p._infer_logical_type(s) == "boolean"


# ---------------------------------------------------------------------------
# Full profiler run never raises ProfilerError on valid DataFrames
# ---------------------------------------------------------------------------


@given(
    n_rows=st.integers(min_value=10, max_value=80),
    n_num=st.integers(min_value=1, max_value=4),
    n_cat=st.integers(min_value=1, max_value=3),
)
@settings(
    max_examples=30, suppress_health_check=[HealthCheck.function_scoped_fixture], deadline=10000
)
def test_profiler_run_never_raises_on_valid_df(n_rows, n_num, n_cat, test_config, tmp_path):
    from tests.conftest import _build_bare_report

    rng = np.random.default_rng(42)
    data = {}
    for i in range(n_num):
        data[f"num_{i}"] = rng.normal(0, 1, n_rows)
    for i in range(n_cat):
        data[f"cat_{i}"] = rng.choice(["a", "b", "c"], n_rows)

    df = pd.DataFrame(data)
    report = _build_bare_report(tmp_path)
    report.metadata = report.metadata.model_copy(update={"original_row_count": n_rows})

    profiler = DataProfiler(test_config)
    # Should not raise ProfilerError
    try:
        profiler.run(df, report)
    except ProfilerError:
        pytest.fail("ProfilerError raised on valid DataFrame")
