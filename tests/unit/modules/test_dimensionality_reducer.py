"""Unit tests for DimensionalityReducer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sleuth.modules.dimensionality_reducer import DimensionalityReducer
from sleuth.schemas.analysis_report import DimensionalityAnalysis


def _reducer(test_config):
    return DimensionalityReducer(test_config)


def _numeric_df(n_rows=60, n_cols=5, seed=42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0, 1, (n_rows, n_cols)),
        columns=[f"f{i}" for i in range(n_cols)],
    )


# ---------------------------------------------------------------------------
# PCA
# ---------------------------------------------------------------------------

class TestRunPCA:
    def test_variance_ratios_sum_leq_one(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df()
        _, ratios = r._run_pca(df)
        assert sum(ratios) <= 1.0 + 1e-9

    def test_variance_ratios_length_bounded_by_pca_n_ratios(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df(n_cols=5)
        _, ratios = r._run_pca(df)
        # test_config sets pca_n_ratios=5; data has 5 features
        assert len(ratios) <= test_config.dimensionality_reduction.pca_n_ratios

    def test_embeddings_have_two_columns(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df()
        embeddings, _ = r._run_pca(df)
        assert embeddings.shape[1] == 2


# ---------------------------------------------------------------------------
# t-SNE
# ---------------------------------------------------------------------------

class TestRunTSNE:
    def test_embeddings_shape(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df(n_rows=40, n_cols=4)
        result = r._run_tsne(df)
        assert result.shape == (40, 2)

    def test_columns_named_tsne(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df(n_rows=20, n_cols=3)
        result = r._run_tsne(df)
        assert all(c.startswith("TSNE") for c in result.columns)


# ---------------------------------------------------------------------------
# UMAP
# ---------------------------------------------------------------------------

class TestRunUMAP:
    def test_embeddings_shape(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df(n_rows=40, n_cols=4)
        result = r._run_umap(df)
        assert result.shape == (40, 2)

    def test_columns_named_umap(self, test_config):
        r = _reducer(test_config)
        df = _numeric_df(n_rows=20, n_cols=3)
        result = r._run_umap(df)
        assert all(c.startswith("UMAP") for c in result.columns)


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

class TestPrepareData:
    def test_returns_none_when_no_numeric_features(self, test_config, bare_report, run_profiler):
        df = pd.DataFrame({"cat": ["a", "b", "c"] * 10})
        report = run_profiler(test_config, df, bare_report)
        r = _reducer(test_config)
        processed, finding = r._prepare_data(df, report)
        assert processed is None
        assert finding is not None

    def test_returns_none_when_fewer_than_3_valid_numerics(self, test_config, bare_report, run_profiler):
        df = pd.DataFrame({
            "a": [1.0, 2.0, 3.0] * 10,
            "b": [4.0, 5.0, 6.0] * 10,
        })
        report = run_profiler(test_config, df, bare_report)
        r = _reducer(test_config)
        processed, finding = r._prepare_data(df, report)
        assert processed is None

    def test_drops_zero_variance_columns(self, test_config, bare_report, run_profiler):
        df = pd.DataFrame({
            "a": [1.0] * 30,  # zero variance → should be dropped
            "b": list(range(30)),
            "c": list(range(30, 60)),
            "d": list(range(60, 90)),
        })
        report = run_profiler(test_config, df, bare_report)
        r = _reducer(test_config)
        processed, _ = r._prepare_data(df, report)
        if processed is not None:
            assert "a" not in processed.columns


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_csv_written_to_embeddings_subdir(self, test_config, tmp_path):
        r = _reducer(test_config)
        df = pd.DataFrame({"PC1": [1.0, 2.0], "PC2": [3.0, 4.0]})
        path = r._save_results(df, tmp_path, "pca")
        from pathlib import Path
        assert Path(path).exists()
        assert "pca_embeddings.csv" in str(path)


# ---------------------------------------------------------------------------
# Full run
# ---------------------------------------------------------------------------

class TestDimensionalityReducerRun:
    def test_run_populates_dim_analysis(self, test_config, bare_report, titanic_small_df, run_profiler):
        report = run_profiler(test_config, titanic_small_df, bare_report)
        result = _reducer(test_config).run(titanic_small_df, report)
        assert result.dimensionality_analysis is not None

    def test_embeddings_files_exist(self, test_config, bare_report, titanic_small_df, run_profiler):
        from pathlib import Path
        report = run_profiler(test_config, titanic_small_df, bare_report)
        result = _reducer(test_config).run(titanic_small_df, report)
        dim = result.dimensionality_analysis
        if dim.pca_embeddings_path:
            assert Path(dim.pca_embeddings_path).exists()

    def test_pca_ratios_present(self, test_config, bare_report, run_profiler):
        # Use synthetic data: 25 unique values per column in 50 rows → 50% unique → 'numerical'
        df = pd.DataFrame({f"num_{i}": list(range(25)) * 2 for i in range(4)})
        report = run_profiler(test_config, df, bare_report)
        result = _reducer(test_config).run(df, report)
        dim = result.dimensionality_analysis
        assert dim.pca_explained_variance_ratios is not None
        assert len(dim.pca_explained_variance_ratios) > 0
