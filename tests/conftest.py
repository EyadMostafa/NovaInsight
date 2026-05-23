"""
Shared pytest fixtures for the Sleuth test suite.

All fixtures create isolated tmp_path-scoped directories so tests never
touch ~/.sleuth_cache or the project output directory.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import matplotlib
matplotlib.use("Agg")

import pandas as pd
import pytest

from sleuth.config.config import (
    SleuthConfig,
    CacheSettings,
    AnalysisSettings,
    OutputSettings,
    ProfilerSettings,
    TargetDetectionSettings,
    StatisticsSettings,
    DimensionalityReduction,
    VisualizationSettings,
    LLMSettings,
    GeneralSettings,
)
from sleuth.llm.providers.base import LLMProvider
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    RunMetadata,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"
TITANIC_SMALL_CSV = FIXTURES_DIR / "titanic_small.csv"
STUDENT_PERF_SMALL_CSV = FIXTURES_DIR / "student_performance_small.csv"


# ---------------------------------------------------------------------------
# Autouse: assert Agg backend is active
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session", autouse=True)
def headless_matplotlib():
    assert matplotlib.get_backend().lower() == "agg", (
        f"Expected Agg backend, got {matplotlib.get_backend()}. "
        "visualizer.py may have lost its matplotlib.use('Agg') guard."
    )


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------

def make_test_config(tmp_path: Path, tsne_max_iter: int = 300) -> SleuthConfig:
    """Build a fully self-contained SleuthConfig writing to tmp_path."""
    return SleuthConfig(
        general=GeneralSettings(debug=False, log_level="WARNING", suppress_user_warnings=True),
        cache=CacheSettings(enabled=True, directory_path=tmp_path / "cache"),
        analysis=AnalysisSettings(
            default_mode="full",
            fast_mode_sample_rows=50,
            output=OutputSettings(default_directory=tmp_path / "out"),
        ),
        profiler=ProfilerSettings(
            duplicate_threshold=0.10,
            dimensionality_threshold=0.5,
            total_memory_threshold=500,
            missing_value_threshold=0.20,
            high_cardinality_threshold=0.90,
            skewness_threshold=1.0,
            memory_hog_threshold=0.20,
            max_categorical_cardinality=15,
        ),
        target_detection=TargetDetectionSettings(
            max_categorical_cardinality=50,
            id_uniqueness_threshold=0.99,
        ),
        statistics=StatisticsSettings(),
        dimensionality_reduction=DimensionalityReduction(
            tsne_max_iter=tsne_max_iter,
            pca_n_ratios=5,
        ),
        visualization=VisualizationSettings(dpi=72, kmeans_n_clusters=3),
        llm=LLMSettings(api_key=None),
    )


@pytest.fixture
def test_config(tmp_path: Path) -> SleuthConfig:
    return make_test_config(tmp_path)


# ---------------------------------------------------------------------------
# DataFrames
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def titanic_df() -> pd.DataFrame:
    return pd.read_csv(TITANIC_SMALL_CSV)


@pytest.fixture(scope="session")
def titanic_small_df() -> pd.DataFrame:
    return pd.read_csv(TITANIC_SMALL_CSV).head(50)


@pytest.fixture(scope="session")
def student_perf_df() -> pd.DataFrame:
    return pd.read_csv(STUDENT_PERF_SMALL_CSV)


# ---------------------------------------------------------------------------
# AnalysisReport builders
# ---------------------------------------------------------------------------

def _make_output_dir(tmp_path: Path) -> Path:
    d = tmp_path / "out" / "Sleuth_reports" / "Test Analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_bare_report(tmp_path: Path) -> AnalysisReport:
    """Single source of truth for a zeroed AnalysisReport."""
    output_dir = _make_output_dir(tmp_path)
    metadata = RunMetadata(
        input_file=tmp_path / "data.csv",
        output_dir=output_dir,
        file_hash="deadbeef" * 8,
        report_title="Test Analysis",
        analysis_mode="fast",
        task="supervised",
        user_target=None,
        original_row_count=0,
    )
    return AnalysisReport(metadata=metadata, findings=[])


@pytest.fixture
def bare_report(tmp_path: Path) -> AnalysisReport:
    """Minimal AnalysisReport before any module has run."""
    return _build_bare_report(tmp_path)


# ---------------------------------------------------------------------------
# LLM mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_llm_provider() -> MagicMock:
    """MagicMock LLMProvider returning the canned LLMSummary JSON."""
    canned = (FIXTURES_DIR / "canned_llm_response.json").read_text()
    provider = MagicMock(spec=LLMProvider)
    provider.generate.return_value = canned
    return provider


# ---------------------------------------------------------------------------
# Pipeline helper: run DataProfiler on a report, updating original_row_count
# ---------------------------------------------------------------------------

@pytest.fixture
def run_profiler():
    """Fixture factory: profile a DataFrame and return the enriched report.

    Usage::
        def test_something(self, test_config, bare_report, titanic_small_df, run_profiler):
            report = run_profiler(test_config, titanic_small_df, bare_report)
    """
    from sleuth.modules.data_profiler import DataProfiler

    def _impl(config, df, report):
        report.metadata = report.metadata.model_copy(update={"original_row_count": len(df)})
        DataProfiler(config).run(df, report)
        return report

    return _impl


# ---------------------------------------------------------------------------
# Convenience: config with a dummy LLM API key (needed for LLMSummarizer init)
# ---------------------------------------------------------------------------

@pytest.fixture
def test_config_with_llm_key(tmp_path: Path) -> SleuthConfig:
    from pydantic import SecretStr
    cfg = make_test_config(tmp_path)
    # Pydantic frozen model — rebuild with key set
    return SleuthConfig(
        **{
            **cfg.model_dump(),
            "llm": {
                **cfg.llm.model_dump(),
                "api_key": SecretStr("test-key"),
            },
        }
    )
