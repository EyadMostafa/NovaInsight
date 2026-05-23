"""Integration: pipeline handles Parquet (.parquet) and Feather (.feather) input files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tests.conftest import STUDENT_PERF_SMALL_CSV, make_test_config
from tests.integration.conftest import MODULES_NO_LLM

# ---------------------------------------------------------------------------
# Session-scoped fixtures: generate columnar fixtures from the CSV once per run
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def student_perf_parquet(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("parquet_fixture")
    path = out_dir / "student_performance_small.parquet"
    pd.read_csv(STUDENT_PERF_SMALL_CSV).to_parquet(path, index=False)
    return path


@pytest.fixture(scope="session")
def student_perf_feather(tmp_path_factory) -> Path:
    out_dir = tmp_path_factory.mktemp("feather_fixture")
    path = out_dir / "student_performance_small.feather"
    pd.read_csv(STUDENT_PERF_SMALL_CSV).to_feather(path)
    return path


# ---------------------------------------------------------------------------
# Parquet tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_parquet_pipeline_completes(tmp_path, student_perf_parquet):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    pipeline = AnalysisPipeline(
        file_path=student_perf_parquet,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
        analysis_mode="full",
    )
    pipeline.run()
    assert pipeline.report.profile is not None


@pytest.mark.slow
@pytest.mark.integration
def test_parquet_row_count_matches_csv(tmp_path, student_perf_parquet):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    pipeline = AnalysisPipeline(
        file_path=student_perf_parquet,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=["profiler"],
        force_rerun=True,
        task="supervised",
    )
    pipeline.run()
    df_csv = pd.read_csv(STUDENT_PERF_SMALL_CSV)
    assert pipeline.report.profile.dataset_stats.analyzed_row_count == len(df_csv)


@pytest.mark.slow
@pytest.mark.integration
def test_parquet_html_report_created(tmp_path, student_perf_parquet):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    pipeline = AnalysisPipeline(
        file_path=student_perf_parquet,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
    )
    pipeline.run()
    html_files = list(Path(pipeline.report.metadata.output_dir).glob("*.html"))
    assert len(html_files) == 1
    assert pipeline.report.html_report_path is not None


@pytest.mark.slow
@pytest.mark.integration
def test_parquet_fast_mode_samples_randomly(tmp_path, student_perf_parquet):
    """Fast mode should return `fast_mode_sample_rows` rows, not the first N."""
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path, tsne_max_iter=50)
    df_full = pd.read_csv(STUDENT_PERF_SMALL_CSV)

    pipeline = AnalysisPipeline(
        file_path=student_perf_parquet,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=["profiler"],
        force_rerun=True,
        task="supervised",
        analysis_mode="fast",
    )
    pipeline.run()

    sample_size = config.analysis.fast_mode_sample_rows
    actual_rows = pipeline.report.profile.dataset_stats.analyzed_row_count
    assert actual_rows == min(sample_size, len(df_full))


# ---------------------------------------------------------------------------
# Feather tests
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_feather_pipeline_completes(tmp_path, student_perf_feather):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    pipeline = AnalysisPipeline(
        file_path=student_perf_feather,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
        analysis_mode="full",
    )
    pipeline.run()
    assert pipeline.report.profile is not None


@pytest.mark.slow
@pytest.mark.integration
def test_feather_row_count_matches_csv(tmp_path, student_perf_feather):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    pipeline = AnalysisPipeline(
        file_path=student_perf_feather,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=["profiler"],
        force_rerun=True,
        task="supervised",
    )
    pipeline.run()
    df_csv = pd.read_csv(STUDENT_PERF_SMALL_CSV)
    assert pipeline.report.profile.dataset_stats.analyzed_row_count == len(df_csv)
