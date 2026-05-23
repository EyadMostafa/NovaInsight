"""Integration test helpers shared across the integration test suite."""
from __future__ import annotations

from pathlib import Path

import pytest

from tests.conftest import make_test_config, STUDENT_PERF_SMALL_CSV

MODULES_NO_LLM = ["profiler", "target", "stats", "dim_reduction", "viz", "report"]
MODULES_NO_LLM_UNSUPERVISED = ["profiler", "stats", "dim_reduction", "viz", "report"]


def build_pipeline(tmp_path: Path, **kwargs):
    """Create an AnalysisPipeline wired to tmp_path with LLM excluded."""
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    defaults = dict(
        file_path=STUDENT_PERF_SMALL_CSV,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
        analysis_mode="full",
    )
    defaults.update(kwargs)
    return AnalysisPipeline(**defaults)
