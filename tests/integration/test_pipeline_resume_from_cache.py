"""Integration: second run restores from cache and skips modules."""
from __future__ import annotations

import pytest

from tests.conftest import make_test_config, STUDENT_PERF_SMALL_CSV
from tests.integration.conftest import MODULES_NO_LLM


@pytest.mark.slow
@pytest.mark.integration
def test_second_run_uses_cache(tmp_path):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)

    # First run — build cache
    p1 = AnalysisPipeline(
        file_path=STUDENT_PERF_SMALL_CSV,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
        analysis_mode="full",
    )
    p1.run()

    # Second run — should restore from cache
    p2 = AnalysisPipeline(
        file_path=STUDENT_PERF_SMALL_CSV,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=False,
        task="supervised",
        analysis_mode="full",
    )
    p2.run()

    timing = p2.report.timing
    assert timing is not None
    # All analytical modules should be cached (time == 0.0)
    for module in MODULES_NO_LLM:
        assert timing.modules.get(module, -1) == 0.0, (
            f"Expected module '{module}' to be cached (time=0.0), "
            f"got {timing.modules.get(module)}"
        )


@pytest.mark.slow
@pytest.mark.integration
def test_force_rerun_ignores_cache(tmp_path):
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)

    p1 = AnalysisPipeline(
        file_path=STUDENT_PERF_SMALL_CSV,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,
        task="supervised",
    )
    p1.run()

    p2 = AnalysisPipeline(
        file_path=STUDENT_PERF_SMALL_CSV,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=MODULES_NO_LLM,
        force_rerun=True,  # force-rerun on second call
        task="supervised",
    )
    p2.run()

    # With force_rerun=True, modules should have non-zero times
    timing = p2.report.timing
    assert timing is not None
    non_zero = [m for m, t in timing.modules.items() if t > 0.0]
    assert len(non_zero) > 0
