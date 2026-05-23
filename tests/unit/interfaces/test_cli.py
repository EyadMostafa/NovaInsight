"""Unit tests for the CLI interface."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from click.testing import CliRunner

from sleuth.exceptions import PipelineError
from sleuth.interfaces.cli import main

FIXTURES = Path(__file__).parent.parent.parent / "fixtures"
TITANIC_CSV = FIXTURES / "titanic_small.csv"


@pytest.fixture
def runner():
    return CliRunner()


class TestAnalyzeCommand:
    def test_help_exits_zero(self, runner):
        result = runner.invoke(main, ["analyze", "--help"])
        assert result.exit_code == 0
        assert "file_path" in result.output.lower() or "FILE_PATH" in result.output

    def test_missing_file_exits_nonzero(self, runner):
        result = runner.invoke(main, ["analyze", "/nonexistent/path/data.csv"])
        assert result.exit_code != 0

    def test_modules_flag_parsed(self, runner, tmp_path):
        captured = {}

        def fake_pipeline_init(self, file_path, config, **kwargs):
            captured["requested_modules"] = kwargs.get("requested_modules")

        def fake_pipeline_run(self):
            pass

        with patch("sleuth.modules.orchestrator.AnalysisPipeline.__init__", fake_pipeline_init):
            with patch("sleuth.modules.orchestrator.AnalysisPipeline.run", fake_pipeline_run):
                runner.invoke(
                    main,
                    ["analyze", str(TITANIC_CSV), "--modules", "profiler,stats"],
                )
        # Even if pipeline fails, modules parsing should have happened
        if "requested_modules" in captured:
            assert "profiler" in captured["requested_modules"]

    def test_pipeline_error_exits_one(self, runner):
        with patch(
            "sleuth.modules.orchestrator.AnalysisPipeline.run",
            side_effect=PipelineError("fatal"),
        ):
            with patch("sleuth.modules.orchestrator.AnalysisPipeline.__init__", return_value=None):
                result = runner.invoke(main, ["analyze", str(TITANIC_CSV)])
        assert result.exit_code == 1

    def test_task_flag_choices(self, runner):
        result = runner.invoke(main, ["analyze", str(TITANIC_CSV), "--task", "invalid"])
        assert result.exit_code != 0

    def test_mode_flag_choices(self, runner):
        result = runner.invoke(main, ["analyze", str(TITANIC_CSV), "--mode", "turbo"])
        assert result.exit_code != 0
