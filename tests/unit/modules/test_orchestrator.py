"""Unit tests for AnalysisPipeline (orchestrator)."""

from __future__ import annotations

from pathlib import Path

from tests.conftest import make_test_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pipeline(tmp_path: Path, csv_path: Path):
    """Return a bare pipeline pointing at csv_path, no modules requested."""
    from sleuth.modules.orchestrator import AnalysisPipeline

    config = make_test_config(tmp_path)
    return AnalysisPipeline(
        file_path=csv_path,
        config=config,
        output_dir=tmp_path / "out",
        requested_modules=["profiler"],
        force_rerun=True,
        task="supervised",
    )


def _write_csv(path: Path, delimiter: str) -> None:
    """Write a tiny three-column CSV using the given delimiter."""
    lines = [
        delimiter.join(["name", "age", "score"]),
        delimiter.join(["alice", "30", "9.5"]),
        delimiter.join(["bob", "25", "8.0"]),
        delimiter.join(["carol", "35", "7.2"]),
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Delimiter detection
# ---------------------------------------------------------------------------


class TestDetectCsvDelimiter:
    def test_comma_detected(self, tmp_path):
        csv_path = tmp_path / "comma.csv"
        _write_csv(csv_path, ",")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        assert pipeline._detect_csv_delimiter() == ","

    def test_semicolon_detected(self, tmp_path):
        csv_path = tmp_path / "semi.csv"
        _write_csv(csv_path, ";")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        assert pipeline._detect_csv_delimiter() == ";"

    def test_tab_detected(self, tmp_path):
        csv_path = tmp_path / "tab.csv"
        _write_csv(csv_path, "\t")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        assert pipeline._detect_csv_delimiter() == "\t"

    def test_pipe_detected(self, tmp_path):
        csv_path = tmp_path / "pipe.csv"
        _write_csv(csv_path, "|")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        assert pipeline._detect_csv_delimiter() == "|"

    def test_fallback_to_comma_on_empty_file(self, tmp_path):
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("", encoding="utf-8")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        assert pipeline._detect_csv_delimiter() == ","

    def test_fallback_to_comma_on_single_column(self, tmp_path):
        """Single-column files give the sniffer nothing to go on; must not crash."""
        csv_path = tmp_path / "single.csv"
        csv_path.write_text("value\n1\n2\n3\n", encoding="utf-8")
        pipeline = _make_pipeline(tmp_path, csv_path)
        pipeline.file_path = csv_path
        result = pipeline._detect_csv_delimiter()
        assert isinstance(result, str) and len(result) == 1


# ---------------------------------------------------------------------------
# End-to-end: non-standard delimiter flows through the pipeline
# ---------------------------------------------------------------------------


class TestNonStandardDelimiterPipeline:
    def test_semicolon_csv_profiled_correctly(self, tmp_path):
        """A semicolon-delimited CSV must produce the right column count."""
        csv_path = tmp_path / "semi.csv"
        _write_csv(csv_path, ";")

        from sleuth.modules.orchestrator import AnalysisPipeline

        config = make_test_config(tmp_path)
        pipeline = AnalysisPipeline(
            file_path=csv_path,
            config=config,
            output_dir=tmp_path / "out",
            requested_modules=["profiler"],
            force_rerun=True,
            task="supervised",
        )
        pipeline.run()

        # Without dialect-aware loading, pandas would see a single column
        # containing the full "name;age;score" string.
        assert pipeline.report.profile.dataset_stats.column_count == 3

    def test_tab_csv_profiled_correctly(self, tmp_path):
        # Users sometimes save tab-delimited data with a .csv extension;
        # this is exactly the fragility we're guarding against.
        csv_path = tmp_path / "tab.csv"
        _write_csv(csv_path, "\t")

        from sleuth.modules.orchestrator import AnalysisPipeline

        config = make_test_config(tmp_path)
        pipeline = AnalysisPipeline(
            file_path=csv_path,
            config=config,
            output_dir=tmp_path / "out",
            requested_modules=["profiler"],
            force_rerun=True,
            task="supervised",
        )
        pipeline.run()

        assert pipeline.report.profile.dataset_stats.column_count == 3
