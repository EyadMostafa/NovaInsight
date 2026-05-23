"""Unit tests for AnalysisReport Pydantic schemas."""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from sleuth.schemas.analysis_report import (
    AnalysisReport,
    CandidateTarget,
    ColumnDetails,
    CorrelationReport,
    DatasetProfile,
    DatasetStats,
    DimensionalityAnalysis,
    Finding,
    LLMSummary,
    Operator,
    OutlierReport,
    PipelineTiming,
    Recommendation,
    Recommendations,
    RunMetadata,
    StatisticalAnalysis,
    TargetVariableAnalysis,
    VisualizationOutput,
)


# ---------------------------------------------------------------------------
# Operator enum
# ---------------------------------------------------------------------------

def test_operator_values_are_lowercase():
    for op in Operator:
        assert op.value == op.name.lower()


def test_operator_string_subclass():
    assert isinstance(Operator.PROFILER, str)
    assert Operator("profiler") is Operator.PROFILER


# ---------------------------------------------------------------------------
# Finding
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("level", ["INFO", "WARNING", "ERROR"])
def test_finding_valid_levels(level):
    f = Finding(level=level, message="test")
    assert f.level == level


def test_finding_invalid_level_raises():
    with pytest.raises(Exception):
        Finding(level="DEBUG", message="bad")


# ---------------------------------------------------------------------------
# DatasetStats
# ---------------------------------------------------------------------------

def test_dataset_stats_construction():
    s = DatasetStats(
        original_row_count=100,
        analyzed_row_count=50,
        column_count=10,
        total_memory_usage_mb=1.5,
        duplicate_rows_count=5,
        column_pct=0.2,
        duplicates_pct=0.05,
    )
    assert s.original_row_count == 100
    assert s.duplicates_pct == 0.05


# ---------------------------------------------------------------------------
# RunMetadata
# ---------------------------------------------------------------------------

def test_run_metadata_timestamp_default():
    m = RunMetadata(
        input_file=Path("/tmp/data.csv"),
        output_dir=Path("/tmp/out"),
        file_hash="abc123",
        report_title="Test",
        analysis_mode="full",
        task="supervised",
    )
    assert isinstance(m.run_timestamp, datetime)
    assert m.run_timestamp.tzinfo is not None


def test_run_metadata_invalid_mode():
    with pytest.raises(Exception):
        RunMetadata(
            input_file="/tmp/x.csv",
            output_dir="/tmp/out",
            file_hash="x",
            report_title="T",
            analysis_mode="turbo",
            task="supervised",
        )


# ---------------------------------------------------------------------------
# AnalysisReport JSON round-trip
# ---------------------------------------------------------------------------

def test_analysis_report_json_roundtrip(bare_report):
    json_str = bare_report.model_dump_json()
    data = json.loads(json_str)
    assert "metadata" in data
    restored = AnalysisReport.model_validate_json(json_str)
    assert restored.metadata.report_title == bare_report.metadata.report_title


def test_analysis_report_optional_fields_default_none(bare_report):
    assert bare_report.target_analysis is None
    assert bare_report.statistical_analysis is None
    assert bare_report.dimensionality_analysis is None
    assert bare_report.visualizations is None
    assert bare_report.llm_summary is None
    assert bare_report.timing is None


# ---------------------------------------------------------------------------
# PipelineTiming
# ---------------------------------------------------------------------------

def test_pipeline_timing():
    pt = PipelineTiming(total_seconds=12.3, modules={"profiler": 1.2, "stats": 3.4})
    assert pt.total_seconds == pytest.approx(12.3)
    assert pt.modules["profiler"] == pytest.approx(1.2)


# ---------------------------------------------------------------------------
# LLMSummary schema
# ---------------------------------------------------------------------------

def test_llm_summary_from_canned(tmp_path):
    import json
    from pathlib import Path

    canned_path = Path(__file__).parent.parent.parent / "fixtures" / "canned_llm_response.json"
    data = json.loads(canned_path.read_text())
    summary = LLMSummary.model_validate(data)
    assert summary.executive_summary
    assert len(summary.recommendations.preprocessing_steps) > 0
    assert all(r.priority in {1, 2, 3} for r in summary.recommendations.preprocessing_steps)


# ---------------------------------------------------------------------------
# CandidateTarget
# ---------------------------------------------------------------------------

def test_candidate_target_confidence_range():
    ct = CandidateTarget(
        column_name="target",
        ml_task="classification",
        justification="test",
        confidence_score=0.85,
    )
    assert 0.0 <= ct.confidence_score <= 1.0
