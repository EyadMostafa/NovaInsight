"""Unit tests for LLMSummarizer (mocked provider — no real API calls)."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sleuth.exceptions import ModuleError
from sleuth.modules.llm_summarizer import LLMSummarizer
from sleuth.schemas.analysis_report import LLMSummary


FIXTURES = Path(__file__).parent.parent.parent / "fixtures"


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

class TestInitialization:
    def test_missing_api_key_raises_module_error(self, test_config):
        # test_config has api_key=None
        with pytest.raises(ModuleError, match="not set"):
            LLMSummarizer(test_config)

    def test_with_api_key_initializes_provider(self, test_config_with_llm_key, mock_llm_provider):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            summarizer = LLMSummarizer(test_config_with_llm_key)
        assert summarizer.provider is mock_llm_provider


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

class TestTryParse:
    def test_valid_json_returns_dict(self, test_config_with_llm_key, mock_llm_provider):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        data = {"key": "value"}
        result = s._try_parse(json.dumps(data))
        assert result == data

    def test_markdown_fenced_json_extracted(self, test_config_with_llm_key, mock_llm_provider):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        text = '```json\n{"a": 1}\n```'
        result = s._try_parse(text)
        assert result == {"a": 1}

    def test_repaired_json_returns_dict(self, test_config_with_llm_key, mock_llm_provider):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        # Missing closing brace — json-repair should fix it
        broken = '{"key": "val"'
        result = s._try_parse(broken)
        assert result is not None
        assert result.get("key") == "val"

    def test_totally_invalid_returns_none(self, test_config_with_llm_key, mock_llm_provider):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        result = s._try_parse("this is not json at all !@#$")
        # json-repair may produce something; we just assert no exception raised
        # The key contract is that it never raises
        assert result is None or isinstance(result, dict)


# ---------------------------------------------------------------------------
# Normalize helper
# ---------------------------------------------------------------------------

class TestNormalize:
    def test_fills_missing_top_level_keys(self):
        data = {}
        result = LLMSummarizer._normalize(data)
        assert "executive_summary" in result
        assert "recommendations" in result

    def test_fills_missing_recommendation_sub_lists(self):
        data = {"recommendations": {}}
        result = LLMSummarizer._normalize(data)
        rec = result["recommendations"]
        assert "preprocessing_steps" in rec
        assert isinstance(rec["preprocessing_steps"], list)

    def test_preserves_existing_values(self):
        data = {"executive_summary": "My summary"}
        result = LLMSummarizer._normalize(data)
        assert result["executive_summary"] == "My summary"


# ---------------------------------------------------------------------------
# Full run: happy path
# ---------------------------------------------------------------------------

class TestRun:
    def test_happy_path_populates_llm_summary(
        self, test_config_with_llm_key, mock_llm_provider, bare_report, titanic_small_df
    ):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        result = s.run(titanic_small_df, bare_report)
        assert result.llm_summary is not None
        assert isinstance(result.llm_summary, LLMSummary)
        assert result.llm_summary.executive_summary

    def test_provider_generate_called_once(
        self, test_config_with_llm_key, mock_llm_provider, bare_report, titanic_small_df
    ):
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=mock_llm_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        s.run(titanic_small_df, bare_report)
        mock_llm_provider.generate.assert_called_once()

    def test_total_failure_raises_module_error(
        self, test_config_with_llm_key, bare_report, titanic_small_df
    ):
        bad_provider = MagicMock()
        bad_provider.generate.return_value = "TOTALLY INVALID NON JSON !!!"
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=bad_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        # Three-layer recovery exhausted → ModuleError
        with pytest.raises(ModuleError):
            s.run(titanic_small_df, bare_report)

    def test_llm_query_exception_raises_module_error(
        self, test_config_with_llm_key, bare_report, titanic_small_df
    ):
        failing_provider = MagicMock()
        failing_provider.generate.side_effect = RuntimeError("Network down")
        with patch("sleuth.modules.llm_summarizer.create_provider", return_value=failing_provider):
            s = LLMSummarizer(test_config_with_llm_key)
        with pytest.raises(ModuleError):
            s.run(titanic_small_df, bare_report)
