"""Unit tests for the Gemini provider (SDK mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sleuth.llm.providers.base import PromptMessage


@pytest.fixture
def gemini_provider():
    with patch.dict("sys.modules", {"google": MagicMock(), "google.genai": MagicMock()}):
        with patch("sleuth.llm.providers.gemini.genai") as mock_genai:
            mock_client = MagicMock()
            mock_genai.Client.return_value = mock_client
            from sleuth.llm.providers.gemini import GeminiProvider

            provider = GeminiProvider(
                api_key="test-key",
                model_name="models/gemini-test",
                temperature=0.3,
                max_tokens=512,
                enable_prompt_caching=False,
            )
            provider._client = mock_client
            return provider, mock_client


class TestGeminiProvider:
    def test_generate_calls_client(self, gemini_provider):
        provider, mock_client = gemini_provider
        mock_response = MagicMock()
        mock_response.text = '{"executive_summary": "test"}'
        mock_client.models.generate_content.return_value = mock_response

        messages = [
            PromptMessage(role="system", content="Be helpful"),
            PromptMessage(role="user", content="Analyse data"),
        ]
        result = provider.generate(messages)
        assert mock_client.models.generate_content.called
        assert isinstance(result, str)

    def test_generate_returns_text(self, gemini_provider):
        provider, mock_client = gemini_provider
        mock_response = MagicMock()
        mock_response.text = "response text"
        mock_client.models.generate_content.return_value = mock_response

        messages = [PromptMessage(role="user", content="hello")]
        result = provider.generate(messages)
        assert "response" in result
