"""Unit tests for the OpenAI provider (SDK mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sleuth.llm.providers.base import PromptMessage


@pytest.fixture
def openai_provider():
    import sleuth.llm.providers.openai as openai_module
    mock_client = MagicMock()
    mock_cls = MagicMock(return_value=mock_client)
    with patch.object(openai_module, "_OpenAI", mock_cls):
        from sleuth.llm.providers.openai import OpenAIProvider
        provider = OpenAIProvider(
            api_key="test-key",
            model_name="gpt-test",
            temperature=0.3,
            max_tokens=512,
            enable_prompt_caching=False,
        )
    return provider, mock_client


class TestOpenAIProvider:
    def test_generate_calls_chat_completions(self, openai_provider):
        provider, mock_client = openai_provider
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content='{"key": "val"}'))]
        mock_client.chat.completions.create.return_value = mock_response

        messages = [
            PromptMessage(role="system", content="sys"),
            PromptMessage(role="user", content="user"),
        ]
        result = provider.generate(messages)
        assert mock_client.chat.completions.create.called
        assert isinstance(result, str)
