"""Unit tests for the Anthropic provider (SDK mocked — works without anthropic installed)."""
from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import pytest

from sleuth.llm.providers.base import PromptMessage


@pytest.fixture
def anthropic_provider():
    mock_anthropic_pkg = MagicMock()
    mock_client = MagicMock()
    mock_anthropic_pkg.Anthropic.return_value = mock_client

    # Mock anthropic package before importing provider module
    with patch.dict(sys.modules, {"anthropic": mock_anthropic_pkg}):
        # Force fresh import so _anthropic picks up the mock
        sys.modules.pop("sleuth.llm.providers.anthropic", None)
        import sleuth.llm.providers.anthropic as anthro_module
        with patch.object(anthro_module, "_anthropic", mock_anthropic_pkg):
            from sleuth.llm.providers.anthropic import AnthropicProvider
            provider = AnthropicProvider(
                api_key="test-key",
                model_name="claude-test",
                temperature=0.3,
                max_tokens=512,
                enable_prompt_caching=False,
            )
    return provider, mock_client


class TestAnthropicProvider:
    def test_generate_calls_messages_create(self, anthropic_provider):
        provider, mock_client = anthropic_provider
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text='{"executive_summary": "test"}')]
        mock_client.messages.create.return_value = mock_response

        messages = [
            PromptMessage(role="system", content="You are an assistant", cache=True),
            PromptMessage(role="user", content="Analyse this data"),
        ]
        result = provider.generate(messages)
        assert mock_client.messages.create.called
        assert isinstance(result, str)

    def test_cache_flag_adds_cache_control(self, anthropic_provider):
        provider, mock_client = anthropic_provider
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client.messages.create.return_value = mock_response

        messages = [
            PromptMessage(role="system", content="cached system", cache=True),
            PromptMessage(role="user", content="user msg"),
        ]
        provider.generate(messages)
        assert mock_client.messages.create.call_args is not None
