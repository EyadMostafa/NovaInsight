"""Unit tests for the Ollama provider (client mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.providers.base import PromptMessage

# ---------------------------------------------------------------------------
# Fixture: OllamaProvider with the ollama client fully mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def ollama_provider():
    """Returns (OllamaProvider instance, mock_client) with ollama SDK mocked."""
    import sleuth.llm.providers.ollama as ollama_module

    mock_client_instance = MagicMock()
    mock_client_cls = MagicMock(return_value=mock_client_instance)

    with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
        # Re-import so the patched module is used.
        import importlib
        importlib.reload(ollama_module)
        from sleuth.llm.providers.ollama import OllamaProvider
        provider = OllamaProvider(
            model_name="llama3.2",
            temperature=0.3,
            max_tokens=512,
        )

    return provider, mock_client_instance


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestOllamaProvider:
    def test_generate_returns_string(self, ollama_provider):
        provider, mock_client = ollama_provider
        mock_response = MagicMock()
        mock_response.message.content = '{"key": "value"}'
        mock_client.chat.return_value = mock_response

        messages = [
            PromptMessage(role="system", content="sys"),
            PromptMessage(role="user", content="user"),
        ]
        result = provider.generate(messages)
        assert isinstance(result, str)
        assert mock_client.chat.called

    def test_json_format_set_when_schema_provided(self, ollama_provider):
        provider, mock_client = ollama_provider
        mock_response = MagicMock()
        mock_response.message.content = "{}"
        mock_client.chat.return_value = mock_response

        messages = [PromptMessage(role="user", content="go")]
        provider.generate(messages, response_schema={"type": "object"})

        call_kwargs = mock_client.chat.call_args[1]
        assert call_kwargs.get("format") == "json"

    def test_no_format_when_schema_is_none(self, ollama_provider):
        provider, mock_client = ollama_provider
        mock_response = MagicMock()
        mock_response.message.content = "plain text"
        mock_client.chat.return_value = mock_response

        messages = [PromptMessage(role="user", content="go")]
        provider.generate(messages, response_schema=None)

        call_kwargs = mock_client.chat.call_args[1]
        assert "format" not in call_kwargs

    def test_connection_error_raises_llm_summarizer_error(self, ollama_provider):
        provider, mock_client = ollama_provider
        mock_client.chat.side_effect = ConnectionRefusedError("connection refused")

        with pytest.raises(LLMSummarizerError, match="ollama serve"):
            provider.generate([PromptMessage(role="user", content="hi")])

    def test_base_url_defaults_to_localhost(self):
        import sleuth.llm.providers.ollama as ollama_module

        mock_client_cls = MagicMock()
        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            import importlib
            importlib.reload(ollama_module)
            from sleuth.llm.providers.ollama import _DEFAULT_HOST, OllamaProvider
            OllamaProvider(model_name="llama3.2", temperature=0.3, max_tokens=512)

        mock_client_cls.assert_called_once_with(host=_DEFAULT_HOST)

    def test_custom_base_url_passed_to_client(self):
        import sleuth.llm.providers.ollama as ollama_module

        mock_client_cls = MagicMock()
        custom_url = "http://10.0.0.5:11434"
        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            import importlib
            importlib.reload(ollama_module)
            from sleuth.llm.providers.ollama import OllamaProvider
            OllamaProvider(model_name="llama3.2", temperature=0.3, max_tokens=512, base_url=custom_url)

        mock_client_cls.assert_called_once_with(host=custom_url)

    def test_enable_prompt_caching_does_not_raise(self):
        import sleuth.llm.providers.ollama as ollama_module

        mock_client_cls = MagicMock()
        with patch.dict("sys.modules", {"ollama": MagicMock(Client=mock_client_cls)}):
            import importlib
            importlib.reload(ollama_module)
            from sleuth.llm.providers.ollama import OllamaProvider
            # Should not raise even though caching is not applicable for Ollama.
            OllamaProvider(
                model_name="llama3.2",
                temperature=0.3,
                max_tokens=512,
                enable_prompt_caching=True,
            )

    def test_missing_ollama_package_raises_llm_error(self):
        import importlib

        # Temporarily hide the ollama package.
        with patch.dict("sys.modules", {"ollama": None}):
            import sleuth.llm.providers.ollama as ollama_module
            importlib.reload(ollama_module)
            from sleuth.llm.providers.ollama import OllamaProvider
            with pytest.raises(LLMSummarizerError, match="ollama.*package"):
                OllamaProvider(model_name="llama3.2", temperature=0.3, max_tokens=512)
