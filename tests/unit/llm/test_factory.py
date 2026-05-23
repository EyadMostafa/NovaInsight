"""Unit tests for the LLM provider factory."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from sleuth.config.config import LLMSettings
from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.factory import _create_provider_cached, create_provider
from sleuth.llm.providers.base import LLMProvider


def _settings(**kwargs) -> LLMSettings:
    defaults = {
        "provider": "google",
        "model_name": "models/test",
        "temperature": 0.3,
        "max_tokens": 512,
        "enable_prompt_caching": False,
        "base_url": None,
    }
    defaults.update(kwargs)
    return LLMSettings(**defaults)


class TestCreateProvider:
    def test_missing_api_key_raises(self):
        cfg = _settings(api_key=None)
        with pytest.raises(LLMSummarizerError, match="No API key"):
            create_provider(cfg)

    def test_ollama_no_api_key_does_not_raise(self):
        """Ollama is a local provider — no API key should be required."""
        import sys
        fake_ollama_mod = MagicMock()
        fake_ollama_mod.OllamaProvider = MagicMock(return_value=MagicMock(spec=LLMProvider))
        _create_provider_cached.cache_clear()
        with patch.dict(sys.modules, {"ollama": MagicMock(), "sleuth.llm.providers.ollama": fake_ollama_mod}):
            cfg = _settings(provider="ollama", model_name="llama3.2", api_key=None)
            create_provider(cfg)  # must not raise

    def test_ollama_base_url_passed_through(self):
        import sys
        fake_ollama_mod = MagicMock()
        fake_provider = MagicMock(spec=LLMProvider)
        fake_ollama_mod.OllamaProvider = MagicMock(return_value=fake_provider)
        _create_provider_cached.cache_clear()
        with patch.dict(sys.modules, {"ollama": MagicMock(), "sleuth.llm.providers.ollama": fake_ollama_mod}):
            cfg = _settings(provider="ollama", model_name="llama3.2", api_key=None, base_url="http://remote:11434")
            create_provider(cfg)
        call_kwargs = fake_ollama_mod.OllamaProvider.call_args[1]
        assert call_kwargs.get("base_url") == "http://remote:11434"

    def test_unsupported_provider_raises(self):
        from pydantic import SecretStr
        cfg = _settings(provider="unknown_llm", api_key=SecretStr("k"))
        with pytest.raises(LLMSummarizerError, match="Unsupported"):
            create_provider(cfg)

    def test_lru_cache_returns_same_instance(self, monkeypatch):

        fake_provider = MagicMock(spec=LLMProvider)

        with patch("sleuth.llm.providers.gemini.GeminiProvider", return_value=fake_provider):
            # Clear lru_cache between test runs
            _create_provider_cached.cache_clear()
            p1 = _create_provider_cached(
                provider="google",
                model_name="models/test",
                api_key="key-abc",
                temperature=0.3,
                max_tokens=512,
                enable_prompt_caching=False,
            )
            p2 = _create_provider_cached(
                provider="google",
                model_name="models/test",
                api_key="key-abc",
                temperature=0.3,
                max_tokens=512,
                enable_prompt_caching=False,
            )
        assert p1 is p2

    def test_google_provider_instantiated(self):

        fake = MagicMock(spec=LLMProvider)
        _create_provider_cached.cache_clear()
        with patch("sleuth.llm.providers.gemini.GeminiProvider", return_value=fake) as mock_cls:
            _create_provider_cached(
                provider="google",
                model_name="models/x",
                api_key="k",
                temperature=0.3,
                max_tokens=100,
                enable_prompt_caching=False,
            )
        mock_cls.assert_called_once()

    def test_anthropic_provider_instantiated(self):
        import sys
        fake = MagicMock(spec=LLMProvider)
        mock_anthro_mod = MagicMock()
        mock_anthro_mod.AnthropicProvider = MagicMock(return_value=fake)
        _create_provider_cached.cache_clear()
        with patch.dict(sys.modules, {
            "anthropic": MagicMock(),
            "sleuth.llm.providers.anthropic": mock_anthro_mod,
        }):
            _create_provider_cached(
                provider="anthropic",
                model_name="claude-test",
                api_key="k",
                temperature=0.3,
                max_tokens=100,
                enable_prompt_caching=False,
            )
        mock_anthro_mod.AnthropicProvider.assert_called_once()

    def test_openai_provider_instantiated(self):
        import importlib
        openai_mod = importlib.import_module("sleuth.llm.providers.openai")
        fake = MagicMock(spec=LLMProvider)
        _create_provider_cached.cache_clear()
        with patch.object(openai_mod, "OpenAIProvider", return_value=fake) as mock_cls:
            _create_provider_cached(
                provider="openai",
                model_name="gpt-test",
                api_key="k",
                temperature=0.3,
                max_tokens=100,
                enable_prompt_caching=False,
            )
        mock_cls.assert_called_once()
