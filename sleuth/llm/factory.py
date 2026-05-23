"""
factory.py

Creates the configured LLMProvider instance.

Caching: _create_provider_cached is keyed on hashable primitives extracted from
LLMSettings so that repeated calls with identical config (common in multi-run
sessions) return the same provider object without re-initialising the SDK client.

Provider SDKs are imported lazily so uninstalled optional dependencies don't
break imports for users who don't need them.
"""

from __future__ import annotations

from functools import lru_cache

from sleuth.config.config import LLMSettings
from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.providers.base import LLMProvider
from sleuth.utils.logger import get_logger

logger = get_logger("sleuth.llm.factory")

_SUPPORTED_PROVIDERS = ("google", "anthropic", "openai", "ollama")

# Providers that run locally and do not require an API key.
_LOCAL_PROVIDERS = frozenset({"ollama"})


def create_provider(config: LLMSettings) -> LLMProvider:
    """
    Return the LLM provider for the given config, reusing a cached instance
    when the config is identical to a previous call.

    Raises:
        LLMSummarizerError: If the provider name is unsupported or api_key is missing.
    """
    provider = config.provider.lower()

    if provider not in _SUPPORTED_PROVIDERS:
        raise LLMSummarizerError(
            f"Unsupported LLM provider: '{config.provider}'. "
            f"Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}"
        )

    if provider not in _LOCAL_PROVIDERS and not config.api_key:
        raise LLMSummarizerError(
            f"No API key found for provider '{config.provider}'. "
            "Set SLEUTH_LLM_API_KEY in your .env file."
        )

    return _create_provider_cached(
        provider=provider,
        model_name=config.model_name,
        api_key=config.api_key.get_secret_value() if config.api_key else "",
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_prompt_caching=config.enable_prompt_caching,
        base_url=config.base_url or "",
    )


@lru_cache(maxsize=8)
def _create_provider_cached(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    enable_prompt_caching: bool,
    base_url: str = "",
) -> LLMProvider:
    """Cached inner factory — arguments must all be hashable primitives."""
    # Common kwargs shared by all cloud providers.
    cloud_kwargs = dict(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_prompt_caching=enable_prompt_caching,
    )

    if provider == "google":
        from sleuth.llm.providers.gemini import GeminiProvider

        logger.info(f"Initializing Gemini provider ({model_name})")
        return GeminiProvider(**cloud_kwargs)

    if provider == "anthropic":
        from sleuth.llm.providers.anthropic import AnthropicProvider

        logger.info(f"Initializing Anthropic provider ({model_name})")
        return AnthropicProvider(**cloud_kwargs)

    if provider == "ollama":
        from sleuth.llm.providers.ollama import OllamaProvider

        logger.info(f"Initializing Ollama provider ({model_name})")
        return OllamaProvider(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            enable_prompt_caching=enable_prompt_caching,
            base_url=base_url or None,
        )

    from sleuth.llm.providers.openai import OpenAIProvider

    logger.info(f"Initializing OpenAI provider ({model_name})")
    return OpenAIProvider(**cloud_kwargs)
