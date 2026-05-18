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

from novainsight.utils.logger import get_logger
from functools import lru_cache

from novainsight.config.config import LLMSettings
from novainsight.exceptions import LLMSummarizerError
from novainsight.llm.providers.base import LLMProvider

logger = get_logger("novainsight.llm.factory")

_SUPPORTED_PROVIDERS = ("google", "anthropic", "openai")


def create_provider(config: LLMSettings) -> LLMProvider:
    """
    Return the LLM provider for the given config, reusing a cached instance
    when the config is identical to a previous call.

    Raises:
        LLMSummarizerError: If the provider name is unsupported or api_key is missing.
    """
    if config.provider.lower() not in _SUPPORTED_PROVIDERS:
        raise LLMSummarizerError(
            f"Unsupported LLM provider: '{config.provider}'. "
            f"Supported providers: {', '.join(_SUPPORTED_PROVIDERS)}"
        )

    if not config.api_key:
        raise LLMSummarizerError(
            f"No API key found for provider '{config.provider}'. "
            "Set NOVA_INSIGHT_LLM_API_KEY in your .env file."
        )

    return _create_provider_cached(
        provider=config.provider.lower(),
        model_name=config.model_name,
        api_key=config.api_key.get_secret_value(),
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        enable_prompt_caching=config.enable_prompt_caching,
    )


@lru_cache(maxsize=8)
def _create_provider_cached(
    provider: str,
    model_name: str,
    api_key: str,
    temperature: float,
    max_tokens: int,
    enable_prompt_caching: bool,
) -> LLMProvider:
    """Cached inner factory — arguments must all be hashable primitives."""
    kwargs = dict(
        api_key=api_key,
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens,
        enable_prompt_caching=enable_prompt_caching,
    )

    if provider == "google":
        from novainsight.llm.providers.gemini import GeminiProvider
        logger.info(f"Initializing Gemini provider ({model_name})")
        return GeminiProvider(**kwargs)

    if provider == "anthropic":
        from novainsight.llm.providers.anthropic import AnthropicProvider
        logger.info(f"Initializing Anthropic provider ({model_name})")
        return AnthropicProvider(**kwargs)

    from novainsight.llm.providers.openai import OpenAIProvider
    logger.info(f"Initializing OpenAI provider ({model_name})")
    return OpenAIProvider(**kwargs)
