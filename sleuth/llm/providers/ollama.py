"""
ollama.py

Local LLM provider via Ollama (https://ollama.com).

Ollama runs a local REST server (default: http://localhost:11434) and exposes
a Python client that mirrors the chat-completions interface.  No API key is
required — the provider works entirely offline.

Prompt caching: not applicable (local inference, zero token cost).
Structured output: JSON mode is requested via `format="json"` when a
response schema is provided.  Full JSON-schema enforcement requires Ollama
≥0.5 with a model that supports it; older versions return free-form JSON.
The existing json_repair fallback in LLMSummarizer handles both cases.
"""

from __future__ import annotations

from typing import Any

from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.providers.base import LLMProvider, PromptMessage
from sleuth.utils.logger import get_logger

logger = get_logger("sleuth.llm.providers.ollama")

_DEFAULT_HOST = "http://localhost:11434"


class OllamaProvider(LLMProvider):
    """Local LLM provider via the Ollama server."""

    def __init__(
        self,
        model_name: str,
        temperature: float,
        max_tokens: int,
        base_url: str | None = None,
        enable_prompt_caching: bool = False,
        **_kwargs: Any,
    ) -> None:
        try:
            import ollama as _ollama
        except ImportError as e:
            raise LLMSummarizerError(
                "The 'ollama' package is required for this provider. "
                "Install it with: pip install 'sleuth[ollama]'"
            ) from e

        host = base_url or _DEFAULT_HOST
        self.client = _ollama.Client(host=host)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        if enable_prompt_caching:
            logger.debug("Ollama runs locally — prompt caching is not applicable.")

        logger.info(f"Ollama provider initialised (model={model_name}, host={host})")

    def generate(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None = None,
    ) -> str:
        ollama_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "messages": ollama_messages,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        if response_schema is not None:
            kwargs["format"] = "json"

        try:
            response = self.client.chat(**kwargs)
            return response.message.content
        except Exception as e:
            raise LLMSummarizerError(
                f"Ollama request failed: {e}. "
                f"Make sure 'ollama serve' is running and the model "
                f"'{self.model_name}' has been pulled ('ollama pull {self.model_name}')."
            ) from e
