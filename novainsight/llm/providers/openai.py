"""
openai.py

OpenAI ChatGPT provider.

Prompt caching: OpenAI automatically caches the first 1024+ tokens of a prompt
for gpt-4o and later models. No opt-in is required — placing the static system
prompt first and keeping it unchanged across calls achieves cache hits transparently.
"""
from __future__ import annotations

from novainsight.utils.logger import get_logger
from typing import Any

from openai import OpenAI as _OpenAI

from novainsight.exceptions import LLMSummarizerError
from novainsight.llm.providers.base import LLMProvider, PromptMessage

logger = get_logger("novainsight.llm.providers.openai")


class OpenAIProvider(LLMProvider):
    """OpenAI ChatCompletion provider. Prompt caching is automatic for gpt-4o+."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        enable_prompt_caching: bool = False,
    ) -> None:
        try:
            self._OpenAI = _OpenAI
        except ImportError as e:
            raise LLMSummarizerError(
                "The 'openai' package is required for this provider. "
                "Install it with: pip install 'novainsight[openai]'"
            ) from e

        self.client = self._OpenAI(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        if enable_prompt_caching:
            logger.info(
                "OpenAI prompt caching is automatic for the first 1024+ tokens. "
                "Keep the system message static across calls to maximise cache hits."
            )

    def generate(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None = None,
    ) -> str:
        openai_messages = [{"role": m.role, "content": m.content} for m in messages]

        kwargs: dict = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if response_schema is not None:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = self.client.chat.completions.create(**kwargs)
            return response.choices[0].message.content
        except Exception as e:
            raise LLMSummarizerError(f"OpenAI API call failed: {e}") from e
