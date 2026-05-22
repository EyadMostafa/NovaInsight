"""
anthropic.py

Anthropic Claude provider.

Prompt caching: system messages marked with cache=True receive a
cache_control = {"type": "ephemeral"} block, activating Anthropic's
server-side prompt cache (5-minute TTL). This cuts cost and latency
significantly when the static system prompt is re-sent across repeated runs.
"""
from __future__ import annotations

from sleuth.utils.logger import get_logger
from typing import Any

import anthropic as _anthropic

from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.providers.base import LLMProvider, PromptMessage

logger = get_logger("sleuth.llm.providers.anthropic")


class AnthropicProvider(LLMProvider):
    """Anthropic Claude provider with explicit prompt caching support."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        enable_prompt_caching: bool = False,
    ) -> None:
        try:
            self._lib = _anthropic
        except ImportError as e:
            raise LLMSummarizerError(
                "The 'anthropic' package is required for this provider. "
                "Install it with: pip install 'Sleuth[anthropic]'"
            ) from e

        self.client = self._lib.Anthropic(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching

        if enable_prompt_caching:
            logger.info(
                "Anthropic prompt caching enabled. System messages marked cache=True "
                "will receive cache_control = ephemeral (5-minute TTL)."
            )

    def generate(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None = None,  # Anthropic doesn't accept a schema arg; the parse layer handles validation.
    ) -> str:
        system_blocks: list[dict] = []
        for m in messages:
            if m.role == "system":
                block: dict = {"type": "text", "text": m.content}
                if self.enable_prompt_caching and m.cache:
                    block["cache_control"] = {"type": "ephemeral"}
                system_blocks.append(block)

        conversation = self._build_conversation(messages)

        create_kwargs: dict = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "messages": conversation,
            "temperature": self.temperature,
        }
        if system_blocks:
            create_kwargs["system"] = system_blocks

        try:
            response = self.client.messages.create(**create_kwargs)
            return response.content[0].text
        except Exception as e:
            raise LLMSummarizerError(f"Anthropic API call failed: {e}") from e

    @staticmethod
    def _build_conversation(messages: list[PromptMessage]) -> list[dict]:
        """
        Build Anthropic conversation turns from PromptMessages.
        Anthropic requires strictly alternating user/assistant turns, so consecutive
        messages with the same role are merged into a single turn.
        """
        turns: list[dict] = []
        for m in messages:
            if m.role not in ("user", "assistant"):
                continue
            if turns and turns[-1]["role"] == m.role:
                turns[-1]["content"] += "\n\n" + m.content
            else:
                turns.append({"role": m.role, "content": m.content})
        return turns
