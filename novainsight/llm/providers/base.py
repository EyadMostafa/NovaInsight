"""
base.py

Defines the PromptMessage dataclass and the LLMProvider abstract base class.
All provider implementations must satisfy this contract.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class PromptMessage:
    """A single turn in a multi-message prompt."""
    role: Literal["system", "user", "assistant"]
    content: str
    # When True, the provider should mark this block for prompt caching where supported.
    # Anthropic: adds cache_control = {"type": "ephemeral"} to the content block.
    # OpenAI: automatic for the first 1024+ tokens — no explicit flag needed.
    # Gemini: noted in logs; context caching requires a separate API workflow.
    cache: bool = field(default=False)


class LLMProvider(ABC):
    """Abstract base class every LLM provider implementation must satisfy."""

    @abstractmethod
    def generate(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None = None,
    ) -> str:
        """
        Send an ordered list of prompt messages to the underlying model.

        Args:
            messages: System → user (→ assistant) turns in conversation order.
            response_schema: Optional Pydantic model for structured JSON output.
                             Providers that do not support schema enforcement ignore this.

        Returns:
            The model's raw text response.

        Raises:
            LLMSummarizerError: On any API or network failure.
        """
