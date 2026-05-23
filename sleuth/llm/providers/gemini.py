"""
gemini.py

Google Gemini provider using the google.genai SDK (v2+).

Context caching behaviour
─────────────────────────
When enable_prompt_caching=True the provider tries to create a Gemini
context cache from every PromptMessage that has cache=True (typically the
static system prompt).

Cache lifecycle:
  - Created on the first generate() call that has cacheable messages.
  - Stored as self._cache_name and reused for subsequent calls that also
    have cacheable messages (e.g. a correction retry without cacheable
    messages falls back to standard generation, avoiding context conflicts).
  - Deleted at the end of the conversation via __del__; also expires via TTL.

Graceful fallback:
  If cache creation fails (payload below the model's minimum token floor,
  network error, etc.) the provider logs a warning and continues with
  standard generation — no exception is surfaced to the caller.
"""
from __future__ import annotations

from typing import Any

from google import genai
from google.genai import types

from sleuth.exceptions import LLMSummarizerError
from sleuth.llm.providers.base import LLMProvider, PromptMessage
from sleuth.utils.logger import get_logger

logger = get_logger("sleuth.llm.providers.gemini")

_CACHE_TTL = "300s"  # 5 minutes — matches Anthropic's ephemeral TTL


class GeminiProvider(LLMProvider):
    """Google Gemini provider with optional server-side context caching."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        enable_prompt_caching: bool = False,
    ) -> None:
        self._client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_prompt_caching = enable_prompt_caching
        self._cache_name: str | None = None

        if enable_prompt_caching:
            logger.info(
                "Gemini context caching enabled. A cache will be created on the first "
                "call with cacheable messages and reused for the session. Cache creation "
                "will fall back silently if the payload is below the model's token floor."
            )

    # ------------------------------------------------------------------
    # LLMProvider protocol
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None = None,
    ) -> str:
        cacheable = [m for m in messages if m.cache]
        non_cacheable = [m for m in messages if not m.cache]

        if self.enable_prompt_caching and cacheable:
            if self._cache_name is None:
                self._try_create_cache(cacheable)
            if self._cache_name:
                try:
                    return self._generate_with_cache(non_cacheable, response_schema)
                except Exception as e:
                    raise LLMSummarizerError(f"Gemini API call failed: {e}") from e

        try:
            return self._generate_standard(messages, response_schema)
        except Exception as e:
            raise LLMSummarizerError(f"Gemini API call failed: {e}") from e

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def _try_create_cache(self, cacheable: list[PromptMessage]) -> None:
        """
        Attempt to create a context cache from cacheable messages.
        System messages become system_instruction; user messages become cached contents.
        Silently falls back on any failure.
        """
        system_text = "\n\n".join(m.content for m in cacheable if m.role == "system")
        user_parts = [m.content for m in cacheable if m.role == "user"]

        cache_contents: list[types.Content] | None = None
        if user_parts:
            cache_contents = [
                types.Content(
                    role="user",
                    parts=[types.Part(text="\n\n".join(user_parts))],
                )
            ]

        try:
            cache = self._client.caches.create(
                model=self.model_name,
                config=types.CreateCachedContentConfig(
                    system_instruction=system_text or None,
                    contents=cache_contents,
                    ttl=_CACHE_TTL,
                ),
            )
            self._cache_name = cache.name
            logger.info(f"Gemini context cache created: {cache.name}")
        except Exception as e:
            logger.warning(
                "Gemini context cache creation failed — payload is likely below the "
                f"model's minimum token floor. Falling back to standard generation. Reason: {e}"
            )

    def _delete_cache(self) -> None:
        cache_name = getattr(self, "_cache_name", None)
        if cache_name:
            try:
                self._client.caches.delete(name=cache_name)
                logger.debug(f"Gemini context cache deleted: {cache_name}")
            except Exception:
                pass  # TTL will expire it regardless
            self._cache_name = None

    def __del__(self) -> None:
        self._delete_cache()

    # ------------------------------------------------------------------
    # Generation paths
    # ------------------------------------------------------------------

    def _generate_with_cache(
        self,
        non_cacheable: list[PromptMessage],
        response_schema: Any | None,
    ) -> str:
        """Use the pre-created context cache; send only non-cached content as new turns."""
        user_text = "\n\n".join(m.content for m in non_cacheable if m.role == "user")

        config_kwargs: dict = {
            "cached_content": self._cache_name,
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_text or "Continue.",
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text

    def _generate_standard(
        self,
        messages: list[PromptMessage],
        response_schema: Any | None,
    ) -> str:
        """Standard generation — system_instruction + user contents, no cache."""
        system_text = "\n\n".join(m.content for m in messages if m.role == "system")
        user_text = "\n\n".join(m.content for m in messages if m.role == "user")

        config_kwargs: dict = {
            "max_output_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text
        if response_schema is not None:
            config_kwargs["response_mime_type"] = "application/json"
            config_kwargs["response_schema"] = response_schema

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=user_text,
            config=types.GenerateContentConfig(**config_kwargs),
        )
        return response.text
