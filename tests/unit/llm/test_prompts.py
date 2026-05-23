"""Unit tests for LLM prompt builders."""
from __future__ import annotations

from sleuth.llm.prompts import (
    SYSTEM_PROMPT,
    build_correction_messages,
    build_messages,
)
from sleuth.llm.providers.base import PromptMessage


class TestSystemPrompt:
    def test_system_prompt_is_non_empty(self):
        assert len(SYSTEM_PROMPT) > 100

    def test_system_prompt_contains_json_schema_hint(self):
        assert "executive_summary" in SYSTEM_PROMPT
        assert "recommendations" in SYSTEM_PROMPT

    def test_system_prompt_contains_output_rules(self):
        assert "JSON" in SYSTEM_PROMPT


class TestBuildMessages:
    def test_returns_two_messages(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert len(msgs) == 2

    def test_first_message_is_system(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert msgs[0].role == "system"

    def test_second_message_is_user(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert msgs[1].role == "user"

    def test_system_message_cache_flag_true(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert msgs[0].cache is True

    def test_user_message_cache_flag_false(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert msgs[1].cache is False

    def test_user_content_includes_task(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert "supervised" in msgs[1].content

    def test_fast_mode_adds_notice(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="fast")
        assert "FAST MODE" in msgs[1].content

    def test_full_mode_no_fast_notice(self):
        msgs = build_messages(toon_context="ctx", task="supervised", analysis_mode="full")
        assert "FAST MODE" not in msgs[1].content

    def test_toon_context_in_user_message(self):
        context = "some_encoded_context_data"
        msgs = build_messages(toon_context=context, task="supervised", analysis_mode="full")
        assert context in msgs[1].content


class TestBuildCorrectionMessages:
    def test_returns_two_messages(self):
        msgs = build_correction_messages("bad json")
        assert len(msgs) == 2

    def test_both_messages_not_cached(self):
        msgs = build_correction_messages("bad json")
        assert msgs[0].cache is False
        assert msgs[1].cache is False

    def test_system_role_is_system(self):
        msgs = build_correction_messages("{broken")
        assert msgs[0].role == "system"

    def test_user_contains_bad_json(self):
        bad = '{"key": "missing_close'
        msgs = build_correction_messages(bad)
        assert bad in msgs[1].content

    def test_system_mentions_json_repair(self):
        msgs = build_correction_messages("bad")
        assert "JSON" in msgs[0].content


class TestPromptMessage:
    def test_default_cache_is_false(self):
        msg = PromptMessage(role="user", content="hello")
        assert msg.cache is False

    def test_cache_can_be_set_true(self):
        msg = PromptMessage(role="system", content="sys", cache=True)
        assert msg.cache is True
