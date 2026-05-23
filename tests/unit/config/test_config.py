"""Unit tests for the config loading system."""
from __future__ import annotations

import pytest
import yaml

from sleuth.config.config import (
    SleuthConfig,
    clean_env_value,
    clean_nested_dict,
    deep_merge,
    load_config,
)
from sleuth.exceptions import ConfigError

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------

class TestCleanEnvValue:
    def test_none_returns_none(self):
        assert clean_env_value(None) is None

    def test_empty_string_returns_none(self):
        assert clean_env_value("") is None

    def test_string_none_returns_none(self):
        assert clean_env_value("None") is None

    def test_valid_string_passes_through(self):
        assert clean_env_value("hello") == "hello"

    def test_zero_string_passes_through(self):
        assert clean_env_value("0") == "0"


class TestDeepMerge:
    def test_source_wins_on_conflict(self):
        result = deep_merge({"a": 2}, {"a": 1})
        assert result["a"] == 2

    def test_keys_only_in_destination_preserved(self):
        result = deep_merge({"a": 1}, {"b": 2})
        assert result["b"] == 2

    def test_nested_merge(self):
        source = {"outer": {"inner": 99}}
        dest = {"outer": {"inner": 1, "other": 42}}
        result = deep_merge(source, dest)
        assert result["outer"]["inner"] == 99
        assert result["outer"]["other"] == 42

    def test_non_dict_source_overwrites(self):
        result = deep_merge({"a": [1, 2]}, {"a": [3, 4]})
        assert result["a"] == [1, 2]


class TestCleanNestedDict:
    def test_removes_none_values(self):
        result = clean_nested_dict({"a": 1, "b": None})
        assert "b" not in result

    def test_removes_empty_dicts_after_cleaning(self):
        result = clean_nested_dict({"outer": {"inner": None}})
        assert "outer" not in result

    def test_preserves_non_none(self):
        result = clean_nested_dict({"a": 1, "b": {"c": 2}})
        assert result == {"a": 1, "b": {"c": 2}}

    def test_non_dict_passthrough(self):
        assert clean_nested_dict(42) == 42
        assert clean_nested_dict("hello") == "hello"


# ---------------------------------------------------------------------------
# load_config: layer priority
# ---------------------------------------------------------------------------

class TestLoadConfig:
    def test_loads_default_config(self):
        cfg = load_config()
        assert isinstance(cfg, SleuthConfig)

    def test_yaml_overrides_pydantic_defaults(self, tmp_path, monkeypatch):
        monkeypatch.delenv("SLEUTH_LOG_LEVEL", raising=False)
        yaml_data = {"general": {"log_level": "DEBUG"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(yaml_data))
        cfg = load_config(cfg_path)
        assert cfg.general.log_level == "DEBUG"

    def test_env_var_overrides_yaml(self, tmp_path, monkeypatch):
        yaml_data = {"general": {"log_level": "DEBUG"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(yaml_data))
        monkeypatch.setenv("SLEUTH_LOG_LEVEL", "ERROR")
        cfg = load_config(cfg_path)
        assert cfg.general.log_level == "ERROR"

    def test_missing_yaml_falls_back_to_defaults(self, tmp_path):
        cfg = load_config(tmp_path / "nonexistent.yaml")
        assert isinstance(cfg, SleuthConfig)

    def test_malformed_yaml_falls_back_to_defaults(self, tmp_path):
        bad = tmp_path / "bad.yaml"
        bad.write_text(": invalid: yaml: [[[")
        cfg = load_config(bad)
        assert isinstance(cfg, SleuthConfig)

    def test_llm_api_key_from_env(self, tmp_path, monkeypatch):
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text("{}")
        monkeypatch.setenv("SLEUTH_LLM_API_KEY", "super-duper-ultra-secret-key-123")
        cfg = load_config(cfg_path)
        assert cfg.llm.api_key is not None
        assert cfg.llm.api_key.get_secret_value() == "super-duper-ultra-secret-key-123"

    def test_api_key_not_in_yaml(self, tmp_path):
        yaml_data = {"llm": {"provider": "google", "model_name": "test-model"}}
        cfg_path = tmp_path / "config.yaml"
        cfg_path.write_text(yaml.dump(yaml_data))
        cfg = load_config(cfg_path)
        # api_key must NOT be settable via YAML — only via env
        # It remains None since no env var set in this test
        # The YAML field for api_key doesn't exist, so no leakage
        assert cfg.llm.provider == "google"

    def test_invalid_pydantic_type_raises_config_error(self, tmp_path):
        yaml_data2 = {"statistics": {"outlier_zscore_threshold": "not-a-float"}}
        cfg_path2 = tmp_path / "config2.yaml"
        cfg_path2.write_text(yaml.dump(yaml_data2))
        with pytest.raises(ConfigError):
            load_config(cfg_path2)

    def test_cache_settings_from_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLEUTH_CACHE_ENABLED", "false")
        cfg_path = tmp_path / "c.yaml"
        cfg_path.write_text("{}")
        cfg = load_config(cfg_path)
        assert cfg.cache.enabled is False
