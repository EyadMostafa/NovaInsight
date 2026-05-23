"""
config.py

Hierarchical configuration system for Sleuth.

Loading priority (highest wins):
1. Environment variables (from .env / system)
2. config.yaml (or config.dev.yaml in dev env)
3. Pydantic model defaults
"""

from __future__ import annotations

import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal

import yaml
from dotenv import find_dotenv, load_dotenv
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from sleuth.exceptions import ConfigError
from sleuth.utils.logger import ColorFormatter, get_logger

# --- ENVIRONMENT DETECTION ---

PROJECT_ENV_PREFIX = "SLEUTH_"

_SLEUTH_ENV = os.getenv("SLEUTH_ENV", "prod").lower()
if _SLEUTH_ENV == "dev":
    _ENV_FILENAME = ".env.dev"
    _YAML_FILENAME = "config.dev.yaml"
else:
    _ENV_FILENAME = ".env"
    _YAML_FILENAME = "config.yaml"

load_dotenv(find_dotenv(_ENV_FILENAME, usecwd=True), override=False)

logger = get_logger("sleuth.config.config")

# --- HELPERS ---


def get_project_root() -> Path:
    return Path(__file__).parent.parent


def deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merges source into destination; source values win on conflicts."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination


def clean_env_value(v: str | None) -> str | None:
    """Treats empty string and the literal 'None' as absent."""
    return None if v in (None, "", "None") else v


def clean_nested_dict(d: Any) -> Any:
    """Recursively removes None values and the empty dicts that result."""
    if not isinstance(d, dict):
        return d
    result: dict = {}
    for k, v in d.items():
        if isinstance(v, dict):
            cleaned = clean_nested_dict(v)
            if cleaned:
                result[k] = cleaned
        elif v is not None:
            result[k] = v
    return result


# --- PYDANTIC MODELS ---
class BaseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)


class GeneralSettings(BaseSettings):
    debug: bool = Field(False)
    log_level: str = Field("INFO")
    suppress_user_warnings: bool = Field(True)


class OutputSettings(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    default_directory: Path = Field(Path("."))


class AnalysisSettings(BaseSettings):
    default_mode: str = Field("full")
    fast_mode_sample_rows: int = Field(5000)
    output: OutputSettings = Field(default_factory=OutputSettings)


class CacheSettings(BaseSettings):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)
    enabled: bool = Field(True)
    directory_path: Path = Field(Path("~/.Sleuth_cache"))


class ProfilerSettings(BaseSettings):
    duplicate_threshold: float = Field(0.10)
    dimensionality_threshold: float = Field(0.5)
    total_memory_threshold: float = Field(500)
    missing_value_threshold: float = Field(0.20)
    high_cardinality_threshold: float = Field(0.90)
    skewness_threshold: float = Field(1.0)
    memory_hog_threshold: float = Field(0.20)
    max_categorical_cardinality: int = Field(50)


class TargetDetectionSettings(BaseSettings):
    max_categorical_cardinality: int = Field(50)
    id_uniqueness_threshold: float = Field(0.99)


class StatisticsSettings(BaseSettings):
    outlier_zscore_threshold: float = Field(3.0)
    multicollinearity_vif_threshold: float = Field(10.0)
    outlier_warning_threshold: float = Field(0.05)
    outlier_detection_method: Literal["mz-score", "z-score"] = Field("mz-score")
    spearman_correlation_threshold: float = Field(0.85)
    cramers_v_correlation_threshold: float = Field(0.5)
    correlation_ratio_threshold: float = Field(0.6)
    spearman_leakage_threshold: float = Field(0.98)
    cramers_v_leakage_threshold: float = Field(0.95)
    correlation_ratio_leakage_threshold: float = Field(0.98)
    class_imbalance_threshold: float = Field(6.0)
    p_value_threshold: float = Field(0.05)
    bivariate_significance_level: float = Field(0.05)


class DimensionalityReduction(BaseSettings):
    imputation_method: Literal["median", "mean"] = Field("median")
    pca_n_ratios: int = Field(10)
    tsne_perplexity: float = Field(30.0)
    tsne_max_iter: int = Field(1000)
    tsne_learning_rate: Literal["auto"] | float = Field("auto")
    umap_n_neighbors: int = Field(15)
    umap_min_dist: float = Field(0.1)
    umap_metric: str = Field("euclidean")


class VisualizationSettings(BaseSettings):
    dpi: int = Field(300)
    theme: str = Field("whitegrid")
    color_palette: str = Field("mako")
    kmeans_n_clusters: int = Field(4)
    univariate_countplot_n: int = Field(10)
    bivariate_top_n: int = Field(-1)


class LLMSettings(BaseSettings):
    provider: str = Field("google")
    model_name: str = Field("models/gemini-flash-latest")
    temperature: float = Field(0.3)
    max_tokens: int = Field(8192)
    api_key: SecretStr | None = None
    enable_prompt_caching: bool = Field(False)
    base_url: str | None = None


class SleuthConfig(BaseSettings):
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    profiler: ProfilerSettings = Field(default_factory=ProfilerSettings)
    target_detection: TargetDetectionSettings = Field(default_factory=TargetDetectionSettings)
    statistics: StatisticsSettings = Field(default_factory=StatisticsSettings)
    dimensionality_reduction: DimensionalityReduction = Field(
        default_factory=DimensionalityReduction
    )
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)


# --- LOGGING SETUP ---


def setup_logging(cfg: SleuthConfig) -> None:
    root_logger = logging.getLogger()
    log_level = getattr(logging, cfg.general.log_level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            ColorFormatter(
                "[%(asctime)s] [%(levelname)s] [%(name)s] --- %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        root_logger.addHandler(handler)


# --- ENVIRONMENT VARIABLE MAPPING ---


def load_environment_variables() -> dict[str, Any]:
    def get(key: str) -> str | None:
        return clean_env_value(os.getenv(f"{PROJECT_ENV_PREFIX}{key}"))

    mapping: dict[str, Any] = {
        "general": {
            "debug": get("DEBUG"),
            "log_level": get("LOG_LEVEL"),
            "suppress_user_warnings": get("SUPPRESS_USER_WARNINGS"),
        },
        "analysis": {
            "default_mode": get("ANALYSIS_DEFAULT_MODE"),
            "fast_mode_sample_rows": get("ANALYSIS_FAST_MODE_SAMPLE_ROWS"),
            "output": {"default_directory": get("OUTPUT_DEFAULT_DIRECTORY")},
        },
        "cache": {
            "enabled": get("CACHE_ENABLED"),
            "directory_path": get("CACHE_DIRECTORY_PATH"),
        },
        "llm": {
            "provider": get("LLM_PROVIDER"),
            "model_name": get("LLM_MODEL_NAME"),
            "api_key": get("LLM_API_KEY"),
            "enable_prompt_caching": get("LLM_ENABLE_PROMPT_CACHING"),
            "base_url": get("LLM_BASE_URL"),
        },
    }
    return clean_nested_dict(mapping)


# --- LOADER ---


def load_config(path: str | Path | None = None) -> SleuthConfig:
    config_path = Path(path) if path else get_project_root() / f"config/{_YAML_FILENAME}"
    final_config_dict: dict[str, Any] = {}

    try:
        with open(config_path, encoding="utf-8") as f:
            yaml_config = yaml.safe_load(f)
        if isinstance(yaml_config, dict):
            final_config_dict = deep_merge(yaml_config, final_config_dict)
            logger.info(f"Loaded configuration from {config_path}")
        else:
            logger.warning(f"Config file at {config_path} is malformed. Using defaults.")
    except FileNotFoundError:
        logger.info(f"Config file not found at {config_path}. Using defaults.")
    except Exception as e:
        logger.error(f"Error reading YAML config: {e}. Using defaults.")

    env_vars = load_environment_variables()
    final_config_dict = deep_merge(env_vars, final_config_dict)

    try:
        loaded_config = SleuthConfig(**final_config_dict)
        setup_logging(loaded_config)
        if loaded_config.general.suppress_user_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
        return loaded_config
    except ValidationError as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e


config = load_config()
