"""
config.py

This module provides a robust, hierarchical configuration system for NovaInsight
using Pydantic for validation, type casting, and default management.

The loading priority is as follows (each level overrides the one below it):
1. Environment Variables (from .env or system) - Highest Priority
2. config.yaml - Project-specific, shared settings
3. Pydantic Model Defaults - Hardcoded, safe fallbacks
"""
from __future__ import annotations

import os
import yaml
from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
from dotenv import load_dotenv, find_dotenv
import logging 
import warnings
from typing import Literal, Union

load_dotenv(find_dotenv(usecwd=True), override=False)
logger = logging.getLogger(__name__)

# --- HELPER FUNCTIONS ---

def get_project_root() -> Path:
    """Returns the absolute path to the project's root directory."""
    return Path(__file__).parent.parent

def deep_merge(source: dict, destination: dict) -> dict:
    """Recursively merges a source dict into a destination dict."""
    for key, value in source.items():
        if isinstance(value, dict) and key in destination and isinstance(destination[key], dict):
            destination[key] = deep_merge(value, destination[key])
        else:
            destination[key] = value
    return destination

# --- PYDANTIC MODELS: THE SINGLE SOURCE OF TRUTH FOR DEFAULTS ---

class GeneralSettings(BaseModel):
    debug: bool = Field(False, description="Enable verbose debug logging.")
    log_level: str = Field("INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR).")
    supress_user_warnings: bool = Field(True, description="Globally supresses all user warnings.")

class OutputSettings(BaseModel):
    default_directory: Path = Field(Path("."), description="Default parent directory for analysis reports.")

    class Config:
        arbitrary_types_allowed = True # Allow Path type

class AnalysisSettings(BaseModel):
    default_mode: str = Field("full", description="Default mode: 'fast' or 'full'.")
    fast_mode_sample_rows: int = Field(5000, description="Number of rows to sample in fast mode.")
    output: OutputSettings = Field(default_factory=OutputSettings)

class CacheSettings(BaseModel):
    enabled: bool = Field(True, description="Enables or disables the pipeline resumption feature.")
    directory_path: Path = Field(Path("~/.novainsight_cache"), description="Root directory for storing all cached analysis workspaces.")

    class Config:
        arbitrary_types_allowed = True # Allow Path type

class ProfilerSettings(BaseModel):
    duplicate_threshold: float = Field(0.10)
    dimensionality_threshold: float = Field(0.5)
    total_memory_threshold: float = Field(500)
    missing_value_threshold: float = Field(0.20)
    high_cardinality_threshold: float = Field(0.90)
    skewness_threshold: float = Field(1.0)
    memory_hog_threshold: float = Field(0.20)
    max_categorical_cardinality: int = Field(50)

class TargetDetectionSettings(BaseModel):
    max_categorical_cardinality: int = Field(50)
    id_uniqueness_threshold: float = Field(0.99)

class StatisticsSettings(BaseModel):
    outlier_zscore_threshold: float = Field(3.0)
    multicollinearity_vif_threshold: float = Field(10.0)
    outlier_warning_threshold: float = Field(0.05)
    outlier_detection_method: Literal['mz-score', 'z-score'] = Field('mz-score')
    spearman_correlation_threshold: float = Field(0.85)
    cramers_v_correlation_threshold: float = Field(0.5)
    correlation_ratio_threshold: float = Field(0.6)
    spearman_leakage_threshold: float = Field(0.98)
    cramers_v_leakage_threshold: float = Field(0.95)
    correlation_ratio_leakage_threshold: float = Field(0.98)
    class_imbalance_threshold: float = Field(6.0)
    p_value_threshold: float = Field(0.05)
    bivariate_significance_level: float = Field(0.05)

class DimensionalityReduction(BaseModel):
    imputation_method: Literal['median', 'mean'] = Field('median')
    pca_n_ratios: int = Field(10)
    tsne_perplexity: float = Field(30.0)
    tsne_max_iter: int = Field(1000)
    tsne_learning_rate: Union[Literal['auto'], float] = Field('auto')
    umap_n_neighbors: int = Field(15)
    umap_min_dist: float = Field(0.1)
    umap_metric: str = Field('euclidean')

class VisualizationSettings(BaseModel):
    dpi: int = Field(300)
    theme: str = Field("whitegrid")
    color_palette: str = Field("mako")
    kmeans_n_clusters: int = Field(4)
    univariate_countplot_n: int = Field(10)
    bivariate_top_n: int = Field(-1)

class LLMSettings(BaseModel):
    provider: str = Field("google")
    model_name: str = Field("models/gemini-flash-latest")
    temperature: float = Field(0.3)
    max_tokens: int = Field(8192)
    api_key: str | None = None

class NovaInsightConfig(BaseModel):
    """The main configuration object, bringing all settings together."""
    general: GeneralSettings = Field(default_factory=GeneralSettings)
    analysis: AnalysisSettings = Field(default_factory=AnalysisSettings)
    cache: CacheSettings = Field(default_factory=CacheSettings)
    profiler: ProfilerSettings = Field(default_factory=ProfilerSettings)
    target_detection: TargetDetectionSettings = Field(default_factory=TargetDetectionSettings)
    statistics: StatisticsSettings = Field(default_factory=StatisticsSettings)
    dimensionality_reduction: DimensionalityReduction = Field(default_factory=DimensionalityReduction)
    visualization: VisualizationSettings = Field(default_factory=VisualizationSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

# --- LOGGING SETUP FUNCTION ---

def setup_logging(config: NovaInsightConfig):
    """
    Configures the root logger based on the loaded application settings.
    This function should be called once at application startup.
    """
    root_logger = logging.getLogger()
    log_level_str = config.general.log_level.upper()
    log_level = getattr(logging, log_level_str, logging.INFO)
    root_logger.setLevel(log_level)

   
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(asctime)s] [%(levelname)-8s] [%(name)s] --- %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
    
    logger.debug(f"Root logger configured to level: {log_level_str}")

# --- THE UNIFIED LOADER FUNCTION ---

def load_config(path: str | Path | None = None) -> NovaInsightConfig:
    """
    Loads configuration from YAML and environment variables, providing defaults for missing values.
    """
    config_path = Path(path) if path else get_project_root() / "config/config.yaml"
    final_config_dict = NovaInsightConfig().model_dump()
    # A temporary, basic logger to catch early messages logged before logger initialization via setup_logging.
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] [%(name)s] %(message)s")

    try:
        with open(config_path, 'r') as f:
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

    # Load environment variables (highest priority)
    def clean_env_value(v):
        return v if v not in ("", "None") else None
    env_vars = {
        "general": {
            "debug": clean_env_value(os.getenv("NOVA_INSIGHT_DEBUG")),
            "log_level": clean_env_value(os.getenv("NOVA_INSIGHT_LOG_LEVEL")),
            "supress_user_warnings": clean_env_value(os.getenv("NOVA_INSIGHT_SUPRESS_USER_WARNINGS"))
        },
        "analysis": {
            "output": {
                "default_directory": clean_env_value(os.getenv("NOVA_INSIGHT_ANALYSIS_OUTPUT_DEFAULT_DIRECTORY"))
            }
        },
        "cache": {
            "enabled": clean_env_value(os.getenv("NOVA_INSIGHT_CACHE_ENABLED")),
            "directory_path": clean_env_value(os.getenv("NOVA_INSIGHT_CACHE_DIRECTORY_PATH")),
        },
        "llm": {
            "api_key": clean_env_value(os.getenv("NOVA_INSIGHT_LLM_API_KEY"))
        }
    }
    # Clean up None values so we only override with set env vars
    def clean_nested_dict(d):
        """
        Recursively removes keys where the value is None.
        Also removes empty dictionaries that result from this cleaning.
        """
        if not isinstance(d, dict):
            return d
            
        clean_d = {}
        for k, v in d.items():
            if isinstance(v, dict):
                nested_clean = clean_nested_dict(v)
                # Only keep the nested dict if it's not empty
                if nested_clean:
                    clean_d[k] = nested_clean
            elif v is not None:
                clean_d[k] = v
        return clean_d
    
    cleaned_env_vars = clean_nested_dict(env_vars)
    cleaned_env_vars.update({k: v for k, v in env_vars.items() if not isinstance(v, dict) and v is not None})
    
    final_config_dict = deep_merge(cleaned_env_vars, final_config_dict)

    try:
        config = NovaInsightConfig(**final_config_dict)
        setup_logging(config)
        if config.general.supress_user_warnings:
            warnings.filterwarnings("ignore", category=UserWarning)
        return config
    except ValidationError as e:
        config = NovaInsightConfig()
        setup_logging(config)
        logger.error(f"Configuration validation error: {e}. Falling back to default settings.")
        return config