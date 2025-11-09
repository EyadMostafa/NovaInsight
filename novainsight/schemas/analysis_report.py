"""
schemas.py

This file defines the complete Pydantic data model for the AnalysisReport.
This structured object is the central "source of truth" that is passed through
the analysis pipeline, with each module enriching its designated section.
"""
from __future__ import annotations

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from datetime import datetime, timezone
from enum import Enum, auto

# ===================================================================
# Section 1: Metadata & Core Profiling Schemas
# ===================================================================

class Operator(str, Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name.lower()

    PROFILER = auto()
    TARGET = auto()
    STATS = auto()
    DIM_REDUCTION = auto()
    VIZ = auto()
    LLM = auto()
    RECOMMENDATIONS = auto()
    REPORT = auto()

class Finding(BaseModel):
    level: Literal['INFO', 'WARNING', 'ERROR'] = Field(..., description="Type/Severity of the finding")
    message: str = Field(..., description="Message describing the finding")

class RunMetadata(BaseModel):
    """Contains metadata about the specific analysis run."""
    input_file: str | Path = Field(..., description="The path to the input dataset.")
    output_dir: str | Path = Field(..., description="The path to the output directory.")
    file_hash: str = Field(..., description="The SHA256 hash of the input file content.")
    report_title: str = Field(..., description="The title for the generated report.")
    run_timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), description="The ISO 8601 timestamp when the analysis was started.")
    analysis_mode: Literal['full', 'fast'] = Field(..., description="The mode of the analysis ('full' or 'fast').")
    task: Literal['supervised', 'unsupervised'] = Field(..., description="The task to be performed using the dataset (supervised or unsupervised)")
    user_target: Optional[str] = Field(None, description="The target provided by the user if any.")

class DatasetStats(BaseModel):
    """Holds overall summary statistics for the entire dataset."""
    original_row_count: int = Field(..., description="Total row count of the entire dataset regardless of analysis mode")
    row_count: int = Field(..., description="Represents the count of (observations/instances) in sampled dataset incase of 'fast' analysis mode, otherwise it is just the full row count.")
    column_count: int = Field(..., description="Total number of columns (features) in the dataset.")
    total_memory_usage_mb: float = Field(..., description="The total memory usage of the entire dataset in megabytes (MB).")
    duplicate_rows_count: int = Field(..., description="Total count of duplicate rows.")
    column_pct: float = Field(..., description="Percentage of features with respect to instances")
    duplicates_pct: float = Field(..., description="Percenteage of duplicates")

class ColumnDetails(BaseModel):
    """A detailed, structured profile for a single column."""
    column_name: str = Field(..., description="The name of the column.")
    inferred_type: str = Field(..., description="The high-level inferred type (e.g., 'numeric', 'categorical', 'datetime', 'text', 'id').")
    dtype: str = Field(..., description="The actual pandas dtype (e.g., 'int64', 'float64', 'object').")
    missing_values_count: int = Field(..., description="The absolute count of missing or null (NaN) values.")
    missing_values_pct: float = Field(..., description="The percentage of values in the column that are missing.")
    unique_values_count: int = Field(..., description="The number of distinct, non-null values in the column.")
    unique_values_pct: float = Field(..., description="The percentage of values in the column that are unique.")
    stats: Dict[str, Any] = Field(..., description="Descriptive statistics (mean, std, etc.) or value counts.")
    memory_usage_mb: float = Field(..., description="The memory usage of the column in megabytes (MB).")

class DatasetProfile(BaseModel):
    """The complete output from the Data Ingestion & Profiling module."""
    dataset_stats: DatasetStats
    column_details: List[ColumnDetails]
    findings: Optional[List[Finding]] = Field([], description="A list to hold findings from any module-level failures, warnings, or info")

# ===================================================================
# Section 2: Target Variable Analysis Schemas
# ===================================================================

class CandidateTarget(BaseModel):
    """Describes a column that is a potential target for a supervised ML model."""
    column_name: str
    ml_task: str = Field(..., description="The suggested machine learning task ('classification' or 'regression').")
    justification: str = Field(..., description="A human-readable reason why this column is a good candidate.")
    confidence_score: float = Field(..., description="A score from 0.0 to 1.0 indicating confidence.")

class TargetVariableAnalysis(BaseModel):
    """The complete output from the Target Variable Detection module."""
    detection_method: Literal['auto', 'user_specified'] = Field(..., description="How the target was chosen ('auto' or 'user_specified').")
    candidate_targets: List[CandidateTarget]
    identified_target: Optional[str] = Field(None, description="The column name chosen as the most likely target.")
    findings: List[Finding] = Field(default_factory=list, description="A list to hold findings from any module-level failures, warnings, or info")

# ===================================================================
# Section 3: Statistical & Structural Analysis Schemas
# ===================================================================

class OutlierReport(BaseModel):
    """A summary of outliers found in a numeric column."""
    outlier_count: int
    outlier_percentage: float
    method: str = Field(..., description="The method used for detection (e.g., 'Modified Z-score', 'Z-score').")

class ClassImbalanceReport(BaseModel):
    """A summary of the class distribution for a classification target."""
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]

class CorrelationReport(BaseModel):
    """Holds paths to the various correlation matrix data files."""
    numerical_numerical_path: Optional[str | Path] = Field(None, description="Path to the Spearman correlation matrix for numeric pairs.")
    categorical_categorical_path: Optional[str | Path] = Field(None, description="Path to the Cramér's V association matrix for categorical pairs.")
    categorical_numerical_path: Optional[str | Path] = Field(None, description="Path to the Correlation Ratio association matrix for categorical-numeric pairs.")

class StatisticalAnalysis(BaseModel):
    """The complete output from the Statistical & Structural Insights module."""
    outlier_report: Dict[str, OutlierReport] = Field(..., description="A mapping of column names to their outlier reports.")
    multicollinearity_report: Dict[str, float] = Field(..., description="A mapping of highly collinear features to their VIF scores.")
    correlation_report: CorrelationReport = Field(...)
    class_imbalance_report: Optional[ClassImbalanceReport] = Field(None)
    findings: List[Finding] = Field(default_factory=list, description="A list to hold findings from any module-level failures, warnings, or info")

# ===================================================================
# Section 4: Advanced Analysis & Visualization Schemas
# ===================================================================

class DimensionalityAnalysis(BaseModel):
    """The complete output from the Dimensionality Reduction module."""
    pca_explained_variance_ratios: Optional[List[float]] = Field([])
    pca_embeddings_path: Optional[str | Path] = Field("")
    tsne_embeddings_path: Optional[str | Path] = Field("")
    umap_embeddings_path: Optional[str | Path] = Field("")
    findings: List[Finding] = Field(default_factory=list, description="A list to hold findings from any module-level failures, warnings, or info")

class VisualizationOutput(BaseModel):
    """A collection of file paths to all generated plot images."""
    univariate_plots: Dict[str, str | Path] = Field(..., description="Mapping of column names to their distribution plot paths.")
    bivariate_plots: Dict[str, str | Path] = Field(..., description="Mapping of plot types (e.g., 'target_vs_feature_x') to their paths.")
    correlation_heatmap_path: Optional[str | Path] = Field("")

# ===================================================================
# Section 5: LLM Summaries & Final Recommendations Schemas
# ===================================================================

class LLMSummary(BaseModel):
    """Holds the natural language summaries generated by the LLM."""
    executive_summary: str
    dataset_overview: str
    # consider replacing the following 2 params with a single findings list
    key_findings_and_patterns: str
    potential_issues_and_warnings: str

class Recommendation(BaseModel):
    """A single, actionable recommendation for the user."""
    category: str = Field(..., description="The area of recommendation (e.g., 'Preprocessing', 'Feature Engineering', 'Modeling').")
    description: str
    priority: int = Field(..., description="Priority from 1 (High) to 3 (Low).")

class Recommendations(BaseModel):
    """A structured and prioritized list of all recommendations."""
    preprocessing_steps: List[Recommendation]
    feature_engineering_ideas: List[Recommendation]
    modeling_suggestions: List[Recommendation]
    pitfall_warnings: List[Recommendation]

# ===================================================================
# The Top-Level Schema: AnalysisReport
# This is the master object that contains all other schemas.
# ===================================================================

class AnalysisReport(BaseModel):
    """
    The central data structure holding all findings from the NovaInsight pipeline.
    This object is initialized by the orchestrator and progressively enriched by each module.
    """
    metadata: RunMetadata
    profile: Optional[DatasetProfile] = None
    target_analysis: Optional[TargetVariableAnalysis] = None
    statistical_analysis: Optional[StatisticalAnalysis] = None
    dimensionality_analysis: Optional[DimensionalityAnalysis] = None
    visualizations: Optional[VisualizationOutput] = None
    llm_summary: Optional[LLMSummary] = None
    recommendations: Optional[Recommendations] = None
    findings: List[Finding] = Field(default_factory=list, description="A list to hold findings from any module-level failures, warnings, or info")

    model_config = {
        "arbitrary_types_allowed": True,
        "frozen": False
    }

