"""
schemas.py

This file defines the complete Pydantic data model for the AnalysisReport.
This structured object is the central "source of truth" that is passed through
the analysis pipeline, with each module enriching its designated section.
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# ===================================================================
# Section 1: Metadata & Core Profiling Schemas
# ===================================================================

class RunMetadata(BaseModel):
    """Contains metadata about the specific analysis run."""
    input_file: str = Field(..., description="The path to the input dataset.")
    file_hash: str = Field(..., description="The SHA256 hash of the input file content.")
    report_title: str = Field(..., description="The title for the generated report.")
    run_timestamp: str = Field(..., description="The ISO 8601 timestamp when the analysis was started.")
    analysis_mode: str = Field(..., description="The mode of the analysis ('fast' or 'full').")

class DatasetStats(BaseModel):
    """Holds overall summary statistics for the entire dataset."""
    row_count: int
    column_count: int
    memory_usage_mb: float
    duplicate_rows: int

class ColumnDetails(BaseModel):
    """A detailed, structured profile for a single column."""
    column_name: str
    inferred_type: str = Field(..., description="The high-level inferred type (e.g., 'numeric', 'categorical', 'datetime', 'text', 'id').")
    dtype: str = Field(..., description="The actual pandas dtype (e.g., 'int64', 'float64', 'object').")
    missing_values: int
    missing_values_pct: float
    unique_values: int
    unique_values_pct: float
    stats: Dict[str, Any] = Field(..., description="Descriptive statistics (mean, std, etc.) or value counts.")

class DatasetProfile(BaseModel):
    """The complete output from the Data Ingestion & Profiling module."""
    dataset_stats: DatasetStats
    column_details: List[ColumnDetails]
    warnings: List[str] = []

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
    detection_method: str = Field(..., description="How the target was chosen ('auto' or 'user_specified').")
    candidate_targets: List[CandidateTarget]
    identified_target: Optional[str] = Field(None, description="The column name chosen as the most likely target.")
    leakage_warnings: List[str] = []

# ===================================================================
# Section 3: Statistical & Structural Analysis Schemas
# ===================================================================

class OutlierReport(BaseModel):
    """A summary of outliers found in a numeric column."""
    outlier_count: int
    outlier_percentage: float
    method: str = Field(..., description="The method used for detection (e.g., 'IQR', 'Z-score').")

class ClassImbalanceReport(BaseModel):
    """A summary of the class distribution for a classification target."""
    class_counts: Dict[str, int]
    class_percentages: Dict[str, float]

class StatisticalAnalysis(BaseModel):
    """The complete output from the Statistical & Structural Insights module."""
    outlier_report: Dict[str, OutlierReport] = Field(..., description="A mapping of column names to their outlier reports.")
    class_imbalance_report: Optional[ClassImbalanceReport] = None
    correlation_matrix_path: Optional[str] = Field(None, description="File path to the saved correlation matrix data.")
    multicollinearity_report: Dict[str, float] = Field(..., description="A mapping of highly collinear features to their VIF scores.")

# ===================================================================
# Section 4: Advanced Analysis & Visualization Schemas
# ===================================================================

class DimensionalityAnalysis(BaseModel):
    """The complete output from the Dimensionality Reduction module."""
    pca_variance_explained: List[float]
    pca_plot_path: str
    umap_plot_path: str
    tsne_plot_path: str

class VisualizationOutput(BaseModel):
    """A collection of file paths to all generated plot images."""
    univariate_plots: Dict[str, str] = Field(..., description="Mapping of column names to their distribution plot paths.")
    bivariate_plots: Dict[str, str] = Field(..., description="Mapping of plot types (e.g., 'target_vs_feature_x') to their paths.")
    correlation_heatmap_path: Optional[str] = None

# ===================================================================
# Section 5: LLM Summaries & Final Recommendations Schemas
# ===================================================================

class LLMSummary(BaseModel):
    """Holds the natural language summaries generated by the LLM."""
    executive_summary: str
    dataset_overview: str
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
    
    # A list to hold warnings from any module-level failures
    warnings: List[str] = []

    class Config:
        arbitrary_types_allowed = True

