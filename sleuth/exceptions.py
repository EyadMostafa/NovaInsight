"""
exceptions.py

Custom exception hierarchy for Sleuth.

Two-tier design:
  - ModuleError    : a pipeline module failed; the orchestrator catches this,
                     records a Finding, skips downstream dependents, and continues.
  - PipelineError  : an infrastructure / unrecoverable failure; the orchestrator
                     re-raises and the CLI surfaces a clean message then exits.

Both inherit from SleuthError so callers can do isinstance(e, SleuthError).
"""

from __future__ import annotations


class SleuthError(Exception):
    """Root exception for all Sleuth errors."""


# ---------------------------------------------------------------------------
# Module-level exceptions  (recoverable — pipeline continues after catch)
# ---------------------------------------------------------------------------


class ModuleError(SleuthError):
    """
    Raised when a pipeline module fails to complete its work.

    Carries optional structured context so the orchestrator can produce a
    richer Finding without parsing the message string.

    Attributes:
        column: The offending column name, if the failure is column-specific.
        module: An explicit module name override (defaults to the class name).
    """

    def __init__(
        self,
        message: str,
        *,
        column: str | None = None,
        module: str | None = None,
    ) -> None:
        super().__init__(message)
        self.column: str | None = column
        self.module: str | None = module


class ProfilerError(ModuleError):
    """Raised by DataProfiler on unrecoverable column or dataset profiling failure."""


class TargetDetectorError(ModuleError):
    """Raised by TargetDetector when target validation or inference fails."""


class StatisticalAnalyzerError(ModuleError):
    """Raised by StatisticalAnalyzer on unrecoverable statistical computation failure."""


class DimensionalityReducerError(ModuleError):
    """Raised by DimensionalityReducer when PCA / t-SNE / UMAP processing fails."""


class VisualizerError(ModuleError):
    """Raised by Visualizer when plot generation fails unrecoverably."""


class LLMSummarizerError(ModuleError):
    """Raised by LLMSummarizer when provider init or querying fails unrecoverably."""


class ReportGeneratorError(ModuleError):
    """Raised by ReportGenerator when HTML rendering or writing fails unrecoverably."""


# ---------------------------------------------------------------------------
# Pipeline-level exceptions  (unrecoverable — orchestrator re-raises, CLI exits)
# ---------------------------------------------------------------------------


class PipelineError(SleuthError):
    """Base for infrastructure / orchestration failures that halt the pipeline."""


class DataLoadError(PipelineError):
    """Raised when the input file cannot be read, parsed, or validated."""


class OrchestratorError(PipelineError):
    """Raised by the orchestrator on DAG / dependency resolution failures."""


class CacheError(PipelineError):
    """Raised by CacheManager on unrecoverable read/write failures."""


class ConfigError(PipelineError):
    """Raised by the config loader when Pydantic validation fails at startup."""
