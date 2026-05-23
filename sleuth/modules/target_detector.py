from __future__ import annotations

from typing import Literal

import pandas as pd

from sleuth.config.config import SleuthConfig
from sleuth.exceptions import TargetDetectorError
from sleuth.modules.base_module import BaseModule
from sleuth.modules.registry import register_module
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    CandidateTarget,
    ColumnDetails,
    Finding,
    Operator,
    TargetVariableAnalysis,
)


@register_module(
    operator=Operator.TARGET,
    dependencies=[Operator.PROFILER],
    is_completed=lambda report: report.target_analysis is not None,
)
class TargetDetector(BaseModule):
    """
    Determines the machine learning context by identifying the target variable.

    This module operates in two modes:
    1.  Validation: If a user specifies a target, it validates its existence.
    2.  Inference: If no target is specified, it uses heuristics to find the most
        likely target for a supervised machine learning task.
    """

    def __init__(self, config: SleuthConfig):
        """Initializes the TargetDetector module with the application configuration."""
        super().__init__(config)

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        The main execution method for the target detection process.

        It orchestrates the validation or inference of a target variable and then
        checks for potential data leakage.
        """
        user_specified_target = report.metadata.user_target

        if user_specified_target:
            candidate_target = self._validate_user_target(
                user_target=user_specified_target,
                column_details=report.profile.column_details
            )

            target_analysis = TargetVariableAnalysis(
                detection_method='user_specified',
                candidate_targets=[candidate_target],
                identified_target=candidate_target.column_name,
                findings=[]
            )

            report.target_analysis = target_analysis

        else:
            target_analysis = self._infer_target_from_profile(df=df, columns=report.profile.column_details)
            report.target_analysis = target_analysis

        return report

    def _validate_user_target(self, user_target: str, column_details: list[ColumnDetails]) -> CandidateTarget:
        """
        Validates that the user-specified target column exists in the dataset.
        """
        target_column_details = None

        try:
            for column in column_details:
                if column.column_name == user_target:
                    target_column_details = column
                    break

            if not target_column_details:
                raise TargetDetectorError(f"User specified target column '{user_target}' does not exist in the dataset.")

            invalid_types = ['text', 'datetime', 'id']
            if target_column_details.inferred_type in invalid_types:
                raise TargetDetectorError(
                    f"Specified target '{user_target}' has an unsuitable type "
                    f"('{target_column_details.inferred_type}'). Please choose a numeric, bool, or categorical column."
                )

            ml_task = self._task_detection(target_column_details)

            if not ml_task:
                raise TargetDetectorError(f"Failed to infer ml task for user specified target column '{user_target}'.")

            candidate_target = CandidateTarget(
                column_name=user_target,
                ml_task=ml_task,
                justification='user_specified',
                confidence_score=1.0
            )

            return candidate_target
        except TargetDetectorError:
            raise
        except Exception as e:
            raise TargetDetectorError(f"Failed to validate user target. Reason: {e}") from e

    def _infer_target_from_profile(self, df: pd.DataFrame, columns: list[ColumnDetails]) -> TargetVariableAnalysis:
        """
        Uses heuristics on column metadata to infer the most likely target.
        """
        candidates = []
        best_candidate = None
        findings: list[Finding] = []

        try:
            for col in columns:
                if col.inferred_type in ['id', 'text', 'datetime']:
                    continue

                column_name = col.column_name
                ml_task = self._task_detection(col)
                confidence_score=1.0

                penalty_words = ['id', 'key', 'name', 'code']

                confidence_score -= col.missing_values_pct
                if any(word in column_name.lower() for word in penalty_words):
                    confidence_score -= 0.4

                if col.inferred_type == 'boolean':
                    confidence_score += 0.2
                elif col.inferred_type == 'categorical':
                    if col.unique_values_count > (0.8 * self.config.target_detection.max_categorical_cardinality):
                        confidence_score -= 0.15
                    elif col.unique_values_count == 2:
                        confidence_score += 0.2
                elif col.inferred_type == 'numerical':
                    skewness = abs(df[column_name].skew())
                    if skewness > self.config.profiler.skewness_threshold:
                        confidence_score -= 0.15
                    elif skewness < 0.5:
                        confidence_score += 0.1

                    if col.unique_values_count < self.config.target_detection.max_categorical_cardinality:
                        confidence_score -= 0.2
                    elif col.unique_values_pct >= 0.8:
                        confidence_score += 0.1

                confidence_score = round(confidence_score, 2)

                candidate = CandidateTarget(
                    column_name=column_name,
                    ml_task=ml_task,
                    justification=f"Inferred as a {ml_task} target with a confidence score of {confidence_score}.",
                    confidence_score=confidence_score
                )
                candidates.append(candidate)

            if candidates:
                candidates.sort(key=lambda c: c.confidence_score, reverse=True)
            else:
                raise TargetDetectorError("No valid candidate target columns found.")

            for cand in candidates:
                if not cand.ml_task:
                    finding = Finding(
                        level='WARNING',
                        message=f"Failed to infer ml task for candidate column '{cand.column_name}'. Skipping to next best candidate."
                    )
                    findings.append(finding)
                else:
                    best_candidate = cand.column_name
                    break

            if not best_candidate:
                raise TargetDetectorError("Failed to identify a best candidate target column.")

            target_analysis = TargetVariableAnalysis(
                detection_method='auto',
                candidate_targets=candidates,
                identified_target=best_candidate,
                findings=findings
            )

            return target_analysis
        except TargetDetectorError:
            raise
        except Exception as e:
            raise TargetDetectorError(f"Failed to infer target from profile. Reason: {e}") from e


    def _task_detection(self, column: ColumnDetails) -> Literal['classification', 'regression'] | None:
        try:
            if (column.inferred_type in ['categorical', 'boolean']
            or (column.inferred_type == 'numerical' and column.unique_values_count <= self.config.target_detection.max_categorical_cardinality)):
                return 'classification'

            if column.inferred_type == 'numerical':
                return 'regression'

            return None
        except Exception as e:
            raise TargetDetectorError(f"Failed to perform task detection. Reason: {e}") from e
