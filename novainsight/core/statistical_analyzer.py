from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from itertools import product, combinations
from typing import List, Dict, Optional, Tuple
from pathlib import Path
from novainsight.core.base_module import BaseModule
from novainsight.config.config import NovaInsightConfig
from novainsight.schemas.analysis_report import (
    AnalysisReport,
    StatisticalAnalysis,
    OutlierReport,
    ClassImbalanceReport,
    Finding,
    ColumnDetails,
    CorrelationReport
)

class StatisticalAnalyzer(BaseModule):
    """
    Performs quantitative statistical analysis to uncover relationships,
    anomalies, and other structural insights in the dataset.
    """

    def __init__(self, config: NovaInsightConfig):
        """Initializes the StatisticalAnalyzer with app configuration."""
        super().__init__(config)

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        The main execution method for the statistical analysis process.

        It orchestrates various statistical tests and is task-aware, running
        certain analyses only when a target variable is present for a
        supervised task.
        """
        findings = []
            
        outlier_report, outlier_findings = self._detect_outliers(df, report.profile.column_details)
        multicollinearity_report, multicollinearity_findings = self._detect_multicollinearity(df, report)
        correlation_report, correlation_findings = self._analyze_correlations(df, report)

        findings += outlier_findings + multicollinearity_findings + correlation_findings

        target_name = report.target_analysis.identified_target
        target_details = next(c for c in report.profile.column_details if c.column_name == target_name)

        if report.metadata.task == 'supervised' and target_details.inferred_type in ['categorical', 'boolean']:
            target_column = df[report.target_analysis.identified_target]
            class_imbalance_report, class_imbalance_finding = self._analyze_class_imbalance(target_column)
            if class_imbalance_finding: findings.append(class_imbalance_finding)
        else: class_imbalance_report = None

        statistical_analysis = StatisticalAnalysis(
            outlier_report=outlier_report,
            multicollinearity_report=multicollinearity_report,
            correlation_report=correlation_report,
            class_imbalance_report=class_imbalance_report,
            findings=findings
        )

        report.statistical_analysis = statistical_analysis
        report.findings += findings

        return report

    def _detect_outliers(self, df: pd.DataFrame, columns_details: List[ColumnDetails]) -> Tuple[Dict[str, OutlierReport], List[Finding]]:
        """
        Identifies outliers in numeric columns using a modified Z-score that is based on the 
        median and median absolute deviation (MAD) instead of the standard mean and standard deviation 
        to mitigate the masking of mild outliers caused by extreme outliers,
        can also use the regular mean based Z-score.
        """
        # Consider adding IQR and Isolation Forests methods.

        columns_outlier_analysis = {}
        findings = []
        detection_method = self.config.statistics.outlier_detection_method
        try:
            for col_details in columns_details:
                if col_details.inferred_type != 'numerical':
                    continue 

                column_name = col_details.column_name
                column = df[column_name]
                scores = None

                if detection_method == 'mz-score':

                    scores = self._compute_modified_z_scores(column)

                    if scores is None:
                        outlier_report = OutlierReport(
                            outlier_count=0,
                            outlier_percentage=0.0,
                            method="Modified Z-score"
                        )
                        columns_outlier_analysis[column_name] = outlier_report
                        continue

                elif detection_method == 'z-score':

                    scores = self._compute_z_scores(column)
                    if scores is None:
                        outlier_report = OutlierReport(
                            outlier_count=0,
                            outlier_percentage=0.0,
                            method="Z-score"
                        )
                        columns_outlier_analysis[column_name] = outlier_report
                        continue

                outlier_count = (np.abs(scores) > self.config.statistics.outlier_zscore_threshold).sum()
                outlier_percentage = outlier_count / column.shape[0]

                if outlier_percentage > self.config.statistics.outlier_warning_threshold:
                    finding = Finding(
                        level='WARNING',
                        message=(f"Column '{column_name}' has a high percentage of potential outliers ({outlier_percentage:.1%}). " 
                                 "This may skew statistical analysis and impact model performance.")
                    )

                    findings.append(finding)

                outlier_report = OutlierReport(
                    outlier_count=outlier_count,
                    outlier_percentage=outlier_percentage,
                    method="Modified Z-score" if detection_method == 'mz-score' else "Z-score"
                )
                columns_outlier_analysis[column_name] = outlier_report

            return columns_outlier_analysis, findings
        except Exception as e:
            raise ValueError(f"Failed to perform outlier detection. Reason: {e}")

    def _detect_multicollinearity(self, df: pd.DataFrame, report: AnalysisReport) -> Tuple[Dict[str, float], List[Finding]]:
        """
        Calculates the Variance Inflation Factor (VIF) for numeric features.
        """
        findings = []
        multicollinearity_report: Dict[str, float] = {}

        target_column = report.target_analysis.identified_target if report.target_analysis else None

        numerical_columns = [
            col.column_name for col in report.profile.column_details 
            if col.inferred_type == 'numerical' and col.column_name != target_column
        ]

        if len(numerical_columns) < 2:
            return multicollinearity_report, findings

        try:
            X = add_constant(df[numerical_columns].dropna())

            for i in range(1, X.shape[1]):
                feature_name = X.columns[i]
                vif_score = variance_inflation_factor(X.values, i)

                if vif_score > self.config.statistics.multicollinearity_vif_threshold:
                    multicollinearity_report[feature_name] = vif_score

                    finding = Finding(
                        level='WARNING',
                        message=(
                            f"High Multicollinearity: The column '{feature_name}' has a high VIF score of {vif_score:.2f}. "
                            "This suggests it is highly correlated with other features and may be redundant, "
                            "which can negatively impact model interpretation."
                        )
                    )
                    findings.append(finding)

            return multicollinearity_report, findings

        except Exception as e:
            raise ValueError(f"Could not perform multicollinearity analysis. Reason: {e}")

    def _analyze_correlations(self, df: pd.DataFrame, report: AnalysisReport) -> Tuple[CorrelationReport, List[Finding]]:
        """
        Calculates a comprehensive correlation matrix for every variable type pair (cat-cat, num-num, cat-num).
        """
        try:
            column_details = report.profile.column_details
            findings = []

            cat_num_corr: Dict[Tuple[str, str], float] = {}
            num_corr: Dict[Tuple[str, str], float] = {}
            cat_corr: Dict[Tuple[str, str], float] = {}

            categorical_columns = [column.column_name for column in column_details if column.inferred_type in ['categorical', 'boolean']]
            numerical_columns = [column.column_name for column in column_details if column.inferred_type == 'numerical']

            cat_num_pairs = list(product(categorical_columns, numerical_columns))
            cat_pairs = list(combinations(categorical_columns, 2))
            num_pairs = list(combinations(numerical_columns, 2))

            for pair in cat_num_pairs:
                categoric, numeric = pair
                corr = self._compute_correlation_ratio(df[categoric], df[numeric])

                if abs(corr) > float(self.config.statistics.correlation_ratio_threshold):
                    finding = Finding(
                        level='WARNING',
                        message= (f"Strong Association: Numeric feature '{numeric}' and categorical feature "
                                  f"'{categoric}' are highly associated (Correlation Ratio η = {corr:.2f}).")
                    )
                    findings.append(finding)

                cat_num_corr[pair] = corr

            for pair in cat_pairs:
                col1, col2 = pair
                corr = self._compute_cramers_v(df[col1], df[col2])

                if abs(corr) > float(self.config.statistics.cramers_v_correlation_threshold):
                    finding = Finding(
                        level='WARNING',
                        message= (f"Strong Association: Categorical features '{col1}' and '{col2}' "
                                  f"are highly associated (Cramér's V = {corr:.2f}).")
                    )
                    findings.append(finding)

                cat_corr[pair] = corr

            for pair in num_pairs:
                col1, col2 = pair
                corr, _ = spearmanr(df[col1], df[col2])

                if abs(corr) > float(self.config.statistics.spearman_correlation_threshold):
                    finding = Finding(
                        level='WARNING',
                        message= (f"High Monotonic Correlation: Features '{col1}' and '{col2}' "
                                  f"have a strong monotonic relationship (Spearman's ρ = {corr:.2f}).")
                    )

                    findings.append(finding)

                num_corr[pair] = corr

            if report.metadata.task == 'supervised':
                findings += self._check_for_data_leakage(df, report, cat_num_corr, cat_corr, num_corr)

            cat_num_corr_matrix: pd.DataFrame = self._dict_to_corr_matrix(cat_num_corr)
            num_corr_matrix: pd.DataFrame = self._dict_to_corr_matrix(num_corr)
            cat_corr_matrix: pd.DataFrame = self._dict_to_corr_matrix(cat_corr)

            cat_num_corr_path = Path(f"{report.metadata.output_dir}/categorical_numerical_corr.csv")
            num_corr_path = Path(f"{report.metadata.output_dir}/numerical_corr.csv")
            cat_corr_path = Path(f"{report.metadata.output_dir}/categorical_corr.csv")

            cat_num_corr_matrix.to_csv(cat_num_corr_path)
            num_corr_matrix.to_csv(num_corr_path)
            cat_corr_matrix.to_csv(cat_corr_path)

            correlation_report = CorrelationReport(
                numerical_numerical_path=num_corr_path,
                categorical_categorical_path=cat_corr_path,
                categorical_numerical_path=cat_num_corr_path
            )

            return correlation_report, findings
        except Exception as e:
            raise ValueError(f"Failed to perform correlation analysis. Reason: {e}")

    def _analyze_class_imbalance(self, target_column: pd.Series) -> Tuple[ClassImbalanceReport, Optional[Finding]]:
        """
        Analyzes the distribution of classes in a categorical target variable.
        (Only called for supervised classification tasks).
        """
        try:
            value_counts = target_column.value_counts()
            total = value_counts.sum()
            percentages = {str(k): float((v / total)) for k, v in value_counts.items()}

            min = (None, float('inf'))
            max = (None, 0.0)

            for value, percentage in percentages.items():
                if percentage < min[1]: min = (value, percentage)
                elif percentage > max[1]: max = (value, percentage)

            max_min_ratio = max[1] / min[1]

            finding = None
            if max_min_ratio > self.config.statistics.class_imbalance_threshold:
                finding = Finding(
                    level='WARNING',
                    message=(f"Class Imbalance Detected: The target variable '{target_column.name}' is imbalanced. "
                             f"The majority class '{max[0]}' is {max_min_ratio:.1f} times more frequent than the minority class "
                             f"'{min[0]}'. This can bias model training.")
                )

            value_counts = {str(k): int(v) for k, v in value_counts.items()}

            class_imbalance_report = ClassImbalanceReport(
                class_counts=value_counts,
                class_percentages=percentages
            )

            return class_imbalance_report, finding
        except Exception as e:
            raise ValueError(f"Could not perform class imbalance analysis. Reason: {e}")

    def _check_for_data_leakage(
        self,
        df: pd.DataFrame, 
        report: AnalysisReport,
        cat_num_corr: Dict[Tuple[str, str], float],
        cat_corr: Dict[Tuple[str, str], float],
        num_corr: Dict[Tuple[str, str], float]) -> List[Finding]:
        """
        Checks for features that are almost perfectly correlated with the target.
        (Only called for supervised tasks).
        """
        findings = []

        target_name = report.target_analysis.identified_target

        try:
            target_type = [column.inferred_type for column in report.profile.column_details if column.column_name == target_name][0]

            for pair, corr in cat_num_corr.items():
                if corr > self.config.statistics.correlation_ratio_leakage_threshold and target_name in pair:
                    column_name = pair[0] if pair[1] == target_name else pair[1]

                    finding = Finding(
                        level='WARNING',
                        message=(f"Potential Data Leakage: The feature '{column_name}' has a near-perfect association"
                                 f"with the target '{target_name}' (Correlation Ratio η = {corr:.2f}), "
                                 f"indicating it can almost perfectly predict the target.")
                    )

                    findings.append(finding)

            if target_type == 'numerical':
                for pair, corr in num_corr.items():
                    if abs(corr) > self.config.statistics.spearman_leakage_threshold and target_name in pair:
                        column_name = pair[0] if pair[1] == target_name else pair[1]

                        finding = Finding(
                            level='WARNING',
                            message=(f"Potential Data Leakage: The feature '{column_name}' "
                                     f"is almost perfectly correlated with the target '{target_name}' "
                                     f"(Spearman's ρ = {corr:.2f}), indicating it is likely a proxy.")
                        )

                        findings.append(finding)
            else:
                for pair, corr in cat_corr.items():
                    if corr > self.config.statistics.cramers_v_leakage_threshold and target_name in pair:
                        column_name = pair[0] if pair[1] == target_name else pair[1]

                        finding = Finding(
                            level='WARNING',
                            message=(f"Potential Data Leakage: The categorical feature '{column_name}' "
                                     f"has a near-perfect association with the target '{target_name}' "
                                     f"(Cramér's V = {corr:.2f}), suggesting it contains redundant information.")
                        )

                        findings.append(finding)

            return findings
        except Exception as e:
            raise ValueError(f"Failed to perform data leakage detection. Reason: {e}")

    @staticmethod
    def _compute_modified_z_scores(column: pd.Series) -> Optional[pd.Series]:
        """
        Computes the modified z-score of a column.
        """
        median = np.median(column)
        deviations_from_median = column - median
        mad = np.median(np.abs(deviations_from_median))
        if mad == 0:
            return None
        
        return (0.6745 * deviations_from_median) / mad

    @staticmethod
    def _compute_z_scores(column: pd.Series) -> Optional[pd.Series]:
        """
        Computes the z-score of a column.
        """
        mean = np.mean(column)
        deviations_from_mean = column - mean
        std = np.std(column)

        if std == 0:
                return None

        return deviations_from_mean / std

    @staticmethod
    def _compute_correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
        """
        Computes the correlation ratio (η) for categorical-numerical association.
        """
        data = pd.DataFrame({'categories': categories, 'values': values}).dropna()
        if data.empty:
            return 0.0

        population_mean = np.mean(data['values'])
        ss_total = np.sum((data['values'] - population_mean) ** 2)

        if ss_total == 0:
            return 0.0

        groups = [data['values'][data['categories'] == cat] for cat in np.unique(data['categories'])]
        n_per_group = [len(group) for group in groups]
        mean_per_group = [np.mean(group) for group in groups]

        ss_between = np.sum(n_per_group * (np.array(mean_per_group) - population_mean) ** 2)

        ratio = ss_between / ss_total
        return np.sqrt(max(0, ratio)) 

    @staticmethod
    def _compute_cramers_v(column1: pd.Series, column2: pd.Series) -> float:
        """
        Computes Cramér's V statistic for categorical-categorical association.
        """
        confusion_matrix = pd.crosstab(column1, column2)
        chi2, _, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()
        k = min(confusion_matrix.shape)
        return np.sqrt(chi2 / (n * (k - 1)))

    @staticmethod
    def _dict_to_corr_matrix(corr_dict: dict) -> pd.DataFrame:
        """
        Converts a dictionary of pairwise correlations (keyed as (col1, col2): value)
        into a symmetric correlation matrix DataFrame.
        """
        if not corr_dict:
            return pd.DataFrame()

        cols = sorted(set([c for pair in corr_dict.keys() for c in pair]))

        matrix = pd.DataFrame(index=cols, columns=cols, dtype=float)

        for (c1, c2), val in corr_dict.items():
            matrix.loc[c1, c2] = val
            matrix.loc[c2, c1] = val

        for c in cols:
            matrix.loc[c, c] = 1.0

        return matrix

        


