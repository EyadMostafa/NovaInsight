from __future__ import annotations

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, spearmanr, f_oneway 
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
from itertools import product, combinations
from typing import List, Dict, Optional, Tuple, Any
from pathlib import Path

from sleuth.modules.base_module import BaseModule
from sleuth.modules.registry import register_module
from sleuth.config.config import SleuthConfig
from sleuth.exceptions import StatisticalAnalyzerError
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    StatisticalAnalysis,
    OutlierReport,
    ClassImbalanceReport,
    Finding,
    ColumnDetails,
    CorrelationReport,
    Operator,
)


@register_module(
    operator=Operator.STATS,
    dependencies=[Operator.PROFILER, Operator.TARGET],
    is_completed=lambda report: report.statistical_analysis is not None,
)
class StatisticalAnalyzer(BaseModule):
    """
    Performs quantitative statistical analysis to uncover relationships,
    anomalies, and other structural insights in the dataset.
    """

    def __init__(self, config: SleuthConfig):
        """Initializes the StatisticalAnalyzer with app configuration."""
        super().__init__(config)

        self.findings: List[Finding] = []

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        The main execution method for the statistical analysis process.
        Orchestrates all statistical tests and consolidates their findings.
        """
        class_imbalance_report = None
        multicollinearity_report = {}

        outlier_report = self._detect_outliers(df, report.profile.column_details)
        multicollinearity_report = self._detect_multicollinearity(df, report)
        correlation_results = self._analyze_correlations(df, report)

        correlation_report = correlation_results['correlation_report']
        cat_num_matrix = correlation_results['cat_num_matrix']
        num_num_matrix = correlation_results['num_num_matrix']
        cat_cat_matrix = correlation_results['cat_cat_matrix']

        try:
            output_dir = Path(report.metadata.output_dir) / "correlations"
            output_dir.mkdir(parents=True, exist_ok=True)

            cat_num_matrix.to_csv(correlation_report.categorical_numerical_path)
            num_num_matrix.to_csv(correlation_report.numerical_numerical_path)
            cat_cat_matrix.to_csv(correlation_report.categorical_categorical_path)
        
        except Exception as e:
            self.findings.append(Finding(
                level='WARNING', 
                message=f"Could not save correlation matrices to disk. Reason: {e}"
            ))
            
            correlation_report.categorical_numerical_path = None
            correlation_report.numerical_numerical_path = None
            correlation_report.categorical_categorical_path = None

        if report.metadata.task == 'supervised':
            target_name = report.target_analysis.identified_target
            target_details = next((c for c in report.profile.column_details if c.column_name == target_name), None)
            if target_details:
                if target_details.inferred_type in ['categorical', 'boolean']:
                    target_column = df[report.target_analysis.identified_target]
                    class_imbalance_report = self._analyze_class_imbalance(target_column)

                self._check_for_data_leakage(
                    report=report,
                    cat_num_corr=correlation_results.get('cat_num_corr_dict', {}),
                    cat_corr=correlation_results.get('cat_corr_dict', {}),
                    num_corr=correlation_results.get('num_corr_dict', {})
                )
            else:
                self.findings.append(Finding(
                    level='WARNING',
                    message=f"Supervised task specified, but target '{target_name}' details not found in profile."
                ))

        statistical_analysis = StatisticalAnalysis(
            outlier_report=outlier_report,
            multicollinearity_report=multicollinearity_report,
            correlation_report=correlation_report,
            class_imbalance_report=class_imbalance_report,
            findings=self.findings
        )

        report.statistical_analysis = statistical_analysis

        return report

    def _detect_outliers(self, df: pd.DataFrame, columns_details: List[ColumnDetails]) -> Dict[str, OutlierReport]:
            """
            Identifies outliers in numeric columns using the configured method.
            Appends findings to self.findings.
            """
            columns_outlier_analysis = {}
            detection_method = self.config.statistics.outlier_detection_method
            try:
                for col_details in columns_details:
                    if col_details.inferred_type != 'numerical':
                        continue 

                    column_name = col_details.column_name
                    column = df[column_name].dropna()

                    if column.empty:
                        continue

                    scores = None
                    method_name = "N/A"

                    if detection_method == 'mz-score':
                        method_name = "Modified Z-score"
                        scores = self._compute_modified_z_scores(column)
                    elif detection_method == 'z-score':
                        method_name = "Z-score"
                        scores = self._compute_z_scores(column)

                    if scores is None:
                        outlier_report = OutlierReport(
                            outlier_count=0,
                            outlier_percentage=0.0,
                            method=method_name
                        )
                        columns_outlier_analysis[column_name] = outlier_report
                        continue

                    outlier_count = (np.abs(scores) > self.config.statistics.outlier_zscore_threshold).sum()

                    outlier_percentage = outlier_count / len(column) if len(column) > 0 else 0.0

                    if outlier_percentage > self.config.statistics.outlier_warning_threshold:
                        self.findings.append(Finding(
                            level='WARNING',
                            message=(f"Column '{column_name}' has a high percentage of potential outliers ({outlier_percentage:.1%}). " 
                                     "This may skew statistical analysis and impact model performance.")
                        ))

                    outlier_report = OutlierReport(
                        outlier_count=int(outlier_count),
                        outlier_percentage=outlier_percentage,
                        method=method_name
                    )
                    columns_outlier_analysis[column_name] = outlier_report

                return columns_outlier_analysis
            except Exception as e:
                raise StatisticalAnalyzerError(f"Failed to perform outlier detection. Reason: {e}") from e

    def _detect_multicollinearity(self, df: pd.DataFrame, report: AnalysisReport) -> Dict[str, float]:
        """
        ... (docstring) ...
        """
        multicollinearity_report: Dict[str, float] = {}

        target_column = report.target_analysis.identified_target if report.target_analysis else None

        numerical_columns = [
            col.column_name for col in report.profile.column_details 
            if col.inferred_type == 'numerical' and col.column_name != target_column
        ]
        
        if len(numerical_columns) < 2:
            return multicollinearity_report

        try:
            X = df[numerical_columns].dropna()
            if X.shape[0] < 2 or X.shape[1] < 2:
                return multicollinearity_report

            X_with_const = add_constant(X)

            for i in range(1, X_with_const.shape[1]):
                feature_name = X_with_const.columns[i]
                vif_score = variance_inflation_factor(X_with_const.values, i)

                if vif_score > self.config.statistics.multicollinearity_vif_threshold:
                    multicollinearity_report[feature_name] = vif_score
                    self.findings.append(Finding(
                        level='WARNING',
                        message=(
                            f"High Multicollinearity: The column '{feature_name}' has a high VIF score of {vif_score:.2f}. "
                            "This suggests it is highly correlated with other features and may be redundant."
                        )
                    ))
            
            return multicollinearity_report

        except Exception as e:
            self.findings.append(Finding(level='WARNING', message=f"Could not perform multicollinearity analysis: {e}"))
            raise StatisticalAnalyzerError(f"Could not perform multicollinearity analysis. Reason: {e}") from e

    def _analyze_correlations(self, df: pd.DataFrame, report: AnalysisReport) -> Dict[str, Any]:
        """
        Calculates comprehensive correlations for all variable type pairs (cat-cat, num-num, cat-num).
        Generates findings based on both strength and statistical significance.
        """
        try:
            
            cat_num_corr: Dict[Tuple[str, str], float] = {}
            num_corr: Dict[Tuple[str, str], float] = {}
            cat_corr: Dict[Tuple[str, str], float] = {}
            
            cat_num_corr_pvals: Dict[Tuple[str, str], float] = {}
            num_corr_pvals: Dict[Tuple[str, str], float] = {}
            cat_corr_pvals: Dict[Tuple[str, str], float] = {}

            p_value_threshold = self.config.statistics.p_value_threshold

            column_details = report.profile.column_details
            categorical_columns = [column.column_name for column in column_details if column.inferred_type in ['categorical', 'boolean']]
            numerical_columns = [column.column_name for column in column_details if column.inferred_type == 'numerical']

            cat_num_pairs = list(product(categorical_columns, numerical_columns))
            cat_pairs = list(combinations(categorical_columns, 2))
            num_pairs = list(combinations(numerical_columns, 2))
            
            corr_ratio_threshold = float(self.config.statistics.correlation_ratio_threshold)
            for cat_col, num_col in cat_num_pairs:
                corr, pvalue = self._compute_correlation_ratio(df[cat_col], df[num_col])
                cat_num_corr[(cat_col, num_col)] = corr
                cat_num_corr_pvals[(cat_col, num_col)] = pvalue
                if corr > corr_ratio_threshold and pvalue < p_value_threshold:
                    self.findings.append(Finding(
                        level='WARNING',
                        message= (f"Strong Association: Numeric feature '{num_col}' and categorical feature "
                                  f"'{cat_col}' are highly associated (Correlation Ratio η = {corr:.2f}, p={pvalue:.3g}).")
                    ))

            cramers_v_threshold = float(self.config.statistics.cramers_v_correlation_threshold)
            for col1, col2 in cat_pairs:
                corr, pvalue = self._compute_cramers_v(df[col1], df[col2])
                cat_corr[(col1, col2)] = corr
                cat_corr_pvals[(col1, col2)] = pvalue
                if corr > cramers_v_threshold and pvalue < p_value_threshold:
                    self.findings.append(Finding(
                        level='WARNING',
                        message= (f"Strong Association: Categorical features '{col1}' and '{col2}' "
                                  f"are highly associated (Cramér's V = {corr:.2f}, p={pvalue:.3g}).")
                    ))

            spearman_threshold = float(self.config.statistics.spearman_correlation_threshold)
            for col1, col2 in num_pairs:

                pair_df = df[[col1, col2]].dropna()
                
                if pair_df.shape[0] < 2:
                    num_corr[(col1, col2)] = 0.0
                    num_corr_pvals[(col1, col2)] = np.nan
                    continue

                corr, pvalue = spearmanr(pair_df[col1], pair_df[col2])
                if pd.isna(corr): corr = 0.0 
                num_corr[(col1, col2)] = corr
                num_corr_pvals[(col1, col2)] = pvalue
                if abs(corr) > spearman_threshold and pvalue < p_value_threshold:
                    self.findings.append(Finding(
                        level='WARNING',
                        message= (f"High Monotonic Correlation: Features '{col1}' and '{col2}' "
                                  f"have a strong monotonic relationship (Spearman's ρ = {corr:.2f}, p={pvalue:.3g}).")
                    ))
            output_dir = Path(report.metadata.output_dir) / "correlations"
            correlation_report = CorrelationReport(
                numerical_numerical_path=output_dir / "numerical_corr.csv",
                categorical_categorical_path=output_dir / "categorical_corr.csv",
                categorical_numerical_path=output_dir / "categorical_numerical_corr.csv"
            )

            results = {
                'correlation_report': correlation_report,
                'cat_num_matrix': self._dict_to_corr_matrix(cat_num_corr),
                'num_num_matrix': self._dict_to_corr_matrix(num_corr),
                'cat_cat_matrix': self._dict_to_corr_matrix(cat_corr),
                'cat_num_corr_dict': {k: (v, cat_num_corr_pvals.get(k, np.nan)) for k, v in cat_num_corr.items()},
                'num_corr_dict': {k: (v, num_corr_pvals.get(k, np.nan)) for k, v in num_corr.items()},
                'cat_corr_dict': {k: (v, cat_corr_pvals.get(k, np.nan)) for k, v in cat_corr.items()},
            }

            return results
        except Exception as e:
            raise StatisticalAnalyzerError(f"Failed to perform correlation analysis. Reason: {e}") from e

    def _analyze_class_imbalance(self, target_column: pd.Series) -> Optional[ClassImbalanceReport]:
        """Analyzes and reports on the class distribution of the target variable."""
        try:
            value_counts = target_column.value_counts()
            total = value_counts.sum()

            if total == 0:
                return ClassImbalanceReport(class_counts={}, class_percentages={})

            percentages = {str(k): float((v / total)) for k, v in value_counts.items()}
            
            if len(percentages) < 2:
                return ClassImbalanceReport(class_counts={str(k): int(v) for k, v in value_counts.items()}, class_percentages=percentages)

            min_pct = min(percentages.values())
            max_pct = max(percentages.values())
            min_class = min(percentages, key=percentages.get)
            max_class = max(percentages, key=percentages.get)

            max_min_ratio = max_pct / min_pct
            if max_min_ratio > self.config.statistics.class_imbalance_threshold:
                self.findings.append(Finding(
                    level='WARNING',
                    message=(f"Class Imbalance Detected: The target variable '{target_column.name}' is imbalanced. "
                             f"The majority class '{max_class}' is {max_min_ratio:.1f} times more frequent than the minority class "
                             f"'{min_class}'. This can bias model training.")
                ))

            value_counts_str_keys = {str(k): int(v) for k, v in value_counts.items()}

            class_imbalance_report = ClassImbalanceReport(
                class_counts=value_counts_str_keys,
                class_percentages=percentages
            )
            
            return class_imbalance_report
        except Exception as e:
            finding = Finding(level='WARNING', message=f"Could not perform class imbalance analysis: {e}")
            self.findings.append(finding)
            return None

    def _check_for_data_leakage(
            self,
            report: AnalysisReport,
            cat_num_corr: Dict[Tuple[str, str], Tuple[float, float]],
            cat_corr: Dict[Tuple[str, str], Tuple[float, float]],
            num_corr: Dict[Tuple[str, str], Tuple[float, float]]
        ) -> None:
            """
            Checks for features that are almost perfectly correlated with the target.
            Appends findings to self.findings. (Only called for supervised tasks).
            """
            p_value_threshold = self.config.statistics.p_value_threshold

            target_name = report.target_analysis.identified_target
            if not target_name: return

            try:
                target_type = next((column.inferred_type for column in report.profile.column_details if column.column_name == target_name), "")

                for pair, (corr, pvalue) in cat_num_corr.items():
                    if corr > self.config.statistics.correlation_ratio_leakage_threshold and target_name in pair and pvalue < p_value_threshold:
                        column_name = pair[0] if pair[1] == target_name else pair[1]
                        self.findings.append(Finding(
                            level='WARNING',
                            message=(f"Potential Data Leakage: The feature '{column_name}' has a near-perfect association"
                                     f"with the target '{target_name}' (Correlation Ratio η = {corr:.2f}, p={pvalue:.3g}), "
                                     f"indicating it can almost perfectly predict the target.")
                        ))

                if target_type == 'numerical':
                    for pair, (corr, pvalue) in num_corr.items():
                        if abs(corr) > self.config.statistics.spearman_leakage_threshold and target_name in pair and pvalue < p_value_threshold:
                            column_name = pair[0] if pair[1] == target_name else pair[1]
                            self.findings.append(Finding(
                                level='WARNING',
                                message=(f"Potential Data Leakage: The feature '{column_name}' "
                                         f"is almost perfectly correlated with the target '{target_name}' "
                                         f"(Spearman's ρ = {corr:.2f}, p={pvalue:.3g}), indicating it is likely a proxy.")
                            ))
                else:
                    for pair, (corr, pvalue) in cat_corr.items():
                        if corr > self.config.statistics.cramers_v_leakage_threshold and target_name in pair and pvalue < p_value_threshold:
                            column_name = pair[0] if pair[1] == target_name else pair[1]
                            self.findings.append(Finding(
                                level='WARNING',
                                message=(f"Potential Data Leakage: The categorical feature '{column_name}' "
                                         f"has a near-perfect association with the target '{target_name}' "
                                         f"(Cramér's V = {corr:.2f}, p={pvalue:.3g}), suggesting it contains redundant information.")
                            ))

            except Exception as e:
                self.findings.append(Finding(level='WARNING', message=f"Failed to perform data leakage detection: {e}"))

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
    def _compute_correlation_ratio(categories: pd.Series, values: pd.Series) -> Tuple[float, float]:
        """Computes Correlation Ratio (η) and its p-value (from F-test)."""
        data = pd.DataFrame({'categories': categories, 'values': values}).dropna()

        if data.empty or data['categories'].nunique() < 2:
            return np.nan, np.nan

        population_mean = np.mean(data['values'])
        ss_total = np.sum((data['values'] - population_mean) ** 2)

        if ss_total == 0:
            return np.nan, np.nan

        groups = [data['values'][data['categories'] == cat] for cat in np.unique(data['categories'])]
        
        groups_for_f_test = [g for g in groups if len(g) >= 2]
        try:
            if len(groups_for_f_test) < 2:
                 f_value, p_value = np.nan, np.nan
            else:
                 f_value, p_value = f_oneway(*groups_for_f_test)
        except ValueError:
             f_value, p_value = np.nan, np.nan

        n_per_group = [len(group) for group in groups]
        mean_per_group = [np.mean(group) for group in groups if not group.empty]
        n_per_group = [n for n in n_per_group if n > 0] 

        ss_between = np.sum(n_per_group * (np.array(mean_per_group) - population_mean) ** 2)

        ratio = ss_between / ss_total
        return np.sqrt(max(0, ratio)), p_value 

    @staticmethod
    def _compute_cramers_v(column1: pd.Series, column2: pd.Series) -> Tuple[float, float]:
        """Computes Cramér's V statistic and its p-value."""
        confusion_matrix = pd.crosstab(column1, column2)
        chi2, pvalue, _, _ = chi2_contingency(confusion_matrix)
        n = confusion_matrix.sum().sum()

        k = min(confusion_matrix.shape)
        
        if k == 1 or n == 0 or (n * (k - 1)) == 0:
             return np.nan, np.nan

        v = np.sqrt(chi2 / (n * (k - 1)))
        return v, pvalue

    @staticmethod
    def _dict_to_corr_matrix(corr_dict: dict) -> pd.DataFrame:
        """
        Converts a dictionary of pairwise correlations (keyed as (col1, col2): value)
        into a symmetric correlation matrix DataFrame.
        """
        if not corr_dict:
            return pd.DataFrame()

        cols_seen = set()
        cols_ordered = []
        for pair in corr_dict.keys():
            for c in pair:
                if c not in cols_seen:
                    cols_seen.add(c)
                    cols_ordered.append(c)

        matrix = pd.DataFrame(index=cols_ordered, columns=cols_ordered, dtype=float)

        for (c1, c2), val in corr_dict.items():
            matrix.loc[c1, c2] = val
            matrix.loc[c2, c1] = val

        for c in cols_ordered:
            matrix.loc[c, c] = 1.0

        return matrix

