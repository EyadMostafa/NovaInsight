import pandas as pd
from pandas.api.types import (
    is_datetime64_any_dtype,
    is_timedelta64_dtype,
    is_period_dtype,
    is_interval_dtype,
    is_bool_dtype,
    is_numeric_dtype
)
from novainsight.schemas.analysis_report import (DatasetProfile, DatasetStats,
                                                 ColumnDetails, Finding, AnalysisReport)
from novainsight.config.config import NovaInsightConfig
from novainsight.core.base_module import BaseModule
from typing import Tuple, List


class DataProfiler(BaseModule):
    """
    Analyzes a DataFrame to produce a detailed structural and statistical profile.
    """
    def __init__(self, config: NovaInsightConfig):
        """
        Initializes the profiler with the data and configuration.
        """
        super().__init__(config)

        self.column_memory_usage: float = None
        self.total_memory_usage: float = None
        self.original_row_count: int | None = None
        self.findings: List[Finding] = []

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        Executes the full profiling process.
        """
        self.original_row_count = report.profile.dataset_stats.original_row_count
        dataset_stats = self._profile_dataset_level(df)
        column_details = self._profile_column_level(df)
        dataset_profile = DatasetProfile(
            dataset_stats=dataset_stats,
            column_details=column_details,
            findings=self.findings
        )

        report.profile = dataset_profile

        return report
    
    def _profile_dataset_level(self, df: pd.DataFrame) -> DatasetStats:
        """
        Calculates statistics for the entire dataset (e.g., rows, columns, memory).
        """
        try:
            row_count, col_count = df.shape
            self.column_memory_usage = df.memory_usage(deep=True)
            self.total_memory_usage = self.column_memory_usage.sum() / (1024**2)
            duplicate_rows_count = df.duplicated().sum()

            cols_pct = col_count / self.original_row_count
            duplicates_pct = duplicate_rows_count / row_count if row_count > 0 else 0.0

            findings : List[Finding] = []

            if duplicates_pct > self.config.profiler.duplicate_threshold:
                finding = Finding(
                        level='WARNING',
                        message=(
                            f"High percentage of duplicate rows detected ({duplicates_pct:.2%}). "
                            "This can skew statistical analysis and bias model training."
                        )
                    )
                findings.append(finding)

            if self.total_memory_usage > self.config.profiler.total_memory_threshold:
                finding = Finding(
                            level='WARNING',
                            message=(
                                f"Large dataset detected. The in-memory size is {self.total_memory_usage:.5f} MB, "
                                "which may lead to slower processing times and high memory consumption."
                            )
                        )
                findings.append(finding)

            if cols_pct > self.config.profiler.dimensionality_threshold:
                finding = Finding(
                        level='WARNING',
                        message=(
                            f"High-dimensionality detected. The number of features ({col_count}) is high relative "
                            f"to the number of rows ({row_count}), which significantly increases the risk of overfitting."
                        )
                    )
                findings.append(finding)
                    
            dataset_stats = DatasetStats(
                original_row_count=self.original_row_count,
                row_count=row_count,
                column_count=col_count,
                total_memory_usage_mb=self.total_memory_usage,
                duplicate_rows_count=duplicate_rows_count,
                column_pct=cols_pct,
                duplicates_pct=duplicates_pct
            )

            self.findings += findings
            return dataset_stats

        except Exception as e:
            raise ValueError(f"Failed to perform dataset level profiling. Reason: {e}")

    def _profile_column_level(self, df: pd.DataFrame) -> List[ColumnDetails]:
        """
        Iterates through each column, calculating and compiling its specific stats.
        """
        columns_details_list = []
        findings = []
        try:
            for column_name in df.columns:
                column = df[column_name]
                inferred_type = self._infer_logical_type(column)
                dtype = str(column.dtype)
                missing_values_count = column.isnull().sum()
                missing_values_pct = missing_values_count / column.shape[0]
                unique_values_count = column.nunique()
                unique_values_pct = unique_values_count / column.shape[0]
                memory_usage = self.column_memory_usage[column_name] / (1024**2)
                memory_usage_pct = memory_usage / self.total_memory_usage if self.total_memory_usage > 0 else 0.0

                stats = self._get_column_stats(column, inferred_type)

                if missing_values_pct > self.config.profiler.missing_value_threshold:
                    finding = Finding(
                        level='WARNING',
                        message=(f"Column '{column_name}' has a high percentage of missing values ({missing_values_pct:.1%}). " 
                                 "This may require advanced imputation or cause the feature to be unusable.")
                    )
                    findings.append(finding)

                if inferred_type in ['categorical', 'text']:
                    if unique_values_pct > self.config.profiler.high_cardinality_threshold:
                        finding = Finding(
                            level='WARNING',
                            message=(f"Column '{column_name}' has very high cardinality ({unique_values_pct:.1%} unique values). "
                                     "It may be an ID, a noisy text feature, or require special encoding.")
                        )
                        findings.append(finding)

                if unique_values_count == 1:
                    finding = Finding(
                        level='WARNING',
                        message=(f"Column '{column_name}' has only one unique value (zero variance) and provides no predictive power. "
                                 "It should be removed.")
                    )
                    findings.append(finding)

                if inferred_type == 'numeric':
                    skewness = column.skew()
                    if abs(skewness) > self.config.profiler.skewness_threshold:
                        finding = Finding(
                            level='WARNING',
                            message=(f"Column '{column_name}' is highly skewed (skewness: {skewness:.5f}). "
                                     "Consider applying a transformation (e.g., log, Box-Cox) for models that assume a normal distribution.")
                        )
                        findings.append(finding)

                if memory_usage_pct > self.config.profiler.memory_hog_threshold:
                    finding = Finding(
                        level='WARNING',
                        message=(f"Column '{column_name}' is a potential memory hog, consuming {memory_usage:.5f} MB "
                                 f"({memory_usage_pct:.1%}) of the total dataset memory. If this is a categorical column, "
                                 "consider converting its type to 'category' to optimize performance.")
                    )
                    findings.append(finding)

                column_details = ColumnDetails(
                    column_name=column_name,
                    inferred_type=inferred_type,
                    dtype=dtype,
                    missing_values_count=missing_values_count,
                    missing_values_pct=missing_values_pct,
                    unique_values_count=unique_values_count,
                    unique_values_pct=unique_values_pct,
                    stats=stats,
                    memory_usage_mb=memory_usage
                )

                columns_details_list.append(column_details)

            self.findings += findings
            return columns_details_list

        except Exception as e:
            raise ValueError(f"Failed to perform column level profiling. Reason: {e}")

    def _infer_logical_type(self, column: pd.Series) -> str:
        """
        Infers the logical type of a column (numeric, categorical, etc.)
        based on its dtype and statistics.
        """
        row_count = column.shape[0] 
        unique_values_count = column.nunique()
        unique_values_pct = unique_values_count / row_count
        sample_count = 500 if row_count > 500 else row_count
        column_sample = column.dropna().sample(n=min(sample_count, len(column.dropna())), random_state=1)

        def is_convertible_to_datetime(column) -> bool:
            try:
                pd.to_datetime(column)
                return True
            except Exception:
                return False
            
        def is_convertible_to_numerical(column):
            try:
                column.astype(float)
                return True
            except Exception:
                return False

        if is_bool_dtype(column): 
            return 'boolean'
        
        if (
            is_datetime64_any_dtype(column)
            or is_timedelta64_dtype(column)
            or is_period_dtype(column)
            or is_interval_dtype(column)
        ):
            return 'datetime'

        if unique_values_pct > 0.99: return 'id'

        if is_numeric_dtype(column):
            if unique_values_count == 2:
                return 'boolean'
            if unique_values_count <= self.config.profiler.max_categorical_cardinality:
                return 'categorical'
            return 'numerical'
        
        if is_convertible_to_datetime(column_sample):
            return 'datetime'
        
        if is_convertible_to_numerical(column_sample):
            if column_sample.nunique() == 2:
                return 'boolean'
            if unique_values_count <= self.config.profiler.max_categorical_cardinality:
                return 'categorical'
            return 'numerical'

        if unique_values_count <= self.config.profiler.max_categorical_cardinality:
            return 'categorical'

        return 'text'

    def _get_column_stats(self, column: pd.Series, inferred_type: str) -> dict:
        """
        Generates a dictionary of descriptive statistics tailored to the
        column's inferred logical type.
        """
        if inferred_type == 'numerical':
            stats_dict = column.describe().to_dict()
            return {key: float(value) for key, value in stats_dict.items()}

        elif inferred_type == 'categorical':
            value_counts = column.value_counts()
            
            if value_counts.empty:
                return {
                    "top": "N/A",
                    "frequency": 0,
                    "value_counts": {}
                }

            top_5_counts = value_counts.head(5).to_dict()
            
            return {
                "top": str(value_counts.index[0]),
                "frequency": int(value_counts.iloc[0]),
                "value_counts": {str(k): int(v) for k, v in top_5_counts.items()}
            }

        elif inferred_type == 'datetime':
            if not is_datetime64_any_dtype(column):
                column = pd.to_datetime(column, errors="coerce")
            
            return {
                "first": str(column.min()),
                "last": str(column.max()),
                "count": int(column.count())
            }

        return {}