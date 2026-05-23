from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from umap import UMAP

from sleuth.config.config import SleuthConfig
from sleuth.exceptions import DimensionalityReducerError
from sleuth.modules.base_module import BaseModule
from sleuth.modules.registry import register_module
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    DimensionalityAnalysis,
    Finding,
    Operator,
)


@register_module(
    operator=Operator.DIM_REDUCTION,
    dependencies=[Operator.PROFILER],
    is_completed=lambda report: report.dimensionality_analysis is not None,
)
class DimensionalityReducer(BaseModule):
    """
    Finds and visualizes latent data structures by running dimensionality
    reduction algorithms (PCA, UMAP, t-SNE) on the numeric features.
    """

    def __init__(self, config: SleuthConfig):
        """Initializes the DimensionalityReducer with app configuration."""
        super().__init__(config)

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        Main execution method for the dimensionality reduction process.
        It prepares the data, runs PCA, UMAP, and t-SNE, saves the results,
        and updates the main analysis report.
        """
        report.dimensionality_analysis = DimensionalityAnalysis()

        processed_df, prep_finding = self._prepare_data(df, report)
        if processed_df is None:
            if prep_finding:
                report.dimensionality_analysis.findings.append(prep_finding)
            return report
        if prep_finding:
             report.dimensionality_analysis.findings.append(prep_finding)

        pca_embeddings, variance_ratios = self._run_pca(processed_df)
        tsne_embeddings = self._run_tsne(processed_df)
        umap_embeddings = self._run_umap(processed_df)

        output_dir = Path(report.metadata.output_dir)

        pca_embeddings_path = self._save_results(pca_embeddings, output_dir, 'pca')
        tsne_embeddings_path = self._save_results(tsne_embeddings, output_dir, 'tsne')
        umap_embeddings_path = self._save_results(umap_embeddings, output_dir, 'umap')

        report.dimensionality_analysis.pca_explained_variance_ratios = variance_ratios
        report.dimensionality_analysis.pca_embeddings_path = pca_embeddings_path
        report.dimensionality_analysis.tsne_embeddings_path = tsne_embeddings_path
        report.dimensionality_analysis.umap_embeddings_path = umap_embeddings_path

        return report

    def _prepare_data(self, df: pd.DataFrame, report: AnalysisReport) -> tuple[pd.DataFrame | None, Finding | None]:
        """
        Selects, filters (for zero variance), imputes, and scales the numeric
        columns from the dataset. Returns a Finding if skipping or filtering.
        """
        try:
            identified_target = report.target_analysis.identified_target if report.target_analysis else None

            initial_numeric_columns = [
                col.column_name for col in report.profile.column_details
                if col.inferred_type == 'numerical' and col.column_name != identified_target
            ]

            if not initial_numeric_columns:
                finding = Finding(
                    level='INFO',
                    message='No numeric features found (excluding target) for dimensionality reduction. Skipping.'
                )
                return None, finding

            df_numeric_initial = df[initial_numeric_columns].copy()

            zero_variance_cols = []
            valid_numeric_columns = []
            for col_name in initial_numeric_columns:
                if df_numeric_initial[col_name].nunique(dropna=False) <= 1:
                    zero_variance_cols.append(col_name)
                else:
                    valid_numeric_columns.append(col_name)

            zero_var_finding = None
            if zero_variance_cols:
                zero_var_finding = Finding(
                    level='INFO',
                    message=(f"Removed constant (zero-variance) numeric columns before dimensionality reduction: "
                             f"{', '.join(zero_variance_cols)}. ")
                )

            final_numeric_count = len(valid_numeric_columns)
            if final_numeric_count < 3:
                skip_finding = Finding(
                    level='INFO',
                    message=(f"Dataset has fewer than 3 valid numeric features ({final_numeric_count}) "
                             f"after excluding non-numerical, target, and zero-variance columns. Skipping Dimensionality reduction.")
                )
                return None, skip_finding

            df_numeric_valid = df_numeric_initial[valid_numeric_columns]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy=self.config.dimensionality_reduction.imputation_method)),
                    ('scaler', StandardScaler())
                ])

            processed_data = num_pipeline.fit_transform(df_numeric_valid)
            df_transformed = pd.DataFrame(processed_data, columns=valid_numeric_columns, index=df_numeric_valid.index)

            return df_transformed, zero_var_finding

        except Exception as e:
            raise DimensionalityReducerError(f"Failed to prepare data for dimensionality reduction. Reason: {e}") from e

    def _run_pca(self, processed_df: pd.DataFrame) -> tuple[pd.DataFrame, list[float]]:
        """
        Performs Principal Component Analysis (PCA) on the processed data.
        Dynamically adjusts n_components based on data shape.
        """
        try:
            n_samples, n_features = processed_df.shape
            max_possible_components = min(n_samples, n_features)

            n_ratios_desired = self.config.dimensionality_reduction.pca_n_ratios
            actual_n_ratios = min(n_ratios_desired, max_possible_components)

            if actual_n_ratios <= 0:
                return pd.DataFrame(index=processed_df.index), []

            pca_variance = PCA(n_components=actual_n_ratios)
            pca_variance.fit(processed_df)
            ratios = [float(r) for r in pca_variance.explained_variance_ratio_]

            n_components_embedding = min(2, max_possible_components)
            if n_components_embedding <= 0:
                return pd.DataFrame(index=processed_df.index), ratios

            embedding_data = pca_variance.transform(processed_df)[:, :n_components_embedding]
            embedding_cols = [f'PC{i+1}' for i in range(n_components_embedding)]
            embeddings_df = pd.DataFrame(embedding_data, columns=embedding_cols, index=processed_df.index)

            return embeddings_df, ratios

        except Exception as e:
            raise DimensionalityReducerError(f"Failed to perform PCA transformation. Reason: {e}") from e

    def _run_tsne(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs t-Distributed Stochastic Neighbor Embedding (t-SNE).
        """
        try:
            n_samples = processed_df.shape[0]
            perplexity_val = min(self.config.dimensionality_reduction.tsne_perplexity, max(1, (n_samples - 1) / 3.1))

            tsne = TSNE(
                n_components=2,
                perplexity=perplexity_val,
                learning_rate=self.config.dimensionality_reduction.tsne_learning_rate,
                max_iter=self.config.dimensionality_reduction.tsne_max_iter,
                init='pca',
                random_state=42
            )

            embeddings_data = tsne.fit_transform(processed_df)
            embedding_cols = [f'TSNE{i+1}' for i in range(embeddings_data.shape[1])]
            embeddings_df = pd.DataFrame(embeddings_data, columns=embedding_cols, index=processed_df.index)

            return embeddings_df
        except Exception as e:
            raise DimensionalityReducerError(f"Failed to perform t-SNE transformation. Reason: {e}") from e

    def _run_umap(self, processed_df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs Uniform Manifold Approximation and Projection (UMAP).
        """
        try:
            n_samples = processed_df.shape[0]
            n_neighbors_val = min(self.config.dimensionality_reduction.umap_n_neighbors, n_samples - 1)
            n_neighbors_val = max(2, n_neighbors_val)

            umap_reducer = UMAP(
                n_components=2,
                n_neighbors=n_neighbors_val,
                min_dist=self.config.dimensionality_reduction.umap_min_dist,
                metric='euclidean',
                random_state=42
            )

            embeddings_data = umap_reducer.fit_transform(processed_df)
            embedding_cols = [f'UMAP{i+1}' for i in range(embeddings_data.shape[1])]
            embeddings_df = pd.DataFrame(embeddings_data, columns=embedding_cols, index=processed_df.index)

            return embeddings_df
        except Exception as e:
            raise DimensionalityReducerError(f"Failed to perform UMAP transformation. Reason: {e}") from e

    def _save_results(self, embeddings_df: pd.DataFrame, output_dir: Path | str, filename_prefix: str) -> str:
        """
        Saves the resulting embeddings DataFrame to a CSV file. Returns str path.
        """
        file_path = None
        try:
            output_dir_path = Path(output_dir)
            final_output_dir = output_dir_path / "embeddings"
            final_output_dir.mkdir(parents=True, exist_ok=True)

            file_path = final_output_dir / f"{filename_prefix}_embeddings.csv"

            embeddings_df.to_csv(file_path, index=(not embeddings_df.index.equals(pd.RangeIndex(start=0, stop=len(embeddings_df), step=1))))

            return file_path
        except Exception as e:
            raise DimensionalityReducerError(f"Failed to save results to {str(file_path) if file_path else filename_prefix}. Reason: {e}") from e

