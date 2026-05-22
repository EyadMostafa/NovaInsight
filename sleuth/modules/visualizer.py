from __future__ import annotations

import matplotlib
matplotlib.use('Agg')  # thread-safe non-interactive backend; must precede pyplot import
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from sleuth.utils.logger import get_logger

from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.feature_selection import f_classif, chi2

from sleuth.modules.base_module import BaseModule
from sleuth.modules.registry import register_module
from sleuth.config.config import SleuthConfig
from sleuth.exceptions import VisualizerError
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    VisualizationOutput,
    Finding,
    ColumnDetails,
    Operator,
)

logger = get_logger("sleuth.modules.visualizer")


@register_module(
    operator=Operator.VIZ,
    dependencies=[Operator.STATS, Operator.DIM_REDUCTION],
    is_completed=lambda report: report.visualizations is not None,
)
class Visualizer(BaseModule):
    """
    Generates a gallery of plots based on all previous modules.
    It is task-aware, changing its plotting (e.g., coloring) based on
    whether the task is 'supervised' or 'unsupervised'.
    """

    def __init__(self, config: SleuthConfig):
        """Initializes the Visualizer with app configuration."""
        super().__init__(config)
        self.output_dir: Optional[Path] = None
        self.findings: List[Finding] = []

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        Main execution method for the visualization process.

        Orchestrates the setup, task-aware hue generation, and creation
        of all plot groups (univariate, bivariate, correlation, dimensionality).
        """
        self._setup_plotting()
        self.output_dir = Path(report.metadata.output_dir) / 'plots'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        hue_data, hue_name = self._get_task_aware_hue(df, report)

        univariate_plots = self._run_univariate_plots(df, report)
        bivariate_plots = self._run_bivariate_plots(df, report, hue_data, hue_name)
        corr_plots = self._run_correlation_visuals(report)
        dim_reduction_plots = self._run_dimensionality_visuals(report, hue_data, hue_name)
        
        viz_output = VisualizationOutput(
            univariate_plots=univariate_plots,
            bivariate_plots=bivariate_plots,
            correlation_heatmap_plots=corr_plots,
            dim_reduction_scatter_plots=dim_reduction_plots,
            findings=self.findings
        )

        report.visualizations = viz_output

        return report

    def _setup_plotting(self) -> None:
        """
        Applies consistent global styling (theme, palette, DPI)
        from the config to seaborn and matplotlib.
        """
        try:
            sns.set_theme(
                style=self.config.visualization.theme,
                palette=self.config.visualization.color_palette
            )

            plt.rcParams['figure.dpi'] = self.config.visualization.dpi
            plt.rcParams['figure.figsize'] = (10, 6)
            
            logger.debug(f"Seaborn theme set to '{self.config.visualization.theme}'")
            
        except Exception as e:
            logger.warning(f"Failed to apply visualization theme: {e}. Using defaults.")

            finding = Finding(
                level='ERROR',
                message=f"Could not apply visualization theme: {e}. Plots may use default styling."
            )

            self.findings.append(finding)

    def _save_plot(self, fig: plt.Figure, filename: str) -> Optional[str | Path]:
        """
        Utility to save a matplotlib figure to the output_dir.
        Returns the path or None if saving fails.
        Handles closing the figure to conserve memory.
        """
        if self.output_dir is None:
            logger.error("Output directory is not set in Visualizer. Cannot save plot.")
            return None

        full_path = None
        try:
            if not filename.endswith('.png'):
                filename = f"{filename}.png"

            full_path = self.output_dir / filename
            
            fig.savefig(full_path, bbox_inches='tight')
            
            plt.close(fig)
            
            return full_path
            
        except Exception as e:
            logger.error(f"Failed to save plot {filename}: {e}")
            
            finding = Finding(
                level="WARNING",
                message=f"Failed to generate plot: {filename}. Reason: {e}"
            )

            self.findings.append(finding)
            
            plt.close(fig)
            
            return None

    def _get_task_aware_hue(self, df: pd.DataFrame, report: AnalysisReport) -> Tuple[Optional[pd.Series], str]:
        """
        Determines what data to use for coloring plots ('hue').

        - Supervised task: Returns the target variable series.
        - Unsupervised task: Runs K-Means and returns the cluster labels.
        """
        hue_data: Optional[pd.Series] = None
        hue_name: str = "N/A"

        if report.metadata.task == 'supervised':
            if report.target_analysis and report.target_analysis.identified_target:
                target_name = report.target_analysis.identified_target
                if target_name in df.columns:
                    hue_data = df[target_name]
                    hue_name = target_name
                else:
                    logger.warning(f"Supervised task specified but target '{target_name}' not found in DataFrame for hue.")
                    finding = Finding(
                        level="WARNING",
                        message=f"Supervised task specified but target '{target_name}' not found. Cannot color plots by target."
                    )
                    self.findings.append(finding)
            else:
                logger.warning("Supervised task specified but no target identified. Will attempt unsupervised clustering for hue.")
                finding = Finding(
                    level="INFO",
                    message="Supervised task specified but no target identified. Attempting K-Means for plot coloring."
                )
                self.findings.append(finding)

        if hue_data is None:
            logger.info("Performing K-Means clustering for plot coloring.")
            try:
                hue_data = self._perform_clustering(df, report)
                hue_name = "Discovered Cluster"
            except Exception as e:
                logger.error(f"K-Means clustering failed for hue data: {e}")
                finding = Finding(
                    level="ERROR",
                    message=f"K-Means clustering for plot coloring failed: {e}. Plots will not be colored."
                )
                self.findings.append(finding)
                hue_data = None
                hue_name = "N/A"

        return hue_data, hue_name

    def _perform_clustering(self, df: pd.DataFrame, report: AnalysisReport) -> pd.Series:
        """
        Runs K-Means clustering on the data to find groups for unsupervised viz.
        """
        try:
            identified_target = report.target_analysis.identified_target if report.target_analysis else None
            
            initial_numeric_columns = [
                col.column_name for col in report.profile.column_details
                if col.inferred_type == 'numerical' and col.column_name != identified_target
            ]

            if not initial_numeric_columns:
                logger.warning("No numeric features found for K-Means clustering.")
                return pd.Series(index=df.index, name="Cluster").fillna(0)

            df_numeric_initial = df[initial_numeric_columns].copy()

            valid_numeric_columns = []
            for col_name in initial_numeric_columns:
                if df_numeric_initial[col_name].nunique(dropna=False) > 1:
                    valid_numeric_columns.append(col_name)

            if not valid_numeric_columns:
                logger.warning("No numeric features with variance found for K-Means clustering.")
                return pd.Series(index=df.index, name="Cluster").fillna(0)

            df_numeric_valid = df_numeric_initial[valid_numeric_columns]

            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy=self.config.dimensionality_reduction.imputation_method)),
                    ('scaler', StandardScaler())
                ])

            processed_data = num_pipeline.fit_transform(df_numeric_valid)

            n_samples = processed_data.shape[0]
            n_clusters_config = self.config.visualization.kmeans_n_clusters
            actual_n_clusters = max(1, min(n_clusters_config, n_samples))
            
            if actual_n_clusters < n_clusters_config:
                logger.warning(f"K-Means n_clusters reduced to {actual_n_clusters} (number of samples).")

            kmeans = KMeans(
                n_clusters=actual_n_clusters,
                random_state=42,
                n_init=10
            )
            labels = kmeans.fit_predict(processed_data)

            return pd.Series(labels, index=df.index, name="Discovered Cluster")

        except Exception as e:
            raise VisualizerError(f"K-Means clustering failed: {e}") from e

    # --- Plot Generation Groups ---

    def _run_univariate_plots(self, df: pd.DataFrame, report: AnalysisReport) -> Dict[str, str | Path]:
        """
        Generates and saves all univariate plots (histograms, count plots).
        Returns a dictionary mapping column_name to plot_path.
        """
        plot_paths: Dict[str, str] = {}

        for col_details in report.profile.column_details:
            col_name = col_details.column_name
            
            if col_details.inferred_type in ['id', 'text', 'datetime']:
                continue 

            plot_func = None
            if col_details.inferred_type == 'numerical':
                plot_func = self._plot_histogram
            elif col_details.inferred_type in ['categorical', 'boolean']:
                plot_func = self._plot_countplot
            
            if plot_func is None:
                continue

            try:
                fig, ax = plt.subplots(figsize=(10, 6))
                plot_func(ax, df[col_name], col_name)
                
                filename = f"univariate_dist_{col_name}"
                path = self._save_plot(fig, filename)
                
                if path:
                    plot_paths[col_name] = path
            except Exception as e:
                logger.error(f"Failed to generate univariate plot for {col_name}: {e}")
                finding = Finding(
                    level="ERROR",
                    message=f"Failed to generate plot for {col_name}: {e}"
                )
                self.findings.append(finding)
                plt.close('all')
                
        return plot_paths

    def _run_bivariate_plots(self, df: pd.DataFrame, report: AnalysisReport, hue_data: pd.Series, hue_name: str) -> Dict[str, str | Path]:
        """
        Generates and saves key feature-vs-target/cluster plots.
        Returns a dictionary mapping plot_name to plot_path.
        """
        plot_paths: Dict[str, str] = {}
        if hue_data is None:
            logger.info("Skipping bivariate plots as no hue (target/cluster) data is available.")
            return plot_paths

        hue_is_numeric = pd.api.types.is_numeric_dtype(hue_data) and hue_data.nunique() > self.config.target_detection.max_categorical_cardinality

        all_features = [
            col for col in report.profile.column_details
            if col.column_name != hue_name and col.inferred_type not in ['id', 'text', 'datetime']
        ]
        
        top_n = min(self.config.visualization.bivariate_top_n, len(all_features))
        top_n = len(all_features) if top_n == -1 else top_n # -1 selects all elligible features

        selected_features: List[ColumnDetails] = []

        try:
            if report.metadata.task == 'supervised':
                selected_features = self._get_top_correlated_features(report, hue_name, all_features, top_n)
            else:
                selected_features = self._get_top_associated_features(df, hue_data, all_features, top_n)
        except Exception as e:
            logger.error(f"Failed to select top features for bivariate plots: {e}")
            self.findings.append(Finding(level="WARNING", message=f"Failed to select top features for bivariate plots: {e}"))
            return plot_paths # Exit if selection fails

        # 4. Loop over selected features and generate plots
        for col in selected_features:
            fig, ax = plt.subplots(figsize=(10, 6))
            plot_func: Optional[callable] = None
            
            try:
                feature_data = df[col.column_name]
                
                if col.inferred_type == 'numerical' and not hue_is_numeric: # num - cat
                    plot_func = self._plot_violinplot
                    plot_func(ax, hue_data, feature_data, hue_name, col.column_name)
                elif col.inferred_type == 'numerical' and hue_is_numeric: # num - num
                    plot_func = self._plot_regplot
                    plot_func(ax, feature_data, hue_data, col.column_name, hue_name)
                elif col.inferred_type in ['categorical', 'boolean'] and not hue_is_numeric: # cat - cat
                    plot_func = self._plot_grouped_countplot
                    plot_func(ax, feature_data, hue_data, col.column_name, hue_name)
                elif col.inferred_type in ['categorical', 'boolean'] and hue_is_numeric: # cat - num
                    plot_func = self._plot_violinplot
                    plot_func(ax, feature_data, hue_data, col.column_name, hue_name) # Flipped
                
                if plot_func is None:
                    plt.close(fig)
                    continue
                    
                filename = f"bivariate_{col.column_name}_vs_{hue_name}"
                path = self._save_plot(fig, filename)
                if path:
                    plot_paths[f"{col.column_name}_vs_{hue_name}"] = path
                    
            except Exception as e:
                logger.error(f"Failed to generate bivariate plot for {col.column_name} vs {hue_name}: {e}")
                self.findings.append(Finding(
                    level="WARNING",
                    message=f"Failed to generate plot for {col.column_name} vs {hue_name}: {e}"
                ))
                plt.close(fig) # Ensure figure is closed on error
        
        return plot_paths
    
    def _run_correlation_visuals(self, report: AnalysisReport) -> Dict[str, str | Path]:
        """
        Generates and saves heatmaps for the correlation matrices.
        Returns a dictionary mapping heatmap_name to plot_path.
        """
        plot_paths: Dict[str, str] = {}
        if not report.statistical_analysis.correlation_report:
            logger.info("Correlation report not found. Skipping correlation heatmaps visualization.")
            finding = Finding(
                level='INFO',
                message="Correlation report not found, skipping correlation heatmaps visualization."
            )
            self.findings.append(finding)
            return plot_paths

        corr_report = report.statistical_analysis.correlation_report.model_dump()
        clean_name = ""

        try:
            for name, path in corr_report.items():
                matrix = pd.read_csv(path, index_col=0)
                if matrix.empty: continue

                clean_name = name.replace('_path', '')

                fig, ax = plt.subplots(figsize=(10, 8))
                title = clean_name.replace('_', ' ').title() + " Correlation"
                self._plot_heatmap(ax=ax, matrix=matrix, title=title)

                filename = f"{clean_name}_heatmap"
                viz_path = self._save_plot(fig, filename)
                if viz_path:
                    plot_paths[clean_name] = viz_path

        except Exception as e:
            logger.error(f"Failed to generate heatmap plot for {clean_name}: {e}")
            self.findings.append(Finding(
                level="WARNING",
                message=f"Failed to generate heatmap plot for {clean_name}: {e}"
            ))
            plt.close('all')

        return plot_paths

    def _run_dimensionality_visuals(self, report: AnalysisReport, hue_data: pd.Series, hue_name: str) -> Dict[str, str | Path]:
        """
        Generates 2D scatter plots for PCA, UMAP, and t-SNE embeddings.
        Returns a dictionary mapping plot_name to plot_path.
        """
        plot_paths: Dict[str, str] = {}
        dimensionality_report = report.dimensionality_analysis.model_dump()
        clean_name = ""

        if not dimensionality_report.items():
            logger.info("Dimensionality analysis report not found. Skipping embeddings visualization.")
            finding = Finding(
                level='INFO',
                message="Dimensionality analysis report not found. Skipping embeddings visualization."
            )
            self.findings.append(finding)
            return plot_paths

        try:
            for k, v in dimensionality_report.items():
                if not "_embeddings_path" in k or not v: continue

                embeddings = pd.read_csv(v)
                if embeddings.empty or embeddings.shape[1] < 2: continue

                fig, ax = plt.subplots(figsize=(10, 8))
                clean_name = k.replace("_path", "")
                title = clean_name.replace('_', ' ').title() + " Scatter Plot"
                self._plot_embedding_scatter(ax, embeddings, title, hue_data, hue_name)

                filename = f"{clean_name}_scatterplot"
                viz_path = self._save_plot(fig, filename)
                if viz_path:
                    plot_paths[clean_name] = viz_path

        except Exception as e:
            logger.error(f"Failed to generate scatter plot for {clean_name}: {e}")
            self.findings.append(Finding(
                level="WARNING",
                message=f"Failed to generate scatter plot for {clean_name}: {e}"
            ))
            plt.close('all')

        return plot_paths

    def _plot_histogram(self, ax: plt.Axes, data: pd.Series, title: str):
        """Generates a histogram and KDE plot on the given axes."""
        data = data.dropna()
        if data.empty:
            ax.text(0.5, 0.5, "No data to plot (all values are NaN)", 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=ax.transAxes, color='gray')
            ax.set_title(f"Distribution of {title}", fontsize=16)
            return

        sns.histplot(data, kde=True, ax=ax)
        ax.set_title(f"Distribution of {title}", fontsize=16)
        ax.set_xlabel(title)
        ax.set_ylabel("Frequency")

    def _plot_countplot(self, ax: plt.Axes, data: pd.Series, title: str):
        """Generates a count plot for categorical data on the given axes."""
        data = data.dropna()

        unique_count = data.nunique()
        n_cats = min(self.config.visualization.univariate_countplot_n, unique_count)
        n_cats = 20 if n_cats > 20 else n_cats 

        top_n_counts = data.value_counts().head(n_cats)
        sns.barplot(x=top_n_counts.index, y=top_n_counts.values, ax=ax, order=top_n_counts.index)
        ax.set_title(f"Count of {title}")
        ax.set_ylabel("Count")
        ax.tick_params(axis='x', rotation=90)
        
    def _plot_heatmap(self, ax: plt.Axes, matrix: pd.DataFrame, title: str):
        """Generates a correlation heatmap on the given axes."""
        sns.heatmap(
            data=matrix, 
            ax=ax, 
            cmap='coolwarm', 
            center=0, 
            annot=True, 
            fmt=".2f", 
            linewidths=0.5
        )
        ax.set_title(title)

    def _plot_embedding_scatter(self, ax: plt.Axes, embeddings: pd.DataFrame, title: str, hue_data: pd.Series, hue_name: str):
        """Generates a 2D scatter plot for embeddings on the given axes."""
        names = embeddings.columns

        embeddings = embeddings.loc[hue_data.index]
        plot_data = pd.concat([embeddings, hue_data], axis=1)
        
        sns.scatterplot(
            data=plot_data, 
            x=names[0], 
            y=names[1], 
            hue=hue_name, 
            ax=ax,
            alpha=0.7,
            palette='viridis' if pd.api.types.is_numeric_dtype(hue_data) else 'tab10'
        )
        ax.set_title(title)

    def _plot_regplot(self, ax: plt.Axes, y_data: pd.Series, x_data: pd.Series, y_name: str, x_name: str):
            """Generates a regression plot (scatter + linear fit) for two numeric variables."""
            plot_data = pd.concat([x_data, y_data], axis=1).dropna()
            if plot_data.empty:
                return

            sns.regplot(
                data=plot_data, x=x_name, y=y_name, ax=ax,
                scatter_kws={'alpha': 0.5, 's': 30}, line_kws={'color': 'red'}
            )
            ax.set_title(f"{x_name} vs {y_name} (with Regression Trend)", fontsize=14)

    def _plot_violinplot(self, ax: plt.Axes, x_data: pd.Series, y_data: pd.Series, x_name: str, y_name: str):
            """
            Generates a violin plot to show the distribution of a numeric variable 
            across levels of a categorical variable.
            """
            plot_data = pd.concat([x_data, y_data], axis=1).dropna()
            if plot_data.empty:
                return
            
            # Current logic assumes x_data is the categorical one based on the caller
            if x_data.nunique() > 20:
                top_cats = x_data.value_counts().head(20).index
                plot_data = plot_data[plot_data[x_name].isin(top_cats)]

            sns.violinplot(data=plot_data, x=x_name, y=y_name, hue=x_name, ax=ax, inner="box", palette='muted', legend=False)
            
            if x_data.nunique() > 4 or x_data.astype(str).str.len().mean() > 8:
                ax.tick_params(axis='x', rotation=45)
    
            ax.set_title(f"Distribution of {y_name} vs {x_name}", fontsize=14)

    def _plot_grouped_countplot(self, ax: plt.Axes, feature_data: pd.Series, hue_data: pd.Series, feature_name: str, hue_name: str):
            """
            Generates a grouped bar chart for two categorical variables with value labels.
            """
            df_temp = pd.DataFrame({feature_name: feature_data, hue_name: hue_data}).dropna()
            if df_temp.empty:
                return

            counts = df_temp.groupby([feature_name, hue_name]).size().reset_index(name='count')

            feature_totals = counts.groupby(feature_name)['count'].sum().sort_values(ascending=False)

            # Limit to top 15 categories of the feature
            if len(feature_totals) > 15:
                top_features = feature_totals.head(15).index
                counts = counts[counts[feature_name].isin(top_features)]
                feature_totals = feature_totals.head(15)

            order = feature_totals.index

            # 3. Plot grouped bar chart
            sns.barplot(
                data=counts, 
                x=feature_name, 
                y='count', 
                hue=hue_name, 
                ax=ax, 
                order=order
            )

            for container in ax.containers:
                ax.bar_label(container, fmt='%d', padding=3, fontsize=10, rotation=90)

            ax.set_title(f"{feature_name} by {hue_name}", fontsize=14)
            ax.set_xlabel(feature_name)
            ax.tick_params(axis='x', rotation=45)
            ax.legend(title=hue_name, bbox_to_anchor=(1.05, 1), loc='upper left')

    def _get_top_correlated_features(self, report: AnalysisReport, target_name: str, all_features: List[ColumnDetails], top_n: int) -> List[ColumnDetails]:
        """(Helper for Bivariate) Gets top N features correlated with the target."""
        all_correlations = {}
        corr_report = report.statistical_analysis.correlation_report
        
        def read_corr_matrix(path_str: str) -> Optional[pd.DataFrame]:
            if not path_str: return None
            try:
                return pd.read_csv(path_str, index_col=0)
            except Exception as e:
                logger.warning(f"Could not read correlation matrix at {path_str}: {e}")
                return None

        df_num_num = read_corr_matrix(corr_report.numerical_numerical_path)
        df_cat_num = read_corr_matrix(corr_report.categorical_numerical_path)
        df_cat_cat = read_corr_matrix(corr_report.categorical_categorical_path)

        if df_num_num is not None and target_name in df_num_num.columns:
            all_correlations.update(df_num_num[target_name].to_dict())
        if df_cat_num is not None:
            if target_name in df_cat_num.columns: # Target is numeric
                all_correlations.update(df_cat_num[target_name].to_dict())
            if target_name in df_cat_num.index: # Target is categorical
                all_correlations.update(df_cat_num.loc[target_name].to_dict())
        if df_cat_cat is not None and target_name in df_cat_cat.columns:
            all_correlations.update(df_cat_cat[target_name].to_dict())
            
        if not all_correlations:
            logger.info("No correlation data found for target, cannot select top features.")
            return []

        corr_series = pd.Series(all_correlations).drop(target_name, errors='ignore').abs()
        selected_feature_names = corr_series.nlargest(top_n).index
        
        return [col for col in all_features if col.column_name in selected_feature_names]

    def _get_top_associated_features(self, df: pd.DataFrame, hue_data: pd.Series, all_features: List[ColumnDetails], top_n: int) -> List[ColumnDetails]:
        """
        (Helper for Bivariate) Gets top N features associated with clusters
        using a robust p-value filter and test statistic ranking.
        """
        numeric_features = [col.column_name for col in all_features if col.inferred_type == 'numerical']
        categorical_features = [col.column_name for col in all_features if col.inferred_type in ['categorical', 'boolean']]
        
        feature_stats = []
        
        p_value_threshold = self.config.statistics.bivariate_significance_level
        
        hue_data_clean = hue_data.dropna()
        df_aligned = df.loc[hue_data_clean.index]

        if numeric_features:
            X_numeric = df_aligned[numeric_features]
            X_numeric_imputed = SimpleImputer(strategy='median').fit_transform(X_numeric)
            
            variance = X_numeric_imputed.var(axis=0)
            valid_numeric_features = [col for i, col in enumerate(numeric_features) if variance[i] > 1e-6]
            X_numeric_final = X_numeric_imputed[:, variance > 1e-6]

            if X_numeric_final.shape[1] > 0:
                f_scores, p_values = f_classif(X_numeric_final, hue_data_clean)
                for i, col_name in enumerate(valid_numeric_features):
                    feature_stats.append({'name': col_name, 'score': f_scores[i], 'pvalue': p_values[i]})

        if categorical_features:
            X_categorical = df_aligned[categorical_features]
            X_cat_imputed = X_categorical.apply(lambda col: col.fillna(col.mode()[0] if not col.mode().empty else 'missing'))
            
            valid_categorical_features = [col for col in categorical_features if X_cat_imputed[col].nunique() > 1]
            if valid_categorical_features:
                X_cat_encoded = X_cat_imputed[valid_categorical_features].apply(lambda col: OrdinalEncoder().fit_transform(col.to_frame())[:, 0])
            
                if X_cat_encoded.shape[1] > 0:
                    chi2_scores, p_values = chi2(X_cat_encoded, hue_data_clean)
                    for i, col_name in enumerate(valid_categorical_features):
                         feature_stats.append({'name': col_name, 'score': chi2_scores[i], 'pvalue': p_values[i]})

        if not feature_stats:
            return []
            
        significant_features = [f for f in feature_stats if f['pvalue'] < p_value_threshold]

        significant_features.sort(key=lambda x: x['score'], reverse=True)
        
        selected_feature_names = [f['name'] for f in significant_features[:top_n]]
        
        return [col for col in all_features if col.column_name in selected_feature_names]