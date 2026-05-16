from __future__ import annotations

import pandas as pd
import openpyxl
import csv
from logging import getLogger
from pathlib import Path
from typing import List

import novainsight.modules  # noqa: F401 — triggers @register_module decorators
from novainsight.config.config import NovaInsightConfig
from novainsight.exceptions import (
    ModuleError, PipelineError,
    DataLoadError, OrchestratorError,
)
from novainsight.modules.registry import get_registry, topological_order
from novainsight.schemas.analysis_report import AnalysisReport, RunMetadata, DatasetProfile, DatasetStats, Finding, Operator
from novainsight.modules.report_generator import ReportGenerator
from novainsight.utils.cache_manager import CacheManager
from novainsight.utils.validators import validate_file_path, validate_directory

logger = getLogger(__name__)


class AnalysisPipeline:
    """The main orchestrator for the NovaInsight analysis pipeline."""

    SUPPORTED_FILE_EXTENSIONS = ['.csv', '.xlsx', '.xls']

    def __init__(
        self,
        file_path: Path,
        config: NovaInsightConfig,
        output_dir: Path | None = None,
        requested_modules: List[str] | None = None,
        force_rerun: bool = False,
        user_target: str | None = None,
        analysis_mode: str | None = None,
        report_title: str | None = None,
        task: str | None = 'supervised',
        advanced: bool = False
    ):
        """Initializes the pipeline with all necessary context from the CLI."""
        self.file_path = file_path
        self.config = config
        self.output_dir = output_dir or self.config.analysis.output.default_directory
        self.force_rerun = force_rerun
        self.user_target = user_target
        self.analysis_mode = analysis_mode or self.config.analysis.default_mode
        self.report_title = report_title
        self.task = task
        self.advanced = advanced
        
        self.cache_manager = CacheManager(config.cache)
        self.report: AnalysisReport | None = None
        self.df: pd.DataFrame | None = None
        
        self.execution_plan = self._resolve_execution_plan(requested_modules)

    def run(self):
        """Executes the full, resolved analysis pipeline from start to finish."""
        try:
            self._initialize_report_and_data()
            self._run_analytical_modules()
            self._generate_outputs()

            report_generator = ReportGenerator(config=self.config)
            report_generator.run(self.df, self.report)

            logger.info("Analysis pipeline completed")
        except PipelineError:
            raise
        except Exception as e:
            raise OrchestratorError(f"Unexpected fatal error in pipeline: {e}") from e

    def _resolve_execution_plan(self, requested: List[str] | None) -> List[Operator]:
        """
        Calculates the full ordered list of operators to run.

        If specific modules were requested, walks their transitive dependencies
        and returns them in topological order. Otherwise returns all registered
        modules in topological order.

        The TARGET operator is silently excluded for unsupervised tasks.
        """
        registry = get_registry()
        full_order = topological_order()

        if not requested:
            plan = [op for op in full_order if not (self.task == 'unsupervised' and op == Operator.TARGET)]
            logger.info(f"Execution plan resolved: {plan}")
            return plan

        # Resolve transitive deps for the explicitly requested subset
        requested_ops: set[Operator] = set()
        for name in requested:
            try:
                op = Operator(name) if isinstance(name, str) else name
            except ValueError:
                logger.warning(f"Unknown module '{name}' requested. Ignoring.")
                continue
            if op not in registry:
                logger.warning(f"Module '{op}' is not registered. Ignoring.")
                continue
            requested_ops.add(op)

        final: set[Operator] = set()

        def collect(op: Operator) -> None:
            if op in final:
                return
            if self.task == 'unsupervised' and op == Operator.TARGET:
                return
            final.add(op)
            for dep in registry[op].dependencies:
                collect(dep)

        for op in requested_ops:
            collect(op)

        plan = [op for op in full_order if op in final]
        logger.info(f"Execution plan resolved: {plan}")
        return plan

    def _initialize_report_and_data(self):
        """
        Validates paths, loads the dataset (with sampling if in fast mode),
        and initializes the AnalysisReport object from cache or from scratch.
        """
        self.file_path = validate_file_path(self.file_path)
        file_extension = self.file_path.suffix.lower() 
        if file_extension not in self.SUPPORTED_FILE_EXTENSIONS:
            raise DataLoadError(f"Unsupported file extension '{file_extension}'. Supported: {', '.join(self.SUPPORTED_FILE_EXTENSIONS)}")

        self.output_dir = Path(self.output_dir / "novainsight_reports") 
        is_valid, message, resolved_path = validate_directory(self.output_dir)
        if not is_valid:
            raise OrchestratorError(f"Invalid output directory. Reason: {message}")
        
        file_hash = self.cache_manager.hash_file(self.file_path)

        if self.force_rerun:
            logger.info("Force rerun requested. Clearing any existing cache for this dataset.")
            self.cache_manager.clear_workspace(file_hash)
            self.report = None
        else:
            self.report = self.cache_manager.load_report(file_hash)

        self.df = self._load_data_frame(file_extension)

        if not self.report:
            logger.info("No valid cache found. Initializing a new analysis report.")
            
            if not self.report_title:
                self.report_title = self._generate_report_title()

            self.output_dir = Path(resolved_path / self.report_title)
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            full_row_count = self.get_full_row_count(file_extension)

            initial_stats = DatasetStats(
                original_row_count=full_row_count,
                # The actual row_count will be filled by the profiler from the provided df
                analyzed_row_count=0, 
                column_count=0, 
                total_memory_usage_mb=0.0, 
                duplicate_rows_count=0, 
                column_pct=0.0,
                duplicates_pct=0.0
            )
            
            initial_profile = DatasetProfile(dataset_stats=initial_stats, column_details=[], findings=[])

            metadata = RunMetadata(
                input_file=self.file_path,
                output_dir=self.output_dir,
                file_hash=file_hash,
                report_title=self.report_title,
                analysis_mode=self.analysis_mode,
                task=self.task,
                user_target=self.user_target
            )

            self.report = AnalysisReport(
                metadata=metadata,
                profile=initial_profile, # The profiler will return a completed instance of this object
                findings=[]
            )
        else:
            self.output_dir = Path(self.report.metadata.output_dir)
        
        logger.info("Analysis report initialized successfully.")

    def _run_analytical_modules(self):
        """
        Iterates through the execution plan and runs each analytical module
        sequentially, with graceful error handling for module-level failures.
        """
        registry = get_registry()
        skip_modules: set[Operator] = set()

        for op in self.execution_plan:
            if op in skip_modules:
                self.report.findings.append(Finding(
                    level='WARNING',
                    message=f"{op.capitalize()} module skipped due to the failure or skipping of a dependency module."
                ))
                continue

            spec = registry[op]
            try:
                if not self.force_rerun and spec.is_completed(self.report):
                    logger.info(f"Skipping {op}: valid results found in cache.")
                    continue
                updated_report = spec.cls(self.config).run(df=self.df, report=self.report)
                if updated_report:
                    self.report = updated_report

            except ModuleError as e:
                msg = f"{op.capitalize()} module failed. Reason: {e}"
                if e.column:
                    msg += f" (column: '{e.column}')"
                self.report.findings.append(Finding(level='ERROR', message=msg))
                skip_modules |= self._find_downstream_dependents(op)
                logger.error(f"{op.capitalize()} module failed — skipping downstream dependents.")
            except Exception as e:
                msg = f"{op.capitalize()} module raised an unexpected error: {e}"
                self.report.findings.append(Finding(level='ERROR', message=msg))
                skip_modules |= self._find_downstream_dependents(op)
                logger.error(msg, exc_info=True)

    def _generate_outputs(self):
        """
        Generates the final user-facing files (JSON, PDF, Notebook) based on
        the final state of the AnalysisReport after the analytical phase.
        """ 
        self.cache_manager.save_report(self.report)
        report_file_path = self.output_dir / "report.json" 
        report_file_path.write_text(self.report.model_dump_json(indent=2))

    def get_full_row_count(self, file_extension: str) -> int:
        """
        Calculates the total number of data rows in a file in a memory-efficient way.

        This function is critical for 'fast mode' to get the context of the full
        dataset without performing a costly full data load. It subtracts one
        for the header row.
        """
        logger.debug(f"Performing pre-scan to get full row count for: {self.file_path}")

        try:
            if file_extension == '.csv':
                with open(self.file_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                    reader = csv.reader(f)
                    line_count = sum(1 for _ in reader) - 1
                return line_count
            elif file_extension in ['.xlsx', '.xls']:
                # This loads the workbook structure but not all the cell data,
                # which is much more memory-efficient than reading with pandas.
                workbook = openpyxl.load_workbook(self.file_path, read_only=True, keep_links=False)
                sheet = workbook.active
                count = sum(1 for row in sheet.iter_rows(values_only=True) if any(row))
                # Subtract 1 for the header row
                return count - 1
            else:
                raise ValueError(f"Unsupported file type for row count pre-scan: {file_extension}")

        except FileNotFoundError:
            logger.error(f"File not found during pre-scan: {self.file_path}")
            raise
        except Exception as e:
            logger.error(f"An unexpected error occurred during row count pre-scan for {self.file_path}: {e}")
            raise DataLoadError(f"Could not determine row count for {self.file_path}. Reason: {e}") from e

    def _load_data_frame(self, file_extension) -> pd.DataFrame:
        """
        Loads the dataset from the file path into a pandas DataFrame.

        In 'fast' mode, this method performs an efficient partial read of the
        data to prioritize speed and low memory usage. In 'full' mode, it
        loads the entire dataset.
        """
        logger.info(f"Loading data from {self.file_path} in '{self.analysis_mode}' mode.")

        try:
            if self.analysis_mode == 'fast':
                sample_size = self.config.analysis.fast_mode_sample_rows
                logger.info(f"Sampling first {sample_size} rows for fast analysis.")

                if file_extension == '.csv':
                    df = pd.read_csv(self.file_path, nrows=sample_size)
                
                elif file_extension in ['.xlsx', '.xls']:
                    # The standard Excel reader doesn't have an efficient 'nrows'.
                    # We load the full sheet but immediately sample the first N rows.
                    # The performance gain here comes from the pre-scan row count.
                    df_full = pd.read_excel(self.file_path)
                    df = df_full.head(sample_size)
            else:
                if file_extension == '.csv':
                    df = pd.read_csv(self.file_path)
                elif file_extension in ['.xlsx', '.xls']:
                    df = pd.read_excel(self.file_path)
            
            logger.info(f"Successfully loaded DataFrame with shape: {df.shape}")
            return df

        except Exception as e:
            raise DataLoadError(f"Failed to read data file at {self.file_path}. Reason: {e}") from e
        
    def _generate_report_title(self) -> str:
        clean_stem = str(self.file_path.stem).replace("_", " ").replace("-", " ")
        return f"{clean_stem.title()} Analysis"

    def _find_downstream_dependents(self, failed_module: Operator) -> set[Operator]:
        """Returns all operators that depend directly or indirectly on failed_module."""
        registry = get_registry()
        skip: set[Operator] = {failed_module}
        added = True
        while added:
            added = False
            for op, spec in registry.items():
                if op not in skip and any(dep in skip for dep in spec.dependencies):
                    skip.add(op)
                    added = True
        return skip