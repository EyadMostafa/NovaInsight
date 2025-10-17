from __future__ import annotations

import pandas as pd
import openpyxl
from logging import getLogger
from pathlib import Path
from typing import List, Set, Optional
from enum import Enum, auto
from novainsight.config.config import NovaInsightConfig
from novainsight.core.base_module import BaseModule
from novainsight.core.data_profiler import DataProfiler
from novainsight.core.target_detector import TargetDetector
from novainsight.core.statistical_analyzer import StatisticalAnalyzer
from novainsight.core.dimensionality_reducer import DimensionalityReducer
from novainsight.core.visualizer import Visualizer
from novainsight.core.llm_summarizer import LLMSummarizer
from novainsight.core.recommender import Recommender
from novainsight.schemas.analysis_report import AnalysisReport, RunMetadata, DatasetProfile, DatasetStats, Finding, Operator
from novainsight.core.report_generator import ReportGenerator
from novainsight.utils.cache_manager import CacheManager
from novainsight.utils.validators import validate_file_path, validate_directory

logger = getLogger(__name__)

class AnalysisPipeline:
    """
    The main orchestrator for the NovaInsight analysis pipeline.
    """
    DEPENDENCY_GRAPH = {
        Operator.PROFILER : [],                                     # profiler           
        Operator.TARGET: [Operator.PROFILER],                       # target --> profiler
        Operator.STATS: [Operator.TARGET],                          # stats --> target --> profiler
        Operator.DIM_REDUCTION: [Operator.PROFILER],                # dim_reduction --> profiler
        Operator.VIZ: [Operator.STATS, Operator.DIM_REDUCTION],     # viz --> dim_reduction --> stats --> target --> profiler
        Operator.LLM: [Operator.STATS],                             # llm --> stats --> target --> profiler
        Operator.RECOMMENDATIONS: [Operator.LLM],                   # recommendations --> llm --> stats --> target --> profiler
        Operator.REPORT: [Operator.RECOMMENDATIONS, Operator.VIZ]   # report --> recommendations --> llm --> viz --> dim_reduction --> stats --> target --> profiler
    }

    MODULES_REGISTRY = {
        Operator.PROFILER: DataProfiler,
        Operator.TARGET: TargetDetector,
        Operator.STATS: StatisticalAnalyzer,
        Operator.DIM_REDUCTION: DimensionalityReducer,
        Operator.VIZ: Visualizer,
        Operator.LLM: LLMSummarizer,
        Operator.RECOMMENDATIONS: Recommender,
        Operator.REPORT: ReportGenerator
    }
    
    EXECUTION_ORDER = [
        Operator.PROFILER, Operator.TARGET, Operator.STATS, Operator.DIM_REDUCTION, 
        Operator.VIZ, Operator.LLM, Operator.RECOMMENDATIONS, Operator.REPORT
    ]

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
        
        modules_to_resolve = requested_modules or [Operator.REPORT]
        self.execution_plan = self._resolve_execution_plan(modules_to_resolve)

    def run(self):
        """Executes the full, resolved analysis pipeline from start to finish."""
        try:
            self._initialize_report_and_data()
            self._run_analytical_modules()
            self._generate_outputs()
            logger.info("Analysis pipeline completed")
        except Exception as e:
            raise ValueError(f"A fatal error halted the pipeline: {e}")

    def _resolve_execution_plan(self, requested: List[str]) -> List[str]:
        """Calculates the full list of modules to run based on dependencies."""
        final_modules: Set[str] = set()
        
        def find_deps(module_name: str):
            if module_name not in self.DEPENDENCY_GRAPH:
                logger.warning(f"Unknown module '{module_name}' requested. Ignoring.")
                return
            final_modules.add(module_name)
            for dep in self.DEPENDENCY_GRAPH.get(module_name, []):
                find_deps(dep)

        for module in requested:
            find_deps(module)
        
        sorted_plan = sorted(list(final_modules), key=self.EXECUTION_ORDER.index)

        if self.task == 'unsupervised' and Operator.TARGET in sorted_plan:
            sorted_plan.remove(Operator.TARGET)
            
        logger.info(f"Execution plan resolved: {sorted_plan}")
        return sorted_plan

    def _initialize_report_and_data(self):
        """
        Validates paths, loads the dataset (with sampling if in fast mode),
        and initializes the AnalysisReport object from cache or from scratch.
        """
        self.file_path = validate_file_path(self.file_path)
        file_extension = self.file_path.suffix.lower() 
        if  file_extension not in self.SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(f"Fatal Error: Unsupported file extension. Supported extensions are: {','.join(self.SUPPORTED_FILE_EXTENSIONS)}") 

        self.output_dir = Path(self.output_dir / "novainsight_reports") 
        is_valid, message, resolved_path = validate_directory(self.output_dir)
        if not is_valid:
            raise IOError(f"Invalid output directory specified. Reason: {message}")
        
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
                row_count=0, 
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
        module: Optional[BaseModule] = None
        skip_modules = set()
        
        for module_name in self.execution_plan:
            if module_name in skip_modules: 
                finding = Finding(
                    level='WARNING',
                    message=f"{module_name.capitalize()} module skipped due to the failure or skipping of a dependency module."
                )
                self.report.findings.append(finding)
                continue
            
            try:
                module = self.MODULES_REGISTRY[module_name](self.config)
                updated_report = module.run(df=self.df, report=self.report)
                if updated_report:
                    self.report = updated_report

            except Exception as e:
                finding = Finding(
                    level='WARNING',
                    message=f"{module_name.capitalize()} module failed. Reason: {e}"
                )
                self.report.findings.append(finding)
                skip_modules = skip_modules | self._find_downstream_dependents(module_name)
                logger.error(f'{module_name.capitalize()} module failed. Skipping and proceeding with the pipeline.')

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
                with open(self.file_path, 'rb') as f:
                    line_count = sum(1 for _ in f)
                # Subtract 1 for the header row
                return line_count - 1
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
            raise IOError(f"Could not read file to determine total row count. Reason: {e}") from e

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
            raise IOError(f"Failed to read the data file at {self.file_path}. Reason: {e}")
        
    def _generate_report_title(self) -> str:
        clean_stem = str(self.file_path.stem).replace("_", " ").replace("-", " ")
        return f"{clean_stem.title()} Analysis"

    def _find_downstream_dependents(self, failed_module: Operator) -> set[Operator]:
        """
        Returns all modules that depend directly or indirectly on the failed module.
        """
        skip_modules = {failed_module}
        added = True

        while added:
            added = False
            for k, dependencies in self.DEPENDENCY_GRAPH.items():
                if any(dep in skip_modules for dep in dependencies):
                    if k not in skip_modules:
                        skip_modules.add(k)
                        added = True

        return skip_modules