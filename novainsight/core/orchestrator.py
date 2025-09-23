from __future__ import annotations

import pandas as pd
from logging import getLogger
from pathlib import Path
from typing import List, Set
from novainsight.config.config import NovaInsightConfig
from novainsight.core.data_profiler import DataProfiler
from novainsight.core.target_detector import TargetDetector
from novainsight.core.statistical_analyzer import StatisticalAnalyzer
from novainsight.core.dimensionality_reducer import DimensionalityReducer
from novainsight.core.visualizer import Visualizer
from novainsight.core.llm_summarizer import LLMSummarizer
from novainsight.core.recommender import Recommender
from novainsight.schemas.analysis_report import AnalysisReport, RunMetadata
from novainsight.core.report_generator import ReportGenerator
from novainsight.utils.cache_manager import CacheManager
from novainsight.utils.validators import (validate_file_path, validate_directory)

logger = getLogger(__name__)

class AnalysisPipeline:
    """
    The main orchestrator for the NovaInsight analysis pipeline.
    """
    DEPENDENCY_GRAPH = {
        'profiler': [],                             # profiler           
        'target': ['profiler'],                     # target --> profiler
        'stats': ['target'],                        # stats --> target --> profiler
        'dim_reduction': ['profiler'],              # dim_reduction --> profiler
        'viz': ['stats', 'dim_reduction'],          # viz --> dim_reduction --> stats --> target --> profiler
        'llm': ['stats'],                           # llm --> stats --> target --> profiler
        'recommendations': ['llm'],                 # recommendations --> llm --> stats --> target --> profiler
        'report': ['recommendations', 'viz']        # report --> recommendations --> llm --> viz --> dim_reduction --> stats --> target --> profiler
    }

    MODULES = {
        'profiler': DataProfiler(),
        'target': TargetDetector(),
        'stats': StatisticalAnalyzer(),
        'dim_reduction': DimensionalityReducer(),
        'viz': Visualizer(),
        'llm': LLMSummarizer(),
        'recommendations': Recommender(),
        'report': ReportGenerator()
    }
    
    EXECUTION_ORDER = [
        'profiler', 'target', 'stats', 'dim_reduction', 
        'viz', 'llm', 'recommendations', 'report'
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
        task: str = 'supervised'
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
        
        self.cache_manager = CacheManager(config.cache)
        self.report: AnalysisReport | None = None
        self.df: pd.DataFrame | None = None
        
        modules_to_resolve = requested_modules or ['report']
        self.execution_plan = self._resolve_execution_plan(modules_to_resolve)

    def run(self):
        """Executes the full, resolved analysis pipeline from start to finish."""
        try:
            self._initialize_report_and_data()
            self._run_analytical_modules()
            self._generate_outputs()
            logger.info("Analysis pipeline completed")
        except Exception as e:
            raise ValueError(f"A fatal error halted the pipeline: {e}", exc_info=self.config.general.debug)

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

        if self.task == 'unsupervised' and 'target' in sorted_plan:
            sorted_plan.remove('target')
            
        logger.info(f"Execution plan resolved: {sorted_plan}")
        return sorted_plan

    def _initialize_report_and_data(self):
        """
        Validates paths, loads the dataset,
        and initializes the AnalysisReport object from cache or from scratch.
        """
        self.file_path = validate_file_path(self.file_path)
        if self.file_path.suffix.lower() not in self.SUPPORTED_FILE_EXTENSIONS:
            raise ValueError(f"Fatal Error: Unsupported file extension. Supported extensions are: {','.join(self.SUPPORTED_FILE_EXTENSIONS)}") 

        is_valid, message, resolved_path = validate_directory(self.output_dir)
        if not is_valid:
            raise IOError(f"Invalid output directory specified. Reason: {message}")
        self.output_dir = resolved_path
        
        file_hash = self.cache_manager.hash_file(self.file_path)

        if self.force_rerun:
            logger.info("Force rerun requested. Clearing any existing cache.")
            self.cache_manager.clear_workspace(file_hash)
            self.report = None
        else:
            self.report = self.cache_manager.load_report(file_hash)

        if not self.report:
            logger.info("No valid cache found. Initializing a new analysis report.")
            
            if not self.report_title:
                clean_stem = str(self.file_path.stem).replace("_", " ").replace("-", " ")
                self.report_title = f"{clean_stem.title()} Analysis"

            metadata = RunMetadata(
                input_file=self.file_path,
                output_dir=self.output_dir,
                file_hash=file_hash,
                report_title=self.report_title,
                analysis_mode=self.analysis_mode,
                task=self.task
            )
            self.report = AnalysisReport(metadata=metadata)

        logger.info(f"Loading data in '{self.analysis_mode}' mode.")
        file_extension = self.file_path.suffix.lower()
        
        try:
            if file_extension == '.csv':
                self.df = pd.read_csv(self.file_path)
            elif file_extension in ['.xlsx', '.xls']:
                self.df = pd.read_excel(self.file_path)

            if self.analysis_mode == 'fast':
                self.df = self.df.sample(self.config.analysis.fast_mode_sample_rows)
            
            logger.info(f"Successfully loaded DataFrame in {self.analysis_mode} analysis mode with shape: {self.df.shape}")
        except Exception as e:
            raise IOError(f"Failed to read the data file at {self.file_path}. Reason: {e}")
        
    def _run_analytical_modules(self):
        """
        Iterates through the execution plan and runs each analytical module
        sequentially, with graceful error handling for module-level failures.
        """
        pass
    
    def _generate_outputs(self):
        """
        Generates the final user-facing files (JSON, PDF, Notebook) based on
        the final state of the AnalysisReport after the analytical phase.
        """ 
        pass

   