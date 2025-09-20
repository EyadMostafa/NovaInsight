from logging import getLogger
from pathlib import Path
from typing import List, Set
from novainsight.schemas.analysis_report import AnalysisReport, RunMetadata
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
    
    EXECUTION_ORDER = [
        'profiler', 'target', 'stats', 'dim_reduction', 
        'viz', 'llm', 'recommendations', 'report'
    ]

    SUPPORTED_FILE_EXTENSIONS = ['.csv', '.xlsx', '.xls']

    def __init__(
        self,
        file_path: Path,
        config: AnalysisSettings,
        output_dir: Path | None = None,
        requested_modules: List[str] | None = None,
        force_rerun: bool = False,
        user_target: str | None = None,
        analysis_mode: str | None = None,
        report_title: str | None = None
    ):
        """Initializes the pipeline with all necessary context from the CLI."""
        self.file_path = file_path
        self.config = config
        self.output_dir = output_dir or self.config.output.default_directory
        self.force_rerun = force_rerun
        self.user_target = user_target
        self.analysis_mode = analysis_mode or self.config.default_mode
        self.report_title = report_title
        
        # self.cache_manager = CacheManager(config.cache)
        self.report: AnalysisReport | None = None
        
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
            for dep in self.DEPENDENCY_GRAPH.get(module_name, [])
                find_deps(dep)

        for module in requested:
            find_deps(module)
        
        sorted_plan = sorted(list(final_modules), key=self.EXECUTION_ORDER.index)
        logger.info(f"Execution plan resolved: {sorted_plan}")
        return sorted_plan

    def _initialize_report_and_data(self):
        """
        Handles loading the dataset and creating the AnalysisReport object,
        either by creating a new one or loading a cached version.
        """
        input_path = self.validate_file_path(self.file_path)
        output_dir = self.validate_directory(self.output_dir)
        file_hash = self.cache_manager.hash_file(input_path)
        if not self.report_title:
            self.report_title = f"{str(input_path.stem).strip().replace("_", " ").replace("-", " ").title()} Analysis"

        metadata = RunMetadata(
            input_file=input_path,
            output_dir=output_dir,
            file_hash=file_hash,
            report_title=self.report_title,
            analysis_mode=self.analysis_mode,
        )

        self.analysis_report = AnalysisReport(
            metadata=metadata
        )


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

   