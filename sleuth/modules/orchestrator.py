from __future__ import annotations

import csv
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import sleuth.modules  # noqa: F401 — triggers @register_module decorators
from sleuth.config.config import SleuthConfig
from sleuth.exceptions import (
    DataLoadError,
    ModuleError,
    OrchestratorError,
    PipelineError,
)
from sleuth.modules.registry import get_registry, topological_order
from sleuth.schemas.analysis_report import (
    AnalysisReport,
    Finding,
    Operator,
    PipelineTiming,
    RunMetadata,
)
from sleuth.utils.cache_manager import CacheManager
from sleuth.utils.logger import get_logger
from sleuth.utils.validators import validate_directory, validate_file_path

logger = get_logger("sleuth.modules.orchestrator")

_RANDOM_SEED = 42


class AnalysisPipeline:
    """The main orchestrator for the Sleuth analysis pipeline."""

    SUPPORTED_FILE_EXTENSIONS = ['.csv', '.parquet', '.feather', '.xlsx']

    def __init__(
        self,
        file_path: Path,
        config: SleuthConfig,
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
        self._csv_delimiter: str = ','  # overwritten by _detect_csv_delimiter for CSV inputs

        self.execution_plan = self._resolve_execution_plan(requested_modules)

    def run(self):
        """Executes the full, resolved analysis pipeline from start to finish."""
        pipeline_start = time.perf_counter()
        try:
            self._initialize_report_and_data()
            module_times = self._run_analytical_modules()
            total = time.perf_counter() - pipeline_start
            self.report.timing = PipelineTiming(total_seconds=total, modules=module_times)
            logger.info(f"Total pipeline time: {total:.2f}s")
            self._generate_outputs()
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
        Validates paths, loads the dataset (with random sampling if in fast mode),
        and initializes the AnalysisReport object from cache or from scratch.
        """
        self.file_path = validate_file_path(self.file_path)
        file_extension = self.file_path.suffix.lower()
        if file_extension not in self.SUPPORTED_FILE_EXTENSIONS:
            raise DataLoadError(
                f"Unsupported file extension '{file_extension}'. "
                f"Supported: {', '.join(self.SUPPORTED_FILE_EXTENSIONS)}"
            )

        self.output_dir = Path(self.output_dir / "sleuth_reports")
        is_valid, message, resolved_path = validate_directory(self.output_dir)
        if not is_valid:
            raise OrchestratorError(f"Invalid output directory. Reason: {message}")

        if file_extension == '.csv':
            self._csv_delimiter = self._detect_csv_delimiter()

        file_hash = self.cache_manager.hash_file(self.file_path)

        if self.force_rerun:
            logger.info("Force rerun requested. Clearing any existing cache for this dataset.")
            self.cache_manager.clear_workspace(file_hash)
            self.report = None
        else:
            self.report = self.cache_manager.load_report(file_hash)

        full_row_count = self.get_full_row_count(file_extension)
        self.df = self._load_data_frame(file_extension, full_row_count)

        if not self.report:
            logger.info("No valid cache found. Initializing a new analysis report.")

            if not self.report_title:
                self.report_title = self._generate_report_title()

            self.output_dir = Path(resolved_path / self.report_title)
            self.output_dir.mkdir(parents=True, exist_ok=True)

            metadata = RunMetadata(
                input_file=self.file_path,
                output_dir=self.output_dir,
                file_hash=file_hash,
                report_title=self.report_title,
                analysis_mode=self.analysis_mode,
                task=self.task,
                user_target=self.user_target,
                original_row_count=full_row_count,
            )

            self.report = AnalysisReport(metadata=metadata, findings=[])
        else:
            self.output_dir = Path(self.report.metadata.output_dir)

        logger.info("Analysis report initialized successfully.")

    @staticmethod
    def _compute_waves(
        plan: list[Operator],
        registry: dict,
    ) -> list[list[Operator]]:
        """Groups plan operators into ordered execution waves.
        A module enters a wave once all its in-plan dependencies are in a prior wave.
        """
        plan_set = set(plan)
        completed: set[Operator] = set()
        waves: list[list[Operator]] = []
        remaining = list(plan)
        while remaining:
            wave = [
                op for op in remaining
                if all(dep in completed for dep in registry[op].dependencies if dep in plan_set)
            ]
            waves.append(wave)
            completed.update(wave)
            remaining = [op for op in remaining if op not in completed]
        return waves

    def _run_module(self, op: Operator, spec) -> None:
        """Executes a single module. Called from worker threads."""
        if not self.force_rerun and spec.is_completed(self.report):
            logger.info(f"Skipping {op}: valid results found in cache.")
            self._module_times[op.value] = 0.0
            return
        t0 = time.perf_counter()
        spec.cls(self.config).run(df=self.df, report=self.report)
        self._module_times[op.value] = time.perf_counter() - t0

    def _run_analytical_modules(self) -> dict[str, float]:
        """
        Runs all modules in the execution plan using wave-based parallelism.
        Modules within the same wave have no mutual dependencies and run concurrently
        via ThreadPoolExecutor. Returns a per-module timing dict.
        """
        registry = get_registry()
        skip_modules: set[Operator] = set()
        self._module_times: dict[str, float] = {}
        self._skipped_op_values: set[str] = set()
        waves = self._compute_waves(self.execution_plan, registry)

        with ThreadPoolExecutor() as executor:
            for wave in waves:
                active = [op for op in wave if op not in skip_modules]
                skipped = [op for op in wave if op in skip_modules]

                for op in skipped:
                    self._module_times[op.value] = 0.0
                    self._skipped_op_values.add(op.value)
                    self.report.findings.append(Finding(
                        level='WARNING',
                        message=f"{op.capitalize()} module skipped due to the failure or skipping of a dependency module.",
                    ))

                if not active:
                    continue

                futures = {
                    executor.submit(self._run_module, op, registry[op]): op
                    for op in active
                }

                for future in as_completed(futures):
                    op = futures[future]
                    try:
                        future.result()
                    except ModuleError as e:
                        self._module_times.setdefault(op.value, 0.0)
                        msg = f"{op.capitalize()} module failed. Reason: {e}"
                        if e.column:
                            msg += f" (column: '{e.column}')"
                        self.report.findings.append(Finding(level='ERROR', message=msg))
                        skip_modules |= self._find_downstream_dependents(op)
                        logger.error(f"{op.capitalize()} module failed — skipping downstream dependents.")
                    except Exception as e:
                        self._module_times.setdefault(op.value, 0.0)
                        msg = f"{op.capitalize()} module raised an unexpected error: {e}"
                        self.report.findings.append(Finding(level='ERROR', message=msg))
                        skip_modules |= self._find_downstream_dependents(op)
                        logger.error(msg, exc_info=True)

        lines = [f"  {'module':<18} {'time':>8}", "  " + "-" * 28]
        for op_val, elapsed in self._module_times.items():
            if op_val in self._skipped_op_values:
                tag = "  (skipped)"
            elif elapsed == 0.0:
                tag = "  (cached)"
            else:
                tag = ""
            lines.append(f"  {op_val:<18} {elapsed:>7.2f}s{tag}")
        logger.info("Module timing breakdown:\n" + "\n".join(lines))

        return self._module_times

    def _generate_outputs(self):
        """Persists the final AnalysisReport to both the cache and the run output dir."""
        # Guarantee HTML is always generated — even when the REPORT module was cascade-
        # skipped because a dependency failed.  This also ensures report.timing (set
        # just before this call) is available inside the template.
        # Local import avoids a load-order issue: orchestrator.py uses
        # `import sleuth.modules` to fire all @register_module decorators, and
        # report_generator is one of those modules; importing it at module-top-level
        # would race against that initialisation sequence.
        # Only attempt report generation when profiling completed — the template
        # requires report.profile (dataset_stats, column_details).  If PROFILER
        # itself failed there is nothing useful to render.
        if (
            Operator.REPORT in self.execution_plan
            and self.report.html_report_path is None
            and self.report.profile is not None
        ):
            from sleuth.modules.report_generator import ReportGenerator  # noqa: PLC0415
            ReportGenerator(self.config).run(df=self.df, report=self.report)

        self.cache_manager.save_report(self.report)
        report_file_path = self.output_dir / "report.json"
        report_file_path.write_text(self.report.model_dump_json(indent=2))

    def get_full_row_count(self, file_extension: str) -> int:
        """
        Returns the total number of data rows (excluding header) for a file without
        loading the full dataset into memory where possible.

        CSV  — streams line-by-line via csv.reader; O(rows) time, O(1) memory.
        Parquet — reads metadata only from file footer; O(1) time and memory.
        Feather — reads the Arrow IPC frame table; fast due to memory-mapping.
        XLSX  — reads a single column via pandas; avoids loading all data.
        """
        logger.debug(f"Performing pre-scan to get full row count for: {self.file_path}")

        try:
            if file_extension == '.csv':
                with open(self.file_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                    reader = csv.reader(f, delimiter=self._csv_delimiter)
                    return sum(1 for _ in reader) - 1  # subtract header row

            if file_extension == '.parquet':
                metadata = pq.read_metadata(self.file_path)
                return metadata.num_rows

            if file_extension == '.feather':
                # Feather (Arrow IPC) is memory-mapped; reading the full table is fast.
                return len(pd.read_feather(self.file_path))

            if file_extension == '.xlsx':
                # Read only the first column to count rows cheaply.
                return len(pd.read_excel(self.file_path, usecols=[0], engine='openpyxl'))

            raise ValueError(f"Unsupported file type for row count pre-scan: {file_extension}")

        except FileNotFoundError:
            logger.error(f"File not found during pre-scan: {self.file_path}")
            raise
        except Exception as e:
            raise DataLoadError(
                f"Could not determine row count for {self.file_path}. Reason: {e}"
            ) from e

    def _load_data_frame(self, file_extension: str, full_row_count: int) -> pd.DataFrame:
        """
        Loads the dataset into a pandas DataFrame.

        In 'fast' mode a true random sample of `fast_mode_sample_rows` rows is
        drawn, not just the first N rows.

        CSV    — uses pandas `skiprows` to stream only the kept rows; memory-efficient.
        Parquet — loads the full file (columnar format is already fast) then samples.
        Feather — loads the full file (memory-mapped) then samples.
        """
        logger.info(f"Loading data from {self.file_path} in '{self.analysis_mode}' mode.")

        try:
            if self.analysis_mode == 'fast':
                sample_size = self.config.analysis.fast_mode_sample_rows

                if full_row_count <= sample_size:
                    logger.info(
                        f"Dataset has {full_row_count} rows which is ≤ sample size "
                        f"({sample_size}). Loading full file."
                    )
                    return self._load_full(file_extension)

                logger.info(
                    f"Randomly sampling {sample_size} of {full_row_count} rows "
                    f"(seed={_RANDOM_SEED})."
                )
                return self._load_sample(file_extension, full_row_count, sample_size)

            return self._load_full(file_extension)

        except DataLoadError:
            raise
        except Exception as e:
            raise DataLoadError(
                f"Failed to read data file at {self.file_path}. Reason: {e}"
            ) from e

    def _load_full(self, file_extension: str) -> pd.DataFrame:
        """Loads the entire dataset without any sampling."""
        if file_extension == '.csv':
            return pd.read_csv(self.file_path, sep=self._csv_delimiter)
        if file_extension == '.parquet':
            return pd.read_parquet(self.file_path)
        if file_extension == '.feather':
            return pd.read_feather(self.file_path)
        if file_extension == '.xlsx':
            return pd.read_excel(self.file_path, engine='openpyxl', sheet_name=0)
        raise DataLoadError(f"No reader available for extension: {file_extension}")

    def _load_sample(
        self, file_extension: str, full_row_count: int, sample_size: int
    ) -> pd.DataFrame:
        """
        Loads a random sample of `sample_size` rows.
        """
        rng = np.random.default_rng(_RANDOM_SEED)

        if file_extension == '.csv':
            all_data_indices = np.arange(1, full_row_count + 1)
            skip_count = full_row_count - sample_size
            skip_indices = rng.choice(all_data_indices, skip_count, replace=False)
            return pd.read_csv(self.file_path, sep=self._csv_delimiter, skiprows=skip_indices)

        if file_extension in ('.parquet', '.feather', '.xlsx'):
            df = self._load_full(file_extension)
            return df.sample(n=sample_size, random_state=_RANDOM_SEED).reset_index(drop=True)

        raise DataLoadError(f"No sampler available for extension: {file_extension}")

    def _detect_csv_delimiter(self) -> str:
        """
        Sniffs the delimiter of a CSV file from its first 16 KB.

        Uses csv.Sniffer which scores each candidate character (comma, semicolon,
        tab, pipe) and picks the one that best explains the file structure.
        Falls back to ',' on any failure so the pipeline never aborts on a
        detection edge-case.
        """
        try:
            with open(self.file_path, 'r', encoding='utf-8', errors='replace', newline='') as f:
                sample = f.read(16_384)
            dialect = csv.Sniffer().sniff(sample, delimiters=',;\t|')
            logger.debug(f"Detected CSV delimiter: {repr(dialect.delimiter)}")
            return dialect.delimiter
        except csv.Error:
            logger.debug("CSV delimiter detection inconclusive; defaulting to ','.")
            return ','

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
