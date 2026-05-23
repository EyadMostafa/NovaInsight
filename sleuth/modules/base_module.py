from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from sleuth.config.config import SleuthConfig
from sleuth.schemas.analysis_report import AnalysisReport


class BaseModule(ABC):
    """
    An abstract base class for all analysis modules in the pipeline.
    """
    def __init__(self, config: SleuthConfig):
        """
        All modules are initialized with the application configuration.
        """
        self.config = config

    @abstractmethod
    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        The main execution method for the module.
        """
        pass
