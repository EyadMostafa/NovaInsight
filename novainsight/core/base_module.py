from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from novainsight.schemas.analysis_report import AnalysisReport
from novainsight.config.config import NovaInsightConfig
from typing import Any

class BaseModule(ABC):
    """
    An abstract base class for all analysis modules in the pipeline.
    """
    def __init__(self, config: NovaInsightConfig):
        """
        All modules are initialized with the application configuration.
        """
        self.config = config

    @abstractmethod
    def run(self, df: pd.DataFrame, report: AnalysisReport) -> Any:
        """
        The main execution method for the module.
        """
        pass
