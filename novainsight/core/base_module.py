from __future__ import annotations

from abc import ABC, abstractmethod
import pandas as pd
from novainsight.schemas.analysis_report import AnalysisReport
from novainsight.config.config import NovaInsightConfig


class BaseModule(ABC):
    @abstractmethod
    def run(self, report: AnalysisReport, config: NovaInsightConfig, df: pd.DataFrame):
        pass
