from __future__ import annotations

import json
import logging
from typing import Optional

import pandas as pd
from json_repair import repair_json
from toon import encode

from novainsight.config.config import NovaInsightConfig
from novainsight.exceptions import LLMSummarizerError
from novainsight.llm.factory import create_provider
from novainsight.llm.prompts import build_correction_messages, build_messages
from novainsight.llm.providers.base import LLMProvider
from novainsight.modules.base_module import BaseModule
from novainsight.schemas.analysis_report import AnalysisReport, Finding, LLMSummary

logger = logging.getLogger(__name__)


class LLMSummarizer(BaseModule):
    """
    The 'Language Brain' of NovaInsight.
    Synthesises statistical findings into a human-readable narrative and
    actionable recommendations via a configurable LLM provider.
    """

    def __init__(self, config: NovaInsightConfig) -> None:
        super().__init__(config)
        self.provider: Optional[LLMProvider] = None
        self._initialize_provider()

    def _initialize_provider(self) -> None:
        if not self.config.llm.api_key:
            logger.warning(
                "LLM API key not set (NOVA_INSIGHT_LLM_API_KEY). "
                "LLM Summarizer will be skipped."
            )
            return
        try:
            self.provider = create_provider(self.config.llm)
        except LLMSummarizerError as e:
            logger.error(f"Failed to initialize LLM provider: {e}")

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        if not self.provider:
            report.findings.append(Finding(
                level="WARNING",
                message="LLM Summarizer skipped: no valid provider initialized.",
            ))
            return report

        logger.info("Constructing TOON prompt context from analysis report...")
        toon_context = self._build_toon_context(report)
        messages = build_messages(
            toon_context=toon_context,
            task=report.metadata.task,
            analysis_mode=report.metadata.analysis_mode,
        )

        llm_output: str | None = None
        try:
            logger.info(
                f"Querying LLM ({self.config.llm.provider} / {self.config.llm.model_name})..."
            )
            llm_output = self.provider.generate(messages, response_schema=LLMSummary)
        except Exception as e:
            logger.error(f"LLM query failed: {e}")
            report.findings.append(Finding(level="ERROR", message=f"LLM query failed: {e}"))

        report.raw_llm_output = llm_output
        if llm_output is None:
            return report

        parsed = self._parse_with_fallback(llm_output, report)
        if parsed is None:
            return report

        try:
            logger.info("Validating LLM response against schema...")
            report.llm_summary = LLMSummary.model_validate(self._normalize(parsed))
            logger.info("LLM summary successfully generated.")
        except Exception as e:
            logger.error(f"LLM schema validation failed: {e}")
            report.findings.append(Finding(
                level="ERROR",
                message=f"LLM schema validation failed: {e}",
            ))

        return report

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_toon_context(self, report: AnalysisReport) -> str:
        context_data = report.model_dump(
            mode="json",
            exclude={"llm_summary", "findings", "visualizations"},
        )
        try:
            return encode(context_data)
        except Exception as e:
            logger.warning(f"TOON encoding failed: {e}. Falling back to str().")
            return str(context_data)

    def _try_parse(self, text: str) -> dict | None:
        """
        Pass 1: direct json.loads on the extracted JSON string.
        Pass 2: json-repair on the same string.
        Returns a dict on success, None on total failure.
        """
        extracted = self._extract_json_string(text)
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass
        try:
            result = repair_json(extracted, return_objects=True)
            if isinstance(result, dict):
                logger.warning("LLM output required JSON repair before parsing.")
                return result
        except Exception:
            pass
        return None

    def _parse_with_fallback(self, raw: str, report: AnalysisReport) -> dict | None:
        """
        Three-layer JSON recovery:
          1. Direct parse + repair.
          2. LLM correction retry.
          3. Degrade gracefully — append Finding, return None.
        """
        result = self._try_parse(raw)
        if result is not None:
            return result

        logger.warning("JSON parse and repair both failed. Attempting LLM correction retry...")
        try:
            correction_msgs = build_correction_messages(raw)
            retry_output = self.provider.generate(correction_msgs, response_schema=None)
            result = self._try_parse(retry_output)
            if result is not None:
                report.raw_llm_output = retry_output
                logger.info("LLM correction retry succeeded.")
                return result
        except Exception as e:
            logger.error(f"LLM correction retry failed: {e}")

        report.findings.append(Finding(
            level="ERROR",
            message=(
                "LLM output could not be parsed as valid JSON after repair and retry. "
                "Raw output is stored in report.raw_llm_output for inspection."
            ),
        ))
        return None

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """Strip markdown fences and extract the outermost JSON object."""
        text = text.strip()
        for fence in ("```json", "```"):
            if text.startswith(fence):
                text = text[len(fence):]
        if text.endswith("```"):
            text = text[:-3]
        text = text.strip()
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        return text

    @staticmethod
    def _normalize(data: dict) -> dict:
        """Fill in missing top-level keys and recommendation sub-lists with safe defaults."""
        defaults: dict = {
            "executive_summary": "",
            "dataset_overview": "",
            "key_findings_and_patterns": "",
            "potential_issues_and_warnings": "",
            "recommendations": {
                "preprocessing_steps": [],
                "feature_engineering_ideas": [],
                "modeling_suggestions": [],
                "pitfall_warnings": [],
            },
        }
        for k, v in defaults.items():
            if k not in data or data[k] is None:
                data[k] = v
        rec = data["recommendations"]
        for k in defaults["recommendations"]:
            if k not in rec or rec[k] is None:
                rec[k] = []
        return data
