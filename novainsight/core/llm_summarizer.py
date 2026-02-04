from __future__ import annotations

import logging
import json
from abc import ABC, abstractmethod
from typing import Optional, Any, List
import google.generativeai as genai
from toon import encode
import pandas as pd

from novainsight.core.base_module import BaseModule
from novainsight.config.config import NovaInsightConfig
from novainsight.schemas.analysis_report import (
    AnalysisReport,
    LLMSummary,
    Finding,
)

logger = logging.getLogger(__name__)

# --- Provider Interface ---

class LLMProvider(ABC):
    """Abstract interface for LLM providers."""
    @abstractmethod
    def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        pass

class GeminiProvider(LLMProvider):
    """Implementation for Google's Gemini models."""
    def __init__(self, api_key: str, model_name: str, temperature: float, max_tokens: int):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(
            model_name=model_name,
        )
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, prompt: str, schema: Optional[Any] = None) -> str:
        generation_config = genai.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )

        if schema:
            generation_config.response_mime_type = "application/json"
            generation_config.response_schema = schema

        try:
            response = self.model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            raise RuntimeError(f"Gemini API call failed: {e}")

class LLMSummarizer(BaseModule):
    """
    The 'Language Brain' of NovaInsight.
    Synthesizes statistical findings into a human-readable narrative and actionable recommendations.
    Uses 'toon' for efficient prompt formatting and JSON mode for structured output.
    """

    def __init__(self, config: NovaInsightConfig):
        super().__init__(config)
        self.provider: Optional[LLMProvider] = None
        self.findings: List[Finding] = []
        
        self._initialize_provider()

    def _initialize_provider(self):
        """Factory method to initialize the correct LLM provider."""
        if self.config.llm.provider.lower() == "google":
            if not self.config.llm.api_key:
                logger.warning("Google API Key not found in config. LLM Summarizer will be disabled.")
                return
            
            try:
                self.provider = GeminiProvider(
                    api_key=self.config.llm.api_key,
                    model_name=self.config.llm.model_name,
                    temperature=self.config.llm.temperature,
                    max_tokens=self.config.llm.max_tokens
                )
                logger.info(f"LLM Provider initialized: Google ({self.config.llm.model_name})")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini Provider: {e}")
        else:
            logger.warning(f"Unsupported LLM provider: {self.config.llm.provider}")

    def run(self, df: pd.DataFrame, report: AnalysisReport) -> AnalysisReport:
        """
        Constructs TOON context, queries LLM with JSON schema, validates response, updates report.
        """
        if not self.provider:
            report.findings.append(Finding(
                level="WARNING",
                message="LLM Summarizer skipped because no valid provider was initialized."
            ))
            return report

        try:
            logger.info("Constructing TOON prompt context from analysis report...")
            prompt = self._construct_prompt(report)
            
            logger.info("Querying LLM for structured analysis summary...")
            llm_output = self.provider.generate(prompt, schema=LLMSummary)

            json_response_text = json.loads(self._extract_json_string(llm_output))

            logger.info("Validating LLM response against schema...")
            normalized = self.normalize_llm_summary(json_response_text)
            summary = LLMSummary.model_validate(normalized)
            
            report.llm_summary = summary
            logger.info("LLM summary and recommendations successfully generated.")
            
        except Exception as e:
            logger.error(f"LLM Summarization failed: {e}")
            report.findings.append(Finding(
                level="ERROR",
                message=f"LLM Summarization failed: {e}"
            ))

        return report

    def _construct_prompt(self, report: AnalysisReport) -> str:
        """
        Constructs a structured context dictionary and serializes it using TOON.
        Passes the full report (excluding top-level aggregated findings) to the LLM.
        """
        context_data = report.model_dump(
            mode='json', 
            exclude={'llm_summary', 'findings', 'visualizations'}
        )

        try:
            toon_context = encode(context_data)
        except Exception as e:
            logger.warning(f"TOON encoding failed: {e}. Falling back to str().")
            toon_context = str(context_data)

        LLM_SCHEMA = """
            {
              "executive_summary": "string",
              "dataset_overview": "string",
              "key_findings_and_patterns": "string",
              "potential_issues_and_warnings": "string",
              "recommendations": {
                "preprocessing_steps": [
                  {
                    "category": "string",
                    "description": "string",
                    "priority": 1
                  }
                ],
                "feature_engineering_ideas": [
                  {
                    "category": "string",
                    "description": "string",
                    "priority": 1
                  }
                ],
                "modeling_suggestions": [
                  {
                    "category": "string",
                    "description": "string",
                    "priority": 1
                  }
                ],
                "pitfall_warnings": [
                  {
                    "category": "string",
                    "description": "string",
                    "priority": 1
                  }
                ]
              }
            }
        """
            
        # 3. Final System Instruction
        prompt = f"""
            You are NovaInsight, an expert Senior Data Scientist.

            Analyze the following dataset summary. The data is formatted in TOON; if TOON serialization failed, the format is JSON.

            Your task is to produce a structured analytical summary intended for downstream machine parsing and validation.

            STRICT OUTPUT RULES (CRITICAL):
            - Return ONLY valid JSON.
            - Do NOT include markdown, code fences, comments, or explanations.
            - Do NOT include ```json or ``` anywhere in the response.
            - Do NOT include any text before or after the JSON.
            - The response MUST start with '{' and end with '}'.
            - The output MUST be directly parsable by a standard JSON parser.

            FAST MODE INTERPRETATION (CRITICAL):
            - Fast mode means that only a subset of rows was analyzed for speed.
            - Actual row count is provided by original_row_count 
            - Do NOT interpret the analyzed_row_count as the true dataset size.
            - Do NOT make modeling recommendations based on sample size alone.
            - Assume the full dataset size is provided separately and is representative.
            - Frame conclusions as preliminary structural insights, not data volume limitations.

            OUTPUT JSON SCHEMA (MUST MATCH EXACTLY):
            All fields are REQUIRED. Arrays may be empty, but must be present.
            If you have no recommendations for a given list, you MUST still include the list as an empty array [].
            Do NOT omit any required fields.
            The `priority` field must be an integer in {1, 2, 3}, where 1 = High priority.

            {LLM_SCHEMA}

            FINAL OUTPUT CONTRACT (CRITICAL):
            You must populate ALL fields in the output JSON.

            Each field has a distinct purpose and must NOT be merged with others:

            - executive_summary:
              Interpretive, high-level narrative intended for decision-makers.
              May include implications and recommendations.
              Must NOT include raw schema metadata.

            - dataset_overview:
              Factual dataset metadata ONLY.
              This field MUST include:
              - total number of rows in the full dataset
              - number of columns
              - brief enumeration of feature groups (numerical, categorical, target)
              This field MUST NOT include analysis, interpretation, or recommendations.
              This field MUST be present even if similar information appears elsewhere.

            - key_findings_and_patterns:
              Analytical observations derived from statistics or structure.
              Do NOT restate dataset metadata here.

            - potential_issues_and_warnings:
              Risks, limitations, and data quality concerns only.

            - recommendations:
              Actionable next steps only, grouped by category.

            FIELD SEMANTICS:
            - executive_summary:
              A concise (3–5 sentences) executive-level overview of the dataset, its quality, and its suitability for the task.

            - dataset_overview:
              Description of dataset structure, size, key columns, data types, and overall composition.

            - key_findings_and_patterns:
              Notable statistical, structural, or distributional patterns relevant to the task.

            - potential_issues_and_warnings:
              Data quality concerns, bias risks, leakage risks, anomalies, missingness, or limitations.

            - preprocessing_steps:
              Concrete data cleaning, imputation, encoding, normalization, or transformation actions.

            - feature_engineering_ideas:
              Suggested derived features, transformations, or aggregations that could improve modeling.

            - modeling_suggestions:
              Appropriate model families, baselines, evaluation strategies, or validation approaches.

            - pitfall_warnings:
              Common or dataset-specific modeling mistakes to avoid.

            ANALYSIS RULES:
            1. Tone: Professional, objective, technical, and concise. No fluff.
            2. Focus on implications for the task: '{report.metadata.task}'.
            3. If the analysis mode is 'fast', explicitly note that findings are based on a structural spot-check.
            4. Recommendations must be specific, actionable, and prioritized.
               Example: "Log-transform 'Salary' due to heavy right skew and extreme outliers."

            CONTEXT DATA (TOON):
            {toon_context}
        """
        return prompt

    @staticmethod
    def _extract_json_string(text: str) -> str:
        """
         robustly extracts the JSON object from the LLM response, handling markdown fences.
        """
        text = text.strip()

        if text.startswith("```json"):
            text = text[7:]
        if text.startswith("```"):
            text = text[3:]
        if text.endswith("```"):
            text = text[:-3]
            
        text = text.strip()
        
        start = text.find("{")
        end = text.rfind("}")
        
        if start != -1 and end != -1 and end > start:
            return text[start : end + 1]
        
        return text
    
    @staticmethod
    def normalize_llm_summary(data: dict) -> dict:
        defaults = {
            "executive_summary": "",
            "dataset_overview": "",
            "key_findings_and_patterns": "",
            "potential_issues_and_warnings": "",
            "recommendations": {
                "preprocessing_steps": [],
                "feature_engineering_ideas": [],
                "modeling_suggestions": [],
                "pitfall_warnings": []
            }
        }

        for k, v in defaults.items():
            if k not in data or data[k] is None:
                data[k] = v

        rec = data["recommendations"]
        for k in defaults["recommendations"]:
            if k not in rec or rec[k] is None:
                rec[k] = []

        return data
