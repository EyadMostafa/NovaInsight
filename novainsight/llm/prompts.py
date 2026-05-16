"""
prompts.py

All prompt templates for the LLM Summarizer.

Architecture:
  - The SYSTEM prompt is fully static: persona, output rules, JSON schema, field semantics.
    It is assembled once at module load and never changes between runs, making it an ideal
    candidate for server-side prompt caching (Anthropic) or automatic prefix caching (OpenAI).
  - The USER prompt is dynamic: it carries the task type, fast-mode notice, and the
    TOON-encoded dataset context for the current run.
  - The CORRECTION prompt is a separate, minimal exchange used only when the primary
    output is malformed JSON.
"""
from __future__ import annotations

from novainsight.llm.providers.base import PromptMessage

# ---------------------------------------------------------------------------
# System prompt components (all static — assembled once at module load)
# ---------------------------------------------------------------------------

_PERSONA = """\
You are NovaInsight, an expert Senior Data Scientist and autonomous analytical agent.
Your role is to synthesize the statistical findings from an Exploratory Data Analysis
pipeline into structured, actionable intelligence for data practitioners.\
"""

_OUTPUT_RULES = """\
STRICT OUTPUT RULES (CRITICAL — violations cause downstream parse failures):
- Return ONLY a valid JSON object. The response MUST start with '{' and end with '}'.
- Do NOT include markdown, code fences (``` or ```json), comments, or explanations.
- Do NOT include any text before or after the JSON object.
- Every JSON field pair must be separated by a comma ','.
- Every string value must be terminated with a closing double-quote '"'.
- Every array '[]' and object '{}' must be properly closed.\
"""

_JSON_SCHEMA = """\
OUTPUT JSON SCHEMA (MUST MATCH EXACTLY):
All fields are REQUIRED. Arrays may be empty [] but must be present.
The 'priority' field must be an integer in {1, 2, 3} where 1 = High priority.

{
  "executive_summary": "string",
  "dataset_overview": "string",
  "key_findings_and_patterns": "string",
  "potential_issues_and_warnings": "string",
  "recommendations": {
    "preprocessing_steps": [
      {"category": "string", "description": "string", "priority": 1}
    ],
    "feature_engineering_ideas": [
      {"category": "string", "description": "string", "priority": 1}
    ],
    "modeling_suggestions": [
      {"category": "string", "description": "string", "priority": 1}
    ],
    "pitfall_warnings": [
      {"category": "string", "description": "string", "priority": 1}
    ]
  }
}\
"""

_FIELD_SEMANTICS = """\
FIELD DEFINITIONS — populate ALL fields; do NOT merge, omit, or re-order them:

executive_summary:
  Interpretive, 3–5 sentence overview for decision-makers. May include implications
  and high-level recommendations. Must NOT contain raw schema metadata or statistics.

dataset_overview:
  Factual metadata ONLY. Must include: total row count (from original_row_count),
  number of columns, brief enumeration of feature groups (numerical, categorical,
  datetime, target). Must NOT include analysis, interpretation, or recommendations.

key_findings_and_patterns:
  Analytical observations derived from statistics, distributions, or structure.
  Do NOT restate dataset metadata here.

potential_issues_and_warnings:
  Risks, data quality concerns, bias risks, leakage risks, anomalies, or missingness.

preprocessing_steps:
  Concrete cleaning, imputation, encoding, normalization, or transformation actions.

feature_engineering_ideas:
  Derived features, transformations, or aggregations that could improve modelling.

modeling_suggestions:
  Appropriate model families, baselines, evaluation strategies, or validation approaches.

pitfall_warnings:
  Common or dataset-specific modelling mistakes to avoid.

Tone: Professional, objective, technical, and concise. No filler phrases.\
"""

SYSTEM_PROMPT: str = "\n\n".join([_PERSONA, _OUTPUT_RULES, _JSON_SCHEMA, _FIELD_SEMANTICS])

# ---------------------------------------------------------------------------
# Public builders
# ---------------------------------------------------------------------------

def build_messages(
    toon_context: str,
    task: str,
    analysis_mode: str,
) -> list[PromptMessage]:
    """
    Build the ordered message list for a primary analysis call.

    The system message is marked cache=True so providers that support explicit
    prompt caching (Anthropic) will cache it. OpenAI caches it automatically.
    The user message carries the dynamic dataset context and is never cached.
    """
    fast_mode_note = ""
    if analysis_mode == "fast":
        fast_mode_note = (
            "\nFAST MODE NOTE (CRITICAL):\n"
            "- Only a structural subset of rows was analysed for speed.\n"
            "- 'original_row_count' in the context is the TRUE dataset size.\n"
            "- Do NOT interpret 'analyzed_row_count' as the full dataset size.\n"
            "- Do NOT make recommendations based on sample size alone.\n"
            "- Frame conclusions as preliminary structural insights.\n"
        )

    user_content = (
        f"Analyse the following dataset. The analytical task is: '{task}'.\n"
        f"{fast_mode_note}\n"
        "The context below is encoded in TOON (Token-Optimised Object Notation). "
        "If any TOON syntax is ambiguous, treat it as structured JSON.\n\n"
        f"CONTEXT DATA:\n{toon_context}"
    )

    return [
        PromptMessage(role="system", content=SYSTEM_PROMPT, cache=True),
        PromptMessage(role="user", content=user_content, cache=False),
    ]


def build_correction_messages(malformed_output: str) -> list[PromptMessage]:
    """
    Build the message list for a JSON correction retry.
    The system message is intentionally short and not marked for caching
    since correction retries are rare and the content varies.
    """
    system = (
        "You are a JSON repair assistant. "
        "Fix the malformed JSON provided by the user. "
        "Return ONLY the corrected JSON object — no markdown, no explanation. "
        "Do not add or remove fields; fix syntax only. "
        "Use double quotes for all strings. "
        "Separate all fields with commas. "
        "Close all arrays and objects."
    )
    user = f"Fix this malformed JSON:\n\n{malformed_output}"
    return [
        PromptMessage(role="system", content=system, cache=False),
        PromptMessage(role="user", content=user, cache=False),
    ]
