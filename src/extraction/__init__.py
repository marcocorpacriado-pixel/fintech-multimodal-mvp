"""Financial document extraction and analysis contracts."""

from .financial_analyzer import (
    METRIC_REGISTRY,
    FinancialMetricSelectionError,
    build_financial_metrics,
    classify_duration,
)
from .pipeline import (
    FINANCIAL_ANALYST_SYSTEM_PROMPT,
    GroundedAnalysisError,
    LLMClient,
    analyze_financials,
    build_analysis_prompt,
    qualitative_analysis_json_schema,
)
from .openrouter_client import (
    LLMConfigurationError,
    LLMProviderError,
    LLMResponseError,
    LLMTransportError,
    LLMUsage,
    OpenRouterLLMClient,
)
from .schemas import (
    ComparisonType,
    DocumentChunk,
    Evidence,
    FinancialAnalysisResult,
    FinancialMetric,
    LoadedDocument,
    LoadedDocumentMetadata,
    ManagementOutlook,
    NormalizedXBRLFact,
    RetrievalResult,
)

__all__ = [
    "ComparisonType",
    "DocumentChunk",
    "Evidence",
    "FINANCIAL_ANALYST_SYSTEM_PROMPT",
    "FinancialAnalysisResult",
    "FinancialMetric",
    "FinancialMetricSelectionError",
    "GroundedAnalysisError",
    "LLMClient",
    "LLMConfigurationError",
    "LLMProviderError",
    "LLMResponseError",
    "LLMTransportError",
    "LLMUsage",
    "LoadedDocument",
    "LoadedDocumentMetadata",
    "ManagementOutlook",
    "METRIC_REGISTRY",
    "NormalizedXBRLFact",
    "OpenRouterLLMClient",
    "RetrievalResult",
    "analyze_financials",
    "build_analysis_prompt",
    "build_financial_metrics",
    "classify_duration",
    "qualitative_analysis_json_schema",
]
