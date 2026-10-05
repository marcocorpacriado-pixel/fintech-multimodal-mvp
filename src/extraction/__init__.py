"""Financial document extraction and analysis contracts."""

from .financial_analyzer import (
    METRIC_REGISTRY,
    FinancialMetricSelectionError,
    build_financial_metrics,
    classify_duration,
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
    "FinancialAnalysisResult",
    "FinancialMetric",
    "FinancialMetricSelectionError",
    "LoadedDocument",
    "LoadedDocumentMetadata",
    "ManagementOutlook",
    "METRIC_REGISTRY",
    "NormalizedXBRLFact",
    "RetrievalResult",
    "build_financial_metrics",
    "classify_duration",
]
