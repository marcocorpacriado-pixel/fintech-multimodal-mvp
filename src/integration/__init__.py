"""Public handoff contracts between analysis, API/UI, and audio modules."""

from .analysis_handoff import (
    AnalysisEvidenceDTO,
    AnalysisHandoff,
    AnalysisMetricDTO,
    AnalysisMode,
    ManagementOutlookDTO,
    PipelineMetadataDTO,
    TokenAttributionDTO,
    TTSInput,
    VerificationDTO,
    VerificationIssueDTO,
    build_analysis_handoff,
    build_tts_input,
)
from .chat_context import (
    CHAT_SYSTEM_PROMPT,
    ChatMessage,
    ChatRequest,
    build_chat_context,
    citation_index,
)
from .errors import (
    IntegrationDiagnostic,
    IntegrationError,
    IntegrationErrorCode,
    diagnose_integration_failure,
    map_integration_error,
)

__all__ = [
    "AnalysisEvidenceDTO",
    "AnalysisHandoff",
    "AnalysisMetricDTO",
    "AnalysisMode",
    "CHAT_SYSTEM_PROMPT",
    "ChatMessage",
    "ChatRequest",
    "IntegrationDiagnostic",
    "IntegrationError",
    "IntegrationErrorCode",
    "ManagementOutlookDTO",
    "PipelineMetadataDTO",
    "TokenAttributionDTO",
    "TTSInput",
    "VerificationDTO",
    "VerificationIssueDTO",
    "build_analysis_handoff",
    "build_chat_context",
    "build_tts_input",
    "citation_index",
    "diagnose_integration_failure",
    "map_integration_error",
]
