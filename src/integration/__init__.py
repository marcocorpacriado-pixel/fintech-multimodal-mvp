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
    "build_tts_input",
    "diagnose_integration_failure",
    "map_integration_error",
]
