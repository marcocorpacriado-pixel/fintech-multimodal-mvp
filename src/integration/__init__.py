"""Public handoff contracts between analysis, API/UI, and audio modules."""

from .analysis_handoff import (
    AnalysisEvidenceDTO,
    AnalysisHandoff,
    AnalysisMetricDTO,
    AnalysisMode,
    ManagementOutlookDTO,
    PipelineMetadataDTO,
    TTSInput,
    VerificationDTO,
    VerificationIssueDTO,
    build_analysis_handoff,
    build_tts_input,
)
from .errors import IntegrationError, IntegrationErrorCode, map_integration_error

__all__ = [
    "AnalysisEvidenceDTO",
    "AnalysisHandoff",
    "AnalysisMetricDTO",
    "AnalysisMode",
    "IntegrationError",
    "IntegrationErrorCode",
    "ManagementOutlookDTO",
    "PipelineMetadataDTO",
    "TTSInput",
    "VerificationDTO",
    "VerificationIssueDTO",
    "build_analysis_handoff",
    "build_tts_input",
    "map_integration_error",
]
