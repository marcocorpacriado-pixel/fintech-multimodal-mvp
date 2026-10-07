# D8C analysis handoff

This document defines the stable boundary between Dani's analysis pipeline,
Marco's API/UI, and Cristian's audio module. It intentionally contains no UI,
HTTP endpoint, model invocation, or financial calculation.

## Responsibilities

- **Dani (`src/extraction/`)**: SEC ingestion, chunking, retrieval, canonical
  metrics, grounded qualitative analysis, citations, and deterministic
  verification.
- **Marco (`src/api/`, `src/visualization/`, `app/`)**: HTTP orchestration,
  presentation, charts, and user experience. These layers display the handoff;
  they do not recompute metrics or call analysis models directly.
- **Cristian (`src/audio/`)**: STT and TTS. TTS receives only the final summary;
  STT produces transcript text and timing metadata.

## Pipeline to Marco

Use the adapter only after `run_analysis_pipeline` returns successfully:

```python
from datetime import date

from src.integration import build_analysis_handoff

handoff = build_analysis_handoff(
    pipeline_result,
    analysis_mode="real",
    provider="openrouter",
    model="deepseek/deepseek-v4-flash",
    filing_date=date(2026, 7, 31),
)
payload = handoff.model_dump(mode="json")
```

`analysis_mode` is mandatory and is either `real` or `demo`. A deterministic
fixture/FakeLLM must always use `demo`; it must never be presented as real
inference.

Reduced JSON shape:

```json
{
  "company": "Apple Inc.",
  "ticker": "AAPL",
  "period": "2026-06-27",
  "filing_type": "10-Q",
  "financial_metrics": [
    {
      "name": "Revenue",
      "current_value": 109417000000.0,
      "previous_value": 111184000000.0,
      "change_pct": -1.5892574471,
      "unit": "usd",
      "comparison_type": "QoQ",
      "current_period": "2026-03-29/2026-06-27",
      "previous_period": "2025-12-28/2026-03-28",
      "source_ids": ["xbrl:current", "xbrl:previous"]
    }
  ],
  "positives": [
    {
      "finding": "...",
      "evidence": "verbatim contiguous source excerpt",
      "source_section": "PART_I_ITEM_2",
      "source_id": "sec-filing:chunk-id",
      "source_type": "filing"
    }
  ],
  "risks": [],
  "management_outlook": {
    "summary": "Insufficient evidence for explicit guidance.",
    "sentiment": "unknown",
    "source_ids": []
  },
  "executive_summary": "...",
  "verification": {
    "valid": true,
    "issues": []
  },
  "pipeline_metadata": {
    "analysis_mode": "real",
    "provider": "openrouter",
    "model": "deepseek/deepseek-v4-flash",
    "filing_date": "2026-07-31",
    "effective_queries": ["revenue operating performance"],
    "retrieval_count": 1,
    "retrieved_source_ids": ["sec-filing:chunk-id"]
  }
}
```

Marco may build metric cards and charts directly from `financial_metrics`.
`change_pct`, period comparability, units, and `comparison_type` are canonical
D5C output and must not be recalculated. The UI must not query XBRL, run BM25,
interpret source identifiers, parse filings, or call the LLM.

`pipeline_metadata.filing_date` is the SEC submission date selected by the
request. Top-level `period` is the financial report period; consumers must not
treat them as interchangeable. A selector can load lightweight metadata from
`GET /api/v1/filings/{ticker}` before submitting the analysis request.

Verification warnings can be displayed without blocking the response.
Pipeline verification errors are raised before a handoff is returned.

## Pipeline to Cristian (TTS)

```python
from src.audio import synthesize
from src.integration import build_tts_input

tts_input = build_tts_input(pipeline_result, analysis_mode="real")
audio = synthesize(tts_input.text, language="en")
```

The `text` field is byte-for-byte/string-equivalent to
`analysis.executive_summary`. The remaining fields (`company`, `ticker`,
`period`, `filing_type`, `analysis_mode`, and `verification_valid`) are compact
display/audit metadata. Financial metrics and evidence are deliberately absent
from the TTS contract.

The integration adapter does not import `src.audio`; this keeps Kokoro/Groq
dependencies optional and owned by Cristian's module.

## Cristian STT to future earnings-call retrieval

The future flow is intentionally additive:

```text
earnings-call audio
  -> src.audio.transcribe(...)
  -> Transcription.text (+ segment timestamps)
  -> transcript loader/chunker adapter
  -> DocumentChunk-compatible records with source_type="earnings_call"
  -> retrieval
  -> grounded analysis and verification
```

Transcript chunks need deterministic `source_id`/`chunk_id`, ticker, period,
offset or timestamp provenance, and `source_type="earnings_call"`. Filing
chunks continue to use `source_type="filing"`; no existing type is replaced.
This D8C phase does not implement that ingestion path.

## API-safe errors

Use `map_integration_error(error)` at the API boundary. It returns only:

```json
{
  "code": "LLM_PROVIDER_ERROR",
  "message": "The language-model provider could not complete the request.",
  "retryable": true
}
```

Stable categories are:

| Code | Typical source | Retryable policy |
|---|---|---|
| `INPUT_ERROR` | invalid request, ticker, path or pipeline input | no (HTTP 422) |
| `FILING_NOT_FOUND` | target or comparable SEC filing absent | no (HTTP 404) |
| `SEC_INGESTION_ERROR` | SEC service/configuration/data preparation | transient service failures only (HTTP 503) |
| `ANALYSIS_ERROR` | XBRL normalization/metric or analysis failure | no |
| `LLM_PROVIDER_ERROR` | provider transport/response | yes, except configuration |
| `GROUNDING_ERROR` | invalid citation/evidence | no |
| `VERIFICATION_ERROR` | deterministic verifier rejection | no |
| `UNKNOWN_ERROR` | unclassified failure | no |

Messages are fixed by category. They never reflect provider response bodies,
prompts, API keys, authorization headers, source documents, or stack traces.

## Real and demo operation

- **Real:** pass `analysis_mode="real"` and, when available, provider/model
  labels. Provider errors are surfaced; there is no silent fallback.
- **Demo:** run the separately selected FakeLLM/validated fixture and pass
  `analysis_mode="demo"`. UI and audio must retain that label.

Fallback selection belongs in Marco's orchestration layer or deployment
configuration, never inside `build_analysis_handoff` and never as an automatic
response to a failed real call.
