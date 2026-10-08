# Cost and latency evidence

## 1. Purpose and evidence policy

This document separates observed measurements from configuration limits and
planning estimates. A timeout or deadline is not reported as observed latency,
and mocked token/cost values are not reported as production spend.

Evidence labels:

- **MEASURED:** captured from a reproducible test, provider response, or
  controlled audit with traceable provenance.
- **CONFIGURED:** a code or infrastructure limit, not an observed duration.
- **ESTIMATED:** a planning assumption or formula that still requires
  measurement.
- **NOT YET MEASURED:** no admissible evidence is currently retained.
- **HISTORICAL OBSERVATION — NOT VERSIONED:** a value recorded in prior task
  notes but not backed by a committed, independently auditable artifact.

## 2. Evidence found in the repository and final audit

| Item | Classification | Evidence | Interpretation |
|---|---|---|---|
| Current test suite | MEASURED | 613 passed, 4 skipped, 0 failed in 9.13 s after integrating Groq TTS at `6f336be` | Engineering regression time, not product latency. |
| Pre-Groq-TTS snapshot | MEASURED | 585 passed, 4 skipped, 0 failed in 7.64 s at the documentation audit base | Earlier engineering regression time, not product latency. |
| Previous forensic snapshot | MEASURED | 583 passed, 4 skipped, 0 failed in 11.73 s at `afead47` | Earlier snapshot before floating-chat tests. |
| Docker/Cloud Run workflow | MEASURED | Current-head workflow completed successfully and ran its health smoke | Confirms build/deploy/run, but does not measure end-user analysis latency. |
| OpenRouter usage parsing | MEASURED BY TEST ONLY | Unit test preserves mocked prompt/completion/total tokens and cost | Demonstrates capture capability; the values are not real spend. |
| Kokoro TTS output | MEASURED FUNCTIONALLY | Warm local audit produced a valid `audio/wav` response | No isolated, repeatable latency benchmark was retained. |

No paid provider call was made for this documentation phase.

### Pitch-ready evidence snapshot

| Signal | Status | Defensible statement |
|---|---|---|
| Regression suite | MEASURED | 613 passed, 4 skipped, 0 failed on the audited local environment. |
| Docker/Cloud Run | MEASURED | Build, deployment, and health smoke completed; no end-user latency benchmark was retained. |
| Historical real-analysis runs | HISTORICAL OBSERVATION — NOT VERSIONED | Candidate runs completed in roughly 25–29 seconds, but they are not certified cost/latency evidence. |
| Analysis reliability ceiling | CONFIGURED | 90-second default deadline per generation and at most one content repair; this is not observed latency. |
| Provider/cloud unit cost | NOT YET MEASURED | Billing-backed unit economics remain open. |
| FinBERT, IG, and Kokoro | ESTIMATED COST CLASS | Local/container compute rather than a per-call model API fee; runtime cost is not yet measured. |
| Optional Groq TTS | NOT YET MEASURED | External English TTS usage; no project billing or latency evidence is retained. |

## 3. Historical R12.2 figures awaiting provenance

The following candidate measurements were supplied in historical task notes,
but searches of the current tree and Git history found no committed report,
log, fixture, or benchmark artifact that independently verifies them.
Consequently they are **HISTORICAL OBSERVATION — NOT VERSIONED**, not
certified measurements.

| Filing date | Candidate SEC | Candidate LLM | Candidate total | Candidate tokens | Candidate cost | Status |
|---|---:|---:|---:|---:|---:|---|
| 2024-08-02 | 1.314 s | 23.324 s | 24.868 s | 13,980 | $0.00067599 | HISTORICAL OBSERVATION — NOT VERSIONED |
| 2023-08-04 | 1.115 s | 11.034 s + 16.539 s | 28.929 s | 32,812 | $0.003188032 | HISTORICAL OBSERVATION — NOT VERSIONED |

To promote these rows to **MEASURED**, retain the sanitized smoke command or
report, filing/accession, model, number of generation attempts, provider usage
payload, timestamp, and environment/commit. Prompts, raw model output, evidence
text, credentials, and authorization headers must not be retained.

## 4. Configured latency and reliability limits

| Component | Setting | Classification | Meaning |
|---|---:|---|---|
| OpenRouter analysis HTTP timeout | 30 s | CONFIGURED | Per-operation transport timeout. |
| OpenRouter total generation deadline | 90 s default | CONFIGURED | One deadline shared by transport attempts and backoff. |
| OpenRouter maximum total deadline | 600 s | CONFIGURED | Upper bound accepted from configuration. |
| OpenRouter transport retries | 2 retries / 3 attempts | CONFIGURED | Attempts remain inside one generation deadline. |
| Content generations | maximum 2 | CONFIGURED | Initial generation plus at most one bounded repair. |
| Default maximum content-generation budget | approximately 180 s | ESTIMATED UPPER BOUND | Two independent 90-second generation deadlines; local stages add time. |
| Streamlit analysis HTTP timeout | 300 s | CONFIGURED | UI request budget, not expected duration. |
| Streamlit chat read timeout | 120 s | CONFIGURED | Does not constitute a full provider benchmark. |
| Streamlit STT timeout | 60 s | CONFIGURED | UI request budget. |
| Streamlit TTS timeout | 300 s | CONFIGURED | Allows model load/cold start. |
| Cloud Run request timeout | 600 s | CONFIGURED | Infrastructure ceiling. |
| Groq upload limit | 25 MB | CONFIGURED | Capacity limit, not latency. |
| Groq TTS chunk size | 200 characters | CONFIGURED | English text is split for the provider; not an observed performance figure. |
| Groq TTS parallel requests | maximum 4 | CONFIGURED | Provider calls per synthesis are bounded; not measured latency. |
| Integrated Gradients steps | 32 | CONFIGURED | Computation depth, not observed latency. |

The analysis deadline applies to each structured OpenRouter generation. The
streaming filing-chat client currently uses transport/read timeouts but does
not implement the same total-generation deadline.

## 5. Cost model

### Variable provider costs

```text
LLM_analysis_cost =
    sum(provider_cost for each completed analysis generation)

optional_chat_cost =
    sum(provider_cost for each chat turn)

optional_STT_cost =
    transcribed_audio_duration * provider_rate_per_duration_unit

optional_Groq_TTS_cost =
    sum(provider_charge for each external TTS chunk/request)
```

The OpenRouter analysis client parses `prompt_tokens`, `completion_tokens`,
`total_tokens`, and optional `cost` from a successful response. The current
pipeline does not persist a complete multi-attempt cost record in
`AnalysisHandoff`; `last_usage` represents only the latest successful client
response. Streaming chat usage is not currently retained.

### Local-model and infrastructure costs

FinBERT, Integrated Gradients, and Kokoro have no per-call external model fee
in the current architecture, but they consume Cloud Run CPU/memory time and
increase container size and build/storage requirements.

```text
allocated_cloud_compute =
    vCPU_seconds * applicable_vCPU_rate
  + GiB_seconds * applicable_memory_rate
  + request_charges

storage_and_network =
    artifact_registry_GiB_month
  + build_cache_storage
  + applicable_network_egress
  + applicable_logging_or_secret_management_cost
```

All rates are **TO BE MEASURED** against the deployed project, region, free
tier, and current provider contracts. This document intentionally does not
invent precise Cloud Run, OpenRouter, or Groq prices.

### Unit and monthly formulas

```text
Cost_per_filing =
    LLM_analysis_cost
  + optional_chat_cost
  + optional_STT_cost
  + optional_Groq_TTS_cost
  + allocated_cloud_compute

Monthly_cost =
    N_filings * variable_cost_per_filing
  + cloud_fixed_or_minimum
  + optional_voice_usage
  + storage_and_network
```

## 6. Measurement status by stage

| Stage | Status | Required next evidence |
|---|---|---|
| SEC discovery/ingestion | NOT YET MEASURED | Separate discovery, narrative, and XBRL timings by accession. |
| BM25/chunking/metrics | NOT YET MEASURED | Local stage timers on representative filings. |
| OpenRouter analysis | PARTIALLY CAPTURABLE | Persist sanitized per-attempt latency and usage/cost. |
| Bounded repair overhead | NOT YET MEASURED | Compare one-attempt and two-attempt filings. |
| FinBERT classification | NOT YET MEASURED | Warm and cold CPU timings. |
| Integrated Gradients | NOT YET MEASURED | 32-step CPU timing and text/token length. |
| Filing chat | NOT YET MEASURED | Time to first token, total time, tokens, and provider cost. |
| Groq STT | NOT YET MEASURED | Audio duration, upload size, provider latency, and billed usage. |
| Kokoro TTS | FUNCTIONALLY MEASURED ONLY | Cold/warm latency, text length, audio duration, and peak memory. |
| Groq TTS | NOT YET MEASURED | Text/chunk count, latency, provider usage, and billed cost. |
| End-to-end real analysis | NOT YET CERTIFIED IN REPO | SEC, local stages, LLM attempt(s), verifier, and total duration. |
| Cloud infrastructure | NOT YET MEASURED | Billing export or provider console by service/SKU. |

## 7. Recommended measurement protocol

1. Use a fixed set of filing accessions and record the exact Git commit,
   configured model, region, and analysis mode.
2. Capture monotonic timings for SEC preparation, deterministic local stages,
   each LLM generation, FinBERT, Integrated Gradients, verification, and total
   request time.
3. Record only sanitized usage metadata: attempt number, token counts, provider
   cost, duration, success category, and repair usage.
4. Benchmark cold and warm FinBERT/Kokoro separately.
5. For STT/TTS, record provider, input duration/text length, Groq chunk count,
   and output duration without retaining personal audio, transcript, or speech
   text content.
6. Report median and p95 across repeated representative runs; do not infer an
   SLA from a single successful request.
7. Reconcile provider usage with Cloud Run and provider billing before using
   the numbers in pricing or unit-economics claims.

## 8. Current conclusion

The repository contains the mechanisms needed to capture part of the LLM usage
and has explicit reliability ceilings, but it does not yet contain a complete,
auditable product-latency or cost dataset. Economic claims should remain
labelled **ESTIMATED / TO BE MEASURED** until the measurement protocol above is
executed and its sanitized results are versioned.
