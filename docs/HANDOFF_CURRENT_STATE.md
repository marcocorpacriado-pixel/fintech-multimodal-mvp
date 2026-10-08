# Current integrated state

This is the current operational handoff. Earlier R11/R12 checkpoints remain in
Git history; they are not the current product baseline.

- Functional audit base before this documentation-only reconciliation:
  `6f336be9796ff616dcc277a3b2ef866761edd0e9`.
- Certified R12.2 checkpoint: `44400bfd6add140cc50922b6fe64f1add061b05c`.
- Validation: **613 passed, 4 skipped, 0 failed, 4 warnings**;
  `compileall`, `pip check`, and `git diff --check` passed.
- The four skips are real Integrated Gradients tests that require `torch`,
  which was absent from the audited local environment. The Docker image and
  `requirements.txt` include PyTorch.
- The current Cloud Run build/deployment workflow and health smoke succeeded.

## Current architecture

SEC narrative and XBRL inputs feed deterministic loading, SEC-aware chunking,
BM25 retrieval, and seven canonical metrics. A structured LLM selects IDs from
a deterministic evidence catalog; the backend reconstructs literal evidence,
source, and section. FinBERT and Integrated Gradients enrich the grounded
outlook before deterministic verification produces `AnalysisHandoff` for
FastAPI, Dash/Plotly, filing chat, local Kokoro TTS, and optional Groq
Orpheus TTS for English speech.

The product UI has four result tabs—Overview, Financials, Narrative, and
Sources—plus the floating **Ask about this filing** experience.

## Completed

- Real SEC filing discovery and distinct filing/report dates.
- Seven canonical XBRL metrics, retrieval, structured analysis, grounding,
  bounded one-shot repair, and deterministic verification.
- Backend-side evidence/source/section reconstruction from `evidence_id`.
- FinBERT financial sentiment and Integrated Gradients local explanation.
- FastAPI, Dash, Plotly, explicit real/demo modes, and safe errors.
- Grounded filing chat over verified handoff context.
- Groq STT as voice input to chat, Kokoro as the default local TTS, and a
  selectable Groq Orpheus path for English chat audio. Spanish speech routes
  to Kokoro.
- Multi-stage Docker image and Cloud Run deployment.

## Current caveats and future work

- Chat citations, abstention, and recommendation blocking are prompt-enforced;
  there is no deterministic post-generation citation verifier yet.
- STT feeds filing chat. Earnings-call audio/transcripts are **not** ingested
  into the SEC retrieval/analysis pipeline.
- Real mode depends on SEC and configured external providers and never falls
  back silently to the deterministic synthetic demo.
- Groq TTS sends normalized speech text to an external provider when selected;
  a provider failure returns an error rather than falling back to Kokoro.
- Product cost/latency measurements, enterprise retention/governance controls,
  and the four skipped local IG tests remain open validation work.

See [`BUSINESS_CASE.md`](BUSINESS_CASE.md) and
[`COST_LATENCY.md`](COST_LATENCY.md) for the business and measurement status.
