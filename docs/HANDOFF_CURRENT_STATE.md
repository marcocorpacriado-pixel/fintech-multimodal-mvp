# Current integrated state

- Main integration commit: `d4acee6ab909d3e875877540f5bdc099a71559c6`
  (the handoff note itself is a later documentation-only commit).
- Integration branch: `integration/final-r12` at
  `a62fd92e31f84b504c9794ca139dfc94fa684c2d`.
- Certified R12.2 checkpoint: `44400bfd6add140cc50922b6fe64f1add061b05c`.
- Validation: 517 tests passed with 4 dependency deprecation warnings;
  `compileall`, `pip check`, and `git diff --check` passed.
- Architecture: SEC narrative and XBRL inputs feed deterministic loading,
  chunking, BM25 retrieval and canonical metrics; a grounded structured LLM
  selects deterministic evidence IDs; the backend reconstructs exact evidence,
  applies deterministic verification, and exposes a serializable handoff to
  FastAPI, Streamlit, and TTS.

# Completed

- R11 real-mode stabilization and live SEC filing discovery.
- R12 professional UX and explicit real/demo behavior.
- R12.1 presentation hardening.
- R12.2 deterministic evidence catalog and `evidence_id` contract.
- Backend-side exact evidence/source/section reconstruction.
- At most one bounded content repair.
- Total OpenRouter generation deadline, including transport retries.
- Marco redesign integration: compatible dark-theme configuration retained;
  legacy static catalog, legacy date contract, relaxed grounding, model
  recommendation, and demo autoload were intentionally excluded.

# Pending

- R13 grounded financial chat.
- R14 STT voice copilot.
- Final manual visual review and optional additional Marco styling port.
- Final end-to-end presentation smoke and demo script/README polish.
- Optional provider/model benchmark only if future evidence justifies it.

Earnings-call STT is **not** integrated into the analysis pipeline. The current
STT module can later provide voice input for grounded chat. Real mode never
falls back silently to demo: it depends on SEC access and the configured
OpenRouter provider/model. Demo remains explicitly synthetic and deterministic.
