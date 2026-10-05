"""Load and normalize source documents for the extraction pipeline.

Future responsibility:
- read narrative SEC filings, XBRL datasets, and earnings-call transcripts;
- preserve source metadata needed for traceability;
- return normalized documents to the chunking stage.

TODO(D2): define the loader input/output contract after inspecting the actual
filing corpus and filename/metadata conventions.
"""
