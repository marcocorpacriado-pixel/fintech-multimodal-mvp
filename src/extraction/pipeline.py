"""Orchestrate the extraction and financial analysis stages.

Future responsibility:
- coordinate loading, chunking, retrieval, and analysis;
- validate the final ``FinancialAnalysisResult`` at the module boundary;
- remain independent from API, visualization, and audio concerns.

TODO: define orchestration only after the individual stage contracts are stable.
"""
