"""Split normalized financial documents into traceable text chunks.

Future responsibility:
- apply section-aware chunking without losing source offsets;
- emit ``DocumentChunk`` instances from ``extraction.schemas``;
- keep chunking policy independent from retrieval strategy.

TODO: choose chunk boundaries and overlap after the document loader contract is
stable and the narrative filing corpus has been profiled.
"""
