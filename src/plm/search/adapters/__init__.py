"""Adapters for integrating extraction systems with search pipeline."""

from .gliner_adapter import (
    document_result_to_chunks,
    gliner_to_enricher_format,
    load_extraction_directory,
)

__all__ = [
    "document_result_to_chunks",
    "gliner_to_enricher_format",
    "load_extraction_directory",
]
