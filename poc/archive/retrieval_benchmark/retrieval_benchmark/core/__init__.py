"""Core types and utilities for retrieval benchmark."""

from .types import (
    Document,
    Section,
    Chunk,
    GroundTruth,
    SearchHit,
    SearchStats,
    SearchResponse,
    IndexStats,
    QueryResult,
    StrategySummary,
)
from .loader import load_documents, load_ground_truth

__all__ = [
    "Document",
    "Section", 
    "Chunk",
    "GroundTruth",
    "SearchHit",
    "SearchStats",
    "SearchResponse",
    "IndexStats",
    "QueryResult",
    "StrategySummary",
    "load_documents",
    "load_ground_truth",
]
