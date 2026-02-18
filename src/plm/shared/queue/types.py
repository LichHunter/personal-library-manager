"""Canonical message schema types for queue integration.

Defines the structure of messages flowing through the extraction pipeline:
- ExtractionMessage: The core extraction result (from fast/slow extraction)
- MessageEnvelope: Wrapper with metadata (message_id, timestamp, source_service)

These types are used by:
- Fast extraction (src/plm/extraction/fast/cli.py) - produces ExtractionMessage
- Slow extraction (src/plm/extraction/slow/cli.py) - produces ExtractionMessage
- Search service watcher (src/plm/search/service/watcher.py) - consumes ExtractionMessage
- Queue integration (future) - wraps in MessageEnvelope for transport
"""

from __future__ import annotations

from typing import TypedDict


class Entity(TypedDict):
    """Extracted entity with confidence score and position."""
    text: str
    label: str
    score: float
    start: int
    end: int


class Chunk(TypedDict):
    """Text chunk with extracted entities and keywords."""
    text: str
    terms: list[str]
    entities: list[Entity]
    keywords: list[str]
    start_char: int
    end_char: int


class HeadingSection(TypedDict):
    """Document section under a heading with chunks."""
    heading: str
    level: int
    chunks: list[Chunk]


class ExtractionMessage(TypedDict):
    """Core extraction result from fast or slow extraction pipeline.
    
    This is the canonical format produced by:
    - Fast extraction: src/plm/extraction/fast/cli.py:_serialize_result()
    - Slow extraction: src/plm/extraction/slow/cli.py (future)
    
    Consumed by:
    - Search service watcher: src/plm/search/service/watcher.py:json_to_chunks()
    """
    source_file: str
    headings: list[HeadingSection]
    avg_confidence: float
    total_entities: int
    is_low_confidence: bool
    error: str | None


class MessageEnvelope(TypedDict):
    """Wrapper for ExtractionMessage with transport metadata.
    
    Used by queue integration to track message provenance and timing.
    """
    message_id: str
    timestamp: str  # ISO 8601 format
    source_service: str  # e.g., "fast-extraction", "slow-extraction"
    payload: ExtractionMessage
