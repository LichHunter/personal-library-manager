"""Unit tests for queue module."""

import os
import pytest

from plm.shared.queue import (
    MessageQueue,
    NullQueue,
    create_queue,
    ExtractionMessage,
    MessageEnvelope,
)
from plm.shared.queue.types import Entity, Chunk, HeadingSection


class TestTypes:
    """Tests for queue type definitions."""

    def test_entity_type(self):
        """Entity TypedDict has correct fields."""
        entity: Entity = {
            "text": "Kubernetes",
            "label": "technology",
            "score": 0.9,
            "start": 0,
            "end": 10,
        }
        assert entity["text"] == "Kubernetes"
        assert entity["label"] == "technology"
        assert entity["score"] == 0.9

    def test_chunk_type(self):
        """Chunk TypedDict has correct fields."""
        chunk: Chunk = {
            "text": "Kubernetes is a container orchestration platform.",
            "terms": ["Kubernetes"],
            "entities": [{"text": "Kubernetes", "label": "technology", "score": 0.9, "start": 0, "end": 10}],
            "keywords": ["container", "orchestration"],
            "start_char": 0,
            "end_char": 48,
        }
        assert chunk["text"].startswith("Kubernetes")
        assert len(chunk["entities"]) == 1

    def test_extraction_message_type(self):
        """ExtractionMessage has all required fields."""
        msg: ExtractionMessage = {
            "source_file": "/test/doc.md",
            "headings": [],
            "avg_confidence": 0.85,
            "total_entities": 5,
            "is_low_confidence": False,
            "error": None,
        }
        assert msg["source_file"] == "/test/doc.md"
        assert msg["avg_confidence"] == 0.85

    def test_message_envelope_type(self):
        """MessageEnvelope wraps ExtractionMessage correctly."""
        payload: ExtractionMessage = {
            "source_file": "/test/doc.md",
            "headings": [],
            "avg_confidence": 0.85,
            "total_entities": 5,
            "is_low_confidence": False,
            "error": None,
        }
        envelope: MessageEnvelope = {
            "message_id": "abc-123",
            "timestamp": "2026-01-01T00:00:00Z",
            "source_service": "fast-extraction",
            "payload": payload,
        }
        assert envelope["message_id"] == "abc-123"
        assert envelope["source_service"] == "fast-extraction"
        assert envelope["payload"]["source_file"] == "/test/doc.md"


class TestNullQueue:
    """Tests for NullQueue implementation."""

    def test_null_queue_satisfies_protocol(self):
        """NullQueue satisfies MessageQueue protocol."""
        queue = NullQueue()
        assert isinstance(queue, MessageQueue)

    def test_null_queue_is_not_available(self):
        """NullQueue.is_available() returns False."""
        queue = NullQueue()
        assert queue.is_available() is False

    def test_null_queue_publish_returns_none(self):
        """NullQueue.publish() returns None."""
        queue = NullQueue()
        result = queue.publish("test:stream", {"data": "test"})
        assert result is None


class TestFactory:
    """Tests for queue factory function."""

    def test_factory_returns_null_queue_when_disabled(self, monkeypatch):
        """Factory returns NullQueue when QUEUE_ENABLED=false."""
        monkeypatch.setenv("QUEUE_ENABLED", "false")
        queue = create_queue()
        assert isinstance(queue, NullQueue)

    def test_factory_returns_null_queue_by_default(self, monkeypatch):
        """Factory returns NullQueue when QUEUE_ENABLED not set."""
        monkeypatch.delenv("QUEUE_ENABLED", raising=False)
        queue = create_queue()
        assert isinstance(queue, NullQueue)


class TestFormatTransformer:
    """Tests for slow extraction format transformer."""

    def test_transform_basic(self):
        """Basic transformation from slow to fast format."""
        from plm.extraction.slow.format_transformer import transform_to_fast_format

        slow = {
            "file": "test.md",
            "processed_at": "2026-01-01T00:00:00Z",
            "chunks": [{
                "text": "Kubernetes is cool",
                "chunk_index": 0,
                "heading": None,
                "terms": [{"term": "Kubernetes", "confidence": 0.9, "level": "HIGH", "sources": ["v6"]}]
            }]
        }
        fast = transform_to_fast_format(slow)

        assert "source_file" in fast
        assert "headings" in fast
        assert fast["source_file"] == "test.md"
        assert len(fast["headings"]) == 1
        assert len(fast["headings"][0]["chunks"]) == 1
        assert fast["headings"][0]["chunks"][0]["entities"][0]["text"] == "Kubernetes"

    def test_transform_with_source_file_override(self):
        """Transformation with explicit source_file."""
        from plm.extraction.slow.format_transformer import transform_to_fast_format

        slow = {
            "file": "original.md",
            "chunks": []
        }
        fast = transform_to_fast_format(slow, source_file="/full/path/to/doc.md")
        assert fast["source_file"] == "/full/path/to/doc.md"

    def test_transform_empty_chunks(self):
        """Transformation with no chunks."""
        from plm.extraction.slow.format_transformer import transform_to_fast_format

        slow = {"file": "empty.md", "chunks": []}
        fast = transform_to_fast_format(slow)
        assert fast["headings"] == []
        assert fast["total_entities"] == 0
        assert fast["avg_confidence"] == 0.0
