"""Tests for SQLite storage layer."""

import tempfile
import sqlite3

import numpy as np
import pytest


class TestSQLiteStorage:
    """Tests for SQLiteStorage."""

    def test_create_tables(self):
        """SQLiteStorage creates tables correctly."""
        from plm.search.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            storage = SQLiteStorage(f.name)
            storage.create_tables()

            conn = sqlite3.connect(f.name)
            tables = [t[0] for t in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()]
            conn.close()

            assert 'documents' in tables
            assert 'chunks' in tables

    def test_add_document(self):
        """SQLiteStorage adds documents correctly."""
        from plm.search.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            storage = SQLiteStorage(f.name)
            storage.create_tables()
            storage.add_document('doc1', 'test.md')

            # Verify document was added
            conn = sqlite3.connect(f.name)
            result = conn.execute(
                "SELECT id, source_file FROM documents WHERE id = ?",
                ('doc1',)
            ).fetchone()
            conn.close()

            assert result is not None
            assert result[0] == 'doc1'
            assert result[1] == 'test.md'

    def test_embedding_roundtrip(self):
        """SQLiteStorage stores and retrieves embeddings correctly."""
        from plm.search.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            storage = SQLiteStorage(f.name)
            storage.create_tables()
            storage.add_document('doc1', 'test.md')

            # Create random embedding
            embedding = np.random.rand(768).astype(np.float32)
            storage.add_chunk(
                'chunk1', 'doc1', 'Test content',
                embedding=embedding, enriched_content='enriched'
            )

            # Retrieve and verify
            retrieved = storage.get_chunk_by_id('chunk1')
            assert retrieved is not None
            assert np.allclose(retrieved['embedding'], embedding)

    def test_get_all_chunks(self):
        """SQLiteStorage retrieves all chunks."""
        from plm.search.storage.sqlite import SQLiteStorage

        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            storage = SQLiteStorage(f.name)
            storage.create_tables()
            storage.add_document('doc1', 'test.md')

            embedding = np.random.rand(768).astype(np.float32)
            storage.add_chunk('chunk1', 'doc1', 'Content 1', embedding=embedding, enriched_content='enriched1')
            storage.add_chunk('chunk2', 'doc1', 'Content 2', embedding=embedding, enriched_content='enriched2')

            chunks = storage.get_all_chunks()
            assert len(chunks) == 2
