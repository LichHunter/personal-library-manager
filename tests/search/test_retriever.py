"""Tests for HybridRetriever orchestrator."""

import tempfile
import os

import pytest


class TestHybridRetriever:
    """Tests for HybridRetriever."""

    def test_instantiation(self):
        """HybridRetriever can be instantiated."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f'{tmpdir}/test.db',
                f'{tmpdir}/bm25_index'
            )
            assert retriever is not None

    @pytest.mark.slow
    def test_ingest_and_retrieve(self):
        """HybridRetriever ingests and retrieves documents."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f'{tmpdir}/test.db',
                f'{tmpdir}/bm25_index'
            )
            retriever.ingest_document('doc1', 'test.md', [
                {
                    'content': 'Kubernetes horizontal pod autoscaler scales pods',
                    'heading': '## HPA',
                    'keywords': ['kubernetes', 'autoscaler'],
                    'entities': {}
                }
            ])
            results = retriever.retrieve('kubernetes autoscaling', k=1)
            assert len(results) > 0

    @pytest.mark.slow
    def test_bm25_persistence(self):
        """BM25 index persists across restarts."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = f'{tmpdir}/test.db'
            bm25_path = f'{tmpdir}/bm25_index'

            # First session: ingest
            retriever1 = HybridRetriever(db_path, bm25_path)
            retriever1.ingest_document('doc1', 'test.md', [
                {
                    'content': 'Kubernetes pod definition',
                    'heading': '## K8s',
                    'keywords': ['kubernetes'],
                    'entities': {}
                }
            ])
            del retriever1

            # Verify BM25 index was saved
            assert os.path.exists(bm25_path)

            # Second session: retrieve without re-indexing
            retriever2 = HybridRetriever(db_path, bm25_path)
            results = retriever2.retrieve('kubernetes', k=1)
            assert len(results) > 0
