"""Tests for explain mode in search API and retriever."""

import tempfile
import uuid

import pytest
from fastapi.testclient import TestClient


class TestExplainModeRetriever:
    """Tests for HybridRetriever explain mode."""

    @pytest.mark.slow
    def test_retrieve_explain_false_returns_list(self):
        """When explain=False, retrieve returns a plain list."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f"{tmpdir}/test.db",
                f"{tmpdir}/bm25_index",
            )
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Kubernetes horizontal pod autoscaler scales pods",
                        "heading": "## HPA",
                        "keywords": ["kubernetes", "autoscaler"],
                        "entities": {},
                    }
                ],
            )
            results = retriever.retrieve("kubernetes autoscaling", k=1, explain=False)
            assert isinstance(results, list)
            assert len(results) > 0
            assert "chunk_id" in results[0]

    @pytest.mark.slow
    def test_retrieve_explain_true_returns_tuple(self):
        """When explain=True, retrieve returns tuple of (results, explain_data)."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f"{tmpdir}/test.db",
                f"{tmpdir}/bm25_index",
            )
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Kubernetes horizontal pod autoscaler scales pods",
                        "heading": "## HPA",
                        "keywords": ["kubernetes", "autoscaler"],
                        "entities": {},
                    }
                ],
            )
            result = retriever.retrieve("kubernetes autoscaling", k=1, explain=True)
            assert isinstance(result, tuple)
            assert len(result) == 2
            results, explain_data = result
            assert isinstance(results, list)
            assert isinstance(explain_data, dict)

    @pytest.mark.slow
    def test_explain_data_has_metadata(self):
        """Explain data contains metadata with query parameters."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f"{tmpdir}/test.db",
                f"{tmpdir}/bm25_index",
            )
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Kubernetes horizontal pod autoscaler scales pods",
                        "heading": "## HPA",
                        "keywords": ["kubernetes", "autoscaler"],
                        "entities": {},
                    }
                ],
            )
            results, explain_data = retriever.retrieve(
                "kubernetes autoscaling", k=1, explain=True
            )

            assert "metadata" in explain_data
            metadata = explain_data["metadata"]
            assert metadata["original_query"] == "kubernetes autoscaling"
            assert "retrieval_mode" in metadata
            assert metadata["retrieval_mode"] in ("hybrid", "splade_only")
            assert "rrf_k" in metadata
            assert "bm25_weight" in metadata
            assert "semantic_weight" in metadata
            assert "expanded_terms" in metadata
            assert isinstance(metadata["expanded_terms"], list)

    @pytest.mark.slow
    def test_explain_data_has_debug_info_per_result(self):
        """Explain data contains debug_info keyed by chunk_id."""
        from plm.search.retriever import HybridRetriever

        with tempfile.TemporaryDirectory() as tmpdir:
            retriever = HybridRetriever(
                f"{tmpdir}/test.db",
                f"{tmpdir}/bm25_index",
            )
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Kubernetes horizontal pod autoscaler scales pods",
                        "heading": "## HPA",
                        "keywords": ["kubernetes", "autoscaler"],
                        "entities": {},
                    }
                ],
            )
            results, explain_data = retriever.retrieve(
                "kubernetes autoscaling", k=1, explain=True
            )

            assert "debug_info" in explain_data
            assert len(results) > 0
            chunk_id = results[0]["chunk_id"]
            assert chunk_id in explain_data["debug_info"]

            debug = explain_data["debug_info"][chunk_id]
            assert "rrf_score" in debug
            assert "retrieval_stage" in debug
            assert debug["retrieval_stage"] in ("rrf", "rerank")
            assert "bm25_score" in debug
            assert "semantic_score" in debug
            assert "bm25_rank" in debug
            assert "semantic_rank" in debug


class TestExplainModeAPI:
    """Tests for explain mode in FastAPI endpoint."""

    @pytest.fixture
    def client_with_data(self):
        """Create test client with ingested data."""
        import os
        import tempfile

        os.environ["INDEX_PATH"] = tempfile.mkdtemp()
        os.environ["WATCH_DIR"] = ""
        os.environ["QUEUE_ENABLED"] = "false"

        from plm.search.service.app import app

        with TestClient(app) as client:
            retriever = app.state.retriever
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Kubernetes horizontal pod autoscaler scales pods",
                        "heading": "## HPA",
                        "keywords": ["kubernetes", "autoscaler"],
                        "entities": {},
                    }
                ],
            )
            yield client

    def test_query_explain_false_no_debug_info(self, client_with_data):
        """POST /query with explain=false returns results without debug_info."""
        response = client_with_data.post(
            "/query",
            json={"query": "kubernetes", "k": 1, "explain": False},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data.get("metadata") is None
        assert data.get("request_id") is None
        if data["results"]:
            assert data["results"][0].get("debug_info") is None

    def test_query_explain_true_has_debug_info(self, client_with_data):
        """POST /query with explain=true returns results with debug_info."""
        response = client_with_data.post(
            "/query",
            json={"query": "kubernetes", "k": 1, "explain": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "metadata" in data
        assert data["metadata"] is not None
        assert "request_id" in data
        assert data["request_id"] is not None

        if data["results"]:
            result = data["results"][0]
            assert "debug_info" in result
            assert result["debug_info"] is not None
            debug = result["debug_info"]
            assert "rrf_score" in debug
            assert "retrieval_stage" in debug

    def test_query_metadata_fields(self, client_with_data):
        """POST /query with explain=true returns metadata with required fields."""
        response = client_with_data.post(
            "/query",
            json={"query": "kubernetes", "k": 1, "explain": True},
        )
        assert response.status_code == 200
        data = response.json()
        metadata = data["metadata"]
        assert metadata["original_query"] == "kubernetes"
        assert "retrieval_mode" in metadata
        assert "rrf_k" in metadata
        assert "bm25_weight" in metadata
        assert "semantic_weight" in metadata
        assert "expanded_terms" in metadata


class TestRequestIDMiddleware:
    """Tests for X-Request-ID middleware."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        import os
        import tempfile

        os.environ["INDEX_PATH"] = tempfile.mkdtemp()
        os.environ["WATCH_DIR"] = ""
        os.environ["QUEUE_ENABLED"] = "false"

        from plm.search.service.app import app

        with TestClient(app) as client:
            yield client

    def test_request_id_generated_when_not_provided(self, client):
        """X-Request-ID is generated when not provided by client."""
        response = client.get("/health")
        assert response.status_code == 200
        assert "X-Request-ID" in response.headers
        request_id = response.headers["X-Request-ID"]
        uuid.UUID(request_id)

    def test_request_id_echoed_when_provided(self, client):
        """X-Request-ID is echoed back when provided by client."""
        custom_id = "test-request-123"
        response = client.get("/health", headers={"X-Request-ID": custom_id})
        assert response.status_code == 200
        assert response.headers["X-Request-ID"] == custom_id

    def test_request_id_in_query_response(self):
        """Request ID is included in QueryResponse when explain=True."""
        import os
        import tempfile

        os.environ["INDEX_PATH"] = tempfile.mkdtemp()
        os.environ["WATCH_DIR"] = ""
        os.environ["QUEUE_ENABLED"] = "false"

        from plm.search.service.app import app

        with TestClient(app) as client:
            retriever = app.state.retriever
            retriever.ingest_document(
                "doc1",
                "test.md",
                [
                    {
                        "content": "Test content",
                        "heading": "## Test",
                        "keywords": ["test"],
                        "entities": {},
                    }
                ],
            )

            custom_id = "correlation-test-456"
            response = client.post(
                "/query",
                json={"query": "test", "k": 1, "explain": True},
                headers={"X-Request-ID": custom_id},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["request_id"] == custom_id
