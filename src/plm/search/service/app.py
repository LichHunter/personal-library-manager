"""FastAPI application for search service with directory watcher.

Endpoints:
- POST /query - Search with query, k, use_rewrite parameters
- GET /health - Service health status
- GET /status - Detailed index statistics

Environment variables:
- INDEX_PATH: Path to index directory (default: /data/index)
- WATCH_DIR: Directory to watch for new documents (default: /data/watch)
- PROCESS_EXISTING: Process existing files on start (default: false)

Usage:
    uv run uvicorn plm.search.service.app:app --port 8000
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, cast

from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from starlette.middleware.base import BaseHTTPMiddleware

from plm.search.retriever import HybridRetriever
from plm.search.service.queue_consumer import SearchQueueConsumer
from plm.search.service.watcher import DirectoryWatcher

logger = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


class QueryRequest(BaseModel):
    """Request body for /query endpoint."""
    query: str = Field(..., description="Search query text")
    k: int = Field(default=5, ge=1, le=100, description="Number of results to return")
    use_rewrite: bool = Field(default=False, description="Whether to use query rewriting")
    use_rerank: bool = Field(default=False, description="Whether to use cross-encoder reranking")
    explain: bool = Field(default=False, description="Whether to return debug info for each result")


class DebugInfo(BaseModel):
    """Per-result debug information for explain mode."""
    bm25_score: float | None = Field(None, description="Raw BM25/SPLADE score")
    semantic_score: float | None = Field(None, description="Cosine similarity score")
    bm25_rank: int | None = Field(None, description="0-indexed position in sparse results")
    semantic_rank: int | None = Field(None, description="0-indexed position by semantic score")
    rrf_score: float = Field(..., description="Final RRF fusion score")
    rerank_score: float | None = Field(None, description="Cross-encoder score")
    retrieval_stage: str = Field(..., description="One of: 'rrf', 'rerank'")


class QueryMetadata(BaseModel):
    """Per-query metadata for explain mode."""
    original_query: str = Field(..., description="Raw query text")
    rewritten_query: str | None = Field(None, description="LLM-rewritten query")
    expanded_terms: list[str] = Field(default_factory=list, description="Terms added by QueryExpander")
    retrieval_mode: str = Field(..., description="'hybrid' or 'splade_only'")
    rrf_k: int = Field(..., description="RRF k parameter used")
    bm25_weight: float = Field(..., description="Weight applied to sparse scores")
    semantic_weight: float = Field(..., description="Weight applied to semantic scores")


class QueryResult(BaseModel):
    """Single search result."""
    chunk_id: str
    doc_id: str
    content: str
    enriched_content: str
    score: float
    heading: str | None
    start_char: int | None
    end_char: int | None
    debug_info: DebugInfo | None = Field(None, description="Debug info (only when explain=True)")


class QueryResponse(BaseModel):
    """Response body for /query endpoint."""
    query: str
    k: int
    use_rewrite: bool
    use_rerank: bool
    results: list[QueryResult]
    elapsed_ms: float
    metadata: QueryMetadata | None = Field(None, description="Query metadata (only when explain=True)")
    request_id: str | None = Field(None, description="Request correlation ID (only when explain=True)")


class HealthResponse(BaseModel):
    """Response body for /health endpoint."""
    status: str = Field(..., description="'ready' or 'starting'")
    chunk_count: int = Field(..., description="Number of indexed chunks")


class StatusResponse(BaseModel):
    """Response body for /status endpoint."""
    status: str
    chunk_count: int
    index_path: str
    watch_dir: str | None
    watcher_running: bool
    bm25_loaded: bool


# Get configuration from environment
INDEX_PATH = os.environ.get("INDEX_PATH", "/data/index")
WATCH_DIR = os.environ.get("WATCH_DIR", None)  # None means no watcher
PROCESS_EXISTING = os.environ.get("PROCESS_EXISTING", "false").lower() == "true"

# Queue configuration
QUEUE_ENABLED = os.environ.get("QUEUE_ENABLED", "false").lower() == "true"
QUEUE_URL = os.environ.get("QUEUE_URL", "redis://localhost:6379")
QUEUE_STREAM = os.environ.get("QUEUE_STREAM", "plm:extraction")

REQUEST_ID_HEADER = "X-Request-ID"


class RequestIDMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        return response


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan: startup and shutdown."""
    # Startup
    logger.info(f"[Service] Starting with INDEX_PATH={INDEX_PATH}, WATCH_DIR={WATCH_DIR}")

    # Ensure index directory exists
    db_path = Path(INDEX_PATH) / "index.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize retriever
    app.state.retriever = HybridRetriever(
        db_path=str(db_path),
        bm25_index_path=INDEX_PATH,
    )
    app.state.startup_time = time.time()

    # Initialize ingestion: queue consumer OR directory watcher (mutually exclusive)
    app.state.watcher = None
    app.state.queue_consumer = None
    
    if QUEUE_ENABLED:
        # Queue mode: use Redis Streams consumer
        import threading
        app.state.queue_consumer = SearchQueueConsumer(
            redis_url=QUEUE_URL,
            retriever=app.state.retriever,
            stream=QUEUE_STREAM,
        )
        # Run consumer in background thread
        consumer_thread = threading.Thread(
            target=app.state.queue_consumer.run,
            daemon=True,
            name="queue-consumer",
        )
        consumer_thread.start()
        app.state.consumer_thread = consumer_thread
        logger.info(f"[Service] Queue consumer started on {QUEUE_STREAM}")
    elif WATCH_DIR:
        # Directory watcher mode (legacy/standalone)
        watch_path = Path(WATCH_DIR)
        watch_path.mkdir(parents=True, exist_ok=True)

        app.state.watcher = DirectoryWatcher(
            watch_dir=watch_path,
            retriever=app.state.retriever,
            debounce_seconds=1.0,
            process_existing=PROCESS_EXISTING,
        )
        app.state.watcher.start()
        logger.info(f"[Service] Directory watcher started on {WATCH_DIR}")
    else:
        logger.info("[Service] No WATCH_DIR or QUEUE_ENABLED configured, ingestion disabled")

    logger.info("[Service] Startup complete")
    yield

    # Shutdown
    logger.info("[Service] Shutting down...")
    if app.state.watcher:
        app.state.watcher.stop()
    if app.state.queue_consumer:
        app.state.queue_consumer.stop()
        # Wait for consumer thread to finish
        if hasattr(app.state, 'consumer_thread'):
            app.state.consumer_thread.join(timeout=5.0)
    logger.info("[Service] Shutdown complete")


app = FastAPI(
    title="PLM Search Service",
    description="Search service with directory watcher for auto-ingestion",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, http_request: Request) -> QueryResponse:
    """Execute a search query.

    Args:
        request: Query parameters including query text, k, and use_rewrite flag.

    Returns:
        QueryResponse with search results and timing info.
    """
    start_time = time.time()

    retriever: HybridRetriever = app.state.retriever

    try:
        result = retriever.retrieve(
            query=request.query,
            k=request.k,
            use_rewrite=request.use_rewrite,
            use_rerank=request.use_rerank,
            explain=request.explain,
        )
    except Exception as e:
        logger.error(f"[Service] Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    elapsed_ms = (time.time() - start_time) * 1000

    if request.explain:
        results, explain_data = result
        query_results = [
            QueryResult(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                content=r["content"],
                enriched_content=r.get("enriched_content", ""),
                score=r["score"],
                heading=r.get("heading"),
                start_char=r.get("start_char"),
                end_char=r.get("end_char"),
                debug_info=DebugInfo(
                    bm25_score=explain_data["debug_info"][r["chunk_id"]].get("bm25_score"),
                    semantic_score=explain_data["debug_info"][r["chunk_id"]].get("semantic_score"),
                    bm25_rank=explain_data["debug_info"][r["chunk_id"]].get("bm25_rank"),
                    semantic_rank=explain_data["debug_info"][r["chunk_id"]].get("semantic_rank"),
                    rrf_score=explain_data["debug_info"][r["chunk_id"]]["rrf_score"],
                    rerank_score=explain_data["debug_info"][r["chunk_id"]].get("rerank_score"),
                    retrieval_stage=explain_data["debug_info"][r["chunk_id"]]["retrieval_stage"],
                ),
            )
            for r in results
        ]
        metadata = QueryMetadata(
            original_query=explain_data["metadata"]["original_query"],
            rewritten_query=explain_data["metadata"].get("rewritten_query"),
            expanded_terms=explain_data["metadata"].get("expanded_terms", []),
            retrieval_mode=explain_data["metadata"]["retrieval_mode"],
            rrf_k=explain_data["metadata"]["rrf_k"],
            bm25_weight=explain_data["metadata"]["bm25_weight"],
            semantic_weight=explain_data["metadata"]["semantic_weight"],
        )
        return QueryResponse(
            query=request.query,
            k=request.k,
            use_rewrite=request.use_rewrite,
            use_rerank=request.use_rerank,
            results=query_results,
            elapsed_ms=elapsed_ms,
            metadata=metadata,
            request_id=http_request.state.request_id,
        )
    else:
        results_list: list[dict] = result  # type: ignore[assignment]
        return QueryResponse(
            query=request.query,
            k=request.k,
            use_rewrite=request.use_rewrite,
            use_rerank=request.use_rerank,
            results=[
                QueryResult(
                    chunk_id=r["chunk_id"],
                    doc_id=r["doc_id"],
                    content=r["content"],
                    enriched_content=r.get("enriched_content", ""),
                    score=r["score"],
                    heading=r.get("heading"),
                    start_char=r.get("start_char"),
                    end_char=r.get("end_char"),
                )
                for r in results_list
            ],
            elapsed_ms=elapsed_ms,
        )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Get service health status.

    Returns 'ready' if the index is loaded and has chunks,
    'starting' if still initializing or empty.
    """
    retriever: HybridRetriever = app.state.retriever
    chunk_count = retriever.storage.get_chunk_count()
    is_ready = chunk_count > 0 and retriever.bm25_index is not None

    return HealthResponse(
        status="ready" if is_ready else "starting",
        chunk_count=chunk_count,
    )


@app.get("/status", response_model=StatusResponse)
async def status() -> StatusResponse:
    """Get detailed index and service status."""
    retriever: HybridRetriever = app.state.retriever
    watcher: DirectoryWatcher | None = app.state.watcher

    chunk_count = retriever.storage.get_chunk_count()
    is_ready = chunk_count > 0 and retriever.bm25_index is not None

    return StatusResponse(
        status="ready" if is_ready else "starting",
        chunk_count=chunk_count,
        index_path=INDEX_PATH,
        watch_dir=WATCH_DIR,
        watcher_running=watcher is not None and watcher._running,
        bm25_loaded=retriever.bm25_index is not None,
    )
