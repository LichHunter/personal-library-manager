"""MCP server exposing PLM search functionality.

This server wraps the HybridRetriever to expose search capabilities
via the Model Context Protocol (MCP) for use with AI assistants.

Environment variables:
    INDEX_PATH: Path to index directory (default: /data/index)

Usage:
    # Run with stdio transport (for OpenCode/Claude Desktop)
    plm-mcp

    # Or directly
    python -m plm.search.mcp.server
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Annotated

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from plm.search.retriever import HybridRetriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

INDEX_PATH = os.environ.get("INDEX_PATH", "/data/index")

mcp = FastMCP(
    name="PLM Search",
    instructions="""Hybrid search over indexed documents (BM25 + semantic + RRF fusion).

## Query Formulation

**Keyword queries work best.** The system combines exact term matching (BM25) with 
semantic similarity, so technical terms and specific phrases are most effective.

Good: "authentication middleware error handling"
Bad: "How do I handle errors when authentication fails in my middleware?"

**Break complex questions into focused queries.** Run 2-3 targeted searches rather 
than one broad query.

**use_rewrite=True**: Enables LLM query rewriting. Use when:
- You must search with natural language questions
- Initial keyword search returned poor results
- Query contains ambiguous terms needing interpretation
Costs ~500ms latency. Default is off.

## Interpreting Results

- **score**: RRF fusion score. Higher = more relevant. Not normalized.
- **heading**: Section context. Use for understanding where chunk appears.
- **doc_id**: Group results by this to see related chunks from same document.
- **start_char/end_char**: Character offsets for precise citation.

## Workflow

1. `get_status()` - Verify index is ready
2. `search(query, k=10)` - Start with k=10, adjust based on result quality
3. `get_chunk(chunk_id)` - Get full enriched content when needed

Multiple searches with different phrasings often outperform a single search.
""",
)

_retriever: HybridRetriever | None = None


def _get_retriever() -> HybridRetriever:
    """Get or initialize the HybridRetriever singleton."""
    global _retriever
    if _retriever is None:
        db_path = Path(INDEX_PATH) / "index.db"
        db_path.parent.mkdir(parents=True, exist_ok=True)
        _retriever = HybridRetriever(
            db_path=str(db_path),
            bm25_index_path=INDEX_PATH,
        )
        logger.info(f"Initialized HybridRetriever with INDEX_PATH={INDEX_PATH}")
    return _retriever


class SearchResult(BaseModel):
    """A single search result with citation information."""

    chunk_id: str = Field(description="Unique identifier for the chunk")
    doc_id: str = Field(description="Document identifier (source file hash)")
    content: str = Field(description="Original chunk text content")
    score: float = Field(description="Relevance score (RRF fusion)")
    heading: str | None = Field(description="Section heading if available")
    start_char: int | None = Field(description="Start character offset in source")
    end_char: int | None = Field(description="End character offset in source")


class SearchResponse(BaseModel):
    """Response from a search query."""

    query: str = Field(description="The original search query")
    k: int = Field(description="Number of results requested")
    result_count: int = Field(description="Number of results returned")
    results: list[SearchResult] = Field(description="List of search results")


class StatusResponse(BaseModel):
    """Index status information."""

    status: str = Field(description="'ready' or 'not_indexed'")
    chunk_count: int = Field(description="Number of indexed chunks")
    document_count: int = Field(description="Number of indexed documents")
    index_path: str = Field(description="Path to index directory")
    bm25_loaded: bool = Field(description="Whether BM25 index is loaded")


class ChunkResponse(BaseModel):
    """Detailed chunk information."""

    chunk_id: str
    doc_id: str
    content: str
    enriched_content: str = Field(description="Keyword-enriched content used for retrieval")
    heading: str | None
    start_char: int | None
    end_char: int | None


class HeadingSearchResult(BaseModel):
    """A heading-level search result."""

    heading_id: str = Field(description="Unique identifier for the heading")
    doc_id: str = Field(description="Document identifier")
    heading_text: str = Field(description="The heading text")
    heading_level: int | None = Field(description="Heading level (0=root, 2=##, etc.)")
    score: float = Field(description="Semantic similarity score")
    keywords_json: str | None = Field(description="Aggregated keywords as JSON")
    entities_json: str | None = Field(description="Aggregated entities as JSON")


class HeadingSearchResponse(BaseModel):
    """Response from a heading-level search."""

    query: str
    k: int
    result_count: int
    results: list[HeadingSearchResult]


class DocumentSearchResult(BaseModel):
    """A document-level search result."""

    doc_id: str = Field(description="Document identifier")
    source_file: str = Field(description="Source file path")
    score: float = Field(description="Semantic similarity score")
    keywords_json: str | None = Field(description="Aggregated keywords as JSON")
    entities_json: str | None = Field(description="Aggregated entities as JSON")


class DocumentSearchResponse(BaseModel):
    """Response from a document-level search."""

    query: str
    k: int
    result_count: int
    results: list[DocumentSearchResult]


@mcp.tool()
def search(
    query: Annotated[str, Field(description="Search query text")],
    k: Annotated[int, Field(description="Number of results to return", ge=1, le=100)] = 10,
    use_rewrite: Annotated[bool, Field(description="Use Claude Haiku to rewrite query for better retrieval")] = False,
) -> SearchResponse:
    """Search the indexed document corpus.

    Performs hybrid search combining BM25 (lexical) and semantic (embedding)
    retrieval with RRF fusion. Returns the most relevant chunks with
    citation information.

    Each result includes:
    - chunk_id: Use with get_chunk() for full details
    - doc_id: Document identifier for grouping
    - content: The actual text content
    - score: Relevance score (higher is better)
    - heading: Section heading for context
    - start_char/end_char: Character offsets for precise citation
    """
    retriever = _get_retriever()

    if not retriever.is_indexed():
        return SearchResponse(
            query=query,
            k=k,
            result_count=0,
            results=[],
        )

    results = retriever.retrieve(query=query, k=k, use_rewrite=use_rewrite)

    return SearchResponse(
        query=query,
        k=k,
        result_count=len(results),
        results=[
            SearchResult(
                chunk_id=r["chunk_id"],
                doc_id=r["doc_id"],
                content=r["content"],
                score=r["score"],
                heading=r.get("heading"),
                start_char=r.get("start_char"),
                end_char=r.get("end_char"),
            )
            for r in results
        ],
    )


@mcp.tool()
def get_status() -> StatusResponse:
    """Get index status and statistics.

    Returns information about the search index including:
    - Whether the index is ready for queries
    - Number of indexed chunks and documents
    - Whether the BM25 index is loaded

    Use this to verify the search service is operational before querying.
    """
    retriever = _get_retriever()

    chunk_count = retriever.storage.get_chunk_count()
    doc_count = retriever.storage.get_document_count()
    is_ready = chunk_count > 0 and retriever.bm25_index is not None

    return StatusResponse(
        status="ready" if is_ready else "not_indexed",
        chunk_count=chunk_count,
        document_count=doc_count,
        index_path=INDEX_PATH,
        bm25_loaded=retriever.bm25_index is not None,
    )


@mcp.tool()
def get_chunk(
    chunk_id: Annotated[str, Field(description="Chunk ID from search results")],
) -> ChunkResponse | None:
    """Get detailed information about a specific chunk.

    Use this to retrieve the full content and metadata for a chunk
    returned from a search query. Includes the enriched content that
    was used for retrieval.

    Returns None if the chunk is not found.
    """
    retriever = _get_retriever()

    chunk = retriever.storage.get_chunk_by_id(chunk_id)
    if chunk is None:
        return None

    return ChunkResponse(
        chunk_id=chunk["id"],
        doc_id=chunk["doc_id"],
        content=chunk["content"],
        enriched_content=chunk.get("enriched_content", ""),
        heading=chunk.get("heading"),
        start_char=chunk.get("start_char"),
        end_char=chunk.get("end_char"),
    )


@mcp.tool()
def search_headings(
    query: Annotated[str, Field(description="Search query text")],
    k: Annotated[int, Field(description="Number of results to return", ge=1, le=100)] = 10,
) -> HeadingSearchResponse:
    """Search at the heading/section level using semantic similarity.

    Returns heading-level results with aggregated metadata from all chunks
    within each heading. Use this for broader context than chunk search.

    Requires documents to be ingested with hierarchical structure.
    """
    retriever = _get_retriever()
    results = retriever.retrieve_headings(query=query, k=k)

    return HeadingSearchResponse(
        query=query,
        k=k,
        result_count=len(results),
        results=[
            HeadingSearchResult(
                heading_id=r["heading_id"],
                doc_id=r["doc_id"],
                heading_text=r["heading_text"],
                heading_level=r.get("heading_level"),
                score=r["score"],
                keywords_json=r.get("keywords_json"),
                entities_json=r.get("entities_json"),
            )
            for r in results
        ],
    )


@mcp.tool()
def search_documents(
    query: Annotated[str, Field(description="Search query text")],
    k: Annotated[int, Field(description="Number of results to return", ge=1, le=100)] = 10,
) -> DocumentSearchResponse:
    """Search at the document level using semantic similarity.

    Returns document-level results with aggregated metadata. Use this
    to find which documents are most relevant before drilling down.

    Requires documents to be ingested with hierarchical structure.
    """
    retriever = _get_retriever()
    results = retriever.retrieve_documents(query=query, k=k)

    return DocumentSearchResponse(
        query=query,
        k=k,
        result_count=len(results),
        results=[
            DocumentSearchResult(
                doc_id=r["doc_id"],
                source_file=r["source_file"],
                score=r["score"],
                keywords_json=r.get("keywords_json"),
                entities_json=r.get("entities_json"),
            )
            for r in results
        ],
    )


def main() -> None:
    """Run the MCP server with stdio transport."""
    logger.info(f"Starting PLM Search MCP server (INDEX_PATH={INDEX_PATH})")
    mcp.run()


if __name__ == "__main__":
    main()
