"""Immutable type system for modular retrieval pipeline.

This module defines frozen dataclasses that form the transformation chain
for the retrieval pipeline. Each type preserves provenance (where data came from)
and transformation history (what was done to it), enabling debugging and
understanding the full data flow.

Transformation chain:
    Query → RewrittenQuery → ExpandedQuery → EmbeddedQuery → ScoredChunk → PipelineResult
"""

from dataclasses import dataclass, field
from typing import Literal, Optional


@dataclass(frozen=True)
class Query:
    """Original user query.

    Immutable representation of the initial search query.
    """

    text: str
    metadata: tuple = field(default_factory=tuple)  # Immutable tuple instead of dict


@dataclass(frozen=True)
class RewrittenQuery:
    """Query after rewriting/reformulation.

    Preserves the original query and adds the rewritten version.
    Tracks which model performed the rewriting.
    """

    original: Query
    rewritten: str
    model: str  # e.g., "claude-3-haiku", "gpt-4"
    metadata: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class ExpandedQuery:
    """Query after expansion with synonyms/related terms.

    Preserves the rewritten query and adds expanded terms.
    Tracks which expansion method was used.
    """

    query: RewrittenQuery
    expanded: str  # Full expanded query text
    expansions: tuple[str, ...] = field(
        default_factory=tuple
    )  # Individual expansion terms
    method: str = "default"  # e.g., "synonym", "wordnet", "domain_specific"
    metadata: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class EmbeddedQuery:
    """Query after embedding into vector space.

    Preserves the expanded query and adds the embedding vector.
    Tracks which embedding model was used.
    """

    query: ExpandedQuery
    embedding: tuple[float, ...] = field(
        default_factory=tuple
    )  # Immutable tuple of floats
    model: str = "bge-base-en-v1.5"  # Embedding model name
    dimension: int = 768  # Vector dimension
    metadata: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class ScoredChunk:
    """A chunk with relevance score and provenance tracking.

    Represents a retrieved chunk with its relevance score and information
    about which retrieval signal produced the score.
    """

    chunk_id: str
    content: str
    score: float  # Relevance score (0.0 to 1.0)
    source: Literal["bm25", "semantic", "rrf"]  # Which signal produced this score
    rank: int  # Rank in the result set
    metadata: tuple = field(default_factory=tuple)  # Immutable chunk metadata


@dataclass(frozen=True)
class FusionConfig:
    """Configuration for Reciprocal Rank Fusion (RRF).

    Immutable configuration for combining multiple retrieval signals.
    """

    k: int = 60  # RRF parameter k
    bm25_weight: float = 0.5  # Weight for BM25 signal
    semantic_weight: float = 0.5  # Weight for semantic signal
    metadata: tuple = field(default_factory=tuple)


@dataclass(frozen=True)
class PipelineResult:
    """Complete result of the retrieval pipeline with full transformation history.

    Captures the entire transformation chain from original query to final results,
    enabling debugging and understanding of the retrieval process.
    """

    # Original query
    original_query: Query

    # Transformation chain
    rewritten_query: Optional[RewrittenQuery] = None
    expanded_query: Optional[ExpandedQuery] = None
    embedded_query: Optional[EmbeddedQuery] = None

    # Results
    scored_chunks: tuple[ScoredChunk, ...] = field(default_factory=tuple)

    # Configuration used
    fusion_config: Optional[FusionConfig] = None

    # Metadata about the pipeline execution
    total_chunks_searched: int = 0
    execution_time_ms: float = 0.0
    pipeline_stages: tuple[str, ...] = field(
        default_factory=tuple
    )  # e.g., ("rewrite", "expand", "embed", "retrieve", "fuse")
    metadata: tuple = field(default_factory=tuple)
