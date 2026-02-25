"""Data types for the retrieval benchmark framework."""

from dataclasses import dataclass, field
from typing import Optional


# ============================================================
# Input Data Types
# ============================================================

@dataclass
class Section:
    """A section within a document."""
    id: str
    document_id: str
    heading: str
    level: int
    content: str
    parent_id: Optional[str] = None


@dataclass
class Document:
    """A document with hierarchical sections."""
    id: str
    title: str
    source: str
    summary: str
    content: str  # Full markdown text
    sections: list[Section]
    
    def get_all_sections(self) -> list[Section]:
        """Get flat list of all sections."""
        return self.sections


@dataclass
class GroundTruth:
    """A ground truth query with expected results."""
    id: str
    question: str
    answer: str
    document_id: str
    section_ids: list[str]
    evidence: list[str]
    difficulty: str
    query_type: str


# ============================================================
# Index Data Types
# ============================================================

@dataclass
class Chunk:
    """A chunk of text for indexing."""
    id: str
    document_id: str
    section_id: Optional[str]
    content: str
    level: int = 0  # 0=chunk, 1=section, 2=doc summary
    metadata: dict = field(default_factory=dict)


# ============================================================
# Search Result Types
# ============================================================

@dataclass
class SearchHit:
    """A single search result."""
    chunk_id: str
    document_id: str
    section_id: Optional[str]
    content: str
    score: float
    level: int = 0


@dataclass
class SearchStats:
    """Statistics from a search operation."""
    duration_ms: float
    embed_calls: int
    llm_calls: int
    vectors_searched: int


@dataclass
class SearchResponse:
    """Complete response from a search operation."""
    hits: list[SearchHit]
    stats: SearchStats


# ============================================================
# Benchmark Result Types
# ============================================================

@dataclass
class IndexStats:
    """Statistics from indexing phase."""
    strategy: str
    backend: str
    embedding_model: str
    llm_model: Optional[str]
    duration_sec: float
    num_documents: int
    num_chunks: int
    num_vectors: int
    llm_calls: int
    embed_calls: int


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its metadata."""
    rank: int
    chunk_id: str
    document_id: str
    section_id: Optional[str]
    content: str
    score: float
    is_correct_doc: bool
    is_correct_section: bool


@dataclass
class ContentMetrics:
    """Content quality metrics comparing retrieved chunks to ground truth."""
    # Tier 1: Deterministic (no additional cost)
    evidence_recall: float       # % of evidence quotes found in retrieved chunks
    answer_token_overlap: float  # Jaccard(answer_keywords, chunk_keywords)
    rouge_l: float               # ROUGE-L F1 between answer and combined context
    
    # Tier 2: Semantic (embedding cost)
    max_evidence_similarity: Optional[float] = None   # Max cosine(evidence, chunk)
    answer_context_similarity: Optional[float] = None # Cosine(answer, combined_context)


@dataclass
class QueryResult:
    """Result of a single query execution."""
    query_id: str
    run_number: int
    strategy: str
    backend: str
    embedding_model: str
    llm_model: Optional[str]
    
    question: str
    expected_doc_id: str
    expected_section_ids: list[str]
    expected_answer: str
    expected_evidence: list[str]
    
    retrieved_chunks: list[RetrievedChunk]
    retrieved_doc_ids: list[str]
    retrieved_section_ids: list[str]
    top_k: int
    
    doc_found: bool
    doc_rank: int
    section_found: bool
    section_rank: int
    
    content_metrics: Optional[ContentMetrics] = None
    
    search_time_ms: float = 0.0
    embed_calls: int = 0
    llm_calls: int = 0


@dataclass
class StrategySummary:
    """Aggregated metrics for one strategy configuration."""
    strategy: str
    backend: str
    embedding_model: str
    llm_model: Optional[str]
    
    # Index metrics
    index_time_sec: float
    num_vectors: int
    
    # Accuracy at different k (keyed by k value)
    doc_recall_at_1: float
    doc_recall_at_3: float
    doc_recall_at_5: float
    doc_recall_at_10: float
    section_recall_at_1: float
    section_recall_at_3: float
    section_recall_at_5: float
    section_recall_at_10: float
    
    # MRR (Mean Reciprocal Rank)
    doc_mrr: float
    section_mrr: float
    
    # Performance
    avg_search_time_ms: float
    p50_search_time_ms: float
    p95_search_time_ms: float
    
    # Consistency (across multiple runs of same query)
    consistency_score: float
    
    # Cost
    total_llm_calls: int
    total_embed_calls: int
    queries_evaluated: int
    runs_per_query: int
