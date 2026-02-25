"""Token-level and document-level evaluation metrics.

Based on Chroma Research (July 2024):
https://research.trychroma.com/evaluating-chunking
https://github.com/brandonstarxel/chunking_evaluation

FIXED in V2.1:
- Token metrics now calculated per-document then aggregated
- Added Key Facts Coverage metric
- Added Efficiency metric
- Calculate against ALL expected docs (not just retrieved)
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TokenMetrics:
    """Token-level evaluation metrics for a single query."""
    
    # Core metrics
    recall: float  # |retrieved ∩ relevant| / |relevant|
    precision: float  # |retrieved ∩ relevant| / |retrieved|
    iou: float  # |retrieved ∩ relevant| / |retrieved ∪ relevant|
    
    # Efficiency: what % of retrieved content is relevant
    efficiency: float  # Same as precision, but named for clarity
    
    # Raw counts for debugging
    relevant_chars: int
    retrieved_chars: int
    overlap_chars: int
    
    # Per-document breakdown
    per_doc_recall: dict[str, float] = field(default_factory=dict)


@dataclass 
class DocumentMetrics:
    """Document-level evaluation metrics for a single query."""
    
    recall_at_k: float  # Did we retrieve docs containing the answer?
    mrr: float  # Reciprocal rank of first relevant doc
    
    # Raw data
    expected_docs: list[str]
    retrieved_docs: list[str]
    k: int


@dataclass
class KeyFactsMetrics:
    """Key facts coverage metrics for a single query."""
    
    coverage: float  # % of key facts found in retrieved chunks
    found_facts: list[str]
    missing_facts: list[str]
    total_facts: int


def spans_to_char_set(spans: list[tuple[int, int]]) -> set[int]:
    """Convert list of (start, end) spans to set of character positions."""
    chars = set()
    for start, end in spans:
        chars.update(range(start, end))
    return chars


def calculate_token_metrics_per_doc(
    retrieved_chunks: list[dict],  # [{"doc_id": str, "start": int, "end": int}, ...]
    relevant_spans: list[dict],    # [{"doc_id": str, "start": int, "end": int}, ...]
    expected_docs: list[str],      # ALL expected docs (not just retrieved)
) -> TokenMetrics:
    """Calculate token-level metrics correctly - per document then aggregate.
    
    Args:
        retrieved_chunks: List of dicts with doc_id, start, end
        relevant_spans: Ground truth spans with doc_id, start, end
        expected_docs: All expected document IDs (to penalize missing docs)
        
    Returns:
        TokenMetrics with recall, precision, IoU calculated per-doc then aggregated
    """
    # Group by doc_id
    retrieved_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for chunk in retrieved_chunks:
        retrieved_by_doc[chunk["doc_id"]].append((chunk["start"], chunk["end"]))
    
    relevant_by_doc: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in relevant_spans:
        relevant_by_doc[span["doc_id"]].append((span["start"], span["end"]))
    
    # Calculate per-doc, then aggregate
    total_overlap = 0
    total_relevant = 0
    total_retrieved = 0
    per_doc_recall = {}
    
    # Include ALL expected docs in calculation (penalizes missing docs)
    all_docs = set(expected_docs) | set(retrieved_by_doc.keys()) | set(relevant_by_doc.keys())
    
    for doc_id in all_docs:
        ret_chars = spans_to_char_set(retrieved_by_doc.get(doc_id, []))
        rel_chars = spans_to_char_set(relevant_by_doc.get(doc_id, []))
        
        overlap = len(ret_chars & rel_chars)
        total_overlap += overlap
        total_relevant += len(rel_chars)
        total_retrieved += len(ret_chars)
        
        # Per-doc recall (useful for debugging)
        if rel_chars:
            per_doc_recall[doc_id] = overlap / len(rel_chars)
        elif doc_id in expected_docs:
            # Expected doc but no relevant spans found - mark as 0 if not retrieved
            per_doc_recall[doc_id] = 1.0 if ret_chars else 0.0
    
    # Calculate aggregate metrics
    if total_relevant == 0:
        recall = 1.0 if total_retrieved == 0 else 0.0
    else:
        recall = total_overlap / total_relevant
    
    if total_retrieved == 0:
        precision = 1.0 if total_relevant == 0 else 0.0
    else:
        precision = total_overlap / total_retrieved
    
    union = total_relevant + total_retrieved - total_overlap
    iou = total_overlap / union if union > 0 else 0.0
    
    return TokenMetrics(
        recall=recall,
        precision=precision,
        iou=iou,
        efficiency=precision,  # Same as precision
        relevant_chars=total_relevant,
        retrieved_chars=total_retrieved,
        overlap_chars=total_overlap,
        per_doc_recall=per_doc_recall,
    )


def calculate_key_facts_coverage(
    retrieved_text: str,
    key_facts: list[str],
) -> KeyFactsMetrics:
    """Calculate what % of key facts appear in retrieved chunks.
    
    This is the most direct measure of "can the LLM answer the question?"
    
    Args:
        retrieved_text: Concatenated text of all retrieved chunks
        key_facts: List of key facts that should be present to answer
        
    Returns:
        KeyFactsMetrics with coverage percentage and details
    """
    if not key_facts:
        return KeyFactsMetrics(
            coverage=1.0,
            found_facts=[],
            missing_facts=[],
            total_facts=0,
        )
    
    retrieved_lower = retrieved_text.lower()
    found = []
    missing = []
    
    for fact in key_facts:
        if fact.lower() in retrieved_lower:
            found.append(fact)
        else:
            missing.append(fact)
    
    return KeyFactsMetrics(
        coverage=len(found) / len(key_facts),
        found_facts=found,
        missing_facts=missing,
        total_facts=len(key_facts),
    )


def calculate_document_metrics(
    retrieved_docs: list[str],
    expected_docs: list[str],
    k: int = 5,
) -> DocumentMetrics:
    """Calculate document-level metrics for a query.
    
    Args:
        retrieved_docs: List of doc IDs in retrieval order (most similar first)
        expected_docs: Ground truth doc IDs that contain the answer
        k: Number of top results to consider for Recall@K
        
    Returns:
        DocumentMetrics with recall@k and MRR
    """
    expected_set = set(expected_docs)
    
    # Recall@K: What fraction of expected docs are in top K?
    top_k = set(retrieved_docs[:k])
    hits = len(expected_set & top_k)
    recall_at_k = hits / len(expected_set) if expected_set else 0.0
    
    # MRR: Reciprocal rank of first relevant doc
    mrr = 0.0
    for rank, doc_id in enumerate(retrieved_docs):
        if doc_id in expected_set:
            mrr = 1.0 / (rank + 1)
            break
    
    return DocumentMetrics(
        recall_at_k=recall_at_k,
        mrr=mrr,
        expected_docs=expected_docs,
        retrieved_docs=retrieved_docs[:k],
        k=k,
    )


def aggregate_metrics(
    token_metrics: list[TokenMetrics],
    doc_metrics: list[DocumentMetrics],
    key_facts_metrics: list[KeyFactsMetrics] | None = None,
) -> dict:
    """Aggregate metrics across all queries.
    
    Returns:
        Dictionary with averaged metrics and distributions
    """
    n_token = len(token_metrics)
    n_doc = len(doc_metrics)
    
    result = {}
    
    # Token-level averages
    if n_token > 0:
        result.update({
            "token_recall": sum(m.recall for m in token_metrics) / n_token,
            "token_precision": sum(m.precision for m in token_metrics) / n_token,
            "token_iou": sum(m.iou for m in token_metrics) / n_token,
            "efficiency": sum(m.efficiency for m in token_metrics) / n_token,
            "total_relevant_chars": sum(m.relevant_chars for m in token_metrics),
            "total_retrieved_chars": sum(m.retrieved_chars for m in token_metrics),
            "total_overlap_chars": sum(m.overlap_chars for m in token_metrics),
            "num_token_queries": n_token,
        })
    
    # Document-level averages
    if n_doc > 0:
        result.update({
            "recall_at_k": sum(m.recall_at_k for m in doc_metrics) / n_doc,
            "mrr": sum(m.mrr for m in doc_metrics) / n_doc,
            "num_doc_queries": n_doc,
        })
    
    # Key facts coverage
    if key_facts_metrics:
        n_kf = len(key_facts_metrics)
        result.update({
            "key_facts_coverage": sum(m.coverage for m in key_facts_metrics) / n_kf,
            "total_facts": sum(m.total_facts for m in key_facts_metrics),
            "total_found_facts": sum(len(m.found_facts) for m in key_facts_metrics),
            "num_kf_queries": n_kf,
        })
    
    result["num_queries"] = max(n_token, n_doc, len(key_facts_metrics) if key_facts_metrics else 0)
    
    return result


# Keep old function for backwards compatibility but mark deprecated
def calculate_token_metrics(
    retrieved_chunks: list[tuple[int, int]],
    relevant_spans: list[tuple[int, int]],
) -> TokenMetrics:
    """DEPRECATED: Use calculate_token_metrics_per_doc instead.
    
    This function doesn't handle multi-document correctly.
    """
    import warnings
    warnings.warn(
        "calculate_token_metrics is deprecated, use calculate_token_metrics_per_doc",
        DeprecationWarning
    )
    
    retrieved_chars = spans_to_char_set(retrieved_chunks)
    relevant_chars = spans_to_char_set(relevant_spans)
    
    if not relevant_chars:
        return TokenMetrics(
            recall=1.0 if not retrieved_chars else 0.0,
            precision=1.0 if not retrieved_chars else 0.0,
            iou=1.0 if not retrieved_chars else 0.0,
            efficiency=1.0 if not retrieved_chars else 0.0,
            relevant_chars=0,
            retrieved_chars=len(retrieved_chars),
            overlap_chars=0,
        )
    
    if not retrieved_chars:
        return TokenMetrics(
            recall=0.0,
            precision=0.0,
            iou=0.0,
            efficiency=0.0,
            relevant_chars=len(relevant_chars),
            retrieved_chars=0,
            overlap_chars=0,
        )
    
    overlap = retrieved_chars & relevant_chars
    union = retrieved_chars | relevant_chars
    
    recall = len(overlap) / len(relevant_chars)
    precision = len(overlap) / len(retrieved_chars)
    iou = len(overlap) / len(union) if union else 0.0
    
    return TokenMetrics(
        recall=recall,
        precision=precision,
        iou=iou,
        efficiency=precision,
        relevant_chars=len(relevant_chars),
        retrieved_chars=len(retrieved_chars),
        overlap_chars=len(overlap),
    )
