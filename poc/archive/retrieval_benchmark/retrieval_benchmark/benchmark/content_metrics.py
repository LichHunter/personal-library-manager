"""Content quality metrics for evaluating retrieved chunks against ground truth."""

import re
from typing import Optional

from ..core.types import ContentMetrics


def _normalize_text(text: str) -> str:
    """Lowercase and normalize whitespace."""
    return " ".join(text.lower().split())


def _tokenize(text: str) -> set[str]:
    """Simple word tokenization."""
    return set(re.findall(r'\b\w+\b', text.lower()))


def compute_evidence_recall(
    retrieved_chunks: list[str],
    evidence_quotes: list[str],
) -> float:
    """
    Compute fraction of evidence quotes found in retrieved chunks.
    
    Evidence is considered "found" if it appears as a substring in any chunk
    after normalization.
    """
    if not evidence_quotes:
        return 1.0
    
    combined_context = _normalize_text(" ".join(retrieved_chunks))
    
    found = 0
    for quote in evidence_quotes:
        normalized_quote = _normalize_text(quote)
        if normalized_quote in combined_context:
            found += 1
    
    return found / len(evidence_quotes)


def compute_answer_token_overlap(
    retrieved_chunks: list[str],
    expected_answer: str,
) -> float:
    """
    Compute Jaccard similarity between answer tokens and retrieved content tokens.
    
    Jaccard = |intersection| / |union|
    """
    if not expected_answer.strip():
        return 0.0
    
    answer_tokens = _tokenize(expected_answer)
    context_tokens = _tokenize(" ".join(retrieved_chunks))
    
    if not answer_tokens or not context_tokens:
        return 0.0
    
    intersection = answer_tokens & context_tokens
    union = answer_tokens | context_tokens
    
    return len(intersection) / len(union)


def compute_rouge_l(
    retrieved_chunks: list[str],
    expected_answer: str,
) -> float:
    """
    Compute ROUGE-L F1 score (longest common subsequence).
    
    This is a simplified implementation without external dependencies.
    For production use, consider rouge-score package.
    """
    if not expected_answer.strip():
        return 0.0
    
    reference_tokens = expected_answer.lower().split()
    hypothesis_tokens = " ".join(retrieved_chunks).lower().split()
    
    if not reference_tokens or not hypothesis_tokens:
        return 0.0
    
    lcs_length = _lcs_length(reference_tokens, hypothesis_tokens)
    
    precision = lcs_length / len(hypothesis_tokens) if hypothesis_tokens else 0.0
    recall = lcs_length / len(reference_tokens) if reference_tokens else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    f1 = 2 * precision * recall / (precision + recall)
    return f1


def _lcs_length(seq1: list[str], seq2: list[str]) -> int:
    """Compute length of longest common subsequence."""
    m, n = len(seq1), len(seq2)
    
    if m > 1000 or n > 1000:
        return _lcs_length_approx(seq1, seq2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]


def _lcs_length_approx(seq1: list[str], seq2: list[str]) -> int:
    """Approximate LCS for long sequences using set intersection."""
    set1 = set(seq1)
    set2 = set(seq2)
    return len(set1 & set2)


def compute_content_metrics(
    retrieved_chunks: list[str],
    expected_answer: str,
    evidence_quotes: list[str],
    embedder=None,
) -> ContentMetrics:
    """
    Compute all content quality metrics.
    
    Args:
        retrieved_chunks: List of retrieved chunk contents
        expected_answer: Ground truth answer
        evidence_quotes: Ground truth evidence quotes
        embedder: Optional embedder for Tier 2 semantic metrics
    
    Returns:
        ContentMetrics with all computed scores
    """
    evidence_recall = compute_evidence_recall(retrieved_chunks, evidence_quotes)
    answer_token_overlap = compute_answer_token_overlap(retrieved_chunks, expected_answer)
    rouge_l = compute_rouge_l(retrieved_chunks, expected_answer)
    
    max_evidence_similarity: Optional[float] = None
    answer_context_similarity: Optional[float] = None
    
    if embedder is not None:
        max_evidence_similarity = compute_max_evidence_similarity(
            retrieved_chunks, evidence_quotes, embedder
        )
        answer_context_similarity = compute_answer_context_similarity(
            retrieved_chunks, expected_answer, embedder
        )
    
    return ContentMetrics(
        evidence_recall=evidence_recall,
        answer_token_overlap=answer_token_overlap,
        rouge_l=rouge_l,
        max_evidence_similarity=max_evidence_similarity,
        answer_context_similarity=answer_context_similarity,
    )


def compute_max_evidence_similarity(
    retrieved_chunks: list[str],
    evidence_quotes: list[str],
    embedder,
) -> float:
    """
    Compute max cosine similarity between any evidence quote and any chunk.
    
    Returns highest similarity score found.
    """
    if not evidence_quotes or not retrieved_chunks:
        return 0.0
    
    import numpy as np
    
    evidence_embeddings = embedder.embed_batch(evidence_quotes)
    chunk_embeddings = embedder.embed_batch(retrieved_chunks)
    
    evidence_arr = np.array(evidence_embeddings)
    chunk_arr = np.array(chunk_embeddings)
    
    evidence_norm = evidence_arr / np.linalg.norm(evidence_arr, axis=1, keepdims=True)
    chunk_norm = chunk_arr / np.linalg.norm(chunk_arr, axis=1, keepdims=True)
    
    similarities = np.dot(evidence_norm, chunk_norm.T)
    
    return float(np.max(similarities))


def compute_answer_context_similarity(
    retrieved_chunks: list[str],
    expected_answer: str,
    embedder,
) -> float:
    """
    Compute cosine similarity between expected answer and combined context.
    """
    if not expected_answer.strip() or not retrieved_chunks:
        return 0.0
    
    import numpy as np
    
    combined_context = " ".join(retrieved_chunks)
    
    answer_emb = np.array(embedder.embed(expected_answer))
    context_emb = np.array(embedder.embed(combined_context))
    
    answer_norm = answer_emb / np.linalg.norm(answer_emb)
    context_norm = context_emb / np.linalg.norm(context_emb)
    
    return float(np.dot(answer_norm, context_norm))
