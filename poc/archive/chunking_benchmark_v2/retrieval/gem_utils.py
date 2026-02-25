"""Shared utilities for gem retrieval strategies.

All strategies should import from this module:
    from .gem_utils import (
        detect_technical_score,
        detect_negation,
        reciprocal_rank_fusion,
        extract_chunk_fields,
        measure_latency,
    )
"""

import re
import time
from contextlib import contextmanager
from typing import Optional, Any

from strategies import Chunk

# ============================================================================
# LATENCY MEASUREMENT
# ============================================================================


@contextmanager
def measure_latency():
    """Context manager to measure execution time.

    Usage:
        with measure_latency() as get_latency:
            # ... do work ...
        latency_ms = get_latency() * 1000
    """
    start = time.perf_counter()
    yield lambda: time.perf_counter() - start


# ============================================================================
# TECHNICAL QUERY DETECTION
# ============================================================================

TECHNICAL_PATTERNS = [
    r"[a-z]+[A-Z][a-z]+",  # camelCase: PgBouncer, minReplicas
    r"[A-Z][a-z]+[A-Z]",  # PascalCase: PostgreSQL
    r"[a-z]+_[a-z]+",  # snake_case: max_connections
    r"\b[A-Z]{2,}\b",  # ALL_CAPS: HS256, JWT, TTL
    r"/[a-z]+",  # URL paths: /health, /ready
    r"\d+[a-z]+",  # Numbers with units: 3600s, 70%
]

TECHNICAL_KEYWORDS = {
    "configuration",
    "parameter",
    "endpoint",
    "yaml",
    "json",
    "api",
    "database",
    "cache",
    "pool",
    "replica",
    "timeout",
    "kubernetes",
    "docker",
    "nginx",
    "redis",
    "kafka",
    "postgresql",
}


def detect_technical_score(query: str) -> float:
    """Calculate 0.0-1.0 technical score for query.

    Returns:
        float: Score from 0.0 (natural language) to 1.0 (highly technical)
    """
    score = 0.0
    query_lower = query.lower()

    # Check pattern matches (up to 0.5)
    pattern_matches = sum(1 for p in TECHNICAL_PATTERNS if re.search(p, query))
    score += min(pattern_matches * 0.1, 0.5)

    # Check keyword matches (up to 0.5)
    keyword_matches = sum(1 for kw in TECHNICAL_KEYWORDS if kw in query_lower)
    score += min(keyword_matches * 0.1, 0.5)

    return min(score, 1.0)


# ============================================================================
# NEGATION DETECTION
# ============================================================================

NEGATION_PATTERNS = {
    "prohibition": [r"\bshould(?:n\'t| not)\b", r"\bavoid\b", r"\bnever\b"],
    "failure": [r"\bdoes(?:n\'t| not)\s+work\b", r"\bwhy\s+can\'t\b"],
    "limitation": [r"\bcan(?:n\'t| not)\b", r"\bminimum\b", r"\bmaximum\b"],
    "consequence": [r"\bwhat\s+happens\s+if\s+(?:I\s+)?(?:don\'t|do not)\b"],
}


def detect_negation(query: str) -> dict:
    """Detect negation type and keywords in query.

    Returns:
        dict: {
            'has_negation': bool,
            'types': list[str],  # e.g. ['prohibition', 'failure']
            'matched_patterns': list[str],
            'negation_keywords': list[str]
        }
    """
    result = {
        "has_negation": False,
        "types": [],
        "matched_patterns": [],
        "negation_keywords": [],
    }

    query_lower = query.lower()

    for neg_type, patterns in NEGATION_PATTERNS.items():
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                result["has_negation"] = True
                if neg_type not in result["types"]:
                    result["types"].append(neg_type)
                result["matched_patterns"].append(pattern)
                result["negation_keywords"].append(match.group())

    return result


# ============================================================================
# RECIPROCAL RANK FUSION
# ============================================================================


def reciprocal_rank_fusion(
    result_sets: list[list[Chunk]], weights: Optional[list[float]] = None, k: int = 60
) -> list[Chunk]:
    """Combine multiple result sets using Reciprocal Rank Fusion.

    Refactored from enriched_hybrid_llm.py lines 267-303.

    Args:
        result_sets: List of ranked result lists (each is list of Chunks)
        weights: Optional weights for each result set (default: equal weights)
        k: RRF parameter (higher = more uniform blending)

    Returns:
        list[Chunk]: Fused results sorted by RRF score (highest first)

    Formula: RRF_score(d) = Σ (weight_i / (k + rank_i(d)))
    """
    if not result_sets:
        return []

    if weights is None:
        weights = [1.0] * len(result_sets)

    # Build chunk -> score mapping
    rrf_scores: dict[str, float] = {}  # chunk.id -> score
    chunk_lookup: dict[str, Chunk] = {}  # chunk.id -> Chunk

    for result_idx, results in enumerate(result_sets):
        weight = weights[result_idx]
        for rank, chunk in enumerate(results):
            chunk_id = chunk.id
            chunk_lookup[chunk_id] = chunk
            rrf_score = weight / (k + rank)
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0.0) + rrf_score

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    return [chunk_lookup[cid] for cid in sorted_ids]


# ============================================================================
# CHUNK FIELD EXTRACTION
# ============================================================================


def extract_chunk_fields(chunk: Chunk) -> dict[str, str]:
    """Extract structured fields from a chunk for BM25F scoring.

    Args:
        chunk: Chunk object with content attribute

    Returns:
        dict: {
            'heading': str,        # First heading found (or empty)
            'first_paragraph': str,  # First non-heading paragraph
            'body': str,           # Remaining content
            'code': str            # All code blocks concatenated
        }
    """
    content = chunk.content
    lines = content.split("\n")

    heading = ""
    first_paragraph = ""
    body_lines = []
    code_blocks = []

    in_code_block = False
    found_first_para = False

    for line in lines:
        # Track code blocks
        if line.strip().startswith("```"):
            in_code_block = not in_code_block
            continue

        if in_code_block:
            code_blocks.append(line)
            continue

        # Extract heading
        if line.startswith("#") and not heading:
            heading = line.lstrip("#").strip()
            continue

        # Extract first paragraph
        if not found_first_para and line.strip():
            first_paragraph = line.strip()
            found_first_para = True
            continue

        # Everything else is body
        body_lines.append(line)

    return {
        "heading": heading,
        "first_paragraph": first_paragraph,
        "body": "\n".join(body_lines),
        "code": "\n".join(code_blocks),
    }
