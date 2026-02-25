#!/usr/bin/env python3
"""
Test script to investigate BMX API behavior for RRF compatibility.

Tests:
1. BMX search return type and structure
2. Whether BMX returns scores for ALL documents or only top-k
3. Score scale comparison between BMX and BM25
4. API compatibility with RRF fusion requirements
"""

import numpy as np
from rank_bm25 import BM25Okapi
from baguetter.indices import BMXSparseIndex

# Sample documents similar to our corpus
SAMPLE_DOCS = [
    "recovery point objective RPO is the maximum acceptable amount of data loss measured in time",
    "recovery time objective RTO is the maximum acceptable downtime for a system",
    "JWT tokens contain claims like iat issued at and exp expiration time",
    "PostgreSQL is a relational database system for storing structured data",
    "Redis is an in-memory data store used for caching and sessions",
    "Kafka is a distributed event streaming platform for real-time data",
    "Prometheus is a monitoring system that collects metrics from applications",
    "Grafana is a visualization platform for displaying metrics and dashboards",
    "Horizontal Pod Autoscaler HPA automatically scales Kubernetes deployments",
    "CPU utilization is a key metric for determining when to scale applications",
]

DOC_IDS = [f"doc_{i}" for i in range(len(SAMPLE_DOCS))]


def test_bmx_api():
    """Test BMX API structure and return types."""
    print("\n" + "=" * 80)
    print("TEST 1: BMX API Structure")
    print("=" * 80)

    idx = BMXSparseIndex()
    idx.add_many(DOC_IDS, SAMPLE_DOCS)

    query = "recovery point objective"
    results = idx.search(query, top_k=5)

    print(f"\nQuery: '{query}'")
    print(f"Search result type: {type(results)}")
    print(f"Result attributes: {dir(results)}")

    # Check what attributes are available
    if hasattr(results, "ids"):
        print(f"\nResult IDs: {results.ids}")
    if hasattr(results, "scores"):
        print(f"Result scores: {results.scores}")
    if hasattr(results, "documents"):
        print(f"Result documents: {results.documents}")

    # Try to access as dict-like
    try:
        print(f"\nTrying dict-like access:")
        print(f"  results['ids']: {results['ids']}")
        print(f"  results['scores']: {results['scores']}")
    except (TypeError, KeyError) as e:
        print(f"  Dict-like access failed: {e}")

    return idx, results


def test_full_corpus_scores(idx):
    """Test if BMX returns scores for ALL documents."""
    print("\n" + "=" * 80)
    print("TEST 2: Full Corpus Scores")
    print("=" * 80)

    query = "JWT authentication"
    n_docs = len(SAMPLE_DOCS)

    # Test 1: Search with top_k = number of docs
    print(f"\nQuery: '{query}'")
    print(f"Total documents: {n_docs}")

    results_all = idx.search(query, top_k=n_docs)

    if hasattr(results_all, "scores"):
        scores = results_all.scores
        print(f"\nScores returned: {len(scores)}")
        print(f"Scores: {scores}")
        print(f"Score range: {min(scores):.6f} to {max(scores):.6f}")

        # Check if we got all documents
        if len(scores) == n_docs:
            print("✓ BMX returns scores for ALL documents when top_k >= n_docs")
        else:
            print(f"✗ BMX only returned {len(scores)} scores for {n_docs} documents")

    # Test 2: Search with top_k < number of docs
    print(f"\n--- Testing with top_k=3 (less than {n_docs} docs) ---")
    results_partial = idx.search(query, top_k=3)

    if hasattr(results_partial, "scores"):
        scores_partial = results_partial.scores
        print(f"Scores returned: {len(scores_partial)}")
        print(f"Scores: {scores_partial}")

        if len(scores_partial) == 3:
            print("✗ BMX only returns top-k scores (not full corpus)")
        else:
            print(f"? Unexpected: got {len(scores_partial)} scores for top_k=3")


def test_score_scale_comparison():
    """Compare BMX score scale to BM25 score scale."""
    print("\n" + "=" * 80)
    print("TEST 3: Score Scale Comparison (BMX vs BM25)")
    print("=" * 80)

    # BM25 setup
    tokenized = [doc.lower().split() for doc in SAMPLE_DOCS]
    bm25 = BM25Okapi(tokenized)

    # BMX setup
    bmx_idx = BMXSparseIndex()
    bmx_idx.add_many(DOC_IDS, SAMPLE_DOCS)

    query = "recovery point objective"
    query_tokens = query.lower().split()

    # Get BM25 scores for all documents
    bm25_scores = bm25.get_scores(query_tokens)

    # Get BMX scores
    bmx_results = bmx_idx.search(query, top_k=len(SAMPLE_DOCS))

    print(f"\nQuery: '{query}'")
    print(f"\nBM25 Scores:")
    print(f"  Type: {type(bm25_scores)}")
    print(
        f"  Shape: {bm25_scores.shape if hasattr(bm25_scores, 'shape') else len(bm25_scores)}"
    )
    print(f"  Range: {min(bm25_scores):.6f} to {max(bm25_scores):.6f}")
    print(f"  Mean: {np.mean(bm25_scores):.6f}")
    print(f"  Std: {np.std(bm25_scores):.6f}")
    print(f"  Top 5 scores: {sorted(bm25_scores, reverse=True)[:5]}")

    if hasattr(bmx_results, "scores"):
        bmx_scores = bmx_results.scores
        print(f"\nBMX Scores:")
        print(f"  Type: {type(bmx_scores)}")
        print(f"  Length: {len(bmx_scores)}")
        print(f"  Range: {min(bmx_scores):.6f} to {max(bmx_scores):.6f}")
        print(f"  Mean: {np.mean(bmx_scores):.6f}")
        print(f"  Std: {np.std(bmx_scores):.6f}")
        print(f"  Top 5 scores: {sorted(bmx_scores, reverse=True)[:5]}")

        # Compare scales
        print(f"\nScale Comparison:")
        print(f"  BM25 max / BMX max: {max(bm25_scores) / max(bmx_scores):.2f}x")
        print(
            f"  BM25 mean / BMX mean: {np.mean(bm25_scores) / np.mean(bmx_scores):.2f}x"
        )


def test_rrf_compatibility():
    """Test if BMX can be used in RRF fusion like BM25."""
    print("\n" + "=" * 80)
    print("TEST 4: RRF Fusion Compatibility")
    print("=" * 80)

    # Setup
    tokenized = [doc.lower().split() for doc in SAMPLE_DOCS]
    bm25 = BM25Okapi(tokenized)
    bmx_idx = BMXSparseIndex()
    bmx_idx.add_many(DOC_IDS, SAMPLE_DOCS)

    query = "database storage"
    query_tokens = query.lower().split()

    # Get scores from both
    bm25_scores = bm25.get_scores(query_tokens)
    bmx_results = bmx_idx.search(query, top_k=len(SAMPLE_DOCS))

    print(f"\nQuery: '{query}'")
    print(f"Number of documents: {len(SAMPLE_DOCS)}")

    # Check if we can compute RRF
    if hasattr(bmx_results, "scores"):
        bmx_scores = bmx_results.scores

        # Simulate RRF calculation
        rrf_k = 60
        n_candidates = min(10, len(SAMPLE_DOCS))

        # Get ranks
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        bmx_ranks = np.argsort(bmx_scores)[::-1]

        print(f"\nRRF Calculation (rrf_k={rrf_k}, n_candidates={n_candidates}):")
        print(f"  BM25 top-5 ranks: {bm25_ranks[:5]}")
        print(f"  BMX top-5 ranks: {bmx_ranks[:5]}")

        # Compute RRF scores
        rrf_scores = {}

        for rank, idx in enumerate(bm25_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)

        for rank, idx in enumerate(bmx_ranks[:n_candidates]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank)

        # Get top results
        top_idx = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[
            :5
        ]

        print(f"\nRRF Fusion Results (top 5):")
        for rank, idx in enumerate(top_idx):
            bm25_rank = np.where(bm25_ranks == idx)[0][0] if idx in bm25_ranks else -1
            bmx_rank = np.where(bmx_ranks == idx)[0][0] if idx in bmx_ranks else -1
            print(
                f"  [{rank + 1}] doc_idx={idx} rrf_score={rrf_scores[idx]:.4f} "
                f"(bm25_rank={bm25_rank}, bmx_rank={bmx_rank})"
            )

        print(f"\n✓ RRF fusion is possible with BMX scores")
    else:
        print(f"✗ Cannot compute RRF: BMX results don't have scores attribute")


def test_api_methods():
    """Explore available BMX API methods."""
    print("\n" + "=" * 80)
    print("TEST 5: BMX API Methods")
    print("=" * 80)

    idx = BMXSparseIndex()
    idx.add_many(DOC_IDS, SAMPLE_DOCS)

    print(f"\nBMXSparseIndex methods:")
    methods = [m for m in dir(idx) if not m.startswith("_")]
    for method in sorted(methods):
        print(f"  - {method}")

    # Try to understand search signature
    import inspect

    print(f"\nBMXSparseIndex.search signature:")
    try:
        sig = inspect.signature(idx.search)
        print(f"  {sig}")
    except Exception as e:
        print(f"  Could not get signature: {e}")

    # Try to understand add_many signature
    print(f"\nBMXSparseIndex.add_many signature:")
    try:
        sig = inspect.signature(idx.add_many)
        print(f"  {sig}")
    except Exception as e:
        print(f"  Could not get signature: {e}")


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("BMX API INVESTIGATION FOR RRF COMPATIBILITY")
    print("=" * 80)

    try:
        # Test 1: API structure
        idx, results = test_bmx_api()

        # Test 2: Full corpus scores
        test_full_corpus_scores(idx)

        # Test 3: Score scale comparison
        test_score_scale_comparison()

        # Test 4: RRF compatibility
        test_rrf_compatibility()

        # Test 5: API methods
        test_api_methods()

        print("\n" + "=" * 80)
        print("INVESTIGATION COMPLETE")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        import traceback

        traceback.print_exc()
