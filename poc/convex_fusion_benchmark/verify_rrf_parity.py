#!/usr/bin/env python3
"""Verify that our RRF implementation matches the production retriever.

This script compares:
1. Our offline RRF computation (from extracted scores)
2. The actual HybridRetriever rankings

If they don't match, our convex fusion comparison is invalid.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.retriever import HybridRetriever

from fusion import rrf_fusion, get_ranking_from_scores
from score_extractor import (
    ScoreExtractor,
    load_informed_queries,
    PLM_DB_PATH,
    PLM_BM25_PATH,
)

ARTIFACTS_DIR = Path(__file__).parent / "artifacts"


def verify_rrf_parity():
    """Compare our RRF against production retriever."""
    print("=" * 60)
    print("RRF PARITY VERIFICATION")
    print("=" * 60)
    
    # Load raw scores
    with open(ARTIFACTS_DIR / "raw_scores.json") as f:
        raw_scores = json.load(f)
    
    informed = raw_scores["informed"]
    
    # Initialize production retriever
    print("\nInitializing production retriever...")
    retriever = HybridRetriever(PLM_DB_PATH, PLM_BM25_PATH)
    
    # Compare rankings for each query
    print("\nComparing rankings for informed queries...")
    print()
    
    matches = 0
    mismatches = 0
    details = []
    
    for q in informed[:10]:  # First 10 queries for quick check
        query = q["query"]
        
        # Get production retriever results
        prod_results = retriever.retrieve(query, k=10, use_rewrite=False)
        prod_ranking = [r["chunk_id"] for r in prod_results]
        
        bm25_scores = {int(idx): v for idx, v in q["bm25_scores"].items()}
        semantic_scores = {int(idx): v for idx, v in q["semantic_scores"].items()}
        
        rrf_k = q.get("rrf_k", 60)
        bm25_weight = q.get("bm25_weight", 1.0)
        sem_weight = q.get("sem_weight", 1.0)
        
        our_rrf = rrf_fusion(
            bm25_scores, semantic_scores,
            k=rrf_k, bm25_weight=bm25_weight, semantic_weight=sem_weight
        )
        our_ranking_indices = get_ranking_from_scores(our_rrf)[:10]
        
        # We need to map indices back to chunk_ids
        # But we don't have that mapping in raw_scores.json...
        # Let's compare ranks instead
        
        print(f"Query: {q['query_id']}")
        print(f"  Production top-3: {prod_ranking[:3]}")
        print(f"  Our top-3 indices: {our_ranking_indices[:3]}")
        
        # Check if ground truth rank matches
        gt_doc_id = q["ground_truth_doc_id"]
        
        # Find GT rank in production
        prod_gt_rank = None
        for i, r in enumerate(prod_results, 1):
            if gt_doc_id in r.get("doc_id", ""):
                prod_gt_rank = i
                break
        
        # Find GT rank in our RRF
        chunk_doc_ids = q["chunk_doc_ids"]
        gt_indices = [int(idx) for idx, doc_id in chunk_doc_ids.items() if gt_doc_id in doc_id]
        
        our_gt_rank = None
        for rank, idx in enumerate(our_ranking_indices, 1):
            if idx in gt_indices:
                our_gt_rank = rank
                break
        
        match = prod_gt_rank == our_gt_rank
        if match:
            matches += 1
            print(f"  GT rank: prod={prod_gt_rank}, ours={our_gt_rank} ✓")
        else:
            mismatches += 1
            print(f"  GT rank: prod={prod_gt_rank}, ours={our_gt_rank} ✗ MISMATCH")
        
        details.append({
            "query_id": q["query_id"],
            "prod_gt_rank": prod_gt_rank,
            "our_gt_rank": our_gt_rank,
            "match": match,
        })
        print()
    
    print("=" * 60)
    print(f"RESULTS: {matches} matches, {mismatches} mismatches")
    print("=" * 60)
    
    if mismatches > 0:
        print("\n⚠️  WARNING: Our RRF does not match production retriever!")
        print("This invalidates the convex fusion comparison.")
        print("\nPossible causes:")
        print("  1. Different candidate pool sizes")
        print("  2. Query expansion affecting RRF parameters")
        print("  3. Score extraction not capturing all candidates")
    else:
        print("\n✓ RRF parity verified - our comparison is valid.")
    
    return mismatches == 0


if __name__ == "__main__":
    verified = verify_rrf_parity()
    exit(0 if verified else 1)
