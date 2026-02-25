#!/usr/bin/env python3
"""Convex fusion testing for SPLADE + Semantic hybrid.

Tests SPLADE + Semantic with convex fusion (weighted sum) instead of RRF,
as specified in TODO 5.4.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python convex_fusion.py
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from splade_encoder import SPLADEEncoder
from splade_index import SPLADEIndex
from baseline_bm25 import (
    load_informed_queries,
    load_needle_queries,
    load_realistic_queries,
    QueryResult,
    BenchmarkResults,
    calculate_ndcg,
)


POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
ARTIFACTS_DIR = POC_DIR / "artifacts"
SPLADE_INDEX_PATH = ARTIFACTS_DIR / "splade_index"

# Alpha values to test (alpha = weight for semantic, 1-alpha = weight for SPLADE)
ALPHA_VALUES = [0.2, 0.3, 0.36, 0.4, 0.5, 0.6, 0.7]
DEFAULT_RRF_K = 60


class ConvexFusionRetriever:
    """Hybrid retriever with convex fusion (weighted combination)."""
    
    def __init__(
        self,
        db_path: Path = TEST_DB_PATH,
        splade_index_path: Path = SPLADE_INDEX_PATH,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        print("[ConvexFusion] Loading SPLADE encoder...")
        self.splade_encoder = SPLADEEncoder()
        
        print("[ConvexFusion] Loading SPLADE index...")
        self.splade_index = SPLADEIndex.load(str(splade_index_path), encoder=self.splade_encoder)
        
        print("[ConvexFusion] Loading semantic embedder...")
        self.semantic_model = SentenceTransformer(embedding_model)
        
        print("[ConvexFusion] Loading embeddings from SQLite...")
        self._load_embeddings(db_path / "index.db")
        
        print(f"[ConvexFusion] Ready: {len(self.doc_ids)} chunks")
    
    def _load_embeddings(self, sqlite_path: Path):
        """Load pre-computed embeddings from SQLite."""
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        
        rows = conn.execute("SELECT doc_id, embedding FROM chunks ORDER BY rowid").fetchall()
        conn.close()
        
        self.doc_ids = [row["doc_id"] for row in rows]
        
        embeddings = []
        for row in rows:
            if row["embedding"]:
                emb = np.frombuffer(row["embedding"], dtype=np.float32)
                embeddings.append(emb)
            else:
                embeddings.append(np.zeros(768, dtype=np.float32))
        
        self.embeddings = np.array(embeddings, dtype=np.float32)
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
    
    def _normalize_scores(self, scores: list[tuple[int, float]]) -> dict[int, float]:
        """Normalize scores to [0, 1] range using min-max normalization."""
        if not scores:
            return {}
        
        values = [s for _, s in scores]
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return {idx: 1.0 for idx, _ in scores}
        
        return {
            idx: (score - min_val) / (max_val - min_val)
            for idx, score in scores
        }
    
    def search_semantic(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Get semantic search results."""
        query_emb = self.semantic_model.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        scores = np.dot(self.embeddings, query_emb)
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_splade(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Get SPLADE search results."""
        query_vec = self.splade_encoder.encode(query, return_tokens=False)
        assert isinstance(query_vec, dict)
        
        results = self.splade_index.search_with_vec(query_vec, k=k)
        return [(r["index"], r["score"]) for r in results]
    
    def search_convex(
        self,
        query: str,
        alpha: float = 0.36,
        k: int = 10,
        n_candidates: int = 50,
    ) -> list[dict]:
        """Search with convex fusion (weighted combination).
        
        Args:
            query: Query string
            alpha: Weight for semantic scores (1-alpha for SPLADE)
            k: Number of final results
            n_candidates: Candidates from each retriever
            
        Returns:
            List of result dicts
        """
        splade_results = self.search_splade(query, k=n_candidates)
        semantic_results = self.search_semantic(query, k=n_candidates)
        
        # Normalize scores
        splade_norm = self._normalize_scores(splade_results)
        semantic_norm = self._normalize_scores(semantic_results)
        
        # Combine all candidates
        all_indices = set(splade_norm.keys()) | set(semantic_norm.keys())
        
        # Calculate combined scores
        combined_scores = {}
        for idx in all_indices:
            splade_score = splade_norm.get(idx, 0.0)
            semantic_score = semantic_norm.get(idx, 0.0)
            combined_scores[idx] = (1 - alpha) * splade_score + alpha * semantic_score
        
        # Sort and return top-k
        sorted_indices = sorted(combined_scores.keys(), key=lambda x: combined_scores[x], reverse=True)[:k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                "index": idx,
                "score": combined_scores[idx],
                "doc_id": self.doc_ids[idx],
                "splade_score": splade_norm.get(idx, 0.0),
                "semantic_score": semantic_norm.get(idx, 0.0),
            })
        
        return results
    
    def search_rrf(
        self,
        query: str,
        k: int = 10,
        n_candidates: int = 50,
        rrf_k: int = DEFAULT_RRF_K,
    ) -> list[dict]:
        """Search with RRF fusion for comparison."""
        splade_results = self.search_splade(query, k=n_candidates)
        semantic_results = self.search_semantic(query, k=n_candidates)
        
        rrf_scores: dict[int, float] = {}
        
        for rank, (idx, _) in enumerate(splade_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1.0 / (rrf_k + rank + 1)
        
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                "index": idx,
                "score": rrf_scores[idx],
                "doc_id": self.doc_ids[idx],
            })
        
        return results


def evaluate_query(
    retriever: ConvexFusionRetriever,
    query_id: str,
    query: str,
    ground_truth_doc_id: str,
    alpha: float,
    k: int = 10,
    use_rrf: bool = False,
) -> QueryResult:
    """Evaluate a single query."""
    start_time = time.time()
    
    if use_rrf:
        results = retriever.search_rrf(query, k=k)
    else:
        results = retriever.search_convex(query, alpha=alpha, k=k)
    
    latency_ms = (time.time() - start_time) * 1000
    
    rank = None
    score = None
    for i, r in enumerate(results):
        if ground_truth_doc_id in r["doc_id"]:
            rank = i + 1
            score = r["score"]
            break
    
    top_k_doc_ids = [r["doc_id"] for r in results]
    
    return QueryResult(
        query_id=query_id,
        query=query,
        ground_truth_doc_id=ground_truth_doc_id,
        rank=rank,
        score=score,
        top_k_doc_ids=top_k_doc_ids,
        latency_ms=latency_ms,
    )


def run_benchmark(
    retriever: ConvexFusionRetriever,
    name: str,
    queries: list[dict],
    alpha: float,
    k: int = 10,
    use_rrf: bool = False,
) -> BenchmarkResults:
    """Run benchmark on a query set."""
    results = []
    
    for q in tqdm(queries, desc=f"[{name}]", leave=False):
        result = evaluate_query(
            retriever,
            q["id"],
            q["query"],
            q["ground_truth_doc_id"],
            alpha=alpha,
            k=k,
            use_rrf=use_rrf,
        )
        results.append(result)
    
    ranks = [r.rank for r in results]
    found_ranks = [r for r in ranks if r is not None]
    
    mrr = sum(1.0 / r for r in found_ranks) / len(ranks) if ranks else 0
    recall_at_k = len(found_ranks) / len(ranks) if ranks else 0
    hit_at_1 = sum(1 for r in found_ranks if r <= 1) / len(ranks) if ranks else 0
    hit_at_5 = sum(1 for r in found_ranks if r <= 5) / len(ranks) if ranks else 0
    hit_at_10 = sum(1 for r in found_ranks if r <= 10) / len(ranks) if ranks else 0
    ndcg_at_10 = calculate_ndcg(ranks, k=10)
    avg_latency = float(np.mean([r.latency_ms for r in results]))
    
    return BenchmarkResults(
        name=name,
        total_queries=len(queries),
        k=k,
        mrr=mrr,
        recall_at_k=recall_at_k,
        hit_at_1=hit_at_1 * 100,
        hit_at_5=hit_at_5 * 100,
        hit_at_10=hit_at_10 * 100,
        ndcg_at_10=ndcg_at_10,
        avg_latency_ms=avg_latency,
        found_count=len(found_ranks),
        not_found_count=len(ranks) - len(found_ranks),
        per_query=results,
    )


def results_to_dict(results: BenchmarkResults) -> dict:
    """Convert BenchmarkResults to serializable dict."""
    return {
        "name": results.name,
        "total_queries": results.total_queries,
        "k": results.k,
        "mrr": results.mrr,
        "recall_at_k": results.recall_at_k,
        "hit_at_1": results.hit_at_1,
        "hit_at_5": results.hit_at_5,
        "hit_at_10": results.hit_at_10,
        "ndcg_at_10": results.ndcg_at_10,
        "avg_latency_ms": results.avg_latency_ms,
        "found_count": results.found_count,
        "not_found_count": results.not_found_count,
        "per_query": [asdict(q) for q in results.per_query],
    }


def main():
    """Run convex fusion benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Convex Fusion Benchmark")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--output", type=str, default="artifacts/convex_fusion_results.json")
    
    args = parser.parse_args()
    
    if not SPLADE_INDEX_PATH.exists():
        print(f"ERROR: SPLADE index not found at {SPLADE_INDEX_PATH}")
        return
    
    retriever = ConvexFusionRetriever()
    
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    all_results = {
        "alpha_sweep": {},
        "rrf_comparison": {},
        "best_alpha": {},
    }
    
    # Test RRF first for comparison
    print("\n" + "="*60)
    print("Testing RRF fusion (baseline)")
    print("="*60)
    
    rrf_informed = run_benchmark(retriever, "informed", informed, alpha=0, k=args.k, use_rrf=True)
    rrf_needle = run_benchmark(retriever, "needle", needle, alpha=0, k=args.k, use_rrf=True)
    rrf_realistic = run_benchmark(retriever, "realistic", realistic, alpha=0, k=args.k, use_rrf=True)
    
    all_results["rrf_comparison"] = {
        "informed": {"mrr": rrf_informed.mrr, "hit_at_1": rrf_informed.hit_at_1},
        "needle": {"mrr": rrf_needle.mrr, "hit_at_1": rrf_needle.hit_at_1},
        "realistic": {"mrr": rrf_realistic.mrr, "hit_at_1": rrf_realistic.hit_at_1},
    }
    
    print(f"\nRRF Results:")
    print(f"  Informed MRR:  {rrf_informed.mrr:.4f}")
    print(f"  Needle MRR:    {rrf_needle.mrr:.4f}")
    print(f"  Realistic MRR: {rrf_realistic.mrr:.4f}")
    
    # Alpha sweep for convex fusion
    print("\n" + "="*60)
    print("Testing Convex Fusion (alpha sweep)")
    print("="*60)
    
    best_informed_mrr = 0
    best_informed_alpha = 0
    
    for alpha in ALPHA_VALUES:
        print(f"\n--- Alpha = {alpha} ---")
        
        informed_results = run_benchmark(retriever, "informed", informed, alpha=alpha, k=args.k)
        needle_results = run_benchmark(retriever, "needle", needle, alpha=alpha, k=args.k)
        realistic_results = run_benchmark(retriever, "realistic", realistic, alpha=alpha, k=args.k)
        
        all_results["alpha_sweep"][str(alpha)] = {
            "informed": {"mrr": informed_results.mrr, "hit_at_1": informed_results.hit_at_1},
            "needle": {"mrr": needle_results.mrr, "hit_at_1": needle_results.hit_at_1},
            "realistic": {"mrr": realistic_results.mrr, "hit_at_1": realistic_results.hit_at_1},
        }
        
        print(f"  Informed MRR:  {informed_results.mrr:.4f}")
        print(f"  Needle MRR:    {needle_results.mrr:.4f}")
        print(f"  Realistic MRR: {realistic_results.mrr:.4f}")
        
        if informed_results.mrr > best_informed_mrr:
            best_informed_mrr = informed_results.mrr
            best_informed_alpha = alpha
    
    all_results["best_alpha"] = {
        "value": best_informed_alpha,
        "informed_mrr": best_informed_mrr,
    }
    
    # Summary
    print("\n" + "="*60)
    print("CONVEX FUSION SUMMARY")
    print("="*60)
    print(f"Best alpha for Informed: {best_informed_alpha} (MRR = {best_informed_mrr:.4f})")
    print(f"RRF Informed MRR: {rrf_informed.mrr:.4f}")
    print(f"\nConvex vs RRF comparison:")
    print(f"  Convex (alpha={best_informed_alpha}): {best_informed_mrr:.4f}")
    print(f"  RRF:                                  {rrf_informed.mrr:.4f}")
    
    if best_informed_mrr > rrf_informed.mrr:
        print(f"  Winner: Convex Fusion (+{(best_informed_mrr - rrf_informed.mrr)*100:.1f}%)")
    else:
        print(f"  Winner: RRF (+{(rrf_informed.mrr - best_informed_mrr)*100:.1f}%)")
    
    # Save results
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
