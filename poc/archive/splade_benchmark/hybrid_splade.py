#!/usr/bin/env python3
"""Hybrid SPLADE + Semantic retrieval with RRF fusion.

Combines SPLADE sparse retrieval with dense semantic search using RRF.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python hybrid_splade.py
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

DEFAULT_RRF_K = 60
DEFAULT_SPLADE_WEIGHT = 1.0
DEFAULT_SEM_WEIGHT = 1.0


class HybridSPLADERetriever:
    """Hybrid retriever combining SPLADE and semantic search."""
    
    def __init__(
        self,
        db_path: Path = TEST_DB_PATH,
        splade_index_path: Path = SPLADE_INDEX_PATH,
        embedding_model: str = "BAAI/bge-base-en-v1.5",
    ):
        """Initialize hybrid retriever.
        
        Args:
            db_path: Path to test_db directory
            splade_index_path: Path to SPLADE index
            embedding_model: Sentence transformer model for query encoding
        """
        print("[HybridSPLADE] Loading SPLADE encoder...")
        self.splade_encoder = SPLADEEncoder()
        
        print("[HybridSPLADE] Loading SPLADE index...")
        self.splade_index = SPLADEIndex.load(str(splade_index_path), encoder=self.splade_encoder)
        
        print("[HybridSPLADE] Loading semantic embedder...")
        self.semantic_model = SentenceTransformer(embedding_model)
        
        print("[HybridSPLADE] Loading embeddings from SQLite...")
        self._load_embeddings(db_path / "index.db")
        
        print(f"[HybridSPLADE] Ready: {len(self.doc_ids)} chunks")
    
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
        
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        self.embeddings = self.embeddings / norms
    
    def search_semantic(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Get semantic search results.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of (index, score) tuples
        """
        query_emb = self.semantic_model.encode(query)
        query_emb = query_emb / np.linalg.norm(query_emb)
        
        scores = np.dot(self.embeddings, query_emb)
        
        top_indices = np.argsort(scores)[::-1][:k]
        return [(int(idx), float(scores[idx])) for idx in top_indices]
    
    def search_splade(self, query: str, k: int = 50) -> list[tuple[int, float]]:
        """Get SPLADE search results.
        
        Args:
            query: Query string
            k: Number of results
            
        Returns:
            List of (index, score) tuples
        """
        query_vec = self.splade_encoder.encode(query, return_tokens=False)
        assert isinstance(query_vec, dict)
        
        results = self.splade_index.search_with_vec(query_vec, k=k)
        return [(r["index"], r["score"]) for r in results]
    
    def search_hybrid(
        self,
        query: str,
        k: int = 10,
        n_candidates: int = 50,
        rrf_k: int = DEFAULT_RRF_K,
        splade_weight: float = DEFAULT_SPLADE_WEIGHT,
        sem_weight: float = DEFAULT_SEM_WEIGHT,
    ) -> list[dict]:
        """Search with SPLADE + Semantic hybrid using RRF.
        
        Args:
            query: Query string
            k: Number of final results
            n_candidates: Candidates from each retriever
            rrf_k: RRF parameter
            splade_weight: Weight for SPLADE scores in RRF
            sem_weight: Weight for semantic scores in RRF
            
        Returns:
            List of result dicts
        """
        splade_results = self.search_splade(query, k=n_candidates)
        semantic_results = self.search_semantic(query, k=n_candidates)
        
        rrf_scores: dict[int, float] = {}
        
        for rank, (idx, _) in enumerate(splade_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + splade_weight / (rrf_k + rank + 1)
        
        for rank, (idx, _) in enumerate(semantic_results):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + sem_weight / (rrf_k + rank + 1)
        
        sorted_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        
        results = []
        for idx in sorted_indices:
            results.append({
                "index": idx,
                "score": rrf_scores[idx],
                "doc_id": self.doc_ids[idx],
            })
        
        return results


class HybridBenchmark:
    """Benchmark for hybrid SPLADE + Semantic retrieval."""
    
    def __init__(
        self,
        db_path: Path = TEST_DB_PATH,
        splade_index_path: Path = SPLADE_INDEX_PATH,
    ):
        self.retriever = HybridSPLADERetriever(db_path, splade_index_path)
    
    def evaluate_query(
        self,
        query_id: str,
        query: str,
        ground_truth_doc_id: str,
        k: int = 10,
        n_candidates: int = 50,
        rrf_k: int = DEFAULT_RRF_K,
        splade_weight: float = DEFAULT_SPLADE_WEIGHT,
        sem_weight: float = DEFAULT_SEM_WEIGHT,
    ) -> QueryResult:
        """Evaluate a single query."""
        start_time = time.time()
        
        results = self.retriever.search_hybrid(
            query,
            k=k,
            n_candidates=n_candidates,
            rrf_k=rrf_k,
            splade_weight=splade_weight,
            sem_weight=sem_weight,
        )
        
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
        self,
        name: str,
        queries: list[dict],
        k: int = 10,
        **kwargs,
    ) -> BenchmarkResults:
        """Run benchmark on a query set."""
        results = []
        
        for q in tqdm(queries, desc=f"[{name}]"):
            result = self.evaluate_query(
                q["id"],
                q["query"],
                q["ground_truth_doc_id"],
                k=k,
                **kwargs,
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
    """Run hybrid SPLADE + Semantic benchmark."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hybrid SPLADE Benchmark")
    parser.add_argument("--k", type=int, default=10, help="Number of results")
    parser.add_argument("--output", type=str, default="artifacts/hybrid_splade_results.json")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF k parameter")
    parser.add_argument("--splade-weight", type=float, default=1.0, help="SPLADE weight")
    parser.add_argument("--sem-weight", type=float, default=1.0, help="Semantic weight")
    
    args = parser.parse_args()
    
    if not SPLADE_INDEX_PATH.exists():
        print(f"ERROR: SPLADE index not found at {SPLADE_INDEX_PATH}")
        return
    
    benchmark = HybridBenchmark()
    
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    print("\n" + "="*60)
    print("Running Hybrid SPLADE+Semantic benchmarks")
    print("="*60)
    
    kwargs = {
        "rrf_k": args.rrf_k,
        "splade_weight": args.splade_weight,
        "sem_weight": args.sem_weight,
    }
    
    informed_results = benchmark.run_benchmark("informed", informed, k=args.k, **kwargs)
    needle_results = benchmark.run_benchmark("needle", needle, k=args.k, **kwargs)
    realistic_results = benchmark.run_benchmark("realistic", realistic, k=args.k, **kwargs)
    
    print("\n" + "="*60)
    print("Hybrid SPLADE+Semantic Results")
    print("="*60)
    
    for results in [informed_results, needle_results, realistic_results]:
        print(f"\n{results.name.upper()}:")
        print(f"  MRR@{results.k}: {results.mrr:.4f}")
        print(f"  Recall@{results.k}: {results.recall_at_k:.4f}")
        print(f"  Hit@1: {results.hit_at_1:.1f}%")
        print(f"  Avg latency: {results.avg_latency_ms:.2f}ms")
        print(f"  Found: {results.found_count}/{results.total_queries}")
    
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_data = {
        "metadata": {
            "k": args.k,
            "rrf_k": args.rrf_k,
            "splade_weight": args.splade_weight,
            "sem_weight": args.sem_weight,
        },
        "informed": results_to_dict(informed_results),
        "needle": results_to_dict(needle_results),
        "realistic": results_to_dict(realistic_results),
    }
    
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
