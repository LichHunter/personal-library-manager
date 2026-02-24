#!/usr/bin/env python3
"""BM25-only baseline extraction for SPLADE benchmark.

This module runs BM25-only retrieval (no semantic component) on all benchmark
queries to establish baselines for comparison with SPLADE.

Usage:
    cd poc/splade_benchmark
    .venv/bin/python baseline_bm25.py
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Optional

import bm25s
import numpy as np
from tqdm import tqdm


# Paths
POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
CORPUS_DIR = POC_DIR / "corpus"
ARTIFACTS_DIR = POC_DIR / "artifacts"

# Query file paths
INFORMED_PATH = CORPUS_DIR / "kubernetes" / "informed_questions.json"
NEEDLE_PATH = CORPUS_DIR / "needle_questions.json"
REALISTIC_PATH = CORPUS_DIR / "kubernetes" / "realistic_questions.json"


@dataclass
class QueryResult:
    """Result for a single query."""
    query_id: str
    query: str
    ground_truth_doc_id: str
    rank: Optional[int]  # None if not found in top-k
    score: Optional[float]
    top_k_doc_ids: list[str]
    latency_ms: float


@dataclass
class BenchmarkResults:
    """Results for a benchmark query set."""
    name: str
    total_queries: int
    k: int
    mrr: float
    recall_at_k: float
    hit_at_1: float
    hit_at_5: float
    hit_at_10: float
    ndcg_at_10: float
    avg_latency_ms: float
    found_count: int
    not_found_count: int
    per_query: list[QueryResult] = field(default_factory=list)


def calculate_ndcg(ranks: list[Optional[int]], k: int = 10) -> float:
    """Calculate NDCG@k: DCG = 1/log2(r+1), IDCG = 1.0, NDCG = DCG/IDCG."""
    if not ranks:
        return 0.0
    
    ndcg_scores = []
    for rank in ranks:
        if rank is not None and rank <= k:
            dcg = 1.0 / np.log2(rank + 1)
            ndcg = dcg
        else:
            ndcg = 0.0
        ndcg_scores.append(ndcg)
    
    return float(np.mean(ndcg_scores))


class BM25Baseline:
    """BM25-only baseline retriever."""
    
    def __init__(self, db_path: Path = TEST_DB_PATH):
        """Initialize BM25 baseline.
        
        Args:
            db_path: Path to test_db directory
        """
        self.db_path = db_path
        
        # Load BM25 index
        print(f"[BM25Baseline] Loading BM25 index from {db_path}")
        self.bm25 = bm25s.BM25.load(str(db_path), load_corpus=True, mmap=False)
        
        # Get corpus - each item has 'id' and 'text'
        raw_corpus = self.bm25.corpus
        self.corpus = [
            item["text"] if isinstance(item, dict) else str(item)
            for item in raw_corpus
        ]
        
        # Clear corpus from BM25 so retrieve() returns int indices
        self.bm25.corpus = None
        
        print(f"[BM25Baseline] Loaded {len(self.corpus)} chunks")
        
        # Load chunk -> doc_id mapping from SQLite
        self._load_doc_id_mapping(db_path / "index.db")
        
        # Index size measurement
        self._measure_index_size(db_path)
    
    def _load_doc_id_mapping(self, sqlite_path: Path):
        """Load mapping from chunk index to doc_id."""
        print(f"[BM25Baseline] Loading doc_id mapping from {sqlite_path}")
        
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        
        # Get all chunks ordered by rowid (matches BM25 index order)
        rows = conn.execute("SELECT doc_id FROM chunks ORDER BY rowid").fetchall()
        
        self.idx_to_doc_id = {i: row["doc_id"] for i, row in enumerate(rows)}
        
        conn.close()
        
        print(f"[BM25Baseline] Mapped {len(self.idx_to_doc_id)} chunks to doc_ids")
    
    def _measure_index_size(self, db_path: Path):
        """Measure BM25 index size on disk."""
        index_files = [
            "data.csc.index.npy",
            "indices.csc.index.npy",
            "indptr.csc.index.npy",
            "params.index.json",
            "vocab.index.json",
        ]
        
        total_size = 0
        for f in index_files:
            fpath = db_path / f
            if fpath.exists():
                total_size += fpath.stat().st_size
        
        self.index_size_bytes = total_size
        self.index_size_mb = total_size / (1024 * 1024)
        print(f"[BM25Baseline] Index size: {self.index_size_mb:.2f} MB")
    
    def search(self, query: str, k: int = 10) -> list[dict]:
        """Search with BM25.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dicts with 'index', 'score', 'doc_id', 'content'
        """
        # Tokenize like production code
        query_tokens = [query.lower().split()]
        actual_k = min(k, len(self.corpus))
        
        # Retrieve
        indices, scores = self.bm25.retrieve(
            query_tokens,
            k=actual_k,
            show_progress=False,
        )
        
        results = []
        for i in range(actual_k):
            idx = int(indices[0, i])
            score = float(scores[0, i])
            doc_id = self.idx_to_doc_id.get(idx, f"unknown_{idx}")
            content = self.corpus[idx] if idx < len(self.corpus) else ""
            
            results.append({
                "index": idx,
                "score": score,
                "doc_id": doc_id,
                "content": content[:200],  # Truncate for storage
            })
        
        return results
    
    def evaluate_query(
        self,
        query_id: str,
        query: str,
        ground_truth_doc_id: str,
        k: int = 10,
    ) -> QueryResult:
        """Evaluate a single query.
        
        Args:
            query_id: Unique query identifier
            query: Query text
            ground_truth_doc_id: Expected document ID
            k: Number of results to retrieve
            
        Returns:
            QueryResult with rank and metrics
        """
        start_time = time.time()
        results = self.search(query, k=k)
        latency_ms = (time.time() - start_time) * 1000
        
        # Find rank of ground truth (substring match)
        rank = None
        score = None
        for i, r in enumerate(results):
            if ground_truth_doc_id in r["doc_id"]:
                rank = i + 1  # 1-indexed
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
    ) -> BenchmarkResults:
        """Run benchmark on a query set.
        
        Args:
            name: Benchmark name (e.g., "informed")
            queries: List of query dicts with 'id', 'query', 'ground_truth_doc_id'
            k: Number of results to retrieve
            
        Returns:
            BenchmarkResults with aggregate metrics
        """
        results = []
        
        for q in tqdm(queries, desc=f"[{name}]"):
            result = self.evaluate_query(
                q["id"],
                q["query"],
                q["ground_truth_doc_id"],
                k=k,
            )
            results.append(result)
        
        # Calculate metrics
        ranks = [r.rank for r in results]
        found_ranks = [r for r in ranks if r is not None]
        
        # MRR@k
        mrr = sum(1.0 / r for r in found_ranks) / len(ranks) if ranks else 0
        
        # Recall@k (fraction found in top-k)
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


def load_informed_queries() -> list[dict]:
    """Load informed benchmark queries (25 total)."""
    with open(INFORMED_PATH) as f:
        data = json.load(f)
    
    queries = []
    for i, q in enumerate(data["questions"]):
        queries.append({
            "id": f"informed_{i:03d}",
            "query": q["original_instruction"],
            "ground_truth_doc_id": q["doc_id"],
        })
    
    return queries


def load_needle_queries() -> list[dict]:
    """Load needle benchmark queries (20 total)."""
    with open(NEEDLE_PATH) as f:
        data = json.load(f)
    
    needle_doc_id = data["needle_doc_id"]
    queries = []
    for q in data["questions"]:
        queries.append({
            "id": q["id"],
            "query": q["question"],
            "ground_truth_doc_id": needle_doc_id,
        })
    
    return queries


def load_realistic_queries(limit: int = 50) -> list[dict]:
    """Load realistic benchmark queries (sample).
    
    Args:
        limit: Maximum number of queries to return
        
    Returns:
        List of query dicts
    """
    with open(REALISTIC_PATH) as f:
        data = json.load(f)
    
    queries = []
    for i, q in enumerate(data["questions"]):
        doc_id = q["doc_id"]
        
        # Add q1 variant
        if q.get("realistic_q1"):
            queries.append({
                "id": f"realistic_{i:03d}_q1",
                "query": q["realistic_q1"],
                "ground_truth_doc_id": doc_id,
            })
        
        # Add q2 variant
        if q.get("realistic_q2"):
            queries.append({
                "id": f"realistic_{i:03d}_q2",
                "query": q["realistic_q2"],
                "ground_truth_doc_id": doc_id,
            })
        
        if len(queries) >= limit:
            break
    
    return queries[:limit]


def main():
    """Run BM25 baseline extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BM25 Baseline Extraction")
    parser.add_argument(
        "--k",
        type=int,
        default=10,
        help="Number of results to retrieve (default: 10)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/baseline_bm25.json",
        help="Output file path",
    )
    
    args = parser.parse_args()
    
    # Initialize baseline
    baseline = BM25Baseline()
    
    # Load queries
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    # Run benchmarks
    print("\n" + "="*60)
    print("Running BM25-only benchmarks")
    print("="*60)
    
    informed_results = baseline.run_benchmark("informed", informed, k=args.k)
    needle_results = baseline.run_benchmark("needle", needle, k=args.k)
    realistic_results = baseline.run_benchmark("realistic", realistic, k=args.k)
    
    # Print summary
    print("\n" + "="*60)
    print("BM25-Only Baseline Results")
    print("="*60)
    
    for results in [informed_results, needle_results, realistic_results]:
        print(f"\n{results.name.upper()}:")
        print(f"  MRR@{results.k}: {results.mrr:.4f}")
        print(f"  Recall@{results.k}: {results.recall_at_k:.4f}")
        print(f"  Hit@1: {results.hit_at_1:.1f}%")
        print(f"  Hit@5: {results.hit_at_5:.1f}%")
        print(f"  Hit@10: {results.hit_at_10:.1f}%")
        print(f"  Avg latency: {results.avg_latency_ms:.2f}ms")
        print(f"  Found: {results.found_count}/{results.total_queries}")
    
    # Save results
    output_path = POC_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    def results_to_dict(results: BenchmarkResults) -> dict:
        d = {
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
        return d
    
    output_data = {
        "metadata": {
            "k": args.k,
            "total_chunks": len(baseline.corpus),
            "index_size_mb": baseline.index_size_mb,
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
