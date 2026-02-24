#!/usr/bin/env python3
"""Score extractor for convex fusion benchmark.

Extracts raw BM25 and semantic scores from the PLM retriever for benchmark queries.
This module does NOT modify the production code - it accesses components directly.

Usage:
    # From project root with main .venv activated:
    cd poc/convex_fusion_benchmark
    python score_extractor.py --output artifacts/raw_scores.json
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

# Add src to path for PLM imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.retriever import HybridRetriever
from plm.search.components.bm25 import BM25Index
from plm.search.components.embedder import EmbeddingEncoder
from plm.search.components.expander import QueryExpander
from plm.search.storage.sqlite import SQLiteStorage
from plm.search.types import Query, RewrittenQuery


# RRF parameters from HybridRetriever (must match production)
DEFAULT_RRF_K = 60
DEFAULT_BM25_WEIGHT = 1.0
DEFAULT_SEM_WEIGHT = 1.0
DEFAULT_CANDIDATE_MULTIPLIER = 10

EXPANDED_RRF_K = 10
EXPANDED_BM25_WEIGHT = 3.0
EXPANDED_SEM_WEIGHT = 0.3
EXPANDED_CANDIDATE_MULTIPLIER = 20


# PLM database paths (same as plm_vs_rag_benchmark)
PLM_DB_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db"
PLM_BM25_PATH = "/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data"

# Query file paths
CORPUS_DIR = Path(__file__).parent / "corpus"
INFORMED_PATH = CORPUS_DIR / "kubernetes" / "informed_questions.json"
NEEDLE_PATH = CORPUS_DIR / "needle_questions.json"
REALISTIC_PATH = CORPUS_DIR / "kubernetes" / "realistic_questions.json"


@dataclass
class QueryScores:
    """Raw scores for a single query."""
    query_id: str
    query: str
    ground_truth_doc_id: str
    bm25_scores: dict[int, float]
    semantic_scores: dict[int, float]
    chunk_doc_ids: dict[int, str]
    expansion_triggered: bool
    expanded_query: str
    rrf_k: int
    bm25_weight: float
    sem_weight: float


class ScoreExtractor:
    """Extract raw BM25 and semantic scores from PLM retriever."""
    
    def __init__(self, db_path: str = PLM_DB_PATH, bm25_path: str = PLM_BM25_PATH):
        """Initialize score extractor with PLM components.
        
        Args:
            db_path: Path to SQLite database
            bm25_path: Path to BM25 index directory
        """
        self.storage = SQLiteStorage(db_path)
        self.bm25_index = BM25Index.load(bm25_path)
        self.embedder = EmbeddingEncoder()
        self.expander = QueryExpander()
        
        # Load all chunks once
        self._all_chunks = self.storage.get_all_chunks()
        if not self._all_chunks:
            raise RuntimeError("No chunks found in storage")
        
        # Build embeddings matrix
        self._embeddings = np.array(
            [chunk["embedding"] for chunk in self._all_chunks],
            dtype=np.float32
        )
        
        # Build chunk_id -> index mapping
        self._chunk_id_to_idx = {
            chunk["id"]: i for i, chunk in enumerate(self._all_chunks)
        }
        
        # Build chunk index -> doc_id mapping
        self._chunk_doc_ids = {
            i: chunk["doc_id"] for i, chunk in enumerate(self._all_chunks)
        }
        
        print(f"[ScoreExtractor] Loaded {len(self._all_chunks)} chunks")
    
    def _expand_query(self, query: str) -> tuple[str, bool]:
        """Expand query using QueryExpander (same as retriever).
        
        Returns:
            (expanded_query, expansion_triggered)
        """
        original_query = Query(text=query)
        rewritten_query = RewrittenQuery(
            original=original_query,
            rewritten=query,
            model="passthrough",
        )
        expanded = self.expander.process(rewritten_query)
        expansion_triggered = len(expanded.expansions) > 0
        return expanded.expanded, expansion_triggered
    
    def get_scores(self, query: str, k: int = 10) -> tuple[dict[int, float], dict[int, float], bool, str, int, float, float]:
        """Get raw BM25 and semantic scores for a query.
        
        Uses same candidate pool size and parameters as production retriever.
        
        Returns:
            (bm25_scores, semantic_scores, expansion_triggered, expanded_query, rrf_k, bm25_weight, sem_weight)
        """
        expanded_query, expansion_triggered = self._expand_query(query)
        
        if expansion_triggered:
            rrf_k = EXPANDED_RRF_K
            bm25_weight = EXPANDED_BM25_WEIGHT
            sem_weight = EXPANDED_SEM_WEIGHT
            multiplier = EXPANDED_CANDIDATE_MULTIPLIER
        else:
            rrf_k = DEFAULT_RRF_K
            bm25_weight = DEFAULT_BM25_WEIGHT
            sem_weight = DEFAULT_SEM_WEIGHT
            multiplier = DEFAULT_CANDIDATE_MULTIPLIER
        
        n_candidates = min(k * multiplier, len(self._all_chunks))
        
        query_emb = np.array(
            self.embedder.process(expanded_query)["embedding"],
            dtype=np.float32
        )
        sem_scores_all = np.dot(self._embeddings, query_emb)
        
        sem_top_indices = np.argsort(sem_scores_all)[::-1][:n_candidates]
        semantic_scores = {
            int(idx): float(sem_scores_all[idx]) for idx in sem_top_indices
        }
        
        bm25_results = self.bm25_index.search(expanded_query, k=n_candidates)
        bm25_scores = {
            result["index"]: result["score"] for result in bm25_results
        }
        
        return bm25_scores, semantic_scores, expansion_triggered, expanded_query, rrf_k, bm25_weight, sem_weight
    
    def extract_query_scores(
        self,
        query_id: str,
        query: str,
        ground_truth_doc_id: str,
        k: int = 10,
    ) -> QueryScores:
        """Extract full score data for a single query."""
        bm25_scores, semantic_scores, expansion_triggered, expanded_query, rrf_k, bm25_weight, sem_weight = self.get_scores(query, k=k)
        
        all_indices = set(bm25_scores.keys()) | set(semantic_scores.keys())
        chunk_doc_ids = {idx: self._chunk_doc_ids[idx] for idx in all_indices}
        
        return QueryScores(
            query_id=query_id,
            query=query,
            ground_truth_doc_id=ground_truth_doc_id,
            bm25_scores=bm25_scores,
            semantic_scores=semantic_scores,
            chunk_doc_ids=chunk_doc_ids,
            expansion_triggered=expansion_triggered,
            expanded_query=expanded_query,
            rrf_k=rrf_k,
            bm25_weight=bm25_weight,
            sem_weight=sem_weight,
        )
    
    def get_chunk_count(self) -> int:
        """Return total number of chunks in index."""
        return len(self._all_chunks)
    
    def find_ground_truth_chunks(self, doc_id: str) -> list[int]:
        """Find all chunk indices belonging to a document.
        
        Args:
            doc_id: Document ID to search for
            
        Returns:
            List of chunk indices belonging to this document
        """
        return [
            i for i, chunk in enumerate(self._all_chunks)
            if doc_id in chunk["doc_id"]  # Substring match
        ]


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
    """Load realistic benchmark queries (sample of expanded variants).
    
    Args:
        limit: Maximum number of queries to return (default 50)
        
    Returns:
        List of query dicts with id, query, ground_truth_doc_id
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


def verify_ground_truth_coverage(
    extractor: ScoreExtractor,
    queries: list[dict],
    label: str,
) -> dict:
    """Verify that ground truth documents exist in the index.
    
    Args:
        extractor: ScoreExtractor instance
        queries: List of query dicts
        label: Label for this query set (e.g., "informed")
        
    Returns:
        Dict with coverage statistics
    """
    found = 0
    missing = []
    
    for q in queries:
        doc_id = q["ground_truth_doc_id"]
        chunks = extractor.find_ground_truth_chunks(doc_id)
        if chunks:
            found += 1
        else:
            missing.append(doc_id)
    
    coverage = {
        "label": label,
        "total": len(queries),
        "found": found,
        "missing_count": len(missing),
        "missing_doc_ids": missing[:5],  # First 5 only
    }
    
    print(f"[{label}] Ground truth coverage: {found}/{len(queries)} ({100*found/len(queries):.1f}%)")
    if missing:
        print(f"  Missing: {missing[:3]}...")
    
    return coverage


def extract_all_scores(
    extractor: ScoreExtractor,
    output_path: Path,
    k: int = 10,
) -> dict:
    """Extract scores for all benchmark queries and save to JSON."""
    results = {
        "metadata": {
            "k": k,
            "total_chunks": extractor.get_chunk_count(),
        },
        "coverage": {},
        "informed": [],
        "needle": [],
        "realistic": [],
    }
    
    # Load all query sets
    informed = load_informed_queries()
    needle = load_needle_queries()
    realistic = load_realistic_queries(limit=50)
    
    print(f"\nLoaded queries: informed={len(informed)}, needle={len(needle)}, realistic={len(realistic)}")
    
    # Verify ground truth coverage
    results["coverage"]["informed"] = verify_ground_truth_coverage(extractor, informed, "informed")
    results["coverage"]["needle"] = verify_ground_truth_coverage(extractor, needle, "needle")
    results["coverage"]["realistic"] = verify_ground_truth_coverage(extractor, realistic, "realistic")
    
    print("\n[Informed] Extracting scores...")
    for q in tqdm(informed, desc="Informed"):
        scores = extractor.extract_query_scores(
            q["id"], q["query"], q["ground_truth_doc_id"], k=k
        )
        results["informed"].append(asdict(scores))
    
    print("\n[Needle] Extracting scores...")
    for q in tqdm(needle, desc="Needle"):
        scores = extractor.extract_query_scores(
            q["id"], q["query"], q["ground_truth_doc_id"], k=k
        )
        results["needle"].append(asdict(scores))
    
    print("\n[Realistic] Extracting scores...")
    for q in tqdm(realistic, desc="Realistic"):
        scores = extractor.extract_query_scores(
            q["id"], q["query"], q["ground_truth_doc_id"], k=k
        )
        results["realistic"].append(asdict(scores))
    
    # Save results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSaved scores to {output_path}")
    print(f"  Informed: {len(results['informed'])} queries")
    print(f"  Needle: {len(results['needle'])} queries")
    print(f"  Realistic: {len(results['realistic'])} queries")
    
    return results


def main():
    """Run score extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract raw BM25 and semantic scores")
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/raw_scores.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Number of top results per retriever (default: 50)",
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default=PLM_DB_PATH,
        help="Path to PLM SQLite database",
    )
    parser.add_argument(
        "--bm25-path",
        type=str,
        default=PLM_BM25_PATH,
        help="Path to BM25 index directory",
    )
    
    args = parser.parse_args()
    
    print("Initializing ScoreExtractor...")
    extractor = ScoreExtractor(db_path=args.db_path, bm25_path=args.bm25_path)
    
    output_path = Path(__file__).parent / args.output
    extract_all_scores(extractor, output_path, k=args.top_k)


if __name__ == "__main__":
    main()
