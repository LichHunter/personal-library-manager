#!/usr/bin/env python3
"""SPLADE inverted index for sparse retrieval.

This module provides SPLADEIndex which:
1. Encodes documents with SPLADE into sparse vectors
2. Builds an inverted index from sparse vectors
3. Provides fast retrieval using dot product scoring

Usage:
    # Build index
    index = SPLADEIndex()
    index.build_from_chunks(chunks, doc_ids)
    index.save("artifacts/splade_index")
    
    # Load and search
    index = SPLADEIndex.load("artifacts/splade_index")
    results = index.search("kubernetes pod", k=10)
"""

from __future__ import annotations

import json
import pickle
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import sparse
from tqdm import tqdm

from splade_encoder import SPLADEEncoder, EncodingStats


POC_DIR = Path(__file__).parent
TEST_DB_PATH = POC_DIR / "test_db"
ARTIFACTS_DIR = POC_DIR / "artifacts"


@dataclass
class IndexStats:
    """Statistics about the SPLADE index."""
    total_chunks: int
    total_terms: int
    avg_terms_per_doc: float
    min_terms_per_doc: int
    max_terms_per_doc: int
    encoding_time_s: float
    index_build_time_s: float
    index_size_mb: float


class SPLADEIndex:
    """SPLADE inverted index for sparse retrieval.
    
    Uses a CSR sparse matrix for efficient storage and retrieval.
    Each row represents a document, each column represents a vocabulary term.
    """
    
    def __init__(self, encoder: Optional[SPLADEEncoder] = None):
        """Initialize SPLADE index.
        
        Args:
            encoder: SPLADEEncoder instance (created if None)
        """
        self.encoder = encoder
        self.sparse_matrix: Optional[sparse.csr_matrix] = None
        self.doc_ids: list[str] = []
        self.contents: list[str] = []
        self.vocab_size: int = 0
        self.stats: Optional[IndexStats] = None
    
    def _ensure_encoder(self):
        """Ensure encoder is loaded."""
        if self.encoder is None:
            self.encoder = SPLADEEncoder()
    
    def build_from_db(
        self,
        db_path: Path = TEST_DB_PATH,
        batch_size: int = 32,
        use_enriched: bool = False,
        limit: Optional[int] = None,
    ) -> IndexStats:
        """Build index from SQLite database.
        
        Args:
            db_path: Path to test_db directory
            batch_size: Batch size for encoding
            use_enriched: If True, use enriched_content instead of content
            
        Returns:
            IndexStats with build statistics
        """
        self._ensure_encoder()
        
        sqlite_path = db_path / "index.db"
        print(f"[SPLADEIndex] Loading chunks from {sqlite_path}")
        
        conn = sqlite3.connect(str(sqlite_path))
        conn.row_factory = sqlite3.Row
        
        content_col = "enriched_content" if use_enriched else "content"
        rows = conn.execute(
            f"SELECT doc_id, {content_col} as content FROM chunks ORDER BY rowid"
        ).fetchall()
        conn.close()
        
        self.doc_ids = [row["doc_id"] for row in rows]
        self.contents = [row["content"] or "" for row in rows]
        
        if limit is not None and limit < len(self.contents):
            print(f"[SPLADEIndex] Limiting to first {limit} chunks")
            self.doc_ids = self.doc_ids[:limit]
            self.contents = self.contents[:limit]
        
        print(f"[SPLADEIndex] Loaded {len(self.contents)} chunks")
        
        return self._build_index(batch_size)
    
    def _build_index(self, batch_size: int = 32) -> IndexStats:
        """Build sparse matrix from encoded documents.
        
        Args:
            batch_size: Batch size for encoding
            
        Returns:
            IndexStats with build statistics
        """
        self._ensure_encoder()
        
        print(f"[SPLADEIndex] Encoding {len(self.contents)} chunks with SPLADE...")
        
        encode_start = time.time()
        sparse_vecs = self.encoder.encode_batch(
            self.contents,
            batch_size=batch_size,
            show_progress=True,
        )
        encode_time = time.time() - encode_start
        
        print(f"[SPLADEIndex] Encoding complete in {encode_time:.2f}s")
        
        terms_per_doc = [len(v) for v in sparse_vecs]
        
        print(f"[SPLADEIndex] Building sparse matrix...")
        build_start = time.time()
        
        self.vocab_size = self.encoder.vocab_size
        
        rows = []
        cols = []
        data = []
        
        for doc_idx, sparse_vec in enumerate(sparse_vecs):
            for term_id, weight in sparse_vec.items():
                rows.append(doc_idx)
                cols.append(term_id)
                data.append(weight)
        
        self.sparse_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(len(self.contents), self.vocab_size),
            dtype=np.float32,
        )
        
        build_time = time.time() - build_start
        print(f"[SPLADEIndex] Index built in {build_time:.2f}s")
        
        nnz = self.sparse_matrix.nnz
        index_size_mb = (
            self.sparse_matrix.data.nbytes +
            self.sparse_matrix.indices.nbytes +
            self.sparse_matrix.indptr.nbytes
        ) / (1024 * 1024)
        
        print(f"[SPLADEIndex] Matrix shape: {self.sparse_matrix.shape}")
        print(f"[SPLADEIndex] Non-zero entries: {nnz:,}")
        print(f"[SPLADEIndex] Index size: {index_size_mb:.2f} MB")
        
        self.stats = IndexStats(
            total_chunks=len(self.contents),
            total_terms=nnz,
            avg_terms_per_doc=np.mean(terms_per_doc),
            min_terms_per_doc=min(terms_per_doc),
            max_terms_per_doc=max(terms_per_doc),
            encoding_time_s=encode_time,
            index_build_time_s=build_time,
            index_size_mb=index_size_mb,
        )
        
        return self.stats
    
    def search(
        self,
        query: str,
        k: int = 10,
    ) -> list[dict]:
        """Search the index.
        
        Args:
            query: Query string
            k: Number of results to return
            
        Returns:
            List of dicts with 'index', 'score', 'doc_id', 'content'
        """
        if self.sparse_matrix is None:
            raise RuntimeError("Index not built. Call build_from_db() first.")
        
        self._ensure_encoder()
        assert self.encoder is not None
        
        query_vec = self.encoder.encode(query, return_tokens=False)
        assert isinstance(query_vec, dict)
        
        query_sparse = sparse.csr_matrix(
            ([query_vec[term_id] for term_id in query_vec.keys()],
             ([0] * len(query_vec), list(query_vec.keys()))),
            shape=(1, self.vocab_size),
            dtype=np.float32,
        )
        
        scores = self.sparse_matrix.dot(query_sparse.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "doc_id": self.doc_ids[idx],
                "content": self.contents[idx][:200],
            })
        
        return results
    
    def search_with_vec(
        self,
        query_vec: dict[int, float],
        k: int = 10,
    ) -> list[dict]:
        """Search with pre-encoded query vector.
        
        Args:
            query_vec: Pre-encoded sparse query vector
            k: Number of results to return
            
        Returns:
            List of dicts with 'index', 'score', 'doc_id', 'content'
        """
        if self.sparse_matrix is None:
            raise RuntimeError("Index not built. Call build_from_db() first.")
        
        query_sparse = sparse.csr_matrix(
            ([query_vec[k] for k in query_vec.keys()],
             ([0] * len(query_vec), list(query_vec.keys()))),
            shape=(1, self.vocab_size),
            dtype=np.float32,
        )
        
        scores = self.sparse_matrix.dot(query_sparse.T).toarray().flatten()
        
        top_indices = np.argsort(scores)[::-1][:k]
        
        results = []
        for idx in top_indices:
            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "doc_id": self.doc_ids[idx],
                "content": self.contents[idx][:200],
            })
        
        return results
    
    def save(self, path: str):
        """Save index to disk.
        
        Args:
            path: Directory path to save index
        """
        if self.sparse_matrix is None:
            raise RuntimeError("Index not built. Nothing to save.")
        
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        sparse.save_npz(save_path / "matrix.npz", self.sparse_matrix)
        
        metadata = {
            "doc_ids": self.doc_ids,
            "vocab_size": self.vocab_size,
            "stats": asdict(self.stats) if self.stats else None,
        }
        with open(save_path / "metadata.json", "w") as f:
            json.dump(metadata, f)
        
        with open(save_path / "contents.pkl", "wb") as f:
            pickle.dump(self.contents, f)
        
        print(f"[SPLADEIndex] Saved to {save_path}")
    
    @classmethod
    def load(cls, path: str, encoder: Optional[SPLADEEncoder] = None) -> "SPLADEIndex":
        """Load index from disk.
        
        Args:
            path: Directory path to load from
            encoder: SPLADEEncoder instance (for search)
            
        Returns:
            Loaded SPLADEIndex
        """
        load_path = Path(path)
        
        instance = cls(encoder=encoder)
        
        instance.sparse_matrix = sparse.load_npz(load_path / "matrix.npz")
        
        with open(load_path / "metadata.json") as f:
            metadata = json.load(f)
        
        instance.doc_ids = metadata["doc_ids"]
        instance.vocab_size = metadata["vocab_size"]
        
        if metadata.get("stats"):
            instance.stats = IndexStats(**metadata["stats"])
        
        with open(load_path / "contents.pkl", "rb") as f:
            instance.contents = pickle.load(f)
        
        print(f"[SPLADEIndex] Loaded from {load_path}")
        print(f"[SPLADEIndex] {len(instance.doc_ids)} chunks, {instance.sparse_matrix.nnz:,} terms")
        
        return instance


def build_and_save_index(
    db_path: Path = TEST_DB_PATH,
    output_path: Path = ARTIFACTS_DIR / "splade_index",
    batch_size: int = 32,
    limit: Optional[int] = None,
) -> IndexStats:
    encoder = SPLADEEncoder()
    index = SPLADEIndex(encoder=encoder)
    
    stats = index.build_from_db(db_path, batch_size=batch_size, limit=limit)
    
    index.save(str(output_path))
    
    stats_path = ARTIFACTS_DIR / "encoding_stats.json"
    with open(stats_path, "w") as f:
        json.dump(asdict(stats), f, indent=2)
    print(f"[SPLADEIndex] Stats saved to {stats_path}")
    
    return stats


def main():
    """Build SPLADE index."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Build SPLADE Index")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for encoding (default: 32)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/splade_index",
        help="Output directory for index",
    )
    parser.add_argument(
        "--test-query",
        type=str,
        default=None,
        help="Test query after building",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to encode (for testing)",
    )
    
    args = parser.parse_args()
    
    output_path = POC_DIR / args.output
    stats = build_and_save_index(
        output_path=output_path,
        batch_size=args.batch_size,
        limit=args.limit,
    )
    
    print("\n" + "="*60)
    print("Index Statistics")
    print("="*60)
    print(f"  Total chunks: {stats.total_chunks:,}")
    print(f"  Total terms: {stats.total_terms:,}")
    print(f"  Avg terms/doc: {stats.avg_terms_per_doc:.1f}")
    print(f"  Min terms/doc: {stats.min_terms_per_doc}")
    print(f"  Max terms/doc: {stats.max_terms_per_doc}")
    print(f"  Encoding time: {stats.encoding_time_s:.2f}s")
    print(f"  Index size: {stats.index_size_mb:.2f} MB")
    
    if args.test_query:
        print(f"\n[Test] Searching for: '{args.test_query}'")
        
        encoder = SPLADEEncoder()
        index = SPLADEIndex.load(str(output_path), encoder=encoder)
        
        start = time.time()
        results = index.search(args.test_query, k=5)
        latency = (time.time() - start) * 1000
        
        print(f"[Test] Latency: {latency:.2f}ms")
        print("[Test] Results:")
        for i, r in enumerate(results):
            print(f"  [{i+1}] score={r['score']:.4f} doc={r['doc_id']}")
            print(f"       {r['content'][:100]}...")


if __name__ == "__main__":
    main()
