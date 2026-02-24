#!/usr/bin/env python3
"""Rebuild BM25 index using raw content instead of enriched content."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from plm.search.components.bm25 import BM25Index
from plm.search.storage.sqlite import SQLiteStorage


def rebuild_bm25_raw(db_path: str, output_path: str) -> None:
    """Rebuild BM25 index using raw content only."""
    storage = SQLiteStorage(db_path)
    
    all_chunks = storage.get_all_chunks()
    if not all_chunks:
        print("No chunks found")
        return
    
    print(f"Found {len(all_chunks)} chunks")
    
    raw_contents = [chunk.get("content", "") for chunk in all_chunks]
    
    print(f"Building BM25 index on raw content...")
    bm25_index = BM25Index()
    bm25_index.index(raw_contents)
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    bm25_index.save(output_path)
    
    print(f"Saved BM25 index to {output_path}")
    print(f"Indexed {len(raw_contents)} documents")


if __name__ == "__main__":
    db_path = "poc/plm_vs_rag_benchmark/test_db/index.db"
    output_path = "poc/plm_vs_rag_benchmark/test_db"
    
    rebuild_bm25_raw(db_path, output_path)
