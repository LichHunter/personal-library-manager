#!/usr/bin/env python3
"""
Label test queries with relevant chunk_ids by searching PLM.

This script:
1. Loads the test query set
2. Searches PLM for each query at chunk, heading, and document levels
3. Stores top results as relevant_*_ids
4. Saves the labeled query set

Requires: PLM search service running (plm-search MCP)
"""

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def search_plm(query: str, level: str, k: int = 5) -> list[dict[str, Any]]:
    """Call PLM search MCP via subprocess (or HTTP if available)."""
    # For now, we'll generate placeholder results
    # In production, this would call the actual PLM search API
    # The real implementation would use the MCP tools
    return []


def main():
    datasets_dir = Path(__file__).parent.parent / "benchmarks" / "datasets"
    
    # Load test queries
    test_path = datasets_dir / "test_queries.json"
    with open(test_path) as f:
        data = json.load(f)
    
    queries = data["queries"]
    total = len(queries)
    print(f"Loaded {total} queries for labeling")
    
    # For now, we'll create a partial labeling based on existing data
    # Full labeling requires running searches against PLM
    
    labeled_count = 0
    for i, q in enumerate(queries):
        # If query already has doc_id from existing data, create doc reference
        if q.get("doc_id"):
            q["relevant_doc_ids"] = [q["doc_id"]]
            labeled_count += 1
        
        # Mark as needing chunk labeling
        if not q.get("relevant_chunk_ids"):
            q["relevant_chunk_ids"] = []  # To be filled by search
    
    print(f"Queries with doc_id labels: {labeled_count}")
    print(f"Queries needing chunk labeling: {total - labeled_count}")
    
    # Save updated queries
    with open(test_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"\nSaved labeled queries to: {test_path}")
    print("\nNote: Full chunk labeling requires running PLM searches.")
    print("Run the oracle_performance.py script to find relevant chunks via search.")


if __name__ == "__main__":
    main()
