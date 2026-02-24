#!/usr/bin/env python3
"""Extract chunks from PLM's SQLite database for baseline comparison.

This script extracts both raw content and enriched content from PLM's
indexed chunks, ensuring all comparison variants use identical chunk boundaries.

Usage:
    python chunk_extractor.py --db-path /path/to/index.db --output chunks.json
"""

import argparse
import json
import sqlite3
from pathlib import Path


def extract_chunks(db_path: str) -> list[dict]:
    """Extract all chunks from PLM's SQLite database.
    
    Args:
        db_path: Path to PLM's SQLite database
        
    Returns:
        List of chunk dictionaries with content, enriched_content, and metadata
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get all chunks with their content
    cursor.execute("""
        SELECT 
            id,
            doc_id,
            content,
            enriched_content,
            heading,
            start_char,
            end_char,
            chunk_index
        FROM chunks
        ORDER BY doc_id, chunk_index
    """)
    
    chunks = []
    for row in cursor.fetchall():
        chunk = {
            "chunk_id": row["id"],
            "doc_id": row["doc_id"],
            "content": row["content"],
            "enriched_content": row["enriched_content"] or row["content"],  # Fallback if no enrichment
            "heading": row["heading"],
            "start_char": row["start_char"],
            "end_char": row["end_char"],
            "chunk_index": row["chunk_index"],
        }
        chunks.append(chunk)
    
    conn.close()
    return chunks


def get_database_stats(db_path: str) -> dict:
    """Get statistics about the PLM database.
    
    Args:
        db_path: Path to PLM's SQLite database
        
    Returns:
        Dictionary with document and chunk counts
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM documents")
    doc_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chunks")
    chunk_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM chunks WHERE enriched_content IS NOT NULL AND enriched_content != content")
    enriched_count = cursor.fetchone()[0]
    
    conn.close()
    
    return {
        "documents": doc_count,
        "chunks": chunk_count,
        "enriched_chunks": enriched_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract chunks from PLM's SQLite database for baseline comparison"
    )
    parser.add_argument(
        "--db-path",
        type=str,
        default="/home/susano/.local/share/docker/volumes/docker_plm-search-index/_data/index.db",
        help="Path to PLM's SQLite database",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="chunks.json",
        help="Output JSON file for extracted chunks",
    )
    parser.add_argument(
        "--stats-only",
        action="store_true",
        help="Only print database statistics, don't extract chunks",
    )
    
    args = parser.parse_args()
    
    # Verify database exists
    db_path = Path(args.db_path)
    if not db_path.exists():
        print(f"ERROR: Database not found at {db_path}")
        return 1
    
    # Get stats
    stats = get_database_stats(args.db_path)
    print(f"Database: {args.db_path}")
    print(f"  Documents: {stats['documents']}")
    print(f"  Chunks: {stats['chunks']}")
    print(f"  Enriched chunks: {stats['enriched_chunks']}")
    
    if args.stats_only:
        return 0
    
    # Extract chunks
    print(f"\nExtracting chunks...")
    chunks = extract_chunks(args.db_path)
    
    # Save to JSON
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump(chunks, f, indent=2)
    
    print(f"Extracted {len(chunks)} chunks to {output_path}")
    
    # Print sample
    if chunks:
        sample = chunks[0]
        print(f"\nSample chunk:")
        print(f"  doc_id: {sample['doc_id']}")
        print(f"  content length: {len(sample['content'])} chars")
        print(f"  enriched_content length: {len(sample['enriched_content'])} chars")
        print(f"  content preview: {sample['content'][:100]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
