#!/usr/bin/env python3
"""
Build the final test query set by merging existing and generated queries.

This script:
1. Loads consolidated existing queries
2. Loads generated queries
3. Merges them ensuring minimum counts are met
4. Outputs the final test_queries.json
"""

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Minimum required queries per type
MINIMUM_COUNTS = {
    "factoid": 50,
    "procedural": 50,
    "explanatory": 50,
    "comparison": 30,
    "troubleshooting": 30,
}


def main():
    datasets_dir = Path(__file__).parent.parent / "benchmarks" / "datasets"
    
    # Load existing queries
    existing_path = datasets_dir / "consolidated_existing.json"
    with open(existing_path) as f:
        existing_data = json.load(f)
    
    existing_queries = existing_data["queries"]
    print(f"Loaded {len(existing_queries)} existing queries")
    
    # Load generated queries
    generated_path = datasets_dir / "generated_queries.json"
    with open(generated_path) as f:
        generated_data = json.load(f)
    
    # Count existing by type
    existing_counts: dict[str, int] = defaultdict(int)
    for q in existing_queries:
        existing_counts[q.get("query_type", "unknown")] += 1
    
    print("\nExisting counts:")
    for qtype, count in existing_counts.items():
        print(f"  {qtype}: {count}")
    
    # Build final query list
    final_queries = []
    
    # Add all existing queries (standardize format)
    for q in existing_queries:
        final_queries.append({
            "id": q.get("id", f"existing_{len(final_queries)}"),
            "query": q.get("query", ""),
            "query_type": q.get("query_type", "unknown"),
            "expected_answer": q.get("expected_answer", ""),
            "optimal_granularity": q.get("optimal_granularity", "heading"),
            "expected_answer_keywords": q.get("expected_answer_keywords", []),
            "difficulty": q.get("difficulty", "medium"),
            "source": "existing",
            "source_file": q.get("source_file", ""),
            "doc_id": q.get("doc_id", ""),
            "relevant_chunk_ids": [],  # To be filled by labeling script
            "relevant_doc_ids": [q.get("doc_id", "")] if q.get("doc_id") else [],
        })
    
    # Add generated queries to meet minimums
    for qtype in MINIMUM_COUNTS.keys():
        current_count = existing_counts.get(qtype, 0)
        needed = MINIMUM_COUNTS[qtype] - current_count
        
        if needed <= 0:
            print(f"\n{qtype}: Already have {current_count}, need {MINIMUM_COUNTS[qtype]} - OK")
            continue
        
        generated_of_type = generated_data.get(qtype, [])
        to_add = generated_of_type[:needed + 5]  # Add a few extra
        
        print(f"\n{qtype}: Have {current_count}, adding {len(to_add)} generated")
        
        for q in to_add:
            final_queries.append({
                "id": q.get("id", f"gen_{len(final_queries)}"),
                "query": q.get("query", ""),
                "query_type": qtype,
                "expected_answer": q.get("expected_answer", ""),
                "optimal_granularity": q.get("optimal_granularity", "heading"),
                "expected_answer_keywords": q.get("expected_answer_keywords", []),
                "difficulty": q.get("difficulty", "medium"),
                "source": "generated",
                "topic": q.get("topic", ""),
                "relevant_chunk_ids": [],  # To be filled by labeling script
                "relevant_doc_ids": [],
            })
    
    # Count final
    final_counts: dict[str, int] = defaultdict(int)
    for q in final_queries:
        final_counts[q["query_type"]] += 1
    
    print("\n=== Final Counts ===")
    for qtype in MINIMUM_COUNTS.keys():
        count = final_counts.get(qtype, 0)
        minimum = MINIMUM_COUNTS[qtype]
        status = "OK" if count >= minimum else f"NEED {minimum - count} MORE"
        print(f"  {qtype:15}: {count:3} / {minimum:3} ({status})")
    
    total = len(final_queries)
    print(f"\nTotal queries: {total}")
    
    # Build output
    output = {
        "metadata": {
            "version": "1.0",
            "created_at": datetime.now().isoformat(),
            "total_queries": total,
            "query_counts": dict(final_counts),
            "corpus": "kubernetes-docs",
            "corpus_stats": {
                "total_chunks": 20801,
                "total_documents": 1569
            },
            "notes": "Queries need chunk_id labeling via search validation"
        },
        "queries": final_queries
    }
    
    # Save
    output_path = datasets_dir / "test_queries.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()
