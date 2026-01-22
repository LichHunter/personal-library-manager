#!/usr/bin/env python3
"""Find which key facts are missing from the corpus."""

import json
from pathlib import Path

def main():
    # Load documents
    v1_corpus = Path(__file__).parent.parent / "chunking_benchmark" / "corpus"
    v2_corpus = Path(__file__).parent / "corpus"
    
    metadata_path = v1_corpus / "corpus_metadata.json"
    docs_dir = v1_corpus / "documents"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    # Concatenate all document content
    all_content = ""
    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            all_content += " " + doc_path.read_text()
    
    all_content_lower = all_content.lower()
    
    # Load queries
    with open(v2_corpus / "ground_truth_v2.json") as f:
        queries = json.load(f)["queries"]
    
    print("=" * 80)
    print("MISSING KEY FACTS ANALYSIS")
    print("=" * 80)
    
    missing_facts = []
    
    for q in queries:
        query_id = q["id"]
        key_facts = q.get("key_facts", [])
        
        for fact in key_facts:
            if fact.lower() not in all_content_lower:
                missing_facts.append({
                    "query_id": query_id,
                    "query": q["query"],
                    "missing_fact": fact,
                    "expected_docs": q.get("expected_docs", []),
                })
    
    print(f"\nFound {len(missing_facts)} missing facts:\n")
    
    for mf in missing_facts:
        print(f"Query: {mf['query_id']}")
        print(f"  Question: {mf['query']}")
        print(f"  Missing fact: '{mf['missing_fact']}'")
        print(f"  Expected docs: {mf['expected_docs']}")
        print()
    
    # Summary
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    found_facts = total_facts - len(missing_facts)
    
    print("=" * 80)
    print(f"SUMMARY: {found_facts}/{total_facts} facts exist in corpus ({found_facts/total_facts:.1%})")
    print(f"Missing {len(missing_facts)} facts - these need to be fixed in ground_truth_v2.json")
    print("=" * 80)


if __name__ == "__main__":
    main()
