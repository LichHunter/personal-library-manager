#!/usr/bin/env python3
"""Generate ground_truth_v2.json with relevant_spans for token-level evaluation.

This script reads the V1 ground_truth.json and enhances it by finding
the actual character offsets of key facts in the source documents.
"""

import json
import re
from pathlib import Path


def find_spans_for_query(query: dict, documents: dict[str, str]) -> list[dict]:
    """Find character spans containing key facts for a query.
    
    Args:
        query: Query dict with expected_docs and key_facts
        documents: Dict of doc_id -> content
        
    Returns:
        List of {"doc_id": str, "start": int, "end": int, "text": str}
    """
    spans = []
    key_facts = query.get("key_facts", [])
    expected_docs = query.get("expected_docs", [])
    
    for doc_id in expected_docs:
        if doc_id not in documents:
            continue
        
        content = documents[doc_id]
        
        # For each key fact, find its location in the document
        for fact in key_facts:
            # Search for the fact (case-insensitive)
            pattern = re.escape(fact)
            
            for match in re.finditer(pattern, content, re.IGNORECASE):
                # Expand to sentence/paragraph boundary for context
                start = match.start()
                end = match.end()
                
                # Expand to include surrounding sentence
                # Look backwards for sentence start
                sentence_start = start
                for i in range(start - 1, max(0, start - 500), -1):
                    if content[i] in '.!?\n' and i < start - 1:
                        sentence_start = i + 1
                        break
                    if i == max(0, start - 500):
                        sentence_start = i
                
                # Look forwards for sentence end
                sentence_end = end
                for i in range(end, min(len(content), end + 500)):
                    if content[i] in '.!?\n':
                        sentence_end = i + 1
                        break
                    if i == min(len(content), end + 500) - 1:
                        sentence_end = i + 1
                
                # Strip whitespace from boundaries
                while sentence_start < sentence_end and content[sentence_start].isspace():
                    sentence_start += 1
                while sentence_end > sentence_start and content[sentence_end - 1].isspace():
                    sentence_end -= 1
                
                spans.append({
                    "doc_id": doc_id,
                    "start": sentence_start,
                    "end": sentence_end,
                    "text": content[sentence_start:sentence_end][:100] + "..." if len(content[sentence_start:sentence_end]) > 100 else content[sentence_start:sentence_end],
                    "matched_fact": fact,
                })
    
    # Deduplicate overlapping spans within same document
    spans = _merge_overlapping_spans(spans)
    
    return spans


def _merge_overlapping_spans(spans: list[dict]) -> list[dict]:
    """Merge overlapping spans within the same document."""
    if not spans:
        return []
    
    # Group by doc_id
    by_doc: dict[str, list[dict]] = {}
    for span in spans:
        doc_id = span["doc_id"]
        if doc_id not in by_doc:
            by_doc[doc_id] = []
        by_doc[doc_id].append(span)
    
    merged = []
    for doc_id, doc_spans in by_doc.items():
        # Sort by start position
        doc_spans.sort(key=lambda s: s["start"])
        
        current = doc_spans[0].copy()
        
        for span in doc_spans[1:]:
            # If overlapping or adjacent, merge
            if span["start"] <= current["end"] + 10:  # 10 char buffer
                current["end"] = max(current["end"], span["end"])
                # Update text preview
                current["text"] = f"[merged span covering {current['end'] - current['start']} chars]"
            else:
                merged.append(current)
                current = span.copy()
        
        merged.append(current)
    
    return merged


def main():
    # Paths
    v1_corpus_dir = Path(__file__).parent.parent.parent / "chunking_benchmark" / "corpus"
    v2_corpus_dir = Path(__file__).parent
    
    # Load V1 ground truth
    with open(v1_corpus_dir / "ground_truth.json") as f:
        v1_data = json.load(f)
    
    # Load all documents
    documents = {}
    docs_dir = v1_corpus_dir / "documents"
    
    # Load corpus metadata
    with open(v1_corpus_dir / "corpus_metadata.json") as f:
        metadata = json.load(f)
    
    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            documents[doc_meta["id"]] = doc_path.read_text()
    
    print(f"Loaded {len(documents)} documents")
    
    # Enhance queries with relevant spans
    v2_queries = []
    
    for query in v1_data["queries"]:
        spans = find_spans_for_query(query, documents)
        
        enhanced_query = query.copy()
        enhanced_query["relevant_spans"] = spans
        v2_queries.append(enhanced_query)
        
        print(f"Query '{query['id']}': found {len(spans)} spans")
    
    # Create V2 ground truth
    v2_data = {
        "description": "Ground truth queries with character-level spans for token-level evaluation",
        "version": "2.0",
        "queries": v2_queries,
    }
    
    # Save
    output_path = v2_corpus_dir / "ground_truth_v2.json"
    with open(output_path, "w") as f:
        json.dump(v2_data, f, indent=2)
    
    print(f"\nSaved to {output_path}")
    
    # Print stats
    total_spans = sum(len(q["relevant_spans"]) for q in v2_queries)
    queries_with_spans = sum(1 for q in v2_queries if q["relevant_spans"])
    print(f"Total spans: {total_spans}")
    print(f"Queries with spans: {queries_with_spans}/{len(v2_queries)}")


if __name__ == "__main__":
    main()
