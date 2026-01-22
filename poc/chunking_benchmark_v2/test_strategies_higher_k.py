#!/usr/bin/env python3
"""Test top strategies with different k values to find path to 95%+.

Also tests semantic fact matching vs exact string matching.
"""

import json
from pathlib import Path
import sys
import re

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from strategies import Document, FixedSizeStrategy, RecursiveSplitterStrategy


def load_documents(corpus_dir: Path) -> list[Document]:
    metadata_path = corpus_dir / "corpus_metadata.json"
    docs_dir = corpus_dir / "documents"
    
    with open(metadata_path) as f:
        metadata = json.load(f)
    
    documents = []
    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            documents.append(Document(
                id=doc_meta["id"],
                title=doc_meta["title"],
                content=doc_path.read_text(),
                path=str(doc_path),
            ))
    return documents


def fuzzy_fact_match(fact: str, text: str) -> bool:
    """More lenient fact matching that handles common patterns."""
    text_lower = text.lower()
    fact_lower = fact.lower()
    
    # Exact match
    if fact_lower in text_lower:
        return True
    
    # Handle "version X" -> "PostgreSQL X" etc
    if fact_lower.startswith("version "):
        version_num = fact_lower.replace("version ", "")
        if version_num in text_lower:
            return True
    
    # Handle "X FK" -> "REFERENCES X" or "X_id"
    if fact_lower.endswith(" fk"):
        table = fact_lower.replace(" fk", "").replace("_id", "")
        if f"references {table}" in text_lower or f"{table}_id" in text_lower:
            return True
    
    # Handle "X tier" -> just "X" in rate limit context
    if "tier" in fact_lower:
        tier_name = fact_lower.replace(" tier", "")
        if tier_name in text_lower:
            return True
    
    # Handle "X for Pro" -> "Pro: X" or "Pro tier: X"
    if " for " in fact_lower:
        parts = fact_lower.split(" for ")
        if len(parts) == 2:
            value, tier = parts
            # Check if both parts exist nearby
            if value in text_lower and tier in text_lower:
                return True
    
    return False


def main():
    print("=" * 80)
    print("STRATEGY COMPARISON WITH VARIABLE K")
    print("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load data
    v1_corpus = Path(__file__).parent.parent / "chunking_benchmark" / "corpus"
    v2_corpus = Path(__file__).parent / "corpus"
    
    documents = load_documents(v1_corpus)
    print(f"Loaded {len(documents)} documents")
    
    with open(v2_corpus / "ground_truth_v2.json") as f:
        queries = json.load(f)["queries"]
    print(f"Loaded {len(queries)} queries")
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    print(f"Total key facts: {total_facts}")
    
    # Load embedder
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    # Strategies to test
    strategies = [
        ("fixed_512_0pct", FixedSizeStrategy(chunk_size=512, overlap=0)),
        ("fixed_400_0pct", FixedSizeStrategy(chunk_size=400, overlap=0)),
        ("recursive_2000", RecursiveSplitterStrategy(chunk_size=2000, chunk_overlap=0)),
        ("recursive_1600", RecursiveSplitterStrategy(chunk_size=1600, chunk_overlap=0)),
        ("fixed_300_0pct", FixedSizeStrategy(chunk_size=300, overlap=0)),
    ]
    
    k_values = [5, 7, 10, 15]
    
    print("\n" + "=" * 80)
    print("EXACT STRING MATCHING (current method)")
    print("=" * 80)
    print(f"\n{'Strategy':<20} | " + " | ".join(f"k={k:<3}" for k in k_values))
    print("-" * 80)
    
    for name, strategy in strategies:
        chunks = strategy.chunk_many(documents)
        embeddings = embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        results = []
        for k in k_values:
            found = 0
            for q in queries:
                query_emb = embedder.encode([q["query"]], normalize_embeddings=True)[0]
                similarities = np.dot(embeddings, query_emb)
                top_indices = np.argsort(similarities)[::-1][:k]
                
                retrieved_text = " ".join(chunks[idx].content for idx in top_indices)
                
                for fact in q.get("key_facts", []):
                    if fact.lower() in retrieved_text.lower():
                        found += 1
            
            results.append(found / total_facts)
        
        print(f"{name:<20} | " + " | ".join(f"{r:>5.1%}" for r in results))
    
    print("\n" + "=" * 80)
    print("FUZZY MATCHING (handles 'version 15' -> 'PostgreSQL 15' etc)")
    print("=" * 80)
    print(f"\n{'Strategy':<20} | " + " | ".join(f"k={k:<3}" for k in k_values))
    print("-" * 80)
    
    for name, strategy in strategies:
        chunks = strategy.chunk_many(documents)
        embeddings = embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        
        results = []
        for k in k_values:
            found = 0
            for q in queries:
                query_emb = embedder.encode([q["query"]], normalize_embeddings=True)[0]
                similarities = np.dot(embeddings, query_emb)
                top_indices = np.argsort(similarities)[::-1][:k]
                
                retrieved_text = " ".join(chunks[idx].content for idx in top_indices)
                
                for fact in q.get("key_facts", []):
                    if fuzzy_fact_match(fact, retrieved_text):
                        found += 1
            
            results.append(found / total_facts)
        
        print(f"{name:<20} | " + " | ".join(f"{r:>5.1%}" for r in results))
    
    # Check theoretical max with fuzzy matching
    print("\n" + "=" * 80)
    print("THEORETICAL MAXIMUM (all docs concatenated)")
    print("=" * 80)
    
    all_content = " ".join(d.content for d in documents)
    
    exact_found = sum(
        1 for q in queries for fact in q.get("key_facts", [])
        if fact.lower() in all_content.lower()
    )
    
    fuzzy_found = sum(
        1 for q in queries for fact in q.get("key_facts", [])
        if fuzzy_fact_match(fact, all_content)
    )
    
    print(f"Exact matching:  {exact_found}/{total_facts} ({exact_found/total_facts:.1%})")
    print(f"Fuzzy matching:  {fuzzy_found}/{total_facts} ({fuzzy_found/total_facts:.1%})")
    print("=" * 80)


if __name__ == "__main__":
    main()
