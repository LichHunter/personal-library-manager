#!/usr/bin/env python3
"""Quick test: How does Key Facts Coverage change with different top-k values?

This helps us understand if the bottleneck is:
1. Chunking (facts not in any chunk) - fixed_512 should have them all
2. Retrieval ranking (facts in chunks but not ranked high enough)
"""

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from strategies import Document, FixedSizeStrategy
from evaluation.metrics import calculate_key_facts_coverage


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


def main():
    print("=" * 70)
    print("TOP-K ANALYSIS: What k do we need for 95%+ Key Facts?")
    print("=" * 70)
    
    # Setup
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
    
    # Chunk with fixed_512 (best strategy)
    strategy = FixedSizeStrategy(chunk_size=512, overlap=0)
    chunks = strategy.chunk_many(documents)
    print(f"Chunked into {len(chunks)} chunks")
    
    # Embed
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    embeddings = embedder.encode(
        [c.content for c in chunks],
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    
    # Test different k values
    k_values = [1, 3, 5, 7, 10, 15, 20, len(chunks)]  # len(chunks) = all chunks
    
    print("\n" + "=" * 70)
    print(f"{'k':<5} {'Key Facts':<12} {'Facts Found':<15} {'Total Facts':<12}")
    print("-" * 70)
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    
    for k in k_values:
        all_found = 0
        
        for q in queries:
            query_emb = embedder.encode([q["query"]], normalize_embeddings=True)[0]
            similarities = np.dot(embeddings, query_emb)
            top_indices = np.argsort(similarities)[::-1][:k]
            
            # Get top-k chunks
            top_k_chunks = [chunks[idx] for idx in top_indices]
            retrieved_text = " ".join(c.content for c in top_k_chunks)
            
            # Count found facts
            key_facts = q.get("key_facts", [])
            for fact in key_facts:
                if fact.lower() in retrieved_text.lower():
                    all_found += 1
        
        coverage = all_found / total_facts if total_facts else 0
        k_label = f"{k}" if k != len(chunks) else f"{k} (ALL)"
        print(f"{k_label:<5} {coverage:>10.1%}   {all_found:<15} {total_facts:<12}")
    
    print("=" * 70)
    
    # Also check: what if we concatenate ALL documents? (theoretical max)
    print("\n" + "=" * 70)
    print("THEORETICAL MAXIMUM: All documents concatenated")
    print("-" * 70)
    
    all_content = " ".join(d.content for d in documents)
    theoretical_found = 0
    
    for q in queries:
        key_facts = q.get("key_facts", [])
        for fact in key_facts:
            if fact.lower() in all_content.lower():
                theoretical_found += 1
    
    print(f"Facts in corpus: {theoretical_found}/{total_facts} ({theoretical_found/total_facts:.1%})")
    print("=" * 70)


if __name__ == "__main__":
    main()
