#!/usr/bin/env python3
"""LangChain baseline RAG implementation for PLM comparison.

This baseline uses:
- FAISS for vector storage (in-memory)
- BGE-base-en-v1.5 for embeddings (same as PLM)
- Pre-extracted chunks from PLM's database (ensures identical chunk boundaries)
- Two modes: raw content vs enriched content

The baseline intentionally DOES NOT include:
- BM25 lexical search
- RRF fusion
- Query expansion
- Custom content enrichment (uses PLM's pre-computed enrichment when in enriched mode)

This allows fair comparison to isolate PLM's retrieval algorithm advantages.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
EMBEDDING_DIM = 768
CACHE_DIR = Path(__file__).parent / ".cache"


class BaselineLangChainRAG:
    """Baseline RAG using FAISS + sentence-transformers.
    
    Supports two modes:
    - use_enriched=False: Embeds raw chunk content
    - use_enriched=True: Embeds enriched content (keywords/entities prepended)
    
    Both modes use normalized embeddings for cosine similarity.
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """Initialize the baseline RAG.
        
        Args:
            model_name: Sentence transformer model name
        """
        self.model = SentenceTransformer(model_name)
        self.index: Optional[faiss.IndexFlatIP] = None
        self.chunks: list[dict] = []
        self.use_enriched: bool = False
    
    def _get_cache_path(self, chunks: list[dict], use_enriched: bool) -> Path:
        """Generate a deterministic cache path based on chunk content hash."""
        content_field = "enriched_content" if use_enriched else "content"
        content_hash = hashlib.md5(
            json.dumps([c.get(content_field, c["content"]) for c in chunks[:100]], sort_keys=True).encode()
        ).hexdigest()[:12]
        mode = "enriched" if use_enriched else "raw"
        return CACHE_DIR / f"faiss_index_{mode}_{len(chunks)}_{content_hash}.index"

    def ingest_chunks(self, chunks: list[dict], use_enriched: bool = False) -> None:
        """Ingest pre-extracted chunks into FAISS index with caching."""
        self.chunks = chunks
        self.use_enriched = use_enriched
        
        cache_path = self._get_cache_path(chunks, use_enriched)
        
        if cache_path.exists():
            print(f"Loading cached FAISS index from {cache_path}...")
            self.index = faiss.read_index(str(cache_path))
            print(f"Loaded {self.index.ntotal} chunks from cache")
            return
        
        content_field = "enriched_content" if use_enriched else "content"
        texts = [chunk.get(content_field) or chunk["content"] for chunk in chunks]
        
        print(f"Embedding {len(texts)} chunks ({'enriched' if use_enriched else 'raw'} mode)...")
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=64,
        )
        
        self.index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.index.add(embeddings.astype(np.float32))
        
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(cache_path))
        print(f"Indexed {self.index.ntotal} chunks (cached to {cache_path})")
    
    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        """Retrieve top-k chunks for a query.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of chunk dicts with added 'score' field, sorted by relevance
        """
        if self.index is None or self.index.ntotal == 0:
            raise ValueError("Index is empty. Call ingest_chunks() first.")
        
        # Embed query with normalization
        query_embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
        ).astype(np.float32)
        
        # Search FAISS index
        scores, indices = self.index.search(query_embedding, k)
        
        # Build results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.chunks):
                continue
            chunk = self.chunks[idx].copy()
            chunk["score"] = float(score)
            results.append(chunk)
        
        return results
    
    def get_stats(self) -> dict:
        """Get statistics about the indexed chunks."""
        return {
            "total_chunks": len(self.chunks),
            "indexed_chunks": self.index.ntotal if self.index else 0,
            "use_enriched": self.use_enriched,
            "embedding_model": EMBEDDING_MODEL,
            "embedding_dim": EMBEDDING_DIM,
        }


def load_chunks(chunks_path: str) -> list[dict]:
    """Load chunks from JSON file.
    
    Args:
        chunks_path: Path to chunks.json
        
    Returns:
        List of chunk dictionaries
    """
    with open(chunks_path) as f:
        return json.load(f)


def main():
    """Demo/test the baseline RAG."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test baseline LangChain RAG")
    parser.add_argument(
        "--chunks",
        type=str,
        default="chunks.json",
        help="Path to extracted chunks JSON",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="how to configure topology manager in kubernetes",
        help="Test query",
    )
    parser.add_argument(
        "--enriched",
        action="store_true",
        help="Use enriched content instead of raw",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=5,
        help="Number of results to retrieve",
    )
    
    args = parser.parse_args()
    
    # Load chunks
    chunks_path = Path(args.chunks)
    if not chunks_path.exists():
        print(f"ERROR: Chunks file not found: {chunks_path}")
        return 1
    
    chunks = load_chunks(args.chunks)
    print(f"Loaded {len(chunks)} chunks from {args.chunks}")
    
    # Initialize and ingest
    rag = BaselineLangChainRAG()
    rag.ingest_chunks(chunks, use_enriched=args.enriched)
    
    # Print stats
    stats = rag.get_stats()
    print(f"\nStats: {stats}")
    
    # Run query
    print(f"\nQuery: {args.query}")
    print(f"Mode: {'enriched' if args.enriched else 'raw'}")
    print(f"Top {args.k} results:")
    print("-" * 60)
    
    results = rag.retrieve(args.query, k=args.k)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Score: {result['score']:.4f}")
        print(f"   Doc ID: {result['doc_id']}")
        print(f"   Content: {result['content'][:150]}...")
    
    return 0


if __name__ == "__main__":
    exit(main())
