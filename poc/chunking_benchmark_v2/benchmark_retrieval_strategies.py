#!/usr/bin/env python3
"""Benchmark retrieval strategies to achieve 95%+ key facts coverage.

Strategies tested:
1. Baseline: Semantic search with all-MiniLM-L6-v2
2. Upgraded embedder: BGE-base-en-v1.5
3. Hybrid: BM25 + Semantic with RRF fusion
4. Reranking: Cross-encoder on top-20 candidates
5. Combined: Hybrid + Reranking + BGE

Target: 98%+ key facts coverage
"""

import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi

from strategies import Document, FixedSizeStrategy


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
    """More lenient fact matching."""
    text_lower = text.lower()
    fact_lower = fact.lower()
    
    if fact_lower in text_lower:
        return True
    
    # Handle "version X" -> "PostgreSQL X"
    if fact_lower.startswith("version "):
        version_num = fact_lower.replace("version ", "")
        if version_num in text_lower:
            return True
    
    # Handle "X FK" -> "REFERENCES"
    if fact_lower.endswith(" fk"):
        table = fact_lower.replace(" fk", "").replace("_id", "")
        if f"references {table}" in text_lower or f"{table}_id" in text_lower:
            return True
    
    # Handle tier names
    if "tier" in fact_lower:
        tier_name = fact_lower.replace(" tier", "")
        if tier_name in text_lower:
            return True
    
    # Handle "X for Y"
    if " for " in fact_lower:
        parts = fact_lower.split(" for ")
        if len(parts) == 2 and all(p in text_lower for p in parts):
            return True
    
    return False


class RetrievalStrategy:
    """Base class for retrieval strategies."""
    
    def __init__(self, name: str):
        self.name = name
    
    def index(self, chunks: list, embedder=None):
        raise NotImplementedError
    
    def retrieve(self, query: str, k: int = 5) -> list:
        raise NotImplementedError


class SemanticRetrieval(RetrievalStrategy):
    """Pure semantic search with bi-encoder."""
    
    def __init__(self, embedder: SentenceTransformer, name: str = None):
        super().__init__(name or f"semantic_{embedder.get_sentence_embedding_dimension()}d")
        self.embedder = embedder
        self.chunks = None
        self.embeddings = None
        self.use_query_prefix = "bge" in str(type(embedder)).lower() or "bge" in getattr(embedder, '_model_card_vars', {}).get('model_name', '').lower()
    
    def index(self, chunks: list, embedder=None):
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
    
    def retrieve(self, query: str, k: int = 5) -> list:
        # BGE models need query prefix
        if self.use_query_prefix:
            query = f"Represent this sentence for searching relevant passages: {query}"
        
        query_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [self.chunks[i] for i in top_indices]


class HybridRetrieval(RetrievalStrategy):
    """BM25 + Semantic with Reciprocal Rank Fusion."""
    
    def __init__(self, embedder: SentenceTransformer, rrf_k: int = 60, name: str = None):
        super().__init__(name or "hybrid_bm25_semantic")
        self.embedder = embedder
        self.rrf_k = rrf_k
        self.chunks = None
        self.embeddings = None
        self.bm25 = None
        self.use_query_prefix = "bge" in str(type(embedder)).lower()
    
    def index(self, chunks: list, embedder=None):
        self.chunks = chunks
        
        # Semantic index
        texts = [c.content for c in chunks]
        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        
        # BM25 index
        tokenized = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, k: int = 5) -> list:
        # Semantic scores
        q_text = f"Represent this sentence for searching relevant passages: {query}" if self.use_query_prefix else query
        query_emb = self.embedder.encode([q_text], normalize_embeddings=True)[0]
        semantic_scores = np.dot(self.embeddings, query_emb)
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # RRF fusion
        semantic_ranks = np.argsort(semantic_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf_scores = {}
        for rank, idx in enumerate(semantic_ranks[:50]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        top_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:k]
        return [self.chunks[i] for i in top_indices]


class RerankingRetrieval(RetrievalStrategy):
    """Retrieve top-N with bi-encoder, rerank with cross-encoder."""
    
    def __init__(
        self,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
        initial_k: int = 20,
        name: str = None,
    ):
        super().__init__(name or f"rerank_top{initial_k}")
        self.embedder = embedder
        self.reranker = reranker
        self.initial_k = initial_k
        self.chunks = None
        self.embeddings = None
        self.use_query_prefix = "bge" in str(type(embedder)).lower()
    
    def index(self, chunks: list, embedder=None):
        self.chunks = chunks
        texts = [c.content for c in chunks]
        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
    
    def retrieve(self, query: str, k: int = 5) -> list:
        # Initial retrieval
        q_text = f"Represent this sentence for searching relevant passages: {query}" if self.use_query_prefix else query
        query_emb = self.embedder.encode([q_text], normalize_embeddings=True)[0]
        similarities = np.dot(self.embeddings, query_emb)
        candidate_indices = np.argsort(similarities)[::-1][:self.initial_k]
        
        # Rerank
        pairs = [[query, self.chunks[i].content] for i in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in reranked[:k]]
        return [self.chunks[i] for i in top_indices]


class HybridRerankRetrieval(RetrievalStrategy):
    """Hybrid (BM25 + Semantic) with cross-encoder reranking."""
    
    def __init__(
        self,
        embedder: SentenceTransformer,
        reranker: CrossEncoder,
        rrf_k: int = 60,
        initial_k: int = 20,
        name: str = None,
    ):
        super().__init__(name or "hybrid_rerank")
        self.embedder = embedder
        self.reranker = reranker
        self.rrf_k = rrf_k
        self.initial_k = initial_k
        self.chunks = None
        self.embeddings = None
        self.bm25 = None
        self.use_query_prefix = "bge" in str(type(embedder)).lower()
    
    def index(self, chunks: list, embedder=None):
        self.chunks = chunks
        texts = [c.content for c in chunks]
        
        self.embeddings = self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        
        tokenized = [c.content.lower().split() for c in chunks]
        self.bm25 = BM25Okapi(tokenized)
    
    def retrieve(self, query: str, k: int = 5) -> list:
        # Semantic scores
        q_text = f"Represent this sentence for searching relevant passages: {query}" if self.use_query_prefix else query
        query_emb = self.embedder.encode([q_text], normalize_embeddings=True)[0]
        semantic_scores = np.dot(self.embeddings, query_emb)
        
        # BM25 scores
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # RRF fusion for initial candidates
        semantic_ranks = np.argsort(semantic_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf_scores = {}
        for rank, idx in enumerate(semantic_ranks[:50]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        candidate_indices = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)[:self.initial_k]
        
        # Rerank
        pairs = [[query, self.chunks[i].content] for i in candidate_indices]
        rerank_scores = self.reranker.predict(pairs)
        
        reranked = sorted(zip(candidate_indices, rerank_scores), key=lambda x: x[1], reverse=True)
        top_indices = [idx for idx, _ in reranked[:k]]
        return [self.chunks[i] for i in top_indices]


def evaluate_strategy(
    strategy: RetrievalStrategy,
    queries: list[dict],
    k: int = 5,
    use_fuzzy: bool = False,
) -> dict:
    """Evaluate a retrieval strategy."""
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    found_facts = 0
    
    query_times = []
    
    for q in queries:
        start = time.perf_counter()
        retrieved = strategy.retrieve(q["query"], k=k)
        query_times.append(time.perf_counter() - start)
        
        retrieved_text = " ".join(c.content for c in retrieved)
        
        for fact in q.get("key_facts", []):
            if use_fuzzy:
                if fuzzy_fact_match(fact, retrieved_text):
                    found_facts += 1
            else:
                if fact.lower() in retrieved_text.lower():
                    found_facts += 1
    
    return {
        "strategy": strategy.name,
        "k": k,
        "key_facts_coverage": found_facts / total_facts if total_facts else 0,
        "found_facts": found_facts,
        "total_facts": total_facts,
        "avg_query_time_ms": np.mean(query_times) * 1000,
        "p95_query_time_ms": np.percentile(query_times, 95) * 1000,
    }


def log(msg: str):
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def main():
    log("=" * 80)
    log("RETRIEVAL STRATEGY BENCHMARK - Target: 98%+ Key Facts Coverage")
    log("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    # Load data
    v1_corpus = Path(__file__).parent.parent / "chunking_benchmark" / "corpus"
    v2_corpus = Path(__file__).parent / "corpus"
    
    documents = load_documents(v1_corpus)
    log(f"Loaded {len(documents)} documents")
    
    with open(v2_corpus / "ground_truth_v2.json") as f:
        queries = json.load(f)["queries"]
    log(f"Loaded {len(queries)} queries")
    
    # Chunk with best strategy
    strategy = FixedSizeStrategy(chunk_size=512, overlap=0)
    chunks = strategy.chunk_many(documents)
    log(f"Chunked into {len(chunks)} chunks (fixed_512_0pct)")
    
    # Load models
    log("\nLoading models...")
    
    log("  - all-MiniLM-L6-v2 (baseline)")
    minilm = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    
    log("  - BAAI/bge-base-en-v1.5 (upgraded)")
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    
    log("  - cross-encoder/ms-marco-MiniLM-L-6-v2 (reranker)")
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # Define strategies
    strategies = [
        # Baseline
        SemanticRetrieval(minilm, name="baseline_minilm"),
        
        # Upgraded embedder
        SemanticRetrieval(bge, name="bge_base"),
        
        # Hybrid
        HybridRetrieval(minilm, name="hybrid_minilm"),
        HybridRetrieval(bge, name="hybrid_bge"),
        
        # Reranking
        RerankingRetrieval(minilm, reranker, initial_k=20, name="rerank_minilm_top20"),
        RerankingRetrieval(bge, reranker, initial_k=20, name="rerank_bge_top20"),
        
        # Combined: Hybrid + Reranking
        HybridRerankRetrieval(minilm, reranker, initial_k=20, name="hybrid_rerank_minilm"),
        HybridRerankRetrieval(bge, reranker, initial_k=20, name="hybrid_rerank_bge"),
    ]
    
    # Index all strategies
    log("\nIndexing strategies...")
    for s in strategies:
        log(f"  - {s.name}")
        s.index(chunks)
    
    # Evaluate
    log("\n" + "=" * 80)
    log("RESULTS - EXACT STRING MATCHING")
    log("=" * 80)
    
    results_exact = []
    for s in strategies:
        result = evaluate_strategy(s, queries, k=5, use_fuzzy=False)
        results_exact.append(result)
        log(f"{s.name:<30} | Key Facts: {result['key_facts_coverage']:>6.1%} | "
            f"Avg: {result['avg_query_time_ms']:>6.1f}ms | P95: {result['p95_query_time_ms']:>6.1f}ms")
    
    log("\n" + "=" * 80)
    log("RESULTS - FUZZY MATCHING (realistic LLM behavior)")
    log("=" * 80)
    
    results_fuzzy = []
    for s in strategies:
        result = evaluate_strategy(s, queries, k=5, use_fuzzy=True)
        results_fuzzy.append(result)
        log(f"{s.name:<30} | Key Facts: {result['key_facts_coverage']:>6.1%} | "
            f"Avg: {result['avg_query_time_ms']:>6.1f}ms | P95: {result['p95_query_time_ms']:>6.1f}ms")
    
    # Summary
    log("\n" + "=" * 80)
    log("SUMMARY")
    log("=" * 80)
    
    best_exact = max(results_exact, key=lambda x: x["key_facts_coverage"])
    best_fuzzy = max(results_fuzzy, key=lambda x: x["key_facts_coverage"])
    
    log(f"\nBest (exact):  {best_exact['strategy']} = {best_exact['key_facts_coverage']:.1%}")
    log(f"Best (fuzzy):  {best_fuzzy['strategy']} = {best_fuzzy['key_facts_coverage']:.1%}")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "retrieval_benchmark.json", "w") as f:
        json.dump({
            "exact_matching": results_exact,
            "fuzzy_matching": results_fuzzy,
        }, f, indent=2)
    
    log(f"\nResults saved to {results_dir / 'retrieval_benchmark.json'}")


if __name__ == "__main__":
    main()
