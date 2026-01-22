#!/usr/bin/env python3
"""Benchmark retrieval strategies on expanded corpus (52 docs, 53 queries, 180 facts)."""

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


def load_expanded_corpus():
    """Load expanded corpus documents."""
    corpus_dir = Path(__file__).parent / "corpus"
    metadata_path = corpus_dir / "corpus_metadata_expanded.json"
    docs_dir = corpus_dir / "expanded_documents"
    
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


def load_expanded_queries():
    """Load expanded ground truth queries."""
    gt_path = Path(__file__).parent / "corpus" / "ground_truth_expanded.json"
    with open(gt_path) as f:
        return json.load(f)["queries"]


def exact_match(fact: str, text: str) -> bool:
    return fact.lower() in text.lower()


def fuzzy_match(fact: str, text: str) -> bool:
    """More lenient matching."""
    text_lower = text.lower()
    fact_lower = fact.lower()
    
    if fact_lower in text_lower:
        return True
    
    # Handle common variations
    words = fact_lower.split()
    if len(words) >= 2 and all(w in text_lower for w in words):
        return True
    
    return False


class SemanticRetrieval:
    def __init__(self, embedder, name, use_prefix=False):
        self.embedder = embedder
        self.name = name
        self.use_prefix = use_prefix
        self.chunks = None
        self.embeddings = None
    
    def index(self, chunks):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
    
    def retrieve(self, query, k=5):
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][:k]
        return [self.chunks[i] for i in top_idx]


class HybridRetrieval:
    def __init__(self, embedder, name, use_prefix=False, rrf_k=60):
        self.embedder = embedder
        self.name = name
        self.use_prefix = use_prefix
        self.rrf_k = rrf_k
        self.chunks = None
        self.embeddings = None
        self.bm25 = None
    
    def index(self, chunks):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self.bm25 = BM25Okapi([c.content.lower().split() for c in chunks])
    
    def retrieve(self, query, k=5):
        # Semantic
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sem_scores = np.dot(self.embeddings, q_emb)
        
        # BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # RRF
        sem_ranks = np.argsort(sem_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf = {}
        for rank, idx in enumerate(sem_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        top_idx = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:k]
        return [self.chunks[i] for i in top_idx]


class HybridRerankRetrieval:
    def __init__(self, embedder, reranker, name, use_prefix=False, rrf_k=60, initial_k=20):
        self.embedder = embedder
        self.reranker = reranker
        self.name = name
        self.use_prefix = use_prefix
        self.rrf_k = rrf_k
        self.initial_k = initial_k
        self.chunks = None
        self.embeddings = None
        self.bm25 = None
    
    def index(self, chunks):
        self.chunks = chunks
        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=True,
            batch_size=32,
        )
        self.bm25 = BM25Okapi([c.content.lower().split() for c in chunks])
    
    def retrieve(self, query, k=5):
        # Semantic
        q = f"Represent this sentence for searching relevant passages: {query}" if self.use_prefix else query
        q_emb = self.embedder.encode([q], normalize_embeddings=True)[0]
        sem_scores = np.dot(self.embeddings, q_emb)
        
        # BM25
        bm25_scores = self.bm25.get_scores(query.lower().split())
        
        # RRF
        sem_ranks = np.argsort(sem_scores)[::-1]
        bm25_ranks = np.argsort(bm25_scores)[::-1]
        
        rrf = {}
        for rank, idx in enumerate(sem_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        for rank, idx in enumerate(bm25_ranks[:50]):
            rrf[idx] = rrf.get(idx, 0) + 1 / (self.rrf_k + rank)
        
        candidates = sorted(rrf.keys(), key=lambda x: rrf[x], reverse=True)[:self.initial_k]
        
        # Rerank
        pairs = [[query, self.chunks[i].content] for i in candidates]
        scores = self.reranker.predict(pairs)
        reranked = sorted(zip(candidates, scores), key=lambda x: x[1], reverse=True)
        top_idx = [idx for idx, _ in reranked[:k]]
        return [self.chunks[i] for i in top_idx]


def evaluate(strategy, queries, k=5, match_fn=exact_match):
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    found = 0
    times = []
    
    for q in queries:
        start = time.perf_counter()
        retrieved = strategy.retrieve(q["query"], k=k)
        times.append(time.perf_counter() - start)
        
        text = " ".join(c.content for c in retrieved)
        for fact in q.get("key_facts", []):
            if match_fn(fact, text):
                found += 1
    
    return {
        "coverage": found / total_facts if total_facts else 0,
        "found": found,
        "total": total_facts,
        "avg_ms": np.mean(times) * 1000,
        "p95_ms": np.percentile(times, 95) * 1000,
    }


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    log("=" * 80)
    log("EXPANDED CORPUS BENCHMARK - 52 docs, 53 queries, 180 key facts")
    log("=" * 80)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Device: {device}")
    
    # Load data
    documents = load_expanded_corpus()
    log(f"Loaded {len(documents)} documents")
    
    queries = load_expanded_queries()
    log(f"Loaded {len(queries)} queries")
    
    total_facts = sum(len(q.get("key_facts", [])) for q in queries)
    log(f"Total key facts: {total_facts}")
    
    # Chunk
    chunker = FixedSizeStrategy(chunk_size=512, overlap=0)
    chunks = chunker.chunk_many(documents)
    log(f"Created {len(chunks)} chunks")
    
    # Load models
    log("\nLoading models...")
    minilm = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    bge = SentenceTransformer("BAAI/bge-base-en-v1.5", device=device)
    reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
    
    # Strategies
    strategies = [
        SemanticRetrieval(minilm, "minilm_semantic"),
        SemanticRetrieval(bge, "bge_semantic", use_prefix=True),
        HybridRetrieval(minilm, "minilm_hybrid"),
        HybridRetrieval(bge, "bge_hybrid", use_prefix=True),
        HybridRerankRetrieval(bge, reranker, "bge_hybrid_rerank", use_prefix=True),
    ]
    
    # Index
    log("\nIndexing...")
    for s in strategies:
        log(f"  {s.name}")
        s.index(chunks)
    
    # Evaluate with different k values
    k_values = [5, 10]
    
    results = {"exact": {}, "fuzzy": {}}
    
    for k in k_values:
        log(f"\n{'='*80}")
        log(f"RESULTS k={k}")
        log(f"{'='*80}")
        
        log(f"\n{'Strategy':<25} | {'Exact':>8} | {'Fuzzy':>8} | {'Avg ms':>8}")
        log("-" * 60)
        
        for s in strategies:
            exact_res = evaluate(s, queries, k=k, match_fn=exact_match)
            fuzzy_res = evaluate(s, queries, k=k, match_fn=fuzzy_match)
            
            results["exact"][f"{s.name}_k{k}"] = exact_res
            results["fuzzy"][f"{s.name}_k{k}"] = fuzzy_res
            
            log(f"{s.name:<25} | {exact_res['coverage']:>7.1%} | {fuzzy_res['coverage']:>7.1%} | {exact_res['avg_ms']:>7.1f}")
    
    # Summary
    log(f"\n{'='*80}")
    log("SUMMARY")
    log(f"{'='*80}")
    
    best_exact_k5 = max((v for k, v in results["exact"].items() if "k5" in k), key=lambda x: x["coverage"])
    best_fuzzy_k5 = max((v for k, v in results["fuzzy"].items() if "k5" in k), key=lambda x: x["coverage"])
    best_exact_k10 = max((v for k, v in results["exact"].items() if "k10" in k), key=lambda x: x["coverage"])
    best_fuzzy_k10 = max((v for k, v in results["fuzzy"].items() if "k10" in k), key=lambda x: x["coverage"])
    
    log(f"\nBest k=5 exact:  {best_exact_k5['coverage']:.1%} ({best_exact_k5['found']}/{best_exact_k5['total']})")
    log(f"Best k=5 fuzzy:  {best_fuzzy_k5['coverage']:.1%} ({best_fuzzy_k5['found']}/{best_fuzzy_k5['total']})")
    log(f"Best k=10 exact: {best_exact_k10['coverage']:.1%} ({best_exact_k10['found']}/{best_exact_k10['total']})")
    log(f"Best k=10 fuzzy: {best_fuzzy_k10['coverage']:.1%} ({best_fuzzy_k10['found']}/{best_fuzzy_k10['total']})")
    
    # Save
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    with open(results_dir / "2025-01-22_expanded_corpus_benchmark.json", "w") as f:
        json.dump(results, f, indent=2)
    
    log(f"\nSaved to {results_dir / '2025-01-22_expanded_corpus_benchmark.json'}")


if __name__ == "__main__":
    main()
