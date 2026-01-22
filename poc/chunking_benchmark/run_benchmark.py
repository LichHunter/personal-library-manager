#!/usr/bin/env python3
"""Lightweight benchmark runner with verbose logging and GPU support."""

import gc
import json
import time
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from strategies import (
    Document,
    FixedSizeStrategy,
    HeadingBasedStrategy,
    HeadingLimitedStrategy,
    HierarchicalStrategy,
    ParagraphStrategy,
    HeadingParagraphStrategy,
)


def log(msg):
    """Print with timestamp."""
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def load_documents(corpus_dir: Path) -> list[Document]:
    """Load documents from corpus directory."""
    log("Loading corpus metadata...")
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


def load_queries(corpus_dir: Path) -> list[dict]:
    """Load ground truth queries."""
    with open(corpus_dir / "ground_truth.json") as f:
        return json.load(f).get("queries", [])


def run_single_strategy(strategy, documents, queries, embedder):
    """Benchmark a single strategy."""
    import numpy as np
    
    log(f"  Chunking documents...")
    start = time.perf_counter()
    chunks = strategy.chunk_many(documents)
    chunk_time = (time.perf_counter() - start) * 1000
    log(f"  Chunked into {len(chunks)} chunks in {chunk_time:.0f}ms")
    
    token_counts = [c.token_count for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    log(f"  Token stats: avg={avg_tokens:.0f}, min={min(token_counts)}, max={max(token_counts)}")
    
    log(f"  Embedding {len(chunks)} chunks...")
    start = time.perf_counter()
    texts = [c.content for c in chunks]
    embeddings = embedder.encode(
        texts, 
        show_progress_bar=True,
        normalize_embeddings=True,
        batch_size=32,
    )
    embed_time = (time.perf_counter() - start) * 1000
    log(f"  Embedded in {embed_time:.0f}ms")
    
    log(f"  Running {len(queries)} retrieval queries...")
    recall_5_scores = []
    mrr_scores = []
    
    for i, q in enumerate(queries):
        query_emb = embedder.encode([q["query"]], normalize_embeddings=True)[0]
        similarities = np.dot(embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:10]
        
        retrieved_docs = [chunks[idx].doc_id for idx in top_indices]
        expected = set(q.get("expected_docs", []))
        
        hits = len(expected & set(retrieved_docs[:5]))
        recall_5_scores.append(hits / len(expected) if expected else 0)
        
        mrr = 0
        for rank, doc_id in enumerate(retrieved_docs):
            if doc_id in expected:
                mrr = 1.0 / (rank + 1)
                break
        mrr_scores.append(mrr)
        
        if (i + 1) % 10 == 0:
            log(f"    Processed {i+1}/{len(queries)} queries")
    
    recall_5 = sum(recall_5_scores) / len(recall_5_scores)
    mrr = sum(mrr_scores) / len(mrr_scores)
    log(f"  Results: Recall@5={recall_5:.1%}, MRR={mrr:.3f}")
    
    return {
        "strategy": strategy.name,
        "num_chunks": len(chunks),
        "avg_tokens": avg_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "chunk_time_ms": chunk_time,
        "embed_time_ms": embed_time,
        "recall_5": recall_5,
        "mrr": mrr,
    }


def main():
    import torch
    
    log("="*60)
    log("CHUNKING BENCHMARK")
    log("="*60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"PyTorch device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
        log(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    corpus_dir = Path(__file__).parent / "corpus"
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    documents = load_documents(corpus_dir)
    log(f"Loaded {len(documents)} documents")
    
    queries = load_queries(corpus_dir)
    log(f"Loaded {len(queries)} queries")
    
    log("Loading embedding model (all-MiniLM-L6-v2)...")
    from sentence_transformers import SentenceTransformer
    embedder = SentenceTransformer('all-MiniLM-L6-v2', device=device)
    log(f"Model loaded on {device}")
    
    strategies = [
        FixedSizeStrategy(chunk_size=512, overlap=50),
        HeadingBasedStrategy(),
        HeadingLimitedStrategy(max_tokens=512),
        HierarchicalStrategy(),
        ParagraphStrategy(min_tokens=50, max_tokens=256),
        HeadingParagraphStrategy(),
    ]
    
    all_results = []
    
    for idx, strategy in enumerate(strategies, 1):
        log("")
        log(f"[{idx}/{len(strategies)}] Testing: {strategy.name}")
        log("-"*50)
        
        result = run_single_strategy(strategy, documents, queries, embedder)
        all_results.append(result)
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    with open(results_dir / "benchmark_summary.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    log("")
    log("="*60)
    log("RESULTS SUMMARY")
    log("="*60)
    log(f"\n{'Strategy':<25} {'Chunks':>7} {'AvgTok':>7} {'Recall@5':>9} {'MRR':>7}")
    log("-" * 60)
    
    sorted_results = sorted(all_results, key=lambda x: x['recall_5'], reverse=True)
    for r in sorted_results:
        log(f"{r['strategy']:<25} {r['num_chunks']:>7} {r['avg_tokens']:>7.0f} {r['recall_5']:>8.1%} {r['mrr']:>7.3f}")
    
    report_lines = [
        "# Chunking Benchmark Results",
        "",
        f"**Corpus:** {len(documents)} documents, {len(queries)} test queries",
        f"**Device:** {device}",
        "",
        "## Summary",
        "",
        "| Strategy | Chunks | Avg Tokens | Recall@5 | MRR | Index Time |",
        "|----------|--------|------------|----------|-----|------------|",
    ]
    
    for r in sorted_results:
        total_time = r['chunk_time_ms'] + r['embed_time_ms']
        report_lines.append(
            f"| {r['strategy']} | {r['num_chunks']} | {r['avg_tokens']:.0f} | "
            f"{r['recall_5']:.1%} | {r['mrr']:.3f} | {total_time:.0f}ms |"
        )
    
    best = sorted_results[0]
    report_lines.extend([
        "",
        "## Recommendation",
        "",
        f"**Best strategy: {best['strategy']}**",
        "",
        f"- Recall@5: {best['recall_5']:.1%}",
        f"- MRR: {best['mrr']:.3f}",
        f"- {best['num_chunks']} chunks averaging {best['avg_tokens']:.0f} tokens",
        "",
        "## Per-Strategy Details",
        "",
    ])
    
    for r in sorted_results:
        report_lines.extend([
            f"### {r['strategy']}",
            f"- Chunks: {r['num_chunks']} (range: {r['min_tokens']}-{r['max_tokens']} tokens)",
            f"- Recall@5: {r['recall_5']:.1%}, MRR: {r['mrr']:.3f}",
            f"- Timing: {r['chunk_time_ms']:.0f}ms chunking, {r['embed_time_ms']:.0f}ms embedding",
            "",
        ])
    
    with open(results_dir / "report.md", "w") as f:
        f.write("\n".join(report_lines))
    
    log(f"\nReport saved to: {results_dir / 'report.md'}")
    log("Done!")


if __name__ == "__main__":
    main()
