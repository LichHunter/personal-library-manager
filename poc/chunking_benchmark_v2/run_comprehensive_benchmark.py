#!/usr/bin/env python3
"""Comprehensive Chunking Benchmark - All V1 and V2 strategies with FIXED metrics.

Tests ALL chunking strategies:
- V1: fixed_size, heading_based, heading_limited, hierarchical, paragraphs, heading_paragraph
- V2: recursive_splitter, cluster_semantic, paragraph_heading, fixed_size (with overlap variants)

Uses corrected evaluation:
- Token metrics per-document (not mixed across docs)
- Key Facts Coverage (primary metric for RAG quality)
- Penalizes missing expected documents
"""

import gc
import json
import time
from pathlib import Path
import sys
import importlib.util

# Setup paths
v2_dir = Path(__file__).parent
v1_dir = v2_dir.parent / "chunking_benchmark"

import numpy as np
import torch
from sentence_transformers import SentenceTransformer


def load_module_with_base(module_name: str, file_path: Path, base_module):
    """Load a module, injecting base classes to handle relative imports."""
    # Read the source and replace relative imports
    source = file_path.read_text()
    
    # Replace relative imports with our injected base
    source = source.replace("from .base import", "# from .base import")
    
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Inject base classes into module namespace before execution
    module.ChunkingStrategy = base_module.ChunkingStrategy
    module.Chunk = base_module.Chunk
    module.Document = base_module.Document
    
    # Compile and execute the modified source
    code = compile(source, str(file_path), 'exec')
    exec(code, module.__dict__)
    
    sys.modules[module_name] = module
    return module


# Load V2 base first (since it has the most complete Chunk class with get_char_set)
v2_base_spec = importlib.util.spec_from_file_location("v2_base", v2_dir / "strategies" / "base.py")
v2_base = importlib.util.module_from_spec(v2_base_spec)
v2_base_spec.loader.exec_module(v2_base)
sys.modules["v2_base"] = v2_base

# Load V2 strategies with base injection
v2_recursive = load_module_with_base("v2_recursive", v2_dir / "strategies" / "recursive_splitter.py", v2_base)
v2_semantic = load_module_with_base("v2_semantic", v2_dir / "strategies" / "cluster_semantic.py", v2_base)
v2_paragraph = load_module_with_base("v2_paragraph", v2_dir / "strategies" / "paragraph_heading.py", v2_base)
v2_fixed = load_module_with_base("v2_fixed", v2_dir / "strategies" / "fixed_size.py", v2_base)

# Load V2 metrics
v2_metrics_spec = importlib.util.spec_from_file_location("v2_metrics", v2_dir / "evaluation" / "metrics.py")
v2_metrics = importlib.util.module_from_spec(v2_metrics_spec)
v2_metrics_spec.loader.exec_module(v2_metrics)

# Load V1 base
v1_base_spec = importlib.util.spec_from_file_location("v1_base", v1_dir / "strategies" / "base.py")
v1_base = importlib.util.module_from_spec(v1_base_spec)
v1_base_spec.loader.exec_module(v1_base)
sys.modules["v1_base"] = v1_base

# Load V1 strategies with V1 base (but we'll convert Documents when calling)
v1_fixed_mod = load_module_with_base("v1_fixed", v1_dir / "strategies" / "fixed_size.py", v1_base)
v1_heading_mod = load_module_with_base("v1_heading", v1_dir / "strategies" / "heading_based.py", v1_base)
v1_heading_limited_mod = load_module_with_base("v1_heading_limited", v1_dir / "strategies" / "heading_limited.py", v1_base)
v1_hierarchical_mod = load_module_with_base("v1_hierarchical", v1_dir / "strategies" / "hierarchical.py", v1_base)
v1_paragraphs_mod = load_module_with_base("v1_paragraphs", v1_dir / "strategies" / "paragraphs.py", v1_base)
v1_heading_para_mod = load_module_with_base("v1_heading_para", v1_dir / "strategies" / "heading_paragraph.py", v1_base)

# Extract classes
Document = v2_base.Document
Chunk = v2_base.Chunk
RecursiveSplitterStrategy = v2_recursive.RecursiveSplitterStrategy
ClusterSemanticStrategy = v2_semantic.ClusterSemanticStrategy
ParagraphHeadingStrategy = v2_paragraph.ParagraphHeadingStrategy
FixedSizeV2 = v2_fixed.FixedSizeStrategy

FixedSizeV1 = v1_fixed_mod.FixedSizeStrategy
HeadingBasedStrategy = v1_heading_mod.HeadingBasedStrategy
HeadingLimitedStrategy = v1_heading_limited_mod.HeadingLimitedStrategy
HierarchicalStrategy = v1_hierarchical_mod.HierarchicalStrategy
ParagraphStrategy = v1_paragraphs_mod.ParagraphStrategy
HeadingParagraphStrategy = v1_heading_para_mod.HeadingParagraphStrategy

calculate_token_metrics_per_doc = v2_metrics.calculate_token_metrics_per_doc
calculate_key_facts_coverage = v2_metrics.calculate_key_facts_coverage
calculate_document_metrics = v2_metrics.calculate_document_metrics
aggregate_metrics = v2_metrics.aggregate_metrics


def log(msg: str):
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
    v2_path = corpus_dir / "ground_truth_v2.json"
    
    if v2_path.exists():
        log("Loading V2 ground truth with spans...")
        with open(v2_path) as f:
            return json.load(f).get("queries", [])
    
    raise FileNotFoundError(f"Ground truth not found at {v2_path}")


def run_single_strategy(
    strategy,
    documents: list[Document],
    queries: list[dict],
    embedder: SentenceTransformer,
) -> dict:
    """Benchmark a single strategy with fixed metrics."""
    
    # Special handling for ClusterSemanticStrategy
    if hasattr(strategy, "set_embedder"):
        strategy.set_embedder(embedder)
    
    log(f"  Chunking documents...")
    start = time.perf_counter()
    chunks = strategy.chunk_many(documents)
    chunk_time = (time.perf_counter() - start) * 1000
    log(f"  Chunked into {len(chunks)} chunks in {chunk_time:.0f}ms")
    
    if not chunks:
        log(f"  WARNING: No chunks produced!")
        return {"strategy": strategy.name, "error": "no_chunks"}
    
    # Chunk statistics
    token_counts = [c.token_count for c in chunks]
    avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
    log(f"  Token stats: avg={avg_tokens:.0f}, min={min(token_counts)}, max={max(token_counts)}")
    
    # Embed chunks
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
    
    # Run queries
    log(f"  Running {len(queries)} retrieval queries...")
    
    all_token_metrics = []
    all_doc_metrics = []
    all_key_facts_metrics = []
    
    for i, q in enumerate(queries):
        # Embed query
        query_emb = embedder.encode([q["query"]], normalize_embeddings=True)[0]
        
        # Get similarities and top results
        similarities = np.dot(embeddings, query_emb)
        top_indices = np.argsort(similarities)[::-1][:10]
        
        # Get top-5 chunks
        top_5_chunks = [chunks[idx] for idx in top_indices[:5]]
        
        # Document-level metrics
        retrieved_docs = []
        seen_docs = set()
        for idx in top_indices:
            doc_id = chunks[idx].doc_id
            if doc_id not in seen_docs:
                retrieved_docs.append(doc_id)
                seen_docs.add(doc_id)
        
        expected_docs = q.get("expected_docs", [])
        
        doc_metrics = calculate_document_metrics(
            retrieved_docs=retrieved_docs,
            expected_docs=expected_docs,
            k=5,
        )
        all_doc_metrics.append(doc_metrics)
        
        # Key Facts Coverage
        key_facts = q.get("key_facts", [])
        if key_facts:
            retrieved_text = " ".join(c.content for c in top_5_chunks)
            kf_metrics = calculate_key_facts_coverage(retrieved_text, key_facts)
            all_key_facts_metrics.append(kf_metrics)
        
        # Token-level metrics - per-document
        relevant_spans = q.get("relevant_spans", [])
        if relevant_spans:
            retrieved_chunk_spans = [
                {"doc_id": c.doc_id, "start": c.start_char, "end": c.end_char}
                for c in top_5_chunks
            ]
            
            token_metrics = calculate_token_metrics_per_doc(
                retrieved_chunks=retrieved_chunk_spans,
                relevant_spans=relevant_spans,
                expected_docs=expected_docs,
            )
            all_token_metrics.append(token_metrics)
        
        if (i + 1) % 10 == 0:
            log(f"    Processed {i+1}/{len(queries)} queries")
    
    # Aggregate
    aggregated = aggregate_metrics(all_token_metrics, all_doc_metrics, all_key_facts_metrics)
    
    log(f"  Results: KeyFacts={aggregated.get('key_facts_coverage', 0):.1%}, "
        f"Recall@5={aggregated.get('recall_at_k', 0):.1%}, "
        f"TokenRecall={aggregated.get('token_recall', 0):.1%}")
    
    return {
        "strategy": strategy.name,
        "num_chunks": len(chunks),
        "avg_tokens": avg_tokens,
        "min_tokens": min(token_counts) if token_counts else 0,
        "max_tokens": max(token_counts) if token_counts else 0,
        "chunk_time_ms": chunk_time,
        "embed_time_ms": embed_time,
        **aggregated,
    }


def main():
    log("=" * 70)
    log("COMPREHENSIVE CHUNKING BENCHMARK - ALL STRATEGIES (V1 + V2)")
    log("=" * 70)
    
    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"PyTorch device: {device}")
    if device == "cuda":
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Paths
    v1_corpus_dir = v1_dir / "corpus"
    v2_corpus_dir = v2_dir / "corpus"
    results_dir = v2_dir / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Load data
    documents = load_documents(v1_corpus_dir)
    log(f"Loaded {len(documents)} documents")
    
    queries = load_queries(v2_corpus_dir)
    log(f"Loaded {len(queries)} queries")
    
    # Load embedder
    log("Loading embedding model...")
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    log(f"Model loaded on {device}")
    
    # Define ALL strategies
    strategies = [
        # ============ V1 STRATEGIES ============
        # Fixed size (V1 implementation with overlap)
        FixedSizeV1(chunk_size=512, overlap=50),
        
        # Heading-based
        HeadingBasedStrategy(max_heading_level=3),
        
        # Heading with size limit
        HeadingLimitedStrategy(max_tokens=512),
        
        # Hierarchical
        HierarchicalStrategy(),
        
        # Paragraph-based (V1 winner)
        ParagraphStrategy(min_tokens=50, max_tokens=256),
        
        # Heading + Paragraph hybrid
        HeadingParagraphStrategy(),
        
        # ============ V2 STRATEGIES ============
        # Paragraph with heading prepend
        ParagraphHeadingStrategy(min_tokens=50, max_tokens=256, prepend_heading=False),
        ParagraphHeadingStrategy(min_tokens=50, max_tokens=256, prepend_heading=True),
        
        # Recursive splitter - various sizes
        RecursiveSplitterStrategy(chunk_size=800, chunk_overlap=0),   # ~200 tokens
        RecursiveSplitterStrategy(chunk_size=1200, chunk_overlap=0),  # ~300 tokens
        RecursiveSplitterStrategy(chunk_size=1600, chunk_overlap=0),  # ~400 tokens
        RecursiveSplitterStrategy(chunk_size=2000, chunk_overlap=0),  # ~500 tokens
        RecursiveSplitterStrategy(chunk_size=1600, chunk_overlap=240),  # 15% overlap
        
        # Fixed size V2 - various sizes
        FixedSizeV2(chunk_size=200, overlap=0),
        FixedSizeV2(chunk_size=300, overlap=0),
        FixedSizeV2(chunk_size=400, overlap=0),
        FixedSizeV2(chunk_size=512, overlap=0),
        FixedSizeV2(chunk_size=512, overlap=77),  # 15% overlap
        
        # Semantic chunking
        ClusterSemanticStrategy(target_chunk_size=200),
        ClusterSemanticStrategy(target_chunk_size=400),
    ]
    
    all_results = []
    
    for idx, strategy in enumerate(strategies, 1):
        log("")
        log(f"[{idx}/{len(strategies)}] Testing: {strategy.name}")
        log("-" * 60)
        
        try:
            result = run_single_strategy(strategy, documents, queries, embedder)
            all_results.append(result)
        except Exception as e:
            log(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results.append({"strategy": strategy.name, "error": str(e)})
            continue
        
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()
    
    # Save results
    with open(results_dir / "comprehensive_benchmark.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    # Filter out errors and sort by Key Facts Coverage
    valid_results = [r for r in all_results if "error" not in r]
    sorted_results = sorted(
        valid_results,
        key=lambda x: (x.get("key_facts_coverage", 0), x.get("recall_at_k", 0)),
        reverse=True
    )
    
    # Print summary
    log("")
    log("=" * 100)
    log("FINAL RESULTS - SORTED BY KEY FACTS COVERAGE")
    log("=" * 100)
    
    log(f"\n{'Rank':<5} {'Strategy':<35} {'Chunks':>7} {'AvgTok':>7} {'KeyFacts':>9} {'Recall@5':>9} {'TokenRec':>9} {'TokenPrec':>10}")
    log("-" * 105)
    
    for rank, r in enumerate(sorted_results, 1):
        kf = f"{r.get('key_facts_coverage', 0):.1%}"
        rec = f"{r.get('recall_at_k', 0):.1%}"
        tr = f"{r.get('token_recall', 0):.1%}"
        tp = f"{r.get('token_precision', 0):.1%}"
        log(f"{rank:<5} {r['strategy']:<35} {r['num_chunks']:>7} {r['avg_tokens']:>7.0f} {kf:>9} {rec:>9} {tr:>9} {tp:>10}")
    
    # Generate report
    _generate_comprehensive_report(sorted_results, results_dir, len(documents), len(queries), device)
    
    log(f"\nReport saved to: {results_dir / 'comprehensive_report.md'}")
    log("Done!")


def _generate_comprehensive_report(
    results: list[dict],
    results_dir: Path,
    num_docs: int,
    num_queries: int,
    device: str,
):
    """Generate comprehensive markdown report."""
    lines = [
        "# Comprehensive Chunking Benchmark Results",
        "",
        f"**Corpus:** {num_docs} documents, {num_queries} test queries",
        f"**Device:** {device}",
        f"**Strategies tested:** {len(results)} (all V1 + V2 strategies)",
        "",
        "## Evaluation Methodology",
        "",
        "**Primary metric: Key Facts Coverage** - % of key facts found in retrieved chunks",
        "- Directly measures: Can the LLM answer the question?",
        "- Token metrics calculated per-document (fixed bug from V2.0)",
        "",
        "## Results Ranked by Key Facts Coverage",
        "",
        "| Rank | Strategy | Chunks | Avg Tokens | Key Facts | Recall@5 | Token Recall | Token Precision |",
        "|------|----------|--------|------------|-----------|----------|--------------|-----------------|",
    ]
    
    for rank, r in enumerate(results, 1):
        kf = f"{r.get('key_facts_coverage', 0):.1%}"
        rec = f"{r.get('recall_at_k', 0):.1%}"
        tr = f"{r.get('token_recall', 0):.1%}"
        tp = f"{r.get('token_precision', 0):.1%}"
        lines.append(
            f"| {rank} | {r['strategy']} | {r['num_chunks']} | {r['avg_tokens']:.0f} | "
            f"{kf} | {rec} | {tr} | {tp} |"
        )
    
    # Winner analysis
    winner = results[0]
    lines.extend([
        "",
        "## Winner Analysis",
        "",
        f"### Best Strategy: `{winner['strategy']}`",
        "",
        f"- **Key Facts Coverage:** {winner.get('key_facts_coverage', 0):.1%}",
        f"- **Recall@5:** {winner.get('recall_at_k', 0):.1%}",
        f"- **Token Recall:** {winner.get('token_recall', 0):.1%}",
        f"- **Token Precision:** {winner.get('token_precision', 0):.1%}",
        f"- **Number of chunks:** {winner['num_chunks']}",
        f"- **Average tokens per chunk:** {winner['avg_tokens']:.0f}",
        "",
    ])
    
    # Group analysis
    lines.extend([
        "## Analysis by Strategy Type",
        "",
        "### Small Chunks (< 100 tokens avg)",
        "",
    ])
    
    small = [r for r in results if r['avg_tokens'] < 100]
    if small:
        best_small = max(small, key=lambda x: x.get('key_facts_coverage', 0))
        lines.append(f"Best: `{best_small['strategy']}` - {best_small.get('key_facts_coverage', 0):.1%} Key Facts")
    
    lines.extend([
        "",
        "### Medium Chunks (100-250 tokens avg)",
        "",
    ])
    
    medium = [r for r in results if 100 <= r['avg_tokens'] < 250]
    if medium:
        best_med = max(medium, key=lambda x: x.get('key_facts_coverage', 0))
        lines.append(f"Best: `{best_med['strategy']}` - {best_med.get('key_facts_coverage', 0):.1%} Key Facts")
    
    lines.extend([
        "",
        "### Large Chunks (250+ tokens avg)",
        "",
    ])
    
    large = [r for r in results if r['avg_tokens'] >= 250]
    if large:
        best_large = max(large, key=lambda x: x.get('key_facts_coverage', 0))
        lines.append(f"Best: `{best_large['strategy']}` - {best_large.get('key_facts_coverage', 0):.1%} Key Facts")
    
    # Recommendation
    lines.extend([
        "",
        "## Final Recommendation",
        "",
        f"Use **`{winner['strategy']}`** for the Personal Knowledge Assistant.",
        "",
        "This strategy provides the best balance of:",
        f"- High Key Facts Coverage ({winner.get('key_facts_coverage', 0):.1%})",
        f"- High Token Recall ({winner.get('token_recall', 0):.1%})",
        f"- Reasonable chunk count ({winner['num_chunks']} chunks)",
        "",
    ])
    
    with open(results_dir / "comprehensive_report.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
