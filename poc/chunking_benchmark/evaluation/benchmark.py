"""Main benchmark runner."""

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional
import sys

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies import (
    ChunkingStrategy, Document, Chunk,
    FixedSizeStrategy,
    HeadingBasedStrategy,
    HeadingLimitedStrategy,
    HierarchicalStrategy,
    ParagraphStrategy,
    HeadingParagraphStrategy,
)

# Handle both relative and absolute imports
try:
    from .embedder import Embedder
    from .retriever import Retriever
    from .metrics import (
        calculate_metrics, 
        aggregate_metrics, 
        calculate_chunk_stats,
        RetrievalMetrics,
        AggregateMetrics,
        ChunkStats,
    )
except ImportError:
    from evaluation.embedder import Embedder
    from evaluation.retriever import Retriever
    from evaluation.metrics import (
        calculate_metrics, 
        aggregate_metrics, 
        calculate_chunk_stats,
        RetrievalMetrics,
        AggregateMetrics,
        ChunkStats,
    )


@dataclass
class StrategyResult:
    """Results for a single chunking strategy."""
    strategy_name: str
    
    # Timing
    chunk_time_ms: float = 0.0
    embed_time_ms: float = 0.0
    total_index_time_ms: float = 0.0
    
    # Chunk statistics
    chunk_stats: Optional[ChunkStats] = None
    
    # Retrieval metrics
    aggregate_metrics: Optional[AggregateMetrics] = None
    per_query_metrics: list[RetrievalMetrics] = field(default_factory=list)
    
    # Raw data for analysis
    chunks: list = field(default_factory=list)


@dataclass
class BenchmarkResult:
    """Full benchmark results."""
    timestamp: str
    corpus_docs: int
    num_queries: int
    
    strategy_results: dict[str, StrategyResult] = field(default_factory=dict)


def load_documents(corpus_dir: Path) -> list[Document]:
    """Load documents from corpus directory.
    
    Args:
        corpus_dir: Path to corpus/documents directory.
        
    Returns:
        List of Document objects.
    """
    documents = []
    metadata_path = corpus_dir.parent / "corpus_metadata.json"
    
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        
        for doc_meta in metadata:
            doc_path = corpus_dir / doc_meta["filename"]
            if doc_path.exists():
                content = doc_path.read_text()
                documents.append(Document(
                    id=doc_meta["id"],
                    title=doc_meta["title"],
                    content=content,
                    path=str(doc_path),
                    metadata={"filename": doc_meta["filename"]}
                ))
    
    return documents


def load_ground_truth(corpus_dir: Path) -> list[dict]:
    """Load ground truth queries.
    
    Args:
        corpus_dir: Path to corpus directory.
        
    Returns:
        List of query dictionaries.
    """
    gt_path = corpus_dir / "ground_truth.json"
    if not gt_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {gt_path}")
    
    with open(gt_path) as f:
        data = json.load(f)
    
    return data.get("queries", [])


def get_all_strategies() -> list[ChunkingStrategy]:
    """Get all chunking strategies to benchmark."""
    return [
        FixedSizeStrategy(chunk_size=512, overlap=50),
        HeadingBasedStrategy(),
        HeadingLimitedStrategy(max_tokens=512),
        HierarchicalStrategy(),
        ParagraphStrategy(min_tokens=50, max_tokens=256),
        HeadingParagraphStrategy(),
    ]


def run_strategy_benchmark(
    strategy: ChunkingStrategy,
    documents: list[Document],
    queries: list[dict],
    embedder: Embedder,
) -> StrategyResult:
    """Run benchmark for a single strategy.
    
    Args:
        strategy: Chunking strategy to benchmark.
        documents: List of documents to chunk.
        queries: List of ground truth queries.
        embedder: Shared embedder instance.
        
    Returns:
        StrategyResult with all metrics.
    """
    result = StrategyResult(strategy_name=strategy.name)
    
    # 1. Chunk documents
    print(f"  Chunking with {strategy.name}...")
    start = time.perf_counter()
    chunks = strategy.chunk_many(documents)
    result.chunk_time_ms = (time.perf_counter() - start) * 1000
    result.chunks = chunks
    
    # 2. Calculate chunk statistics
    result.chunk_stats = calculate_chunk_stats(chunks)
    print(f"    {len(chunks)} chunks, avg {result.chunk_stats.avg_tokens:.0f} tokens")
    
    # 3. Embed and index chunks
    print(f"  Embedding {len(chunks)} chunks...")
    retriever = Retriever(embedder)
    start = time.perf_counter()
    retriever.index(chunks, show_progress=False)
    result.embed_time_ms = (time.perf_counter() - start) * 1000
    result.total_index_time_ms = result.chunk_time_ms + result.embed_time_ms
    
    # 4. Run retrieval evaluation
    print(f"  Running {len(queries)} queries...")
    query_metrics = []
    
    for q in queries:
        query_text = q["query"]
        expected_docs = q.get("expected_docs", [])
        
        # Retrieve
        results = retriever.search(query_text, k=10)
        
        # Calculate metrics
        metrics = calculate_metrics(
            query_id=q["id"],
            query=query_text,
            results=results,
            expected_docs=expected_docs,
        )
        query_metrics.append(metrics)
    
    result.per_query_metrics = query_metrics
    result.aggregate_metrics = aggregate_metrics(query_metrics)
    
    return result


def run_benchmark(
    corpus_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
) -> BenchmarkResult:
    """Run full benchmark across all strategies.
    
    Args:
        corpus_dir: Path to corpus directory. Defaults to ./corpus.
        output_dir: Path to output directory. Defaults to ./results.
        
    Returns:
        BenchmarkResult with all metrics.
    """
    # Set default paths
    if corpus_dir is None:
        corpus_dir = Path(__file__).parent.parent / "corpus"
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    
    output_dir.mkdir(exist_ok=True)
    docs_dir = corpus_dir / "documents"
    
    # Load data
    print("Loading documents...")
    documents = load_documents(docs_dir)
    print(f"Loaded {len(documents)} documents")
    
    print("Loading ground truth...")
    queries = load_ground_truth(corpus_dir)
    print(f"Loaded {len(queries)} queries")
    
    # Create shared embedder
    print("Initializing embedder...")
    embedder = Embedder()
    
    # Get strategies
    strategies = get_all_strategies()
    
    # Run benchmarks
    result = BenchmarkResult(
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        corpus_docs=len(documents),
        num_queries=len(queries),
    )
    
    for strategy in strategies:
        print(f"\n{'='*50}")
        print(f"Benchmarking: {strategy.name}")
        print('='*50)
        
        strategy_result = run_strategy_benchmark(
            strategy=strategy,
            documents=documents,
            queries=queries,
            embedder=embedder,
        )
        result.strategy_results[strategy.name] = strategy_result
        
        # Print summary
        agg = strategy_result.aggregate_metrics
        print(f"\n  Results:")
        print(f"    Doc Recall@5: {agg.avg_doc_recall_at_5:.1%}")
        print(f"    MRR: {agg.mean_mrr:.3f}")
        print(f"    Index time: {strategy_result.total_index_time_ms:.0f}ms")
    
    # Save results
    save_results(result, output_dir)
    
    return result


def save_results(result: BenchmarkResult, output_dir: Path) -> None:
    """Save benchmark results to files.
    
    Args:
        result: Benchmark results.
        output_dir: Directory to save results.
    """
    # Save summary JSON (without raw chunks)
    summary = {
        "timestamp": result.timestamp,
        "corpus_docs": result.corpus_docs,
        "num_queries": result.num_queries,
        "strategies": {},
    }
    
    for name, sr in result.strategy_results.items():
        summary["strategies"][name] = {
            "chunk_time_ms": sr.chunk_time_ms,
            "embed_time_ms": sr.embed_time_ms,
            "total_index_time_ms": sr.total_index_time_ms,
            "chunk_stats": asdict(sr.chunk_stats) if sr.chunk_stats else None,
            "aggregate_metrics": asdict(sr.aggregate_metrics) if sr.aggregate_metrics else None,
        }
    
    with open(output_dir / "benchmark_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Generate markdown report
    generate_report(result, output_dir / "report.md")


def generate_report(result: BenchmarkResult, report_path: Path) -> None:
    """Generate markdown report from benchmark results.
    
    Args:
        result: Benchmark results.
        report_path: Path to save report.
    """
    lines = [
        "# Chunking Benchmark Results",
        "",
        f"**Date:** {result.timestamp}",
        f"**Corpus:** {result.corpus_docs} documents",
        f"**Queries:** {result.num_queries} test queries",
        "",
        "## Summary",
        "",
        "| Strategy | Chunks | Avg Tokens | Recall@5 | MRR | Index Time |",
        "|----------|--------|------------|----------|-----|------------|",
    ]
    
    # Sort by Recall@5 descending
    sorted_strategies = sorted(
        result.strategy_results.items(),
        key=lambda x: x[1].aggregate_metrics.avg_doc_recall_at_5 if x[1].aggregate_metrics else 0,
        reverse=True
    )
    
    for name, sr in sorted_strategies:
        cs = sr.chunk_stats
        am = sr.aggregate_metrics
        lines.append(
            f"| {name} | {cs.num_chunks} | {cs.avg_tokens:.0f} | "
            f"{am.avg_doc_recall_at_5:.1%} | {am.mean_mrr:.3f} | "
            f"{sr.total_index_time_ms:.0f}ms |"
        )
    
    lines.extend([
        "",
        "## Detailed Metrics",
        "",
    ])
    
    for name, sr in sorted_strategies:
        am = sr.aggregate_metrics
        cs = sr.chunk_stats
        
        lines.extend([
            f"### {name}",
            "",
            "**Retrieval Metrics:**",
            f"- Document Recall@1: {am.avg_doc_recall_at_1:.1%}",
            f"- Document Recall@3: {am.avg_doc_recall_at_3:.1%}",
            f"- Document Recall@5: {am.avg_doc_recall_at_5:.1%}",
            f"- Document Recall@10: {am.avg_doc_recall_at_10:.1%}",
            f"- Mean Reciprocal Rank: {am.mean_mrr:.3f}",
            "",
            "**Chunk Statistics:**",
            f"- Number of chunks: {cs.num_chunks}",
            f"- Average tokens: {cs.avg_tokens:.0f}",
            f"- Min/Max tokens: {cs.min_tokens} / {cs.max_tokens}",
            f"- Std dev: {cs.std_tokens:.0f}",
            "",
            "**Size Distribution:**",
            f"- Under 100 tokens: {cs.pct_under_100:.1%}",
            f"- 100-300 tokens: {cs.pct_100_300:.1%}",
            f"- 300-500 tokens: {cs.pct_300_500:.1%}",
            f"- 500-800 tokens: {cs.pct_500_800:.1%}",
            f"- Over 800 tokens: {cs.pct_over_800:.1%}",
            "",
            "**Performance:**",
            f"- Chunking time: {sr.chunk_time_ms:.0f}ms",
            f"- Embedding time: {sr.embed_time_ms:.0f}ms",
            f"- Total index time: {sr.total_index_time_ms:.0f}ms",
            "",
        ])
    
    # Per-category analysis
    lines.extend([
        "## Performance by Query Category",
        "",
    ])
    
    # Get all categories
    all_categories = set()
    for sr in result.strategy_results.values():
        if sr.aggregate_metrics:
            all_categories.update(sr.aggregate_metrics.category_metrics.keys())
    
    for category in sorted(all_categories):
        lines.extend([
            f"### {category.replace('_', ' ').title()}",
            "",
            "| Strategy | Recall@5 | MRR | Hit Rate@5 |",
            "|----------|----------|-----|------------|",
        ])
        
        for name, sr in sorted_strategies:
            if sr.aggregate_metrics and category in sr.aggregate_metrics.category_metrics:
                cat = sr.aggregate_metrics.category_metrics[category]
                lines.append(
                    f"| {name} | {cat['avg_doc_recall_at_5']:.1%} | "
                    f"{cat['avg_mrr']:.3f} | {cat['hit_rate_at_5']:.1%} |"
                )
        
        lines.append("")
    
    # Recommendation
    best = sorted_strategies[0]
    lines.extend([
        "## Recommendation",
        "",
        f"**Best performing strategy: {best[0]}**",
        "",
        f"Achieves {best[1].aggregate_metrics.avg_doc_recall_at_5:.1%} document recall at K=5 "
        f"with {best[1].chunk_stats.num_chunks} chunks averaging "
        f"{best[1].chunk_stats.avg_tokens:.0f} tokens each.",
        "",
    ])
    
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    run_benchmark()
