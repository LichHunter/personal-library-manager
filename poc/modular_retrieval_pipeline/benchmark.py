#!/usr/bin/env python3
"""Benchmark comparing enriched_hybrid_llm baseline vs modular pipeline.

This benchmark validates that the new modular pipeline achieves comparable
accuracy to the existing enriched_hybrid_llm strategy (90% baseline).

Usage:
    python poc/modular_retrieval_pipeline/benchmark.py --questions poc/chunking_benchmark_v2/corpus/needle_questions.json

Comparison:
    - Baseline: enriched_hybrid_llm (BM25 + BGE + YAKE + spaCy + Claude Haiku + RRF)
    - Modular: Component-based pipeline with equivalent configuration

Metrics:
    - Accuracy: % of questions where needle document found in top-5
    - Latency: Average query time (ms)
    - Memory: Peak memory usage (MB)
"""

import argparse
import json
import sys
import time
import tracemalloc
from datetime import datetime
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

# Add paths for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "chunking_benchmark_v2"))

from strategies import Document, Chunk, MarkdownSemanticStrategy
from retrieval import create_retrieval_strategy

# Import modular pipeline components
poc_path = Path(__file__).parent.parent
sys.path.insert(0, str(poc_path))
from modular_retrieval_pipeline.types import Query
from modular_retrieval_pipeline.components.query_rewriter import QueryRewriter
from modular_retrieval_pipeline.components.query_expander import QueryExpander
from modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor
from modular_retrieval_pipeline.components.entity_extractor import EntityExtractor
from modular_retrieval_pipeline.components.content_enricher import ContentEnricher
from modular_retrieval_pipeline.base import Pipeline
from modular_retrieval_pipeline.modular_enriched_hybrid_llm import (
    ModularEnrichedHybridLLM,
)
from modular_retrieval_pipeline.modular_enriched_hybrid import (
    ModularEnrichedHybrid,
)

# Import caching components
from modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
from modular_retrieval_pipeline.cache.cached_keyword_extractor import (
    CachedKeywordExtractor,
)
from modular_retrieval_pipeline.cache.cached_entity_extractor import (
    CachedEntityExtractor,
)
from modular_retrieval_pipeline.cache.caching_embedder import CachingEmbedder


CORPUS_DIR = Path("poc/chunking_benchmark_v2/corpus/kubernetes")
DEFAULT_QUESTIONS = Path("poc/chunking_benchmark_v2/corpus/needle_questions.json")


def load_questions(questions_file: Path) -> tuple[list[dict], str]:
    """Load questions from JSON file.

    Returns:
        (questions, needle_doc_id)
    """
    with open(questions_file) as f:
        data = json.load(f)
    return data["questions"], data["needle_doc_id"]


def load_documents() -> list[Document]:
    """Load all documents from corpus."""
    documents = []
    for doc_path in sorted(CORPUS_DIR.glob("*.md")):
        doc_id = doc_path.stem
        content = doc_path.read_text()

        # Extract title
        title = doc_id
        for line in content.split("\n"):
            if line.startswith("title:"):
                title = line.replace("title:", "").strip().strip("\"'")
                break
            elif line.startswith("# "):
                title = line[2:].strip()
                break

        doc = Document(id=doc_id, title=title, content=content, path=str(doc_path))
        documents.append(doc)

    return documents


def chunk_documents(documents: list[Document]) -> list[Chunk]:
    """Chunk documents using MarkdownSemanticStrategy."""
    strategy = MarkdownSemanticStrategy(
        max_heading_level=4,
        target_chunk_size=400,
        min_chunk_size=50,
        max_chunk_size=800,
        overlap_sentences=1,
    )

    all_chunks = []
    for doc in documents:
        chunks = strategy.chunk(doc)
        all_chunks.extend(chunks)

    return all_chunks


def run_baseline_benchmark(
    questions: list[dict],
    needle_doc_id: str,
    chunks: list[Chunk],
    documents: list[Document],
    embedder: SentenceTransformer,
    cache: RedisCacheClient | None = None,
) -> dict:
    """Run baseline enriched_hybrid_llm benchmark.

    Returns:
        {
            'accuracy': float,
            'avg_latency_ms': float,
            'peak_memory_mb': float,
            'results': list[dict],
        }
    """
    print("\n" + "=" * 60)
    print("BASELINE: enriched_hybrid_llm")
    print("=" * 60)

    # Initialize strategy
    strategy = create_retrieval_strategy("enriched_hybrid_llm", debug=False)
    strategy.set_embedder(embedder)

    # Log cache status
    if cache and cache.is_connected():
        print("Cache: enabled (Redis connected)")
    else:
        print("Cache: disabled")

    # Index chunks
    print("Indexing chunks...")
    tracemalloc.start()
    index_start = time.time()
    strategy.index(chunks, documents)
    index_time = time.time() - index_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    print(f"Indexed {len(chunks)} chunks in {index_time:.1f}s")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")

    # Log cache statistics if available
    if cache:
        stats = (
            strategy.get_cache_stats() if hasattr(strategy, "get_cache_stats") else None
        )
        if stats:
            hits = stats.get("total_hits", 0)
            misses = stats.get("total_misses", 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            print(
                f"Cache stats: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)"
            )

    # Run retrieval for each question
    print(f"\nRunning retrieval for {len(questions)} questions...")
    results = []
    latencies = []

    for i, q in enumerate(questions):
        query_start = time.time()
        retrieved = strategy.retrieve(q["question"], k=5)
        latency = (time.time() - query_start) * 1000  # ms
        latencies.append(latency)

        # Check if needle found
        needle_found = any(c.doc_id == needle_doc_id for c in retrieved)

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "needle_found": needle_found,
            "latency_ms": round(latency, 1),
        }
        results.append(result)

        status = "✓" if needle_found else "✗"
        print(
            f"  [{i + 1:2d}/{len(questions)}] {status} ({latency:.0f}ms) {q['question'][:50]}..."
        )

    # Calculate metrics
    accuracy = sum(1 for r in results if r["needle_found"]) / len(results) * 100
    avg_latency = sum(latencies) / len(latencies)

    print(
        f"\nAccuracy: {accuracy:.1f}% ({sum(1 for r in results if r['needle_found'])}/{len(results)})"
    )
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print(f"Peak Memory: {peak_memory_mb:.1f}MB")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "peak_memory_mb": peak_memory_mb,
        "results": results,
    }


def run_modular_benchmark(
    questions: list[dict],
    needle_doc_id: str,
    chunks: list[Chunk],
    documents: list[Document],
    embedder: SentenceTransformer,
    cache: RedisCacheClient | None = None,
) -> dict:
    """Run modular pipeline benchmark using ModularEnrichedHybridLLM.

    This benchmark uses the ModularEnrichedHybridLLM orchestrator which replicates
    the exact behavior of the baseline enriched_hybrid_llm strategy using modular
    components.

    Configuration:
    - Embedder: BAAI/bge-base-en-v1.5 (same as baseline)
    - k: 5 (same as baseline)
    - RRF: Semantic first, BM25 second (same as baseline)
    - Adaptive weights based on query expansion

    Returns:
        {
            'accuracy': float,
            'avg_latency_ms': float,
            'peak_memory_mb': float,
            'results': list[dict],
        }
    """
    print("\n" + "=" * 60)
    print("MODULAR PIPELINE: ModularEnrichedHybridLLM")
    print("=" * 60)

    # Initialize modular strategy
    strategy = ModularEnrichedHybridLLM(cache=cache, debug=False)
    if cache and cache.is_connected():
        strategy.set_cached_embedder(embedder, cache)
    else:
        strategy.set_embedder(embedder)

    # Log cache status
    if cache and cache.is_connected():
        print("Cache: enabled (Redis connected)")
    else:
        print("Cache: disabled")

    # Index chunks
    print("Indexing chunks...")
    tracemalloc.start()
    index_start = time.time()
    strategy.index(chunks, documents)
    index_time = time.time() - index_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    print(f"Indexed {len(chunks)} chunks in {index_time:.1f}s")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")

    # Run retrieval for each question
    print(f"\nRunning retrieval for {len(questions)} questions...")
    results = []
    latencies = []

    for i, q in enumerate(questions):
        query_start = time.time()
        retrieved = strategy.retrieve(q["question"], k=5)
        latency = (time.time() - query_start) * 1000  # ms
        latencies.append(latency)

        # Check if needle found
        needle_found = any(c.doc_id == needle_doc_id for c in retrieved)

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "needle_found": needle_found,
            "latency_ms": round(latency, 1),
        }
        results.append(result)

        status = "✓" if needle_found else "✗"
        print(
            f"  [{i + 1:2d}/{len(questions)}] {status} ({latency:.0f}ms) {q['question'][:50]}..."
        )

    # Calculate metrics
    accuracy = sum(1 for r in results if r["needle_found"]) / len(results) * 100
    avg_latency = sum(latencies) / len(latencies)

    print(
        f"\nAccuracy: {accuracy:.1f}% ({sum(1 for r in results if r['needle_found'])}/{len(results)})"
    )
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print(f"Peak Memory: {peak_memory_mb:.1f}MB")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "peak_memory_mb": peak_memory_mb,
        "results": results,
    }


def run_modular_no_llm_benchmark(
    questions: list[dict],
    needle_doc_id: str,
    chunks: list[Chunk],
    documents: list[Document],
    embedder: SentenceTransformer,
    cache: RedisCacheClient | None = None,
) -> dict:
    """Run modular pipeline benchmark using ModularEnrichedHybrid (no LLM).

    This benchmark uses the ModularEnrichedHybrid orchestrator which skips
    LLM-based query rewriting for faster latency.

    Configuration:
    - Embedder: BAAI/bge-base-en-v1.5 (same as others)
    - k: 5 (same as others)
    - RRF: Semantic first, BM25 second (same as others)
    - Adaptive weights based on query expansion (same as others)
    - NO LLM query rewriting (key difference)

    Returns:
        {
            'accuracy': float,
            'avg_latency_ms': float,
            'peak_memory_mb': float,
            'results': list[dict],
        }
    """
    print("\n" + "=" * 60)
    print("MODULAR PIPELINE: ModularEnrichedHybrid (No LLM)")
    print("=" * 60)

    # Initialize strategy (NO rewrite_timeout parameter!)
    strategy = ModularEnrichedHybrid(cache=cache, debug=False)
    if cache and cache.is_connected():
        strategy.set_cached_embedder(embedder, cache)
    else:
        strategy.set_embedder(embedder)

    # Log cache status
    if cache and cache.is_connected():
        print("Cache: enabled (Redis connected)")
    else:
        print("Cache: disabled")

    # Index chunks
    print("Indexing chunks...")
    tracemalloc.start()
    index_start = time.time()
    strategy.index(chunks, documents)
    index_time = time.time() - index_start
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_memory_mb = peak / 1024 / 1024

    print(f"Indexed {len(chunks)} chunks in {index_time:.1f}s")
    print(f"Peak memory: {peak_memory_mb:.1f} MB")

    # Log cache statistics if available
    if cache:
        stats = (
            strategy.get_cache_stats() if hasattr(strategy, "get_cache_stats") else None
        )
        if stats:
            hits = stats.get("total_hits", 0)
            misses = stats.get("total_misses", 0)
            total = hits + misses
            hit_rate = (hits / total * 100) if total > 0 else 0
            print(
                f"Cache stats: {hits} hits, {misses} misses ({hit_rate:.1f}% hit rate)"
            )

    # Run retrieval for each question
    print(f"\nRunning retrieval for {len(questions)} questions...")
    results = []
    latencies = []

    for i, q in enumerate(questions):
        query_start = time.time()
        retrieved = strategy.retrieve(q["question"], k=5)
        latency = (time.time() - query_start) * 1000  # ms
        latencies.append(latency)

        # Check if needle found
        needle_found = any(c.doc_id == needle_doc_id for c in retrieved)

        result = {
            "question_id": q["id"],
            "question": q["question"],
            "needle_found": needle_found,
            "latency_ms": round(latency, 1),
        }
        results.append(result)

        status = "✓" if needle_found else "✗"
        print(
            f"  [{i + 1:2d}/{len(questions)}] {status} ({latency:.0f}ms) {q['question'][:50]}..."
        )

    # Calculate metrics
    accuracy = sum(1 for r in results if r["needle_found"]) / len(results) * 100
    avg_latency = sum(latencies) / len(latencies)

    print(
        f"\nAccuracy: {accuracy:.1f}% ({sum(1 for r in results if r['needle_found'])}/{len(results)})"
    )
    print(f"Avg Latency: {avg_latency:.1f}ms")
    print(f"Peak Memory: {peak_memory_mb:.1f}MB")

    return {
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "peak_memory_mb": peak_memory_mb,
        "results": results,
    }


def generate_report(
    baseline: dict,
    modular: dict,
    questions_file: Path,
    output_file: Path,
) -> None:
    """Generate comparison report."""
    report = {
        "benchmark_run_at": datetime.now().isoformat(),
        "questions_file": str(questions_file),
        "baseline": {
            "strategy": "enriched_hybrid_llm",
            "accuracy": baseline["accuracy"],
            "avg_latency_ms": baseline["avg_latency_ms"],
            "peak_memory_mb": baseline["peak_memory_mb"],
        },
        "modular": {
            "strategy": "enriched_hybrid_llm (modular)",
            "accuracy": modular["accuracy"],
            "avg_latency_ms": modular["avg_latency_ms"],
            "peak_memory_mb": modular["peak_memory_mb"],
            "status": modular.get("status", "complete"),
        },
        "comparison": {
            "accuracy_diff": modular["accuracy"] - baseline["accuracy"],
            "latency_diff_ms": modular["avg_latency_ms"] - baseline["avg_latency_ms"],
            "memory_diff_mb": modular["peak_memory_mb"] - baseline["peak_memory_mb"],
        },
    }

    # Save JSON report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (enriched_hybrid_llm):")
    print(f"  Accuracy: {baseline['accuracy']:.1f}%")
    print(f"  Avg Latency: {baseline['avg_latency_ms']:.1f}ms")
    print(f"  Peak Memory: {baseline['peak_memory_mb']:.1f}MB")

    print(f"\nModular Pipeline:")
    print(f"  Accuracy: {modular['accuracy']:.1f}%")
    print(f"  Avg Latency: {modular['avg_latency_ms']:.1f}ms")
    print(f"  Peak Memory: {modular['peak_memory_mb']:.1f}MB")
    print(f"  Status: {modular.get('status', 'complete')}")

    if modular.get("status") != "incomplete":
        print(f"\nComparison:")
        print(f"  Accuracy Diff: {report['comparison']['accuracy_diff']:+.1f}%")
        print(f"  Latency Diff: {report['comparison']['latency_diff_ms']:+.1f}ms")
        print(f"  Memory Diff: {report['comparison']['memory_diff_mb']:+.1f}MB")

        # Verdict
        accuracy_pass = modular["accuracy"] >= 85.0
        verdict = "✓ PASS" if accuracy_pass else "✗ FAIL"
        print(f"\nVERDICT: {verdict} (target: ≥85% accuracy)")

    print(f"\nReport saved to: {output_file}")


def generate_three_way_report(
    baseline: dict,
    modular: dict,
    modular_no_llm: dict,
    questions_file: Path,
    output_file: Path,
) -> None:
    """Generate three-way comparison report."""
    report = {
        "benchmark_run_at": datetime.now().isoformat(),
        "questions_file": str(questions_file),
        "baseline": {
            "strategy": "enriched_hybrid_llm",
            "accuracy": baseline["accuracy"],
            "avg_latency_ms": baseline["avg_latency_ms"],
            "peak_memory_mb": baseline["peak_memory_mb"],
        },
        "modular_with_llm": {
            "strategy": "enriched_hybrid_llm (modular with LLM)",
            "accuracy": modular["accuracy"],
            "avg_latency_ms": modular["avg_latency_ms"],
            "peak_memory_mb": modular["peak_memory_mb"],
        },
        "modular_no_llm": {
            "strategy": "enriched_hybrid (modular no LLM)",
            "accuracy": modular_no_llm["accuracy"],
            "avg_latency_ms": modular_no_llm["avg_latency_ms"],
            "peak_memory_mb": modular_no_llm["peak_memory_mb"],
        },
        "comparison": {
            "modular_vs_baseline_accuracy": modular["accuracy"] - baseline["accuracy"],
            "modular_vs_baseline_latency_ms": modular["avg_latency_ms"]
            - baseline["avg_latency_ms"],
            "modular_vs_baseline_memory_mb": modular["peak_memory_mb"]
            - baseline["peak_memory_mb"],
            "no_llm_vs_baseline_accuracy": modular_no_llm["accuracy"]
            - baseline["accuracy"],
            "no_llm_vs_baseline_latency_ms": modular_no_llm["avg_latency_ms"]
            - baseline["avg_latency_ms"],
            "no_llm_vs_baseline_memory_mb": modular_no_llm["peak_memory_mb"]
            - baseline["peak_memory_mb"],
            "no_llm_vs_with_llm_accuracy": modular_no_llm["accuracy"]
            - modular["accuracy"],
            "no_llm_vs_with_llm_latency_ms": modular_no_llm["avg_latency_ms"]
            - modular["avg_latency_ms"],
            "no_llm_vs_with_llm_memory_mb": modular_no_llm["peak_memory_mb"]
            - modular["peak_memory_mb"],
        },
    }

    # Save JSON report
    with open(output_file, "w") as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("THREE-WAY BENCHMARK SUMMARY")
    print("=" * 60)
    print(f"\nBaseline (enriched_hybrid_llm):")
    print(f"  Accuracy: {baseline['accuracy']:.1f}%")
    print(f"  Avg Latency: {baseline['avg_latency_ms']:.1f}ms")
    print(f"  Peak Memory: {baseline['peak_memory_mb']:.1f}MB")

    print(f"\nModular Pipeline (with LLM):")
    print(f"  Accuracy: {modular['accuracy']:.1f}%")
    print(f"  Avg Latency: {modular['avg_latency_ms']:.1f}ms")
    print(f"  Peak Memory: {modular['peak_memory_mb']:.1f}MB")

    print(f"\nModular Pipeline (no LLM):")
    print(f"  Accuracy: {modular_no_llm['accuracy']:.1f}%")
    print(f"  Avg Latency: {modular_no_llm['avg_latency_ms']:.1f}ms")
    print(f"  Peak Memory: {modular_no_llm['peak_memory_mb']:.1f}MB")

    print(f"\nComparison (vs Baseline):")
    print(
        f"  With LLM - Accuracy Diff: {report['comparison']['modular_vs_baseline_accuracy']:+.1f}%"
    )
    print(
        f"  With LLM - Latency Diff: {report['comparison']['modular_vs_baseline_latency_ms']:+.1f}ms"
    )
    print(
        f"  With LLM - Memory Diff: {report['comparison']['modular_vs_baseline_memory_mb']:+.1f}MB"
    )
    print(
        f"  No LLM - Accuracy Diff: {report['comparison']['no_llm_vs_baseline_accuracy']:+.1f}%"
    )
    print(
        f"  No LLM - Latency Diff: {report['comparison']['no_llm_vs_baseline_latency_ms']:+.1f}ms"
    )
    print(
        f"  No LLM - Memory Diff: {report['comparison']['no_llm_vs_baseline_memory_mb']:+.1f}MB"
    )

    print(f"\nComparison (No LLM vs With LLM):")
    print(
        f"  Accuracy Diff: {report['comparison']['no_llm_vs_with_llm_accuracy']:+.1f}%"
    )
    print(
        f"  Latency Diff: {report['comparison']['no_llm_vs_with_llm_latency_ms']:+.1f}ms"
    )
    print(
        f"  Memory Diff: {report['comparison']['no_llm_vs_with_llm_memory_mb']:+.1f}MB"
    )

    print(f"\nReport saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark modular pipeline vs enriched_hybrid_llm baseline"
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=DEFAULT_QUESTIONS,
        help="Path to questions JSON file",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("poc/modular_retrieval_pipeline/benchmark_results.json"),
        help="Output file for results",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode: test with first 5 questions only",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["baseline", "modular", "modular-no-llm", "all"],
        default="all",
        help="Which strategy to run (default: all)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable Redis caching (cache enabled by default)",
    )

    args = parser.parse_args()

    # Load data
    print("Loading questions...")
    questions, needle_doc_id = load_questions(args.questions)

    if args.quick:
        questions = questions[:5]
        print(f"QUICK MODE: Using first {len(questions)} questions only")

    print(f"Loaded {len(questions)} questions for needle: {needle_doc_id}")

    print("\nLoading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} documents")

    print("\nChunking documents...")
    chunks = chunk_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("\nLoading embedder...")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    print("Embedder loaded")

    # Initialize cache if enabled
    cache = None if args.no_cache else RedisCacheClient()

    # Run benchmarks based on --strategy flag
    baseline = None
    modular = None
    modular_no_llm = None

    if args.strategy in ["baseline", "all"]:
        baseline = run_baseline_benchmark(
            questions, needle_doc_id, chunks, documents, embedder, cache
        )

    if args.strategy in ["modular", "all"]:
        modular = run_modular_benchmark(
            questions, needle_doc_id, chunks, documents, embedder, cache
        )

    if args.strategy in ["modular-no-llm", "all"]:
        modular_no_llm = run_modular_no_llm_benchmark(
            questions, needle_doc_id, chunks, documents, embedder, cache
        )

    # Generate report
    if args.strategy == "all":
        # All three must be non-None here
        assert (
            baseline is not None and modular is not None and modular_no_llm is not None
        )
        generate_three_way_report(
            baseline, modular, modular_no_llm, args.questions, args.output
        )
    else:
        # Single strategy report - get the non-None result
        if baseline is not None:
            result = baseline
            strategy_name = "baseline"
        elif modular is not None:
            result = modular
            strategy_name = "modular"
        else:
            assert modular_no_llm is not None
            result = modular_no_llm
            strategy_name = "modular-no-llm"

        print(f"\n{'=' * 60}")
        print(f"RESULTS: {strategy_name}")
        print(f"{'=' * 60}")
        print(f"Accuracy: {result['accuracy']:.1f}%")
        print(f"Avg Latency: {result['avg_latency_ms']:.2f}ms")
        print(f"Peak Memory: {result['peak_memory_mb']:.2f}MB")
        print(f"\nResults saved to: {args.output}")

        # Save single strategy result
        with open(args.output, "w") as f:
            json.dump(
                {
                    "strategy": strategy_name,
                    "benchmark_run_at": datetime.now().isoformat(),
                    "questions_file": str(args.questions),
                    **result,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
