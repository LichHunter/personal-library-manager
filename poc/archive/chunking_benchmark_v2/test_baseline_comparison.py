#!/usr/bin/env python3
"""Test runner for baseline comparison across 3 retrieval strategies.

Runs adaptive_hybrid, synthetic_variants, and enriched_hybrid_llm strategies
against 10 baseline questions and generates individual result files plus a
side-by-side comparison report.

Usage:
    python test_baseline_comparison.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from strategies import Document, Chunk, MarkdownSemanticStrategy
from retrieval import create_retrieval_strategy, RETRIEVAL_STRATEGIES


def load_baseline_queries(queries_file: str = "corpus/baseline_queries.json") -> dict:
    """Load baseline queries from JSON file.

    Args:
        queries_file: Path to baseline_queries.json

    Returns:
        Dictionary with 'queries' list

    Raises:
        FileNotFoundError: If queries file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    queries_path = Path(queries_file)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    with open(queries_path) as f:
        data = json.load(f)
        return data.get("queries", [])


def load_corpus(
    corpus_dir: str = "corpus/realistic_documents",
    metadata_file: str = "corpus/corpus_metadata_realistic.json",
) -> list[Document]:
    """Load documents from corpus."""
    docs_dir = Path(corpus_dir)
    metadata_path = Path(metadata_file)

    if not docs_dir.exists():
        raise FileNotFoundError(f"Corpus directory not found: {corpus_dir}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_file}")

    with open(metadata_path) as f:
        metadata = json.load(f)

    documents = []
    for doc_meta in metadata:
        doc_path = docs_dir / doc_meta["filename"]
        if doc_path.exists():
            doc = Document(
                id=doc_meta["id"],
                title=doc_meta["title"],
                content=doc_path.read_text(),
                path=str(doc_path),
            )
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
    return strategy.chunk_many(documents)


def run_retrieval(
    strategy_name: str, queries: list[dict], top_k: int = 5
) -> dict[str, list[tuple[Chunk, float]]]:
    """Run retrieval for a strategy on all queries."""
    print(f"Loading corpus...", file=sys.stderr)
    documents = load_corpus()
    print(f"  Loaded {len(documents)} documents", file=sys.stderr)

    print(
        f"Chunking documents (MarkdownSemanticStrategy, target=400 words)...",
        file=sys.stderr,
    )
    chunks = chunk_documents(documents)
    print(f"  Created {len(chunks)} chunks", file=sys.stderr)

    print(f"Initializing strategy '{strategy_name}'...", file=sys.stderr)
    if strategy_name not in RETRIEVAL_STRATEGIES:
        strategy_class = import_strategy(strategy_name)
        strategy = strategy_class(name=strategy_name)
    else:
        strategy = create_retrieval_strategy(strategy_name)

    print(f"Loading embedder (BAAI/bge-base-en-v1.5)...", file=sys.stderr)
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    if hasattr(strategy, "set_embedder"):
        strategy.set_embedder(embedder, use_prefix=True)

    print(f"Indexing {len(chunks)} chunks...", file=sys.stderr)
    strategy.index(chunks, documents)

    print(f"Retrieving for {len(queries)} queries...", file=sys.stderr)
    results: dict[str, list[tuple[Chunk, float]]] = {}
    for query in queries:
        query_id = query["id"]
        query_text = query["query"]
        retrieved = strategy.retrieve(query_text, k=top_k)
        scored_results = []
        for i, chunk in enumerate(retrieved):
            score = getattr(chunk, "score", None)
            if score is None:
                score = 1.0 / (i + 1)
            scored_results.append((chunk, score))
        results[query_id] = scored_results
        print(f"  {query_id}: retrieved {len(retrieved)} chunks", file=sys.stderr)

    return results


def import_strategy(strategy_name: str):
    """Dynamically import a retrieval strategy class."""
    SPECIAL_CASES = {
        "bm25f_hybrid": "BM25FHybridRetrieval",
    }

    if strategy_name in SPECIAL_CASES:
        class_name = SPECIAL_CASES[strategy_name]
    else:
        class_name = (
            "".join(word.capitalize() for word in strategy_name.split("_"))
            + "Retrieval"
        )
    module_name = f"retrieval.{strategy_name}"

    try:
        module = __import__(module_name, fromlist=[class_name])
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"Cannot import module '{module_name}': {e}")
    except AttributeError as e:
        raise AttributeError(
            f"Class '{class_name}' not found in module '{module_name}': {e}"
        )


def generate_individual_markdown(
    strategy_name: str,
    queries: list[dict],
    retrieved_results: Optional[dict[str, list[tuple[Chunk, float]]]] = None,
) -> str:
    """Generate markdown for individual strategy results."""
    timestamp = datetime.now().isoformat()

    lines = [
        "# Gem Strategy Test Results",
        "",
        f"**Strategy**: {strategy_name}",
        f"**Date**: {timestamp}",
        f"**Queries Tested**: {len(queries)}",
        "",
    ]

    for query in queries:
        query_id = query["id"]
        lines.extend(
            [
                f"## Query: {query_id}",
                f"**Type**: {query['type']}",
            ]
        )

        lines.extend(
            [
                "",
                f"**Query**: {query['query']}",
                "",
            ]
        )

        if "expected_answer" in query and query["expected_answer"]:
            lines.extend(
                [
                    f"**Expected Answer**: {query['expected_answer']}",
                    "",
                ]
            )

        lines.append("**Retrieved Chunks**:")

        if retrieved_results and query_id in retrieved_results:
            chunks_with_scores = retrieved_results[query_id]
            if chunks_with_scores:
                for i, (chunk, score) in enumerate(chunks_with_scores, 1):
                    lines.append(
                        f"{i}. [doc_id: {chunk.doc_id}, chunk_id: {chunk.id}, score: {score:.3f}]"
                    )
                    content_preview = chunk.content[:500].replace("\n", "\n   > ")
                    if len(chunk.content) > 500:
                        content_preview += "..."
                    lines.append(f"   > {content_preview}")
                    lines.append("")
            else:
                lines.extend(
                    [
                        "   (No chunks retrieved)",
                        "",
                    ]
                )
        else:
            lines.extend(
                [
                    "1. [doc_id: ???, chunk_id: ???]",
                    "   > (chunk content will appear here after running retrieval)",
                    "",
                ]
            )

        baseline = query.get("baseline_score", "?")
        lines.extend(
            [
                f"**Baseline Score**: {baseline}/10",
                "**New Score**: ___/10 (FILL IN)",
                "**Notes**: _______________",
                "",
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def generate_comparison_markdown(
    queries: list[dict],
    all_results: dict[str, dict[str, list[tuple[Chunk, float]]]],
) -> str:
    """Generate side-by-side comparison markdown for all strategies."""
    timestamp = datetime.now().isoformat()
    strategies = list(all_results.keys())

    lines = [
        "# Baseline Questions - Strategy Comparison",
        "",
        f"**Date**: {timestamp}",
        f"**Strategies**: {', '.join(strategies)}",
        f"**Questions**: {len(queries)}",
        "",
    ]

    for query in queries:
        query_id = query["id"]
        lines.extend(
            [
                f"## Question {query_id}: {query['query'][:80]}...",
                "",
            ]
        )

        if "expected_answer" in query and query["expected_answer"]:
            lines.extend(
                [
                    f"**Expected Answer**: {query['expected_answer']}",
                    "",
                ]
            )

        for strategy_name in strategies:
            lines.append(f"### {strategy_name}")

            if strategy_name in all_results and query_id in all_results[strategy_name]:
                chunks_with_scores = all_results[strategy_name][query_id]
                if chunks_with_scores:
                    chunk, score = chunks_with_scores[0]  # Top chunk only
                    lines.append(
                        f"**Top Chunk**: [doc_id: {chunk.doc_id}, chunk_id: {chunk.id}, score: {score:.3f}]"
                    )
                    # Preview first 200 chars
                    preview = chunk.content[:200].replace("\n", " ")
                    if len(chunk.content) > 200:
                        preview += "..."
                    lines.append(f"> {preview}")
                else:
                    lines.append("**Top Chunk**: (No chunks retrieved)")
            else:
                lines.append("**Top Chunk**: (No results)")

            lines.append("")

        lines.extend(
            [
                "---",
                "",
            ]
        )

    return "\n".join(lines)


def save_results(strategy_name: str, content: str, prefix: str = "baseline") -> Path:
    """Save results to timestamped markdown file."""
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = results_dir / f"{prefix}_{strategy_name}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(content)

    return filename


def main():
    """Run all 3 strategies and generate comparison report."""
    print("\n" + "=" * 70, file=sys.stderr)
    print("BASELINE COMPARISON TEST", file=sys.stderr)
    print("=" * 70 + "\n", file=sys.stderr)

    # Load baseline queries
    print("Loading baseline queries...", file=sys.stderr)
    queries = load_baseline_queries()
    print(f"  Loaded {len(queries)} baseline questions\n", file=sys.stderr)

    # Strategies to test
    strategies = ["adaptive_hybrid", "synthetic_variants", "enriched_hybrid_llm"]

    # Run all strategies
    all_results = {}
    for strategy_name in strategies:
        print(f"\n{'=' * 70}", file=sys.stderr)
        print(f"Testing strategy: {strategy_name}", file=sys.stderr)
        print(f"{'=' * 70}\n", file=sys.stderr)

        try:
            retrieved_results = run_retrieval(strategy_name, queries, top_k=5)
            all_results[strategy_name] = retrieved_results

            # Generate individual result file
            markdown = generate_individual_markdown(
                strategy_name, queries, retrieved_results
            )
            output_file = save_results(strategy_name, markdown, prefix="baseline")
            print(f"\n✓ Saved individual results to: {output_file}", file=sys.stderr)

        except Exception as e:
            print(f"\n✗ Error testing {strategy_name}: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc(file=sys.stderr)

    # Generate comparison report
    print(f"\n{'=' * 70}", file=sys.stderr)
    print("Generating comparison report...", file=sys.stderr)
    print(f"{'=' * 70}\n", file=sys.stderr)

    comparison_markdown = generate_comparison_markdown(queries, all_results)
    comparison_file = save_results("COMPARISON", comparison_markdown, prefix="BASELINE")
    print(f"✓ Saved comparison report to: {comparison_file}", file=sys.stderr)

    print(f"\n{'=' * 70}", file=sys.stderr)
    print("BASELINE COMPARISON COMPLETE", file=sys.stderr)
    print(f"{'=' * 70}\n", file=sys.stderr)

    print("Generated files:", file=sys.stderr)
    for strategy_name in strategies:
        if strategy_name in all_results:
            print(f"  ✓ baseline_{strategy_name}_*.md", file=sys.stderr)
    print(f"  ✓ BASELINE_COMPARISON_*.md", file=sys.stderr)
    print()


if __name__ == "__main__":
    main()
