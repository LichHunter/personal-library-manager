#!/usr/bin/env python3
"""Test runner for gem retrieval strategies.

Loads edge case queries and runs retrieval strategies for manual grading.
Outputs markdown suitable for manual evaluation of retrieval quality.

Usage:
    python test_gems.py --strategy adaptive_hybrid
    python test_gems.py --strategy negation_aware --queries neg_001,neg_002
    python test_gems.py --strategy adaptive_hybrid --regression
    python test_gems.py --compare adaptive_hybrid,enriched_hybrid_llm
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from strategies import Document, Chunk, FixedSizeStrategy
from retrieval import create_retrieval_strategy, RETRIEVAL_STRATEGIES


def load_queries(queries_file: str = "corpus/edge_case_queries.json") -> dict:
    """Load edge case queries from JSON file.

    Args:
        queries_file: Path to edge_case_queries.json

    Returns:
        Dictionary with 'failed_queries' and 'passing_queries' lists

    Raises:
        FileNotFoundError: If queries file doesn't exist
        json.JSONDecodeError: If file is not valid JSON
    """
    queries_path = Path(queries_file)
    if not queries_path.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")

    with open(queries_path) as f:
        return json.load(f)


def load_corpus(
    corpus_dir: str = "corpus/realistic_documents",
    metadata_file: str = "corpus/corpus_metadata_realistic.json",
) -> list[Document]:
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


def chunk_documents(
    documents: list[Document], chunk_size: int = 512, overlap: int = 0
) -> list[Chunk]:
    strategy = FixedSizeStrategy(chunk_size=chunk_size, overlap=overlap)
    return strategy.chunk_many(documents)


def run_retrieval(
    strategy_name: str, queries: list[dict], top_k: int = 5
) -> dict[str, list[tuple[Chunk, float]]]:
    print(f"Loading corpus...", file=sys.stderr)
    documents = load_corpus()
    print(f"  Loaded {len(documents)} documents", file=sys.stderr)

    print(f"Chunking documents (512 tokens, no overlap)...", file=sys.stderr)
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
    """Dynamically import a retrieval strategy class.

    Converts snake_case strategy name to CamelCase class name.
    Example: 'adaptive_hybrid' -> 'AdaptiveHybridRetrieval'

    Args:
        strategy_name: Strategy name in snake_case (e.g., 'adaptive_hybrid')

    Returns:
        The strategy class

    Raises:
        ImportError: If module or class not found
        AttributeError: If class doesn't exist in module
    """
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


def generate_markdown(
    strategy_name: str,
    queries: list[dict],
    retrieved_results: Optional[dict[str, list[tuple[Chunk, float]]]] = None,
) -> str:
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

        if "root_causes" in query and query["root_causes"]:
            causes = ", ".join(query["root_causes"])
            lines.append(f"**Root Causes**: {causes}")

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


def save_results(strategy_name: str, content: str) -> Path:
    """Save results to timestamped markdown file.

    Creates results/ directory if it doesn't exist.
    Filename format: gems_[strategy]_[YYYY-MM-DD-HHMMSS].md

    Args:
        strategy_name: Name of the strategy
        content: Markdown content to save

    Returns:
        Path to the saved file
    """
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d-%H%M%S")
    filename = results_dir / f"gems_{strategy_name}_{timestamp}.md"

    with open(filename, "w") as f:
        f.write(content)

    return filename


def run_test(
    strategy_name: str, query_ids: Optional[list[str]] = None, regression: bool = False
) -> None:
    queries_data = load_queries()

    if regression:
        queries = queries_data.get("passing_queries", [])
        test_type = "regression"
    elif query_ids:
        all_queries = queries_data.get("failed_queries", []) + queries_data.get(
            "passing_queries", []
        )
        queries = [q for q in all_queries if q["id"] in query_ids]
        test_type = f"specific ({len(query_ids)} queries)"
    else:
        queries = queries_data.get("failed_queries", [])
        test_type = "failed queries"

    if not queries:
        print(f"No queries found for {test_type}", file=sys.stderr)
        sys.exit(1)

    print(f"\n{'=' * 60}", file=sys.stderr)
    print(f"Testing strategy: {strategy_name}", file=sys.stderr)
    print(f"Queries: {len(queries)} ({test_type})", file=sys.stderr)
    print(f"{'=' * 60}\n", file=sys.stderr)

    try:
        retrieved_results = run_retrieval(strategy_name, queries, top_k=5)
    except (FileNotFoundError, ImportError, ValueError) as e:
        print(f"Error running retrieval: {e}", file=sys.stderr)
        print("Falling back to template-only mode...", file=sys.stderr)
        retrieved_results = None

    markdown = generate_markdown(strategy_name, queries, retrieved_results)
    output_file = save_results(strategy_name, markdown)

    print(f"\n{'=' * 60}", file=sys.stderr)
    if retrieved_results:
        print(f"Retrieval complete for {len(queries)} {test_type}")
    else:
        print(f"Generated template for {len(queries)} {test_type}")
    print(f"Saved to: {output_file}")
    print()
    print("Next steps:")
    print("  1. Open the markdown file")
    print("  2. Grade each query (1-10 scale)")
    print("  3. Compare baseline vs new scores")


def main():
    """Parse arguments and run tests."""
    parser = argparse.ArgumentParser(
        description="Test gem retrieval strategies for manual grading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Test single strategy on all failed queries
  python test_gems.py --strategy adaptive_hybrid
  
  # Test on specific queries
  python test_gems.py --strategy negation_aware --queries neg_001,neg_002
  
  # Regression test on passing queries
  python test_gems.py --strategy adaptive_hybrid --regression
  
  # Compare two strategies
  python test_gems.py --compare adaptive_hybrid,enriched_hybrid_llm
        """,
    )

    parser.add_argument(
        "--strategy", help="Strategy name (e.g., adaptive_hybrid, enriched_hybrid_llm)"
    )
    parser.add_argument(
        "--queries",
        help="Comma-separated query IDs to test (e.g., neg_001,neg_002,mh_002)",
    )
    parser.add_argument(
        "--regression",
        action="store_true",
        help="Test on passing_queries instead of failed_queries",
    )
    parser.add_argument(
        "--compare",
        help="Compare two strategies (comma-separated, e.g., adaptive_hybrid,enriched_hybrid_llm)",
    )

    args = parser.parse_args()

    # Validate arguments
    if not args.strategy and not args.compare:
        parser.print_help()
        sys.exit(1)

    if args.strategy and args.compare:
        print("Error: Cannot use both --strategy and --compare", file=sys.stderr)
        sys.exit(1)

    # Run tests
    if args.strategy:
        query_ids = args.queries.split(",") if args.queries else None
        run_test(args.strategy, query_ids, args.regression)
    elif args.compare:
        strategies = [s.strip() for s in args.compare.split(",")]
        for strategy in strategies:
            print(f"\n{'=' * 60}")
            print(f"Testing strategy: {strategy}")
            print(f"{'=' * 60}\n")
            run_test(strategy)


if __name__ == "__main__":
    main()
