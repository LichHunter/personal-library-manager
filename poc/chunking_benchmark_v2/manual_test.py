"""
Manual Testing Tool for RAG Retrieval Validation

This module provides explicit grading criteria and validation thresholds for
manually evaluating retrieval results from the enriched_hybrid_llm strategy.

The grading system uses a 1-10 rubric embedded in Claude Sonnet prompts to
deterministically score whether retrieved content answers the user's question.
Average scores are mapped to validation verdicts:
  - VALIDATED: >= 7.5 (strong evidence the 88.7% benchmark is accurate)
  - INCONCLUSIVE: 5.5-7.4 (mixed results, need more testing)
  - INVALIDATED: < 5.5 (evidence the benchmark claim is overstated)

This tool is part of Task 1 of the manual-testing-interface plan, which aims
to validate the 88.7% coverage claim for enriched_hybrid_llm through agent-driven
qualitative testing.

Usage:
    python manual_test.py                    # Default: agent decides question count
    python manual_test.py --questions 10     # Specific count
    python manual_test.py --output report.md # Custom output path
    python manual_test.py --help             # Show usage
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from sentence_transformers import SentenceTransformer

from retrieval import create_retrieval_strategy, EmbedderMixin
from strategies.base import Document

# ============================================================================
# GRADING RUBRIC
# ============================================================================

GRADING_RUBRIC = """
GRADING RUBRIC (1-10 Scale)

10: Perfect
    - All requested information retrieved
    - Directly answers the question
    - No irrelevant content

9: Excellent
    - Complete answer provided
    - Minor irrelevant content present
    - Core facts are clear and accurate

8: Very Good
    - Answer is present but buried in some noise
    - Requires some parsing to extract the answer
    - Supporting details may be incomplete

7: Good
    - Core answer is present
    - Missing some supporting details
    - Requires inference to fully answer the question

6: Adequate
    - Partial answer provided
    - Enough information to be useful
    - Significant gaps in coverage

5: Borderline
    - Some relevant information present
    - Misses key point or central fact
    - Requires substantial additional context

4: Poor
    - Tangentially related content only
    - Does not directly address the question
    - Requires significant interpretation

3: Very Poor
    - Mostly irrelevant content
    - Hint of the topic present
    - Unlikely to help answer the question

2: Bad
    - Almost entirely irrelevant
    - Minimal connection to the question
    - Misleading or confusing

1: Failed
    - No relevant content retrieved
    - Completely off-topic
    - Useless for answering the question
"""

# ============================================================================
# VALIDATION THRESHOLDS
# ============================================================================

VALIDATION_THRESHOLDS = {
    "VALIDATED": 7.5,  # >= 7.5: Strong evidence benchmark is accurate
    "INCONCLUSIVE": 5.5,  # 5.5-7.4: Mixed results, need more testing
    "INVALIDATED": 0.0,  # < 5.5: Evidence benchmark is overstated
}


# Verdict mapping helper
def get_verdict(average_score: float) -> str:
    """
    Map average score to validation verdict.

    Args:
        average_score: Average of all graded responses (1-10 scale)

    Returns:
        One of: "VALIDATED", "INCONCLUSIVE", "INVALIDATED"
    """
    if average_score >= VALIDATION_THRESHOLDS["VALIDATED"]:
        return "VALIDATED"
    elif average_score >= VALIDATION_THRESHOLDS["INCONCLUSIVE"]:
        return "INCONCLUSIVE"
    else:
        return "INVALIDATED"


# ============================================================================
# CORPUS LOADING AND STRATEGY INITIALIZATION
# ============================================================================


def load_corpus_documents() -> list[Document]:
    """Load corpus documents from realistic_documents directory.

    Returns:
        List of Document objects with id, title, and content
    """
    base_dir = Path(__file__).parent
    docs_dir = base_dir / "corpus" / "realistic_documents"
    metadata_path = base_dir / "corpus" / "corpus_metadata_realistic.json"

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


def initialize_strategy():
    """Initialize enriched_hybrid_llm retrieval strategy with embedder.

    Returns:
        Configured EnrichedHybridLLMRetrieval strategy
    """
    strategy = create_retrieval_strategy("enriched_hybrid_llm", name="manual_test")
    embedder = SentenceTransformer("BAAI/bge-base-en-v1.5")
    if isinstance(strategy, EmbedderMixin):
        strategy.set_embedder(embedder, use_prefix=True)
    return strategy


# ============================================================================
# CLI AND MAIN
# ============================================================================


def main():
    """Main entry point for manual testing tool."""
    parser = argparse.ArgumentParser(
        description="Manual testing tool for enriched_hybrid_llm retrieval validation"
    )
    parser.add_argument(
        "--questions",
        type=int,
        default=None,
        help="Number of questions to test (default: agent decides)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for results (default: results/manual_test_<timestamp>.md)",
    )

    args = parser.parse_args()

    # Set default output path if not provided
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(__file__).parent / "results"
        output_dir.mkdir(exist_ok=True)
        args.output = str(output_dir / f"manual_test_{timestamp}.md")

    # Start timing
    start_time = time.time()

    try:
        print("Loading corpus...")
        documents = load_corpus_documents()
        print(f"  Loaded {len(documents)} documents")

        print("Initializing strategy...")
        strategy = initialize_strategy()
        print("  Strategy initialized: enriched_hybrid_llm")

        # Placeholder for future steps
        print("\nManual testing interface ready")
        if args.questions:
            print(f"  Will test {args.questions} questions")
        print(f"  Output will be saved to: {args.output}")

        # End timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nSetup completed in {duration:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
