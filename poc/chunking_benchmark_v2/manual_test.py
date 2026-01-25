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

from enrichment.provider import call_llm
from retrieval import create_retrieval_strategy, EmbedderMixin
from strategies.base import Document
from strategies import MarkdownSemanticStrategy

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
# QUESTION GENERATION
# ============================================================================


def generate_questions(
    documents: list[Document], num_questions: Optional[int] = None
) -> list[dict]:
    """Generate test questions from corpus documents using Claude.

    Uses Claude Haiku to read corpus documents and generate human-like test
    questions with expected answers grounded in the corpus. Each question
    includes the query text, expected answer (verified to exist in corpus),
    source document, and difficulty level.

    The function automatically determines the number of questions based on
    document coverage if num_questions is None (max 15 questions).

    Args:
        documents: List of Document objects with id, title, and content
        num_questions: Number of questions to generate. If None, agent decides
                      based on document coverage (max 15). Must be <= 15.

    Returns:
        List of question dicts with keys:
        - query: str - The test question
        - expected_answer: str - Direct quote or paraphrase from corpus
        - source_doc: str - Document id where answer is found
        - difficulty: str - One of "easy", "medium", "hard"

        Returns empty list if LLM generation fails.

    Example:
        >>> docs = load_corpus_documents()
        >>> questions = generate_questions(docs, num_questions=5)
        >>> len(questions)
        5
        >>> questions[0].keys()
        dict_keys(['query', 'expected_answer', 'source_doc', 'difficulty'])
    """
    if not documents:
        print("Warning: No documents provided to generate_questions()")
        return []

    # Load style guide from ground truth
    base_dir = Path(__file__).parent
    ground_truth_path = base_dir / "corpus" / "ground_truth_realistic.json"

    try:
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        print(f"Warning: Could not load ground truth from {ground_truth_path}")
        ground_truth = {"queries": []}

    # Extract sample queries for style guide (first 3 queries, all dimensions)
    style_guide_samples = []
    for query_obj in ground_truth.get("queries", [])[:3]:
        original = query_obj.get("original_query", "")
        if original:
            style_guide_samples.append(f"- Original: {original}")
            for human_query in query_obj.get("human_queries", [])[:2]:
                dimension = human_query.get("dimension", "")
                query = human_query.get("query", "")
                if dimension and query:
                    style_guide_samples.append(f"  {dimension}: {query}")

    style_guide_text = "\n".join(style_guide_samples) if style_guide_samples else ""

    # Determine number of questions
    if num_questions is None:
        # Agent decides based on document coverage: 2-3 questions per document, max 15
        num_questions = min(len(documents) * 2, 15)
    else:
        num_questions = min(num_questions, 15)

    # Build document content string with clear separators
    doc_contents = []
    for doc in documents:
        doc_contents.append(
            f"=== DOCUMENT: {doc.title} (id: {doc.id}) ===\n{doc.content}\n"
        )

    documents_text = "\n".join(doc_contents)

    # Build the prompt
    prompt = f"""You are testing a RAG retrieval system. Read these documents carefully and generate {num_questions} test questions.

DOCUMENTS:
{documents_text}

STYLE GUIDE (from existing queries - shows different question dimensions):
{style_guide_text}

For each question, output a JSON object with these fields:
- query: The test question (string)
- expected_answer: A direct quote or paraphrase from the documents (string)
- source_doc: The document id where the answer is found (string)
- difficulty: One of "easy" (direct lookup), "medium" (requires inference), or "hard" (cross-document or complex reasoning)

Rules:
1. Questions MUST have answers that exist IN the documents (verify before including)
2. Mix difficulty levels: aim for roughly equal distribution
3. Cover all documents proportionally
4. Use realistic human phrasing: casual, problem-oriented, technical
5. Expected answer must be a direct quote or close paraphrase from the document
6. Verify each answer exists in the document before including the question
7. Do NOT generate questions about topics not in the corpus

Output ONLY a valid JSON array of objects, no markdown formatting, no explanation.
Example format:
[
  {{"query": "What is the API rate limit?", "expected_answer": "100 requests per minute", "source_doc": "api_reference", "difficulty": "easy"}},
  {{"query": "How do I deploy to production?", "expected_answer": "Use the deployment guide...", "source_doc": "deployment_guide", "difficulty": "medium"}}
]"""

    try:
        response = call_llm(prompt, model="claude-haiku", timeout=90)
    except Exception as e:
        print(f"Warning: LLM generation failed: {e}")
        return []

    # Parse JSON response
    try:
        # Try to extract JSON from response (in case there's extra text)
        response_text = response.strip()
        if response_text.startswith("["):
            questions = json.loads(response_text)
        else:
            # Try to find JSON array in response
            start_idx = response_text.find("[")
            end_idx = response_text.rfind("]") + 1
            if start_idx >= 0 and end_idx > start_idx:
                questions = json.loads(response_text[start_idx:end_idx])
            else:
                print("Warning: Could not find JSON array in LLM response")
                return []
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse LLM response as JSON: {e}")
        return []

    # Validate each question has required fields
    validated_questions = []
    for q in questions:
        if isinstance(q, dict) and all(
            key in q for key in ["query", "expected_answer", "source_doc", "difficulty"]
        ):
            # Validate difficulty is one of the allowed values
            if q.get("difficulty") in ["easy", "medium", "hard"]:
                validated_questions.append(q)

    return validated_questions


# ============================================================================
# RETRIEVAL EXECUTION
# ============================================================================


def run_retrieval(strategy, query: str, k: int = 5) -> dict:
    """Execute retrieval and capture results with timing.

    Executes a retrieval query against the initialized strategy, measures
    latency with high precision, and extracts chunk data including content,
    score (if available), and document ID.

    Args:
        strategy: Initialized retrieval strategy (enriched_hybrid_llm)
        query: Query string to search for
        k: Number of chunks to retrieve (default: 5)

    Returns:
        Dict with:
            - query (str): The input query
            - chunks (list): List of dicts with content, score, doc_id
            - latency_ms (float): Retrieval time in milliseconds

    Example:
        >>> strategy = initialize_strategy()
        >>> result = run_retrieval(strategy, "What is CloudFlow?", k=5)
        >>> len(result["chunks"])
        5
        >>> result["chunks"][0].keys()
        dict_keys(['content', 'score', 'doc_id'])
        >>> result["latency_ms"] > 0
        True
    """
    # Start high-precision timer
    start = time.perf_counter()

    # Execute retrieval
    retrieved = strategy.retrieve(query, k=k)

    # Calculate latency in milliseconds
    latency_ms = (time.perf_counter() - start) * 1000

    # Extract chunk data
    chunks = []
    for chunk in retrieved:
        chunk_data = {
            "content": chunk.content,
            "score": getattr(chunk, "score", None),
            "doc_id": chunk.doc_id,
        }
        chunks.append(chunk_data)

    return {
        "query": query,
        "chunks": chunks,
        "latency_ms": latency_ms,
    }


# ============================================================================
# GRADING LOGIC
# ============================================================================


def grade_result(question: dict, retrieval_result: dict) -> dict:
    """Stub function - manual grading is now performed by humans.

    This function is kept for backward compatibility but no longer performs
    automated LLM grading. Manual grading instructions are included in the
    generated report instead.

    Args:
        question: Dict with question data (unused)
        retrieval_result: Dict with retrieval result (unused)

    Returns:
        Empty dict (grading is now manual)
    """
    return {}


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(
    questions: list[dict], results: list[dict], output_path: str
) -> None:
    """Generate manual grading template for human evaluation.

    Creates a markdown template with questions, expected answers, and retrieved
    chunks for manual human grading. Includes grading rubric and instructions.

    Args:
        questions: List of question dicts with keys:
            - query: str - The test question
            - expected_answer: str - Direct quote or paraphrase from corpus
            - source_doc: str - Document id where answer is found
            - difficulty: str - One of "easy", "medium", "hard"

        results: List of result dicts with keys:
            - question: dict - The question object
            - retrieval: dict - Retrieval result with chunks, latency_ms

        output_path: Path to write markdown report to

    Returns:
        None (writes to file)
    """
    if not results:
        print("Warning: No results to report")
        return

    # Build markdown report
    lines = []

    # Header
    lines.append("# Manual Testing Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Strategy: enriched_hybrid_llm")
    lines.append(f"Questions: {len(questions)}")
    lines.append("")

    # Instructions section
    lines.append("## Instructions")
    lines.append("")
    lines.append("For each question below, manually grade the retrieved chunks:")
    lines.append("")
    lines.append("1. Read the **QUESTION**")
    lines.append("2. Read the **EXPECTED ANSWER**")
    lines.append("3. Review the **RETRIEVED CHUNKS**")
    lines.append("4. Assign a score 1-10 using the rubric below")
    lines.append("")

    # Grading rubric
    lines.append("### Grading Rubric")
    lines.append("")
    lines.append(GRADING_RUBRIC)
    lines.append("")

    # Questions section
    lines.append("## Questions")
    lines.append("")

    for i, result in enumerate(results, 1):
        question = result["question"]
        retrieval = result["retrieval"]

        lines.append(f"### Question {i}: {question['query']}")
        lines.append("")

        # Expected answer
        lines.append(f"**Expected Answer**: {question['expected_answer']}")
        lines.append("")

        # Grading fields
        lines.append("**Your Score**: ___ / 10")
        lines.append("")
        lines.append("**Your Notes**: ")
        lines.append("")

        # Retrieved chunks
        lines.append("**Retrieved Chunks**:")
        lines.append("")

        chunks = retrieval.get("chunks", [])
        for j, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")

            # Escape markdown special characters
            content_display = content.replace("|", "\\|")

            lines.append(f"{j}. **[{doc_id}]**")
            lines.append("")
            lines.append(f"   {content_display}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # Write to file
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines))


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

        print("\nGenerating test questions...")
        questions = generate_questions(documents, args.questions)
        print(f"  Generated {len(questions)} questions")

        print("Creating chunks...")
        chunker = MarkdownSemanticStrategy(
            max_heading_level=4,  # Split on h1-h4
            target_chunk_size=400,  # Target ~400 words per chunk
            min_chunk_size=50,  # Merge tiny sections
            max_chunk_size=800,  # Split very large sections
            overlap_sentences=1,  # 1 sentence overlap
        )
        if hasattr(chunker, "set_embedder"):
            chunker.set_embedder(
                strategy.embedder if hasattr(strategy, "embedder") else None
            )
        chunks = chunker.chunk_many(documents)
        print(f"  Created {len(chunks)} chunks using {chunker.name}")

        print("Indexing strategy with documents...")
        strategy.index(chunks=chunks, documents=documents)
        print("  Strategy indexed")

        print("\nRunning retrieval...")
        results = []
        for i, question in enumerate(questions, 1):
            print(f"  Testing question {i}/{len(questions)}...")
            retrieval_result = run_retrieval(strategy, question["query"])
            results.append(
                {
                    "question": question,
                    "retrieval": retrieval_result,
                }
            )

        print("\nGenerating report...")
        generate_report(questions, results, args.output)
        print(f"  Report saved to: {args.output}")

        print(f"\n{'=' * 60}")
        print(f"MANUAL GRADING TEMPLATE GENERATED")
        print(f"{'=' * 60}")
        print(f"Questions: {len(questions)}")
        print(f"Output: {args.output}")
        print(f"\nNext steps:")
        print(f"1. Open the report file")
        print(f"2. For each question, review the retrieved chunks")
        print(f"3. Assign a score 1-10 using the provided rubric")
        print(f"4. Add notes explaining your score")
        print(f"{'=' * 60}")

        # End timing
        end_time = time.time()
        duration = end_time - start_time
        print(f"\nCompleted in {duration:.2f}s")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
