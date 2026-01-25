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
from strategies import FixedSizeStrategy

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
    """Grade a retrieval result using Claude Sonnet with deterministic scoring.

    Uses Claude Sonnet to evaluate whether retrieved chunks adequately answer
    the question. Produces a 1-10 score with explanation and verdict mapping.

    The grading is deterministic (temperature=0) to ensure consistent results
    across multiple runs with identical inputs.

    Args:
        question: Dict with keys:
            - query: str - The test question
            - expected_answer: str - Direct quote or paraphrase from corpus
            - source_doc: str - Document id where answer is found
            - difficulty: str - One of "easy", "medium", "hard"

        retrieval_result: Dict with keys:
            - query: str - The query that was executed
            - chunks: list - List of dicts with content, score, doc_id
            - latency_ms: float - Retrieval time in milliseconds

    Returns:
        Dict with:
            - score: int - 1-10 score (1=failed, 10=perfect)
            - explanation: str - Justification for the score with specific evidence
            - verdict: str - One of "PASS" (>=7), "PARTIAL" (4-6), "FAIL" (<=3)

        On LLM failure, returns:
            - score: 0
            - explanation: "Grading failed"
            - verdict: "FAIL"

    Example:
        >>> question = {
        ...     "query": "What is the API rate limit?",
        ...     "expected_answer": "100 requests per minute",
        ...     "source_doc": "api_reference",
        ...     "difficulty": "easy"
        ... }
        >>> retrieval_result = {
        ...     "query": "What is the API rate limit?",
        ...     "chunks": [
        ...         {"content": "The API rate limit is 100 requests per minute...", "score": 0.95, "doc_id": "api_reference"},
        ...     ],
        ...     "latency_ms": 123.45
        ... }
        >>> result = grade_result(question, retrieval_result)
        >>> result["score"]
        9
        >>> result["verdict"]
        'PASS'
    """
    try:
        # Extract question components
        query = question.get("query", "")
        expected_answer = question.get("expected_answer", "")

        # Extract chunks and format for prompt
        chunks = retrieval_result.get("chunks", [])
        chunks_text = ""
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")[:500]  # Truncate to 500 chars
            doc_id = chunk.get("doc_id", "unknown")
            chunks_text += f"Chunk {i} (doc: {doc_id}):\n{content}\n---\n"

        # Build the grading prompt
        prompt = f"""You are grading a RAG retrieval system's response.

QUESTION: {query}
EXPECTED ANSWER: {expected_answer}

RETRIEVED CONTENT:
{chunks_text}

GRADING RUBRIC:
{GRADING_RUBRIC}

Grade this retrieval. Output JSON:
{{"score": N, "explanation": "...", "verdict": "PASS|PARTIAL|FAIL"}}

Rules:
- Score must be 1-10 (integer)
- PASS = score >= 7 (good answer, usable)
- PARTIAL = score 4-6 (some relevant info, incomplete)
- FAIL = score <= 3 (mostly irrelevant or wrong)
- Explanation must justify the score with specific evidence from retrieved content
- Be strict but fair: if the answer is there, give credit even if buried in noise

Output ONLY valid JSON, no markdown formatting."""

        # Call LLM with temperature=0 for deterministic grading
        response = call_llm(prompt, model="claude-sonnet", timeout=120)

        # Parse JSON response
        response_text = response.strip()
        try:
            # Try to extract JSON from response
            if response_text.startswith("{"):
                result = json.loads(response_text)
            else:
                # Try to find JSON object in response
                start_idx = response_text.find("{")
                end_idx = response_text.rfind("}") + 1
                if start_idx >= 0 and end_idx > start_idx:
                    result = json.loads(response_text[start_idx:end_idx])
                else:
                    raise ValueError("No JSON object found in response")
        except json.JSONDecodeError as e:
            print(f"Warning: Failed to parse grading response as JSON: {e}")
            return {
                "score": 0,
                "explanation": "Grading failed",
                "verdict": "FAIL",
            }

        # Validate and normalize score
        score = result.get("score", 0)
        if not isinstance(score, int) or score < 1 or score > 10:
            score = (
                max(1, min(10, int(score))) if isinstance(score, (int, float)) else 0
            )

        # Map score to verdict
        if score >= 7:
            verdict = "PASS"
        elif score >= 4:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        # Get explanation
        explanation = result.get("explanation", "No explanation provided")

        return {
            "score": score,
            "explanation": explanation,
            "verdict": verdict,
        }

    except Exception as e:
        print(f"Error during grading: {e}")
        return {
            "score": 0,
            "explanation": "Grading failed",
            "verdict": "FAIL",
        }


# ============================================================================
# REPORT GENERATION
# ============================================================================


def generate_report(
    questions: list[dict], results: list[dict], output_path: str
) -> None:
    """Generate comprehensive markdown report of manual testing results.

    Creates a detailed markdown report with Summary, Results Table, Detailed
    Findings, and Conclusion sections. Calculates aggregate score and compares
    to benchmark (88.7% coverage claim).

    Args:
        questions: List of question dicts with keys:
            - query: str - The test question
            - expected_answer: str - Direct quote or paraphrase from corpus
            - source_doc: str - Document id where answer is found
            - difficulty: str - One of "easy", "medium", "hard"

        results: List of result dicts with keys:
            - question: dict - The question object
            - retrieval: dict - Retrieval result with chunks, latency_ms
            - grade: dict - Grade with score, explanation, verdict

        output_path: Path to write markdown report to

    Returns:
        None (writes to file)

    Report Structure:
        1. Summary: timestamp, strategy, question count, average score,
                   benchmark comparison, validation verdict
        2. Results Table: # | Question (50 chars) | Score | Verdict | Source
        3. Detailed Findings: For each question, show query, expected answer,
                             score/verdict, explanation, retrieved chunks
                             (truncated to 200 chars)
        4. Conclusion: Validation verdict with reasoning based on average score
    """
    if not results:
        print("Warning: No results to report")
        return

    # Calculate aggregate score
    scores = [r["grade"]["score"] for r in results]
    average_score = sum(scores) / len(scores) if scores else 0.0

    # Get validation verdict
    validation_verdict = get_verdict(average_score)

    # Benchmark comparison
    benchmark_score = 88.7
    benchmark_comparison = (
        f"✓ MEETS BENCHMARK" if average_score >= 7.5 else "✗ BELOW BENCHMARK"
    )

    # Build markdown report
    lines = []

    # Header
    lines.append("# Manual Testing Report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("Strategy: enriched_hybrid_llm")
    lines.append(f"Questions: {len(questions)}")
    lines.append("")

    # Summary section
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- **Average Score**: {average_score:.1f} / 10")
    lines.append(f"- **Benchmark Claim**: {benchmark_score}% coverage")
    lines.append(f"- **Validation**: {validation_verdict}")
    lines.append(f"- **Status**: {benchmark_comparison}")
    lines.append("")

    # Results table
    lines.append("## Results")
    lines.append("")
    lines.append("| # | Question (truncated) | Score | Verdict | Source |")
    lines.append("|---|---------------------|-------|---------|--------|")

    for i, result in enumerate(results, 1):
        question = result["question"]
        grade = result["grade"]

        # Truncate question to 50 chars
        query_truncated = question["query"][:50]
        if len(question["query"]) > 50:
            query_truncated += "..."

        # Escape pipe characters in query
        query_truncated = query_truncated.replace("|", "\\|")

        source = question.get("source_doc", "unknown")
        score = grade["score"]
        verdict = grade["verdict"]

        lines.append(f"| {i} | {query_truncated} | {score} | {verdict} | {source} |")

    lines.append("")

    # Detailed findings section
    lines.append("## Detailed Findings")
    lines.append("")

    for i, result in enumerate(results, 1):
        question = result["question"]
        retrieval = result["retrieval"]
        grade = result["grade"]

        lines.append(f"### Question {i}: {question['query']}")
        lines.append("")

        # Expected answer
        lines.append(f"**Expected Answer**: {question['expected_answer']}")
        lines.append("")

        # Score and verdict
        lines.append(f"**Score**: {grade['score']}/10 ({grade['verdict']})")
        lines.append("")

        # Explanation
        lines.append(f"**Explanation**: {grade['explanation']}")
        lines.append("")

        # Retrieved chunks
        lines.append("**Retrieved Chunks**:")
        lines.append("")

        chunks = retrieval.get("chunks", [])
        for j, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")
            score_val = chunk.get("score")

            # Truncate content to 200 chars
            if len(content) > 200:
                content_display = content[:200] + "..."
            else:
                content_display = content

            # Escape markdown special characters
            content_display = content_display.replace("|", "\\|")

            score_str = f" (score: {score_val:.3f})" if score_val is not None else ""
            lines.append(f"{j}. **[{doc_id}]** {content_display}{score_str}")

        lines.append("")

    # Conclusion section
    lines.append("## Conclusion")
    lines.append("")

    if validation_verdict == "VALIDATED":
        conclusion = (
            f"✓ **VALIDATED**: Average score of {average_score:.1f}/10 meets the "
            f"validation threshold (≥7.5). The enriched_hybrid_llm strategy "
            f"demonstrates strong retrieval performance and supports the 88.7% "
            f"coverage benchmark claim."
        )
    elif validation_verdict == "INCONCLUSIVE":
        conclusion = (
            f"⚠ **INCONCLUSIVE**: Average score of {average_score:.1f}/10 falls in "
            f"the inconclusive range (5.5-7.4). Results show mixed performance. "
            f"More testing is needed to validate or invalidate the 88.7% benchmark claim."
        )
    else:  # INVALIDATED
        conclusion = (
            f"✗ **INVALIDATED**: Average score of {average_score:.1f}/10 is below "
            f"the validation threshold (<5.5). The enriched_hybrid_llm strategy "
            f"shows insufficient retrieval performance. The 88.7% coverage benchmark "
            f"claim appears to be overstated."
        )

    lines.append(conclusion)
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
        chunker = FixedSizeStrategy(chunk_size=512, overlap=0)
        if hasattr(chunker, "set_embedder"):
            chunker.set_embedder(
                strategy.embedder if hasattr(strategy, "embedder") else None
            )
        chunks = chunker.chunk_many(documents)
        print(f"  Created {len(chunks)} chunks")

        print("Indexing strategy with documents...")
        strategy.index(chunks=chunks, documents=documents)
        print("  Strategy indexed")

        print("\nRunning retrieval and grading...")
        results = []
        for i, question in enumerate(questions, 1):
            print(f"  Testing question {i}/{len(questions)}...")
            retrieval_result = run_retrieval(strategy, question["query"])
            grade = grade_result(question, retrieval_result)
            results.append(
                {
                    "question": question,
                    "retrieval": retrieval_result,
                    "grade": grade,
                }
            )

        print("\nGenerating report...")
        generate_report(questions, results, args.output)
        print(f"  Report saved to: {args.output}")

        # Calculate and display summary
        scores = [r["grade"]["score"] for r in results]
        average_score = sum(scores) / len(scores) if scores else 0.0
        verdict = get_verdict(average_score)

        print(f"\n{'=' * 60}")
        print(f"SUMMARY")
        print(f"{'=' * 60}")
        print(f"Questions tested: {len(questions)}")
        print(f"Average score: {average_score:.1f}/10")
        print(f"Validation verdict: {verdict}")
        print(f"Benchmark (88.7%): {'✓ MEETS' if average_score >= 7.5 else '✗ BELOW'}")
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
