"""Retrieval quality grading component for modular retrieval pipeline.

This module provides retrieval quality grading functionality using Claude Sonnet.
It evaluates whether retrieved chunks contain sufficient information to answer
a user's question, with configurable timeout handling.

The component is stateless (no stored LLM client) and provides a standalone
utility for grading retrieval quality in RAG pipelines.

Example:
    >>> from poc.modular_retrieval_pipeline.components.retrieval_grader import RetrievalGrader, GradeResult
    >>> grader = RetrievalGrader(timeout=30.0)
    >>> chunks = [{"content": "...", "doc_id": "doc1"}]
    >>> result = grader.grade("What is X?", "X is...", chunks)
    >>> print(result.grade)  # 8 (1-10 scale)
"""

import json
import re
import time
from dataclasses import dataclass
from typing import Optional

from ..utils.llm_provider import call_llm
from ..utils.logger import get_logger


GRADING_PROMPT = """You are an impartial judge evaluating retrieval quality for a documentation search system.

USER QUESTION: {question}

EXPECTED ANSWER (ground truth): {expected_answer}

RETRIEVED CHUNKS:
---
{chunks_text}
---

TASK: Determine if the retrieved chunks contain sufficient information to answer the user's question correctly.

EVALUATION CRITERIA:
1. Does the retrieved content contain the key facts from the expected answer?
2. Would a user be able to solve their problem using ONLY these chunks?
3. Ignore style, formatting, or verbosity - focus on factual completeness.

SCORING GUIDE:
- 10: PERFECT - Chunks contain the complete answer with all necessary details
- 8-9: EXCELLENT - Chunks contain the core answer, minor details may be missing
- 6-7: GOOD - Chunks contain most relevant information, user could likely solve their problem
- 4-5: PARTIAL - Chunks have some relevant info but missing key details needed to fully answer
- 2-3: POOR - Chunks are tangentially related but don't address the actual question
- 1: IRRELEVANT - Chunks have no useful information for this question

IMPORTANT:
- Compare chunks against the EXPECTED ANSWER to verify factual coverage
- A chunk mentioning the topic is NOT enough - it must contain actionable information
- Grade based on whether the user could SOLVE THEIR PROBLEM, not just "learn something"

First, analyze what key facts from the expected answer appear in the chunks.
Then provide your grade.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"grade": <integer 1-10>, "reasoning": "<which key facts are present/missing>"}}"""


@dataclass
class GradeResult:
    """Result of retrieval quality grading.

    Attributes:
        grade: Numeric grade 1-10 (or None if grading failed)
        reasoning: Explanation of the grade (or None if grading failed)
        latency_ms: Time taken to grade in milliseconds
    """

    grade: Optional[int]
    reasoning: Optional[str]
    latency_ms: float


class RetrievalGrader:
    """Grades retrieval quality using Claude Sonnet.

    Evaluates whether retrieved chunks contain sufficient information to answer
    a user's question by comparing against ground truth answers.

    The component is stateless: no LLM client is stored as an instance variable.
    Each call to grade() independently invokes the LLM provider.

    Features:
    - Accepts question, expected answer, and chunks
    - Returns GradeResult with grade (1-10), reasoning, and latency
    - Configurable timeout (default 30.0 seconds)
    - Graceful error handling (returns None on failure)
    - Pure function interface (no state mutation)

    Example:
        >>> grader = RetrievalGrader(timeout=30.0)
        >>> chunks = [{"content": "...", "doc_id": "doc1"}]
        >>> result = grader.grade("What is X?", "X is...", chunks)
        >>> assert isinstance(result, GradeResult)
        >>> assert result.grade is not None or result.latency_ms > 0
    """

    def __init__(self, timeout: float = 30.0):
        """Initialize RetrievalGrader component.

        Args:
            timeout: Timeout in seconds for LLM call (default 30.0).
                    If the LLM takes longer than this, a GradeResult with
                    None values is returned.
        """
        self.timeout = timeout
        self._log = get_logger()

    def grade(
        self, question: str, expected_answer: str, chunks: list[dict]
    ) -> GradeResult:
        """Grade retrieval quality using Claude Sonnet.

        Evaluates whether the retrieved chunks contain sufficient information
        to answer the user's question by comparing against the expected answer.

        Args:
            question: User's question
            expected_answer: Ground truth answer to compare against
            chunks: List of retrieved chunk dicts with 'content' and 'doc_id' fields

        Returns:
            GradeResult with:
            - grade: Numeric score 1-10 (or None on failure)
            - reasoning: Explanation of the grade (or None on failure)
            - latency_ms: Time taken to grade in milliseconds
        """
        self._log.debug(f"[retrieval-grader] Grading question: {question[:50]}...")
        self._log.trace(f"[retrieval-grader] Expected: {expected_answer[:100]}...")
        self._log.trace(
            f"[retrieval-grader] Chunks ({len(chunks)}): {[c.get('doc_id', 'unknown') for c in chunks]}"
        )

        start_time = time.time()

        try:
            # Format chunks into text
            chunks_text = self._format_chunks(chunks)

            # Build prompt
            prompt = GRADING_PROMPT.format(
                question=question,
                expected_answer=expected_answer,
                chunks_text=chunks_text,
            )

            # Call LLM
            response = call_llm(
                prompt, model="claude-sonnet", timeout=int(self.timeout)
            )
            elapsed = time.time() - start_time

            # Parse response
            if not response or not response.strip():
                self._log.debug(f"[retrieval-grader] Empty response in {elapsed:.3f}s")
                return GradeResult(
                    grade=None, reasoning=None, latency_ms=elapsed * 1000
                )

            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response_clean = self._strip_markdown_json(response)
                    data = json.loads(response_clean)
                    grade = data.get("grade")
                    reasoning = data.get("reasoning")

                    if grade is not None:
                        if isinstance(grade, (int, float)):
                            grade = int(grade)
                            if grade < 1:
                                grade = 1
                            elif grade > 10:
                                grade = 10
                        else:
                            grade = None

                    self._log.info(
                        f"[retrieval-grader] Grade={grade}/10 - {reasoning[:80] if reasoning else 'N/A'}..."
                    )
                    return GradeResult(
                        grade=grade, reasoning=reasoning, latency_ms=elapsed * 1000
                    )

                except json.JSONDecodeError as e:
                    self._log.debug(
                        f"[retrieval-grader] Raw response: {response[:200]}..."
                    )
                    if attempt < max_retries - 1:
                        self._log.debug(
                            f"[retrieval-grader] Parse fail attempt {attempt + 1}/{max_retries}, retrying..."
                        )
                        continue
                    else:
                        self._log.warn(
                            f"[retrieval-grader] All {max_retries} attempts failed"
                        )
                        return GradeResult(
                            grade=None, reasoning=None, latency_ms=elapsed * 1000
                        )

        except Exception as e:
            elapsed = time.time() - start_time
            self._log.warn(
                f"[retrieval-grader] ERROR after {elapsed:.3f}s: {type(e).__name__}: {e}"
            )
            return GradeResult(grade=None, reasoning=None, latency_ms=elapsed * 1000)

    def _strip_markdown_json(self, text: str) -> str:
        """Strip markdown code blocks from JSON response.

        Handles responses wrapped in ```json ... ``` blocks.

        Args:
            text: Raw response text that may contain markdown wrappers

        Returns:
            Clean JSON string with markdown removed
        """
        text = re.sub(r"^```json\s*", "", text.strip())
        text = re.sub(r"\s*```$", "", text.strip())
        return text

    def _format_chunks(self, chunks: list[dict]) -> str:
        """Format chunks into readable text for the prompt.

        Args:
            chunks: List of chunk dicts with 'content' and 'doc_id' fields

        Returns:
            Formatted chunks text with numbering and doc IDs
        """
        if not chunks:
            return "(No chunks retrieved)"

        formatted_chunks = []
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get("content", "")
            doc_id = chunk.get("doc_id", "unknown")
            formatted_chunks.append(f"[Chunk {i} - {doc_id}]\n{content}")

        return "\n\n".join(formatted_chunks)
