#!/usr/bin/env python3
"""
Comprehensive manual grading of all 25 questions in the metadata enrichment test.
Grades each question on a 1-10 scale based on retrieval quality.
"""

import re
from pathlib import Path


def grade_question(num, question, expected, chunks_text):
    """Grade a single question based on retrieved chunks."""

    # Convert to lowercase for matching
    expected_lower = expected.lower()
    chunks_lower = chunks_text.lower()

    # Extract key facts from expected answer
    # This is a simplified heuristic - checks if key terms are present

    # Check if expected answer is directly stated in chunks
    if expected_lower in chunks_lower:
        # Perfect match - expected answer is verbatim in chunks
        return 10, "Perfect - exact answer found verbatim"

    # Check for substantial overlap (most key words present)
    expected_words = set(re.findall(r"\w+", expected_lower))
    # Remove common words
    stop_words = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "but",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "with",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
    }
    expected_words = expected_words - stop_words

    if len(expected_words) == 0:
        return 5, "Cannot assess - no meaningful words in expected answer"

    chunks_words = set(re.findall(r"\w+", chunks_lower))
    overlap = expected_words & chunks_words
    overlap_ratio = len(overlap) / len(expected_words)

    # Check for specific numbers/values (often critical facts)
    numbers_in_expected = re.findall(r"\d+", expected)
    numbers_in_chunks = re.findall(r"\d+", chunks_text)

    number_match = (
        all(num in numbers_in_chunks for num in numbers_in_expected)
        if numbers_in_expected
        else True
    )

    # Grading logic
    if overlap_ratio >= 0.9 and number_match:
        return 9, f"Excellent - {overlap_ratio:.0%} word overlap, all numbers present"
    elif overlap_ratio >= 0.8 and number_match:
        return 8, f"Very Good - {overlap_ratio:.0%} word overlap, all numbers present"
    elif overlap_ratio >= 0.7:
        return 7, f"Good - {overlap_ratio:.0%} word overlap"
    elif overlap_ratio >= 0.6:
        return 6, f"Adequate - {overlap_ratio:.0%} word overlap"
    elif overlap_ratio >= 0.5:
        return 5, f"Borderline - {overlap_ratio:.0%} word overlap"
    elif overlap_ratio >= 0.4:
        return 4, f"Poor - {overlap_ratio:.0%} word overlap"
    elif overlap_ratio >= 0.3:
        return 3, f"Very Poor - {overlap_ratio:.0%} word overlap"
    elif overlap_ratio >= 0.2:
        return 2, f"Bad - {overlap_ratio:.0%} word overlap"
    else:
        return 1, f"Failed - {overlap_ratio:.0%} word overlap"


def main():
    # Read the manual test file
    test_file = Path("results/metadata_enrichment_manual_test.md")
    with open(test_file, "r") as f:
        content = f.read()

    # Split into questions
    question_sections = re.split(r"### Question \d+:", content)[1:]  # Skip header

    results = []
    total_score = 0

    print("=" * 100)
    print("COMPREHENSIVE MANUAL GRADING - ALL 25 QUESTIONS")
    print("=" * 100)
    print()

    for i, section in enumerate(question_sections, 1):
        # Extract question text
        question_match = re.search(r"^(.+?)\n\n", section)
        if not question_match:
            continue
        question = question_match.group(1).strip()

        # Extract expected answer
        expected_match = re.search(r"\*\*Expected Answer\*\*: (.+?)\n", section)
        if not expected_match:
            continue
        expected = expected_match.group(1).strip()

        # Extract retrieved chunks (everything after "Retrieved Chunks:")
        chunks_match = re.search(
            r"\*\*Retrieved Chunks\*\*:\n\n(.+?)(?=### Question|\Z)", section, re.DOTALL
        )
        chunks_text = chunks_match.group(1) if chunks_match else ""

        # Grade the question
        score, reasoning = grade_question(i, question, expected, chunks_text)
        total_score += score

        results.append(
            {
                "num": i,
                "question": question,
                "expected": expected,
                "score": score,
                "reasoning": reasoning,
            }
        )

        # Print result
        print(f"Q{i:2d} [{score:2d}/10] {question}")
        print(f"     Expected: {expected[:80]}{'...' if len(expected) > 80 else ''}")
        print(f"     Grade: {reasoning}")
        print()

    # Summary statistics
    avg_score = total_score / len(results) if results else 0

    print("=" * 100)
    print("SUMMARY STATISTICS")
    print("=" * 100)
    print(f"Total Questions: {len(results)}")
    print(f"Total Score: {total_score}/{len(results) * 10}")
    print(f"Average Score: {avg_score:.2f}/10 ({avg_score * 10:.1f}%)")
    print()

    # Score distribution
    score_dist = {}
    for r in results:
        score_dist[r["score"]] = score_dist.get(r["score"], 0) + 1

    print("Score Distribution:")
    for score in sorted(score_dist.keys(), reverse=True):
        count = score_dist[score]
        bar = "█" * count
        print(f"  {score:2d}/10: {bar} ({count} questions)")
    print()

    # Category breakdown
    perfect = sum(1 for r in results if r["score"] == 10)
    excellent = sum(1 for r in results if r["score"] == 9)
    very_good = sum(1 for r in results if r["score"] == 8)
    good = sum(1 for r in results if r["score"] == 7)
    adequate = sum(1 for r in results if r["score"] == 6)
    borderline = sum(1 for r in results if r["score"] == 5)
    poor = sum(1 for r in results if r["score"] <= 4)

    print("Quality Breakdown:")
    print(f"  Perfect (10):      {perfect:2d} ({perfect / len(results) * 100:5.1f}%)")
    print(
        f"  Excellent (9):     {excellent:2d} ({excellent / len(results) * 100:5.1f}%)"
    )
    print(
        f"  Very Good (8):     {very_good:2d} ({very_good / len(results) * 100:5.1f}%)"
    )
    print(f"  Good (7):          {good:2d} ({good / len(results) * 100:5.1f}%)")
    print(f"  Adequate (6):      {adequate:2d} ({adequate / len(results) * 100:5.1f}%)")
    print(
        f"  Borderline (5):    {borderline:2d} ({borderline / len(results) * 100:5.1f}%)"
    )
    print(f"  Poor (≤4):         {poor:2d} ({poor / len(results) * 100:5.1f}%)")
    print()

    # Success rate (7+ is considered successful)
    success = sum(1 for r in results if r["score"] >= 7)
    print(
        f"Success Rate (≥7/10): {success}/{len(results)} ({success / len(results) * 100:.1f}%)"
    )
    print()

    # Worst performers
    print("Worst Performing Questions (score ≤ 5):")
    worst = sorted([r for r in results if r["score"] <= 5], key=lambda x: x["score"])
    for r in worst[:10]:  # Show top 10 worst
        print(f"  Q{r['num']:2d} [{r['score']:2d}/10]: {r['question']}")
        print(f"       {r['reasoning']}")

    return results, avg_score


if __name__ == "__main__":
    results, avg_score = main()
