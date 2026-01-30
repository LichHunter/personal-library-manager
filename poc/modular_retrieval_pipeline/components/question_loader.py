"""Question loader component for benchmark datasets.

This module provides a stateless component that loads benchmark questions from JSON files.
It validates required fields and returns a flat list of question dictionaries.

Field Contracts:
    REQUIRED fields (must be present in every question):
        - question (str): The query text
        - expected_answer (str): Ground truth for grading
        - doc_id (str): Target document ID

    OPTIONAL fields (may be present):
        - id (str): Question identifier
        - difficulty (str): e.g., "easy", "medium", "hard"
        - type (str): e.g., "factual", "comparison"
        - section (str): Document section reference
        - quality_score (float): Quality rating 0-1
        - source (str): Origin of question
        - variant (str): Variant identifier (e.g., "q1", "q2")

Example:
    >>> from question_loader import load_questions
    >>> questions = load_questions("questions.json")
    >>> print(len(questions))
    20
    >>> print(questions[0].keys())
    dict_keys(['question', 'expected_answer', 'doc_id', 'id', 'quality_score'])

Expected JSON file structure:
    {
        "questions": [
            {
                "question": "What is a ServiceAccount in Kubernetes?",
                "expected_answer": "A ServiceAccount is...",
                "doc_id": "tasks_configure-pod-container_configure-service-account",
                "id": "q_024_q1",
                "quality_score": 1.0,
                "variant": "informed_q1"
            }
        ]
    }
"""

import json
from pathlib import Path


def load_questions(filepath: str) -> list[dict]:
    """Load benchmark questions from JSON file.

    Loads questions from a JSON file with the structure:
    {
        "questions": [
            {
                "question": str,
                "expected_answer": str,
                "doc_id": str,
                ... optional fields ...
            }
        ]
    }

    Validates that all required fields are present in each question.

    Args:
        filepath: Path to JSON file containing questions

    Returns:
        List of question dictionaries with validated required fields

    Raises:
        FileNotFoundError: If filepath does not exist
        json.JSONDecodeError: If file is not valid JSON
        ValueError: If required fields are missing from any question
        KeyError: If "questions" key is missing from JSON

    Example:
        >>> questions = load_questions("benchmark_questions.json")
        >>> assert len(questions) > 0
        >>> assert all("question" in q for q in questions)
    """
    # Load JSON file
    filepath_obj = Path(filepath)
    if not filepath_obj.exists():
        raise FileNotFoundError(f"Questions file not found: {filepath}")

    try:
        with open(filepath_obj) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in {filepath}: {e.msg}",
            e.doc,
            e.pos,
        )

    # Extract questions list
    if "questions" not in data:
        raise KeyError(f"Missing 'questions' key in {filepath}")

    questions = data["questions"]
    if not isinstance(questions, list):
        raise ValueError(f"'questions' must be a list, got {type(questions).__name__}")

    # Validate required fields in each question
    required_fields = {"question", "expected_answer", "doc_id"}
    validated_questions = []

    for i, question in enumerate(questions):
        if not isinstance(question, dict):
            raise ValueError(
                f"Question {i} is not a dict, got {type(question).__name__}"
            )

        # Check for missing required fields
        missing_fields = required_fields - set(question.keys())
        if missing_fields:
            raise ValueError(
                f"Question {i} missing required fields: {', '.join(sorted(missing_fields))}. "
                f"Required: {', '.join(sorted(required_fields))}. "
                f"Got: {', '.join(sorted(question.keys()))}"
            )

        # Validate field types
        if not isinstance(question.get("question"), str):
            raise ValueError(
                f"Question {i}: 'question' must be str, got {type(question.get('question')).__name__}"
            )
        if not isinstance(question.get("expected_answer"), str):
            raise ValueError(
                f"Question {i}: 'expected_answer' must be str, got {type(question.get('expected_answer')).__name__}"
            )
        if not isinstance(question.get("doc_id"), str):
            raise ValueError(
                f"Question {i}: 'doc_id' must be str, got {type(question.get('doc_id')).__name__}"
            )

        validated_questions.append(question)

    return validated_questions
