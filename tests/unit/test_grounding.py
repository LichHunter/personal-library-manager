"""Tests for grounding functions."""
from plm.extraction.slow.grounding import verify_span, normalize_term


def test_verify_span_exact():
    assert verify_span("React", "Use React for UI")[0] == True


def test_verify_span_not_found():
    assert verify_span("Vue", "Use React for UI")[0] == False


def test_normalize_term():
    assert normalize_term("Hello-World") == "hello world"
    assert normalize_term("HELLO_WORLD") == "hello world"
