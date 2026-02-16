"""Tests for fast heuristic extraction."""
from plm.extraction.fast.heuristic import (
    extract_camel_case,
    extract_dot_paths,
    extract_all_heuristic,
)


def test_extract_camel_case():
    text = "Use HelloWorld and FooBar"
    result = list(extract_camel_case(text))
    assert "HelloWorld" in result
    assert "FooBar" in result


def test_extract_dot_paths():
    text = "Import React.Component and os.path.join"
    result = list(extract_dot_paths(text))
    assert "React.Component" in result
    assert "os.path.join" in result


def test_extract_all_heuristic(sample_text):
    result = extract_all_heuristic(sample_text)
    assert len(result) > 0
    assert "React.Component" in result
