"""Integration tests for extraction pipeline."""
from plm.extraction import extract, fast_extract, ExtractionConfig


def test_fast_extract_basic(sample_text):
    result = fast_extract(sample_text)
    assert isinstance(result, list)
    assert len(result) > 0


def test_extract_with_config(sample_text):
    config = ExtractionConfig(use_fast_only=True)
    result = extract(sample_text, config)
    assert result.terms is not None
    assert len(result.fast_candidates) > 0
