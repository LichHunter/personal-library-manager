"""
Unit tests for Confidence Signal Calculators

Comprehensive test coverage for all signal functions including:
- Normal cases
- Edge cases (empty inputs, zero values)
- Boundary conditions
"""

import pytest
from signals import (
    known_term_ratio,
    coverage_score,
    entity_density,
    section_type_mismatch,
    technical_pattern_ratio,
    avg_term_length,
    text_grounding_score,
)


class TestKnownTermRatio:
    """Tests for known_term_ratio signal function."""

    def test_all_terms_known(self):
        """Test when all terms are in vocabulary."""
        terms = ["python", "java", "c++"]
        vocab = {"python", "java", "c++", "rust"}
        assert known_term_ratio(terms, vocab) == 1.0

    def test_no_terms_known(self):
        """Test when no terms are in vocabulary."""
        terms = ["xyz", "abc", "def"]
        vocab = {"python", "java", "c++"}
        assert known_term_ratio(terms, vocab) == 0.0

    def test_partial_terms_known(self):
        """Test when some terms are in vocabulary."""
        terms = ["python", "xyz", "java"]
        vocab = {"python", "java", "c++"}
        assert known_term_ratio(terms, vocab) == pytest.approx(2 / 3)

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        terms = []
        vocab = {"python", "java"}
        assert known_term_ratio(terms, vocab) == 0.0

    def test_empty_vocabulary(self):
        """Test with empty vocabulary."""
        terms = ["python", "java"]
        vocab = set()
        assert known_term_ratio(terms, vocab) == 0.0

    def test_case_insensitive_matching(self):
        """Test that matching is case-insensitive."""
        terms = ["Python", "JAVA", "c++"]
        vocab = {"python", "java", "c++"}
        assert known_term_ratio(terms, vocab) == 1.0

    def test_single_term_known(self):
        """Test with single term that is known."""
        terms = ["python"]
        vocab = {"python"}
        assert known_term_ratio(terms, vocab) == 1.0

    def test_single_term_unknown(self):
        """Test with single term that is unknown."""
        terms = ["xyz"]
        vocab = {"python"}
        assert known_term_ratio(terms, vocab) == 0.0


class TestCoverageScore:
    """Tests for coverage_score signal function."""

    def test_full_coverage(self):
        """Test when all text is covered by terms."""
        terms = ["hello", "world"]
        text = "hello world"
        # "hello" = 5 chars, "world" = 5 chars, total = 10 chars covered
        # text = "hello world" = 11 chars (includes space)
        # coverage = 10/11 = 0.909...
        assert coverage_score(terms, text) == pytest.approx(10 / 11)

    def test_partial_coverage(self):
        """Test when part of text is covered by terms."""
        terms = ["hello"]
        text = "hello world"
        # "hello" = 5 chars, total = 11 chars, coverage = 5/11
        assert coverage_score(terms, text) == pytest.approx(5 / 11)

    def test_no_coverage(self):
        """Test when no text is covered by terms."""
        terms = ["xyz"]
        text = "hello world"
        assert coverage_score(terms, text) == 0.0

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        terms = []
        text = "hello world"
        assert coverage_score(terms, text) == 0.0

    def test_empty_text(self):
        """Test with empty text."""
        terms = ["hello"]
        text = ""
        assert coverage_score(terms, text) == 0.0

    def test_both_empty(self):
        """Test with both empty terms and text."""
        terms = []
        text = ""
        assert coverage_score(terms, text) == 0.0

    def test_case_insensitive_coverage(self):
        """Test that coverage matching is case-insensitive."""
        terms = ["Hello"]
        text = "hello world"
        assert coverage_score(terms, text) == pytest.approx(5 / 11)

    def test_repeated_terms_in_text(self):
        """Test when term appears multiple times in text."""
        terms = ["a"]
        text = "a a a"  # 5 chars total, "a" appears 3 times = 3 chars covered
        assert coverage_score(terms, text) == pytest.approx(3 / 5)

    def test_coverage_capped_at_one(self):
        """Test that coverage is capped at 1.0."""
        terms = ["hello", "hello", "hello"]
        text = "hello"
        # 5 + 5 + 5 = 15 chars covered, but capped at 1.0
        assert coverage_score(terms, text) == 1.0

    def test_overlapping_terms(self):
        """Test with overlapping terms."""
        terms = ["hello", "ello"]
        text = "hello"
        # "hello" = 5 chars, "ello" = 4 chars, total = 9, but capped at 1.0
        assert coverage_score(terms, text) == 1.0


class TestEntityDensity:
    """Tests for entity_density signal function."""

    def test_single_term_three_tokens(self):
        """Test with single term and three tokens."""
        terms = ["python"]
        text = "python is great"
        # 1 term / 3 tokens * 100 = 33.33...
        assert entity_density(terms, text) == pytest.approx(100 / 3)

    def test_multiple_terms(self):
        """Test with multiple terms."""
        terms = ["python", "java"]
        text = "python and java are languages"
        # 2 terms / 5 tokens * 100 = 40
        assert entity_density(terms, text) == 40.0

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        terms = []
        text = "python is great"
        assert entity_density(terms, text) == 0.0

    def test_empty_text(self):
        """Test with empty text."""
        terms = ["python"]
        text = ""
        assert entity_density(terms, text) == 0.0

    def test_single_token(self):
        """Test with single token."""
        terms = ["python"]
        text = "python"
        # 1 term / 1 token * 100 = 100
        assert entity_density(terms, text) == 100.0

    def test_high_density(self):
        """Test with high entity density."""
        terms = ["a", "b", "c"]
        text = "a b c"
        # 3 terms / 3 tokens * 100 = 100
        assert entity_density(terms, text) == 100.0

    def test_low_density(self):
        """Test with low entity density."""
        terms = ["python"]
        text = "python is a great programming language for data science"
        # 1 term / 9 tokens * 100 = 11.11...
        assert entity_density(terms, text) == pytest.approx(100 / 9)

    def test_whitespace_handling(self):
        """Test that multiple spaces are handled correctly."""
        terms = ["python"]
        text = "python  is  great"  # double spaces
        # split() handles multiple spaces, so 3 tokens
        assert entity_density(terms, text) == pytest.approx(100 / 3)


class TestSectionTypeMismatch:
    """Tests for section_type_mismatch signal function."""

    def test_none_section_type(self):
        """Test with None section type (SO data)."""
        terms = ["python"]
        assert section_type_mismatch(terms, None) == 0.0

    def test_body_section_type(self):
        """Test with body section type."""
        terms = ["python"]
        assert section_type_mismatch(terms, "body") == 0.0

    def test_title_section_type(self):
        """Test with title section type."""
        terms = ["python"]
        assert section_type_mismatch(terms, "title") == 0.0

    def test_code_section_type(self):
        """Test with code section type."""
        terms = ["python"]
        assert section_type_mismatch(terms, "code") == 0.0

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        assert section_type_mismatch([], None) == 0.0

    def test_multiple_terms(self):
        """Test with multiple terms."""
        terms = ["python", "java", "c++"]
        assert section_type_mismatch(terms, "body") == 0.0

    def test_stub_always_returns_zero(self):
        """Test that stub implementation always returns 0.0."""
        # This test documents the stub behavior
        test_cases = [
            (["python"], None),
            (["python"], "body"),
            (["python"], "title"),
            ([], None),
            (["a", "b", "c"], "code"),
        ]
        for terms, section_type in test_cases:
            assert section_type_mismatch(terms, section_type) == 0.0


class TestTechnicalPatternRatio:
    """Tests for technical_pattern_ratio signal function."""

    def test_all_technical_patterns(self):
        """Test with all terms matching technical patterns."""
        terms = ["DirectoryStream", "google.maps.Api", "MAX_VALUE"]
        assert technical_pattern_ratio(terms) == 1.0

    def test_no_technical_patterns(self):
        """Test with no terms matching technical patterns."""
        terms = ["hello", "world", "test"]
        assert technical_pattern_ratio(terms) == 0.0

    def test_partial_technical_patterns(self):
        """Test with some terms matching technical patterns."""
        terms = ["DirectoryStream", "hello", "config.json"]
        assert technical_pattern_ratio(terms) == pytest.approx(2 / 3)

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        assert technical_pattern_ratio([]) == 0.0

    def test_camel_case(self):
        """Test camelCase pattern recognition."""
        terms = ["myVariable", "someMethod"]
        assert technical_pattern_ratio(terms) == 1.0

    def test_pascal_case(self):
        """Test PascalCase pattern recognition."""
        terms = ["DirectoryStream", "MyClass"]
        assert technical_pattern_ratio(terms) == 1.0

    def test_snake_case(self):
        """Test snake_case pattern recognition."""
        terms = ["my_variable", "some_method_name"]
        assert technical_pattern_ratio(terms) == 1.0

    def test_dot_notation(self):
        """Test dot.notation pattern recognition."""
        terms = ["google.maps.Api", "config.json"]
        assert technical_pattern_ratio(terms) == 1.0

    def test_constant_case_excludes_short(self):
        """Test CONSTANT_CASE excludes short terms like CODE, API."""
        terms = ["CODE", "API", "MAX_VALUE"]
        # CODE and API should NOT match (too short/generic), MAX_VALUE should
        assert technical_pattern_ratio(terms) == pytest.approx(1 / 3)

    def test_file_path(self):
        """Test file path pattern recognition."""
        terms = ["config.json", "build.xml", "test.py"]
        assert technical_pattern_ratio(terms) == 1.0


class TestAvgTermLength:
    """Tests for avg_term_length signal function."""

    def test_basic_calculation(self):
        """Test basic average length calculation."""
        terms = ["python", "java"]  # 6 + 4 = 10, avg = 5
        assert avg_term_length(terms) == 5.0

    def test_single_term(self):
        """Test with single term."""
        terms = ["DirectoryStream"]  # 15 chars
        assert avg_term_length(terms) == 15.0

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        assert avg_term_length([]) == 0.0

    def test_varying_lengths(self):
        """Test with varying term lengths."""
        terms = ["a", "bb", "ccc"]  # 1 + 2 + 3 = 6, avg = 2
        assert avg_term_length(terms) == 2.0

    def test_long_terms(self):
        """Test with long terms."""
        terms = ["google.maps.PolylineOptions", "Files.newDirectoryStream"]
        assert avg_term_length(terms) == 25.5


class TestTextGroundingScore:
    """Tests for text_grounding_score signal function."""

    def test_all_terms_grounded(self):
        """Test when all terms are found in text."""
        terms = ["python", "java"]
        text = "Python and Java are languages"
        assert text_grounding_score(terms, text) == 1.0

    def test_no_terms_grounded(self):
        """Test when no terms are found in text."""
        terms = ["xyz", "abc"]
        text = "Python is great"
        assert text_grounding_score(terms, text) == 0.0

    def test_partial_grounding(self):
        """Test when some terms are found in text."""
        terms = ["python", "xyz"]
        text = "Python is great"
        assert text_grounding_score(terms, text) == 0.5

    def test_empty_terms_list(self):
        """Test with empty terms list."""
        assert text_grounding_score([], "any text") == 0.0

    def test_empty_text(self):
        """Test with empty text."""
        assert text_grounding_score(["python"], "") == 0.0

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        terms = ["PYTHON", "Java"]
        text = "python and JAVA are here"
        assert text_grounding_score(terms, text) == 1.0

    def test_substring_matching(self):
        """Test that partial substring matches work."""
        terms = ["Stream"]
        text = "DirectoryStream is a class"
        assert text_grounding_score(terms, text) == 1.0


class TestIntegration:
    """Integration tests for all signal functions together."""

    def test_all_signals_with_realistic_data(self):
        """Test all signals with realistic extraction data."""
        terms = ["python", "machine learning", "data science"]
        text = "Python is great for machine learning and data science applications"
        vocab = {"python", "machine learning", "data science", "applications"}

        ktr = known_term_ratio(terms, vocab)
        cs = coverage_score(terms, text)
        ed = entity_density(terms, text)
        stm = section_type_mismatch(terms, "body")

        assert 0.0 <= ktr <= 1.0
        assert 0.0 <= cs <= 1.0
        assert ed >= 0.0
        assert stm == 0.0

    def test_all_signals_with_empty_data(self):
        """Test all signals with empty data."""
        terms = []
        text = ""
        vocab = set()

        ktr = known_term_ratio(terms, vocab)
        cs = coverage_score(terms, text)
        ed = entity_density(terms, text)
        stm = section_type_mismatch(terms, None)

        assert ktr == 0.0
        assert cs == 0.0
        assert ed == 0.0
        assert stm == 0.0

    def test_all_signals_with_single_term(self):
        """Test all signals with single term."""
        terms = ["python"]
        text = "python"
        vocab = {"python"}

        ktr = known_term_ratio(terms, vocab)
        cs = coverage_score(terms, text)
        ed = entity_density(terms, text)
        stm = section_type_mismatch(terms, None)

        assert ktr == 1.0
        assert cs == 1.0
        assert ed == 100.0
        assert stm == 0.0

    def test_new_signals_with_good_extraction(self):
        """Test new signals with a good extraction example."""
        terms = ["DirectoryStream", "Files.newDirectoryStream", "NIO.2"]
        text = "You can use DirectoryStream with Files.newDirectoryStream from NIO.2 API"

        tpr = technical_pattern_ratio(terms)
        atl = avg_term_length(terms)
        tgs = text_grounding_score(terms, text)

        assert tpr >= 0.6  # Most terms are technical patterns
        assert 5.0 <= atl <= 25.0  # Reasonable term lengths
        assert tgs >= 0.6  # Most terms are grounded

    def test_new_signals_with_poor_extraction(self):
        """Test new signals with a poor extraction example."""
        terms = ["CODE", "UPDATE"]
        text = "Here is some Python code for image processing"

        tpr = technical_pattern_ratio(terms)
        atl = avg_term_length(terms)
        tgs = text_grounding_score(terms, text)

        assert tpr == 0.0
        assert tgs <= 0.5
