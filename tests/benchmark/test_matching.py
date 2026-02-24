"""Unit tests for matching utilities."""

import pytest

from plm.benchmark.matching import (
    MIN_QUOTE_LENGTH,
    MIN_RECIPROCAL_WORDS,
    GENERIC_BLACKLIST,
    QuoteMatch,
    ReciprocalMatch,
    extract_text_from_html,
    extract_words,
    find_quote_matches,
    find_reciprocal_matches,
    is_generic_text,
    normalize_anchor,
    normalize_text,
)


class TestNormalizeText:
    def test_html_decode(self):
        assert normalize_text("&lt;code&gt;") == "<code>"
        assert normalize_text("&amp;") == "&"
        assert normalize_text("&quot;") == '"'

    def test_collapse_whitespace(self):
        assert normalize_text("a  b") == "a b"
        assert normalize_text("a\n\nb") == "a b"
        assert normalize_text("a\t\tb") == "a b"
        assert normalize_text("a  \n  b") == "a b"

    def test_strip_whitespace(self):
        assert normalize_text("  text  ") == "text"
        assert normalize_text("\ntext\n") == "text"

    def test_lowercase(self):
        assert normalize_text("Kubectl") == "kubectl"
        assert normalize_text("KUBERNETES") == "kubernetes"

    def test_combined(self):
        result = normalize_text("  &lt;Kubectl&gt;  \n  get  pods  ")
        assert result == "<kubectl> get pods"


class TestExtractTextFromHtml:
    def test_extract_code_tags(self):
        html = "<code>kubectl get pods</code>"
        result = extract_text_from_html(html)
        assert "kubectl get pods" in result["code"]

    def test_extract_pre_tags(self):
        html = "<pre>#!/bin/bash\necho hello</pre>"
        result = extract_text_from_html(html)
        assert "#!/bin/bash\necho hello" in result["code"]

    def test_extract_blockquote_tags(self):
        html = "<blockquote>This is a quote</blockquote>"
        result = extract_text_from_html(html)
        assert "This is a quote" in result["blockquote"]

    def test_extract_prose_tags(self):
        html = "<p>This is prose</p>"
        result = extract_text_from_html(html)
        assert "This is prose" in result["prose"]

    def test_skip_nested_tags(self):
        html = "<p>Text with <code>code inside</code></p>"
        result = extract_text_from_html(html)
        assert "code inside" in result["code"]
        assert "Text with" not in result["prose"]

    def test_empty_tags_skipped(self):
        html = "<code></code><p>  </p>"
        result = extract_text_from_html(html)
        assert len(result["code"]) == 0
        assert len(result["prose"]) == 0


class TestIsGenericText:
    def test_generic_blacklist_items(self):
        for item in GENERIC_BLACKLIST:
            assert is_generic_text(item)

    def test_case_insensitive(self):
        assert is_generic_text("RUN THE FOLLOWING COMMAND")
        assert is_generic_text("Run The Following Command")

    def test_non_generic_text(self):
        assert not is_generic_text("kubectl get pods --all-namespaces")
        assert not is_generic_text("This is a specific technical detail")


class TestFindQuoteMatches:
    def test_exact_code_match(self):
        answer_html = "<code>kubectl get pods --all-namespaces</code>"
        chunk_content = "kubectl get pods --all-namespaces"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0
        assert matches[0].match_length >= MIN_QUOTE_LENGTH
        assert matches[0].source_type == "code"
        assert matches[0].chunk_id == "chunk-1"

    def test_minimum_length_threshold(self):
        answer_html = "<code>short text</code>"
        chunk_content = "short text"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) == 0

    def test_blockquote_match(self):
        answer_html = "<blockquote>A Pod is the smallest deployable unit in Kubernetes</blockquote>"
        chunk_content = "A Pod is the smallest deployable unit in Kubernetes"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0
        assert matches[0].source_type == "blockquote"

    def test_no_match_when_text_absent(self):
        answer_html = "<code>kubectl get pods</code>"
        chunk_content = "completely different content"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) == 0

    def test_generic_text_excluded(self):
        answer_html = "<code>run the following command</code>"
        chunk_content = "run the following command"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) == 0

    def test_whitespace_normalization(self):
        answer_html = "<code>kubectl  get   pods  --all-namespaces  --show-labels</code>"
        chunk_content = "kubectl get pods --all-namespaces --show-labels"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0

    def test_case_insensitive_matching(self):
        answer_html = "<code>Kubectl Get Pods --All-Namespaces --Show-Labels</code>"
        chunk_content = "kubectl get pods --all-namespaces --show-labels"
        matches = find_quote_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0


class TestExtractWords:
    def test_basic_extraction(self):
        words = extract_words("hello world test")
        assert words == ["hello", "world", "test"]

    def test_lowercase_conversion(self):
        words = extract_words("Hello WORLD Test")
        assert words == ["hello", "world", "test"]

    def test_punctuation_removal(self):
        words = extract_words("hello, world! test.")
        assert words == ["hello", "world", "test"]

    def test_empty_string(self):
        words = extract_words("")
        assert words == []


class TestFindReciprocalMatches:
    def test_chunk_in_answer(self):
        answer_html = "<p>As the documentation says, a Pod is the smallest deployable unit in Kubernetes and it is the basic building block of the system for container orchestration</p>"
        chunk_content = "A Pod is the smallest deployable unit in Kubernetes and it is the basic building block of the system for container orchestration"
        matches = find_reciprocal_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0
        assert any(m.direction == "chunk_in_answer" for m in matches)

    def test_answer_in_chunk(self):
        answer_html = "pod lifecycle management is critical for container orchestration and resource allocation in distributed systems and cloud infrastructure platforms today always"
        chunk_content = "pod lifecycle management is critical for container orchestration and resource allocation in distributed systems and cloud infrastructure platforms today always and Kubernetes"
        matches = find_reciprocal_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0
        assert any(m.direction == "answer_in_chunk" for m in matches)

    def test_minimum_word_threshold(self):
        answer_html = "<p>short text</p>"
        chunk_content = "short text"
        matches = find_reciprocal_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) == 0

    def test_word_count_accuracy(self):
        answer_html = "<p>one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty</p>"
        chunk_content = "one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty"
        matches = find_reciprocal_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) > 0
        assert matches[0].word_count >= MIN_RECIPROCAL_WORDS

    def test_no_match_when_absent(self):
        answer_html = "<p>completely different content here</p>"
        chunk_content = "unrelated text about something else"
        matches = find_reciprocal_matches(answer_html, chunk_content, "chunk-1")

        assert len(matches) == 0


class TestNormalizeAnchor:
    def test_remove_hash(self):
        assert normalize_anchor("#pod-lifecycle") == "pod-lifecycle"
        assert normalize_anchor("pod-lifecycle") == "pod-lifecycle"

    def test_lowercase(self):
        assert normalize_anchor("#PodLifecycle") == "podlifecycle"
        assert normalize_anchor("#POD-LIFECYCLE") == "pod-lifecycle"

    def test_underscore_to_hyphen(self):
        assert normalize_anchor("#pod_lifecycle") == "pod-lifecycle"
        assert normalize_anchor("#pod_life_cycle") == "pod-life-cycle"

    def test_space_to_hyphen(self):
        assert normalize_anchor("#pod lifecycle") == "pod-lifecycle"
        assert normalize_anchor("#pod life cycle") == "pod-life-cycle"

    def test_collapse_multiple_hyphens(self):
        assert normalize_anchor("#pod--lifecycle") == "pod-lifecycle"
        assert normalize_anchor("#pod---lifecycle") == "pod-lifecycle"

    def test_remove_special_characters(self):
        assert normalize_anchor("#pod@lifecycle") == "podlifecycle"
        assert normalize_anchor("#pod!lifecycle") == "podlifecycle"

    def test_strip_leading_trailing_hyphens(self):
        assert normalize_anchor("#-pod-lifecycle-") == "pod-lifecycle"

    def test_combined_transformations(self):
        assert normalize_anchor("#Pod_Life-Cycle") == "pod-life-cycle"
        assert normalize_anchor("#Pod--Life__Cycle") == "pod-life-cycle"
        assert normalize_anchor("#Pod Life-Cycle!") == "pod-life-cycle"


class TestQuoteMatchDataclass:
    def test_creation(self):
        match = QuoteMatch(
            matched_text="test text",
            match_length=9,
            source_type="code",
            chunk_id="chunk-1",
            chunk_offset=0,
            answer_offset=0,
        )
        assert match.matched_text == "test text"
        assert match.match_length == 9
        assert match.source_type == "code"
        assert match.chunk_id == "chunk-1"


class TestReciprocalMatchDataclass:
    def test_creation(self):
        match = ReciprocalMatch(
            matched_words=["word1", "word2"],
            word_count=2,
            chunk_id="chunk-1",
            direction="chunk_in_answer",
        )
        assert match.matched_words == ["word1", "word2"]
        assert match.word_count == 2
        assert match.chunk_id == "chunk-1"
        assert match.direction == "chunk_in_answer"
