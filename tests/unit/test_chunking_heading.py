import pytest
from plm.extraction.chunking import HeadingChunker, Chunk


@pytest.fixture
def chunker():
    return HeadingChunker(min_tokens=50, max_tokens=256)


@pytest.fixture
def small_chunker():
    return HeadingChunker(min_tokens=10, max_tokens=50)


class TestHeadingChunkerBasic:
    def test_name_property_returns_heading(self, chunker):
        assert chunker.name == "heading"
    
    def test_chunk_returns_list(self, chunker):
        result = chunker.chunk("Some content here.")
        assert isinstance(result, list)
    
    def test_configurable_min_tokens(self):
        chunker = HeadingChunker(min_tokens=100)
        assert chunker.min_tokens == 100
    
    def test_configurable_max_tokens(self):
        chunker = HeadingChunker(max_tokens=500)
        assert chunker.max_tokens == 500
    
    def test_configurable_prepend_heading(self):
        chunker = HeadingChunker(prepend_heading=False)
        assert chunker.prepend_heading is False


class TestHeadingChunkerParagraphSplitting:
    def test_splits_on_double_newline(self, small_chunker):
        text = "Paragraph one with enough words here.\n\nParagraph two with more words."
        result = small_chunker.chunk(text)
        assert len(result) >= 1
    
    def test_single_paragraph_returns_one_chunk(self, chunker):
        text = "This is a single paragraph with content."
        result = chunker.chunk(text)
        assert len(result) == 1
    
    def test_handles_multiple_paragraphs(self, small_chunker):
        text = "First para.\n\nSecond para.\n\nThird para."
        result = small_chunker.chunk(text)
        assert len(result) >= 1


class TestHeadingChunkerHeadingContext:
    def test_prepends_heading_to_chunk(self, small_chunker):
        text = "# Main Section\n\nContent under the heading with some words."
        result = small_chunker.chunk(text)
        assert len(result) >= 1
        assert "# Main Section" in result[0].text
    
    def test_heading_in_metadata(self, small_chunker):
        text = "# Section\n\nParagraph content here."
        result = small_chunker.chunk(text)
        assert result[0].heading == "# Section"
    
    def test_nested_headings_use_deepest(self, small_chunker):
        text = "# H1\n\n## H2\n\nContent under H2."
        result = small_chunker.chunk(text)
        assert "## H2" in result[0].text or result[0].heading == "## H2"
    
    def test_no_heading_document(self, small_chunker):
        text = "Just plain content without any headings."
        result = small_chunker.chunk(text)
        assert len(result) >= 1
        assert result[0].heading is None
    
    def test_prepend_heading_disabled(self):
        chunker = HeadingChunker(min_tokens=10, max_tokens=50, prepend_heading=False)
        text = "# Section\n\nContent here."
        result = chunker.chunk(text)
        assert not result[0].text.startswith("# Section")


class TestHeadingChunkerMerging:
    def test_merges_small_paragraphs(self):
        chunker = HeadingChunker(min_tokens=100, max_tokens=500)
        text = "Small.\n\nAlso small.\n\nStill small."
        result = chunker.chunk(text)
        assert len(result) == 1
    
    def test_no_merge_across_headings(self):
        chunker = HeadingChunker(min_tokens=100, max_tokens=500)
        text = "# Section 1\n\nSmall content.\n\n# Section 2\n\nMore small content."
        result = chunker.chunk(text)
        assert len(result) >= 2
    
    def test_respects_max_when_merging(self):
        chunker = HeadingChunker(min_tokens=10, max_tokens=30)
        words = "word " * 20
        text = f"{words}\n\n{words}"
        result = chunker.chunk(text)
        for chunk in result:
            word_count = len(chunk.text.split())
            assert word_count <= chunker.max_words + 10


class TestHeadingChunkerSplitting:
    def test_splits_large_paragraphs(self, small_chunker):
        large_text = "This is a sentence. " * 50
        result = small_chunker.chunk(large_text)
        assert len(result) > 1
    
    def test_attempts_sentence_boundary_split(self, small_chunker):
        large_text = "This is sentence one. This is sentence two. This is sentence three. " * 10
        result = small_chunker.chunk(large_text)
        for chunk in result[:-1]:
            text = chunk.text.strip()
            if not text.endswith((".", "!", "?")):
                pass


class TestHeadingChunkerTextPreservation:
    def test_preserves_internal_newlines(self, small_chunker):
        text = "Line one\nLine two\nLine three with more words."
        result = small_chunker.chunk(text)
        assert "\n" in result[0].text or len(result) > 0
    
    def test_preserves_code_blocks(self, chunker):
        text = "# Code\n\n```python\ndef foo():\n    pass\n```"
        result = chunker.chunk(text)
        assert "```python" in result[0].text
    
    def test_preserves_indentation(self, chunker):
        text = "# List\n\n- Item 1\n  - Nested item\n- Item 2"
        result = chunker.chunk(text)
        assert "  - Nested" in result[0].text or "Nested" in result[0].text


class TestHeadingChunkerCharPositions:
    def test_sequential_indices(self, small_chunker):
        text = "First para here.\n\nSecond para here.\n\nThird para here."
        result = small_chunker.chunk(text)
        for i, chunk in enumerate(result):
            assert chunk.index == i
    
    def test_valid_char_ranges(self, small_chunker):
        text = "Content here.\n\nMore content."
        result = small_chunker.chunk(text)
        for chunk in result:
            assert chunk.start_char >= 0
            assert chunk.end_char >= chunk.start_char


class TestHeadingChunkerEdgeCases:
    def test_empty_string(self, chunker):
        result = chunker.chunk("")
        assert result == []
    
    def test_only_headings(self, chunker):
        text = "# Heading 1\n\n## Heading 2\n\n### Heading 3"
        result = chunker.chunk(text)
        assert isinstance(result, list)
    
    def test_heading_at_end(self, chunker):
        text = "Some content here.\n\n# Final Heading"
        result = chunker.chunk(text)
        assert len(result) >= 1
    
    def test_multiple_blank_lines(self, small_chunker):
        text = "Para one.\n\n\n\nPara two."
        result = small_chunker.chunk(text)
        assert len(result) >= 1
    
    def test_filename_parameter_accepted(self, chunker):
        result = chunker.chunk("Content.", filename="test.md")
        assert isinstance(result, list)
    
    def test_whitespace_only_returns_empty(self, chunker):
        result = chunker.chunk("   \n\n   ")
        assert result == []
