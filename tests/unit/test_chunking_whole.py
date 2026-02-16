import pytest
from plm.extraction.chunking import WholeChunker, Chunk


@pytest.fixture
def chunker():
    return WholeChunker()


class TestWholeChunkerBasic:
    def test_name_property_returns_whole(self, chunker):
        assert chunker.name == "whole"
    
    def test_chunk_returns_list(self, chunker):
        result = chunker.chunk("text")
        assert isinstance(result, list)
    
    def test_chunk_returns_single_chunk(self, chunker):
        result = chunker.chunk("any content")
        assert len(result) == 1
    
    def test_chunk_index_is_zero(self, chunker):
        result = chunker.chunk("content")
        assert result[0].index == 0
    
    def test_chunk_returns_chunk_instance(self, chunker):
        result = chunker.chunk("test")
        assert isinstance(result[0], Chunk)


class TestWholeChunkerTextPreservation:
    def test_exact_text_preserved(self, chunker):
        text = "Hello, World!"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_single_newlines_preserved(self, chunker):
        text = "line1\nline2\nline3"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_double_newlines_preserved(self, chunker):
        text = "paragraph1\n\nparagraph2"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_leading_whitespace_preserved(self, chunker):
        text = "   indented"
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_trailing_whitespace_preserved(self, chunker):
        text = "content   "
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_markdown_syntax_preserved(self, chunker):
        text = "# Heading\n\n**bold** and *italic*\n\n- list item"
        result = chunker.chunk(text)
        assert result[0].text == text


class TestWholeChunkerCharPositions:
    def test_start_char_is_zero(self, chunker):
        result = chunker.chunk("content")
        assert result[0].start_char == 0
    
    def test_end_char_equals_text_length(self, chunker):
        text = "hello world"
        result = chunker.chunk(text)
        assert result[0].end_char == len(text)
    
    def test_unicode_char_positions_correct(self, chunker):
        text = "Hello 世界 🎉"
        result = chunker.chunk(text)
        assert result[0].start_char == 0
        assert result[0].end_char == len(text)
    
    def test_char_range_spans_entire_document(self, chunker):
        text = "a" * 1000
        result = chunker.chunk(text)
        assert result[0].end_char - result[0].start_char == len(text)


class TestWholeChunkerEdgeCases:
    def test_empty_string(self, chunker):
        result = chunker.chunk("")
        assert len(result) == 1
        assert result[0].text == ""
        assert result[0].end_char == 0
    
    def test_whitespace_only(self, chunker):
        text = "   \n\n\t  "
        result = chunker.chunk(text)
        assert result[0].text == text
    
    def test_very_long_text(self, chunker):
        text = "word " * 10000
        result = chunker.chunk(text)
        assert len(result) == 1
        assert result[0].text == text
    
    def test_heading_is_none(self, chunker):
        result = chunker.chunk("any content")
        assert result[0].heading is None
    
    def test_filename_parameter_accepted(self, chunker):
        result = chunker.chunk("content", filename="test.md")
        assert len(result) == 1
    
    def test_filename_none_accepted(self, chunker):
        result = chunker.chunk("content", filename=None)
        assert len(result) == 1
