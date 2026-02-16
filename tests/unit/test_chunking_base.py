import pytest
from plm.extraction.chunking import Chunk, Chunker, get_chunker, list_chunkers


class TestChunkDataclass:
    def test_required_fields_text_and_index(self):
        chunk = Chunk(text="hello", index=0)
        assert chunk.text == "hello"
        assert chunk.index == 0
    
    def test_optional_defaults(self):
        chunk = Chunk(text="hello", index=0)
        assert chunk.heading is None
        assert chunk.start_char == 0
        assert chunk.end_char == 0
    
    def test_all_fields_set(self):
        chunk = Chunk(
            text="content",
            index=5,
            heading="# Section",
            start_char=100,
            end_char=200,
        )
        assert chunk.text == "content"
        assert chunk.index == 5
        assert chunk.heading == "# Section"
        assert chunk.start_char == 100
        assert chunk.end_char == 200
    
    def test_multiline_text_preserved(self):
        multiline = "line1\nline2\nline3"
        chunk = Chunk(text=multiline, index=0)
        assert chunk.text == multiline
        assert "\n" in chunk.text
    
    def test_empty_text_allowed(self):
        chunk = Chunk(text="", index=0)
        assert chunk.text == ""
    
    def test_unicode_text_preserved(self):
        unicode_text = "Hello 世界 🎉"
        chunk = Chunk(text=unicode_text, index=0)
        assert chunk.text == unicode_text


class TestChunkerRegistry:
    def test_get_chunker_whole_returns_instance(self):
        chunker = get_chunker("whole")
        assert isinstance(chunker, Chunker)
        assert chunker.name == "whole"
    
    def test_get_chunker_heading_returns_instance(self):
        chunker = get_chunker("heading")
        assert isinstance(chunker, Chunker)
        assert chunker.name == "heading"
    
    def test_get_chunker_unknown_raises_valueerror(self):
        with pytest.raises(ValueError) as exc_info:
            get_chunker("unknown_chunker")
        assert "Unknown chunker" in str(exc_info.value)
        assert "unknown_chunker" in str(exc_info.value)
    
    def test_list_chunkers_returns_available(self):
        available = list_chunkers()
        assert isinstance(available, list)
        assert "whole" in available
        assert "heading" in available
    
    def test_get_chunker_each_call_returns_new_instance(self):
        chunker1 = get_chunker("whole")
        chunker2 = get_chunker("whole")
        assert chunker1 is not chunker2
    
    def test_error_message_includes_available_chunkers(self):
        with pytest.raises(ValueError) as exc_info:
            get_chunker("nonexistent")
        error_msg = str(exc_info.value)
        assert "whole" in error_msg or "Available" in error_msg
