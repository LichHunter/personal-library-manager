"""WholeChunker - returns entire document as a single chunk."""

from .base import Chunk, Chunker, register_chunker


@register_chunker
class WholeChunker(Chunker):
    @property
    def name(self) -> str:
        return "whole"
    
    def chunk(self, text: str, filename: str | None = None) -> list[Chunk]:
        return [
            Chunk(
                text=text,
                index=0,
                heading=None,
                start_char=0,
                end_char=len(text),
            )
        ]
