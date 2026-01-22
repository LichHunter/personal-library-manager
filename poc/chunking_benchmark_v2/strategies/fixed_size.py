"""Fixed-size chunking with overlap support.

Simple baseline strategy that splits text into fixed-size chunks
with optional overlap. Tries to split at sentence boundaries when possible.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class FixedSizeStrategy(ChunkingStrategy):
    """Fixed-size chunking with configurable overlap.
    
    Splits documents into chunks of approximately chunk_size tokens,
    with chunk_overlap tokens shared between adjacent chunks.
    """
    
    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 0,
        respect_sentences: bool = True,
    ):
        """
        Args:
            chunk_size: Target chunk size in tokens (approx)
            overlap: Number of tokens to overlap between chunks
            respect_sentences: Try to split at sentence boundaries
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.respect_sentences = respect_sentences
        # Approximate tokens to words ratio
        self.words_per_chunk = int(chunk_size * 0.75)
        self.overlap_words = int(overlap * 0.75)
    
    @property
    def name(self) -> str:
        overlap_pct = int(self.overlap / self.chunk_size * 100) if self.chunk_size else 0
        return f"fixed_{self.chunk_size}_{overlap_pct}pct"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        text = document.content
        words = text.split()
        
        if not words:
            return []
        
        chunks = []
        chunk_idx = 0
        start_word = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.words_per_chunk, len(words))
            
            # Try to find a sentence boundary if enabled
            if self.respect_sentences and end_word < len(words):
                # Look backwards for sentence end
                for i in range(end_word, max(start_word + self.words_per_chunk // 2, start_word), -1):
                    if words[i-1].endswith(('.', '!', '?', '."', '?"', '!"')):
                        end_word = i
                        break
            
            # Extract chunk text
            chunk_words = words[start_word:end_word]
            chunk_text = ' '.join(chunk_words)
            
            # Calculate character positions
            # Find start position in original text
            words_before = ' '.join(words[:start_word])
            start_char = len(words_before) + (1 if start_word > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            chunks.append(Chunk(
                id=f"{document.id}_fix_{chunk_idx}",
                doc_id=document.id,
                content=chunk_text,
                start_char=start_char,
                end_char=end_char,
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": chunk_idx,
                    "word_range": (start_word, end_word),
                }
            ))
            chunk_idx += 1
            
            # Move to next chunk with overlap
            if self.overlap_words > 0 and end_word < len(words):
                start_word = end_word - self.overlap_words
            else:
                start_word = end_word
        
        return chunks
