"""Strategy 1: Fixed-size chunks with overlap."""

import re
from .base import ChunkingStrategy, Chunk, Document


class FixedSizeStrategy(ChunkingStrategy):
    """Split documents into fixed-size chunks with overlap.
    
    This is the baseline strategy - simple but loses semantic boundaries.
    """
    
    def __init__(self, chunk_size: int = 512, overlap: int = 50):
        """
        Args:
            chunk_size: Target chunk size in tokens (approx words * 1.3)
            overlap: Number of tokens to overlap between chunks
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        # Approximate: 1 token ≈ 0.75 words, so chunk_size tokens ≈ chunk_size * 0.75 words
        self.chunk_words = int(chunk_size * 0.75)
        self.overlap_words = int(overlap * 0.75)
    
    @property
    def name(self) -> str:
        return f"fixed_size_{self.chunk_size}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into fixed-size chunks."""
        chunks = []
        content = document.content
        words = content.split()
        
        if not words:
            return []
        
        chunk_idx = 0
        start_word = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.chunk_words, len(words))
            
            # Try to end at sentence boundary
            if end_word < len(words):
                # Look for sentence end in last 20% of chunk
                search_start = start_word + int(self.chunk_words * 0.8)
                for i in range(end_word, search_start, -1):
                    if i < len(words) and words[i-1].endswith(('.', '!', '?')):
                        end_word = i
                        break
            
            chunk_words_list = words[start_word:end_word]
            chunk_text = ' '.join(chunk_words_list)
            
            # Calculate character positions
            start_char = len(' '.join(words[:start_word])) + (1 if start_word > 0 else 0)
            end_char = start_char + len(chunk_text)
            
            chunk = Chunk(
                id=f"{document.id}_chunk_{chunk_idx}",
                doc_id=document.id,
                content=chunk_text,
                start_char=start_char,
                end_char=end_char,
                level=0,
                metadata={"strategy": self.name, "chunk_idx": chunk_idx}
            )
            chunks.append(chunk)
            
            chunk_idx += 1
            new_start = end_word - self.overlap_words
            
            if new_start <= start_word:
                break
            start_word = new_start
        
        return chunks
