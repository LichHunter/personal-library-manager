"""RecursiveCharacterTextSplitter - LangChain-style chunking.

Tries to split on larger semantic boundaries first (paragraphs, sentences),
falling back to smaller boundaries (words, characters) when chunks are too large.

This is the most commonly used splitter in production RAG systems.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class RecursiveSplitterStrategy(ChunkingStrategy):
    """Recursive character text splitter with configurable separators.
    
    Similar to LangChain's RecursiveCharacterTextSplitter.
    Tries separators in order: paragraphs -> newlines -> sentences -> words -> chars
    """
    
    DEFAULT_SEPARATORS = [
        "\n\n",  # Paragraphs
        "\n",    # Lines
        ". ",    # Sentences
        "? ",    # Questions
        "! ",    # Exclamations
        "; ",    # Clauses
        ", ",    # Phrases
        " ",     # Words
        "",      # Characters (fallback)
    ]
    
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 0,
        separators: list[str] | None = None,
    ):
        """
        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Number of characters to overlap between chunks
            separators: List of separators to try, in order of preference
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS
    
    @property
    def name(self) -> str:
        overlap_pct = int(self.chunk_overlap / self.chunk_size * 100) if self.chunk_size else 0
        return f"recursive_{self.chunk_size}_{overlap_pct}pct"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document using recursive character splitting."""
        text = document.content
        
        # Get raw splits with character positions
        splits = self._split_text(text, self.separators)
        
        if not splits:
            return []
        
        # Merge splits into chunks of target size with overlap
        chunks = self._merge_splits(document.id, splits)
        
        return chunks
    
    def _split_text(
        self,
        text: str,
        separators: list[str],
    ) -> list[dict]:
        """Recursively split text, tracking character positions.
        
        Returns list of {"text": str, "start": int, "end": int}
        """
        if not text:
            return []
        
        # Base case: no separators left, return as is
        if not separators:
            return [{"text": text, "start": 0, "end": len(text)}]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        # Empty separator = character-level split
        if separator == "":
            # Don't actually split by character, just return the text
            return [{"text": text, "start": 0, "end": len(text)}]
        
        # Split by current separator
        parts = text.split(separator)
        
        if len(parts) == 1:
            # Separator not found, try next
            return self._split_text(text, remaining_separators)
        
        # Build splits with positions
        splits = []
        current_pos = 0
        
        for i, part in enumerate(parts):
            if not part:  # Skip empty parts
                current_pos += len(separator)
                continue
            
            # If this part is too large, recursively split it
            if len(part) > self.chunk_size:
                sub_splits = self._split_text(part, remaining_separators)
                # Adjust positions relative to current_pos
                for sub in sub_splits:
                    splits.append({
                        "text": sub["text"],
                        "start": current_pos + sub["start"],
                        "end": current_pos + sub["end"],
                    })
            else:
                splits.append({
                    "text": part,
                    "start": current_pos,
                    "end": current_pos + len(part),
                })
            
            # Move past this part and the separator
            current_pos += len(part)
            if i < len(parts) - 1:
                current_pos += len(separator)
        
        return splits
    
    def _merge_splits(
        self,
        doc_id: str,
        splits: list[dict],
    ) -> list[Chunk]:
        """Merge small splits into chunks of target size with overlap."""
        if not splits:
            return []
        
        chunks = []
        chunk_idx = 0
        
        current_texts = []
        current_start = splits[0]["start"]
        current_length = 0
        
        for split in splits:
            split_len = len(split["text"])
            
            # If adding this split would exceed chunk_size, finalize current chunk
            if current_length > 0 and current_length + split_len + 1 > self.chunk_size:
                # Create chunk from current texts
                chunk_text = " ".join(current_texts)
                chunk_end = current_start + len(chunk_text)
                
                chunks.append(Chunk(
                    id=f"{doc_id}_rec_{chunk_idx}",
                    doc_id=doc_id,
                    content=chunk_text,
                    start_char=current_start,
                    end_char=chunk_end,
                    level=0,
                    metadata={
                        "strategy": self.name,
                        "chunk_idx": chunk_idx,
                    }
                ))
                chunk_idx += 1
                
                # Handle overlap: keep some texts for next chunk
                if self.chunk_overlap > 0:
                    overlap_texts = []
                    overlap_len = 0
                    for t in reversed(current_texts):
                        if overlap_len + len(t) + 1 <= self.chunk_overlap:
                            overlap_texts.insert(0, t)
                            overlap_len += len(t) + 1
                        else:
                            break
                    current_texts = overlap_texts
                    current_length = sum(len(t) for t in current_texts) + len(current_texts) - 1 if current_texts else 0
                    # Adjust start position for overlap
                    if current_texts:
                        current_start = split["start"] - current_length - 1
                else:
                    current_texts = []
                    current_length = 0
                    current_start = split["start"]
            
            # Add split to current chunk
            current_texts.append(split["text"])
            if current_length == 0:
                current_start = split["start"]
                current_length = split_len
            else:
                current_length += split_len + 1  # +1 for space
        
        # Don't forget the last chunk
        if current_texts:
            chunk_text = " ".join(current_texts)
            chunks.append(Chunk(
                id=f"{doc_id}_rec_{chunk_idx}",
                doc_id=doc_id,
                content=chunk_text,
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": chunk_idx,
                }
            ))
        
        return chunks
