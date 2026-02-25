"""Paragraph-based chunking strategy.

Splits documents by paragraph boundaries. Merges small paragraphs and splits
large ones. Natural boundaries but loses heading context.

Copied from V1 benchmark for comprehensive testing.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class ParagraphStrategy(ChunkingStrategy):
    """Split documents by paragraph boundaries.
    
    Merges small paragraphs and splits large ones.
    Natural boundaries but loses heading context.
    """
    
    def __init__(self, min_tokens: int = 50, max_tokens: int = 512):
        """
        Args:
            min_tokens: Minimum tokens per chunk (merge if smaller)
            max_tokens: Maximum tokens per chunk (split if larger)
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.min_words = int(min_tokens * 0.75)
        self.max_words = int(max_tokens * 0.75)
    
    @property
    def name(self) -> str:
        return f"paragraphs_{self.min_tokens}_{self.max_tokens}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by paragraphs."""
        content = document.content
        
        # Split by double newlines (paragraph boundaries)
        # Also handle markdown headings as paragraph starters
        raw_paragraphs = re.split(r'\n\s*\n', content)
        
        # Filter empty paragraphs and track positions
        paragraphs = []
        current_pos = 0
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                # Skip empty but track position
                current_pos = content.find('\n\n', current_pos)
                if current_pos >= 0:
                    current_pos += 2
                continue
            
            # Find actual position in content
            start = content.find(para, current_pos)
            if start < 0:
                start = current_pos
            
            paragraphs.append({
                'text': para,
                'start': start,
                'end': start + len(para)
            })
            current_pos = start + len(para)
        
        if not paragraphs:
            return []
        
        # Merge small paragraphs and split large ones
        chunks = []
        chunk_idx = 0
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            words = para['text'].split()
            
            # If paragraph is too large, split it
            if len(words) > self.max_words:
                sub_chunks = self._split_large_paragraph(
                    document.id, para, chunk_idx
                )
                chunks.extend(sub_chunks)
                chunk_idx += len(sub_chunks)
                i += 1
                continue
            
            # If paragraph is too small, try to merge with next
            merged_text = para['text']
            merged_start = para['start']
            merged_end = para['end']
            
            while len(merged_text.split()) < self.min_words and i + 1 < len(paragraphs):
                next_para = paragraphs[i + 1]
                next_words = next_para['text'].split()
                
                # Don't merge if it would exceed max
                if len(merged_text.split()) + len(next_words) > self.max_words:
                    break
                
                merged_text = merged_text + "\n\n" + next_para['text']
                merged_end = next_para['end']
                i += 1
            
            chunk = Chunk(
                id=f"{document.id}_para_{chunk_idx}",
                doc_id=document.id,
                content=merged_text,
                start_char=merged_start,
                end_char=merged_end,
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": chunk_idx
                }
            )
            chunks.append(chunk)
            chunk_idx += 1
            i += 1
        
        return chunks
    
    def _split_large_paragraph(
        self,
        doc_id: str,
        para: dict,
        base_idx: int
    ) -> list[Chunk]:
        """Split a large paragraph into smaller chunks."""
        chunks = []
        words = para['text'].split()
        
        sub_idx = 0
        start_word = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.max_words, len(words))
            
            # Try to end at sentence boundary
            if end_word < len(words):
                for i in range(end_word, start_word + int(self.max_words * 0.7), -1):
                    if words[i-1].endswith(('.', '!', '?')):
                        end_word = i
                        break
            
            chunk_text = ' '.join(words[start_word:end_word])
            
            # Calculate approximate character positions
            chars_before = len(' '.join(words[:start_word])) + (1 if start_word > 0 else 0)
            
            chunk = Chunk(
                id=f"{doc_id}_para_{base_idx + sub_idx}",
                doc_id=doc_id,
                content=chunk_text,
                start_char=para['start'] + chars_before,
                end_char=para['start'] + chars_before + len(chunk_text),
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": base_idx + sub_idx,
                    "is_split": True
                }
            )
            chunks.append(chunk)
            
            sub_idx += 1
            start_word = end_word
        
        return chunks
