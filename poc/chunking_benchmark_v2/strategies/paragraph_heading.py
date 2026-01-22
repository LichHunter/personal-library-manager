"""Paragraph-based chunking with heading context prepending.

This extends V1's winning paragraph strategy by prepending the current
section heading to each chunk for additional context.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class ParagraphHeadingStrategy(ChunkingStrategy):
    """Paragraph-based chunking with heading context.
    
    Splits on paragraph boundaries, merges small paragraphs,
    and prepends the current section heading to each chunk.
    """
    
    def __init__(
        self,
        min_tokens: int = 50,
        max_tokens: int = 256,
        prepend_heading: bool = True,
        heading_separator: str = "\n\n",
    ):
        """
        Args:
            min_tokens: Minimum tokens per chunk (merge if smaller)
            max_tokens: Maximum tokens per chunk (split if larger)
            prepend_heading: Whether to prepend section heading to chunks
            heading_separator: Separator between heading and content
        """
        self.min_tokens = min_tokens
        self.max_tokens = max_tokens
        self.prepend_heading = prepend_heading
        self.heading_separator = heading_separator
        self.min_words = int(min_tokens * 0.75)
        self.max_words = int(max_tokens * 0.75)
    
    @property
    def name(self) -> str:
        suffix = "_heading" if self.prepend_heading else ""
        return f"paragraph_{self.min_tokens}_{self.max_tokens}{suffix}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by paragraphs with heading context."""
        content = document.content
        
        # Parse heading structure
        heading_map = self._build_heading_map(content)
        
        # Split into paragraphs with positions
        paragraphs = self._split_paragraphs(content)
        
        if not paragraphs:
            return []
        
        # Merge small and split large paragraphs
        chunks = []
        chunk_idx = 0
        
        i = 0
        while i < len(paragraphs):
            para = paragraphs[i]
            words = para['text'].split()
            
            # Get current heading context
            current_heading = self._get_heading_at(heading_map, para['start'])
            
            # If paragraph is too large, split it
            if len(words) > self.max_words:
                sub_chunks = self._split_large_paragraph(
                    document.id, para, current_heading, chunk_idx
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
                
                # Don't merge across major heading boundaries
                next_heading = self._get_heading_at(heading_map, next_para['start'])
                if next_heading != current_heading:
                    break
                
                merged_text = merged_text + "\n\n" + next_para['text']
                merged_end = next_para['end']
                i += 1
            
            # Create chunk with optional heading prefix
            if self.prepend_heading and current_heading:
                display_text = f"{current_heading}{self.heading_separator}{merged_text}"
            else:
                display_text = merged_text
            
            chunk = Chunk(
                id=f"{document.id}_para_{chunk_idx}",
                doc_id=document.id,
                content=display_text,
                start_char=merged_start,
                end_char=merged_end,
                heading=current_heading,
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": chunk_idx,
                    "has_heading": bool(current_heading),
                }
            )
            chunks.append(chunk)
            chunk_idx += 1
            i += 1
        
        return chunks
    
    def _build_heading_map(self, content: str) -> list[dict]:
        """Build a map of heading positions for quick lookup.
        
        Returns list of {"pos": int, "level": int, "text": str}
        sorted by position.
        """
        headings = []
        
        # Match markdown headings
        for match in re.finditer(r'^(#{1,6})\s+(.+?)$', content, re.MULTILINE):
            level = len(match.group(1))
            text = match.group(2).strip()
            headings.append({
                "pos": match.start(),
                "level": level,
                "text": text,
            })
        
        return sorted(headings, key=lambda h: h["pos"])
    
    def _get_heading_at(self, heading_map: list[dict], pos: int) -> str | None:
        """Get the current heading context at a given position.
        
        Returns the most recent heading path (e.g., "## Section > ### Subsection").
        """
        if not heading_map:
            return None
        
        # Find all headings before this position
        active_headings = []
        for h in heading_map:
            if h["pos"] > pos:
                break
            # Remove headings at same or lower level
            while active_headings and active_headings[-1]["level"] >= h["level"]:
                active_headings.pop()
            active_headings.append(h)
        
        if not active_headings:
            return None
        
        # Return just the deepest heading for simplicity
        return "#" * active_headings[-1]["level"] + " " + active_headings[-1]["text"]
    
    def _split_paragraphs(self, content: str) -> list[dict]:
        """Split content into paragraphs with character positions."""
        raw_paragraphs = re.split(r'\n\s*\n', content)
        
        paragraphs = []
        current_pos = 0
        
        for para in raw_paragraphs:
            para = para.strip()
            if not para:
                current_pos = content.find('\n\n', current_pos)
                if current_pos >= 0:
                    current_pos += 2
                continue
            
            # Skip pure heading lines (they'll be prepended to next paragraph)
            if re.match(r'^#{1,6}\s+.+$', para) and '\n' not in para:
                current_pos = content.find(para, current_pos)
                if current_pos >= 0:
                    current_pos += len(para)
                continue
            
            start = content.find(para, current_pos)
            if start < 0:
                start = current_pos
            
            paragraphs.append({
                'text': para,
                'start': start,
                'end': start + len(para),
            })
            current_pos = start + len(para)
        
        return paragraphs
    
    def _split_large_paragraph(
        self,
        doc_id: str,
        para: dict,
        heading: str | None,
        base_idx: int,
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
            
            # Prepend heading if enabled
            if self.prepend_heading and heading:
                display_text = f"{heading}{self.heading_separator}{chunk_text}"
            else:
                display_text = chunk_text
            
            # Calculate character positions
            chars_before = len(' '.join(words[:start_word])) + (1 if start_word > 0 else 0)
            
            chunk = Chunk(
                id=f"{doc_id}_para_{base_idx + sub_idx}",
                doc_id=doc_id,
                content=display_text,
                start_char=para['start'] + chars_before,
                end_char=para['start'] + chars_before + len(chunk_text),
                heading=heading,
                level=0,
                metadata={
                    "strategy": self.name,
                    "chunk_idx": base_idx + sub_idx,
                    "is_split": True,
                    "has_heading": bool(heading),
                }
            )
            chunks.append(chunk)
            
            sub_idx += 1
            start_word = end_word
        
        return chunks
