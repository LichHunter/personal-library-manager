"""Strategy 3: Heading-based with size limits."""

import re
from .base import ChunkingStrategy, Chunk, Document


class HeadingLimitedStrategy(ChunkingStrategy):
    """Split documents by headings, but subdivide large sections.
    
    Combines semantic boundaries of heading-based with size control.
    Large sections are split into sub-chunks while preserving heading context.
    """
    
    def __init__(self, max_heading_level: int = 3, max_tokens: int = 512, overlap: int = 50):
        """
        Args:
            max_heading_level: Deepest heading level to split on
            max_tokens: Maximum tokens per chunk before splitting
            overlap: Overlap tokens when splitting large sections
        """
        self.max_heading_level = max_heading_level
        self.max_tokens = max_tokens
        self.overlap = overlap
        self.max_words = int(max_tokens * 0.75)
        self.overlap_words = int(overlap * 0.75)
    
    @property
    def name(self) -> str:
        return f"heading_limited_{self.max_tokens}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by headings, subdividing large sections."""
        chunks = []
        content = document.content
        
        # Pattern to match markdown headings
        heading_pattern = r'^(#{1,' + str(self.max_heading_level) + r'})\s+(.+)$'
        
        # Find all headings
        headings = []
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            headings.append({
                'level': level,
                'title': title,
                'start': match.start(),
                'end': match.end()
            })
        
        # If no headings, chunk as fixed-size
        if not headings:
            return self._chunk_text(
                document.id, content, 0, document.title, [document.title]
            )
        
        # Build heading path stack
        heading_stack = []
        all_chunks = []
        
        # Handle content before first heading
        if headings[0]['start'] > 0:
            intro_content = content[:headings[0]['start']].strip()
            if intro_content:
                intro_chunks = self._chunk_text(
                    document.id, intro_content, 0,
                    "Introduction", ["Introduction"], is_intro=True
                )
                all_chunks.extend(intro_chunks)
        
        for i, heading in enumerate(headings):
            # Determine section end
            section_end = headings[i + 1]['start'] if i + 1 < len(headings) else len(content)
            
            # Get section content
            section_start = heading['end'] + 1
            section_content = content[section_start:section_end].strip()
            
            # Update heading stack
            while heading_stack and heading_stack[-1]['level'] >= heading['level']:
                heading_stack.pop()
            heading_stack.append(heading)
            
            # Build heading path
            heading_path = [h['title'] for h in heading_stack]
            
            # Chunk this section (may produce multiple chunks if large)
            section_chunks = self._chunk_text(
                document.id,
                section_content,
                heading['start'],
                heading['title'],
                heading_path,
                heading_level=heading['level'],
                section_idx=i
            )
            all_chunks.extend(section_chunks)
        
        return all_chunks
    
    def _chunk_text(
        self,
        doc_id: str,
        content: str,
        base_offset: int,
        heading: str,
        heading_path: list[str],
        heading_level: int = 1,
        section_idx: int = 0,
        is_intro: bool = False
    ) -> list[Chunk]:
        """Chunk text, splitting if too large."""
        if not content.strip():
            return []
        
        words = content.split()
        
        # If small enough, return as single chunk
        if len(words) <= self.max_words:
            # Include heading in content for context
            heading_prefix = f"{'#' * heading_level} {heading}\n\n" if not is_intro else ""
            full_content = heading_prefix + content
            
            chunk = Chunk(
                id=f"{doc_id}_section_{section_idx}" if not is_intro else f"{doc_id}_intro",
                doc_id=doc_id,
                content=full_content.strip(),
                start_char=base_offset,
                end_char=base_offset + len(content),
                heading=heading,
                heading_path=heading_path,
                level=0,
                metadata={
                    "strategy": self.name,
                    "heading_level": heading_level,
                    "split": False
                }
            )
            return [chunk]
        
        # Split into sub-chunks
        chunks = []
        chunk_idx = 0
        start_word = 0
        
        while start_word < len(words):
            end_word = min(start_word + self.max_words, len(words))
            
            # Try to end at sentence boundary
            if end_word < len(words):
                search_start = start_word + int(self.max_words * 0.8)
                for i in range(end_word, search_start, -1):
                    if words[i-1].endswith(('.', '!', '?')):
                        end_word = i
                        break
            
            chunk_words = words[start_word:end_word]
            chunk_text = ' '.join(chunk_words)
            
            # Include heading in first sub-chunk, indicate continuation in others
            if chunk_idx == 0:
                heading_prefix = f"{'#' * heading_level} {heading}\n\n"
            else:
                heading_prefix = f"{'#' * heading_level} {heading} (continued)\n\n"
            
            full_content = heading_prefix + chunk_text
            
            chunk = Chunk(
                id=f"{doc_id}_section_{section_idx}_part_{chunk_idx}",
                doc_id=doc_id,
                content=full_content.strip(),
                start_char=base_offset + len(' '.join(words[:start_word])),
                end_char=base_offset + len(' '.join(words[:end_word])),
                heading=heading,
                heading_path=heading_path,
                level=0,
                metadata={
                    "strategy": self.name,
                    "heading_level": heading_level,
                    "split": True,
                    "part": chunk_idx
                }
            )
            chunks.append(chunk)
            
            chunk_idx += 1
            start_word = end_word - self.overlap_words
            
            if start_word >= end_word:
                break
        
        return chunks
