"""Strategy 2: Heading-based sections."""

import re
from .base import ChunkingStrategy, Chunk, Document


class HeadingBasedStrategy(ChunkingStrategy):
    """Split documents by markdown headings.
    
    Each section (content under a heading) becomes one chunk.
    Preserves semantic boundaries but produces variable-sized chunks.
    """
    
    def __init__(self, max_heading_level: int = 3):
        """
        Args:
            max_heading_level: Deepest heading level to split on (1-6)
        """
        self.max_heading_level = max_heading_level
    
    @property
    def name(self) -> str:
        return f"heading_based_h{self.max_heading_level}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Split document by headings."""
        chunks = []
        content = document.content
        
        # Pattern to match markdown headings up to max_level
        heading_pattern = r'^(#{1,' + str(self.max_heading_level) + r'})\s+(.+)$'
        
        # Find all headings with their positions
        headings = []
        for match in re.finditer(heading_pattern, content, re.MULTILINE):
            level = len(match.group(1))
            title = match.group(2).strip()
            start = match.start()
            headings.append({
                'level': level,
                'title': title,
                'start': start,
                'end': match.end()
            })
        
        # If no headings, treat entire document as one chunk
        if not headings:
            chunk = Chunk(
                id=f"{document.id}_section_0",
                doc_id=document.id,
                content=content.strip(),
                start_char=0,
                end_char=len(content),
                heading=document.title,
                heading_path=[document.title],
                level=0,
                metadata={"strategy": self.name}
            )
            return [chunk] if content.strip() else []
        
        # Build heading path stack for nested headings
        heading_stack = []
        
        for i, heading in enumerate(headings):
            # Determine section end (start of next heading or end of doc)
            section_end = headings[i + 1]['start'] if i + 1 < len(headings) else len(content)
            
            # Get section content (after heading line to before next heading)
            section_start = heading['end'] + 1  # Skip newline after heading
            section_content = content[section_start:section_end].strip()
            
            # Update heading stack based on level
            while heading_stack and heading_stack[-1]['level'] >= heading['level']:
                heading_stack.pop()
            heading_stack.append(heading)
            
            # Build heading path
            heading_path = [h['title'] for h in heading_stack]
            
            # Create chunk even if content is empty (heading might be important)
            # But skip if truly empty
            if not section_content and not heading['title']:
                continue
            
            # Include heading in content for context
            full_content = f"{'#' * heading['level']} {heading['title']}\n\n{section_content}"
            
            chunk = Chunk(
                id=f"{document.id}_section_{i}",
                doc_id=document.id,
                content=full_content.strip(),
                start_char=heading['start'],
                end_char=section_end,
                heading=heading['title'],
                heading_path=heading_path,
                level=0,
                metadata={
                    "strategy": self.name,
                    "heading_level": heading['level'],
                    "section_idx": i
                }
            )
            chunks.append(chunk)
        
        # Handle content before first heading (if any)
        if headings and headings[0]['start'] > 0:
            intro_content = content[:headings[0]['start']].strip()
            if intro_content:
                intro_chunk = Chunk(
                    id=f"{document.id}_intro",
                    doc_id=document.id,
                    content=intro_content,
                    start_char=0,
                    end_char=headings[0]['start'],
                    heading="Introduction",
                    heading_path=["Introduction"],
                    level=0,
                    metadata={"strategy": self.name, "is_intro": True}
                )
                chunks.insert(0, intro_chunk)
        
        return chunks
