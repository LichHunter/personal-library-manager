"""Heading + Paragraph hybrid chunking.

Combines heading structure with paragraph granularity.
Each section is stored as a unit containing its paragraphs.
Retrieval can happen at section or paragraph level.

Copied from V1 benchmark for comprehensive testing.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class HeadingParagraphStrategy(ChunkingStrategy):
    """Combine heading structure with paragraph granularity.
    
    Each section is stored as a unit containing its paragraphs.
    Retrieval can happen at section or paragraph level.
    """
    
    def __init__(self, max_heading_level: int = 3, include_heading_in_paragraphs: bool = True):
        """
        Args:
            max_heading_level: Deepest heading level to split on
            include_heading_in_paragraphs: Whether to prepend heading to each paragraph
        """
        self.max_heading_level = max_heading_level
        self.include_heading_in_paragraphs = include_heading_in_paragraphs
    
    @property
    def name(self) -> str:
        return f"heading_paragraph_h{self.max_heading_level}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Create chunks at both section and paragraph levels."""
        content = document.content
        
        # Pattern to match headings
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
        
        all_chunks = []
        heading_stack = []
        
        # Handle content before first heading
        if headings:
            intro_end = headings[0]['start']
        else:
            intro_end = len(content)
        
        intro_content = content[:intro_end].strip()
        if intro_content:
            intro_chunks = self._process_section(
                document.id,
                intro_content,
                0,
                "Introduction",
                ["Introduction"],
                heading_level=0,
                section_idx=-1
            )
            all_chunks.extend(intro_chunks)
        
        # Process each section
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
            
            # Process this section
            section_chunks = self._process_section(
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
    
    def _process_section(
        self,
        doc_id: str,
        content: str,
        base_offset: int,
        heading: str,
        heading_path: list[str],
        heading_level: int,
        section_idx: int
    ) -> list[Chunk]:
        """Process a section into section-level and paragraph-level chunks."""
        chunks = []
        
        if not content.strip():
            return []
        
        # Create section-level chunk
        heading_prefix = f"{'#' * heading_level} {heading}\n\n" if heading_level > 0 else ""
        section_chunk = Chunk(
            id=f"{doc_id}_section_{section_idx}" if section_idx >= 0 else f"{doc_id}_intro",
            doc_id=doc_id,
            content=(heading_prefix + content).strip(),
            start_char=base_offset,
            end_char=base_offset + len(content),
            heading=heading,
            heading_path=heading_path,
            level=1,  # Section level
            metadata={
                "strategy": self.name,
                "type": "section",
                "heading_level": heading_level
            }
        )
        chunks.append(section_chunk)
        
        # Split into paragraphs
        paragraphs = re.split(r'\n\s*\n', content)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]
        
        current_pos = 0
        for para_idx, para_text in enumerate(paragraphs):
            # Find position in content
            para_start = content.find(para_text, current_pos)
            if para_start < 0:
                para_start = current_pos
            
            # Optionally prepend heading for context
            if self.include_heading_in_paragraphs and heading_level > 0:
                para_content = f"{'#' * heading_level} {heading}\n\n{para_text}"
            else:
                para_content = para_text
            
            para_chunk = Chunk(
                id=f"{doc_id}_section_{section_idx}_para_{para_idx}" if section_idx >= 0 else f"{doc_id}_intro_para_{para_idx}",
                doc_id=doc_id,
                content=para_content,
                start_char=base_offset + para_start,
                end_char=base_offset + para_start + len(para_text),
                heading=heading,
                heading_path=heading_path,
                level=0,  # Paragraph level (leaf)
                parent_id=section_chunk.id,
                metadata={
                    "strategy": self.name,
                    "type": "paragraph",
                    "para_idx": para_idx,
                    "heading_level": heading_level
                }
            )
            chunks.append(para_chunk)
            
            current_pos = para_start + len(para_text)
        
        # Link section to its paragraphs
        section_chunk.children_ids = [
            c.id for c in chunks if c.metadata.get("type") == "paragraph" and c.parent_id == section_chunk.id
        ]
        
        return chunks
    
    def get_section_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Get only section-level chunks."""
        return [c for c in chunks if c.metadata.get("type") == "section"]
    
    def get_paragraph_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Get only paragraph-level chunks."""
        return [c for c in chunks if c.metadata.get("type") == "paragraph"]
