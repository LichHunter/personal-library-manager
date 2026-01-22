"""Hierarchical parent-child structure chunking.

Creates hierarchical tree structure from document headings.
Stores chunks at multiple levels with parent-child relationships.
Enables flexible retrieval at different granularities.

Copied from V1 benchmark for comprehensive testing.
"""

import re
from .base import ChunkingStrategy, Chunk, Document


class HierarchicalStrategy(ChunkingStrategy):
    """Create hierarchical tree structure from document headings.
    
    Stores chunks at multiple levels with parent-child relationships.
    Enables flexible retrieval at different granularities.
    """
    
    def __init__(self, max_heading_level: int = 4):
        """
        Args:
            max_heading_level: Deepest heading level to include
        """
        self.max_heading_level = max_heading_level
    
    @property
    def name(self) -> str:
        return f"hierarchical_h{self.max_heading_level}"
    
    def chunk(self, document: Document) -> list[Chunk]:
        """Create hierarchical chunks from document."""
        content = document.content
        
        # Pattern to match all headings up to max level
        heading_pattern = r'^(#{1,' + str(self.max_heading_level) + r'})\s+(.+)$'
        
        # Parse all headings
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
        
        # Build tree structure
        chunks = []
        
        # Create root node (document level)
        root_id = f"{document.id}_root"
        
        # Get document intro (content before first heading)
        intro_end = headings[0]['start'] if headings else len(content)
        intro_content = content[:intro_end].strip()
        
        root_chunk = Chunk(
            id=root_id,
            doc_id=document.id,
            content=f"# {document.title}\n\n{intro_content}" if intro_content else f"# {document.title}",
            start_char=0,
            end_char=intro_end,
            heading=document.title,
            heading_path=[document.title],
            level=0,  # Root level
            parent_id=None,
            children_ids=[],
            metadata={"strategy": self.name, "is_root": True}
        )
        chunks.append(root_chunk)
        
        if not headings:
            return chunks
        
        # Track parent at each level
        level_parents = {0: root_chunk}
        
        for i, heading in enumerate(headings):
            # Determine section end
            section_end = headings[i + 1]['start'] if i + 1 < len(headings) else len(content)
            
            # Get section content (just this section, not children)
            section_start = heading['end'] + 1
            
            # Find where child sections start (if any)
            child_start = section_end
            for j in range(i + 1, len(headings)):
                if headings[j]['level'] > heading['level']:
                    child_start = min(child_start, headings[j]['start'])
                    break
                elif headings[j]['level'] <= heading['level']:
                    break
            
            section_content = content[section_start:child_start].strip()
            
            # Find parent
            parent_level = heading['level'] - 1
            while parent_level >= 0 and parent_level not in level_parents:
                parent_level -= 1
            parent = level_parents.get(parent_level, root_chunk)
            
            # Build heading path
            heading_path = parent.heading_path + [heading['title']]
            
            # Create chunk
            chunk_id = f"{document.id}_h{heading['level']}_{i}"
            chunk = Chunk(
                id=chunk_id,
                doc_id=document.id,
                content=f"{'#' * heading['level']} {heading['title']}\n\n{section_content}".strip(),
                start_char=heading['start'],
                end_char=child_start,
                heading=heading['title'],
                heading_path=heading_path,
                level=heading['level'],
                parent_id=parent.id,
                children_ids=[],
                metadata={
                    "strategy": self.name,
                    "heading_level": heading['level'],
                    "has_content": bool(section_content)
                }
            )
            
            # Link parent to child
            parent.children_ids.append(chunk_id)
            
            # Update level parents
            level_parents[heading['level']] = chunk
            # Clear deeper levels
            for lvl in list(level_parents.keys()):
                if lvl > heading['level']:
                    del level_parents[lvl]
            
            chunks.append(chunk)
        
        return chunks
    
    def get_leaf_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """Get only leaf nodes (no children)."""
        return [c for c in chunks if not c.children_ids]
    
    def get_chunks_with_context(self, chunks: list[Chunk], chunk_id: str) -> str:
        """Get a chunk with its parent context prepended."""
        chunk_map = {c.id: c for c in chunks}
        chunk = chunk_map.get(chunk_id)
        if not chunk:
            return ""
        
        # Build context from parents
        context_parts = []
        current = chunk
        while current:
            context_parts.insert(0, current.content)
            current = chunk_map.get(current.parent_id) if current.parent_id else None
        
        return "\n\n---\n\n".join(context_parts)
