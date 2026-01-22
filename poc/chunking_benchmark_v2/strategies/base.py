"""Base classes for chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Document:
    """A document to be chunked."""
    id: str
    title: str
    content: str
    path: Optional[str] = None
    metadata: dict = field(default_factory=dict)


@dataclass
class Chunk:
    """A chunk produced by a chunking strategy."""
    id: str
    doc_id: str
    content: str
    
    # Position in original document (character offsets)
    start_char: int
    end_char: int
    
    # Structural metadata
    heading: Optional[str] = None
    heading_path: list[str] = field(default_factory=list)
    level: int = 0  # 0 = leaf, higher = more abstract
    
    # For hierarchical strategies
    parent_id: Optional[str] = None
    children_ids: list[str] = field(default_factory=list)
    
    # Additional metadata
    metadata: dict = field(default_factory=dict)
    
    @property
    def token_count(self) -> int:
        """Approximate token count (words * 1.3)."""
        return int(len(self.content.split()) * 1.3)
    
    def get_char_set(self) -> set[int]:
        """Get set of character positions covered by this chunk.
        
        Used for token-level IoU calculation.
        """
        return set(range(self.start_char, self.end_char))


class ChunkingStrategy(ABC):
    """Base class for chunking strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for reporting."""
        pass
    
    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Chunk a document into pieces."""
        pass
    
    def chunk_many(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk(doc))
        return all_chunks
