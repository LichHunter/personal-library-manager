"""Base classes and registry for chunking strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class Chunk:
    """A chunk produced by a chunking strategy.
    
    Attributes:
        text: The chunk text content.
        index: The chunk index in the sequence.
        heading: Optional heading context for the chunk.
        start_char: Start character position in original document.
        end_char: End character position in original document.
    """
    text: str
    index: int
    heading: str | None = None
    start_char: int = 0
    end_char: int = 0


class Chunker(ABC):
    """Abstract base class for chunking strategies."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy name for identification."""
        pass
    
    @abstractmethod
    def chunk(self, text: str, filename: str | None = None) -> list[Chunk]:
        """Chunk text into pieces.
        
        Args:
            text: The document text to chunk.
            filename: Optional filename for context.
            
        Returns:
            List of Chunk objects.
        """
        pass


CHUNKERS: dict[str, type[Chunker]] = {}


def register_chunker(cls: type[Chunker]) -> type[Chunker]:
    """Decorator to register a chunker class.
    
    Usage:
        @register_chunker
        class MyChunker(Chunker):
            ...
    """
    instance = cls()
    CHUNKERS[instance.name] = cls
    return cls


def get_chunker(name: str) -> Chunker:
    """Factory function to get a chunker instance by name.
    
    Args:
        name: The chunker name (e.g., "whole", "heading").
        
    Returns:
        An instance of the requested chunker.
        
    Raises:
        ValueError: If the chunker name is unknown.
    """
    if name not in CHUNKERS:
        available = list(CHUNKERS.keys())
        raise ValueError(f"Unknown chunker: {name!r}. Available: {available}")
    return CHUNKERS[name]()


def list_chunkers() -> list[str]:
    """List all available chunker names."""
    return list(CHUNKERS.keys())
