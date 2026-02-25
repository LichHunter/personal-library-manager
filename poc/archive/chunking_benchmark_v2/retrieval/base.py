"""Base classes for retrieval strategies."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from strategies import Chunk, Document


@dataclass
class Section:
    """A section within a document."""
    id: str
    heading: str
    content: str
    level: int = 1


@dataclass
class StructuredDocument:
    """Document with section structure for hierarchical strategies."""
    id: str
    title: str
    content: str
    summary: str
    sections: list[Section] = field(default_factory=list)


def parse_document_sections(doc: Document) -> StructuredDocument:
    """Parse markdown document into structured sections."""
    import re

    content = doc.content
    lines = content.split('\n')

    # Extract title from first H1 or use doc title
    title = doc.title
    for line in lines:
        if line.startswith('# '):
            title = line[2:].strip()
            break

    # Find all headings
    heading_pattern = r'^(#{1,4})\s+(.+)$'
    sections = []
    current_section = None
    current_content = []

    for line in lines:
        match = re.match(heading_pattern, line)
        if match:
            # Save previous section
            if current_section:
                current_section['content'] = '\n'.join(current_content).strip()
                if current_section['content']:
                    sections.append(Section(
                        id=f"sec_{len(sections)}",
                        heading=current_section['heading'],
                        content=current_section['content'],
                        level=current_section['level'],
                    ))

            # Start new section
            level = len(match.group(1))
            heading = match.group(2).strip()
            current_section = {'heading': heading, 'level': level}
            current_content = []
        else:
            current_content.append(line)

    # Save final section
    if current_section:
        current_section['content'] = '\n'.join(current_content).strip()
        if current_section['content']:
            sections.append(Section(
                id=f"sec_{len(sections)}",
                heading=current_section['heading'],
                content=current_section['content'],
                level=current_section['level'],
            ))

    # If no sections found, treat whole content as one section
    if not sections:
        sections.append(Section(
            id="sec_0",
            heading=title,
            content=content,
            level=1,
        ))

    # Create summary from first 500 chars
    summary = content[:500].strip()
    if len(content) > 500:
        summary += "..."

    return StructuredDocument(
        id=doc.id,
        title=title,
        content=content,
        summary=summary,
        sections=sections,
    )


class RetrievalStrategy(ABC):
    """Base class for retrieval strategies."""

    def __init__(self, name: str, **kwargs):
        self._name = name
        self.config = kwargs

    @property
    def name(self) -> str:
        return self._name

    @abstractmethod
    def index(self, chunks: list[Chunk], documents: Optional[list[Document]] = None, 
              structured_docs: Optional[list[StructuredDocument]] = None) -> None:
        """Index chunks/documents for retrieval.
        
        Args:
            chunks: Flat list of chunks (for flat strategies)
            documents: Original documents (for strategies that need full docs)
            structured_docs: Parsed documents with sections (for hierarchical strategies)
        """
        pass

    @abstractmethod
    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks for a query."""
        pass

    def get_index_stats(self) -> dict[str, Any]:
        """Return statistics about the index (for reporting)."""
        return {}


class EmbedderMixin:
    """Mixin for strategies that need embedding models."""

    embedder: Any = None
    use_prefix: bool = False

    def set_embedder(self, embedder: Any, use_prefix: bool = False):
        """Set the embedding model."""
        self.embedder = embedder
        self.use_prefix = use_prefix

    def encode_query(self, query: str) -> np.ndarray:
        """Encode a query with optional prefix."""
        if self.use_prefix:
            query = f"Represent this sentence for searching relevant passages: {query}"
        return self.embedder.encode([query], normalize_embeddings=True)[0]

    def encode_texts(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts."""
        return self.embedder.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )


class RerankerMixin:
    """Mixin for strategies that use cross-encoder reranking."""

    reranker: Any = None

    def set_reranker(self, reranker: Any):
        """Set the reranker model."""
        self.reranker = reranker

    def rerank(self, query: str, chunks: list[Chunk], k: int) -> list[Chunk]:
        """Rerank chunks using cross-encoder."""
        if not chunks or self.reranker is None:
            return chunks[:k]

        pairs = [[query, c.content] for c in chunks]
        scores = self.reranker.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [c for c, _ in ranked[:k]]
