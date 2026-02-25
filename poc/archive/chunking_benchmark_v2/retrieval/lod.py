"""Level-of-Detail (LOD) retrieval - 3-level hierarchy with embedding search."""

from typing import Optional

import numpy as np

from strategies import Chunk, Document, FixedSizeStrategy
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument, Section


class LODRetrieval(RetrievalStrategy, EmbedderMixin):
    """Level-of-Detail retrieval with 3-level hierarchy.
    
    Hierarchy:
    - Level 2 (coarse): Document summaries
    - Level 1 (medium): Section summaries  
    - Level 0 (fine): Chunks
    
    Search proceeds top-down:
    1. Find top-k documents by summary similarity
    2. Within selected docs, find top-k sections
    3. Within selected sections, find top-k chunks
    
    Args:
        chunk_size: Size of chunks at level 0 (default 512 tokens).
        doc_top_k: Number of documents to select at level 2.
        section_top_k: Number of sections to select at level 1.
        summary_max_chars: Max chars for document/section summaries.
    """

    def __init__(
        self,
        name: str = "lod",
        chunk_size: int = 512,
        doc_top_k: int = 3,
        section_top_k: int = 5,
        summary_max_chars: int = 500,
        **kwargs
    ):
        super().__init__(name, **kwargs)
        self.chunk_size = chunk_size
        self.doc_top_k = doc_top_k
        self.section_top_k = section_top_k
        self.summary_max_chars = summary_max_chars
        
        # Level 2: Documents (id, text, embedding)
        self.doc_entries: list[tuple[str, str, np.ndarray]] = []
        
        # Level 1: Sections (doc_id, section_id, text, embedding)
        self.section_entries: list[tuple[str, str, str, np.ndarray]] = []
        
        # Level 0: Chunks (doc_id, section_id, chunk, embedding)
        self.chunk_entries: list[tuple[str, str, Chunk, np.ndarray]] = []

    def index(
        self,
        chunks: Optional[list[Chunk]] = None,
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Build 3-level index from structured documents."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")
        
        if structured_docs is None:
            raise ValueError("LOD requires structured_docs. Parse documents first.")

        # Level 2: Document summaries
        doc_texts = []
        for doc in structured_docs:
            text = f"{doc.title}\n\n{doc.summary[:self.summary_max_chars]}"
            self.doc_entries.append((doc.id, text, np.array([])))  # placeholder
            doc_texts.append(text)

        doc_embeddings = self.encode_texts(doc_texts)
        self.doc_entries = [
            (entry[0], entry[1], emb) 
            for entry, emb in zip(self.doc_entries, doc_embeddings)
        ]

        # Level 1: Section summaries
        section_texts = []
        for doc in structured_docs:
            for section in doc.sections:
                summary = section.content[:self.summary_max_chars]
                text = f"{section.heading}\n\n{summary}"
                self.section_entries.append((doc.id, section.id, text, np.array([])))
                section_texts.append(text)

        if section_texts:
            section_embeddings = self.encode_texts(section_texts)
            self.section_entries = [
                (entry[0], entry[1], entry[2], emb)
                for entry, emb in zip(self.section_entries, section_embeddings)
            ]

        # Level 0: Chunks within sections
        chunker = FixedSizeStrategy(chunk_size=self.chunk_size, overlap=0)
        chunk_texts = []

        for doc in structured_docs:
            for section in doc.sections:
                # Create temp document for chunking
                temp_doc = Document(
                    id=f"{doc.id}_{section.id}",
                    title=section.heading,
                    content=section.content,
                )
                section_chunks = chunker.chunk(temp_doc)
                
                for chunk in section_chunks:
                    # Fix doc_id to be the actual document ID
                    chunk.doc_id = doc.id
                    self.chunk_entries.append((doc.id, section.id, chunk, np.array([])))
                    chunk_texts.append(chunk.content)

        if chunk_texts:
            chunk_embeddings = self.encode_texts(chunk_texts)
            self.chunk_entries = [
                (entry[0], entry[1], entry[2], emb)
                for entry, emb in zip(self.chunk_entries, chunk_embeddings)
            ]

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve chunks using top-down hierarchical search."""
        if not self.chunk_entries:
            return []

        q_emb = self.encode_query(query)

        # Level 2: Find top documents
        doc_sims = [np.dot(entry[2], q_emb) for entry in self.doc_entries]
        top_doc_indices = np.argsort(doc_sims)[::-1][:self.doc_top_k]
        selected_doc_ids = {self.doc_entries[i][0] for i in top_doc_indices}

        # Level 1: Find top sections within selected docs
        filtered_sections = [
            (i, entry) for i, entry in enumerate(self.section_entries)
            if entry[0] in selected_doc_ids
        ]

        if not filtered_sections:
            return []

        section_sims = [(i, np.dot(entry[3], q_emb)) for i, entry in filtered_sections]
        section_sims.sort(key=lambda x: x[1], reverse=True)
        top_section_indices = [i for i, _ in section_sims[:self.section_top_k]]
        selected_section_ids = {
            (self.section_entries[i][0], self.section_entries[i][1])
            for i in top_section_indices
        }

        # Level 0: Find top chunks within selected sections
        filtered_chunks = [
            (i, entry) for i, entry in enumerate(self.chunk_entries)
            if (entry[0], entry[1]) in selected_section_ids
        ]

        if not filtered_chunks:
            return []

        chunk_sims = [(i, np.dot(entry[3], q_emb)) for i, entry in filtered_chunks]
        chunk_sims.sort(key=lambda x: x[1], reverse=True)
        top_chunk_indices = [i for i, _ in chunk_sims[:k]]

        return [self.chunk_entries[i][2] for i in top_chunk_indices]

    def get_index_stats(self) -> dict:
        return {
            "num_documents": len(self.doc_entries),
            "num_sections": len(self.section_entries),
            "num_chunks": len(self.chunk_entries),
            "doc_top_k": self.doc_top_k,
            "section_top_k": self.section_top_k,
            "chunk_size": self.chunk_size,
        }
