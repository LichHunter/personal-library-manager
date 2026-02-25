"""LOD (Level of Detail) retrieval strategies."""

import logging
import time
from typing import Optional

from ..core.protocols import Embedder, VectorStore, LLM, RetrievalStrategy
from ..core.types import (
    Document,
    Chunk,
    IndexStats,
    SearchResponse,
    SearchHit,
    SearchStats,
)

logger = logging.getLogger(__name__)


class LODEmbedStrategy(RetrievalStrategy):
    """
    LOD Embedding-Only strategy - hierarchical search using document structure.
    
    Index:
      Level 2: Document summaries
      Level 1: Section summaries (headings + first paragraph)
      Level 0: Chunks (same as flat)
    
    Search:
      1. Search L2 (documents) -> filter to top doc_top_k
      2. Search L1 (sections) within filtered docs -> filter to top section_top_k
      3. Search L0 (chunks) within filtered sections -> return top_k
    
    No LLM at search time - pure embedding similarity.
    """
    
    COLLECTION_DOCS = "lod_docs"
    COLLECTION_SECTIONS = "lod_sections"
    COLLECTION_CHUNKS = "lod_chunks"
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        doc_top_k: int = 5,
        section_top_k: int = 10,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._doc_top_k = doc_top_k
        self._section_top_k = section_top_k
        
        self._embedder: Optional[Embedder] = None
        self._store: Optional[VectorStore] = None
        self._llm: Optional[LLM] = None
        
        self._is_indexed = False
        self._index_stats: Optional[IndexStats] = None
    
    @property
    def name(self) -> str:
        return "lod_embed"
    
    @property
    def requires_llm(self) -> bool:
        return False
    
    def configure(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: Optional[LLM] = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._llm = llm
        
        dim = self._embedder.dimension
        self._store.create_collection(self.COLLECTION_DOCS, dimension=dim)
        self._store.create_collection(self.COLLECTION_SECTIONS, dimension=dim)
        self._store.create_collection(self.COLLECTION_CHUNKS, dimension=dim)
    
    def index(self, documents: list[Document]) -> IndexStats:
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        start_time = time.perf_counter()
        total_embed_calls = 0
        
        doc_entries: list[Chunk] = []
        section_entries: list[Chunk] = []
        chunk_entries: list[Chunk] = []
        
        for doc in documents:
            doc_entry = Chunk(
                id=f"doc_{doc.id}",
                document_id=doc.id,
                section_id=None,
                content=f"{doc.title}\n\n{doc.summary}",
                level=2,
                metadata={"title": doc.title},
            )
            doc_entries.append(doc_entry)
            
            for section in doc.sections:
                section_text = f"{section.heading}\n\n{section.content[:500]}"
                section_entry = Chunk(
                    id=f"sec_{doc.id}_{section.id}",
                    document_id=doc.id,
                    section_id=section.id,
                    content=section_text,
                    level=1,
                    metadata={"heading": section.heading},
                )
                section_entries.append(section_entry)
                
                section_chunks = self._chunk_text(
                    text=section.content,
                    document_id=doc.id,
                    section_id=section.id,
                )
                chunk_entries.extend(section_chunks)
        
        logger.info(
            f"LOD index: {len(doc_entries)} docs, {len(section_entries)} sections, "
            f"{len(chunk_entries)} chunks"
        )
        
        for collection, entries in [
            (self.COLLECTION_DOCS, doc_entries),
            (self.COLLECTION_SECTIONS, section_entries),
            (self.COLLECTION_CHUNKS, chunk_entries),
        ]:
            if not entries:
                continue
            
            texts = [e.content for e in entries]
            embeddings = self._embedder.embed_batch(texts)
            total_embed_calls += 1
            
            ids = [e.id for e in entries]
            metadata = [
                {
                    "document_id": e.document_id,
                    "section_id": e.section_id or "",
                    "content": e.content,
                    "level": e.level,
                    **e.metadata,
                }
                for e in entries
            ]
            
            self._store.insert(
                collection=collection,
                ids=ids,
                embeddings=embeddings,
                metadata=metadata,
            )
        
        duration = time.perf_counter() - start_time
        total_vectors = len(doc_entries) + len(section_entries) + len(chunk_entries)
        
        self._index_stats = IndexStats(
            strategy=self.name,
            backend=self._store.backend_name,
            embedding_model=self._embedder.model_name,
            llm_model=None,
            duration_sec=duration,
            num_documents=len(documents),
            num_chunks=len(chunk_entries),
            num_vectors=total_vectors,
            llm_calls=0,
            embed_calls=total_embed_calls,
        )
        
        self._is_indexed = True
        logger.info(f"LOD indexed: {total_vectors} vectors in {duration:.2f}s")
        
        return self._index_stats
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        if not self._is_indexed:
            return SearchResponse(
                hits=[],
                stats=SearchStats(duration_ms=0.0, embed_calls=0, llm_calls=0, vectors_searched=0),
            )
        
        start_time = time.perf_counter()
        
        query_embedding = self._embedder.embed(query)
        
        doc_hits = self._store.search(
            collection=self.COLLECTION_DOCS,
            query_embedding=query_embedding,
            top_k=self._doc_top_k,
        )
        
        if not doc_hits:
            return self._empty_response(start_time)
        
        filtered_doc_ids = {hit.document_id for hit in doc_hits}
        
        section_hits = self._store.search(
            collection=self.COLLECTION_SECTIONS,
            query_embedding=query_embedding,
            top_k=self._section_top_k * len(filtered_doc_ids),
        )
        
        filtered_sections = [
            hit for hit in section_hits 
            if hit.document_id in filtered_doc_ids
        ][:self._section_top_k]
        
        if not filtered_sections:
            return self._empty_response(start_time)
        
        filtered_section_ids = {hit.section_id for hit in filtered_sections}
        
        chunk_hits = self._store.search(
            collection=self.COLLECTION_CHUNKS,
            query_embedding=query_embedding,
            top_k=top_k * len(filtered_section_ids),
        )
        
        final_hits = [
            hit for hit in chunk_hits
            if hit.section_id in filtered_section_ids
        ][:top_k]
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        vectors_searched = (
            self._store.count(self.COLLECTION_DOCS) +
            self._store.count(self.COLLECTION_SECTIONS) +
            self._store.count(self.COLLECTION_CHUNKS)
        )
        
        return SearchResponse(
            hits=final_hits,
            stats=SearchStats(
                duration_ms=duration_ms,
                embed_calls=1,
                llm_calls=0,
                vectors_searched=vectors_searched,
            ),
        )
    
    def clear(self) -> None:
        if self._store is not None:
            for collection in [self.COLLECTION_DOCS, self.COLLECTION_SECTIONS, self.COLLECTION_CHUNKS]:
                self._store.delete_collection(collection)
                if self._embedder is not None:
                    self._store.create_collection(collection, dimension=self._embedder.dimension)
        
        self._is_indexed = False
        self._index_stats = None
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        section_id: str,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        
        chunks: list[Chunk] = []
        text = text.strip()
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            if end < len(text):
                for delim in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_delim = text[start:end].rfind(delim)
                    if last_delim > self._chunk_size // 2:
                        end = start + last_delim + len(delim)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"{document_id}_{section_id}_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        section_id=section_id,
                        content=chunk_text,
                        level=0,
                        metadata={"chunk_idx": chunk_idx},
                    )
                )
                chunk_idx += 1
            
            start = end - self._chunk_overlap
            if start >= end:
                break
        
        return chunks
    
    def _empty_response(self, start_time: float) -> SearchResponse:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return SearchResponse(
            hits=[],
            stats=SearchStats(duration_ms=duration_ms, embed_calls=1, llm_calls=0, vectors_searched=0),
        )


class LODLLMStrategy(RetrievalStrategy):
    """
    LOD LLM-Guided strategy - LLM makes routing decisions at search time.
    
    Index: Same as LODEmbedStrategy (3-level hierarchy)
    
    Search:
      1. Retrieve L2 summaries -> LLM selects relevant docs
      2. Retrieve L1 summaries from selected docs -> LLM selects sections
      3. Embedding search L0 chunks within selected sections -> return top_k
    """
    
    COLLECTION_DOCS = "lod_llm_docs"
    COLLECTION_SECTIONS = "lod_llm_sections"
    COLLECTION_CHUNKS = "lod_llm_chunks"
    
    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        doc_top_k: int = 5,
        section_top_k: int = 10,
    ):
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._doc_top_k = doc_top_k
        self._section_top_k = section_top_k
        
        self._embedder: Optional[Embedder] = None
        self._store: Optional[VectorStore] = None
        self._llm: Optional[LLM] = None
        
        self._is_indexed = False
        self._index_stats: Optional[IndexStats] = None
        
        self._doc_summaries: dict[str, str] = {}
        self._section_summaries: dict[str, str] = {}
    
    @property
    def name(self) -> str:
        return "lod_llm"
    
    @property
    def requires_llm(self) -> bool:
        return True
    
    def configure(
        self,
        embedder: Embedder,
        store: VectorStore,
        llm: Optional[LLM] = None,
    ) -> None:
        if llm is None:
            raise ValueError("LOD LLM strategy requires an LLM")
        
        self._embedder = embedder
        self._store = store
        self._llm = llm
        
        dim = self._embedder.dimension
        self._store.create_collection(self.COLLECTION_DOCS, dimension=dim)
        self._store.create_collection(self.COLLECTION_SECTIONS, dimension=dim)
        self._store.create_collection(self.COLLECTION_CHUNKS, dimension=dim)
    
    def index(self, documents: list[Document]) -> IndexStats:
        if self._embedder is None or self._store is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        start_time = time.perf_counter()
        total_embed_calls = 0
        
        doc_entries: list[Chunk] = []
        section_entries: list[Chunk] = []
        chunk_entries: list[Chunk] = []
        
        for doc in documents:
            doc_text = f"{doc.title}\n\n{doc.summary}"
            self._doc_summaries[doc.id] = doc_text
            
            doc_entry = Chunk(
                id=f"doc_{doc.id}",
                document_id=doc.id,
                section_id=None,
                content=doc_text,
                level=2,
                metadata={"title": doc.title},
            )
            doc_entries.append(doc_entry)
            
            for section in doc.sections:
                section_text = f"{section.heading}\n\n{section.content[:500]}"
                section_key = f"{doc.id}_{section.id}"
                self._section_summaries[section_key] = section_text
                
                section_entry = Chunk(
                    id=f"sec_{doc.id}_{section.id}",
                    document_id=doc.id,
                    section_id=section.id,
                    content=section_text,
                    level=1,
                    metadata={"heading": section.heading},
                )
                section_entries.append(section_entry)
                
                section_chunks = self._chunk_text(
                    text=section.content,
                    document_id=doc.id,
                    section_id=section.id,
                )
                chunk_entries.extend(section_chunks)
        
        logger.info(
            f"LOD-LLM index: {len(doc_entries)} docs, {len(section_entries)} sections, "
            f"{len(chunk_entries)} chunks"
        )
        
        for collection, entries in [
            (self.COLLECTION_DOCS, doc_entries),
            (self.COLLECTION_SECTIONS, section_entries),
            (self.COLLECTION_CHUNKS, chunk_entries),
        ]:
            if not entries:
                continue
            
            texts = [e.content for e in entries]
            embeddings = self._embedder.embed_batch(texts)
            total_embed_calls += 1
            
            ids = [e.id for e in entries]
            metadata = [
                {
                    "document_id": e.document_id,
                    "section_id": e.section_id or "",
                    "content": e.content,
                    "level": e.level,
                    **e.metadata,
                }
                for e in entries
            ]
            
            self._store.insert(
                collection=collection,
                ids=ids,
                embeddings=embeddings,
                metadata=metadata,
            )
        
        duration = time.perf_counter() - start_time
        total_vectors = len(doc_entries) + len(section_entries) + len(chunk_entries)
        
        self._index_stats = IndexStats(
            strategy=self.name,
            backend=self._store.backend_name,
            embedding_model=self._embedder.model_name,
            llm_model=self._llm.model_name if self._llm else None,
            duration_sec=duration,
            num_documents=len(documents),
            num_chunks=len(chunk_entries),
            num_vectors=total_vectors,
            llm_calls=0,
            embed_calls=total_embed_calls,
        )
        
        self._is_indexed = True
        logger.info(f"LOD-LLM indexed: {total_vectors} vectors in {duration:.2f}s")
        
        return self._index_stats
    
    def search(self, query: str, top_k: int = 5) -> SearchResponse:
        if self._embedder is None or self._store is None or self._llm is None:
            raise RuntimeError("Strategy not configured. Call configure() first.")
        
        if not self._is_indexed:
            return SearchResponse(
                hits=[],
                stats=SearchStats(duration_ms=0.0, embed_calls=0, llm_calls=0, vectors_searched=0),
            )
        
        start_time = time.perf_counter()
        llm_calls = 0
        
        query_embedding = self._embedder.embed(query)
        
        doc_hits = self._store.search(
            collection=self.COLLECTION_DOCS,
            query_embedding=query_embedding,
            top_k=self._doc_top_k * 2,
        )
        
        if not doc_hits:
            return self._empty_response(start_time, llm_calls)
        
        doc_summaries_text = "\n\n".join(
            f"[{hit.document_id}] {hit.content}"
            for i, hit in enumerate(doc_hits)
        )
        
        doc_select_prompt = f"""Select documents most likely to answer this question.

Question: {query}

Documents:
{doc_summaries_text}

Instructions:
- Return ONLY document IDs from the list above, comma-separated
- Choose 1-3 most relevant documents
- Document IDs look like: wiki_abc123

Selected:"""
        
        doc_response = self._llm.generate(doc_select_prompt, max_tokens=100, temperature=0.0)
        llm_calls += 1
        logger.debug(f"Doc selection LLM response: {doc_response}")
        
        selected_doc_ids = self._parse_doc_ids(doc_response, [h.document_id for h in doc_hits])
        logger.debug(f"Parsed doc IDs: {selected_doc_ids}")
        
        if not selected_doc_ids:
            selected_doc_ids = {doc_hits[0].document_id}
        
        section_hits = self._store.search(
            collection=self.COLLECTION_SECTIONS,
            query_embedding=query_embedding,
            top_k=self._section_top_k * len(selected_doc_ids),
        )
        
        filtered_sections = [
            hit for hit in section_hits
            if hit.document_id in selected_doc_ids
        ]
        
        if not filtered_sections:
            return self._empty_response(start_time, llm_calls)
        
        section_summaries_text = "\n\n".join(
            f"[{hit.section_id}] {hit.content}"
            for i, hit in enumerate(filtered_sections[:12])
        )
        
        section_select_prompt = f"""Select sections most likely to answer this question.

Question: {query}

Sections:
{section_summaries_text}

Instructions:
- Return ONLY section IDs from the list above, comma-separated
- Choose 2-4 most relevant sections
- Section IDs look like: sec_2, sec_7_6, sec_3_1

Selected:"""
        
        section_response = self._llm.generate(section_select_prompt, max_tokens=100, temperature=0.0)
        llm_calls += 1
        logger.debug(f"Section selection LLM response: {section_response}")
        
        candidate_sections = [h.section_id for h in filtered_sections if h.section_id]
        logger.debug(f"Candidate sections: {candidate_sections[:10]}")
        
        selected_section_ids = self._parse_section_ids(section_response, candidate_sections)
        logger.debug(f"Parsed section IDs: {selected_section_ids}")
        
        if not selected_section_ids:
            selected_section_ids = {filtered_sections[0].section_id}
        
        chunk_hits = self._store.search(
            collection=self.COLLECTION_CHUNKS,
            query_embedding=query_embedding,
            top_k=top_k * len(selected_section_ids),
        )
        
        final_hits = [
            hit for hit in chunk_hits
            if hit.section_id in selected_section_ids
        ][:top_k]
        
        if not final_hits and chunk_hits:
            final_hits = [
                hit for hit in chunk_hits
                if hit.document_id in selected_doc_ids
            ][:top_k]
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        
        vectors_searched = (
            self._store.count(self.COLLECTION_DOCS) +
            self._store.count(self.COLLECTION_SECTIONS) +
            self._store.count(self.COLLECTION_CHUNKS)
        )
        
        return SearchResponse(
            hits=final_hits,
            stats=SearchStats(
                duration_ms=duration_ms,
                embed_calls=1,
                llm_calls=llm_calls,
                vectors_searched=vectors_searched,
            ),
        )
    
    def clear(self) -> None:
        if self._store is not None:
            for collection in [self.COLLECTION_DOCS, self.COLLECTION_SECTIONS, self.COLLECTION_CHUNKS]:
                self._store.delete_collection(collection)
                if self._embedder is not None:
                    self._store.create_collection(collection, dimension=self._embedder.dimension)
        
        self._doc_summaries.clear()
        self._section_summaries.clear()
        self._is_indexed = False
        self._index_stats = None
    
    def _chunk_text(
        self,
        text: str,
        document_id: str,
        section_id: str,
    ) -> list[Chunk]:
        if not text.strip():
            return []
        
        chunks: list[Chunk] = []
        text = text.strip()
        start = 0
        chunk_idx = 0
        
        while start < len(text):
            end = start + self._chunk_size
            
            if end < len(text):
                for delim in [". ", ".\n", "! ", "? ", "\n\n"]:
                    last_delim = text[start:end].rfind(delim)
                    if last_delim > self._chunk_size // 2:
                        end = start + last_delim + len(delim)
                        break
            
            chunk_text = text[start:end].strip()
            
            if chunk_text:
                chunk_id = f"{document_id}_{section_id}_{chunk_idx}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        document_id=document_id,
                        section_id=section_id,
                        content=chunk_text,
                        level=0,
                        metadata={"chunk_idx": chunk_idx},
                    )
                )
                chunk_idx += 1
            
            start = end - self._chunk_overlap
            if start >= end:
                break
        
        return chunks
    
    def _parse_doc_ids(self, response: str, candidates: list[str]) -> set[str]:
        result = set()
        response_lower = response.lower()
        for doc_id in candidates:
            if doc_id.lower() in response_lower:
                result.add(doc_id)
        return result
    
    def _parse_section_ids(self, response: str, candidates: list[str]) -> set[str]:
        result = set()
        response_clean = response.replace(",", " ").replace("\n", " ")
        response_parts = response_clean.split()
        
        for section_id in candidates:
            if not section_id:
                continue
            for part in response_parts:
                part_clean = part.strip().lower()
                if part_clean == section_id.lower():
                    result.add(section_id)
                    break
        
        return result
    
    def _empty_response(self, start_time: float, llm_calls: int) -> SearchResponse:
        duration_ms = (time.perf_counter() - start_time) * 1000
        return SearchResponse(
            hits=[],
            stats=SearchStats(duration_ms=duration_ms, embed_calls=1, llm_calls=llm_calls, vectors_searched=0),
        )
