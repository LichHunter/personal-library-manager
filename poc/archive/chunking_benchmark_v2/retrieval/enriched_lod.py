"""Enriched Level-of-Detail retrieval with configurable enrichment types."""

import time
from typing import Optional

import numpy as np

from strategies import Chunk, Document, FixedSizeStrategy
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument, Section

import sys

sys.path.insert(0, str(__file__).rsplit("/", 2)[0])
from enrichment import (
    EnrichmentCache,
    KeywordEnricher,
    ContextualEnricher,
    QuestionEnricher,
    SummaryEnricher,
    EntityEnricher,
    EnrichmentResult,
)


ENRICHER_CLASSES = {
    "keywords": KeywordEnricher,
    "contextual": ContextualEnricher,
    "questions": QuestionEnricher,
    "summary": SummaryEnricher,
    "entities": EntityEnricher,
}


class EnrichedLODRetrieval(RetrievalStrategy, EmbedderMixin):
    def __init__(
        self,
        name: str = "enriched_lod",
        enrichment_types: list[str] = None,
        llm_model: str = "llama3.2:3b",
        chunk_size: int = 512,
        doc_top_k: int = 3,
        section_top_k: int = 5,
        summary_max_chars: int = 500,
        use_cache: bool = True,
        cache_dir: str = "enrichment_cache",
        verbose: bool = True,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.enrichment_types = enrichment_types or ["keywords"]
        self.llm_model = llm_model
        self.chunk_size = chunk_size
        self.doc_top_k = doc_top_k
        self.section_top_k = section_top_k
        self.summary_max_chars = summary_max_chars
        self.use_cache = use_cache
        self.verbose = verbose

        self.cache = EnrichmentCache(cache_dir) if use_cache else None
        self.enrichers = {
            etype: ENRICHER_CLASSES[etype](model=llm_model)
            for etype in self.enrichment_types
            if etype in ENRICHER_CLASSES
        }

        self.doc_entries: list[tuple[str, str, np.ndarray]] = []
        self.section_entries: list[tuple[str, str, str, np.ndarray]] = []
        self.chunk_entries: list[tuple[str, str, Chunk, np.ndarray, str]] = []

        self.enrichment_time_s = 0.0
        self._enrichment_count = 0
        self._cache_hits = 0

    def set_llm_model(self, model: str):
        self.llm_model = model
        self.enrichers = {
            etype: ENRICHER_CLASSES[etype](model=model)
            for etype in self.enrichment_types
            if etype in ENRICHER_CLASSES
        }

    def _enrich_content(self, content: str, context: Optional[dict] = None) -> str:
        if not self.enrichers:
            return content

        for etype in self.enrichment_types:
            if self.cache:
                cached = self.cache.get(content, etype, self.llm_model)
                if cached:
                    self._cache_hits += 1
                    return cached.enhanced_content

            enricher = self.enrichers.get(etype)
            if enricher:
                self._enrichment_count += 1
                if self.verbose and self._enrichment_count % 10 == 0:
                    print(
                        f"    [enriching {self._enrichment_count}] {etype} with {self.llm_model}...",
                        flush=True,
                    )
                result = enricher.enrich(content, context)
                if self.cache:
                    self.cache.put(content, etype, self.llm_model, result)
                return result.enhanced_content

        return content

    def index(
        self,
        chunks: Optional[list[Chunk]] = None,
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        if structured_docs is None:
            raise ValueError("EnrichedLOD requires structured_docs.")

        enrichment_start = time.time()

        doc_texts = []
        for doc in structured_docs:
            text = f"{doc.title}\n\n{doc.summary[: self.summary_max_chars]}"
            enriched = self._enrich_content(
                text, {"doc_title": doc.title, "section": "Summary"}
            )
            self.doc_entries.append((doc.id, text, np.array([])))
            doc_texts.append(enriched)

        doc_embeddings = self.encode_texts(doc_texts)
        self.doc_entries = [
            (entry[0], entry[1], emb)
            for entry, emb in zip(self.doc_entries, doc_embeddings)
        ]

        section_texts = []
        section_enriched = []
        for doc in structured_docs:
            for section in doc.sections:
                summary = section.content[: self.summary_max_chars]
                text = f"{section.heading}\n\n{summary}"
                enriched = self._enrich_content(
                    text, {"doc_title": doc.title, "section": section.heading}
                )
                self.section_entries.append((doc.id, section.id, text, np.array([])))
                section_texts.append(text)
                section_enriched.append(enriched)

        if section_enriched:
            section_embeddings = self.encode_texts(section_enriched)
            self.section_entries = [
                (entry[0], entry[1], entry[2], emb)
                for entry, emb in zip(self.section_entries, section_embeddings)
            ]

        chunker = FixedSizeStrategy(chunk_size=self.chunk_size, overlap=0)
        chunk_originals = []
        chunk_enriched = []

        for doc in structured_docs:
            for section in doc.sections:
                temp_doc = Document(
                    id=f"{doc.id}_{section.id}",
                    title=section.heading,
                    content=section.content,
                )
                section_chunks = chunker.chunk(temp_doc)

                for chunk in section_chunks:
                    chunk.doc_id = doc.id
                    enriched = self._enrich_content(
                        chunk.content,
                        {"doc_title": doc.title, "section": section.heading},
                    )
                    self.chunk_entries.append(
                        (doc.id, section.id, chunk, np.array([]), chunk.content)
                    )
                    chunk_originals.append(chunk.content)
                    chunk_enriched.append(enriched)

        if chunk_enriched:
            chunk_embeddings = self.encode_texts(chunk_enriched)
            self.chunk_entries = [
                (entry[0], entry[1], entry[2], emb, entry[4])
                for entry, emb in zip(self.chunk_entries, chunk_embeddings)
            ]

        self.enrichment_time_s = time.time() - enrichment_start

        if self.verbose:
            total = (
                len(self.doc_entries)
                + len(self.section_entries)
                + len(self.chunk_entries)
            )
            print(
                f"    [enriched] docs={len(self.doc_entries)} sections={len(self.section_entries)} chunks={len(self.chunk_entries)} total={total}",
                flush=True,
            )
            print(
                f"    [stats] new_enrichments={self._enrichment_count} cache_hits={self._cache_hits} time={self.enrichment_time_s:.1f}s",
                flush=True,
            )

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if not self.chunk_entries:
            return []

        q_emb = self.encode_query(query)

        doc_sims = [np.dot(entry[2], q_emb) for entry in self.doc_entries]
        top_doc_indices = np.argsort(doc_sims)[::-1][: self.doc_top_k]
        selected_doc_ids = {self.doc_entries[i][0] for i in top_doc_indices}

        filtered_sections = [
            (i, entry)
            for i, entry in enumerate(self.section_entries)
            if entry[0] in selected_doc_ids
        ]

        if not filtered_sections:
            return []

        section_sims = [(i, np.dot(entry[3], q_emb)) for i, entry in filtered_sections]
        section_sims.sort(key=lambda x: x[1], reverse=True)
        top_section_indices = [i for i, _ in section_sims[: self.section_top_k]]
        selected_section_ids = {
            (self.section_entries[i][0], self.section_entries[i][1])
            for i in top_section_indices
        }

        filtered_chunks = [
            (i, entry)
            for i, entry in enumerate(self.chunk_entries)
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
            "enrichment_types": self.enrichment_types,
            "llm_model": self.llm_model,
            "doc_top_k": self.doc_top_k,
            "section_top_k": self.section_top_k,
            "chunk_size": self.chunk_size,
            "enrichment_time_s": self.enrichment_time_s,
        }
