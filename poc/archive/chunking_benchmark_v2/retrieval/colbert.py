"""ColBERT late-interaction retrieval using RAGatouille."""

from typing import Optional
import tempfile
import os

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument


class ColBERTRetrieval(RetrievalStrategy):
    """ColBERT late-interaction retrieval via RAGatouille.

    Uses token-level embeddings and MaxSim for more precise matching.
    """

    def __init__(
        self,
        name: str = "colbert",
        model_name: str = "colbert-ir/colbertv2.0",
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.rag = None
        self.chunks: Optional[list[Chunk]] = None
        self.index_path: Optional[str] = None
        self._temp_dir = None

    def _ensure_model(self):
        if self.rag is None:
            try:
                from ragatouille import RAGPretrainedModel

                self.rag = RAGPretrainedModel.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "RAGatouille not installed. Run: pip install ragatouille"
                )

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        self._ensure_model()

        self.chunks = chunks

        self._temp_dir = tempfile.mkdtemp(prefix="colbert_")

        contents = [c.content for c in chunks]
        doc_ids = [c.id for c in chunks]

        self.index_path = self.rag.index(
            collection=contents,
            document_ids=doc_ids,
            index_name="benchmark_index",
            max_document_length=512,
            split_documents=False,
        )

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.rag is None or self.chunks is None:
            return []

        results = self.rag.search(query, k=k)

        chunk_map = {c.id: c for c in self.chunks}
        retrieved = []
        for result in results:
            doc_id = result.get("document_id") or result.get("id")
            if doc_id and doc_id in chunk_map:
                retrieved.append(chunk_map[doc_id])

        return retrieved

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "model": self.model_name,
            "index_path": self.index_path,
        }


class ColBERTReranker(RetrievalStrategy, EmbedderMixin):
    """Use ColBERT as a reranker on top of semantic retrieval."""

    def __init__(
        self,
        name: str = "colbert_rerank",
        model_name: str = "colbert-ir/colbertv2.0",
        initial_k: int = 50,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.model_name = model_name
        self.initial_k = initial_k
        self.rag = None
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings = None

    def _ensure_model(self):
        if self.rag is None:
            try:
                from ragatouille import RAGPretrainedModel

                self.rag = RAGPretrainedModel.from_pretrained(self.model_name)
            except ImportError:
                raise ImportError(
                    "RAGatouille not installed. Run: pip install ragatouille"
                )

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self._ensure_model()
        self.chunks = chunks

        import numpy as np

        self.embeddings = self.embedder.encode(
            [c.content for c in chunks],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.chunks is None or self.embeddings is None:
            return []

        import numpy as np

        q_emb = self.embedder.encode([query], normalize_embeddings=True)[0]
        sims = np.dot(self.embeddings, q_emb)
        top_idx = np.argsort(sims)[::-1][: self.initial_k]
        candidates = [self.chunks[i] for i in top_idx]

        candidate_texts = [c.content for c in candidates]
        reranked = self.rag.rerank(query=query, documents=candidate_texts, k=k)

        content_to_chunk = {c.content: c for c in candidates}
        result = []
        for item in reranked:
            content = item.get("content") or item.get("text") or item.get("document")
            if content and content in content_to_chunk:
                result.append(content_to_chunk[content])

        return result

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "model": self.model_name,
            "initial_k": self.initial_k,
        }
