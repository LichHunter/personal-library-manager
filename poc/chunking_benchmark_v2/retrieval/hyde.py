"""HyDE (Hypothetical Document Embeddings) retrieval strategy.

Uses an LLM to generate a hypothetical document that answers the query,
then embeds that document instead of the query for retrieval.
This bridges the semantic gap between how users ask questions and how documents are written.
"""

import subprocess
import json
from typing import Optional

import numpy as np

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, RerankerMixin, StructuredDocument


def call_ollama(prompt: str, model: str = "llama3.2:3b", timeout: int = 60) -> str:
    """Call Ollama for text generation."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return ""
    except Exception as e:
        print(f"Ollama error: {e}")
        return ""


class HyDERetrieval(RetrievalStrategy, EmbedderMixin, RerankerMixin):
    """Hypothetical Document Embeddings retrieval.

    Instead of embedding the query directly, generates a hypothetical
    document that would answer the query, then embeds that document.
    This often produces better retrieval for natural language queries.

    Optionally supports reranking the final results.
    """

    HYDE_PROMPT = """You are a technical documentation writer. Given a question, write a short passage (2-3 sentences) that would answer it directly. Write as if you're quoting from official documentation.

Question: {query}

Documentation passage:"""

    def __init__(
        self,
        name: str = "hyde",
        llm_model: str = "llama3.2:3b",
        num_hypotheses: int = 1,
        use_reranker: bool = False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.llm_model = llm_model
        self.num_hypotheses = num_hypotheses
        self.use_reranker = use_reranker
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks by computing embeddings."""
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks
        self.embeddings = self.encode_texts([c.content for c in chunks])

    def generate_hypothetical_document(self, query: str) -> str:
        """Generate a hypothetical document that would answer the query."""
        prompt = self.HYDE_PROMPT.format(query=query)
        return call_ollama(prompt, self.llm_model)

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve using HyDE: generate hypothetical doc, embed it, search."""
        if self.chunks is None or self.embeddings is None:
            return []

        # Generate hypothetical document(s)
        hypotheses = []
        for _ in range(self.num_hypotheses):
            hypo = self.generate_hypothetical_document(query)
            if hypo:
                hypotheses.append(hypo)

        if not hypotheses:
            # Fallback to regular semantic search
            q_emb = self.encode_query(query)
        else:
            # Average embeddings of hypothetical docs
            hypo_embeddings = self.encode_texts(hypotheses)
            q_emb = np.mean(hypo_embeddings, axis=0)
            # Normalize
            q_emb = q_emb / np.linalg.norm(q_emb)

        # Semantic search with hypothetical embedding
        sims = np.dot(self.embeddings, q_emb)

        if self.use_reranker and self.reranker is not None:
            # Get more candidates for reranking
            top_idx = np.argsort(sims)[::-1][: k * 4]
            candidates = [self.chunks[i] for i in top_idx]
            return self.rerank(query, candidates, k)
        else:
            top_idx = np.argsort(sims)[::-1][:k]
            return [self.chunks[i] for i in top_idx]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "llm_model": self.llm_model,
            "num_hypotheses": self.num_hypotheses,
        }
