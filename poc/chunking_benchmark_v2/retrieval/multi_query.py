"""Multi-Query Expansion retrieval strategy.

Generates multiple query variations using an LLM, retrieves for each,
and merges results using Reciprocal Rank Fusion.
"""

import subprocess
from typing import Optional
from collections import defaultdict

import numpy as np

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, RerankerMixin, StructuredDocument


def call_ollama(prompt: str, model: str = "llama3.2:3b", timeout: int = 60) -> str:
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


def reciprocal_rank_fusion(
    ranked_lists: list[list[int]], k: int = 60
) -> list[tuple[int, float]]:
    """Combine multiple ranked lists using RRF."""
    scores = defaultdict(float)
    for ranked_list in ranked_lists:
        for rank, doc_idx in enumerate(ranked_list):
            scores[doc_idx] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class MultiQueryRetrieval(RetrievalStrategy, EmbedderMixin, RerankerMixin):
    """Multi-Query expansion with RRF fusion.

    Generates query variations (synonyms, rephrases) and retrieves
    for each, then combines results using Reciprocal Rank Fusion.
    """

    EXPANSION_PROMPT = """Generate 3 different ways to ask this question. Keep each on a separate line. Be concise.

Original: {query}

Variations:"""

    def __init__(
        self,
        name: str = "multi_query",
        llm_model: str = "llama3.2:3b",
        num_variations: int = 3,
        rrf_k: int = 60,
        use_reranker: bool = False,
        **kwargs,
    ):
        super().__init__(name, **kwargs)
        self.llm_model = llm_model
        self.num_variations = num_variations
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks
        self.embeddings = self.encode_texts([c.content for c in chunks])

    def expand_query(self, query: str) -> list[str]:
        """Generate query variations using LLM."""
        prompt = self.EXPANSION_PROMPT.format(query=query)
        response = call_ollama(prompt, self.llm_model)

        variations = [query]
        for line in response.split("\n"):
            line = line.strip()
            if line and not line[0].isdigit():
                variations.append(line)
            elif line and line[0].isdigit():
                parts = line.split(". ", 1)
                if len(parts) > 1:
                    variations.append(parts[1])

        return variations[: self.num_variations + 1]

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        if self.chunks is None or self.embeddings is None:
            return []

        queries = self.expand_query(query)

        ranked_lists = []
        for q in queries:
            q_emb = self.encode_query(q)
            sims = np.dot(self.embeddings, q_emb)
            top_idx = np.argsort(sims)[::-1][: k * 3].tolist()
            ranked_lists.append(top_idx)

        fused = reciprocal_rank_fusion(ranked_lists, self.rrf_k)

        if self.use_reranker and self.reranker is not None:
            candidate_idx = [idx for idx, _ in fused[: k * 4]]
            candidates = [self.chunks[i] for i in candidate_idx]
            return self.rerank(query, candidates, k)
        else:
            top_idx = [idx for idx, _ in fused[:k]]
            return [self.chunks[i] for i in top_idx]

    def get_index_stats(self) -> dict:
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "llm_model": self.llm_model,
            "num_variations": self.num_variations,
        }
