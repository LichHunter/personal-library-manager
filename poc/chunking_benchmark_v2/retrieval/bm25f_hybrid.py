"""BM25F field-weighted hybrid retrieval strategy.

Implements field-weighted BM25 scoring where different parts of chunks
get different weights:
- Heading: 3.0x weight
- First paragraph: 2.0x weight
- Body: 1.0x weight
- Code: 0.5x weight (reduced importance)

Combines field-weighted BM25 with semantic retrieval using RRF fusion.
"""

from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from strategies import Chunk, Document
from .base import RetrievalStrategy, EmbedderMixin, StructuredDocument
from .gem_utils import extract_chunk_fields, reciprocal_rank_fusion, measure_latency


# Field weights for BM25F scoring
FIELD_WEIGHTS = {
    "heading": 3.0,
    "first_paragraph": 2.0,
    "body": 1.0,
    "code": 0.5,
}


class BM25FHybridRetrieval(RetrievalStrategy, EmbedderMixin):
    """BM25F field-weighted hybrid retrieval strategy.

    Combines field-weighted BM25 scoring with semantic retrieval.
    Different chunk fields get different weights based on their importance.
    """

    def __init__(self, name: str = "bm25f_hybrid", **kwargs):
        super().__init__(name, **kwargs)
        self.chunks: Optional[list[Chunk]] = None
        self.embeddings: Optional[np.ndarray] = None
        self.bm25: Optional[BM25Okapi] = None
        self.chunk_fields: Optional[list[dict]] = None

    def index(
        self,
        chunks: list[Chunk],
        documents: Optional[list[Document]] = None,
        structured_docs: Optional[list[StructuredDocument]] = None,
    ) -> None:
        """Index chunks for retrieval.

        Args:
            chunks: List of chunks to index
            documents: Optional list of source documents
            structured_docs: Optional structured documents (unused)
        """
        if self.embedder is None:
            raise ValueError("Embedder not set. Call set_embedder() first.")

        self.chunks = chunks

        # Extract fields from all chunks
        self.chunk_fields = [extract_chunk_fields(chunk) for chunk in chunks]

        # Create embeddings
        texts = [chunk.content for chunk in chunks]
        self.embeddings = self.encode_texts(texts)

        # Create BM25 index on full content
        tokenized = [text.lower().split() for text in texts]
        self.bm25 = BM25Okapi(tokenized)

    def _bm25f_score(self, query_tokens: list[str], chunk_fields: dict) -> float:
        """Calculate field-weighted BM25 score.

        Args:
            query_tokens: Tokenized query
            chunk_fields: Dict with 'heading', 'first_paragraph', 'body', 'code'

        Returns:
            Weighted BM25 score
        """
        total_score = 0.0

        for field, weight in FIELD_WEIGHTS.items():
            field_content = chunk_fields.get(field, "")
            if not field_content:
                continue

            field_tokens = field_content.lower().split()

            # Calculate BM25 score for this field
            # Use BM25Okapi's scoring formula on field tokens
            field_score = 0.0
            for token in query_tokens:
                # Count occurrences in field
                token_count = field_tokens.count(token)
                if token_count > 0:
                    # Simple BM25-like scoring: frequency with saturation
                    # IDF approximation: log(1 + count)
                    idf = np.log(1.0 + token_count)
                    # Saturation: count / (count + 1) to avoid over-weighting
                    saturation = token_count / (token_count + 1.0)
                    field_score += idf * saturation

            total_score += field_score * weight

        return total_score

    def retrieve(self, query: str, k: int = 5) -> list[Chunk]:
        """Retrieve top-k chunks with field-weighted BM25 + semantic fusion.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of top-k chunks
        """
        if (
            self.chunks is None
            or self.embeddings is None
            or self.bm25 is None
            or self.chunk_fields is None
        ):
            return []

        query_tokens = query.lower().split()

        # 1. Run field-weighted BM25 search
        bm25f_scores = np.array(
            [self._bm25f_score(query_tokens, fields) for fields in self.chunk_fields]
        )
        bm25f_ranks = np.argsort(bm25f_scores)[::-1][:20]
        bm25f_results = [self.chunks[i] for i in bm25f_ranks]

        # 2. Run semantic search
        q_emb = self.encode_query(query)
        sem_scores = np.dot(self.embeddings, q_emb)
        sem_ranks = np.argsort(sem_scores)[::-1][:20]
        sem_results = [self.chunks[i] for i in sem_ranks]

        # 3. RRF fusion with equal weights
        fused = reciprocal_rank_fusion(
            [bm25f_results, sem_results], weights=[1.0, 1.0], k=60
        )

        return fused[:k]

    def get_index_stats(self) -> dict:
        """Return index statistics."""
        return {
            "num_chunks": len(self.chunks) if self.chunks else 0,
            "embedding_dim": self.embeddings.shape[1]
            if self.embeddings is not None
            else 0,
            "bm25_avg_doc_len": self.bm25.avgdl if self.bm25 else 0,
        }
