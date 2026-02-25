"""In-memory vector store using NumPy."""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.protocols import VectorStore
from ..core.types import SearchHit

logger = logging.getLogger(__name__)


@dataclass
class Collection:
    """A collection of vectors in memory."""
    name: str
    dimension: int
    ids: list[str] = field(default_factory=list)
    embeddings: Optional[np.ndarray] = None
    metadata: list[dict] = field(default_factory=list)
    
    def add(
        self,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Add vectors to the collection."""
        new_embeddings = np.array(embeddings, dtype=np.float32)
        
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.ids.extend(ids)
        self.metadata.extend(metadata)
    
    def search(
        self,
        query_embedding: list[float],
        top_k: int,
        filter: Optional[dict] = None,
    ) -> list[tuple[str, float, dict]]:
        """Search for similar vectors. Returns (id, score, metadata) tuples."""
        if self.embeddings is None or len(self.ids) == 0:
            return []
        
        query = np.array(query_embedding, dtype=np.float32)
        
        # Cosine similarity
        # Normalize query
        query_norm = query / (np.linalg.norm(query) + 1e-10)
        
        # Normalize embeddings
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-10
        normalized = self.embeddings / norms
        
        # Compute similarities
        similarities = np.dot(normalized, query_norm)
        
        # Apply filter if provided
        if filter:
            mask = np.ones(len(self.ids), dtype=bool)
            for key, value in filter.items():
                if isinstance(value, dict) and "$in" in value:
                    # Handle $in operator
                    allowed_values = value["$in"]
                    for i, meta in enumerate(self.metadata):
                        if meta.get(key) not in allowed_values:
                            mask[i] = False
                else:
                    # Exact match
                    for i, meta in enumerate(self.metadata):
                        if meta.get(key) != value:
                            mask[i] = False
            
            # Set filtered items to -inf so they won't be selected
            similarities = np.where(mask, similarities, -np.inf)
        
        # Get top-k indices
        if top_k >= len(self.ids):
            top_indices = np.argsort(similarities)[::-1]
        else:
            top_indices = np.argpartition(similarities, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
        
        # Filter out -inf scores (filtered items)
        results = []
        for idx in top_indices:
            score = float(similarities[idx])
            if score > -np.inf:
                results.append((self.ids[idx], score, self.metadata[idx]))
            if len(results) >= top_k:
                break
        
        return results


class MemoryStore(VectorStore):
    """In-memory vector store using NumPy for cosine similarity."""
    
    def __init__(self):
        self._collections: dict[str, Collection] = {}
    
    @property
    def backend_name(self) -> str:
        return "memory"
    
    def create_collection(self, name: str, dimension: int) -> None:
        """Create a new collection."""
        if name in self._collections:
            logger.warning(f"Collection {name} already exists, overwriting")
        self._collections[name] = Collection(name=name, dimension=dimension)
        logger.debug(f"Created collection: {name} (dim={dimension})")
    
    def insert(
        self,
        collection: str,
        ids: list[str],
        embeddings: list[list[float]],
        metadata: list[dict],
    ) -> None:
        """Insert vectors into a collection."""
        if collection not in self._collections:
            raise ValueError(f"Collection not found: {collection}")
        
        self._collections[collection].add(ids, embeddings, metadata)
        logger.debug(f"Inserted {len(ids)} vectors into {collection}")
    
    def search(
        self,
        collection: str,
        query_embedding: list[float],
        top_k: int,
        filter: Optional[dict] = None,
    ) -> list[SearchHit]:
        """Search for similar vectors."""
        if collection not in self._collections:
            raise ValueError(f"Collection not found: {collection}")
        
        results = self._collections[collection].search(
            query_embedding, top_k, filter
        )
        
        hits = []
        for id_, score, meta in results:
            hit = SearchHit(
                chunk_id=id_,
                document_id=meta.get("document_id", ""),
                section_id=meta.get("section_id"),
                content=meta.get("content", ""),
                score=score,
                level=meta.get("level", 0),
            )
            hits.append(hit)
        
        return hits
    
    def count(self, collection: str) -> int:
        """Return the number of vectors in a collection."""
        if collection not in self._collections:
            return 0
        return len(self._collections[collection].ids)
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        if name in self._collections:
            del self._collections[name]
            logger.debug(f"Deleted collection: {name}")
    
    def close(self) -> None:
        """Close the store and free resources."""
        self._collections.clear()
