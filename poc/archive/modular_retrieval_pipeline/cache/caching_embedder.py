"""Caching wrapper for SentenceTransformer with Redis-backed embedding caching.

This module provides a transparent caching layer for SentenceTransformer embeddings.
It wraps a SentenceTransformer instance and caches embedding results using RedisCacheClient.

The wrapper implements the same encode(texts: list[str]) -> np.ndarray interface as
SentenceTransformer, making it a drop-in replacement for transparent caching.

Example:
    >>> from sentence_transformers import SentenceTransformer
    >>> from redis_client import RedisCacheClient
    >>> from caching_embedder import CachingEmbedder
    >>> import numpy as np
    >>>
    >>> cache = RedisCacheClient()
    >>> base_embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
    >>> embedder = CachingEmbedder(base_embedder, cache, model_name='BAAI/bge-base-en-v1.5')
    >>>
    >>> texts = ['Kubernetes pods', 'Docker containers', 'Kubernetes pods']
    >>>
    >>> # First call - cache misses for unique texts
    >>> embeddings1 = embedder.encode(texts)
    >>> print(embeddings1.shape)  # (3, 768)
    >>> print(embedder.hits, embedder.misses)  # 0, 2 (duplicate hit on first call)
    >>>
    >>> # Second call - all cache hits
    >>> embeddings2 = embedder.encode(texts)
    >>> print(np.allclose(embeddings1, embeddings2))  # True
    >>> print(embedder.hits, embedder.misses)  # 3, 2 (all hits on second call)
"""

import hashlib
import json
import logging
from typing import Any, Optional

import numpy as np

from .redis_client import RedisCacheClient
from ..utils.logger import get_logger

logger = logging.getLogger(__name__)

# Embedding dimension for BGE model
EMBEDDING_DIM = 768


class CachingEmbedder:
    """Transparent caching wrapper for SentenceTransformer.

    Wraps a SentenceTransformer instance and caches embedding results using Redis.
    For each text in the input:
    - If cached: retrieves from cache
    - If not cached: encodes with wrapped embedder, stores in cache
    - Returns combined results in original order

    Handles duplicates efficiently: if the same text appears multiple times in input,
    it's only encoded once and the cached result is reused.

    Tracks hit/miss counts for monitoring cache effectiveness.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
    """

    def __init__(
        self,
        embedder: Any,
        cache: Optional[RedisCacheClient] = None,
        model_name: str = "BAAI/bge-base-en-v1.5",
    ):
        """Initialize CachingEmbedder.

        Args:
            embedder: SentenceTransformer instance to wrap
            cache: RedisCacheClient instance for caching (optional)
            model_name: Model name for cache key generation (e.g., 'BAAI/bge-base-en-v1.5')
        """
        self._embedder = embedder
        self._cache = cache
        self._model_name = model_name
        self.hits = 0
        self.misses = 0
        logger.debug(
            f"[{self.__class__.__name__}] initialized with cache={'enabled' if cache else 'disabled'}, "
            f"model={model_name}"
        )

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encode texts to embeddings with caching.

        Implements the SentenceTransformer.encode() interface. For each text:
        1. Check cache for existing embedding
        2. Collect cache misses
        3. Batch encode only misses
        4. Store new embeddings in cache
        5. Return combined results in original order

        Handles duplicates efficiently: same text only encoded once.
        Tracks hits/misses per text occurrence (duplicates count as hits if already processed).

        Args:
            texts: List of text strings to encode
            normalize_embeddings: Whether to normalize embeddings (default: True)
            show_progress_bar: Whether to show progress bar (default: False)
            **kwargs: Additional arguments passed to wrapped embedder

        Returns:
            numpy array of shape (len(texts), 768) with embeddings
        """
        if not texts:
            return np.array([], dtype=np.float32).reshape(0, EMBEDDING_DIM)

        # Build config for cache key generation
        config = self._build_config(normalize_embeddings)

        # Track which texts need encoding and their indices
        # Use dict to handle duplicates: text -> list of indices
        text_to_indices = {}
        for idx, text in enumerate(texts):
            if text not in text_to_indices:
                text_to_indices[text] = []
            text_to_indices[text].append(idx)

        # Try to get embeddings from cache
        embeddings_dict = {}  # text -> embedding array
        texts_to_encode = []  # texts that need encoding
        cache_hits = set()  # texts that were found in cache

        for text in text_to_indices.keys():
            cache_key = None
            if self._cache:
                cache_key = RedisCacheClient.make_key("embeddings", config, text)

            # Try cache first
            if cache_key and self._cache:
                cached_bytes = self._cache.get(cache_key)
                if cached_bytes is not None:
                    try:
                        embedding = np.frombuffer(
                            cached_bytes, dtype=np.float32
                        ).reshape(EMBEDDING_DIM)
                        embeddings_dict[text] = embedding
                        cache_hits.add(text)
                        logger.debug(
                            f"[{self.__class__.__name__}] cache hit for text length={len(text)}"
                        )
                        continue
                    except Exception as e:
                        logger.warning(
                            f"[{self.__class__.__name__}] failed to decode cached embedding: {e}"
                        )

            # Cache miss - need to encode this text
            texts_to_encode.append(text)
            self.misses += 1
            logger.debug(
                f"[{self.__class__.__name__}] cache miss for text length={len(text)}"
            )

        # Batch encode only the texts that weren't cached
        if texts_to_encode:
            logger.debug(
                f"[{self.__class__.__name__}] batch encoding {len(texts_to_encode)} texts"
            )
            new_embeddings = self._embedder.encode(
                texts_to_encode,
                normalize_embeddings=normalize_embeddings,
                show_progress_bar=show_progress_bar,
                **kwargs,
            )

            # Store new embeddings in cache and dict
            for text, embedding in zip(texts_to_encode, new_embeddings):
                embeddings_dict[text] = embedding

                # Cache the embedding
                cache_key = None
                if self._cache:
                    cache_key = RedisCacheClient.make_key("embeddings", config, text)

                if cache_key and self._cache:
                    try:
                        embedding_bytes = embedding.astype(np.float32).tobytes()
                        self._cache.set(cache_key, embedding_bytes)
                        logger.debug(
                            f"[{self.__class__.__name__}] cached embedding for text length={len(text)}"
                        )
                    except Exception as e:
                        logger.warning(
                            f"[{self.__class__.__name__}] failed to cache embedding: {e}"
                        )

        # Count hits for text occurrences in input
        # First occurrence of each unique text: hit if in cache_hits
        # Duplicate occurrences: always hit (already processed in this call)
        seen_in_this_call = set()
        for text in texts:
            if text not in seen_in_this_call:
                # First occurrence of this text in this call
                if text in cache_hits:
                    # It was in cache
                    self.hits += 1
                # else: it was a cache miss, already counted in misses above
                seen_in_this_call.add(text)
            else:
                # Duplicate occurrence in this call - always a hit
                self.hits += 1

        # Assemble result in original order
        result = np.zeros((len(texts), EMBEDDING_DIM), dtype=np.float32)
        for idx, text in enumerate(texts):
            result[idx] = embeddings_dict[text]

        logger.debug(
            f"[{self.__class__.__name__}] encode complete: shape={result.shape}, "
            f"hits={self.hits}, misses={self.misses}"
        )

        return result

    def _build_config(self, normalize_embeddings: bool) -> dict[str, Any]:
        """Build configuration dict for cache key generation.

        Includes model name and normalization setting, which affect embedding output.

        Args:
            normalize_embeddings: Whether embeddings are normalized

        Returns:
            Configuration dictionary for hashing
        """
        config = {
            "model": self._model_name,
            "normalize_embeddings": normalize_embeddings,
        }
        return config
