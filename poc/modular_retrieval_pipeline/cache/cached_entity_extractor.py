"""Cached wrapper for EntityExtractor with Redis-backed caching.

This module provides a transparent caching layer for the EntityExtractor component.
It wraps an EntityExtractor instance and caches extraction results using RedisCacheClient.

The wrapper implements the same process(data: dict) -> dict interface as EntityExtractor,
making it a drop-in replacement for transparent caching.

Example:
    >>> from entity_extractor import EntityExtractor
    >>> from redis_client import RedisCacheClient
    >>> from cached_entity_extractor import CachedEntityExtractor
    >>>
    >>> cache = RedisCacheClient()
    >>> extractor = EntityExtractor()
    >>> cached = CachedEntityExtractor(extractor, cache)
    >>>
    >>> # First call - cache miss
    >>> result1 = cached.process({'content': 'Google Cloud Platform runs Kubernetes Engine'})
    >>> print(cached.hits, cached.misses)  # 0, 1
    >>>
    >>> # Second call - cache hit
    >>> result2 = cached.process({'content': 'Google Cloud Platform runs Kubernetes Engine'})
    >>> print(cached.hits, cached.misses)  # 1, 1
    >>> print(result1['entities'] == result2['entities'])  # True
"""

import json
import logging
from typing import Any, Optional

from .redis_client import RedisCacheClient

logger = logging.getLogger(__name__)


class CachedEntityExtractor:
    """Transparent caching wrapper for EntityExtractor.

    Wraps an EntityExtractor instance and caches extraction results using Redis.
    On cache hit, returns cached entities without calling the wrapped extractor.
    On cache miss, calls the wrapped extractor, caches the result, and returns it.

    Tracks hit/miss counts for monitoring cache effectiveness.

    Attributes:
        hits: Number of cache hits
        misses: Number of cache misses
    """

    def __init__(
        self,
        extractor: Any,
        cache: Optional[RedisCacheClient] = None,
    ):
        """Initialize CachedEntityExtractor.

        Args:
            extractor: EntityExtractor instance to wrap
            cache: RedisCacheClient instance for caching (optional)
        """
        self._extractor = extractor
        self._cache = cache
        self.hits = 0
        self.misses = 0
        logger.debug(
            f"[{self.__class__.__name__}] initialized with cache={'enabled' if cache else 'disabled'}"
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract entities with caching.

        Implements the Component protocol. Accepts a dict with a 'content' field,
        checks cache for entities, and returns a new dict with an added 'entities' field.
        All input fields are preserved.

        Cache behavior:
        - If cache hit: returns cached entities, increments hits counter
        - If cache miss: calls wrapped extractor, caches result, increments misses counter
        - If cache unavailable: calls wrapped extractor without caching

        Args:
            data: Input dict with 'content' field containing text to extract entities from

        Returns:
            New dict with added 'entities' field (dict mapping entity type to list of entity texts).
            All input fields are preserved.
        """
        # Validate input
        if "content" not in data:
            raise KeyError("Input dict must have 'content' field")

        content = data["content"]

        # Generate cache key if cache is available
        cache_key = None
        if self._cache:
            config = self._build_config()
            cache_key = RedisCacheClient.make_key("entities", config, content)

        # Try to get from cache
        if cache_key and self._cache:
            cached_value = self._cache.get(cache_key)
            if cached_value is not None:
                try:
                    entities = json.loads(cached_value.decode("utf-8"))
                    self.hits += 1
                    logger.debug(
                        f"[{self.__class__.__name__}] cache hit for content length={len(content)}"
                    )
                    # Return input dict with cached entities
                    result = dict(data)
                    result["entities"] = entities
                    return result
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    logger.warning(
                        f"[{self.__class__.__name__}] failed to decode cached value: {e}"
                    )

        # Cache miss - call wrapped extractor
        self.misses += 1
        logger.debug(
            f"[{self.__class__.__name__}] cache miss for content length={len(content)}"
        )
        result = self._extractor.process(data)

        # Cache the result if cache is available
        if cache_key and self._cache and "entities" in result:
            try:
                entities_json = json.dumps(result["entities"])
                self._cache.set(cache_key, entities_json.encode("utf-8"))
                entity_count = sum(len(v) for v in result["entities"].values())
                logger.debug(
                    f"[{self.__class__.__name__}] cached {entity_count} entities across {len(result['entities'])} types"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] failed to cache result: {e}"
                )

        return result

    def _build_config(self) -> dict[str, Any]:
        """Build configuration dict for cache key generation.

        Extracts configuration from wrapped EntityExtractor instance.
        Includes entity_types and spacy_model (which is hardcoded in EntityExtractor).

        Returns:
            Configuration dictionary for hashing
        """
        config = {
            "entity_types": sorted(list(self._extractor.entity_types)),
            "spacy_model": "en_core_web_sm",
        }
        return config
