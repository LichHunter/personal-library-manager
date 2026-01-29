"""Cached wrapper for KeywordExtractor with Redis-backed caching.

This module provides a transparent caching layer for the KeywordExtractor component.
It wraps a KeywordExtractor instance and caches extraction results using RedisCacheClient.

The wrapper implements the same process(data: dict) -> dict interface as KeywordExtractor,
making it a drop-in replacement for transparent caching.

Example:
    >>> from keyword_extractor import KeywordExtractor
    >>> from redis_client import RedisCacheClient
    >>> from cached_keyword_extractor import CachedKeywordExtractor
    >>>
    >>> cache = RedisCacheClient()
    >>> extractor = KeywordExtractor(max_keywords=10)
    >>> cached = CachedKeywordExtractor(extractor, cache)
    >>>
    >>> # First call - cache miss
    >>> result1 = cached.process({'content': 'Kubernetes pod autoscaling'})
    >>> print(cached.hits, cached.misses)  # 0, 1
    >>>
    >>> # Second call - cache hit
    >>> result2 = cached.process({'content': 'Kubernetes pod autoscaling'})
    >>> print(cached.hits, cached.misses)  # 1, 1
    >>> print(result1['keywords'] == result2['keywords'])  # True
"""

import json
import logging
from typing import Any, Optional

from .redis_client import RedisCacheClient
from ..utils.logger import get_logger

logger = logging.getLogger(__name__)


class CachedKeywordExtractor:
    """Transparent caching wrapper for KeywordExtractor.

    Wraps a KeywordExtractor instance and caches extraction results using Redis.
    On cache hit, returns cached keywords without calling the wrapped extractor.
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
        """Initialize CachedKeywordExtractor.

        Args:
            extractor: KeywordExtractor instance to wrap
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
        """Extract keywords with caching.

        Implements the Component protocol. Accepts a dict with a 'content' field,
        checks cache for keywords, and returns a new dict with an added 'keywords' field.
        All input fields are preserved.

        Cache behavior:
        - If cache hit: returns cached keywords, increments hits counter
        - If cache miss: calls wrapped extractor, caches result, increments misses counter
        - If cache unavailable: calls wrapped extractor without caching

        Args:
            data: Input dict with 'content' field containing text to extract keywords from

        Returns:
            New dict with added 'keywords' field (list of keyword strings).
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
            cache_key = RedisCacheClient.make_key("keywords", config, content)

        # Try to get from cache
        if cache_key and self._cache:
            cached_value = self._cache.get(cache_key)
            if cached_value is not None:
                try:
                    keywords = json.loads(cached_value.decode("utf-8"))
                    self.hits += 1
                    logger.debug(
                        f"[{self.__class__.__name__}] cache hit for content length={len(content)}"
                    )
                    get_logger().trace(
                        f"[CachedKeywordExtractor] CACHE HIT - content_len={len(content)}, keywords={len(keywords)}, total_hits={self.hits}"
                    )
                    # Return input dict with cached keywords
                    result = dict(data)
                    result["keywords"] = keywords
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
        get_logger().trace(
            f"[CachedKeywordExtractor] CACHE MISS - content_len={len(content)}, total_misses={self.misses}"
        )
        result = self._extractor.process(data)

        # Cache the result if cache is available
        if cache_key and self._cache and "keywords" in result:
            try:
                keywords_json = json.dumps(result["keywords"])
                self._cache.set(cache_key, keywords_json.encode("utf-8"))
                logger.debug(
                    f"[{self.__class__.__name__}] cached {len(result['keywords'])} keywords"
                )
            except Exception as e:
                logger.warning(
                    f"[{self.__class__.__name__}] failed to cache result: {e}"
                )

        return result

    def _build_config(self) -> dict[str, Any]:
        """Build configuration dict for cache key generation.

        Extracts configuration from wrapped KeywordExtractor instance.
        Includes max_keywords and YAKE parameters (which are hardcoded in KeywordExtractor).

        Returns:
            Configuration dictionary for hashing
        """
        config = {
            "max_keywords": self._extractor.max_keywords,
            # YAKE parameters (hardcoded in KeywordExtractor._get_yake_extractor)
            "lan": "en",
            "n": 2,
            "top": 10,
            "dedupLim": 0.9,
            "dedupFunc": "seqm",
            "windowsSize": 1,
        }
        return config
