"""Keyword extraction component using YAKE.

This module provides a stateless component that wraps YAKE keyword extraction
from FastEnricher. It follows the Component protocol for pipeline integration.

The component accepts a dict with a 'content' field and returns a new dict with
an added 'keywords' field, preserving all input fields (Unix pipe accumulation).

Example:
    >>> from keyword_extractor import KeywordExtractor
    >>> extractor = KeywordExtractor(max_keywords=5)
    >>> data = {'content': 'Kubernetes horizontal pod autoscaler scales replicas'}
    >>> result = extractor.process(data)
    >>> print(result['keywords'])
    ['kubernetes', 'autoscaler', 'replicas', 'pod', 'scales']
    >>> print(result['content'])  # Original content preserved
    'Kubernetes horizontal pod autoscaler scales replicas'

Can be wrapped with CacheableComponent for caching:
    >>> from cache import CacheableComponent
    >>> cached = CacheableComponent(KeywordExtractor(), cache_dir="cache/keywords")
    >>> result = cached.process(data)  # First call: runs extraction
    >>> result = cached.process(data)  # Second call: returns cached result
"""

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Component(Protocol):
    """Protocol for pipeline components."""

    def process(self, data: Any) -> Any:
        """Process input data and return transformed output."""
        ...


class KeywordExtractor:
    """Extract keywords from content using YAKE.

    This component wraps YAKE (Yet Another Keyword Extractor) for statistical
    keyword extraction without machine learning. It's designed to be fast and
    stateless, with YAKE initialized fresh in each process() call.

    The component accepts a dict with a 'content' field and returns a new dict
    with an added 'keywords' field. All input fields are preserved in the output
    (Unix pipe accumulation pattern).

    Args:
        max_keywords: Maximum number of keywords to extract (default 10)

    Attributes:
        max_keywords: Maximum number of keywords to extract

    Example:
        >>> extractor = KeywordExtractor(max_keywords=10)
        >>> data = {
        ...     'content': 'Kubernetes horizontal pod autoscaler scales replicas based on CPU',
        ...     'source': 'docs.md'
        ... }
        >>> result = extractor.process(data)
        >>> # result = {
        >>> #     'content': 'Kubernetes horizontal pod autoscaler...',
        >>> #     'source': 'docs.md',
        >>> #     'keywords': ['kubernetes', 'autoscaler', 'replicas', 'cpu', ...]
        >>> # }
    """

    def __init__(self, max_keywords: int = 10):
        """Initialize KeywordExtractor.

        Args:
            max_keywords: Maximum number of keywords to extract (default 10)
        """
        self.max_keywords = max_keywords

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract keywords from content and return enriched dict.

        Implements the Component protocol. Accepts a dict with a 'content' field,
        extracts keywords using YAKE, and returns a new dict with an added
        'keywords' field. All input fields are preserved.

        Args:
            data: Input dict with 'content' field containing text to extract keywords from

        Returns:
            New dict with added 'keywords' field (list of keyword strings).
            All input fields are preserved.

        Raises:
            KeyError: If 'content' field is missing from input dict
            ImportError: If yake package is not installed
            Exception: Any exception from YAKE extraction

        Example:
            >>> extractor = KeywordExtractor(max_keywords=5)
            >>> data = {'content': 'Kubernetes pod autoscaler', 'doc_id': '123'}
            >>> result = extractor.process(data)
            >>> assert 'keywords' in result
            >>> assert 'doc_id' in result  # Original field preserved
            >>> assert len(result['keywords']) <= 5
        """
        # Validate input
        if "content" not in data:
            raise KeyError("Input dict must have 'content' field")

        content = data["content"]

        # Handle empty or short content
        if not content or len(content.strip()) < 10:
            # Return input dict with empty keywords
            result = dict(data)
            result["keywords"] = []
            return result

        # Extract keywords using YAKE
        keywords = self._extract_keywords(content, self.max_keywords)

        # Create output dict preserving all input fields
        result = dict(data)
        result["keywords"] = keywords

        return result

    def _extract_keywords(self, content: str, max_keywords: int) -> list[str]:
        """Extract keywords from content using YAKE.

        This is a stateless helper method that initializes YAKE fresh for each call.
        YAKE is initialized in the method (not __init__) to keep the component
        stateless and avoid storing model state.

        Args:
            content: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of keyword strings (lowercase), ordered by relevance score

        Raises:
            ImportError: If yake package is not installed
        """
        import yake

        # Initialize YAKE extractor (stateless - fresh for each call)
        kw_extractor = yake.KeywordExtractor(
            lan="en",  # English language
            n=3,  # Maximum n-gram size (1-3 word phrases)
            dedupLim=0.9,  # Deduplication threshold
            top=max_keywords,  # Return top N keywords
            features=None,  # Use all features
        )

        # Extract keywords: returns list of (keyword, score) tuples
        keywords_with_scores = kw_extractor.extract_keywords(content)

        # Extract just the keyword strings (discard scores)
        keywords = [kw for kw, score in keywords_with_scores]

        return keywords
