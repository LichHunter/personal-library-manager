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

import re
from typing import Any, Protocol, runtime_checkable

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

# Module-level cache for YAKE extractor (matches FastEnricher)
_yake_extractor = None


def _get_yake_extractor():
    """Lazy-load YAKE extractor."""
    global _yake_extractor
    if _yake_extractor is None:
        import yake

        _yake_extractor = yake.KeywordExtractor(
            lan="en",
            n=2,
            top=10,
            dedupLim=0.9,
            dedupFunc="seqm",
            windowsSize=1,
        )
    return _yake_extractor


def _calculate_code_ratio(text: str) -> float:
    """Calculate ratio of code blocks to total text."""
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _remove_code_blocks(text: str) -> str:
    """Remove code blocks from text for NLP processing."""
    return CODE_BLOCK_PATTERN.sub(" ", text)


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

        code_ratio = _calculate_code_ratio(content)
        text_for_nlp = _remove_code_blocks(content) if code_ratio > 0.3 else content

        # Extract keywords using YAKE
        keywords = self._extract_keywords(text_for_nlp, self.max_keywords)

        # Create output dict preserving all input fields
        result = dict(data)
        result["keywords"] = keywords

        return result

    def _extract_keywords(self, content: str, max_keywords: int) -> list[str]:
        """Extract keywords from content using YAKE.

        Uses module-level cached YAKE extractor for performance.
        YAKE is initialized once and reused across all calls.

        Args:
            content: Text to extract keywords from
            max_keywords: Maximum number of keywords to extract

        Returns:
            List of keyword strings (lowercase), ordered by relevance score

        Raises:
            ImportError: If yake package is not installed
        """
        kw_extractor = _get_yake_extractor()

        keywords_with_scores = kw_extractor.extract_keywords(content)

        keywords = [kw for kw, score in keywords_with_scores]

        return keywords
