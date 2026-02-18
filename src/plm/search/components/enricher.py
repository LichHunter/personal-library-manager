"""Content enrichment component for formatting accumulated pipeline data.

This module provides a stateless component that formats keywords and entities
into an enriched content string. It follows the Component protocol for pipeline
integration.

The component accepts a dict with 'content', 'keywords', and 'entities' fields
(accumulated from previous pipeline components) and returns a formatted enriched
string: "keywords | entities\n\noriginal_content"

Example:
    >>> from plm.search.components.enricher import ContentEnricher
    >>> enricher = ContentEnricher()
    >>> data = {
    ...     'content': 'Kubernetes horizontal pod autoscaler scales replicas',
    ...     'keywords': ['kubernetes', 'autoscaler', 'replicas'],
    ...     'entities': {'PRODUCT': ['Kubernetes']}
    ... }
    >>> result = enricher.process(data)
    >>> print(result)
    'kubernetes, autoscaler, replicas | Kubernetes\\n\\nKubernetes horizontal pod autoscaler scales replicas'
"""

from typing import Any


class ContentEnricher:
    """Format accumulated pipeline data into enriched content string.

    This component is the final step in the enrichment pipeline. It receives
    accumulated data from KeywordExtractor and EntityExtractor, then formats
    it into the enriched content string used for retrieval.

    The component accepts a dict with 'content', 'keywords', and 'entities'
    fields and returns a formatted enriched string. The format is:
    "keywords | entities\n\noriginal_content"

    This is pure string formatting with no extraction logic - it simply
    combines the accumulated data from previous pipeline components.

    Example:
        >>> enricher = ContentEnricher()
        >>> data = {
        ...     'content': 'Original text here',
        ...     'keywords': ['key1', 'key2'],
        ...     'entities': {'ORG': ['Company'], 'PRODUCT': ['Tool']},
        ...     'source': 'docs.md'
        ... }
        >>> result = enricher.process(data)
        >>> # result = "key1, key2 | Company, Tool\\n\\nOriginal text here"
    """

    def __init__(self):
        """Initialize ContentEnricher.

        No configuration needed - this component is stateless and performs
        pure string formatting.
        """
        pass

    def process(self, data: dict[str, Any]) -> str:
        """Format accumulated pipeline data into enriched content string.

        Implements the Component protocol. Accepts a dict with 'content',
        'keywords', and 'entities' fields, and returns a formatted enriched
        string combining all three.

        The format is: "keywords | entities\n\noriginal_content"
        - Keywords are formatted as comma-separated list (first 7 only)
        - Entities are formatted as comma-separated list (first 2 per type, max 5 total)
        - Multiple parts are separated by " | "
        - If no keywords or entities, returns original content unchanged

        Args:
            data: Input dict with 'content', 'keywords', and 'entities' fields.
                  - 'content' (str): Original content text
                  - 'keywords' (list[str]): Keywords extracted by KeywordExtractor
                  - 'entities' (dict[str, list[str]]): Entities extracted by EntityExtractor

        Returns:
            Formatted enriched string: "keywords | entities\n\noriginal_content"
            If no keywords or entities, returns original content unchanged.

        Raises:
            KeyError: If required fields ('content', 'keywords', 'entities') are missing
            TypeError: If field types are incorrect

        Example:
            >>> enricher = ContentEnricher()
            >>> data = {
            ...     'content': 'Kubernetes pod autoscaler',
            ...     'keywords': ['kubernetes', 'autoscaler'],
            ...     'entities': {'PRODUCT': ['Kubernetes']}
            ... }
            >>> result = enricher.process(data)
            >>> assert result == 'kubernetes, autoscaler | Kubernetes\\n\\nKubernetes pod autoscaler'
        """
        # Validate required fields
        if "content" not in data:
            raise KeyError("Input dict must have 'content' field")
        if "keywords" not in data:
            raise KeyError("Input dict must have 'keywords' field")
        if "entities" not in data:
            raise KeyError("Input dict must have 'entities' field")

        content = data["content"]
        keywords = data["keywords"]
        entities = data["entities"]

        # Validate types
        if not isinstance(content, str):
            raise TypeError(f"'content' must be str, got {type(content).__name__}")
        if not isinstance(keywords, list):
            raise TypeError(f"'keywords' must be list, got {type(keywords).__name__}")
        if not isinstance(entities, dict):
            raise TypeError(f"'entities' must be dict, got {type(entities).__name__}")

        # Format the enriched string
        enriched = self._format_enriched_content(content, keywords, entities)

        return enriched

    def _format_enriched_content(
        self, content: str, keywords: list[str], entities: dict[str, list[str]]
    ) -> str:
        """Format keywords and entities into enriched content string.

        This is a stateless helper method that performs pure string formatting.
        It combines keywords and entities into a prefix, then prepends the
        prefix to the original content.

        Args:
            content: Original content text
            keywords: List of keyword strings
            entities: Dict mapping entity type to list of entity texts

        Returns:
            Formatted enriched string: "keywords | entities\n\noriginal_content"
            If no keywords or entities, returns original content unchanged.
        """
        prefix_parts = []

        # Format keywords as comma-separated list (first 7 only)
        if keywords:
            keyword_str = ", ".join(keywords[:7])
            prefix_parts.append(keyword_str)

        # Format entities as flat list (first 2 per type, max 5 total, NO type labels)
        if entities:
            entity_values = []
            for label, values in entities.items():
                entity_values.extend(values[:2])
            if entity_values:
                prefix_parts.append(", ".join(entity_values[:5]))

        # Combine all parts with " | " separator
        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            enriched = f"{prefix}\n\n{content}"
        else:
            # No keywords or entities - return original content
            enriched = content

        return enriched
