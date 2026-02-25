"""Entity extraction component using spaCy NER.

This module provides a stateless component that wraps spaCy named entity recognition
from FastEnricher. It follows the Component protocol for pipeline integration.

The component accepts a dict with a 'content' field and returns a new dict with
an added 'entities' field, preserving all input fields (Unix pipe accumulation).

Example:
    >>> from entity_extractor import EntityExtractor
    >>> extractor = EntityExtractor()
    >>> data = {'content': 'Google Cloud Platform offers Kubernetes Engine'}
    >>> result = extractor.process(data)
    >>> print(result['entities'])
    {'ORG': ['Google Cloud Platform'], 'PRODUCT': ['Kubernetes Engine']}
    >>> print(result['content'])  # Original content preserved
    'Google Cloud Platform offers Kubernetes Engine'

Can be wrapped with CacheableComponent for caching:
    >>> from cache import CacheableComponent
    >>> cached = CacheableComponent(EntityExtractor(), cache_dir="cache/entities")
    >>> result = cached.process(data)  # First call: runs extraction
    >>> result = cached.process(data)  # Second call: returns cached result
"""

import re
from typing import Any

from ..utils.logger import get_logger

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")

# Module-level cache for spaCy model (matches FastEnricher)
_spacy_nlp = None


def _get_spacy_nlp():
    """Lazy-load spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli.download import download

            download("en_core_web_sm")
            _spacy_nlp = spacy.load("en_core_web_sm")
    return _spacy_nlp


def _calculate_code_ratio(text: str) -> float:
    """Calculate ratio of code blocks to total text."""
    if not text:
        return 0.0
    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    return code_chars / len(text)


def _remove_code_blocks(text: str) -> str:
    """Remove code blocks from text for NLP processing."""
    return CODE_BLOCK_PATTERN.sub(" ", text)


class EntityExtractor:
    """Extract named entities from content using spaCy NER.

    This component wraps spaCy (Spacy) named entity recognition for identifying
    organizations, products, people, and technical entities. It's designed to be
    fast and stateless, with spaCy model loaded fresh in each process() call.

    The component accepts a dict with a 'content' field and returns a new dict
    with an added 'entities' field. All input fields are preserved in the output
    (Unix pipe accumulation pattern).

    Attributes:
        entity_types: Set of entity types to extract (default: ORG, PRODUCT, PERSON, TECH)

    Example:
        >>> extractor = EntityExtractor()
        >>> data = {
        ...     'content': 'Google Cloud Platform offers Kubernetes Engine',
        ...     'source': 'docs.md'
        ... }
        >>> result = extractor.process(data)
        >>> # result = {
        >>> #     'content': 'Google Cloud Platform offers Kubernetes Engine',
        >>> #     'source': 'docs.md',
        >>> #     'entities': {
        >>> #         'ORG': ['Google Cloud Platform'],
        >>> #         'PRODUCT': ['Kubernetes Engine']
        >>> #     }
        >>> # }
    """

    # Default entity types to extract
    DEFAULT_ENTITY_TYPES = {
        "ORG",
        "PRODUCT",
        "GPE",
        "PERSON",
        "WORK_OF_ART",
        "LAW",
        "EVENT",
        "FAC",
        "NORP",
    }

    def __init__(self, entity_types: set[str] | None = None):
        """Initialize EntityExtractor.

        Args:
            entity_types: Set of entity types to extract (default: ORG, PRODUCT, PERSON, TECH)
        """
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self._log = get_logger()
        self._log.debug(
            f"[{self.__class__.__name__}] initialized with entity_types={self.entity_types}"
        )

    def process(self, data: dict[str, Any]) -> dict[str, Any]:
        """Extract entities from content and return enriched dict.

        Implements the Component protocol. Accepts a dict with a 'content' field,
        extracts named entities using spaCy, and returns a new dict with an added
        'entities' field. All input fields are preserved.

        Args:
            data: Input dict with 'content' field containing text to extract entities from

        Returns:
            New dict with added 'entities' field (dict mapping entity type to list of entity texts).
            All input fields are preserved.

        Raises:
            KeyError: If 'content' field is missing from input dict
            ImportError: If spacy package is not installed
            Exception: Any exception from spaCy NER

        Example:
            >>> extractor = EntityExtractor()
            >>> data = {'content': 'Google Cloud Platform offers Kubernetes', 'doc_id': '123'}
            >>> result = extractor.process(data)
            >>> assert 'entities' in result
            >>> assert 'doc_id' in result  # Original field preserved
            >>> assert isinstance(result['entities'], dict)
        """
        # Validate input
        if "content" not in data:
            raise KeyError("Input dict must have 'content' field")

        content = data["content"]
        content_len = len(content) if content else 0
        self._log.debug(
            f"[{self.__class__.__name__}] processing content, length={content_len}"
        )

        # Handle empty or short content
        if (
            not content or len(content.strip()) < 50
        ):  # Min length matches FastEnricher.min_text_length default
            # Return input dict with empty entities
            self._log.trace(
                f"[{self.__class__.__name__}] content too short, returning empty entities"
            )
            result = dict(data)
            result["entities"] = {}
            return result

        # Extract entities using spaCy
        entities = self._extract_entities(content, self.entity_types)

        # Create output dict preserving all input fields
        result = dict(data)
        result["entities"] = entities
        entity_count = sum(len(v) for v in entities.values())
        self._log.debug(
            f"[{self.__class__.__name__}] extracted {entity_count} entities across {len(entities)} types"
        )
        self._log.trace(
            f"[{self.__class__.__name__}] entity types found: {list(entities.keys())}"
        )

        return result

    def _extract_entities(
        self, content: str, entity_types: set[str]
    ) -> dict[str, list[str]]:
        """Extract named entities from content using spaCy.

        Uses module-level cached spaCy model for performance.
        spaCy model is loaded once and reused across all calls.

        Args:
            content: Text to extract entities from
            entity_types: Set of entity types to extract

        Returns:
            Dict mapping entity type to list of entity texts (deduplicated)

        Raises:
            ImportError: If spacy package is not installed
        """
        nlp = _get_spacy_nlp()

        code_ratio = _calculate_code_ratio(content)
        text_for_nlp = _remove_code_blocks(content) if code_ratio > 0.3 else content
        doc = nlp(text_for_nlp[:5000])

        # Extract entities
        entities: dict[str, list[str]] = {}
        for ent in doc.ents:
            if ent.label_ in entity_types:
                if ent.label_ not in entities:
                    entities[ent.label_] = []
                # Avoid duplicates
                if ent.text not in entities[ent.label_]:
                    entities[ent.label_].append(ent.text)

        for label in entities:
            entities[label] = entities[label][:5]

        return entities
