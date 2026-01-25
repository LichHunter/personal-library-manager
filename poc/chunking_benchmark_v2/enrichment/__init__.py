"""Chunk enrichment module for enhanced retrieval."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from .provider import (
    call_llm,
    call_ollama,
    get_provider,
    LLMProvider,
    OllamaProvider,
    AnthropicProvider,
)


@dataclass
class EnrichmentResult:
    """Result of enriching a chunk."""

    original_content: str
    enhanced_content: str
    enrichment_type: str
    metadata: dict = field(default_factory=dict)
    keywords: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    summary: str = ""
    entities: dict = field(default_factory=dict)
    contextual_prefix: str = ""


class Enricher(ABC):
    """Base class for chunk enrichers."""

    def __init__(self, model: str = "llama3.2:3b"):
        self.model = model

    @property
    @abstractmethod
    def enrichment_type(self) -> str:
        """Name of this enrichment type."""
        pass

    @abstractmethod
    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        """Enrich a chunk with additional information.

        Args:
            content: Original chunk content
            context: Optional context dict with doc_title, section, etc.

        Returns:
            EnrichmentResult with original and enhanced content
        """
        pass

    def enrich_many(
        self, contents: list[str], contexts: Optional[list[dict]] = None
    ) -> list[EnrichmentResult]:
        """Enrich multiple chunks."""
        if contexts is None:
            contexts = [None] * len(contents)
        return [self.enrich(content, ctx) for content, ctx in zip(contents, contexts)]


from .keywords import KeywordEnricher
from .contextual import ContextualEnricher
from .questions import QuestionEnricher
from .summary import SummaryEnricher
from .entities import EntityEnricher
from .cache import EnrichmentCache
from .fast import FastEnricher

__all__ = [
    "call_llm",
    "call_ollama",
    "get_provider",
    "LLMProvider",
    "OllamaProvider",
    "AnthropicProvider",
    "EnrichmentResult",
    "Enricher",
    "KeywordEnricher",
    "ContextualEnricher",
    "QuestionEnricher",
    "SummaryEnricher",
    "EntityEnricher",
    "EnrichmentCache",
    "FastEnricher",
]
