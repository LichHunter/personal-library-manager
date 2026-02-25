"""Fast keyword and entity extraction without LLM.

Uses YAKE for keyword extraction and spaCy for named entity recognition.
This is ~500-1000x faster than LLM-based enrichment.
"""

import re
import time
from typing import Optional

from . import Enricher, EnrichmentResult

_yake_extractor = None
_spacy_nlp = None

CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")


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


def _get_spacy_nlp():
    """Lazy-load spaCy model."""
    global _spacy_nlp
    if _spacy_nlp is None:
        import spacy

        try:
            _spacy_nlp = spacy.load("en_core_web_sm")
        except OSError:
            from spacy.cli import download

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


class FastEnricher(Enricher):
    """Fast keyword and entity extraction using YAKE + spaCy.

    This enricher combines:
    - YAKE: Statistical keyword extraction (no ML, very fast)
    - spaCy NER: Named entity recognition for ORG, PRODUCT, PERSON, etc.

    It's designed to be a fast alternative to LLM-based enrichment,
    trading some quality for 500-1000x speed improvement.

    Args:
        max_keywords: Maximum number of keywords to extract (default 10)
        include_entities: Whether to include spaCy NER entities (default True)
        entity_types: Which entity types to extract. Default includes
                     ORG, PRODUCT, GPE, PERSON, WORK_OF_ART, LAW, EVENT
        min_text_length: Minimum text length to process (default 50 chars)
        debug: Enable detailed debug logging
    """

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

    def __init__(
        self,
        model: str = "fast",
        max_keywords: int = 10,
        include_entities: bool = True,
        entity_types: set[str] | None = None,
        min_text_length: int = 50,
        debug: bool = False,
        **kwargs,
    ):
        super().__init__(model)
        self.max_keywords = max_keywords
        self.include_entities = include_entities
        self.entity_types = entity_types or self.DEFAULT_ENTITY_TYPES
        self.min_text_length = min_text_length
        self.debug = debug
        self._total_processed = 0
        self._total_time = 0.0

    @property
    def enrichment_type(self) -> str:
        return "fast"

    def _debug_log(self, msg: str):
        if self.debug:
            from logger import get_logger

            get_logger().debug(f"[fast-enricher] {msg}")

    def _trace_log(self, msg: str):
        if self.debug:
            from logger import get_logger

            get_logger().trace(f"[fast-enricher] {msg}")

    def enrich(self, content: str, context: Optional[dict] = None) -> EnrichmentResult:
        """Extract keywords and entities from content.

        Args:
            content: Text to enrich
            context: Optional context dict (doc_title, section, etc.)

        Returns:
            EnrichmentResult with keywords and entities as metadata
        """
        start_time = time.perf_counter()

        content_preview = content[:500].replace("\n", " ") if content else ""
        self._trace_log(f"INPUT [{len(content)} chars]: {content_preview}...")

        if not content or len(content.strip()) < self.min_text_length:
            self._debug_log(f"Skipping short content: {len(content)} chars")
            return EnrichmentResult(
                original_content=content,
                enhanced_content=content,
                enrichment_type=self.enrichment_type,
                keywords=[],
                entities={},
                metadata={"skipped": True, "reason": "too_short"},
            )

        code_ratio = _calculate_code_ratio(content)
        text_for_nlp = _remove_code_blocks(content) if code_ratio > 0.3 else content

        keywords = []
        entities = {}

        try:
            yake_start = time.perf_counter()
            yake_extractor = _get_yake_extractor()
            yake_results = yake_extractor.extract_keywords(text_for_nlp)
            keywords = [kw for kw, score in yake_results[: self.max_keywords]]
            yake_elapsed = time.perf_counter() - yake_start

            self._debug_log(
                f"YAKE extracted {len(keywords)} keywords: {keywords[:5]}..."
            )

            if yake_results:
                scored_keywords = [
                    f"{kw}({score:.3f})"
                    for kw, score in yake_results[: self.max_keywords]
                ]
                self._trace_log(f"YAKE keywords with scores: {scored_keywords}")
            self._trace_log(f"YAKE extraction took {yake_elapsed * 1000:.1f}ms")
        except Exception as e:
            self._debug_log(f"YAKE error: {e}")

        if self.include_entities:
            try:
                spacy_start = time.perf_counter()
                nlp = _get_spacy_nlp()
                doc = nlp(text_for_nlp[:5000])

                for ent in doc.ents:
                    if ent.label_ in self.entity_types:
                        if ent.label_ not in entities:
                            entities[ent.label_] = []
                        if ent.text not in entities[ent.label_]:
                            entities[ent.label_].append(ent.text)

                for label in entities:
                    entities[label] = entities[label][:5]

                spacy_elapsed = time.perf_counter() - spacy_start
                entity_count = sum(len(v) for v in entities.values())
                self._debug_log(f"spaCy extracted {entity_count} entities: {entities}")

                for label, values in entities.items():
                    self._trace_log(f"spaCy {label}: {values}")
                self._trace_log(f"spaCy NER took {spacy_elapsed * 1000:.1f}ms")
            except Exception as e:
                self._debug_log(f"spaCy error: {e}")

        prefix_parts = []

        if keywords:
            prefix_parts.append(", ".join(keywords[:7]))

        if entities:
            entity_values = []
            for label, values in entities.items():
                entity_values.extend(values[:2])
            if entity_values:
                prefix_parts.append(", ".join(entity_values[:5]))

        prefix = ""
        if prefix_parts:
            prefix = " | ".join(prefix_parts)
            enhanced = f"{prefix}\n\n{content}"
            self._trace_log(f"Generated prefix: {prefix}")
        else:
            enhanced = content
            self._trace_log("No prefix generated (no keywords or entities)")

        elapsed = time.perf_counter() - start_time
        self._total_processed += 1
        self._total_time += elapsed

        prefix_len = len(prefix)
        self._debug_log(
            f"Enriched in {elapsed * 1000:.1f}ms | "
            f"keywords={len(keywords)} entities={sum(len(v) for v in entities.values())} | "
            f"prefix_len={prefix_len}"
        )

        return EnrichmentResult(
            original_content=content,
            enhanced_content=enhanced,
            enrichment_type=self.enrichment_type,
            keywords=keywords,
            entities=entities,
            metadata={
                "code_ratio": round(code_ratio, 2),
                "processing_time_ms": round(elapsed * 1000, 2),
                "keyword_count": len(keywords),
                "entity_count": sum(len(v) for v in entities.values()),
            },
        )

    def get_stats(self) -> dict:
        """Get processing statistics."""
        return {
            "total_processed": self._total_processed,
            "total_time_s": round(self._total_time, 2),
            "avg_time_ms": round(
                self._total_time / max(1, self._total_processed) * 1000, 2
            ),
        }
