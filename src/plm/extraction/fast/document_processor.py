"""Document processor — reads a file, chunks it, extracts entities and keywords, returns structured output.

Also provides ``prepare_chunks_for_search`` which is the bridge between any
chunking strategy and ``HybridRetriever.batch_ingest``: it takes raw text
chunks and runs YAKE keyword + NER entity extraction, returning dicts ready
for ingestion.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from plm.extraction.chunking.gliner_chunker import GLiNERChunker
from plm.extraction.fast.gliner import ExtractedEntity, extract_entities

if TYPE_CHECKING:
    from gliner import GLiNER

logger = logging.getLogger(__name__)

# YAKE keyword extraction setup
CODE_BLOCK_PATTERN = re.compile(r"```[\s\S]*?```|`[^`\n]+`")
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


def _extract_keywords(text: str, max_keywords: int = 10) -> list[str]:
    """Extract keywords from text using YAKE.

    Args:
        text: Text to extract keywords from.
        max_keywords: Maximum number of keywords to return.

    Returns:
        List of keyword strings ordered by relevance.
    """
    if not text or len(text.strip()) < 50:
        return []

    code_chars = sum(len(m.group()) for m in CODE_BLOCK_PATTERN.finditer(text))
    code_ratio = code_chars / len(text) if text else 0.0
    text_for_nlp = CODE_BLOCK_PATTERN.sub(" ", text) if code_ratio > 0.3 else text

    try:
        kw_extractor = _get_yake_extractor()
        keywords_with_scores = kw_extractor.extract_keywords(text_for_nlp)
        return [kw for kw, score in keywords_with_scores[:max_keywords]]
    except Exception:
        logger.exception("YAKE extraction failed")
        return []


@dataclass
class ChunkResult:
    text: str
    terms: list[str]
    entities: list[ExtractedEntity]
    keywords: list[str]
    start_char: int
    end_char: int


@dataclass
class HeadingSection:
    heading: str
    level: int
    chunks: list[ChunkResult]


@dataclass
class DocumentResult:
    source_file: str
    headings: list[HeadingSection]
    avg_confidence: float
    total_entities: int
    is_low_confidence: bool
    error: str | None = None


def _read_file(filepath: Path) -> str:
    """Read file content with encoding fallback."""
    try:
        return filepath.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 failed for %s, falling back to latin-1", filepath)
        return filepath.read_text(encoding="latin-1")


def process_document(
    filepath: Path,
    confidence_threshold: float = 0.7,
    extraction_threshold: float = 0.3,
    model: GLiNER | None = None,
) -> DocumentResult:
    """Process a single document: chunk → extract → structure.

    Args:
        filepath: Path to the document file.
        confidence_threshold: Below this avg confidence, flag as low-confidence.
        extraction_threshold: GLiNER entity confidence threshold.
        model: Optional GLiNER model for dependency injection.

    Returns:
        DocumentResult with heading hierarchy, entities, and confidence info.
    """
    try:
        text = _read_file(filepath)
    except (FileNotFoundError, PermissionError, OSError) as exc:
        logger.error("Cannot read %s: %s", filepath, exc)
        return DocumentResult(
            source_file=str(filepath),
            headings=[],
            avg_confidence=0.0,
            total_entities=0,
            is_low_confidence=True,
            error=str(exc),
        )

    if not text.strip():
        return DocumentResult(
            source_file=str(filepath),
            headings=[],
            avg_confidence=0.0,
            total_entities=0,
            is_low_confidence=True,
        )

    chunker = GLiNERChunker()
    chunks = chunker.chunk(text, filename=filepath.name)

    # Group chunks by heading
    heading_groups: dict[str | None, list] = {}
    heading_levels: dict[str | None, int] = {}

    for chunk in chunks:
        heading_key = chunk.heading
        if heading_key not in heading_groups:
            heading_groups[heading_key] = []
            # Parse level from heading like "## Foo" → 2
            if heading_key:
                match = re.match(r"^(#{1,6})\s+", heading_key)
                heading_levels[heading_key] = len(match.group(1)) if match else 0
            else:
                heading_levels[heading_key] = 0
        heading_groups[heading_key].append(chunk)

    all_entities: list[ExtractedEntity] = []
    sections: list[HeadingSection] = []

    for heading_key, group_chunks in heading_groups.items():
        chunk_results: list[ChunkResult] = []
        for chunk in group_chunks:
            try:
                entities = extract_entities(
                    chunk.text,
                    threshold=extraction_threshold,
                    model=model,
                )
            except Exception:
                logger.exception("Extraction failed for chunk %d in %s", chunk.index, filepath)
                entities = []

            all_entities.extend(entities)
            terms = list(dict.fromkeys(e.text for e in entities))
            keywords = _extract_keywords(chunk.text)

            chunk_results.append(ChunkResult(
                text=chunk.text,
                terms=terms,
                entities=entities,
                keywords=keywords,
                start_char=chunk.start_char,
                end_char=chunk.end_char,
            ))

        display_heading = heading_key or "(root)"
        level = heading_levels.get(heading_key, 0)
        sections.append(HeadingSection(
            heading=display_heading,
            level=level,
            chunks=chunk_results,
        ))

    avg_confidence = (
        sum(e.score for e in all_entities) / len(all_entities)
        if all_entities
        else 0.0
    )

    return DocumentResult(
        source_file=str(filepath),
        headings=sections,
        avg_confidence=avg_confidence,
        total_entities=len(all_entities),
        is_low_confidence=avg_confidence < confidence_threshold,
    )
