"""GLiNER adapter — transforms GLiNER output to ContentEnricher format.

This module bridges the fast extraction system output (GLiNER entities + YAKE keywords)
to the format expected by the ContentEnricher component.
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from plm.extraction.fast.document_processor import DocumentResult
    from plm.extraction.fast.gliner import ExtractedEntity


def gliner_to_enricher_format(entities: list[ExtractedEntity]) -> dict[str, list[str]]:
    """Transform GLiNER entities to ContentEnricher format.

    Groups entities by their label and returns a dict mapping labels to entity texts.
    Deduplicates entity texts within each label.

    Args:
        entities: List of ExtractedEntity from GLiNER.

    Returns:
        Dict mapping entity labels to lists of unique entity texts.
        Example: {'library': ['React', 'Vue'], 'framework': ['Next.js']}
    """
    result: dict[str, list[str]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    
    for entity in entities:
        if entity.text not in seen[entity.label]:
            result[entity.label].append(entity.text)
            seen[entity.label].add(entity.text)
    
    return dict(result)


def document_result_to_chunks(doc: DocumentResult) -> list[dict]:
    """Flatten DocumentResult to list of chunk dicts for storage/enrichment.

    Each chunk dict contains all information needed for indexing:
    - content: The raw chunk text
    - keywords: YAKE-extracted keywords (list[str])
    - entities: GLiNER entities grouped by label (dict[str, list[str]])
    - heading: The section heading this chunk belongs to
    - start_char: Character offset start
    - end_char: Character offset end

    Args:
        doc: DocumentResult from the fast extraction system.

    Returns:
        List of dicts, one per chunk, ready for ContentEnricher/storage.
    """
    chunks = []
    for section in doc.headings:
        for chunk in section.chunks:
            chunks.append({
                'content': chunk.text,
                'keywords': chunk.keywords,
                'entities': gliner_to_enricher_format(chunk.entities),
                'heading': section.heading,
                'start_char': chunk.start_char,
                'end_char': chunk.end_char,
            })
    return chunks
