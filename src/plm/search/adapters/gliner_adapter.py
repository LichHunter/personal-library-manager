"""GLiNER adapter — transforms GLiNER output to ContentEnricher format.

This module bridges the fast extraction system output (GLiNER entities + YAKE keywords)
to the format expected by the ContentEnricher component.

Also provides utilities for loading extraction output directories into the
format expected by HybridRetriever.batch_ingest().
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from plm.search.service.watcher import json_to_chunks

if TYPE_CHECKING:
    from plm.extraction.fast.document_processor import DocumentResult
    from plm.extraction.fast.gliner import ExtractedEntity

logger = logging.getLogger(__name__)


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


def load_extraction_directory(json_dir: str | Path) -> list[dict]:
    """Load extraction JSON files from a directory into batch_ingest format.

    Reads all .json files from the given directory (including processed/
    subdirectory) and converts them into the format expected by
    HybridRetriever.batch_ingest().

    The doc_id is derived from the JSON filename stem (e.g.,
    "concepts_architecture_leases.json" -> "concepts_architecture_leases"),
    which matches the question format used in benchmarks.

    Args:
        json_dir: Path to directory containing extraction JSON files.
            Also checks json_dir/processed/ for already-processed files.

    Returns:
        List of document dicts ready for batch_ingest(), each with:
            - 'doc_id': Filename stem
            - 'source_file': Original source file from JSON metadata
            - 'chunks': List of chunk dicts from json_to_chunks()
    """
    json_dir = Path(json_dir)
    documents: list[dict] = []

    json_files: list[Path] = []
    for search_dir in [json_dir, json_dir / "processed"]:
        if search_dir.is_dir():
            json_files.extend(sorted(search_dir.glob("*.json")))

    for json_path in json_files:
        doc_dict = json.loads(json_path.read_text(encoding="utf-8"))
        chunks = json_to_chunks(doc_dict)
        if not chunks:
            logger.warning(f"No chunks extracted from {json_path}")
            continue

        documents.append({
            "doc_id": json_path.stem,
            "source_file": doc_dict.get("source_file", str(json_path)),
            "chunks": chunks,
        })

    logger.info(f"Loaded {len(documents)} documents from {json_dir}")
    return documents
