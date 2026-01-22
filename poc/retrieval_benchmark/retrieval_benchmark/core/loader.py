"""Load documents and ground truth from test_data output."""

import json
import logging
from pathlib import Path
from typing import Optional

from .types import Document, Section, GroundTruth

logger = logging.getLogger(__name__)


def _parse_section(data: dict, document_id: str) -> Section:
    """Parse a section from JSON data."""
    return Section(
        id=data["id"],
        document_id=document_id,
        heading=data["heading"],
        level=data["level"],
        content=data["content"],
        parent_id=None,  # We could infer this from nesting if needed
    )


def _parse_sections_recursive(
    sections_data: list[dict],
    document_id: str,
) -> list[Section]:
    """Parse sections recursively, flattening the hierarchy."""
    result = []
    
    for section_data in sections_data:
        section = _parse_section(section_data, document_id)
        result.append(section)
        
        # Recursively parse subsections
        if "subsections" in section_data:
            subsections = _parse_sections_recursive(
                section_data["subsections"],
                document_id,
            )
            result.extend(subsections)
    
    return result


def load_document_from_json(json_path: Path) -> Document:
    """Load a single document from JSON file."""
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)
    
    document_id = data["id"]
    sections = _parse_sections_recursive(data.get("sections", []), document_id)
    
    # Load markdown content if available
    md_path = json_path.with_suffix(".md")
    if md_path.exists():
        with open(md_path, encoding="utf-8") as f:
            content = f.read()
    else:
        # Fallback: reconstruct content from sections
        content = f"# {data['title']}\n\n{data.get('summary', '')}\n\n"
        for section in sections:
            content += f"\n## {section.heading}\n\n{section.content}\n"
    
    return Document(
        id=document_id,
        title=data["title"],
        source=data.get("source", "unknown"),
        summary=data.get("summary", ""),
        content=content,
        sections=sections,
    )


def load_documents(
    documents_dir: str | Path,
    max_docs: Optional[int] = None,
) -> list[Document]:
    """Load all documents from a directory.
    
    Args:
        documents_dir: Path to directory containing document JSON files
        max_docs: Maximum number of documents to load (None = all)
        
    Returns:
        List of Document objects
    """
    documents_dir = Path(documents_dir)
    
    if not documents_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {documents_dir}")
    
    # Find all JSON files
    json_files = sorted(documents_dir.glob("*.json"))
    
    if max_docs is not None:
        json_files = json_files[:max_docs]
    
    documents = []
    for json_path in json_files:
        try:
            doc = load_document_from_json(json_path)
            documents.append(doc)
            logger.debug(f"Loaded document: {doc.title} ({len(doc.sections)} sections)")
        except Exception as e:
            logger.warning(f"Failed to load {json_path}: {e}")
    
    logger.info(f"Loaded {len(documents)} documents from {documents_dir}")
    return documents


def load_ground_truth(
    ground_truth_path: str | Path,
    max_queries: Optional[int] = None,
) -> list[GroundTruth]:
    """Load ground truth Q&A pairs.
    
    Args:
        ground_truth_path: Path to ground truth JSON file
        max_queries: Maximum number of queries to load (None = all)
        
    Returns:
        List of GroundTruth objects
    """
    ground_truth_path = Path(ground_truth_path)
    
    if not ground_truth_path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {ground_truth_path}")
    
    with open(ground_truth_path, encoding="utf-8") as f:
        data = json.load(f)
    
    ground_truths = []
    for item in data:
        gt = GroundTruth(
            id=item["id"],
            question=item["question"],
            answer=item["answer"],
            document_id=item["document_id"],
            section_ids=item["section_ids"],
            evidence=item["evidence"],
            difficulty=item.get("difficulty", "unknown"),
            query_type=item.get("query_type", "unknown"),
        )
        ground_truths.append(gt)
    
    if max_queries is not None:
        ground_truths = ground_truths[:max_queries]
    
    logger.info(f"Loaded {len(ground_truths)} ground truth queries")
    return ground_truths
