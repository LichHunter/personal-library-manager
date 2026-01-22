#!/usr/bin/env python3
"""
Batch generate test dataset from Wikipedia articles.

Usage:
    python -m test_data.batch_generate [--output OUTPUT_DIR] [--questions N] [--no-gt]

This script:
1. Fetches a predefined list of Wikipedia articles
2. Saves each document as both JSON (structured) and Markdown (full text)
3. Optionally generates ground truth Q&A pairs
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from .ground_truth import GroundTruthGenerator
from .models import Document, Section
from .sources.wikipedia import WikipediaSource

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Diverse set of Wikipedia articles for testing retrieval
# Categories: Technology, Science, History, Arts, Geography, People
ARTICLE_LIST = [
    # Programming Languages (familiar domain for testing)
    "Python (programming language)",
    "Rust (programming language)",
    "Go (programming language)",
    "TypeScript",
    "JavaScript",
    
    # Computer Science Concepts
    "Machine learning",
    "Artificial neural network",
    "Database",
    "Operating system",
    "Computer network",
    
    # Science
    "Quantum mechanics",
    "Theory of relativity",
    "DNA",
    "Climate change",
    "Black hole",
    
    # History
    "World War II",
    "Ancient Rome",
    "Industrial Revolution",
    "Renaissance",
    "Cold War",
    
    # Geography
    "Amazon rainforest",
    "Pacific Ocean",
    "Mount Everest",
    "Sahara",
    "Great Barrier Reef",
    
    # Notable People
    "Albert Einstein",
    "Marie Curie",
    "Leonardo da Vinci",
    "Nikola Tesla",
    "Ada Lovelace",
    
    # Arts & Culture
    "Jazz",
    "Impressionism",
    "Film noir",
    "Greek mythology",
    "Shakespeare",
    
    # Technology & Engineering
    "Electric vehicle",
    "Solar energy",
    "Internet",
    "Smartphone",
    "Space exploration",
    
    # Biology & Medicine
    "Human brain",
    "Immune system",
    "Evolution",
    "Antibiotic",
    "Photosynthesis",
    
    # Economics & Society
    "Capitalism",
    "United Nations",
    "Democracy",
    "Globalization",
    "Cryptocurrency",
]


def section_to_markdown(section: Section, doc_title: str = "") -> str:
    """Convert a Section to markdown format."""
    lines = []
    
    # Heading level based on section level
    heading_prefix = "#" * (section.level + 1)  # +1 because level 1 = ##
    lines.append(f"{heading_prefix} {section.heading}")
    lines.append("")
    
    if section.content:
        lines.append(section.content)
        lines.append("")
    
    for subsection in section.subsections:
        lines.append(section_to_markdown(subsection, doc_title))
    
    return "\n".join(lines)


def document_to_markdown(doc: Document) -> str:
    """Convert a Document to full markdown format."""
    lines = []
    
    # Title
    lines.append(f"# {doc.title}")
    lines.append("")
    
    # Metadata
    lines.append(f"> Source: {doc.source}")
    lines.append(f"> Document ID: {doc.id}")
    lines.append("")
    
    # Summary
    if doc.summary:
        lines.append("## Summary")
        lines.append("")
        lines.append(doc.summary)
        lines.append("")
    
    # Sections
    for section in doc.sections:
        lines.append(section_to_markdown(section, doc.title))
    
    return "\n".join(lines)


def save_document(doc: Document, output_dir: Path) -> tuple[Path, Path]:
    """Save document as both JSON and Markdown. Returns (json_path, md_path)."""
    docs_dir = output_dir / "documents"
    docs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save JSON (structured)
    json_path = docs_dir / f"{doc.id}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(doc.to_dict(), f, indent=2, ensure_ascii=False)
    
    # Save Markdown (full text)
    md_path = docs_dir / f"{doc.id}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(document_to_markdown(doc))
    
    return json_path, md_path


def batch_generate(
    output_dir: Path,
    articles: List[str] | None = None,
    generate_gt: bool = True,
    questions_per_doc: int = 3,
    ollama_model: str = "llama3.2:3b",
) -> dict:
    """
    Batch generate test dataset.
    
    Returns a summary dict with stats.
    """
    if articles is None:
        articles = ARTICLE_LIST
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize sources
    wiki_source = WikipediaSource()
    gt_generator = GroundTruthGenerator(model=ollama_model) if generate_gt else None
    
    # Track results
    documents: List[Document] = []
    ground_truths = []
    failed_articles = []
    
    total = len(articles)
    
    # Fetch articles
    logger.info(f"Fetching {total} Wikipedia articles...")
    for i, title in enumerate(articles, 1):
        logger.info(f"[{i}/{total}] Fetching: {title}")
        
        try:
            doc = wiki_source.fetch(title)
            if doc is None:
                logger.warning(f"  -> Article not found: {title}")
                failed_articles.append({"title": title, "error": "not found"})
                continue
            
            # Save document
            json_path, md_path = save_document(doc, output_dir)
            documents.append(doc)
            
            num_sections = len(doc.get_all_sections())
            logger.info(f"  -> Saved: {doc.id} ({num_sections} sections)")
            
        except Exception as e:
            logger.error(f"  -> Failed: {e}")
            failed_articles.append({"title": title, "error": str(e)})
    
    logger.info(f"Fetched {len(documents)}/{total} articles successfully")
    
    # Generate ground truth (if enabled)
    if generate_gt and documents:
        logger.info(f"Generating ground truth Q&A pairs...")
        for i, doc in enumerate(documents, 1):
            logger.info(f"[{i}/{len(documents)}] Generating Q&A for: {doc.title}")
            try:
                gts = gt_generator.generate(doc, num_questions=questions_per_doc)
                ground_truths.extend(gts)
                logger.info(f"  -> Generated {len(gts)} Q&A pairs")
            except Exception as e:
                logger.error(f"  -> Failed to generate Q&A: {e}")
        
        # Save ground truth
        gt_path = output_dir / "ground_truth.json"
        with open(gt_path, "w", encoding="utf-8") as f:
            json.dump([gt.to_dict() for gt in ground_truths], f, indent=2, ensure_ascii=False)
        logger.info(f"Saved {len(ground_truths)} ground truth entries to {gt_path}")
    
    # Create manifest
    manifest = {
        "created_at": datetime.now().isoformat(),
        "num_documents": len(documents),
        "num_ground_truths": len(ground_truths),
        "num_failed": len(failed_articles),
        "documents": [
            {
                "id": doc.id,
                "title": doc.title,
                "source": doc.source,
                "num_sections": len(doc.get_all_sections()),
                "files": {
                    "json": f"documents/{doc.id}.json",
                    "markdown": f"documents/{doc.id}.md",
                },
            }
            for doc in documents
        ],
        "failed_articles": failed_articles,
        "ground_truth_stats": {
            "total": len(ground_truths),
            "by_difficulty": {
                "easy": sum(1 for gt in ground_truths if gt.difficulty == "easy"),
                "medium": sum(1 for gt in ground_truths if gt.difficulty == "medium"),
                "hard": sum(1 for gt in ground_truths if gt.difficulty == "hard"),
            },
            "by_query_type": {
                "factual": sum(1 for gt in ground_truths if gt.query_type == "factual"),
                "synthesis": sum(1 for gt in ground_truths if gt.query_type == "synthesis"),
                "comparison": sum(1 for gt in ground_truths if gt.query_type == "comparison"),
            },
        } if ground_truths else None,
    }
    
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved manifest to {manifest_path}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH GENERATION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Documents: {len(documents)}/{total}")
    logger.info(f"Ground truths: {len(ground_truths)}")
    logger.info(f"Failed: {len(failed_articles)}")
    logger.info(f"Output: {output_dir}")
    
    return manifest


def main():
    parser = argparse.ArgumentParser(
        description="Batch generate test dataset from Wikipedia articles"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("./output"),
        help="Output directory (default: ./output)",
    )
    parser.add_argument(
        "--questions", "-q",
        type=int,
        default=3,
        help="Number of Q&A pairs per document (default: 3)",
    )
    parser.add_argument(
        "--no-gt",
        action="store_true",
        help="Skip ground truth generation (faster, just fetch docs)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="llama3.2:3b",
        help="Ollama model for Q&A generation (default: llama3.2:3b)",
    )
    parser.add_argument(
        "--articles",
        type=str,
        nargs="+",
        help="Custom list of article titles (default: use built-in list)",
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of articles to fetch (for testing)",
    )
    
    args = parser.parse_args()
    
    articles = args.articles if args.articles else ARTICLE_LIST
    if args.limit:
        articles = articles[:args.limit]
    
    batch_generate(
        output_dir=args.output,
        articles=articles,
        generate_gt=not args.no_gt,
        questions_per_doc=args.questions,
        ollama_model=args.model,
    )


if __name__ == "__main__":
    main()
