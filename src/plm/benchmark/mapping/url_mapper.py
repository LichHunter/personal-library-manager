from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator
from urllib.parse import urlparse

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class ChunkInfo:
    chunk_id: str
    doc_id: str
    source_file: str
    heading: str | None
    heading_id: str | None
    start_char: int | None
    end_char: int | None


def normalize_url(url: str) -> tuple[str, str | None]:
    """
    Normalize URL and extract path + fragment.
    
    Returns: (normalized_path, fragment)
    - Converts http:// to https://
    - Removes trailing slashes (except root)
    - Extracts fragment separately
    """
    parsed = urlparse(url)
    
    path = parsed.path.rstrip("/") if parsed.path != "/" else parsed.path
    
    fragment = parsed.fragment if parsed.fragment else None
    
    return path, fragment


def slugify_heading(heading: str) -> str:
    """
    Convert heading text to anchor slug.
    
    Rules:
    1. Remove markdown heading prefix (##, ###, etc.)
    2. Convert to lowercase
    3. Replace spaces with hyphens
    4. Remove special characters except hyphens
    5. Collapse multiple hyphens to single
    """
    text = heading.strip()
    
    text = re.sub(r"^#+\s+", "", text)
    
    text = text.lower()
    
    text = re.sub(r"\s+", "-", text)
    
    text = re.sub(r"[^a-z0-9\-]", "", text)
    
    text = re.sub(r"-+", "-", text)
    
    text = text.strip("-")
    
    return text


def read_chunks_from_db(db_path: Path) -> Iterator[ChunkInfo]:
    """Read all chunks from SQLite database with document info."""
    log.info(f"Opening database: {db_path}")
    
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    
    try:
        cursor = conn.execute("PRAGMA table_info(chunks)")
        columns = {row[1] for row in cursor.fetchall()}
        
        has_heading_id = "heading_id" in columns
        has_chunk_index = "chunk_index" in columns
        
        if has_heading_id:
            query = """
            SELECT 
                c.id as chunk_id,
                c.doc_id,
                d.source_file,
                c.heading,
                c.heading_id,
                c.start_char,
                c.end_char
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            ORDER BY c.doc_id, c.chunk_index
            """
        else:
            query = """
            SELECT 
                c.id as chunk_id,
                c.doc_id,
                d.source_file,
                c.heading,
                NULL as heading_id,
                c.start_char,
                c.end_char
            FROM chunks c
            JOIN documents d ON c.doc_id = d.id
            ORDER BY c.doc_id
            """
        
        cursor = conn.execute(query)
        batch_size = 1000
        row_count = 0
        
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            
            for row in rows:
                row_count += 1
                if row_count % 5000 == 0:
                    log.info(f"Processed {row_count} chunks...")
                
                yield ChunkInfo(
                    chunk_id=row["chunk_id"],
                    doc_id=row["doc_id"],
                    source_file=row["source_file"],
                    heading=row["heading"],
                    heading_id=row["heading_id"],
                    start_char=row["start_char"],
                    end_char=row["end_char"],
                )
        
        log.info(f"Total chunks read: {row_count}")
    finally:
        conn.close()


def build_mappings(chunks: list[ChunkInfo], unmapped_log: Path) -> tuple[dict, dict, dict]:
    """
    Build three mapping structures from chunks.
    
    Returns:
    - url_to_docid: {url_path: [doc_id, ...]}
    - url_to_chunks: {url_path: [chunk_id, ...]}
    - anchor_to_heading: {anchor_slug: [{doc_id, heading_id, heading_text}, ...]}
    """
    url_to_docid: dict[str, list[str]] = defaultdict(list)
    url_to_chunks: dict[str, list[str]] = defaultdict(list)
    anchor_to_heading: dict[str, list[dict]] = defaultdict(list)
    unmapped_urls: list[str] = []
    
    doc_id_to_source: dict[str, str] = {}
    
    for chunk in chunks:
        doc_id = chunk.doc_id
        source_file = chunk.source_file
        
        if doc_id not in doc_id_to_source:
            doc_id_to_source[doc_id] = source_file
        
        if source_file.startswith("http://") or source_file.startswith("https://"):
            path, fragment = normalize_url(source_file)
            
            if doc_id not in url_to_docid[path]:
                url_to_docid[path].append(doc_id)
            
            url_to_chunks[path].append(chunk.chunk_id)
        else:
            unmapped_urls.append(f"{source_file} (not a URL)")
        
        if chunk.heading and chunk.heading_id:
            slug = slugify_heading(chunk.heading)
            
            entry = {
                "doc_id": doc_id,
                "heading_id": chunk.heading_id,
                "heading_text": chunk.heading,
            }
            
            if entry not in anchor_to_heading[slug]:
                anchor_to_heading[slug].append(entry)
    
    if unmapped_urls:
        log.info(f"Found {len(unmapped_urls)} unmapped URLs, writing to log")
        unmapped_log.parent.mkdir(parents=True, exist_ok=True)
        unmapped_log.write_text("\n".join(unmapped_urls) + "\n")
    
    return (
        dict(url_to_docid),
        dict(url_to_chunks),
        dict(anchor_to_heading),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build URL-to-Chunk mappings from indexed corpus.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python -m plm.benchmark.mapping.url_mapper \\
    --db /path/to/index.db \\
    --output artifacts/benchmark/mappings/
""",
    )
    parser.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to SQLite index database",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Output directory for mapping JSON files",
    )
    
    args = parser.parse_args(argv)
    
    if not args.db.exists():
        log.error(f"Database file not found: {args.db}")
        return 1
    
    args.output.mkdir(parents=True, exist_ok=True)
    
    log.info("=" * 60)
    log.info("URL-to-Chunk Mapping Builder")
    log.info("=" * 60)
    log.info(f"Database: {args.db}")
    log.info(f"Output: {args.output}")
    
    log.info("-" * 60)
    log.info("Reading chunks from database...")
    
    chunks: list[ChunkInfo] = []
    try:
        for chunk in read_chunks_from_db(args.db):
            chunks.append(chunk)
    except sqlite3.OperationalError as e:
        log.error(f"Database error: {e}")
        log.error("Make sure the database has chunks and documents tables")
        return 1
    
    if not chunks:
        log.warning("No chunks found in database!")
        return 1
    
    log.info("-" * 60)
    log.info("Building mappings...")
    
    unmapped_log = args.output / "unmapped_urls.log"
    url_to_docid, url_to_chunks, anchor_to_heading = build_mappings(chunks, unmapped_log)
    
    log.info("-" * 60)
    log.info("MAPPING STATISTICS")
    log.info("-" * 60)
    log.info(f"Total chunks:           {len(chunks)}")
    log.info(f"Unique URL paths:       {len(url_to_docid)}")
    log.info(f"Unique anchor slugs:    {len(anchor_to_heading)}")
    
    log.info("-" * 60)
    log.info("Writing mapping files...")
    
    url_to_docid_file = args.output / "url_to_docid.json"
    url_to_chunks_file = args.output / "url_to_chunks.json"
    anchor_to_heading_file = args.output / "anchor_to_heading.json"
    
    url_to_docid_file.write_text(json.dumps(url_to_docid, indent=2, ensure_ascii=False))
    log.info(f"Wrote {url_to_docid_file}")
    
    url_to_chunks_file.write_text(json.dumps(url_to_chunks, indent=2, ensure_ascii=False))
    log.info(f"Wrote {url_to_chunks_file}")
    
    anchor_to_heading_file.write_text(json.dumps(anchor_to_heading, indent=2, ensure_ascii=False))
    log.info(f"Wrote {anchor_to_heading_file}")
    
    if unmapped_log.exists():
        log.info(f"Wrote {unmapped_log}")
    
    log.info("=" * 60)
    log.info("MAPPING COMPLETE")
    log.info("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
