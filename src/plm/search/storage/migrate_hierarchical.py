#!/usr/bin/env python3
"""Migration script: Recover structured metadata from Redis and build hierarchical storage.

Reads extraction messages from Redis Streams, matches to existing chunks,
updates with keywords_json/entities_json, creates heading records, and
aggregates embeddings/metadata to heading and document levels.

Usage:
    python -m plm.search.storage.migrate_hierarchical \
        --db-path /path/to/index.db \
        --redis-url redis://localhost:6379 \
        --stream plm:extraction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def connect_redis(redis_url: str):
    import redis
    return redis.from_url(redis_url)


def read_all_messages(redis_client, stream: str) -> list[dict]:
    messages = []
    last_id = "0"
    batch_size = 100
    
    while True:
        results = redis_client.xrange(stream, min=last_id, count=batch_size)
        if not results:
            break
        
        for msg_id, data in results:
            msg_id_str = msg_id.decode() if isinstance(msg_id, bytes) else msg_id
            last_id = f"({msg_id_str}"
            
            raw = data.get(b"data") or data.get("data")
            if raw:
                raw_str = raw.decode() if isinstance(raw, bytes) else raw
                try:
                    parsed = json.loads(raw_str)
                    messages.append(parsed)
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse message {msg_id_str}")
        
        if len(results) < batch_size:
            break
    
    return messages


def normalize_source_file(source_file: str) -> str:
    """Normalize source file path for matching (basename only)."""
    return Path(source_file).name


def dedupe_keywords(keywords_list: list[list[str]]) -> list[str]:
    seen_lower: set[str] = set()
    result: list[str] = []
    for keywords in keywords_list:
        for kw in keywords:
            if kw.lower() not in seen_lower:
                result.append(kw)
                seen_lower.add(kw.lower())
    return sorted(result)


def dedupe_entities(entities_lists: list[list[dict]]) -> dict[str, list[str]]:
    by_label: dict[str, set[str]] = defaultdict(set)
    seen_lower: dict[str, set[str]] = defaultdict(set)
    
    for entities in entities_lists:
        for ent in entities:
            label = ent.get("label", "unknown")
            text = ent.get("text", "")
            if text.lower() not in seen_lower[label]:
                by_label[label].add(text)
                seen_lower[label].add(text.lower())
    
    return {label: sorted(texts) for label, texts in by_label.items()}


def aggregate_embeddings(embeddings: list[np.ndarray]) -> np.ndarray | None:
    valid = [e for e in embeddings if e is not None and len(e) > 0]
    if not valid:
        return None
    return np.mean(valid, axis=0).astype(np.float32)


def run_migration(db_path: str, redis_url: str, stream: str, dry_run: bool = False):
    from plm.search.storage.sqlite import SQLiteStorage
    
    logger.info(f"Connecting to Redis at {redis_url}")
    redis_client = connect_redis(redis_url)
    
    logger.info(f"Reading messages from stream {stream}")
    messages = read_all_messages(redis_client, stream)
    logger.info(f"Found {len(messages)} messages")
    
    if not messages:
        logger.warning("No messages found in Redis. Migration cannot proceed.")
        return
    
    logger.info(f"Opening database at {db_path}")
    storage = SQLiteStorage(db_path)
    storage.create_tables()
    
    existing_chunks = storage.get_all_chunks()
    existing_docs = storage.get_all_documents()
    logger.info(f"Found {len(existing_chunks)} existing chunks, {len(existing_docs)} documents")
    
    source_to_doc_id: dict[str, str] = {}
    for doc in existing_docs:
        src = doc.get("source_file", "")
        source_to_doc_id[normalize_source_file(src)] = doc["id"]
    
    chunk_lookup: dict[tuple[str, int, int], dict] = {}
    for chunk in existing_chunks:
        doc_id = chunk["doc_id"]
        start = chunk.get("start_char")
        end = chunk.get("end_char")
        if start is not None and end is not None:
            chunk_lookup[(doc_id, start, end)] = chunk
    
    stats = {
        "chunks_updated": 0,
        "headings_created": 0,
        "documents_updated": 0,
        "chunks_not_found": 0,
        "docs_not_found": 0,
    }
    
    total = len(messages)
    for msg_idx, msg in enumerate(messages):
        if (msg_idx + 1) % 50 == 0 or msg_idx == 0:
            logger.info(f"Processing message {msg_idx + 1}/{total}...")
        
        payload = msg.get("payload", msg)
        source_file = payload.get("source_file", "")
        headings_data = payload.get("headings", [])
        
        normalized = normalize_source_file(source_file)
        doc_id = source_to_doc_id.get(normalized)
        
        if not doc_id:
            stats["docs_not_found"] += 1
            continue
        
        doc_keywords_lists: list[list[str]] = []
        doc_entities_lists: list[list[dict]] = []
        doc_embeddings: list[np.ndarray] = []
        
        for heading_idx, heading_section in enumerate(headings_data):
            heading_text = heading_section.get("heading", "(root)")
            heading_level = heading_section.get("level", 0)
            chunks_data = heading_section.get("chunks", [])
            
            heading_id = f"{doc_id}_h{heading_idx}"
            
            heading_keywords_lists: list[list[str]] = []
            heading_entities_lists: list[list[dict]] = []
            heading_embeddings: list[np.ndarray] = []
            heading_start = None
            heading_end = None
            
            for chunk_idx, chunk_data in enumerate(chunks_data):
                start_char = chunk_data.get("start_char")
                end_char = chunk_data.get("end_char")
                keywords = chunk_data.get("keywords", [])
                entities = chunk_data.get("entities", [])
                
                if heading_start is None or (start_char is not None and start_char < heading_start):
                    heading_start = start_char
                if heading_end is None or (end_char is not None and end_char > heading_end):
                    heading_end = end_char
                
                existing = chunk_lookup.get((doc_id, start_char, end_char))
                if existing:
                    keywords_json = json.dumps(keywords) if keywords else None
                    entities_json = json.dumps(entities) if entities else None
                    
                    if not dry_run:
                        storage.update_chunk_heading(existing["id"], heading_id, chunk_idx)
                        with storage._connect() as conn:
                            conn.execute(
                                "UPDATE chunks SET keywords_json = ?, entities_json = ? WHERE id = ?",
                                (keywords_json, entities_json, existing["id"]),
                            )
                    
                    stats["chunks_updated"] += 1
                    
                    heading_keywords_lists.append(keywords)
                    heading_entities_lists.append(entities)
                    if existing.get("embedding") is not None:
                        heading_embeddings.append(existing["embedding"])
                else:
                    stats["chunks_not_found"] += 1
            
            if chunks_data:
                heading_keywords = dedupe_keywords(heading_keywords_lists)
                heading_entities = dedupe_entities(heading_entities_lists)
                heading_embedding = aggregate_embeddings(heading_embeddings)
                
                if not dry_run:
                    storage.add_heading(
                        heading_id=heading_id,
                        doc_id=doc_id,
                        heading_text=heading_text,
                        heading_level=heading_level,
                        start_char=heading_start,
                        end_char=heading_end,
                        embedding=heading_embedding,
                        keywords_json=json.dumps(heading_keywords) if heading_keywords else None,
                        entities_json=json.dumps(heading_entities) if heading_entities else None,
                    )
                
                stats["headings_created"] += 1
                
                doc_keywords_lists.append(heading_keywords)
                doc_entities_lists.extend(heading_entities_lists)
                if heading_embedding is not None:
                    doc_embeddings.append(heading_embedding)
        
        if headings_data:
            doc_keywords = dedupe_keywords(doc_keywords_lists)
            doc_entities = dedupe_entities(doc_entities_lists)
            doc_embedding = aggregate_embeddings(doc_embeddings)
            
            if not dry_run:
                storage.update_document_aggregates(
                    doc_id=doc_id,
                    embedding=doc_embedding,
                    keywords_json=json.dumps(doc_keywords) if doc_keywords else None,
                    entities_json=json.dumps(doc_entities) if doc_entities else None,
                )
            
            stats["documents_updated"] += 1
    
    logger.info("Migration complete!")
    logger.info(f"  Chunks updated: {stats['chunks_updated']}")
    logger.info(f"  Headings created: {stats['headings_created']}")
    logger.info(f"  Documents updated: {stats['documents_updated']}")
    logger.info(f"  Chunks not found: {stats['chunks_not_found']}")
    logger.info(f"  Documents not found: {stats['docs_not_found']}")
    
    if dry_run:
        logger.info("(DRY RUN - no changes were made)")


def main():
    parser = argparse.ArgumentParser(description="Migrate to hierarchical storage")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--stream", default="plm:extraction", help="Redis stream name")
    parser.add_argument("--dry-run", action="store_true", help="Dry run (no changes)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.db_path):
        logger.error(f"Database not found: {args.db_path}")
        sys.exit(1)
    
    run_migration(args.db_path, args.redis_url, args.stream, args.dry_run)


if __name__ == "__main__":
    main()
