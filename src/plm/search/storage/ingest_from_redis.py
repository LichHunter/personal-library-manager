#!/usr/bin/env python3
"""Ingest documents from Redis stream into the search index.

Reads extraction messages from Redis Streams and ingests them using
the HybridRetriever's hierarchical ingestion method.

Usage:
    python -m plm.search.storage.ingest_from_redis \
        --db-path /path/to/index.db \
        --bm25-path /path/to/bm25_index \
        --redis-url redis://localhost:6379 \
        --stream plm:extraction
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def connect_redis(redis_url: str):
    import redis
    return redis.from_url(redis_url)


def read_all_messages(redis_client, stream: str) -> list[dict]:
    """Read all messages from Redis stream."""
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


def compute_doc_id(source_file: str, content: str) -> str:
    """Compute doc_id matching watcher.py pattern: {stem}_{content_hash}."""
    stem = Path(source_file).stem
    content_hash = hashlib.sha256(content.encode()).hexdigest()[:16]
    return f"{stem}_{content_hash}"


def extract_full_content(headings: list[dict]) -> str:
    """Extract all chunk text to compute content hash."""
    texts = []
    for section in headings:
        for chunk in section.get("chunks", []):
            text = chunk.get("text", "")
            if text:
                texts.append(text)
    return "\n\n".join(texts)


def run_ingestion(db_path: str, bm25_path: str, redis_url: str, stream: str, limit: int | None = None):
    from plm.search.retriever import HybridRetriever
    
    logger.info(f"Connecting to Redis at {redis_url}")
    redis_client = connect_redis(redis_url)
    
    logger.info(f"Reading messages from stream {stream}")
    messages = read_all_messages(redis_client, stream)
    logger.info(f"Found {len(messages)} messages")
    
    if not messages:
        logger.warning("No messages found in Redis. Nothing to ingest.")
        return
    
    if limit:
        messages = messages[:limit]
        logger.info(f"Processing first {limit} messages")
    
    logger.info(f"Initializing retriever (db={db_path}, bm25={bm25_path})")
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    retriever = HybridRetriever(db_path=db_path, bm25_index_path=bm25_path)
    
    stats = {
        "ingested": 0,
        "skipped": 0,
        "errors": 0,
    }
    
    seen_doc_ids: set[str] = set()
    
    for i, msg in enumerate(messages):
        payload = msg.get("payload", msg)
        source_file = payload.get("source_file", "")
        headings = payload.get("headings", [])
        
        if not headings:
            logger.debug(f"Skipping message {i}: no headings")
            stats["skipped"] += 1
            continue
        
        # Compute doc_id from content
        full_content = extract_full_content(headings)
        if not full_content:
            logger.debug(f"Skipping message {i}: no content")
            stats["skipped"] += 1
            continue
        
        doc_id = compute_doc_id(source_file, full_content)
        
        # Skip duplicates (same doc_id)
        if doc_id in seen_doc_ids:
            logger.debug(f"Skipping duplicate: {doc_id}")
            stats["skipped"] += 1
            continue
        seen_doc_ids.add(doc_id)
        
        try:
            # Defer BM25 rebuild until end
            retriever.ingest_document_hierarchical(
                doc_id=doc_id,
                source_file=source_file,
                headings=headings,
                rebuild_index=False,
            )
            stats["ingested"] += 1
            
            if (i + 1) % 100 == 0:
                logger.info(f"Progress: {i + 1}/{len(messages)} messages processed")
                
        except Exception as e:
            logger.error(f"Error ingesting {source_file}: {e}")
            stats["errors"] += 1
    
    # Rebuild BM25 index once at the end
    logger.info("Building BM25 index...")
    retriever.rebuild_bm25_index()
    
    logger.info("Ingestion complete!")
    logger.info(f"  Documents ingested: {stats['ingested']}")
    logger.info(f"  Skipped: {stats['skipped']}")
    logger.info(f"  Errors: {stats['errors']}")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents from Redis")
    parser.add_argument("--db-path", required=True, help="Path to SQLite database")
    parser.add_argument("--bm25-path", required=True, help="Path to BM25 index directory")
    parser.add_argument("--redis-url", default="redis://localhost:6379", help="Redis URL")
    parser.add_argument("--stream", default="plm:extraction", help="Redis stream name")
    parser.add_argument("--limit", type=int, help="Limit number of messages to process")
    
    args = parser.parse_args()
    
    run_ingestion(args.db_path, args.bm25_path, args.redis_url, args.stream, args.limit)


if __name__ == "__main__":
    main()
