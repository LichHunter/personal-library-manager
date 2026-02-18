"""Queue consumer for search service document ingestion.

Consumes extraction messages from Redis Streams and ingests them into
the search index. Replaces DirectoryWatcher when QUEUE_ENABLED=true.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import time
from typing import TYPE_CHECKING

from plm.search.service.watcher import json_to_chunks
from plm.shared.queue import QueueConsumer

if TYPE_CHECKING:
    from plm.search.retriever import HybridRetriever

logger = logging.getLogger(__name__)

# Batch settings for index rebuilds
DEFAULT_BATCH_SIZE = 10
DEFAULT_BATCH_TIMEOUT_S = 5.0


class SearchQueueConsumer(QueueConsumer):
    """Queue consumer that ingests extraction results into search index.
    
    Processes MessageEnvelope messages containing ExtractionMessage payloads.
    Batches ingestions and rebuilds index periodically for efficiency.
    """

    def __init__(
        self,
        redis_url: str,
        retriever: HybridRetriever,
        stream: str = "plm:extraction",
        group: str = "search-service",
        consumer: str | None = None,
        max_retries: int = 3,
        dlq_stream: str | None = None,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_timeout_s: float = DEFAULT_BATCH_TIMEOUT_S,
    ) -> None:
        """Initialize search queue consumer.

        Args:
            redis_url: Redis connection URL.
            retriever: HybridRetriever instance for ingestion.
            stream: Stream to consume from.
            group: Consumer group name.
            consumer: Consumer name (auto-generated if None).
            max_retries: Max retries before moving to DLQ.
            dlq_stream: DLQ stream name.
            batch_size: Number of messages before forced index rebuild.
            batch_timeout_s: Seconds since last ingest before index rebuild.
        """
        super().__init__(
            redis_url=redis_url,
            stream=stream,
            group=group,
            consumer=consumer,
            max_retries=max_retries,
            dlq_stream=dlq_stream or f"{stream}:dlq",
        )
        self._retriever = retriever
        self._batch_size = batch_size
        self._batch_timeout_s = batch_timeout_s
        self._pending_count = 0
        self._last_ingest_time = 0.0
        self._batch_lock = threading.Lock()

    def process_message(self, data: dict) -> None:
        """Process a single extraction message.

        Args:
            data: MessageEnvelope containing ExtractionMessage payload.
            
        Raises:
            ValueError: If message is malformed.
            KeyError: If required fields are missing.
        """
        # Extract payload from envelope (or use data directly if no envelope)
        if "payload" in data:
            payload = data["payload"]
            source_service = data.get("source_service", "unknown")
            message_id = data.get("message_id", "unknown")
        else:
            # Direct ExtractionMessage without envelope
            payload = data
            source_service = "unknown"
            message_id = "unknown"

        # Validate required fields
        source_file = payload.get("source_file")
        if not source_file:
            raise ValueError("Missing 'source_file' in payload")
        
        headings = payload.get("headings")
        if headings is None:
            raise ValueError("Missing 'headings' in payload")

        # Convert to chunks using existing watcher logic
        chunks = json_to_chunks(payload)
        if not chunks:
            logger.warning(f"[QueueConsumer] No chunks extracted from {source_file}")
            return

        all_content = "".join(c["content"] for c in chunks)
        content_hash = hashlib.sha256(all_content.encode()).hexdigest()[:12]
        filename = source_file.split("/")[-1].rsplit(".", 1)[0]
        doc_id = f"{filename}_{content_hash}"

        deleted = self._retriever.storage.delete_documents_by_content_hash(content_hash)
        if deleted > 0:
            logger.info(f"[QueueConsumer] Deduplicated {deleted} existing document(s) with hash {content_hash}")

        self._retriever.ingest_document(
            doc_id=doc_id,
            source_file=source_file,
            chunks=chunks,
            rebuild_index=False,
        )
        
        logger.info(
            f"[QueueConsumer] Ingested {len(chunks)} chunks from {source_file} "
            f"(source: {source_service}, msg: {message_id})"
        )

        # Track for batched index rebuild
        with self._batch_lock:
            self._pending_count += 1
            self._last_ingest_time = time.time()
            
            if self._pending_count >= self._batch_size:
                self._rebuild_index()

    def _rebuild_index(self) -> None:
        """Rebuild index and reset pending counter."""
        if self._pending_count > 0:
            try:
                self._retriever.rebuild_index()
                logger.info(f"[QueueConsumer] Rebuilt index after {self._pending_count} documents")
            except Exception as e:
                logger.error(f"[QueueConsumer] Failed to rebuild index: {e}")
            finally:
                self._pending_count = 0

    def check_batch_timeout(self) -> None:
        """Check if batch timeout elapsed and rebuild if needed.
        
        Should be called periodically from the run loop.
        """
        with self._batch_lock:
            if self._pending_count > 0:
                elapsed = time.time() - self._last_ingest_time
                if elapsed >= self._batch_timeout_s:
                    self._rebuild_index()

    def run(self) -> None:
        """Run the consumer loop with periodic batch timeout checks."""
        self._setup_signal_handlers()
        self._ensure_group()
        
        logger.info(f"[QueueConsumer] Starting on stream '{self._stream}'")
        
        # First process any pending messages
        pending = self.process_pending()
        if pending:
            logger.info(f"[QueueConsumer] Processed {pending} pending messages")
            self._rebuild_index()
        
        # Then consume new messages continuously
        while not self._shutdown.is_set():
            try:
                messages = self._read_messages(">", block_ms=1000)  # 1s block for timeout checks
                for msg_id, fields in messages:
                    self._process_one(msg_id, fields)
                
                # Check batch timeout even if no messages
                self.check_batch_timeout()
                        
            except Exception as e:
                logger.error(f"[QueueConsumer] Error in run loop: {e}")
                if not self._shutdown.is_set():
                    self._shutdown.wait(1.0)
        
        # Final index rebuild on shutdown
        self._rebuild_index()
        logger.info("[QueueConsumer] Shutdown complete")
        self._restore_signal_handlers()
