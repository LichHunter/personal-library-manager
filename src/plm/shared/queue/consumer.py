"""Queue consumer base class with DLQ support.

Provides a base class for consuming messages from Redis Streams with:
- Consumer group management (XGROUP CREATE)
- Message consumption (XREADGROUP)  
- Acknowledgment (XACK)
- Immediate retry with configurable max_retries
- Dead-letter queue after N failures
- Graceful shutdown
"""

from __future__ import annotations

import json
import logging
import os
import signal
import socket
import threading
from abc import ABC, abstractmethod
from typing import Any

import redis

logger = logging.getLogger(__name__)

DEFAULT_MAX_RETRIES = 3
DEFAULT_BLOCK_MS = 5000  # Block for 5 seconds waiting for messages
DEFAULT_BATCH_SIZE = 10


class QueueConsumer(ABC):
    """Base class for Redis Streams consumers with DLQ support.
    
    Subclasses must implement process_message() to handle each message.
    Failed messages are retried immediately up to max_retries times,
    then moved to DLQ.
    """

    def __init__(
        self,
        redis_url: str,
        stream: str,
        group: str,
        consumer: str | None = None,
        max_retries: int = DEFAULT_MAX_RETRIES,
        dlq_stream: str | None = None,
        block_ms: int = DEFAULT_BLOCK_MS,
        batch_size: int = DEFAULT_BATCH_SIZE,
    ) -> None:
        """Initialize consumer.

        Args:
            redis_url: Redis connection URL.
            stream: Stream to consume from.
            group: Consumer group name.
            consumer: Consumer name (default: hostname-pid).
            max_retries: Max retries before moving to DLQ.
            dlq_stream: DLQ stream name (default: {stream}:dlq).
            block_ms: Block time in milliseconds for XREADGROUP.
            batch_size: Number of messages to read per batch.
        """
        self._url = redis_url
        self._stream = stream
        self._group = group
        self._consumer = consumer or f"{socket.gethostname()}-{os.getpid()}"
        self._max_retries = max_retries
        self._dlq_stream = dlq_stream or f"{stream}:dlq"
        self._block_ms = block_ms
        self._batch_size = batch_size
        
        self._client: redis.Redis | None = None
        self._shutdown = threading.Event()
        self._original_sigterm = None
        self._original_sigint = None

    def _get_client(self) -> redis.Redis:
        """Get or create Redis client."""
        if self._client is None:
            self._client = redis.from_url(self._url, decode_responses=True)
        return self._client

    def _ensure_group(self) -> None:
        """Create consumer group if it doesn't exist."""
        client = self._get_client()
        try:
            # MKSTREAM creates the stream if it doesn't exist
            client.xgroup_create(self._stream, self._group, id="0", mkstream=True)
            logger.info(f"[Consumer] Created group '{self._group}' on '{self._stream}'")
        except redis.exceptions.ResponseError as e:
            if "BUSYGROUP" in str(e):
                # Group already exists, that's fine
                logger.debug(f"[Consumer] Group '{self._group}' already exists")
            else:
                raise

    @abstractmethod
    def process_message(self, data: dict) -> None:
        """Process a single message. Override in subclass.

        Args:
            data: Parsed message payload.
            
        Raises:
            Exception: Any exception triggers retry/DLQ logic.
        """
        ...

    def _move_to_dlq(self, msg_id: str, data: dict, error: Exception) -> None:
        """Move a failed message to dead-letter queue."""
        client = self._get_client()
        dlq_entry = {
            "original_stream": self._stream,
            "original_id": msg_id,
            "data": json.dumps(data),
            "error_type": type(error).__name__,
            "error_message": str(error),
            "consumer": self._consumer,
        }
        client.xadd(self._dlq_stream, dlq_entry)
        logger.warning(f"[Consumer] Moved {msg_id} to DLQ: {error}")

    def _process_one(self, msg_id: str, fields: dict[str, str]) -> bool:
        """Process a single message with immediate retry.

        Retries up to max_retries times before moving to DLQ.

        Returns:
            True if processed successfully, False if moved to DLQ.
        """
        client = self._get_client()
        
        # Parse message data
        try:
            raw_data = fields.get("data", "{}")
            data = json.loads(raw_data)
        except json.JSONDecodeError as e:
            logger.error(f"[Consumer] Invalid JSON in {msg_id}: {e}")
            self._move_to_dlq(msg_id, {"raw": fields}, e)
            client.xack(self._stream, self._group, msg_id)
            return False
        
        # Try to process with immediate retries
        last_error: Exception | None = None
        for attempt in range(1, self._max_retries + 1):
            try:
                self.process_message(data)
                client.xack(self._stream, self._group, msg_id)
                logger.debug(f"[Consumer] Processed {msg_id}")
                return True
            except Exception as e:
                last_error = e
                if attempt < self._max_retries:
                    logger.warning(f"[Consumer] Attempt {attempt}/{self._max_retries} failed for {msg_id}: {e}")
                else:
                    logger.error(f"[Consumer] All {self._max_retries} attempts failed for {msg_id}: {e}")
        
        # All retries exhausted, move to DLQ
        if last_error:
            self._move_to_dlq(msg_id, data, last_error)
        client.xack(self._stream, self._group, msg_id)
        return False

    def _read_messages(self, start_id: str, block_ms: int = 100) -> list[tuple[str, dict[str, str]]]:
        """Read messages from stream.
        
        Args:
            start_id: ">" for new messages, "0" for pending messages.
            block_ms: Block time in milliseconds.
            
        Returns:
            List of (msg_id, fields) tuples.
        """
        client = self._get_client()
        result: Any = client.xreadgroup(
            self._group,
            self._consumer,
            {self._stream: start_id},
            count=self._batch_size,
            block=block_ms,
        )
        
        if not result:
            return []
        
        messages = []
        for stream_name, stream_messages in result:
            for msg_id, fields in stream_messages:
                messages.append((msg_id, fields))
        return messages

    def process_pending(self) -> int:
        """Process pending and new messages (for recovery/testing).
        
        First processes any messages previously delivered but not ACKed,
        then processes new messages until none remain.
        
        Returns:
            Number of messages processed.
        """
        self._ensure_group()
        processed = 0
        
        # First, process pending messages (previously delivered, not ACKed)
        while not self._shutdown.is_set():
            messages = self._read_messages("0", block_ms=100)
            if not messages:
                break
            for msg_id, fields in messages:
                self._process_one(msg_id, fields)
                processed += 1
        
        # Then, process new messages (not yet delivered)
        while not self._shutdown.is_set():
            messages = self._read_messages(">", block_ms=100)
            if not messages:
                break
            for msg_id, fields in messages:
                self._process_one(msg_id, fields)
                processed += 1
        
        return processed

    def run(self) -> None:
        """Run the consumer loop until shutdown."""
        self._setup_signal_handlers()
        self._ensure_group()
        
        logger.info(f"[Consumer] Starting consumer '{self._consumer}' on '{self._stream}'")
        
        # First process any pending messages
        pending = self.process_pending()
        if pending:
            logger.info(f"[Consumer] Processed {pending} pending messages")
        
        # Then consume new messages continuously
        while not self._shutdown.is_set():
            try:
                messages = self._read_messages(">", block_ms=self._block_ms)
                for msg_id, fields in messages:
                    self._process_one(msg_id, fields)
                        
            except redis.exceptions.ConnectionError as e:
                logger.error(f"[Consumer] Redis connection error: {e}")
                if not self._shutdown.is_set():
                    self._shutdown.wait(1.0)
        
        logger.info("[Consumer] Shutdown complete")
        self._restore_signal_handlers()

    def stop(self) -> None:
        """Signal the consumer to stop gracefully."""
        logger.info("[Consumer] Shutdown requested")
        self._shutdown.set()

    def _setup_signal_handlers(self) -> None:
        """Set up graceful shutdown on SIGTERM/SIGINT.
        
        Only works in main thread - silently skips in worker threads.
        """
        import threading
        if threading.current_thread() is not threading.main_thread():
            logger.debug("[Consumer] Skipping signal handlers (not main thread)")
            return
            
        def handler(signum, frame):
            logger.info(f"[Consumer] Received signal {signum}")
            self.stop()
        
        self._original_sigterm = signal.signal(signal.SIGTERM, handler)
        self._original_sigint = signal.signal(signal.SIGINT, handler)

    def _restore_signal_handlers(self) -> None:
        """Restore original signal handlers."""
        import threading
        if threading.current_thread() is not threading.main_thread():
            return
        if self._original_sigterm is not None:
            signal.signal(signal.SIGTERM, self._original_sigterm)
        if self._original_sigint is not None:
            signal.signal(signal.SIGINT, self._original_sigint)
