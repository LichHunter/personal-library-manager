"""RedisStreamQueue implementation using Redis Streams.

Publishes messages to Redis Streams using XADD command.
Fails loudly on connection errors (no silent fallback).
"""

from __future__ import annotations

import json
import logging
import warnings
from typing import Any

import redis

logger = logging.getLogger(__name__)

# Default max stream length (bounded stream to prevent unbounded growth)
DEFAULT_MAXLEN = 10000


class RedisStreamQueue:
    """Redis Streams-based message queue implementation.
    
    Uses XADD for publishing with bounded stream length.
    Raises ConnectionError on Redis unavailability (fail loudly).
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        maxlen: int = DEFAULT_MAXLEN,
    ) -> None:
        """Initialize Redis connection.

        Args:
            url: Redis connection URL.
            maxlen: Maximum stream length (approximate, uses ~).
        """
        self._url = url
        self._maxlen = maxlen
        self._client: redis.Redis | None = None
        self._connected: bool | None = None  # None = not yet probed

    def _get_client(self) -> redis.Redis:
        """Get or create Redis client with lazy connection probe.
        
        Returns:
            Connected Redis client.
            
        Raises:
            redis.exceptions.ConnectionError: If Redis is unavailable.
        """
        if self._client is None:
            self._client = redis.from_url(self._url, decode_responses=True)
        
        # Probe connection on first use
        if self._connected is None:
            try:
                self._client.ping()
                self._connected = True
                logger.info(f"[RedisQueue] Connected to {self._url}")
            except redis.exceptions.ConnectionError:
                self._connected = False
                raise
        
        return self._client

    def publish(self, stream: str, message: dict) -> str | None:
        """Publish a message to a Redis stream.

        Args:
            stream: Stream name (e.g., 'plm:extraction').
            message: Message payload as a dictionary.

        Returns:
            Message ID assigned by Redis (e.g., '1234567890-0').
            
        Raises:
            redis.exceptions.ConnectionError: If Redis is unavailable.
        """
        client = self._get_client()
        
        # Serialize message to JSON for storage
        # Redis Streams require flat key-value pairs, so we store as single 'data' field
        serialized = json.dumps(message)
        
        # Warn on large messages (>1MB)
        size = len(serialized)
        if size > 1_000_000:
            warnings.warn(
                f"Large message ({size} bytes) being published to {stream}. "
                "Consider chunking or compressing.",
                stacklevel=2,
            )
        
        # XADD with approximate maxlen for bounded stream
        msg_id: Any = client.xadd(
            stream,
            {"data": serialized},
            maxlen=self._maxlen,
            approximate=True,
        )
        
        logger.debug(f"[RedisQueue] Published to {stream}: {msg_id}")
        return str(msg_id)

    def is_available(self) -> bool:
        """Check if Redis is available and connected.

        Returns:
            True if Redis is reachable, False otherwise.
        """
        try:
            client = self._get_client()
            client.ping()
            return True
        except redis.exceptions.ConnectionError:
            return False
