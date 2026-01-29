"""Redis cache client with graceful fallback for modular retrieval pipeline."""

import hashlib
import json
import logging
from typing import Optional

try:
    import redis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class RedisCacheClient:
    """Redis cache client with graceful fallback when Redis is unavailable."""

    def __init__(self, host: str = "localhost", port: int = 6379):
        """
        Initialize Redis cache client with graceful fallback.

        Args:
            host: Redis server hostname (default: localhost)
            port: Redis server port (default: 6379)
        """
        self._client = None
        self._connected = False

        if not REDIS_AVAILABLE:
            logger.warning(
                "redis package not available, cache operations will be disabled"
            )
            return

        try:
            self._client = redis.Redis(
                host=host,
                port=port,
                decode_responses=False,
                socket_connect_timeout=2,
                socket_keepalive=True,
            )
            # Test connection
            self._client.ping()
            self._connected = True
            logger.info(f"Connected to Redis at {host}:{port}")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis at {host}:{port}: {e}")
            self._client = None
            self._connected = False

    def is_connected(self) -> bool:
        """
        Check if Redis is available and connected.

        Returns:
            True if connected, False otherwise
        """
        return self._connected

    def get(self, key: str) -> Optional[bytes]:
        """
        Get value from cache.

        Args:
            key: Cache key

        Returns:
            Cached value as bytes, or None if not found or Redis unavailable
        """
        if not self._connected or self._client is None:
            return None

        try:
            return self._client.get(key)
        except Exception as e:
            logger.warning(f"Error getting key '{key}' from Redis: {e}")
            return None

    def set(self, key: str, value: bytes, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache.

        Args:
            key: Cache key
            value: Value to cache (bytes)
            ttl: Time to live in seconds (optional, None = indefinite)

        Returns:
            True if successful, False if Redis unavailable
        """
        if not self._connected or self._client is None:
            return False

        try:
            if ttl is not None:
                self._client.setex(key, ttl, value)
            else:
                self._client.set(key, value)
            return True
        except Exception as e:
            logger.warning(f"Error setting key '{key}' in Redis: {e}")
            return False

    def delete(self, key: str) -> bool:
        """
        Delete key from cache.

        Args:
            key: Cache key

        Returns:
            True if successful, False if Redis unavailable
        """
        if not self._connected or self._client is None:
            return False

        try:
            self._client.delete(key)
            return True
        except Exception as e:
            logger.warning(f"Error deleting key '{key}' from Redis: {e}")
            return False

    def exists(self, key: str) -> bool:
        """
        Check if key exists in cache.

        Args:
            key: Cache key

        Returns:
            True if key exists, False if not found or Redis unavailable
        """
        if not self._connected or self._client is None:
            return False

        try:
            return bool(self._client.exists(key))
        except Exception as e:
            logger.warning(f"Error checking existence of key '{key}' in Redis: {e}")
            return False

    def dbsize(self) -> int:
        """
        Get number of keys in Redis database.

        Returns:
            Number of keys, or 0 if Redis unavailable
        """
        if not self._connected or self._client is None:
            return 0

        try:
            return self._client.dbsize()
        except Exception as e:
            logger.warning(f"Error getting dbsize from Redis: {e}")
            return 0

    @staticmethod
    def make_key(component: str, config: dict, content: str) -> str:
        """
        Generate cache key from component, config, and content.

        Key format: v1:{component}:{config_hash}:{content_hash}

        Args:
            component: Component name (e.g., "keywords", "entities")
            config: Configuration dictionary
            content: Content string to cache

        Returns:
            Cache key string
        """
        # Hash config: SHA-256 of sorted JSON
        config_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
        config_hash = hashlib.sha256(config_json.encode()).hexdigest()

        # Hash content: SHA-256 of content string
        content_hash = hashlib.sha256(content.encode()).hexdigest()

        return f"v1:{component}:{config_hash}:{content_hash}"
