"""Queue factory for creating queue instances from environment variables.

Environment Variables:
    QUEUE_ENABLED: "true" to enable Redis queue, "false" (default) for NullQueue
    QUEUE_URL: Redis URL (default: "redis://localhost:6379")
"""

from __future__ import annotations

import logging
import os

from plm.shared.queue.null_queue import NullQueue
from plm.shared.queue.protocol import MessageQueue

logger = logging.getLogger(__name__)


def create_queue() -> MessageQueue:
    """Create a queue instance based on environment configuration.

    Reads QUEUE_ENABLED and QUEUE_URL from environment.
    
    Returns:
        MessageQueue: NullQueue if disabled, RedisStreamQueue if enabled.
        
    Raises:
        ImportError: If QUEUE_ENABLED=true but redis package not installed.
    """
    enabled = os.environ.get("QUEUE_ENABLED", "false").lower() == "true"
    
    if not enabled:
        logger.debug("[Queue] QUEUE_ENABLED=false, using NullQueue")
        return NullQueue()
    
    # Import Redis queue only when needed (lazy import)
    url = os.environ.get("QUEUE_URL", "redis://localhost:6379")
    logger.info(f"[Queue] QUEUE_ENABLED=true, connecting to {url}")
    
    try:
        from plm.shared.queue.redis_queue import RedisStreamQueue
        return RedisStreamQueue(url)
    except ImportError as e:
        logger.error(f"[Queue] Redis queue requested but redis package not installed: {e}")
        raise
