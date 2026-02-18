"""NullQueue implementation for standalone mode.

When QUEUE_ENABLED=false (default), services use NullQueue which
silently discards messages. This allows all services to work without Redis.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class NullQueue:
    """No-op queue implementation for standalone mode.
    
    Satisfies MessageQueue protocol but discards all messages.
    Used when QUEUE_ENABLED=false or Redis is unavailable.
    """

    def publish(self, stream: str, message: dict) -> str | None:
        """Discard the message silently.

        Args:
            stream: Stream name (ignored).
            message: Message payload (ignored).

        Returns:
            None (no message ID since nothing is published).
        """
        logger.debug(f"[NullQueue] Discarding message to stream '{stream}'")
        return None

    def is_available(self) -> bool:
        """NullQueue is never 'available' in the connected sense.

        Returns:
            False always.
        """
        return False
