"""MessageQueue Protocol for queue integration.

This module defines the abstract MessageQueue interface that all queue
implementations must satisfy.
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class MessageQueue(Protocol):
    """Protocol for message queue implementations.
    
    Defines the interface that all queue backends (Redis, NullQueue, etc.)
    must implement.
    """

    def publish(self, stream: str, message: dict) -> str | None:
        """Publish a message to a stream.

        Args:
            stream: Stream/topic name (e.g., 'extraction-tasks').
            message: Message payload as a dictionary.

        Returns:
            Message ID if published successfully, None if queue is unavailable
            or disabled.
        """
        ...

    def is_available(self) -> bool:
        """Check if the queue is available and connected.

        Returns:
            True if queue is ready to accept messages, False otherwise.
        """
        ...
