"""Queue integration module for PLM services.

Provides optional Redis Streams-based message queue communication between
extraction and search services. All services retain standalone capability
when QUEUE_ENABLED=false (default).

Usage:
    from plm.shared.queue import create_queue, MessageQueue
    
    queue = create_queue()  # Returns NullQueue or RedisStreamQueue based on env
    if queue.is_available():
        queue.publish("plm:extraction", {"source_file": "doc.md", ...})
"""

from plm.shared.queue.consumer import QueueConsumer
from plm.shared.queue.factory import create_queue
from plm.shared.queue.null_queue import NullQueue
from plm.shared.queue.protocol import MessageQueue
from plm.shared.queue.redis_queue import RedisStreamQueue
from plm.shared.queue.types import (
    Chunk,
    Entity,
    ExtractionMessage,
    HeadingSection,
    MessageEnvelope,
)

__all__ = [
    # Factory
    "create_queue",
    # Protocol & implementations
    "MessageQueue",
    "NullQueue",
    "RedisStreamQueue",
    # Consumer
    "QueueConsumer",
    # Types
    "Entity",
    "Chunk",
    "HeadingSection",
    "ExtractionMessage",
    "MessageEnvelope",
]
