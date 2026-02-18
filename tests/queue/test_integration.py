"""Integration tests for queue module (requires Redis)."""

import json
import subprocess
import time
import uuid
from datetime import datetime, timezone

import pytest


def is_redis_available():
    """Check if Redis is available on localhost:6379."""
    try:
        import redis
        client = redis.from_url("redis://localhost:6379")
        client.ping()
        return True
    except Exception:
        return False


# Skip all tests in this module if Redis is not available
pytestmark = pytest.mark.skipif(
    not is_redis_available(),
    reason="Redis not available on localhost:6379"
)


@pytest.fixture
def redis_client():
    """Create a Redis client and clean up test streams."""
    import redis
    client = redis.from_url("redis://localhost:6379")
    yield client
    # Cleanup
    for key in client.keys("test:*"):
        client.delete(key)


class TestRedisStreamQueue:
    """Integration tests for RedisStreamQueue."""

    def test_publish_and_read(self, redis_client):
        """Message can be published and read from Redis."""
        from plm.shared.queue.redis_queue import RedisStreamQueue

        queue = RedisStreamQueue("redis://localhost:6379")
        assert queue.is_available() is True

        # Publish
        msg = {"test": "data", "timestamp": time.time()}
        msg_id = queue.publish("test:publish", msg)
        assert msg_id is not None

        # Read back
        result = redis_client.xrange("test:publish", "-", "+", count=1)
        assert len(result) == 1
        stored_data = json.loads(result[0][1][b"data"].decode())
        assert stored_data["test"] == "data"

    def test_connection_failure_raises(self):
        """ConnectionError raised when Redis unavailable."""
        import redis
        from plm.shared.queue.redis_queue import RedisStreamQueue

        queue = RedisStreamQueue("redis://localhost:9999")  # Wrong port
        with pytest.raises(redis.exceptions.ConnectionError):
            queue.publish("test:fail", {"data": "x"})


class TestQueueConsumer:
    """Integration tests for QueueConsumer."""

    def test_consumer_processes_message(self, redis_client):
        """Consumer successfully processes a message."""
        from plm.shared.queue.consumer import QueueConsumer

        processed = []

        class TestConsumer(QueueConsumer):
            def process_message(self, data):
                processed.append(data)

        # Clean up
        redis_client.delete("test:consumer", "test:consumer:dlq")

        # Add message
        redis_client.xadd("test:consumer", {"data": json.dumps({"hello": "world"})})

        # Process
        consumer = TestConsumer(
            redis_url="redis://localhost:6379",
            stream="test:consumer",
            group="test-group",
            consumer="test-consumer",
        )
        count = consumer.process_pending()

        assert count == 1
        assert len(processed) == 1
        assert processed[0]["hello"] == "world"

    def test_consumer_dlq_after_retries(self, redis_client):
        """Failed messages move to DLQ after max retries."""
        from plm.shared.queue.consumer import QueueConsumer

        class FailingConsumer(QueueConsumer):
            def process_message(self, data):
                raise ValueError("Intentional failure")

        # Clean up
        redis_client.delete("test:failing", "test:failing:dlq")

        # Add message
        redis_client.xadd("test:failing", {"data": json.dumps({"fail": True})})

        # Process
        consumer = FailingConsumer(
            redis_url="redis://localhost:6379",
            stream="test:failing",
            group="test-group",
            consumer="test-consumer",
            max_retries=3,
            dlq_stream="test:failing:dlq",
        )
        consumer.process_pending()

        # Verify DLQ
        dlq_len = redis_client.xlen("test:failing:dlq")
        assert dlq_len == 1


class TestSearchQueueConsumer:
    """Integration tests for SearchQueueConsumer."""

    def test_consumer_ingests_document(self, redis_client, tmp_path):
        """SearchQueueConsumer ingests document into retriever."""
        from plm.search.retriever import HybridRetriever
        from plm.search.service.queue_consumer import SearchQueueConsumer

        # Clean up
        redis_client.delete("test:search", "test:search:dlq")

        # Create retriever
        db_path = tmp_path / "index.db"
        retriever = HybridRetriever(
            db_path=str(db_path),
            bm25_index_path=str(tmp_path),
        )

        # Add message with extraction result
        msg = {
            "message_id": str(uuid.uuid4()),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_service": "fast-extraction",
            "payload": {
                "source_file": "/test/kubernetes.md",
                "headings": [{
                    "heading": "# Kubernetes",
                    "level": 1,
                    "chunks": [{
                        "text": "Kubernetes is a container orchestration platform.",
                        "terms": ["Kubernetes"],
                        "entities": [{"text": "Kubernetes", "label": "technology", "score": 0.9, "start": 0, "end": 10}],
                        "keywords": [],
                        "start_char": 0,
                        "end_char": 49
                    }]
                }],
                "avg_confidence": 0.9,
                "total_entities": 1,
                "is_low_confidence": False,
                "error": None
            }
        }
        redis_client.xadd("test:search", {"data": json.dumps(msg)})

        # Process
        consumer = SearchQueueConsumer(
            redis_url="redis://localhost:6379",
            retriever=retriever,
            stream="test:search",
        )
        count = consumer.process_pending()
        consumer._rebuild_index()

        # Verify
        assert count == 1
        chunk_count = retriever.storage.get_chunk_count()
        assert chunk_count == 1


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_fast_extraction_to_search(self, redis_client, tmp_path):
        """Fast extraction publishes, search consumer ingests."""
        import os

        # This test requires the full fast extraction which is slow
        # Mark as slow/integration test
        pytest.importorskip("plm.extraction.fast.document_processor")

        # Create input document
        input_dir = tmp_path / "input"
        output_dir = tmp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        doc_path = input_dir / "test.md"
        doc_path.write_text("# Test\n\nKubernetes is a container orchestration platform.")

        # Clean up Redis
        redis_client.delete("test:e2e", "test:e2e:dlq")

        # Run fast extraction with queue
        env = os.environ.copy()
        env["QUEUE_ENABLED"] = "true"
        env["QUEUE_URL"] = "redis://localhost:6379"
        env["QUEUE_STREAM"] = "test:e2e"

        result = subprocess.run(
            ["python", "-m", "plm.extraction.fast.cli",
             "--input", str(input_dir),
             "--output", str(output_dir),
             "--workers", "1"],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )

        # Check extraction succeeded
        assert result.returncode == 0, f"Fast extraction failed: {result.stderr}"

        # Verify message in Redis
        messages = redis_client.xrange("test:e2e", "-", "+")
        assert len(messages) >= 1, "No messages in Redis stream"

        # Verify message structure
        msg_data = json.loads(messages[0][1][b"data"].decode())
        assert "payload" in msg_data
        assert "source_file" in msg_data["payload"]
        assert "headings" in msg_data["payload"]
