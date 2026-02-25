# Queue Integration for PLM Services

## TL;DR

> **Quick Summary**: Connect fast extraction, slow extraction, and search services via Redis Streams message queue. All services retain standalone capability via `QUEUE_ENABLED=false` (default). When enabled, extraction services publish to queue instead of writing files; search service consumes from queue instead of watching directories.
> 
> **Deliverables**:
> - `src/plm/shared/queue/` module (Protocol, RedisStreamQueue, NullQueue, factory)
> - Modified extraction CLIs with optional queue publishing
> - Search service queue consumer component
> - Combined `docker/docker-compose.full.yml` with Redis + all services
> - Integration tests for queue communication
> 
> **Estimated Effort**: Large (8-12 tasks, cross-cutting changes)
> **Parallel Execution**: YES - 5 waves
> **Critical Path**: Task 1 → Task 3 → Task 5 → Task 9 → Task 10

---

## Context

### Original Request
User wants to connect fast extraction, slow extraction, and search services via message queue. Currently services communicate via filesystem (files dropped into watched directories). Queue should be optional - all services must work standalone without Redis.

### Interview Summary
**Key Discussions**:
- **Queue technology**: Redis Streams (already proven in POC, lightweight, Kafka-like semantics)
- **Publishers**: Both fast AND slow extraction publish to queue
- **Failure handling**: Dead-letter queue after N retries (robust production pattern)
- **Docker scope**: Full pipeline (Redis + Fast + Slow + Search)
- **Format normalization**: Slow extraction transforms output to match fast extraction format
- **Output mode**: Queue replaces files when `QUEUE_ENABLED=true`
- **Redis failure**: Fail loudly (extraction stops with error)
- **DirectoryWatcher**: Disable when queue enabled (queue consumer is only ingestion path)

**Research Findings**:
- Fast extraction: `_serialize_result()` at `cli.py:19-28` produces JSON, output at lines 208-211
- Slow extraction: `write_output()` at `cli.py:169`, format differs from fast (needs transformation)
- Search service: `DirectoryWatcher` + `json_to_chunks()` at `watcher.py:36-74`
- Existing Redis client pattern: `poc/modular_retrieval_pipeline/cache/redis_client.py` (graceful fallback)
- Adapter pattern in codebase: `src/plm/shared/llm/base.py` (Protocol pattern)

### Metis Review
**Identified Gaps** (addressed):
- Format mismatch between slow/fast extraction → Normalize at producer (slow transforms to fast format)
- Redis failure mode undefined → Fail loudly (stop with error)
- DirectoryWatcher coexistence → Disable when queue enabled
- Consumer group configuration → Use service name + hostname-pid defaults
- Message size limits → Warn at 1MB, don't block
- DLQ processing strategy → Manual via redis-cli (out of scope for MVP)

---

## Work Objectives

### Core Objective
Add optional Redis Streams-based message queue communication between extraction and search services, maintaining full standalone capability when queue is disabled.

### Concrete Deliverables
- `src/plm/shared/queue/protocol.py` - MessageQueue Protocol definition
- `src/plm/shared/queue/types.py` - Canonical message schema (TypedDict)
- `src/plm/shared/queue/null_queue.py` - No-op queue for standalone mode
- `src/plm/shared/queue/redis_queue.py` - Redis Streams implementation
- `src/plm/shared/queue/consumer.py` - Queue consumer base with DLQ support
- `src/plm/shared/queue/factory.py` - Queue factory from env vars
- `src/plm/shared/queue/__init__.py` - Public API
- Modified `src/plm/extraction/fast/cli.py` - Optional queue publishing
- Modified `src/plm/extraction/slow/cli.py` - Format transform + optional queue publishing
- `src/plm/search/service/queue_consumer.py` - Queue consumer for search service
- Modified `src/plm/search/service/app.py` - Conditional queue consumer startup
- `docker/docker-compose.full.yml` - Full pipeline with Redis
- `tests/queue/` - Integration tests

### Definition of Done
- [ ] `QUEUE_ENABLED=false`: All services work exactly as before (file I/O)
- [ ] `QUEUE_ENABLED=true`: Extraction publishes to Redis, search consumes from Redis
- [ ] Failed messages move to DLQ after 3 retries
- [ ] Docker Compose starts full pipeline successfully
- [ ] All existing tests pass
- [ ] New queue integration tests pass

### Must Have
- Protocol-based abstraction (NullQueue for standalone, RedisStreamQueue for connected)
- Environment variable configuration (`QUEUE_ENABLED`, `QUEUE_URL`, etc.)
- Dead-letter queue for failed messages
- Graceful shutdown (drain pending messages)
- Slow-to-fast format transformation at producer

### Must NOT Have (Guardrails)
- **NO changes to extraction logic** (entity detection, confidence scoring)
- **NO changes to search indexing logic** (BM25 build, embedding)
- **NO metrics/monitoring/observability** (separate follow-up)
- **NO message compression or encryption**
- **NO multi-queue routing or priority queues**
- **NO changes to `json_to_chunks()`** - slow extraction must transform to match its expected input
- **NO admin UI for queue inspection** (use redis-cli)
- **NO circuit breaker pattern** (nice-to-have, not MVP)
- **NO message deduplication** (at-least-once is acceptable)

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.

### Test Decision
- **Infrastructure exists**: YES (pytest in project)
- **Automated tests**: Tests after implementation
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

> The executing agent will directly verify each deliverable by running it.

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| Queue module | Bash (pytest) | Unit tests for protocol, factory |
| CLI changes | Bash (python -m) | Run CLI with queue enabled, check Redis |
| Docker Compose | Bash (docker compose) | Start services, verify connectivity |
| Integration | Bash (pytest + docker) | End-to-end message flow |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Define canonical message schema (types.py)
└── Task 2: Create queue protocol + NullQueue

Wave 2 (After Wave 1):
├── Task 3: Implement RedisStreamQueue
└── Task 4: Implement queue consumer base with DLQ

Wave 3 (After Wave 2):
├── Task 5: Fast extraction queue integration
└── Task 6: Search service queue consumer

Wave 4 (After Wave 3):
├── Task 7: Slow extraction format transformer
└── Task 8: Slow extraction queue integration

Wave 5 (After Wave 4):
└── Task 9: Docker Compose full pipeline

Wave 6 (After Wave 5):
└── Task 10: Integration tests
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 5, 6, 7 | 2 |
| 2 | None | 3, 4, 5, 6 | 1 |
| 3 | 1, 2 | 5, 6 | 4 |
| 4 | 2 | 6 | 3 |
| 5 | 3 | 9, 10 | 6 |
| 6 | 3, 4 | 9, 10 | 5 |
| 7 | 1 | 8 | 3, 4, 5, 6 |
| 8 | 3, 7 | 9, 10 | None |
| 9 | 5, 6, 8 | 10 | None |
| 10 | 9 | None | None |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Approach |
|------|-------|---------------------|
| 1 | 1, 2 | Parallel: foundational types |
| 2 | 3, 4 | Parallel: Redis implementations |
| 3 | 5, 6 | Parallel: producer + consumer |
| 4 | 7, 8 | Sequential: transform then integrate |
| 5 | 9 | Single: Docker orchestration |
| 6 | 10 | Single: integration tests |

---

## TODOs

- [ ] 1. Define canonical message schema

  **What to do**:
  - Create `src/plm/shared/queue/types.py`
  - Define `ExtractionMessage` TypedDict matching fast extraction output format
  - Include envelope fields: `message_id`, `timestamp`, `source_service`
  - Define `MessageEnvelope` wrapper type
  - Add JSON schema validation function (optional, for debugging)

  **Must NOT do**:
  - Change existing fast extraction output structure
  - Add complex nested types beyond what fast extraction produces

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, clear schema from existing code
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 5, 6, 7
  - **Blocked By**: None

  **References**:
  - `src/plm/extraction/fast/cli.py:19-28` - `_serialize_result()` shows exact output structure
  - `src/plm/extraction/fast/document_processor.py:91-97` - `DocumentResult` dataclass
  - `src/plm/search/service/watcher.py:36-74` - `json_to_chunks()` shows expected input format

  **Acceptance Criteria**:
  - [ ] `src/plm/shared/queue/types.py` exists
  - [ ] `ExtractionMessage` TypedDict defined with all fields from fast extraction output
  - [ ] `MessageEnvelope` includes `message_id: str`, `timestamp: str`, `source_service: str`, `payload: ExtractionMessage`
  - [ ] Type imports work: `python -c "from plm.shared.queue.types import ExtractionMessage, MessageEnvelope"`

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Types are importable and valid
    Tool: Bash
    Steps:
      1. python -c "from plm.shared.queue.types import ExtractionMessage, MessageEnvelope; print('OK')"
    Expected Result: Prints "OK" with exit code 0
    Evidence: stdout captured

  Scenario: Type structure matches fast extraction output
    Tool: Bash
    Steps:
      1. python -c "
         from plm.shared.queue.types import ExtractionMessage
         import typing
         hints = typing.get_type_hints(ExtractionMessage)
         assert 'source_file' in hints
         assert 'headings' in hints
         print('Schema valid')
         "
    Expected Result: Prints "Schema valid"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(queue): add canonical message schema types`
  - Files: `src/plm/shared/queue/types.py`, `src/plm/shared/queue/__init__.py`

---

- [ ] 2. Create queue protocol, NullQueue, and factory

  **What to do**:
  - Create `src/plm/shared/queue/protocol.py` with `MessageQueue` Protocol
  - Create `src/plm/shared/queue/null_queue.py` with `NullQueue` implementation
  - Create `src/plm/shared/queue/factory.py` with `create_queue()` factory function
  - Protocol methods: `publish(stream, message) -> str | None`, `is_available() -> bool`
  - NullQueue: returns immediately, logs at DEBUG level
  - Factory: reads `QUEUE_ENABLED`, `QUEUE_URL` env vars, returns NullQueue or RedisStreamQueue
  - Update `src/plm/shared/queue/__init__.py` (created in Task 1) to export protocol, NullQueue, factory

  **Must NOT do**:
  - Implement Redis logic (that's Task 3)
  - Add complex configuration (that's Task 4)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Two small files, clear pattern from existing codebase
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Tasks 3, 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `src/plm/shared/llm/base.py` - Existing Protocol pattern in codebase
  - `poc/modular_retrieval_pipeline/cache/redis_client.py:55-62` - `is_connected()` pattern

  **Acceptance Criteria**:
  - [ ] `src/plm/shared/queue/protocol.py` exists with `MessageQueue` Protocol
  - [ ] `src/plm/shared/queue/null_queue.py` exists with `NullQueue` class
  - [ ] `src/plm/shared/queue/factory.py` exists with `create_queue()` function
  - [ ] `NullQueue` satisfies `MessageQueue` protocol (runtime checkable)
  - [ ] `__init__.py` updated to export protocol, NullQueue, create_queue
  - [ ] Import works: `python -c "from plm.shared.queue import MessageQueue, NullQueue, create_queue"`

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: NullQueue satisfies Protocol
    Tool: Bash
    Steps:
      1. python -c "
         from plm.shared.queue import MessageQueue, NullQueue
         q = NullQueue()
         assert isinstance(q, MessageQueue)
         assert q.is_available() == False
         assert q.publish('test', {'key': 'value'}) is None
         print('Protocol satisfied')
         "
    Expected Result: Prints "Protocol satisfied"
    Evidence: stdout captured

  Scenario: Factory returns NullQueue when QUEUE_ENABLED=false
    Tool: Bash
    Steps:
      1. QUEUE_ENABLED=false python -c "
         from plm.shared.queue import create_queue, NullQueue
         q = create_queue()
         assert isinstance(q, NullQueue)
         print('Factory returns NullQueue')
         "
    Expected Result: Prints "Factory returns NullQueue"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(queue): add MessageQueue protocol, NullQueue, and factory`
  - Files: `src/plm/shared/queue/protocol.py`, `src/plm/shared/queue/null_queue.py`, `src/plm/shared/queue/factory.py`
  - Note: `__init__.py` was created in Task 1, this task updates it to add new exports

---

- [ ] 3. Implement RedisStreamQueue

  **What to do**:
  - **First**: Add `redis>=5.0.0` to `pyproject.toml` dependencies and run `uv sync`
  - Create `src/plm/shared/queue/redis_queue.py`
  - Implement `RedisStreamQueue` class satisfying `MessageQueue` Protocol
  - Use `redis-py` library (added to dependencies in this task)
  - Implement `XADD` for publishing with `maxlen` parameter
  - Lazy connection probe on first use (like POC pattern)
  - Raise exception on connection failure when publishing (fail loudly)

  **Must NOT do**:
  - Implement consumer logic (that's Task 4)
  - Add retry logic (fail loudly per requirements)
  - Add connection pooling (single connection is fine for MVP)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Single file, moderate complexity, clear interface
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 4)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/cache/redis_client.py:38-53` - Connection pattern with lazy probe
  - Librarian research: `XADD` with `maxlen` for bounded streams
  - `src/plm/shared/queue/protocol.py` - Protocol to implement (from Task 2)

  **Acceptance Criteria**:
  - [ ] `src/plm/shared/queue/redis_queue.py` exists
  - [ ] `RedisStreamQueue` implements `MessageQueue` Protocol
  - [ ] `publish()` uses `XADD` with `maxlen` parameter
  - [ ] Connection failure raises `ConnectionError`
  - [ ] `is_available()` returns True when Redis reachable

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: RedisStreamQueue publishes to Redis
    Tool: Bash
    Preconditions: No test-redis container running
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine
      2. sleep 2
      3. python -c "
         from plm.shared.queue.redis_queue import RedisStreamQueue
         q = RedisStreamQueue('redis://localhost:6379')
         assert q.is_available() == True
         msg_id = q.publish('test:stream', {'data': 'test'})
         assert msg_id is not None
         print(f'Published: {msg_id}')
         " && docker exec test-redis redis-cli XLEN test:stream
      4. docker stop test-redis && docker rm test-redis
    Expected Result: XLEN returns 1
    Failure Cleanup: If any step fails, run `docker stop test-redis; docker rm test-redis` to clean up
    Evidence: stdout captured

  Scenario: RedisStreamQueue fails loudly when Redis unavailable
    Tool: Bash
    Preconditions: No Redis running on port 9999
    Steps:
      1. python -c "
         from plm.shared.queue.redis_queue import RedisStreamQueue
         import redis
         q = RedisStreamQueue('redis://localhost:9999')
         try:
             q.publish('test', {'data': 'x'})
             print('ERROR: Should have raised')
         except redis.exceptions.ConnectionError:
             print('Correctly raised ConnectionError')
         "
    Expected Result: Prints "Correctly raised ConnectionError"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(queue): add RedisStreamQueue implementation`
  - Files: `src/plm/shared/queue/redis_queue.py`

---

- [ ] 4. Implement queue consumer base with DLQ support

  **What to do**:
  - Create `src/plm/shared/queue/consumer.py`
  - Implement `QueueConsumer` base class with:
    - Consumer group management (`XGROUP CREATE`)
    - Message consumption (`XREADGROUP`)
    - Acknowledgment (`XACK`)
    - Retry tracking (in-memory counter per message ID)
    - Dead-letter queue (`XADD` to DLQ stream after N failures)
  - Graceful shutdown handler (drain pending, then exit)
  - Abstract `process_message()` method for subclasses

  **Must NOT do**:
  - Implement search-specific logic (that's Task 6)
  - Add persistent retry tracking (in-memory is fine for MVP)
  - Add metrics or alerting

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex state machine, error handling, graceful shutdown
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Task 3)
  - **Blocks**: Task 6
  - **Blocked By**: Task 2

  **References**:
  - `src/plm/search/service/watcher.py:161-174` - `drain()` pattern for graceful shutdown
  - `src/plm/search/service/watcher.py:274-288` - Signal handler setup
  - Librarian research: `XREADGROUP`, `XACK`, `XGROUP CREATE` patterns

  **Acceptance Criteria**:
  - [ ] `src/plm/shared/queue/consumer.py` exists
  - [ ] `QueueConsumer` has abstract `process_message(data: dict)` method
  - [ ] Consumer creates group with `XGROUP CREATE ... MKSTREAM`
  - [ ] Messages move to DLQ after `max_retries` failures (default 3)
  - [ ] `BUSYGROUP` error handled gracefully (group already exists)
  - [ ] Graceful shutdown on SIGTERM

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Consumer moves failed messages to DLQ after 3 retries
    Tool: Bash
    Preconditions: Redis running
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine
      2. sleep 2
      3. python -c "
         from plm.shared.queue.consumer import QueueConsumer
         import redis

         class FailingConsumer(QueueConsumer):
             def process_message(self, data):
                 raise ValueError('Intentional failure')

         client = redis.from_url('redis://localhost:6379')
         # Add a test message
         client.xadd('test:stream', {'data': 'fail-me'})
         
         c = FailingConsumer(
             redis_url='redis://localhost:6379',
             stream='test:stream',
             group='test-group',
             consumer='test-consumer',
             max_retries=3,
             dlq_stream='test:dlq'
         )
         # Process once (will retry 3 times then DLQ)
         c.process_pending()
         
         dlq_len = client.xlen('test:dlq')
         print(f'DLQ length: {dlq_len}')
         assert dlq_len == 1
         "
      4. docker stop test-redis && docker rm test-redis
    Expected Result: DLQ length is 1
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(queue): add QueueConsumer base with DLQ support`
  - Files: `src/plm/shared/queue/consumer.py`

---

- [ ] 5. Fast extraction queue integration

  **What to do**:
  - Modify `src/plm/extraction/fast/cli.py`
  - Add env var parsing: `QUEUE_ENABLED`, `QUEUE_URL`, `QUEUE_STREAM`
  - Import queue factory function
  - In `_process_one()`: after `_serialize_result()`, either publish OR write file
  - Wrap serialized result in `MessageEnvelope` before publishing
  - When queue enabled and publish fails: raise exception (fail loudly)

  **Must NOT do**:
  - Change `_serialize_result()` output structure
  - Change `process_document()` logic
  - Add dual-write (queue replaces files)
  - Add retry logic (fail loudly)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Modifying existing CLI, clear injection point
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Task 3

  **References**:
  - `src/plm/extraction/fast/cli.py:208-211` - Current file write location (injection point)
  - `src/plm/extraction/fast/cli.py:19-28` - `_serialize_result()` function
  - `src/plm/shared/queue/factory.py` - Factory function (from Task 2)
  - `src/plm/shared/queue/types.py` - `MessageEnvelope` type (from Task 1)

  **Acceptance Criteria**:
  - [ ] `QUEUE_ENABLED=false` (default): writes files exactly as before
  - [ ] `QUEUE_ENABLED=true`: publishes to queue, does NOT write files
  - [ ] Message includes envelope with `source_service: "fast-extraction"`
  - [ ] Redis unavailable + `QUEUE_ENABLED=true`: exits with error code 1

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Standalone mode unchanged
    Tool: Bash
    Steps:
      1. mkdir -p /tmp/fast-test-in /tmp/fast-test-out
      2. echo "# Test Doc\n\nKubernetes is cool" > /tmp/fast-test-in/test.md
      3. QUEUE_ENABLED=false python -m plm.extraction.fast.cli \
           --input /tmp/fast-test-in --output /tmp/fast-test-out --workers 1
      4. ls /tmp/fast-test-out/*.json | wc -l
    Expected Result: Output is 1 (file created)
    Evidence: file count captured

  Scenario: Queue mode publishes instead of writing files
    Tool: Bash
    Preconditions: Redis running
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine && sleep 2
      2. mkdir -p /tmp/fast-test-in /tmp/fast-test-out
      3. echo "# Test\n\nKubernetes" > /tmp/fast-test-in/test.md
      4. QUEUE_ENABLED=true QUEUE_URL=redis://localhost:6379 QUEUE_STREAM=plm:extraction \
           python -m plm.extraction.fast.cli \
           --input /tmp/fast-test-in --output /tmp/fast-test-out --workers 1
      5. docker exec test-redis redis-cli XLEN plm:extraction
      6. ls /tmp/fast-test-out/*.json 2>/dev/null | wc -l
      7. docker stop test-redis && docker rm test-redis
    Expected Result: XLEN >= 1, file count = 0
    Evidence: Redis output + file count captured
  ```

  **Commit**: YES
  - Message: `feat(fast-extraction): add optional queue publishing`
  - Files: `src/plm/extraction/fast/cli.py`

---

- [ ] 6. Search service queue consumer

  **What to do**:
  - Create `src/plm/search/service/queue_consumer.py`
  - Subclass `QueueConsumer` from shared queue module
  - Implement `process_message()` to:
    - Parse `MessageEnvelope` and extract payload
    - Call `json_to_chunks()` on payload
    - Call `retriever.ingest_document(rebuild_index=False)`
  - Add batch processing: after N messages OR timeout, call `retriever.rebuild_index()`
  - Modify `src/plm/search/service/app.py`:
    - Parse `QUEUE_ENABLED` env var
    - If enabled: start queue consumer instead of DirectoryWatcher
    - If disabled: start DirectoryWatcher as before

  **Must NOT do**:
  - Modify `json_to_chunks()` function
  - Modify `HybridRetriever` internals
  - Run both queue consumer AND DirectoryWatcher

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: New component + app.py modification, batch processing logic
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 3, 4

  **References**:
  - `src/plm/search/service/watcher.py:297-323` - `_process_batch()` for batch ingestion pattern
  - `src/plm/search/service/watcher.py:324-354` - `_process_file()` for single doc pattern
  - `src/plm/search/service/watcher.py:36-74` - `json_to_chunks()` transformation
  - `src/plm/search/service/app.py:108-123` - DirectoryWatcher setup (conditional replacement)
  - `src/plm/shared/queue/consumer.py` - Base consumer class (from Task 4)

  **Acceptance Criteria**:
  - [ ] `src/plm/search/service/queue_consumer.py` exists
  - [ ] `QUEUE_ENABLED=false`: DirectoryWatcher starts as before
  - [ ] `QUEUE_ENABLED=true`: QueueConsumer starts, DirectoryWatcher does NOT start
  - [ ] Consumer calls `rebuild_index()` after batch (not per message)
  - [ ] Failed messages go to DLQ

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Standalone mode uses DirectoryWatcher
    Tool: Bash
    Steps:
      1. QUEUE_ENABLED=false INDEX_PATH=/tmp/search-test WATCH_DIR=/tmp/watch-test \
           timeout 5 python -c "
           import asyncio
           from plm.search.service.app import app, lifespan
           async def check():
               async with lifespan(app):
                   assert app.state.watcher is not None
                   print('DirectoryWatcher active')
           asyncio.run(check())
           " || true
    Expected Result: Prints "DirectoryWatcher active"
    Evidence: stdout captured

  Scenario: Queue mode starts consumer instead of watcher
    Tool: Bash
    Preconditions: Redis running
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine && sleep 2
      2. QUEUE_ENABLED=true QUEUE_URL=redis://localhost:6379 INDEX_PATH=/tmp/search-test \
           timeout 5 python -c "
           import asyncio
           from plm.search.service.app import app, lifespan
           async def check():
               async with lifespan(app):
                   assert app.state.watcher is None
                   assert app.state.queue_consumer is not None
                   print('QueueConsumer active')
           asyncio.run(check())
           " || true
      3. docker stop test-redis && docker rm test-redis
    Expected Result: Prints "QueueConsumer active"
    Failure Cleanup: Run `docker stop test-redis; docker rm test-redis`
    Evidence: stdout captured

  Scenario: Malformed message goes to DLQ
    Tool: Bash
    Preconditions: Redis running, search service configured
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine && sleep 2
      2. # Publish malformed message (missing required fields)
         docker exec test-redis redis-cli XADD plm:extraction '*' data '{"not_valid": true}'
      3. # Start consumer briefly to process the message
         QUEUE_ENABLED=true QUEUE_URL=redis://localhost:6379 INDEX_PATH=/tmp/search-test \
           QUEUE_MAX_RETRIES=1 timeout 10 python -c "
           from plm.search.service.queue_consumer import SearchQueueConsumer
           c = SearchQueueConsumer(redis_url='redis://localhost:6379', index_path='/tmp/search-test')
           c.process_pending()
           " || true
      4. # Verify message moved to DLQ
         docker exec test-redis redis-cli XLEN plm:dlq
      5. docker stop test-redis && docker rm test-redis
    Expected Result: DLQ length is 1
    Failure Cleanup: Run `docker stop test-redis; docker rm test-redis`
    Evidence: XLEN output captured
  ```

  **Commit**: YES
  - Message: `feat(search): add queue consumer for document ingestion`
  - Files: `src/plm/search/service/queue_consumer.py`, `src/plm/search/service/app.py`

---

- [ ] 7. Slow extraction format transformer

  **What to do**:
  - Create `src/plm/extraction/slow/format_transformer.py`
  - Implement `transform_to_fast_format(slow_result: dict) -> dict`
  - Transform slow extraction output structure to fast extraction structure:
    - `file` → `source_file`
    - `chunks[].terms[].term` → `headings[0].chunks[].entities[].text`
    - `chunks[].terms[].confidence` → `headings[0].chunks[].entities[].score`
    - Map `level` to appropriate `label` field
  - Unit test the transformation

  **Must NOT do**:
  - Change slow extraction pipeline logic
  - Change fast extraction output structure
  - Add complex validation

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function, clear input/output mapping
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Tasks 5, 6) OR Wave 4 start
  - **Blocks**: Task 8
  - **Blocked By**: Task 1

  **References**:
  - `src/plm/extraction/slow/cli.py:150-166` - Slow extraction output structure
  - `src/plm/extraction/fast/cli.py:19-28` - Fast extraction output structure (target)
  - `src/plm/shared/queue/types.py` - `ExtractionMessage` schema (from Task 1)

  **Acceptance Criteria**:
  - [ ] `src/plm/extraction/slow/format_transformer.py` exists
  - [ ] `transform_to_fast_format()` converts slow→fast structure
  - [ ] Output matches `ExtractionMessage` TypedDict
  - [ ] Keywords preserved (slow has `terms`, fast has `entities` + `keywords`)

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Transform matches fast extraction schema
    Tool: Bash
    Steps:
      1. python -c "
         from plm.extraction.slow.format_transformer import transform_to_fast_format
         slow = {
             'file': 'test.md',
             'processed_at': '2026-01-01T00:00:00Z',
             'chunks': [{
                 'text': 'Kubernetes is cool',
                 'chunk_index': 0,
                 'heading': None,
                 'terms': [{'term': 'Kubernetes', 'confidence': 0.9, 'level': 'HIGH', 'sources': ['v6']}]
             }]
         }
         fast = transform_to_fast_format(slow)
         assert 'source_file' in fast
         assert 'headings' in fast
         assert fast['headings'][0]['chunks'][0]['entities'][0]['text'] == 'Kubernetes'
         print('Transform valid')
         "
    Expected Result: Prints "Transform valid"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(slow-extraction): add format transformer for queue publishing`
  - Files: `src/plm/extraction/slow/format_transformer.py`

---

- [ ] 8. Slow extraction queue integration

  **What to do**:
  - Modify `src/plm/extraction/slow/cli.py`
  - Add env var parsing: `QUEUE_ENABLED`, `QUEUE_URL`, `QUEUE_STREAM`
  - Import queue factory and format transformer
  - In `watch_loop()` after `process_document()`:
    - Transform result using `transform_to_fast_format()`
    - Either publish OR write file (not both)
  - When queue enabled and publish fails: raise exception (fail loudly)

  **Must NOT do**:
  - Change `process_document()` or pipeline logic
  - Change V6 extraction strategy
  - Add dual-write

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Modifying existing CLI, similar to Task 5
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential after Task 7
  - **Blocks**: Tasks 9, 10
  - **Blocked By**: Tasks 3, 7

  **References**:
  - `src/plm/extraction/slow/cli.py:205-206` - `write_output()` call and `processed.add()` (injection point)
  - `src/plm/extraction/slow/format_transformer.py` - Transform function (from Task 7)
  - `src/plm/extraction/fast/cli.py` - Reference for queue integration pattern (from Task 5)

  **Acceptance Criteria**:
  - [ ] `QUEUE_ENABLED=false` (default): writes files exactly as before
  - [ ] `QUEUE_ENABLED=true`: transforms and publishes to queue, no files
  - [ ] Message includes envelope with `source_service: "slow-extraction"`
  - [ ] Published message has fast extraction format (after transform)

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Queue mode publishes transformed message
    Tool: Bash
    Preconditions: 
      - Redis running
      - vocab files available (data/vocabularies/auto_vocab.json, train_documents.json)
      - ANTHROPIC_API_KEY or OPENCODE_AUTH_PATH set (slow extraction uses Claude LLM)
      - Note: First run downloads SentenceTransformer model (~90MB), takes 30+ seconds
    Steps:
      1. docker run -d --name test-redis -p 6379:6379 redis:7-alpine && sleep 2
      2. mkdir -p /tmp/slow-test-in /tmp/slow-test-out
      3. echo "Kubernetes deployments use pods" > /tmp/slow-test-in/test.txt
      4. QUEUE_ENABLED=true QUEUE_URL=redis://localhost:6379 QUEUE_STREAM=plm:extraction \
           INPUT_DIR=/tmp/slow-test-in OUTPUT_DIR=/tmp/slow-test-out \
           VOCAB_PATH=data/vocabularies/auto_vocab.json \
           TRAIN_DOCS_PATH=data/vocabularies/train_documents.json \
           PROCESS_ONCE=true python -m plm.extraction.slow.cli
      5. docker exec test-redis redis-cli XRANGE plm:extraction - + COUNT 1
      6. docker stop test-redis && docker rm test-redis
    Expected Result: Message contains "headings" key (fast format, after transformation)
    Failure Cleanup: Run `docker stop test-redis; docker rm test-redis`
    Evidence: Redis XRANGE output captured
    
  Scenario: Standalone mode unchanged (no LLM required for this test)
    Tool: Bash
    Preconditions: vocab files available
    Steps:
      1. mkdir -p /tmp/slow-test-in /tmp/slow-test-out
      2. echo "Test content" > /tmp/slow-test-in/test.txt
      3. QUEUE_ENABLED=false INPUT_DIR=/tmp/slow-test-in OUTPUT_DIR=/tmp/slow-test-out \
           VOCAB_PATH=data/vocabularies/auto_vocab.json \
           TRAIN_DOCS_PATH=data/vocabularies/train_documents.json \
           DRY_RUN=true PROCESS_ONCE=true python -m plm.extraction.slow.cli 2>&1 | grep -q "Would write"
    Expected Result: "Would write" appears in output (dry run mode)
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(slow-extraction): add optional queue publishing with format transform`
  - Files: `src/plm/extraction/slow/cli.py`

---

- [ ] 9. Docker Compose full pipeline

  **What to do**:
  - Create `docker/docker-compose.full.yml`
  - Define services: `redis`, `fast-extraction`, `slow-extraction`, `search-service`
  - Add shared network `plm-network`
  - Redis service:
    - `redis:7-alpine`
    - `appendonly yes` for persistence
    - Healthcheck with `redis-cli ping`
  - All services:
    - `QUEUE_ENABLED=true`
    - `QUEUE_URL=redis://redis:6379`
    - `depends_on: redis: condition: service_healthy`
  - Document usage in comments

  **Must NOT do**:
  - Change individual service compose files
  - Add monitoring services (prometheus, grafana)
  - Add external network exposure beyond ports needed

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single YAML file, clear structure from existing compose files
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 5)
  - **Blocks**: Task 10
  - **Blocked By**: Tasks 5, 6, 8

  **References**:
  - `docker/docker-compose.search.yml` - Search service compose pattern
  - `src/plm/extraction/fast/docker-compose.yml` - Fast extraction compose pattern
  - `src/plm/extraction/slow/docker-compose.yml` - Slow extraction compose pattern
  - `poc/modular_retrieval_pipeline/docker-compose.yml` - Redis service pattern
  - Librarian research: `depends_on: condition: service_healthy` pattern

  **Acceptance Criteria**:
  - [ ] `docker/docker-compose.full.yml` exists
  - [ ] All 4 services defined: redis, fast-extraction, slow-extraction, search-service
  - [ ] Shared network `plm-network` connecting all services
  - [ ] Redis has healthcheck
  - [ ] `docker compose -f docker/docker-compose.full.yml config` validates without error

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Docker Compose config is valid
    Tool: Bash
    Steps:
      1. docker compose -f docker/docker-compose.full.yml config --quiet
    Expected Result: Exit code 0 (no output = valid)
    Evidence: Exit code captured

  Scenario: Services start and connect
    Tool: Bash
    Steps:
      1. docker compose -f docker/docker-compose.full.yml up -d redis
      2. sleep 5
      3. docker compose -f docker/docker-compose.full.yml exec redis redis-cli ping
      4. docker compose -f docker/docker-compose.full.yml down
    Expected Result: Redis responds "PONG"
    Evidence: Redis response captured
  ```

  **Commit**: YES
  - Message: `feat(docker): add full pipeline compose with Redis queue`
  - Files: `docker/docker-compose.full.yml`

---

- [ ] 10. Integration tests

  **What to do**:
  - Create `tests/queue/` directory
  - Create `tests/queue/test_queue_module.py` - Unit tests for queue module
  - Create `tests/queue/test_integration.py` - End-to-end queue flow test
  - Integration test should:
    - Start Redis container (pytest-docker or subprocess)
    - Run fast extraction with queue enabled
    - Verify message in Redis stream
    - Run search consumer to ingest
    - Verify document indexed in search service
  - Add pytest markers for `@pytest.mark.integration` (skip without Docker)

  **Must NOT do**:
  - Add performance/load tests
  - Add stress tests
  - Add flaky timing-dependent tests

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple test files, Docker integration, complex setup
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 6, final)
  - **Blocks**: None (final task)
  - **Blocked By**: Task 9

  **References**:
  - `tests/` - Existing test structure and patterns
  - `src/plm/shared/queue/` - Module to test (from Tasks 1-4)
  - `pytest.ini` or `pyproject.toml` - Test configuration

  **Acceptance Criteria**:
  - [ ] `tests/queue/test_queue_module.py` exists with unit tests
  - [ ] `tests/queue/test_integration.py` exists with e2e test
  - [ ] Unit tests pass: `pytest tests/queue/test_queue_module.py -v`
  - [ ] Integration tests pass: `pytest tests/queue/test_integration.py -v --integration` (with Docker)
  - [ ] Tests don't break existing tests: `pytest tests/ -v`

  **Agent-Executed QA Scenarios:**
  ```
  Scenario: Unit tests pass
    Tool: Bash
    Steps:
      1. pytest tests/queue/test_queue_module.py -v
    Expected Result: All tests pass
    Evidence: pytest output captured

  Scenario: Integration test passes (with Docker)
    Tool: Bash
    Preconditions: Docker available
    Steps:
      1. pytest tests/queue/test_integration.py -v -m integration
    Expected Result: All tests pass
    Evidence: pytest output captured

  Scenario: Existing tests not broken
    Tool: Bash
    Steps:
      1. pytest tests/ -v --ignore=tests/queue/test_integration.py
    Expected Result: No regressions
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `test(queue): add unit and integration tests`
  - Files: `tests/queue/test_queue_module.py`, `tests/queue/test_integration.py`, `tests/queue/__init__.py`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(queue): add canonical message schema types` | types.py, __init__.py | python -c import |
| 2 | `feat(queue): add MessageQueue protocol and NullQueue` | protocol.py, null_queue.py | python -c import |
| 3 | `feat(queue): add RedisStreamQueue implementation` | redis_queue.py | manual redis test |
| 4 | `feat(queue): add QueueConsumer base with DLQ support` | consumer.py | manual redis test |
| 5 | `feat(fast-extraction): add optional queue publishing` | cli.py | QUEUE_ENABLED test |
| 6 | `feat(search): add queue consumer for document ingestion` | queue_consumer.py, app.py | QUEUE_ENABLED test |
| 7 | `feat(slow-extraction): add format transformer for queue publishing` | format_transformer.py | python -c import |
| 8 | `feat(slow-extraction): add optional queue publishing with format transform` | cli.py | QUEUE_ENABLED test |
| 9 | `feat(docker): add full pipeline compose with Redis queue` | docker-compose.full.yml | docker compose config |
| 10 | `test(queue): add unit and integration tests` | tests/queue/*.py | pytest |

---

## Success Criteria

### Verification Commands
```bash
# Unit tests
pytest tests/queue/test_queue_module.py -v

# Integration tests (requires Docker)
pytest tests/queue/test_integration.py -v -m integration

# All existing tests still pass
pytest tests/ -v --ignore=tests/queue/test_integration.py

# Docker Compose validation
docker compose -f docker/docker-compose.full.yml config

# End-to-end smoke test
docker compose -f docker/docker-compose.full.yml up -d
# ... wait for healthy ...
docker compose -f docker/docker-compose.full.yml down
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] All unit tests pass
- [ ] Integration tests pass
- [ ] Docker Compose starts full pipeline
- [ ] Standalone mode (QUEUE_ENABLED=false) works for all services
- [ ] Queue mode (QUEUE_ENABLED=true) works end-to-end
