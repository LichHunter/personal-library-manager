# Draft: Queue Integration for PLM Services

## Requirements (confirmed from user)

- **Goal**: Connect fast extraction, slow extraction, and search services via message queue
- **Data flow**: Both extraction services → outputs to queue → Search service reads and processes
- **Standalone mode**: All services MUST retain ability to work independently (queue is optional)
- **Docker**: Combined docker-compose with all 3 services + queue + full pipeline
- **Non-Docker**: Support running services as simple processes with queue

## Technical Decisions (confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Queue technology | Redis Streams | Already proven in POC, lightweight, Kafka-like semantics |
| Queue publishers | Both fast AND slow extraction | Unified ingestion pipeline |
| Failure handling | Dead-letter queue | Robust: failed messages move to DLQ after N retries |
| Docker scope | Full pipeline | Redis + Fast + Slow + Search with queue consumer |
| Test strategy | Tests after implementation | Agent-executed QA + unit tests after |
| Standalone mode | `QUEUE_ENABLED=false` default | Adapter pattern with NullQueue |
| Message format | Normalize at producer | Slow extraction transforms to fast extraction format |
| Output mode | Queue replaces files | When `QUEUE_ENABLED=true`, no file output |
| Redis failure | Fail loudly | Extraction stops with error if Redis unavailable |
| DirectoryWatcher | Disable when queue enabled | Queue consumer is the only ingestion path |

## Auto-Resolved (Sensible Defaults)

| Decision | Default | Rationale |
|----------|---------|-----------|
| Consumer group name | `{service-name}` | Simple, identifiable |
| Consumer name | `{hostname}-{pid}` | Unique per instance |
| Max message size | Warn at 1MB | Log warning, don't block |
| DLQ processing | Manual via redis-cli | Out of scope for MVP |
| Message ordering | Last write wins | Documents can be re-indexed |

## Research Findings (from codebase exploration)

### Current Architecture

**Fast Extraction** (`src/plm/extraction/fast/cli.py`):
- Input: Reads from `--input` directory (glob patterns: `**/*.md,**/*.txt`)
- Output: Writes JSON files to `--output` directory (lines 208-211)
- Key function: `_serialize_result()` converts `DocumentResult` to dict
- Threading support: Uses ThreadPoolExecutor for parallel processing
- **Extension point**: `_serialize_result()` already separates serialization from writing
- **No output abstraction exists** — writes inline with `Path.write_text()`

**Slow Extraction** (`src/plm/extraction/slow/cli.py`):
- Input: Reads from `INPUT_DIR` env var (polling-based)
- Output: Writes JSON to `OUTPUT_DIR` via `write_output()` (lines 169-176)
- Watch mode: Polls every `POLL_INTERVAL` seconds
- Process-once mode: Processes existing files and exits
- **Extension point**: `write_output()` function is separate from processing

**Search Service** (`src/plm/search/service/`):
- `DirectoryWatcher` in `watcher.py` watches for JSON files
- Uses `json_to_chunks()` to convert extraction output to chunks
- Calls `HybridRetriever.ingest_document()` for indexing
- Batch debounce: Collects files for 1s before processing
- Graceful shutdown: Drains queue before exit
- **No HTTP `/ingest` endpoint** — query-only API
- **BM25 rebuild is full rebuild** — O(N) in corpus size, not incremental

**Existing Redis Pattern** (POC only):
- `poc/modular_retrieval_pipeline/cache/redis_client.py`
- Used for CACHING (embeddings, keywords), NOT for messaging
- Has graceful fallback pattern when Redis unavailable
- **Can be promoted to production** — clean implementation

### Current Docker Setup

| Service | Compose File | Networking |
|---------|--------------|------------|
| Fast extraction | `src/plm/extraction/fast/docker-compose.yml` | Standalone |
| Slow extraction | `src/plm/extraction/slow/docker-compose.yml` | Standalone |
| Search service | `docker/docker-compose.search.yml` | Standalone |

- **No shared Docker network** — services isolated per compose file
- **Filesystem-as-message-bus** — data flows via bind-mounted host directories
- **`watch` profile pattern** — established convention for long-running vs. one-shot services

### JSON Message Format (already standardized)

```json
{
  "source_file": "path/to/original.md",
  "headings": [
    {
      "heading": "Section Title",
      "chunks": [
        {
          "text": "chunk text",
          "keywords": ["kw1", "kw2"],
          "entities": [{"text": "Kubernetes", "label": "technology", "score": 0.85}],
          "start_char": 0,
          "end_char": 412
        }
      ]
    }
  ],
  "avg_confidence": 0.72,
  "total_entities": 15
}
```
This format is produced by fast extraction and consumed by `watcher.py:json_to_chunks()`.

## Open Questions

1. **Queue technology choice**: Redis Streams vs RabbitMQ vs other?
2. **Slow extraction in scope?**: User mentioned fast → search, but slow also?
3. **Acknowledgment pattern**: At-least-once? At-most-once?
4. **Retry/dead-letter handling**: What happens on failure?
5. **Non-Docker mode**: How to configure queue connection (localhost vs service name)?

## Architectural Observations

**Cleanest integration points identified:**
- Fast extraction: `_process_one()` in cli.py after `_serialize_result()` (line 210)
- Slow extraction: After `process_document()` before `write_output()` (line 206)
- Search service: Alongside or replacing `DirectoryWatcher` pattern

**Message format already standardized:**
- Fast extraction output JSON is already consumed by `watcher.py:json_to_chunks()`
- Same format can be used for queue messages (no transformation needed)

**Redis Streams recommended:**
- Already proven in POC (`poc/modular_retrieval_pipeline/cache/redis_client.py`)
- Lower operational overhead than RabbitMQ
- Graceful fallback pattern already implemented
- Can reuse existing Docker image (`redis:7-alpine`)

**Known issues to address:**
- BM25 rebuild is not thread-safe (needs lock)
- Chunk orphaning on re-ingest (no DELETE before INSERT)
- Embedder not thread-safe (concurrent ingestion races)

## Scope Boundaries

- **INCLUDE**: *Pending confirmation*
- **EXCLUDE**: *Pending confirmation*
