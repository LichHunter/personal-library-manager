# Docker Redis Cache for Modular Retrieval Pipeline

## TL;DR

> **Quick Summary**: Add Redis-backed caching layer for expensive enrichment computations (keywords, entities, embeddings) to dramatically speed up benchmark runs on unchanged corpora.
> 
> **Deliverables**:
> - Redis cache client with graceful fallback
> - Three cached component wrappers (keywords, entities, embeddings)
> - Docker Compose for Redis with persistence
> - Benchmark integration with `--no-cache` flag
> 
> **Estimated Effort**: Medium
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (Redis client) -> Tasks 2,3,4 (wrappers) -> Task 5,6 (integration)

---

## Context

### Original Request
Create a tool to interact with Docker database (Redis) for storing enrichment cache. Maximize cache usage during benchmarks but only use cache when the cache algorithm parameters match exactly.

### Interview Summary
**Key Discussions**:
- Docker for clean isolation (easy to reset/destroy)
- Redis chosen for speed and TTL support
- Strict cache validity (ALL params must match)
- Benchmarks only (no production concerns)
- Component wrappers for transparent caching
- Graceful fallback if Redis unavailable
- Cache enabled by default, `--no-cache` to disable

**Research Findings**:
- Current indexing: ~140-180s for 7,000 chunks
- Embedding encoding: 100-120s (70% of time) - BIGGEST WIN
- Enrichment (YAKE + spaCy): 30-40s (25% of time)
- Expected speedup: ~90% faster on subsequent runs (~140s -> ~15s)
- Baseline has `EnrichmentCache` (disk-based), modular has none
- Cache size estimate: ~30-50MB for full corpus

### Metis Review
**Identified Gaps** (addressed):
- Embedding caching approach: CachingEmbedder wrapper (cleaner, reusable)
- Cache key format: Full config in key (explicit, future-proof)
- Volume mount: Named Docker volume (cleaner management)

---

## Work Objectives

### Core Objective
Add transparent Redis caching to the modular retrieval pipeline's expensive components (keywords, entities, embeddings) to eliminate redundant computation during repeated benchmark runs.

### Concrete Deliverables
1. `poc/modular_retrieval_pipeline/cache/redis_client.py` - Redis connection and operations
2. `poc/modular_retrieval_pipeline/cache/cached_keyword_extractor.py` - Wrapper for KeywordExtractor
3. `poc/modular_retrieval_pipeline/cache/cached_entity_extractor.py` - Wrapper for EntityExtractor
4. `poc/modular_retrieval_pipeline/cache/caching_embedder.py` - Wrapper for SentenceTransformer
5. `poc/modular_retrieval_pipeline/docker-compose.yml` - Redis service definition
6. Modified `benchmark.py` with `--no-cache` flag
7. Modified `modular_enriched_hybrid.py` to use cached components

### Definition of Done
- [ ] `docker compose up -d` starts Redis 7 with persistence
- [ ] First benchmark run populates cache (7,000+ entries)
- [ ] Second benchmark run shows >80% cache hit rate
- [ ] Second benchmark indexing time <20s (vs ~140s without cache)
- [ ] `--no-cache` flag bypasses all caching
- [ ] Redis down = graceful fallback (no errors, just slower)

### Must Have
- Strict cache key: `{component}:{config_hash}:{content_hash}`
- Graceful fallback on Redis connection failure
- Cache enabled by default
- Logging of cache hits/misses

### Must NOT Have (Guardrails)
- No CLI management tools (use redis-cli directly)
- No cache warming utilities
- No cross-machine cache sharing
- No production deployment considerations
- No automated tests (manual verification only)
- No caching of ContentEnricher (too fast, <1ms)
- No caching of QueryRewriter/LLM responses

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (adding new)
- **User wants tests**: NO - manual verification
- **Framework**: N/A

### Manual Verification Procedures

Each task includes verification via:
1. **Benchmark output logs** - Cache hit/miss counts
2. **Timing comparisons** - First run vs second run
3. **redis-cli inspection** - `DBSIZE`, `KEYS *`, `GET key`

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Docker Compose + Redis client (foundation)
└── (No parallel tasks - foundation required)

Wave 2 (After Wave 1):
├── Task 2: CachedKeywordExtractor
├── Task 3: CachedEntityExtractor
└── Task 4: CachingEmbedder
    (All three can run in parallel - independent wrappers)

Wave 3 (After Wave 2):
├── Task 5: Benchmark integration
└── Task 6: Orchestrator integration
    (Both depend on wrappers existing)

Wave 4 (After Wave 3):
└── Task 7: End-to-end verification
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5, 6 | None (foundation) |
| 2 | 1 | 5, 6 | 3, 4 |
| 3 | 1 | 5, 6 | 2, 4 |
| 4 | 1 | 5, 6 | 2, 3 |
| 5 | 2, 3, 4 | 7 | 6 |
| 6 | 2, 3, 4 | 7 | 5 |
| 7 | 5, 6 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Agents |
|------|-------|-------------------|
| 1 | 1 | delegate_task(category="quick", load_skills=[]) |
| 2 | 2, 3, 4 | Three parallel delegate_task(category="quick", ..., run_in_background=true) |
| 3 | 5, 6 | Two parallel delegate_task(category="quick", ...) |
| 4 | 7 | Manual verification by executor |

---

## TODOs

- [x] 1. Create Docker Compose and Redis Cache Client

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/docker-compose.yml` with Redis 7 service
  - Use named volume `plm_redis_cache` for persistence
  - Expose port 6379 (standard Redis port)
  - Create `poc/modular_retrieval_pipeline/cache/__init__.py`
  - Create `poc/modular_retrieval_pipeline/cache/redis_client.py` with:
    - `RedisCacheClient` class
    - Connection with graceful fallback (returns None on connection error)
    - `get(key: str) -> bytes | None`
    - `set(key: str, value: bytes, ttl: int | None = None)`
    - `delete(key: str)`
    - `exists(key: str) -> bool`
    - `dbsize() -> int`
    - Static method `make_key(component: str, config: dict, content: str) -> str`
    - Config hash using `hashlib.sha256` of sorted JSON
    - Content hash using `hashlib.sha256`

  **Must NOT do**:
  - No complex connection pooling (single connection is fine for benchmarks)
  - No retry logic (graceful fallback is sufficient)
  - No async support (benchmarks are synchronous)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward implementation, single file, clear specs
  - **Skills**: `[]`
    - No special skills needed for basic Python + Docker
  - **Skills Evaluated but Omitted**:
    - `playwright`: Not a browser task
    - `frontend-ui-ux`: Not a frontend task

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (foundation)
  - **Blocks**: Tasks 2, 3, 4, 5, 6
  - **Blocked By**: None (can start immediately)

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:102-139` - Existing EnrichmentCache pattern for cache get/put interface

  **API/Type References**:
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py:24-37` - YAKE config params to include in config hash
  - `poc/modular_retrieval_pipeline/components/entity_extractor.py:16-22` - Entity types and spaCy model to include in config hash
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py:35-45` - Embedding model params to include in config hash

  **External References**:
  - Redis Python client: `pip install redis` - https://redis-py.readthedocs.io/
  - Docker Compose Redis: https://hub.docker.com/_/redis (use redis:7-alpine)

  **WHY Each Reference Matters**:
  - EnrichmentCache pattern shows the get/put interface that existing code expects
  - Component files show exactly which config params affect output (for cache key)
  - Redis docs for proper connection handling and error types

  **Acceptance Criteria**:

  ```bash
  # 1. Start Redis
  cd poc/modular_retrieval_pipeline && docker compose up -d
  # Assert: Container 'plm-redis-cache' running
  docker compose ps --format json | jq '.[0].State'
  # Expected: "running"
  
  # 2. Test client connection
  cd /home/fujin/Code/personal-library-manager
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
client = RedisCacheClient()
print('Connected:', client.is_connected())
client.set('test', b'hello')
print('Get test:', client.get('test'))
client.delete('test')
print('After delete:', client.get('test'))
"
  # Expected output:
  # Connected: True
  # Get test: b'hello'
  # After delete: None
  
  # 3. Test cache key generation
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
key = RedisCacheClient.make_key('keywords', {'max_keywords': 10}, 'test content')
print('Key format:', key)
print('Key length:', len(key))
"
  # Expected: Key like 'keywords:abc123:def456' (component:config_hash:content_hash)
  # Key length should be ~50-60 chars
  
  # 4. Test graceful fallback
  docker compose down
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
client = RedisCacheClient()
print('Connected:', client.is_connected())
result = client.get('anything')
print('Get returns:', result)
"
  # Expected: Connected: False, Get returns: None (no exception)
  ```

  **Evidence to Capture:**
  - Terminal output from all verification commands
  - docker-compose.yml content
  - redis_client.py content

  **Commit**: YES
  - Message: `feat(cache): add Redis cache client with Docker Compose`
  - Files: `poc/modular_retrieval_pipeline/docker-compose.yml`, `poc/modular_retrieval_pipeline/cache/__init__.py`, `poc/modular_retrieval_pipeline/cache/redis_client.py`
  - Pre-commit: N/A (no tests)

---

- [x] 2. Create CachedKeywordExtractor Wrapper

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/cache/cached_keyword_extractor.py`
  - Implement `CachedKeywordExtractor` class that:
    - Wraps `KeywordExtractor` component
    - Implements same `process(data: dict) -> dict` interface
    - On cache hit: return cached keywords
    - On cache miss: call wrapped extractor, cache result, return
    - Tracks hit/miss counts for logging
    - Uses config hash from KeywordExtractor's actual params
  - Cache key format: `keywords:{config_hash}:{content_hash}`
  - Cache value: JSON-encoded list of keywords
  - Include version prefix for future invalidation: `v1:`

  **Must NOT do**:
  - Don't modify original KeywordExtractor
  - Don't add TTL (cache is valid indefinitely for same config)
  - Don't add compression (keywords are small strings)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, clear wrapper pattern, straightforward
  - **Skills**: `[]`
    - No special skills needed
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable to this backend Python task

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 1 (needs RedisCacheClient)

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py` - Original component to wrap, understand input/output
  - `poc/modular_retrieval_pipeline/base.py:25-45` - Component protocol interface (`process` method signature)
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:122-139` - Cache get/put pattern

  **API/Type References**:
  - `poc/modular_retrieval_pipeline/cache/redis_client.py` - RedisCacheClient class (from Task 1)

  **WHY Each Reference Matters**:
  - KeywordExtractor shows exact config params (max_keywords, YAKE settings) for cache key
  - Component protocol ensures wrapper is drop-in compatible
  - EnrichmentCache shows the caching pattern to follow

  **Acceptance Criteria**:

  ```bash
  # Ensure Redis is running
  cd poc/modular_retrieval_pipeline && docker compose up -d
  
  # Test wrapper
  cd /home/fujin/Code/personal-library-manager
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
from poc.modular_retrieval_pipeline.cache.cached_keyword_extractor import CachedKeywordExtractor
from poc.modular_retrieval_pipeline.components.keyword_extractor import KeywordExtractor

cache = RedisCacheClient()
extractor = KeywordExtractor(max_keywords=10)
cached = CachedKeywordExtractor(extractor, cache)

# First call - cache miss
result1 = cached.process({'content': 'Kubernetes pod autoscaling with HPA'})
print('Keywords:', result1.get('keywords', []))
print('Hits:', cached.hits, 'Misses:', cached.misses)

# Second call - cache hit
result2 = cached.process({'content': 'Kubernetes pod autoscaling with HPA'})
print('Same result:', result1['keywords'] == result2['keywords'])
print('Hits:', cached.hits, 'Misses:', cached.misses)
"
  # Expected output:
  # Keywords: ['kubernetes', 'pod', 'autoscaling', ...] (non-empty list)
  # Hits: 0 Misses: 1
  # Same result: True
  # Hits: 1 Misses: 1
  
  # Verify in Redis
  docker exec plm-redis-cache redis-cli KEYS 'v1:keywords:*' | head -1
  # Expected: At least one key like 'v1:keywords:abc123:def456'
  ```

  **Evidence to Capture:**
  - Terminal output showing cache miss then hit
  - Redis key verification

  **Commit**: YES (groups with Tasks 3, 4)
  - Message: `feat(cache): add cached component wrappers for keywords, entities, embeddings`
  - Files: All three wrapper files
  - Pre-commit: N/A

---

- [x] 3. Create CachedEntityExtractor Wrapper

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/cache/cached_entity_extractor.py`
  - Implement `CachedEntityExtractor` class that:
    - Wraps `EntityExtractor` component
    - Implements same `process(data: dict) -> dict` interface
    - On cache hit: return cached entities
    - On cache miss: call wrapped extractor, cache result, return
    - Tracks hit/miss counts for logging
    - Uses config hash from EntityExtractor's actual params (entity_types, spacy_model)
  - Cache key format: `v1:entities:{config_hash}:{content_hash}`
  - Cache value: JSON-encoded dict of entities

  **Must NOT do**:
  - Don't modify original EntityExtractor
  - Don't add TTL
  - Don't cache spaCy model loading (already cached at module level)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, same pattern as Task 2
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 1 (needs RedisCacheClient)

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/components/entity_extractor.py` - Original component, config params
  - `poc/modular_retrieval_pipeline/cache/cached_keyword_extractor.py` - Same wrapper pattern (from Task 2)

  **API/Type References**:
  - `poc/modular_retrieval_pipeline/cache/redis_client.py` - RedisCacheClient class

  **WHY Each Reference Matters**:
  - EntityExtractor shows entity_types and spacy_model for cache key
  - Task 2's wrapper provides the exact pattern to follow

  **Acceptance Criteria**:

  ```bash
  cd /home/fujin/Code/personal-library-manager
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
from poc.modular_retrieval_pipeline.cache.cached_entity_extractor import CachedEntityExtractor
from poc.modular_retrieval_pipeline.components.entity_extractor import EntityExtractor

cache = RedisCacheClient()
extractor = EntityExtractor()
cached = CachedEntityExtractor(extractor, cache)

# First call - cache miss
result1 = cached.process({'content': 'Google Cloud Platform runs Kubernetes Engine for container orchestration.'})
print('Entities:', result1.get('entities', {}))
print('Hits:', cached.hits, 'Misses:', cached.misses)

# Second call - cache hit
result2 = cached.process({'content': 'Google Cloud Platform runs Kubernetes Engine for container orchestration.'})
print('Same result:', result1['entities'] == result2['entities'])
print('Hits:', cached.hits, 'Misses:', cached.misses)
"
  # Expected output:
  # Entities: {'ORG': ['Google Cloud Platform'], 'PRODUCT': ['Kubernetes Engine'], ...}
  # Hits: 0 Misses: 1
  # Same result: True
  # Hits: 1 Misses: 1
  ```

  **Evidence to Capture:**
  - Terminal output showing cache miss then hit

  **Commit**: YES (groups with Tasks 2, 4)
  - Message: `feat(cache): add cached component wrappers for keywords, entities, embeddings`
  - Files: All three wrapper files
  - Pre-commit: N/A

---

- [x] 4. Create CachingEmbedder Wrapper

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/cache/caching_embedder.py`
  - Implement `CachingEmbedder` class that:
    - Wraps `SentenceTransformer` (not EmbeddingEncoder)
    - Implements `encode(texts: list[str]) -> np.ndarray` interface
    - Checks cache for each text, collects misses
    - Batch encodes only misses
    - Stores new embeddings in cache
    - Returns combined results in original order
    - Tracks hit/miss counts
  - Cache key format: `v1:embeddings:{model_hash}:{content_hash}`
  - Cache value: numpy array as bytes (`embedding.tobytes()`)
  - Config hash includes: model name, normalize_embeddings setting
  - Store embedding dimension for validation

  **Must NOT do**:
  - Don't modify how SentenceTransformer is loaded
  - Don't add compression (embeddings are already dense)
  - Don't change batch_size behavior

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file, clear interface
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Tasks 5, 6
  - **Blocked By**: Task 1 (needs RedisCacheClient)

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:255-275` - How embedder.encode() is called
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py:35-55` - Model loading and encode params

  **API/Type References**:
  - `sentence_transformers.SentenceTransformer` - Interface to wrap (encode method)
  - `numpy.ndarray.tobytes()` / `numpy.frombuffer()` - Binary serialization

  **External References**:
  - SentenceTransformers docs: https://www.sbert.net/docs/usage/embeddings.html

  **WHY Each Reference Matters**:
  - modular_enriched_hybrid shows exact encode() call pattern to match
  - embedding_encoder shows model params that affect output

  **Acceptance Criteria**:

  ```bash
  cd /home/fujin/Code/personal-library-manager
  python -c "
from sentence_transformers import SentenceTransformer
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
from poc.modular_retrieval_pipeline.cache.caching_embedder import CachingEmbedder
import numpy as np

cache = RedisCacheClient()
base_embedder = SentenceTransformer('BAAI/bge-base-en-v1.5')
embedder = CachingEmbedder(base_embedder, cache, model_name='BAAI/bge-base-en-v1.5')

texts = ['Kubernetes pods', 'Docker containers', 'Kubernetes pods']  # Note: duplicate

# First call - cache misses for unique texts
embeddings1 = embedder.encode(texts)
print('Shape:', embeddings1.shape)
print('Hits:', embedder.hits, 'Misses:', embedder.misses)

# Second call - should be all cache hits
embeddings2 = embedder.encode(texts)
print('Same embeddings:', np.allclose(embeddings1, embeddings2))
print('Hits:', embedder.hits, 'Misses:', embedder.misses)
"
  # Expected output:
  # Shape: (3, 768)
  # Hits: 1 Misses: 2  (duplicate was hit on first call)
  # Same embeddings: True
  # Hits: 4 Misses: 2  (all hits on second call)
  ```

  **Evidence to Capture:**
  - Terminal output showing embedding caching works
  - Shape verification (768 dimensions)

  **Commit**: YES (groups with Tasks 2, 3)
  - Message: `feat(cache): add cached component wrappers for keywords, entities, embeddings`
  - Files: All three wrapper files
  - Pre-commit: N/A

---

- [x] 5. Integrate Caching into Benchmark

  **What to do**:
  - Modify `poc/modular_retrieval_pipeline/benchmark.py`:
    - Add `--no-cache` flag (cache enabled by default)
    - Import cached wrapper classes
    - Pass cache client to orchestrator when cache enabled
    - Log cache hit/miss statistics after indexing
  - Add cache status to benchmark output:
    - "Cache: enabled (Redis connected)" or "Cache: disabled"
    - After indexing: "Cache stats: X hits, Y misses (Z% hit rate)"

  **Must NOT do**:
  - Don't add `--cache` flag (cache is default ON)
  - Don't add cache statistics to JSON output (just console logs)
  - Don't change existing benchmark logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Modifying existing file, clear integration points
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 4 (needs all wrappers)

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:1-50` - Existing argument parsing
  - `poc/modular_retrieval_pipeline/benchmark.py:150-200` - Where orchestrator is created

  **API/Type References**:
  - All cached wrapper classes from Tasks 2, 3, 4
  - `poc/modular_retrieval_pipeline/cache/redis_client.py` - RedisCacheClient

  **WHY Each Reference Matters**:
  - benchmark.py structure shows where to add flag and integration
  - Wrappers provide the interface to use

  **Acceptance Criteria**:

  ```bash
  cd /home/fujin/Code/personal-library-manager
  
  # Test --no-cache flag exists
  python poc/modular_retrieval_pipeline/benchmark.py --help | grep -i cache
  # Expected: Shows --no-cache option
  
  # Test cache disabled mode (should work without Redis)
  docker compose -f poc/modular_retrieval_pipeline/docker-compose.yml down
  python poc/modular_retrieval_pipeline/benchmark.py --no-cache --strategy modular-no-llm --questions poc/chunking_benchmark_v2/corpus/needle_questions.json 2>&1 | head -20
  # Expected: "Cache: disabled" in output, benchmark runs successfully
  
  # Test cache enabled mode
  docker compose -f poc/modular_retrieval_pipeline/docker-compose.yml up -d
  python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --questions poc/chunking_benchmark_v2/corpus/needle_questions.json 2>&1 | grep -i cache
  # Expected: "Cache: enabled (Redis connected)" and cache statistics
  ```

  **Evidence to Capture:**
  - Terminal output showing --no-cache flag works
  - Terminal output showing cache statistics

  **Commit**: YES (groups with Task 6)
  - Message: `feat(benchmark): integrate Redis caching with --no-cache flag`
  - Files: `poc/modular_retrieval_pipeline/benchmark.py`, `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py`
  - Pre-commit: N/A

---

- [x] 6. Integrate Caching into Orchestrator

  **What to do**:
  - Modify `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py`:
    - Add optional `cache: RedisCacheClient | None = None` parameter to `__init__`
    - If cache provided, wrap components with cached versions:
      - `self._keyword_extractor = CachedKeywordExtractor(KeywordExtractor(...), cache)`
      - `self._entity_extractor = CachedEntityExtractor(EntityExtractor(), cache)`
    - Add method to set cached embedder: `set_cached_embedder(embedder, cache)`
    - Add method to get cache stats: `get_cache_stats() -> dict`
  - Also update `modular_enriched_hybrid_llm.py` with same changes (keep parity)

  **Must NOT do**:
  - Don't change component creation if no cache provided (preserve non-cached behavior)
  - Don't auto-create cache client (caller provides it)
  - Don't add cache to ContentEnricher

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Modifying existing files, clear pattern
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 4 (needs all wrappers)

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:100-130` - Component initialization
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:145-160` - set_embedder method

  **API/Type References**:
  - All cached wrapper classes from Tasks 2, 3, 4
  - `poc/modular_retrieval_pipeline/cache/redis_client.py` - RedisCacheClient

  **WHY Each Reference Matters**:
  - Shows exactly where components are created and how to wrap them
  - set_embedder shows pattern for optional injection

  **Acceptance Criteria**:

  ```bash
  cd /home/fujin/Code/personal-library-manager
  python -c "
from poc.modular_retrieval_pipeline.cache.redis_client import RedisCacheClient
from poc.modular_retrieval_pipeline.modular_enriched_hybrid import ModularEnrichedHybrid

# Without cache
strategy1 = ModularEnrichedHybrid()
print('No cache:', strategy1.get_cache_stats())

# With cache
cache = RedisCacheClient()
strategy2 = ModularEnrichedHybrid(cache=cache)
print('With cache:', strategy2.get_cache_stats())
print('Cache enabled:', cache.is_connected())
"
  # Expected:
  # No cache: None or {'enabled': False}
  # With cache: {'enabled': True, 'keyword_hits': 0, ...}
  # Cache enabled: True
  ```

  **Evidence to Capture:**
  - Terminal output showing cache integration works

  **Commit**: YES (groups with Task 5)
  - Message: `feat(benchmark): integrate Redis caching with --no-cache flag`
  - Files: Both orchestrator files
  - Pre-commit: N/A

---

- [ ] 7. End-to-End Verification

  **What to do**:
  - Run full benchmark twice, compare times
  - Verify cache hit rate on second run
  - Verify --no-cache bypasses caching
  - Document observed speedup

  **Must NOT do**:
  - Don't add new code (verification only)
  - Don't push results to git (just local verification)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Verification task only
  - **Skills**: `[]`
  - **Skills Evaluated but Omitted**:
    - All skills: Not applicable

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None (final task)
  - **Blocked By**: Tasks 5, 6

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md` - Where to document results

  **Acceptance Criteria**:

  ```bash
  # Clear any existing cache
  docker exec plm-redis-cache redis-cli FLUSHDB
  
  # First run (cold cache)
  cd /home/fujin/Code/personal-library-manager
  time python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
  # Note the indexing time (should be ~140-180s)
  
  # Second run (warm cache)
  time python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
  # Note the indexing time (should be <20s)
  
  # Verify cache stats
  docker exec plm-redis-cache redis-cli DBSIZE
  # Expected: ~7000+ keys
  
  # Test --no-cache
  time python poc/modular_retrieval_pipeline/benchmark.py --no-cache --strategy modular-no-llm --questions poc/chunking_benchmark_v2/corpus/needle_questions.json
  # Should take full time again (~140-180s)
  ```

  **Expected Results**:
  - First run: ~140-180s indexing
  - Second run: <20s indexing (>85% faster)
  - Cache hit rate: >95% on second run
  - --no-cache: Full time, no cache usage

  **Evidence to Capture:**
  - Terminal output with timing comparisons
  - DBSIZE output showing cache entries

  **Commit**: NO (verification only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(cache): add Redis cache client with Docker Compose` | docker-compose.yml, cache/__init__.py, cache/redis_client.py | Manual |
| 2, 3, 4 | `feat(cache): add cached component wrappers for keywords, entities, embeddings` | cache/cached_*.py, cache/caching_embedder.py | Manual |
| 5, 6 | `feat(benchmark): integrate Redis caching with --no-cache flag` | benchmark.py, modular_enriched_hybrid*.py | Manual |
| 7 | No commit (verification only) | - | - |

---

## Success Criteria

### Verification Commands
```bash
# Redis running
docker compose -f poc/modular_retrieval_pipeline/docker-compose.yml ps
# Expected: plm-redis-cache running

# Cache populated
docker exec plm-redis-cache redis-cli DBSIZE
# Expected: >7000 keys after full run

# Second run speedup
# Expected: <20s vs ~140s first run (>85% faster)
```

### Final Checklist
- [ ] Docker Compose starts Redis with `docker compose up -d`
- [ ] Redis persists data across restarts
- [ ] First benchmark run populates cache (~7000 entries)
- [ ] Second benchmark run shows >85% speedup
- [ ] --no-cache flag bypasses all caching
- [ ] Graceful fallback when Redis unavailable (no errors)
- [ ] Cache statistics logged during benchmark
