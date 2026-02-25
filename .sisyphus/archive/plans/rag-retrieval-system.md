# RAG Retrieval System: One-to-One POC Port

## TL;DR

> **Quick Summary**: Port the modular retrieval pipeline from POC to production code with enhanced fast extraction (GLiNER + YAKE), BM25S persistence, and SQLite storage in Docker volumes.
>
> **Deliverables**:
> - Enhanced `src/plm/extraction/fast/` with YAKE keyword extraction
> - `src/plm/search/` package with all retrieval components
> - BM25S with disk persistence (10-180x faster than rank-bm25)
> - SQLite storage layer for documents, chunks, and embeddings
> - Docker volumes for BM25 index + SQLite persistence
> - Tests validating identical behavior to POC
>
> **Estimated Effort**: Medium-Large (14 tasks: 1 extraction enhancement + 8 components + storage + orchestrator + docker + tests)
> **Parallel Execution**: YES - 4 waves
> **Critical Path**: Task 0 (YAKE) → Task 1 (types) → Tasks 3-8 (components) → Task 11 (orchestrator) → Task 14 (integration)

---

## Context

### Original Request
One-to-one implementation of embedding and retrieval flow from `poc/modular_retrieval_pipeline/`. Only difference is ingestion: use GLiNER fast system output instead of YAKE/spaCy inline extraction.

### Interview Summary
**Key Discussions**:
- Direct port of all POC components with minimal changes
- Storage: SQLite in Docker volume (not in-memory like POC)
- Test strategy: Tests after implementation

**Research Findings**:
- POC achieved 90% accuracy on needle questions
- RRF parameters: k=60 default, adaptive (k=10, bm25=3.0, sem=0.3) when expanded
- ContentEnricher format: `"keywords | entities\n\noriginal_content"`
- Embedding model: `BAAI/bge-base-en-v1.5`

### Metis Review
**Identified Gaps** (addressed):
- Entity format mismatch: GLiNER outputs `list[ExtractedEntity]`, ContentEnricher expects `dict[str, list[str]]` → **Adapter pattern applied**
- Domain expansions hardcoded for Kubernetes → **Made configurable via file**
- Chunk ID strategy unclear → **Hash of (doc_path, start_char, end_char)**
- Heading hierarchy storage → **Flattened to single table with heading column**
- BM25 persistence → **Store tokens, rebuild index on startup**

### Oracle Review (Critical Issues Addressed)
**Issue 1: Keywords source gap** — YAKE extracts keyphrases, GLiNER extracts only entities
- **Resolution**: Add YAKE to fast extraction system, output both `keywords` (YAKE) and `terms` (GLiNER entities) as separate fields

**Issue 2: BM25 persistence not implemented**
- **Resolution**: Replace rank-bm25 with BM25S library (built-in disk persistence, 10-180x faster, memory-mapped loading)

**Issue 3: Error handling for embedding model**
- **Resolution**: Add try/except in lazy loading with clear error messages

---

## Work Objectives

### Core Objective
Port POC retrieval pipeline to production with SQLite persistence and GLiNER ingestion, achieving identical retrieval accuracy.

### Concrete Deliverables
- `src/plm/search/types.py` — Immutable type definitions
- `src/plm/search/pipeline.py` — Pipeline/Component base classes
- `src/plm/search/components/` — All retrieval components (7 files)
- `src/plm/search/adapters/gliner_adapter.py` — GLiNER → ContentEnricher format
- `src/plm/search/storage/sqlite.py` — SQLite storage layer
- `src/plm/search/retriever.py` — Main orchestrator
- `tests/search/` — Tests for all components
- `docker/docker-compose.search.yml` — Docker compose for SQLite volume

### Definition of Done
- [ ] `python -c "from plm.search import HybridRetriever"` → Exit 0
- [ ] `pytest tests/search/ -v` → All tests pass
- [ ] SQLite tables created with correct schema
- [ ] ContentEnricher output format matches POC exactly

### Must Have
- Exact RRF algorithm: semantic FIRST, BM25 SECOND
- Adaptive weights: k=60 default, k=10/bm25=3.0/sem=0.3 when expanded
- Same embedding model: BAAI/bge-base-en-v1.5
- BM25S library with disk persistence (replaces rank-bm25)
- Same enrichment format: `"keywords | entities\n\ncontent"`
- YAKE keywords in fast extraction output (separate from GLiNER entities)
- Robust error handling for model loading failures

### Must NOT Have (Guardrails)
- NO Redis/caching layer (out of scope)
- NO cross-encoder reranker (not in core pipeline)
- NO LLM query rewriting (POC proved no accuracy gain)
- NO async/batch processing API (scope creep)
- NO vector database (SQLite is the requirement)
- NO multiple embedding models (BGE only, hardcoded)
- NO migrations system (simple schema, no versioning for MVP)
- NO rank-bm25 (replaced by BM25S)

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> Every criterion is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: YES (pytest, tests/ directory)
- **Automated tests**: YES (Tests-after)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY — ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Python imports** | Bash (python -c) | Import module, assert no errors |
| **Component behavior** | Bash (pytest) | Run tests, assert all pass |
| **SQLite schema** | Bash (python -c) | Connect, query sqlite_master |
| **Docker** | Bash (docker compose) | Up, verify volume created |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 0 (Pre-requisite - Fast Extraction Enhancement):
└── Task 0: Add YAKE to fast extraction - no dependencies (can run first)

Wave 1 (Start after Wave 0):
├── Task 1: Types (types.py) - no dependencies
├── Task 2: Pipeline base (pipeline.py) - no dependencies
└── Task 9: SQLite storage (sqlite.py) - no dependencies

Wave 2 (After Wave 1):
├── Task 3: ContentEnricher - depends: 1, 2
├── Task 4: EmbeddingEncoder (with error handling) - depends: 1, 2
├── Task 5: BM25Scorer (using BM25S) - depends: 1, 2
├── Task 6: SimilarityScorer - depends: 1, 2
├── Task 7: QueryExpander - depends: 1, 2
├── Task 8: RRFFuser - depends: 1, 2
└── Task 10: GLiNER Adapter - depends: 0, 1

Wave 3 (After Wave 2):
├── Task 11: Orchestrator (retriever.py) - depends: 3-10
└── Task 12: Docker compose (SQLite + BM25 volumes) - depends: 9

Wave 4 (After Wave 3):
├── Task 13: Accuracy regression test - depends: 0, 11
└── Task 14: Integration tests - depends: 11, 12, 13

Critical Path: Task 0 → Task 1 → Task 5 → Task 11 → Task 14
Parallel Speedup: ~50% faster than sequential
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 0 | None | 10, 13 | None (run first) |
| 1 | None | 3-8, 10 | 2, 9 |
| 2 | None | 3-8 | 1, 9 |
| 9 | None | 11, 12 | 1, 2 |
| 3 | 1, 2 | 11 | 4, 5, 6, 7, 8, 10 |
| 4 | 1, 2 | 11 | 3, 5, 6, 7, 8, 10 |
| 5 | 1, 2 | 11 | 3, 4, 6, 7, 8, 10 |
| 6 | 1, 2 | 11 | 3, 4, 5, 7, 8, 10 |
| 7 | 1, 2 | 11 | 3, 4, 5, 6, 8, 10 |
| 8 | 1, 2 | 11 | 3, 4, 5, 6, 7, 10 |
| 10 | 0, 1 | 11 | 3-8 |
| 11 | 3-10 | 13, 14 | 12 |
| 12 | 9 | 14 | 11 |
| 13 | 0, 11 | 14 | 12 |
| 14 | 11, 12, 13 | None | None (final) |

---

## TODOs

- [ ] 0. Add YAKE keyword extraction to fast extraction system

  **What to do**:
  - Modify `src/plm/extraction/fast/document_processor.py`
  - Add `keywords: list[str]` field to `ChunkResult` dataclass
  - Add YAKE extraction in `process_document()` for each chunk
  - Port YAKE logic from `poc/modular_retrieval_pipeline/components/keyword_extractor.py`
  - Update CLI output JSON to include `keywords` field

  **Must NOT do**:
  - Change existing `terms` field (GLiNER entity texts)
  - Change `entities` field structure
  - Remove any existing functionality

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding one field and calling existing YAKE code
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (must run first)
  - **Parallel Group**: Wave 0 (prerequisite)
  - **Blocks**: Tasks 10, 13
  - **Blocked By**: None

  **References**:
  - `src/plm/extraction/fast/document_processor.py:20-27` — ChunkResult dataclass to modify
  - `src/plm/extraction/fast/document_processor.py:116-138` — Where to add YAKE extraction
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py:37-51` — YAKE extractor initialization
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py:186-208` — YAKE extraction logic

  **Acceptance Criteria**:
  - [ ] ChunkResult has `keywords: list[str]` field
  - [ ] YAKE extraction runs on each chunk
  - [ ] JSON output includes `keywords` array

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: ChunkResult has keywords field
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.extraction.fast.document_processor import ChunkResult
c = ChunkResult(text='test', terms=['t1'], entities=[], keywords=['k1', 'k2'], start_char=0, end_char=4)
assert c.keywords == ['k1', 'k2']
print('PASS: keywords field exists')
"
    Expected Result: Prints "PASS: keywords field exists"
    Evidence: stdout captured

  Scenario: YAKE extraction produces keywords
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from pathlib import Path
import tempfile
from plm.extraction.fast.document_processor import process_document
# Create test file
with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
    f.write('# Kubernetes\n\nThe horizontal pod autoscaler automatically scales the number of pods based on CPU utilization.')
    path = Path(f.name)
result = process_document(path)
# Check first chunk has keywords
chunk = result.headings[0].chunks[0]
assert len(chunk.keywords) > 0, f'No keywords extracted'
assert any('autoscaler' in kw.lower() or 'kubernetes' in kw.lower() for kw in chunk.keywords), f'Expected relevant keywords, got: {chunk.keywords}'
print(f'PASS: Extracted keywords: {chunk.keywords[:3]}')
"
    Expected Result: Prints "PASS: Extracted keywords: [...]"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(extraction): add YAKE keyword extraction to fast system`
  - Files: `src/plm/extraction/fast/document_processor.py`
  - Pre-commit: `python -c "from plm.extraction.fast.document_processor import ChunkResult; assert hasattr(ChunkResult, '__dataclass_fields__') and 'keywords' in ChunkResult.__dataclass_fields__"`

---

- [ ] 1. Port types.py — Immutable type definitions

  **What to do**:
  - Create `src/plm/search/types.py`
  - Copy all dataclass definitions from `poc/modular_retrieval_pipeline/types.py`
  - Preserve exact field names and types
  - Keep frozen=True for immutability

  **Must NOT do**:
  - Change field names or types
  - Add new fields
  - Remove frozen=True

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct copy-paste with minimal adaptation
  - **Skills**: []
    - No special skills needed for direct port

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 2, 9)
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8, 10
  - **Blocked By**: None (can start immediately)

  **References**:
  - `poc/modular_retrieval_pipeline/types.py:1-133` — Source file to port (Query, RewrittenQuery, ExpandedQuery, EmbeddedQuery, ScoredChunk, FusionConfig, PipelineResult)
  - `src/plm/extraction/fast/document_processor.py:20-43` — Reference for production dataclass conventions

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/types.py`
  - [ ] `python -c "from plm.search.types import Query, ScoredChunk, FusionConfig"` → Exit 0

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Import all types successfully
    Tool: Bash (python -c)
    Preconditions: src/plm/search/types.py exists
    Steps:
      1. python -c "from plm.search.types import Query, RewrittenQuery, ExpandedQuery, EmbeddedQuery, ScoredChunk, FusionConfig, PipelineResult; print('PASS')"
    Expected Result: Prints "PASS", exit code 0
    Evidence: stdout captured

  Scenario: Types are frozen (immutable)
    Tool: Bash (python -c)
    Preconditions: types.py exists
    Steps:
      1. python -c "from plm.search.types import Query; q = Query(text='test'); q.text = 'changed'" 2>&1 || echo "PASS: Frozen works"
    Expected Result: Raises FrozenInstanceError or prints "PASS: Frozen works"
    Evidence: stderr/stdout captured
  ```

  **Commit**: YES (group with 2)
  - Message: `feat(search): add type definitions for retrieval pipeline`
  - Files: `src/plm/search/types.py`, `src/plm/search/__init__.py`
  - Pre-commit: `python -c "from plm.search.types import Query"`

---

- [ ] 2. Port pipeline.py — Pipeline/Component base classes

  **What to do**:
  - Create `src/plm/search/pipeline.py`
  - Copy Component protocol, Pipeline class, PipelineError, TypeValidationError
  - Preserve exact behavior

  **Must NOT do**:
  - Add async support
  - Change Pipeline.run() behavior
  - Modify error handling

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct port with no changes
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 9)
  - **Blocks**: Tasks 3, 4, 5, 6, 7, 8
  - **Blocked By**: None

  **References**:
  - `poc/modular_retrieval_pipeline/base.py:1-228` — Source file to port (Component protocol, Pipeline class, PipelineError)

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/pipeline.py`
  - [ ] `python -c "from plm.search.pipeline import Pipeline, Component"` → Exit 0

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Pipeline chains components correctly
    Tool: Bash (python -c)
    Preconditions: pipeline.py exists
    Steps:
      1. python -c "
from plm.search.pipeline import Pipeline, Component
class Upper:
    def process(self, data): return data.upper()
class Exclaim:
    def process(self, data): return data + '!'
p = Pipeline().add(Upper()).add(Exclaim())
result = p.run('hello')
assert result == 'HELLO!', f'Got: {result}'
print('PASS')
"
    Expected Result: Prints "PASS"
    Evidence: stdout captured
  ```

  **Commit**: YES (group with 1)
  - Message: `feat(search): add type definitions for retrieval pipeline`
  - Files: `src/plm/search/pipeline.py`
  - Pre-commit: `python -c "from plm.search.pipeline import Pipeline"`

---

- [ ] 3. Port ContentEnricher component

  **What to do**:
  - Create `src/plm/search/components/enricher.py`
  - Port from `poc/modular_retrieval_pipeline/components/content_enricher.py`
  - Preserve exact output format: `"keywords | entities\n\ncontent"`
  - Keep limit: first 7 keywords, first 2 per entity type, max 5 total entities

  **Must NOT do**:
  - Change output format
  - Change limits (7 keywords, 5 entities)
  - Add GLiNER-specific handling (that's the adapter's job)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct port
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 4, 5, 6, 7, 8, 10)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/content_enricher.py:1-182` — Source file to port
  - `poc/modular_retrieval_pipeline/components/content_enricher.py:160-171` — Entity formatting logic (first 2 per type, max 5 total, NO type labels)

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/enricher.py`
  - [ ] Output format test passes

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: ContentEnricher produces correct format
    Tool: Bash (python -c)
    Preconditions: enricher.py exists
    Steps:
      1. python -c "
from plm.search.components.enricher import ContentEnricher
enricher = ContentEnricher()
result = enricher.process({
    'content': 'Test content here',
    'keywords': ['key1', 'key2', 'key3'],
    'entities': {'library': ['React', 'Vue'], 'framework': ['Next.js']}
})
expected = 'key1, key2, key3 | React, Vue, Next.js\n\nTest content here'
assert result == expected, f'Expected:\n{expected}\nGot:\n{result}'
print('PASS: Format correct')
"
    Expected Result: Prints "PASS: Format correct"
    Evidence: stdout captured

  Scenario: Empty keywords/entities returns content unchanged
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.components.enricher import ContentEnricher
enricher = ContentEnricher()
result = enricher.process({'content': 'Just content', 'keywords': [], 'entities': {}})
assert result == 'Just content', f'Got: {result}'
print('PASS: Empty case works')
"
    Expected Result: Prints "PASS: Empty case works"
    Evidence: stdout captured
  ```

  **Commit**: NO (group with other components)

---

- [ ] 4. Port EmbeddingEncoder component (with error handling)

  **What to do**:
  - Create `src/plm/search/components/embedder.py`
  - Port from `poc/modular_retrieval_pipeline/components/embedding_encoder.py`
  - Keep lazy loading pattern (model loaded on first use)
  - Use model: `BAAI/bge-base-en-v1.5`
  - **Add robust error handling** for model loading failures:
    - Network errors (model download fails)
    - OOM errors (insufficient RAM)
    - Corrupted cache
  - Wrap `_load_model()` in try/except with clear error message

  **Must NOT do**:
  - Change model name
  - Add model selection
  - Remove lazy loading
  - Silently fail (must raise with clear message)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct port with error handling addition
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py:1-227` — Source file to port
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py:80-103` — Lazy loading pattern to wrap with error handling

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/embedder.py`
  - [ ] Embedding dimension is 768
  - [ ] Clear error message on model loading failure

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: EmbeddingEncoder produces 768-dim vectors
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.components.embedder import EmbeddingEncoder
encoder = EmbeddingEncoder()
result = encoder.process('test text')
assert len(result['embedding']) == 768, f'Got dim: {len(result[\"embedding\"])}'
assert result['model'] == 'BAAI/bge-base-en-v1.5'
print('PASS: 768-dim BGE embedding')
"
    Expected Result: Prints "PASS: 768-dim BGE embedding"
    Evidence: stdout captured

  Scenario: Clear error on invalid model
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.components.embedder import EmbeddingEncoder
encoder = EmbeddingEncoder(model='nonexistent/model-xyz-404')
try:
    encoder.process('test')
    print('FAIL: Should have raised')
except Exception as e:
    assert 'model' in str(e).lower() or 'load' in str(e).lower(), f'Error not descriptive: {e}'
    print('PASS: Clear error message')
" 2>&1
    Expected Result: Prints "PASS: Clear error message"
    Evidence: stdout/stderr captured
  ```

  **Commit**: NO (group with other components)

---

- [ ] 5. Create BM25Scorer component (using BM25S with persistence)

  **What to do**:
  - Create `src/plm/search/components/bm25.py`
  - Use **bm25s** library instead of rank_bm25 (10-180x faster, built-in persistence)
  - Implement BM25Index class that wraps bm25s.BM25:
    - `index(chunks: list[str])` — Build index
    - `save(path: str)` — Persist to disk
    - `load(path: str)` — Load from disk (memory-mapped)
    - `search(query: str, k: int)` — Return ranked results
  - Use bm25s tokenizer or simple `content.lower().split()` for compatibility

  **Must NOT do**:
  - Use rank_bm25 (deprecated, no persistence)
  - Add stemming/lemmatization (keep simple tokenization for POC parity)
  - Change scoring algorithm

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: New implementation using bm25s library
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/bm25_scorer.py:1-166` — Original POC implementation (for behavior reference)
  - https://bm25s.github.io/ — BM25S documentation
  - https://github.com/xhluca/bm25s — BM25S source code

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/bm25.py`
  - [ ] Uses bm25s library (not rank_bm25)
  - [ ] save()/load() persistence works
  - [ ] Memory-mapped loading works

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: BM25Index ranks matching content higher
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.components.bm25 import BM25Index
index = BM25Index()
index.index(['kubernetes pod definition', 'docker container', 'kubernetes deployment'])
results = index.search('kubernetes pod', k=3)
# kubernetes pod definition should rank first (exact match)
assert results[0]['content'] == 'kubernetes pod definition', f'Top result: {results[0]}'
print('PASS: BM25 ranking works')
"
    Expected Result: Prints "PASS: BM25 ranking works"
    Evidence: stdout captured

  Scenario: BM25 index persistence works
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
import os
from plm.search.components.bm25 import BM25Index
# Create and save
index = BM25Index()
index.index(['kubernetes pod definition', 'docker container'])
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'bm25_index')
    index.save(path)
    assert os.path.exists(path), 'Index not saved'
    # Load and verify
    loaded = BM25Index.load(path)
    results = loaded.search('kubernetes', k=1)
    assert len(results) > 0, 'No results from loaded index'
    print('PASS: Persistence roundtrip works')
"
    Expected Result: Prints "PASS: Persistence roundtrip works"
    Evidence: stdout captured

  Scenario: Memory-mapped loading is efficient
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
import os
import time
from plm.search.components.bm25 import BM25Index
# Create larger index
chunks = [f'document number {i} about kubernetes and pods' for i in range(1000)]
index = BM25Index()
index.index(chunks)
with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'bm25_index')
    index.save(path)
    # Time memory-mapped load
    start = time.time()
    loaded = BM25Index.load(path, mmap=True)
    load_time = time.time() - start
    assert load_time < 2.0, f'Load too slow: {load_time}s'
    print(f'PASS: Memory-mapped load in {load_time:.2f}s')
"
    Expected Result: Prints "PASS: Memory-mapped load in X.XXs" (under 2s)
    Evidence: stdout captured
  ```

  **Commit**: NO (group with other components)

---

- [ ] 6. Port SimilarityScorer component

  **What to do**:
  - Create `src/plm/search/components/semantic.py`
  - Port from `poc/modular_retrieval_pipeline/components/similarity_scorer.py`
  - Use cosine similarity (normalized dot product)

  **Must NOT do**:
  - Change similarity metric
  - Add other distance metrics

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct port
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/similarity_scorer.py:1-208` — Source file to port
  - `poc/modular_retrieval_pipeline/components/similarity_scorer.py:166-183` — Normalization and dot product logic

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/semantic.py`
  - [ ] Identical vectors produce score ~1.0

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: SimilarityScorer ranks identical vectors highest
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.components.semantic import SimilarityScorer
scorer = SimilarityScorer()
results = scorer.process({
    'query_embedding': (1.0, 0.0, 0.0),
    'chunk_embeddings': [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.5, 0.5, 0.0)]
})
assert abs(results[0].score - 1.0) < 0.01, f'Expected ~1.0, got {results[0].score}'
assert results[0].source == 'semantic'
print('PASS: Similarity scoring works')
"
    Expected Result: Prints "PASS: Similarity scoring works"
    Evidence: stdout captured
  ```

  **Commit**: NO (group with other components)

---

- [ ] 7. Port QueryExpander component

  **What to do**:
  - Create `src/plm/search/components/expander.py`
  - Port from `poc/modular_retrieval_pipeline/components/query_expander.py`
  - Make DOMAIN_EXPANSIONS configurable (load from JSON file if exists, else use defaults)
  - Default expansions remain Kubernetes-focused (as in POC)

  **Must NOT do**:
  - Remove expansion functionality
  - Change expansion algorithm

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mostly direct port with minor config addition
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/query_expander.py:1-184` — Source file to port
  - `poc/modular_retrieval_pipeline/components/query_expander.py:30-50` — DOMAIN_EXPANSIONS dictionary

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/expander.py`
  - [ ] Default expansions loaded
  - [ ] Configurable via JSON file

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: QueryExpander expands known terms
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.types import Query, RewrittenQuery
from plm.search.components.expander import QueryExpander
expander = QueryExpander()
query = Query(text='token authentication')
rewritten = RewrittenQuery(original=query, rewritten='token auth', model='none')
result = expander.process(rewritten)
assert 'JWT' in result.expanded, f'Expected JWT in: {result.expanded}'
assert 'token' in result.expansions
print('PASS: Expansion works')
"
    Expected Result: Prints "PASS: Expansion works"
    Evidence: stdout captured

  Scenario: No expansion when no match
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.types import Query, RewrittenQuery
from plm.search.components.expander import QueryExpander
expander = QueryExpander()
query = Query(text='random query')
rewritten = RewrittenQuery(original=query, rewritten='random query', model='none')
result = expander.process(rewritten)
assert result.expanded == 'random query', f'Got: {result.expanded}'
assert result.expansions == ()
print('PASS: No false expansion')
"
    Expected Result: Prints "PASS: No false expansion"
    Evidence: stdout captured
  ```

  **Commit**: NO (group with other components)

---

- [ ] 8. Port RRFFuser component

  **What to do**:
  - Create `src/plm/search/components/rrf.py`
  - Port from `poc/modular_retrieval_pipeline/components/rrf_fuser.py`
  - Preserve exact RRF formula: `score(d) = Σ (weight_i / (k + rank_i(d)))`
  - Default k=60, equal weights

  **Must NOT do**:
  - Change RRF formula
  - Change default k value
  - Change weight extraction logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Direct port
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/rrf_fuser.py:1-208` — Source file to port
  - `poc/modular_retrieval_pipeline/components/rrf_fuser.py:150-152` — RRF score calculation: `weight / (k + rank)`

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/components/rrf.py`
  - [ ] RRF scores calculated correctly

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: RRFFuser combines scores correctly
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.types import ScoredChunk, FusionConfig
from plm.search.components.rrf import RRFFuser
config = FusionConfig(k=60, bm25_weight=1.0, semantic_weight=1.0)
fuser = RRFFuser(config)
bm25 = [ScoredChunk(chunk_id='a', content='text', score=10.0, source='bm25', rank=1)]
semantic = [ScoredChunk(chunk_id='a', content='text', score=0.9, source='semantic', rank=1)]
result = fuser.process([bm25, semantic])
# RRF score for chunk 'a' = 1.0/(60+1) + 1.0/(60+1) = 2/61 ≈ 0.0328
assert abs(result[0].score - (2/61)) < 0.001, f'Expected ~0.0328, got {result[0].score}'
assert result[0].source == 'rrf'
print('PASS: RRF fusion works')
"
    Expected Result: Prints "PASS: RRF fusion works"
    Evidence: stdout captured
  ```

  **Commit**: YES (group all components)
  - Message: `feat(search): add retrieval components (enricher, embedder, bm25, semantic, expander, rrf)`
  - Files: `src/plm/search/components/*.py`, `src/plm/search/components/__init__.py`
  - Pre-commit: `python -c "from plm.search.components import ContentEnricher, BM25Scorer, RRFFuser"`

---

- [ ] 9. Create SQLite storage layer

  **What to do**:
  - Create `src/plm/search/storage/sqlite.py`
  - Define SQLiteStorage class with schema creation
  - Tables: documents, chunks (with embeddings as BLOB)
  - Methods: create_tables(), add_document(), add_chunk(), get_all_chunks(), get_chunk_by_id()
  - Store embeddings as `np.ndarray.tobytes()`, load with `np.frombuffer()`

  **Must NOT do**:
  - Add FTS5 or other extensions
  - Add async support
  - Add connection pooling (simple single connection)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: New code, but straightforward SQLite operations
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Tasks 1, 2)
  - **Blocks**: Tasks 11, 12
  - **Blocked By**: None

  **References**:
  - Draft schema in `.sisyphus/drafts/rag-storage-search.md` — Proposed schema
  - `src/plm/extraction/fast/document_processor.py:20-43` — DocumentResult structure for ingestion

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/storage/sqlite.py`
  - [ ] Tables created with correct schema
  - [ ] CRUD operations work

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: SQLite creates tables correctly
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
import sqlite3
from plm.search.storage.sqlite import SQLiteStorage
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    storage = SQLiteStorage(f.name)
    storage.create_tables()
    conn = sqlite3.connect(f.name)
    tables = [t[0] for t in conn.execute(\"SELECT name FROM sqlite_master WHERE type='table'\").fetchall()]
    assert 'documents' in tables, f'Missing documents table. Got: {tables}'
    assert 'chunks' in tables, f'Missing chunks table. Got: {tables}'
    print('PASS: Tables created')
"
    Expected Result: Prints "PASS: Tables created"
    Evidence: stdout captured

  Scenario: Store and retrieve embedding
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
import numpy as np
from plm.search.storage.sqlite import SQLiteStorage
with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
    storage = SQLiteStorage(f.name)
    storage.create_tables()
    storage.add_document('doc1', 'test.md')
    embedding = np.random.rand(768).astype(np.float32)
    storage.add_chunk('chunk1', 'doc1', 'Test content', embedding=embedding, enriched_content='enriched')
    retrieved = storage.get_chunk_by_id('chunk1')
    assert retrieved is not None
    assert np.allclose(retrieved['embedding'], embedding)
    print('PASS: Embedding roundtrip works')
"
    Expected Result: Prints "PASS: Embedding roundtrip works"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(search): add SQLite storage layer for chunks and embeddings`
  - Files: `src/plm/search/storage/sqlite.py`, `src/plm/search/storage/__init__.py`
  - Pre-commit: `python -c "from plm.search.storage import SQLiteStorage"`

---

- [ ] 10. Create GLiNER adapter (with YAKE keywords support)

  **What to do**:
  - Create `src/plm/search/adapters/gliner_adapter.py`
  - Function: `gliner_to_enricher_format(entities: list[ExtractedEntity]) -> dict[str, list[str]]`
  - Groups entities by label, returns dict format expected by ContentEnricher
  - Function: `document_result_to_chunks(doc: DocumentResult) -> list[dict]`
  - Flattens DocumentResult → list of chunk dicts ready for storage/enrichment
  - **Include both `keywords` (from YAKE) and `entities` (from GLiNER) in output**:
    ```python
    {
        'content': chunk.text,
        'keywords': chunk.keywords,  # From YAKE (Task 0)
        'entities': gliner_to_enricher_format(chunk.entities),  # From GLiNER
        'heading': section.heading,
        'start_char': chunk.start_char,
        'end_char': chunk.end_char,
    }
    ```

  **Must NOT do**:
  - Modify ContentEnricher
  - Conflate keywords and entities (keep separate)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple transformation functions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3-8)
  - **Blocks**: Task 11
  - **Blocked By**: Tasks 0, 1

  **References**:
  - `src/plm/extraction/fast/gliner.py:18-24` — ExtractedEntity dataclass definition
  - `src/plm/extraction/fast/document_processor.py:20-27` — ChunkResult (now with keywords field from Task 0)
  - Metis analysis — Entity format transformation logic

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/adapters/gliner_adapter.py`
  - [ ] Entity transformation works
  - [ ] Output includes both `keywords` and `entities`

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: GLiNER entities transformed correctly
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.adapters.gliner_adapter import gliner_to_enricher_format
from plm.extraction.fast.gliner import ExtractedEntity
entities = [
    ExtractedEntity(text='React', label='library', score=0.9, start=0, end=5),
    ExtractedEntity(text='Vue', label='library', score=0.8, start=10, end=13),
    ExtractedEntity(text='Next.js', label='framework', score=0.85, start=20, end=27),
]
result = gliner_to_enricher_format(entities)
assert result == {'library': ['React', 'Vue'], 'framework': ['Next.js']}, f'Got: {result}'
print('PASS: Entity transformation correct')
"
    Expected Result: Prints "PASS: Entity transformation correct"
    Evidence: stdout captured

  Scenario: DocumentResult flattened with keywords and entities
    Tool: Bash (python -c)
    Steps:
      1. python -c "
from plm.search.adapters.gliner_adapter import document_result_to_chunks
from plm.extraction.fast.document_processor import DocumentResult, HeadingSection, ChunkResult
from plm.extraction.fast.gliner import ExtractedEntity
entity = ExtractedEntity(text='Kubernetes', label='technology', score=0.9, start=0, end=10)
chunk = ChunkResult(text='Test content', terms=['Kubernetes'], entities=[entity], keywords=['kubernetes', 'autoscaler'], start_char=0, end_char=12)
section = HeadingSection(heading='## Test', level=2, chunks=[chunk])
doc = DocumentResult(source_file='test.md', headings=[section], avg_confidence=0.9, total_entities=1, is_low_confidence=False)
result = document_result_to_chunks(doc)
assert len(result) == 1
assert result[0]['content'] == 'Test content'
assert result[0]['keywords'] == ['kubernetes', 'autoscaler'], f'Keywords: {result[0].get(\"keywords\")}'
assert result[0]['entities'] == {'technology': ['Kubernetes']}, f'Entities: {result[0].get(\"entities\")}'
print('PASS: Document flattening with keywords and entities works')
"
    Expected Result: Prints "PASS: Document flattening with keywords and entities works"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(search): add GLiNER adapter for ingestion`
  - Files: `src/plm/search/adapters/gliner_adapter.py`, `src/plm/search/adapters/__init__.py`
  - Pre-commit: `python -c "from plm.search.adapters import gliner_to_enricher_format"`

---

- [ ] 11. Create main orchestrator (HybridRetriever with BM25S persistence)

  **What to do**:
  - Create `src/plm/search/retriever.py`
  - Port from `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py`
  - Adapt for SQLite storage instead of in-memory
  - Adapt for GLiNER ingestion using adapter
  - **Use BM25Index (bm25s) with disk persistence**:
    - On `__init__`: Load BM25 index from disk if exists
    - On `index()`: Build and save BM25 index to disk
    - Constructor: `HybridRetriever(db_path: str, bm25_index_path: str)`
  - Preserve EXACT retrieval logic:
    - RRF order: semantic FIRST, BM25 SECOND (lines 404-424)
    - Adaptive weights: k=60 default, k=10/bm25=3.0/sem=0.3 when expanded
    - dict.get(idx, 0) accumulation pattern

  **Must NOT do**:
  - Change RRF algorithm
  - Change adaptive weight logic
  - Add caching beyond BM25 persistence
  - Add async support
  - Use rank_bm25 (must use bm25s)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Complex orchestration, critical to preserve exact behavior
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (after all components)
  - **Blocks**: Tasks 13, 14
  - **Blocked By**: Tasks 3, 4, 5, 6, 7, 8, 9, 10

  **References**:
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:1-541` — Main source file to port
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:278-311` — Query expansion with adaptive weights
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:399-424` — RRF fusion (semantic FIRST, BM25 SECOND)
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py:346-355` — Adaptive weights selection
  - Task 5 (BM25Index) — BM25S wrapper with persistence

  **Acceptance Criteria**:
  - [ ] File created: `src/plm/search/retriever.py`
  - [ ] `from plm.search import HybridRetriever` works
  - [ ] Retrieval returns ranked chunks
  - [ ] BM25 index persists to disk
  - [ ] Restart loads existing BM25 index

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: HybridRetriever can be instantiated
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
from plm.search.retriever import HybridRetriever
with tempfile.TemporaryDirectory() as tmpdir:
    retriever = HybridRetriever(f'{tmpdir}/test.db', f'{tmpdir}/bm25_index')
    print('PASS: Instantiation works')
"
    Expected Result: Prints "PASS: Instantiation works"
    Evidence: stdout captured

  Scenario: Ingest and retrieve workflow
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
from plm.search.retriever import HybridRetriever
with tempfile.TemporaryDirectory() as tmpdir:
    retriever = HybridRetriever(f'{tmpdir}/test.db', f'{tmpdir}/bm25_index')
    # Ingest test document
    retriever.ingest_document('doc1', 'test.md', [
        {'content': 'Kubernetes horizontal pod autoscaler scales pods', 'heading': '## HPA', 'keywords': ['kubernetes', 'autoscaler'], 'entities': {}}
    ])
    # Retrieve
    results = retriever.retrieve('kubernetes autoscaling', k=1)
    assert len(results) > 0, 'No results returned'
    print('PASS: Ingest and retrieve works')
"
    Expected Result: Prints "PASS: Ingest and retrieve works"
    Evidence: stdout captured

  Scenario: BM25 index persists across restarts
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
import os
from plm.search.retriever import HybridRetriever
with tempfile.TemporaryDirectory() as tmpdir:
    db_path = f'{tmpdir}/test.db'
    bm25_path = f'{tmpdir}/bm25_index'
    # First session: ingest
    retriever1 = HybridRetriever(db_path, bm25_path)
    retriever1.ingest_document('doc1', 'test.md', [
        {'content': 'Kubernetes pod definition', 'heading': '## K8s', 'keywords': ['kubernetes'], 'entities': {}}
    ])
    del retriever1
    # Second session: retrieve without re-indexing
    retriever2 = HybridRetriever(db_path, bm25_path)
    results = retriever2.retrieve('kubernetes', k=1)
    assert len(results) > 0, 'BM25 index not persisted'
    print('PASS: BM25 persistence across restarts')
"
    Expected Result: Prints "PASS: BM25 persistence across restarts"
    Evidence: stdout captured

  Scenario: Adaptive weights triggered on expansion
    Tool: Bash (python -c)
    Steps:
      1. python -c "
import tempfile
from plm.search.retriever import HybridRetriever
with tempfile.TemporaryDirectory() as tmpdir:
    retriever = HybridRetriever(f'{tmpdir}/test.db', f'{tmpdir}/bm25_index')
    retriever.ingest_document('doc1', 'test.md', [
        {'content': 'JWT token authentication with iat claims', 'heading': '## Auth', 'keywords': ['jwt', 'token'], 'entities': {}}
    ])
    # Query with expansion trigger 'jwt'
    results = retriever.retrieve('jwt authentication', k=1)
    assert len(results) > 0, 'No results'
    # Verify adaptive weights were used (k=10, bm25=3.0, sem=0.3)
    # This is implicit - just verify it doesn't crash
    print('PASS: Adaptive weights work')
"
    Expected Result: Prints "PASS: Adaptive weights work"
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `feat(search): add HybridRetriever orchestrator with BM25S persistence`
  - Files: `src/plm/search/retriever.py`
  - Pre-commit: `python -c "from plm.search import HybridRetriever"`

---

- [ ] 12. Create Docker compose for SQLite + BM25 volumes

  **What to do**:
  - Create `docker/docker-compose.search.yml`
  - Define TWO named volumes:
    - `plm-sqlite-data`: For SQLite database file
    - `plm-bm25-index`: For BM25S index directory
  - Create a simple Python service that mounts both volumes
  - Example volume mounts:
    ```yaml
    volumes:
      - plm-sqlite-data:/data/sqlite
      - plm-bm25-index:/data/bm25
    ```

  **Must NOT do**:
  - Add Redis
  - Add complex networking
  - Add multiple services

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple docker compose file
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 11)
  - **Blocks**: Task 14
  - **Blocked By**: Task 9

  **References**:
  - `poc/modular_retrieval_pipeline/docker-compose.yml` — Reference for compose structure

  **Acceptance Criteria**:
  - [ ] File created: `docker/docker-compose.search.yml`
  - [ ] SQLite volume defined
  - [ ] BM25 index volume defined

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Docker compose file is valid
    Tool: Bash (docker compose)
    Steps:
      1. docker compose -f docker/docker-compose.search.yml config
    Expected Result: Exit code 0, valid YAML output
    Evidence: stdout captured

  Scenario: Both volumes defined
    Tool: Bash (grep)
    Steps:
      1. grep -E 'plm-sqlite-data|plm-bm25-index' docker/docker-compose.search.yml | wc -l
    Expected Result: At least 4 lines (2 definitions + 2 usages)
    Evidence: stdout captured
  ```

  **Commit**: YES
  - Message: `infra(docker): add compose file for search with SQLite and BM25 volumes`
  - Files: `docker/docker-compose.search.yml`
  - Pre-commit: `docker compose -f docker/docker-compose.search.yml config`

---

- [ ] 13. Create accuracy regression test

  **What to do**:
  - Create `tests/search/test_accuracy.py`
  - Test against POC's 90% accuracy baseline using needle questions
  - Load at least 3 questions from `poc/modular_retrieval_pipeline/corpus/`
  - Verify new system achieves ≥85% accuracy (allow 5% regression tolerance)
  - This validates that YAKE + GLiNER + BM25S produces comparable results to POC

  **Must NOT do**:
  - Run full 20-question benchmark (too slow for CI)
  - Require exact 90% match (some variance expected)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard pytest test with accuracy check
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 4 (with Task 14)
  - **Blocks**: None
  - **Blocked By**: Tasks 0, 11

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark.py` — POC benchmark logic
  - `poc/modular_retrieval_pipeline/corpus/` — Test corpus
  - `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md` — 90% accuracy baseline

  **Acceptance Criteria**:
  - [ ] File created: `tests/search/test_accuracy.py`
  - [ ] Test passes with ≥85% accuracy on sample queries

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Accuracy test passes
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/search/test_accuracy.py -v --tb=short
    Expected Result: All tests pass, accuracy ≥85%
    Evidence: pytest output captured
  ```

  **Commit**: NO (group with Task 14)

---

- [ ] 14. Write integration tests

  **What to do**:
  - Create `tests/search/test_pipeline.py`
  - Create `tests/search/test_storage.py`
  - Create `tests/search/test_components.py`
  - Test full ingest → retrieve workflow
  - Verify RRF behavior matches POC
  - Test BM25S persistence

  **Must NOT do**:
  - Add performance tests (separate concern)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Standard pytest tests
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (final)
  - **Blocks**: None
  - **Blocked By**: Tasks 11, 12, 13

  **References**:
  - `tests/extraction/test_gliner_chunking.py` — Test file structure example
  - All component acceptance criteria — Test cases to formalize

  **Acceptance Criteria**:
  - [ ] Files created in `tests/search/`
  - [ ] `pytest tests/search/ -v` → All tests pass

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: All search tests pass
    Tool: Bash (pytest)
    Steps:
      1. pytest tests/search/ -v --tb=short
    Expected Result: All tests pass, exit code 0
    Evidence: pytest output captured
  ```

  **Commit**: YES
  - Message: `test(search): add tests for retrieval pipeline with accuracy validation`
  - Files: `tests/search/*.py`, `tests/search/__init__.py`
  - Pre-commit: `pytest tests/search/ -v`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 0 | `feat(extraction): add YAKE keyword extraction to fast system` | document_processor.py | Import test |
| 1, 2 | `feat(search): add type definitions for retrieval pipeline` | types.py, pipeline.py, __init__.py | Import test |
| 3-8 | `feat(search): add retrieval components (with BM25S)` | components/*.py | Import test |
| 9 | `feat(search): add SQLite storage layer` | storage/*.py | Import test |
| 10 | `feat(search): add GLiNER adapter` | adapters/*.py | Import test |
| 11 | `feat(search): add HybridRetriever orchestrator with persistence` | retriever.py | Import test |
| 12 | `infra(docker): add compose file for search with SQLite and BM25 volumes` | docker-compose.search.yml | Config validate |
| 13, 14 | `test(search): add tests for retrieval pipeline with accuracy validation` | tests/search/*.py | pytest |

---

## Success Criteria

### Verification Commands
```bash
# 1. YAKE keywords in fast extraction
python -c "from plm.extraction.fast.document_processor import ChunkResult; assert 'keywords' in ChunkResult.__dataclass_fields__"  # Exit 0

# 2. All imports work
python -c "from plm.search import HybridRetriever, SQLiteStorage"  # Exit 0

# 3. BM25S persistence works
python -c "from plm.search.components.bm25 import BM25Index; i = BM25Index(); i.index(['test'])"  # Exit 0

# 4. All tests pass (including accuracy regression)
pytest tests/search/ -v  # All pass, accuracy ≥85%

# 5. Docker compose valid
docker compose -f docker/docker-compose.search.yml config  # Valid output
```

### Final Checklist
- [ ] All "Must Have" present (RRF algorithm, adaptive weights, BGE model, BM25S persistence)
- [ ] All "Must NOT Have" absent (no Redis, no reranker, no LLM rewriting, no rank-bm25)
- [ ] All tests pass
- [ ] ContentEnricher format matches POC exactly
- [ ] YAKE keywords extracted in fast system
- [ ] BM25 index persists across restarts
- [ ] Accuracy ≥85% on needle questions
