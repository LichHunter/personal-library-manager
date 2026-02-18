# Pipeline Service for Manual Testing

## TL;DR

> **Quick Summary**: Port missing QueryRewriter from POC to production (using shared LLM module), build HTTP service with auto-ingestion, and enable direct comparison with POC results. Haiku query enrichment is OPTIONAL (can be disabled).
> 
> **Deliverables**:
> - `src/plm/search/components/query_rewriter.py` - QueryRewriter using shared LLM module
> - Updated HybridRetriever with optional query rewriting
> - FastAPI service wrapping HybridRetriever (`src/plm/search/service/`)
> - Directory watcher for auto-ingestion
> - CLI tool (`plm-query`) with `--no-rewrite` flag
> - POC comparison test script
> - Docker Compose setup
> 
> **Estimated Effort**: Medium-Large
> **Parallel Execution**: YES - 3 waves
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 7 (comparison test)

---

## Context

### Original Request
Replicate the modular POC test pattern using production code. Fast system ingests data, outputs to pipeline, which builds indexes. Then manually test by sending requests directly to production code - NO middleware. Compare results with POC.

### Interview Summary
**Key Discussions**:
- Pipeline must run as standalone process (not just Python class)
- Communication via HTTP API (FastAPI)
- Data flow: Fast system writes to directory → Pipeline watches and auto-ingests
- Index persistence in Docker volume (survives restarts)
- **CRITICAL**: QueryRewriter is in POC but MISSING from production - must port it
- **CRITICAL**: Must use shared LLM module (`src/plm/shared/llm/`)
- **CRITICAL**: Haiku query enrichment must be OPTIONAL (can be disabled)
- **CRITICAL**: Direct comparison with POC results (same CONTENT, not IDs - IDs use UUIDs)

### Metis Review
**Identified Gaps** (addressed):
- QueryRewriter NOT ported to production - MUST FIX
- POC uses local LLM provider, production must use shared module
- No flag to disable Haiku enrichment - MUST ADD
- No comparison test infrastructure - MUST ADD

---

## Work Objectives

### Core Objective
Port QueryRewriter from POC to production using shared LLM module, enable optional Haiku enrichment, and verify production matches POC output.

### Concrete Deliverables
- `src/plm/search/components/query_rewriter.py` - QueryRewriter using shared LLM
- Updated `HybridRetriever.retrieve()` with `use_rewrite` parameter
- FastAPI service with `/query` endpoint (supports `use_rewrite` flag)
- CLI with `--no-rewrite` flag
- `scripts/compare_with_poc.py` - Direct comparison test
- Docker Compose for persistent deployment

### Definition of Done
- [ ] `QueryRewriter` uses `src/plm/shared/llm/base.py:call_llm()`
- [ ] `HybridRetriever.retrieve(query, use_rewrite=True/False)` works
- [ ] CLI `plm-query --no-rewrite` skips Haiku
- [ ] Comparison script shows same CONTENT as POC (≥90% match)
- [ ] Service runs in Docker with persistent index
- [ ] Directory watcher handles failures gracefully (failed/ directory)

### Must Have
- QueryRewriter using shared LLM module
- Optional query rewriting (flag to enable/disable)
- HTTP query endpoint with rewrite toggle
- CLI with `--no-rewrite` flag
- Direct comparison with POC results
- Directory watching for auto-ingestion

### Must NOT Have (Guardrails)
- No custom LLM provider (must use shared module)
- No changes to fast system output format
- No mocking of HybridRetriever internals during comparison
- No authentication/authorization

---

## Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**

### Test Decision
- **Automated tests**: NO - Manual testing infrastructure
- **Agent-Executed QA**: MANDATORY for all tasks

### Comparison Strategy
**CRITICAL FIX from Oracle review:**
- POC uses strategy-generated IDs (`doc_id:heading:idx`)
- Production uses UUID-based IDs (`doc_id_i_uuid8`) - NEVER deterministic
- **Compare CONTENT TEXT, not chunk IDs**

**Approach:**
- Both pipelines use `use_rewrite=False` for determinism (no LLM calls)
- Run same queries on both pipelines
- Compare: `{r['content'] for r in results}` as sets
- Same content = MATCH, different = DIFF with explanation
- Target: ≥90% (18/20) content matches

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Port QueryRewriter to production (shared LLM)
└── Task 4: Add dependencies (fastapi, etc.)

Wave 2 (After Wave 1):
├── Task 2: Update HybridRetriever with optional rewrite
├── Task 3: Create FastAPI service + watcher
└── Task 5: Create CLI with --no-rewrite

Wave 3 (After Wave 2):
├── Task 6: Create Docker setup
└── Task 7: POC comparison test
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2 | 4 |
| 2 | 1 | 3, 5, 7 | 4 |
| 3 | 2 | 7 | 5, 6 |
| 4 | None | 3 | 1 |
| 5 | 2 | 7 | 3, 6 |
| 6 | 4 | 7 | 3, 5 |
| 7 | 3, 5, 6 | None | None (final) |

---

## TODOs

- [ ] 1. Port QueryRewriter to production using shared LLM module

  **What to do**:
  - Create `src/plm/search/components/query_rewriter.py`
  - Port logic from `poc/modular_retrieval_pipeline/components/query_rewriter.py`
  - **CRITICAL**: Use `from plm.shared.llm import call_llm` (NOT local utils)
  - Keep same prompt template (QUERY_REWRITE_PROMPT)
  - Keep same timeout handling (default 5s)
  - Return `RewrittenQuery` type (already exists in `src/plm/search/types.py`)

  **Must NOT do**:
  - Don't copy POC's local `utils/llm_provider.py`
  - Don't modify the shared LLM module
  - Don't change the prompt template

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward port, change only the import
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 4)
  - **Blocks**: Task 2
  - **Blocked By**: None

  **References**:
  - `poc/modular_retrieval_pipeline/components/query_rewriter.py` - Source to port (lines 1-168)
  - `src/plm/shared/llm/base.py:call_llm` - Target LLM function (line 99-129)
  - `src/plm/search/types.py:Query,RewrittenQuery` - Existing types to use
  - `src/plm/search/components/` - Where to create new file

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: QueryRewriter imports successfully
    Tool: Bash
    Preconditions: File created
    Steps:
      1. uv run python -c "from plm.search.components.query_rewriter import QueryRewriter; print('OK')"
      2. Assert: Output contains "OK"
    Expected Result: No import errors
    Evidence: Output captured

  Scenario: QueryRewriter uses shared LLM module
    Tool: Bash
    Preconditions: File created
    Steps:
      1. grep -n "from plm.shared.llm" src/plm/search/components/query_rewriter.py
      2. Assert: Output shows import line
      3. grep -n "utils.llm_provider" src/plm/search/components/query_rewriter.py
      4. Assert: No output (NOT using POC's local provider)
    Expected Result: Uses shared module, not POC's local one
    Evidence: Grep output captured

  Scenario: QueryRewriter processes query (with mock)
    Tool: Bash
    Preconditions: QueryRewriter created, ANTHROPIC_API_KEY not set (will fall back)
    Steps:
      1. uv run python -c "
         from plm.search.components.query_rewriter import QueryRewriter
         from plm.search.types import Query
         qr = QueryRewriter(timeout=1.0)
         result = qr.process(Query(text='test query'))
         print(f'original: {result.original.text}')
         print(f'rewritten: {result.rewritten}')
         "
      2. Assert: Output shows original and rewritten fields
    Expected Result: QueryRewriter returns result (even if fallback to original)
    Evidence: Output captured
  ```

  **Commit**: YES
  - Message: `feat(search): port QueryRewriter from POC using shared LLM module`
  - Files: `src/plm/search/components/query_rewriter.py`
  - Pre-commit: Import check passes

---

- [ ] 2. Update HybridRetriever with optional query rewriting

  **What to do**:
  - Import QueryRewriter in `src/plm/search/retriever.py`
  - Add `use_rewrite: bool = False` parameter to `retrieve()` method
  - Add `rewrite_timeout: float = 5.0` parameter to `__init__()`
  - When `use_rewrite=True`: run query through QueryRewriter before expansion
  - When `use_rewrite=False`: skip rewriting (current behavior)
  - Lazy-initialize QueryRewriter (only when first needed)

  **Must NOT do**:
  - Don't change default behavior (default should be `use_rewrite=False` for backward compat)
  - Don't modify other methods
  - Don't add required dependencies

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small, focused change to existing file
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 3, 5, 7
  - **Blocked By**: Task 1

  **References**:
  - `src/plm/search/retriever.py:HybridRetriever` - Class to modify
  - `src/plm/search/retriever.py:retrieve` - Method to update (line 314-443)
  - `src/plm/search/components/query_rewriter.py` - QueryRewriter to import (from Task 1)
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid_llm.py:351` - How POC uses rewriter

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: retrieve() accepts use_rewrite parameter
    Tool: Bash
    Preconditions: HybridRetriever updated
    Steps:
      1. uv run python -c "
         import inspect
         from plm.search.retriever import HybridRetriever
         sig = inspect.signature(HybridRetriever.retrieve)
         params = list(sig.parameters.keys())
         print('use_rewrite' in params)
         "
      2. Assert: Output is "True"
    Expected Result: Parameter exists
    Evidence: Output captured

  Scenario: use_rewrite=False preserves original behavior
    Tool: Bash
    Preconditions: Retriever with indexed data
    Steps:
      1. Run retrieve() with use_rewrite=False
      2. Assert: No LLM call made (check logs or timing)
    Expected Result: Fast retrieval without LLM
    Evidence: Timing/log output

  Scenario: use_rewrite=True triggers QueryRewriter
    Tool: Bash
    Preconditions: Retriever with indexed data, ANTHROPIC_API_KEY set
    Steps:
      1. Run retrieve() with use_rewrite=True
      2. Assert: QueryRewriter was invoked (check logs)
    Expected Result: LLM rewriting happens
    Evidence: Log output showing rewrite
  ```

  **Commit**: YES
  - Message: `feat(search): add optional query rewriting to HybridRetriever.retrieve()`
  - Files: `src/plm/search/retriever.py`
  - Pre-commit: Import check passes

---

- [ ] 3. Create FastAPI service with directory watcher

  **What to do**:
  - Create `src/plm/search/service/__init__.py`
  - Create `src/plm/search/service/app.py` with FastAPI app:
    - POST /query - Accept `{"query": str, "k": int, "use_rewrite": bool}`
    - GET /health - Return service status
    - GET /status - Return index stats
  - Create `src/plm/search/service/watcher.py`:
    - Watch directory for new JSON files
    - **ROBUSTNESS (from Oracle review):**
      - Atomic file handling: only process files NOT ending in `.tmp` (writers rename on complete)
      - Failure handling: move bad files to `failed/` with `.error` metadata file
      - Batch debounce: collect files for 1s, then batch ingest with single rebuild
      - Call `ingest_document(..., rebuild_index=False)` per file, then `rebuild_index()` once
      - SIGTERM handler: drain queue before shutdown, don't lose in-flight files
      - Startup flag: `--process-existing` (default: ignore existing files on start)
    - Move successfully processed files to "processed/"
  - Service lifecycle: starting → ready

  **Must NOT do**:
  - Don't add authentication
  - Don't hardcode paths (use environment variables)
  - Don't process incomplete files (watch for `.tmp` suffix)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Multiple files, threading, service architecture
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 5, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 4

  **References**:
  - `src/plm/search/retriever.py:HybridRetriever` - Class to wrap
  - `src/plm/search/adapters/gliner_adapter.py` - Adapter for ingestion
  - `src/plm/extraction/fast/document_processor.py:DocumentResult` - Input type
  - FastAPI lifespan: https://fastapi.tiangolo.com/advanced/events/

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Service starts and responds to health check
    Tool: Bash
    Steps:
      1. Start: uv run uvicorn plm.search.service.app:app --port 8000 &
      2. Wait: 15 seconds
      3. curl -s http://localhost:8000/health
      4. Assert: Response contains "status"
      5. Kill uvicorn
    Expected Result: Health endpoint works
    Evidence: Response JSON

  Scenario: Query endpoint accepts use_rewrite flag
    Tool: Bash
    Preconditions: Service running with indexed data
    Steps:
      1. curl -X POST http://localhost:8000/query \
           -H "Content-Type: application/json" \
           -d '{"query": "test", "k": 5, "use_rewrite": false}'
      2. Assert: HTTP 200
      3. Assert: Response contains "results"
    Expected Result: Query with rewrite flag works
    Evidence: Response JSON

  Scenario: Watcher auto-ingests new files
    Tool: Bash
    Steps:
      1. Start service with empty watched directory
      2. Copy DocumentResult JSON to watched/
      3. Wait: 5 seconds
      4. Assert: File moved to watched/processed/
      5. GET /status shows chunk_count > 0
    Expected Result: Auto-ingestion works
    Evidence: Status response

  Scenario: Watcher ignores incomplete writes (.tmp files)
    Tool: Bash
    Steps:
      1. Start service with empty watched directory
      2. Write file to watched/test.json.tmp (incomplete)
      3. Wait: 3 seconds
      4. Assert: File NOT moved (still in watched/)
      5. Rename watched/test.json.tmp → watched/test.json
      6. Wait: 3 seconds
      7. Assert: File moved to watched/processed/
    Expected Result: Only processes complete files
    Evidence: Directory listing at each step

  Scenario: Processing failure creates error record
    Tool: Bash
    Steps:
      1. Copy malformed JSON to watched/bad.json
      2. Wait: 5 seconds
      3. Assert: File moved to watched/failed/bad.json
      4. Assert: watched/failed/bad.json.error exists with exception
    Expected Result: Failures tracked, not lost
    Evidence: Error file contents

  Scenario: Graceful shutdown preserves in-flight files
    Tool: Bash
    Steps:
      1. Start service, begin processing large file
      2. Send SIGTERM mid-processing
      3. Wait: 2 seconds for shutdown
      4. Assert: File either in processed/ OR still in watched/ (NOT lost)
    Expected Result: No data loss on shutdown
    Evidence: Directory listing
  ```

  **Commit**: YES
  - Message: `feat(search): add FastAPI service with directory watcher`
  - Files: `src/plm/search/service/*.py`

---

- [ ] 4. Add dependencies to pyproject.toml

  **What to do**:
  - Add: fastapi, uvicorn[standard], watchdog, click
  - Run `uv sync`

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:
  - `pyproject.toml` - Add to dependencies

  **Acceptance Criteria**:

  ```
  Scenario: Dependencies install
    Tool: Bash
    Steps:
      1. uv sync
      2. uv run python -c "import fastapi, uvicorn, watchdog, click; print('OK')"
      3. Assert: Output is "OK"
  ```

  **Commit**: YES
  - Message: `feat(search): add service dependencies`
  - Files: `pyproject.toml`, `uv.lock`

---

- [ ] 5. Create CLI with --no-rewrite flag

  **What to do**:
  - Create `src/plm/search/service/cli.py` with Click
  - Command: `plm-query <text>`
  - Options: `--url`, `--k`, `--no-rewrite` (flag to disable Haiku)
  - Add entry point to pyproject.toml

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 6)
  - **Blocks**: Task 7
  - **Blocked By**: Task 2

  **References**:
  - `src/plm/extraction/fast/cli.py` - Existing CLI pattern
  - Click docs: https://click.palletsprojects.com/

  **Acceptance Criteria**:

  ```
  Scenario: CLI has --no-rewrite flag
    Tool: Bash
    Steps:
      1. uv run plm-query --help
      2. Assert: Output contains "--no-rewrite"
    Evidence: Help output

  Scenario: --no-rewrite skips Haiku
    Tool: Bash
    Preconditions: Service running
    Steps:
      1. uv run plm-query "test" --no-rewrite
      2. Assert: Fast response (< 1s, no LLM call)
  ```

  **Commit**: YES
  - Message: `feat(search): add CLI with --no-rewrite flag`
  - Files: `src/plm/search/service/cli.py`, `pyproject.toml`

---

- [ ] 6. Create Docker Compose setup

  **What to do**:
  - Create `docker/docker-compose.pipeline.yml`
  - Create `docker/Dockerfile.pipeline`
  - Volumes for index persistence and watched directory
  - Environment variables for paths

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 5)
  - **Blocks**: Task 7
  - **Blocked By**: Task 4

  **References**:
  - `docker/` - Existing Docker directory

  **Acceptance Criteria**:

  ```
  Scenario: Docker image builds
    Tool: Bash
    Steps:
      1. docker build -f docker/Dockerfile.pipeline -t plm-pipeline .
      2. Assert: Exit code 0

  Scenario: Container serves health endpoint
    Tool: Bash
    Steps:
      1. docker compose -f docker/docker-compose.pipeline.yml up -d
      2. Wait: 30 seconds
      3. curl http://localhost:8000/health
      4. Assert: Response contains "status"
      5. docker compose down
  ```

  **Commit**: YES
  - Message: `feat(docker): add pipeline service Docker setup`
  - Files: `docker/docker-compose.pipeline.yml`, `docker/Dockerfile.pipeline`

---

- [ ] 7. Create POC comparison test

  **What to do**:
  - Create `scripts/compare_with_poc.py` that:
    1. Loads same test data (kubernetes_sample_200, first 3 docs like POC test)
    2. **Use `use_rewrite=False` in BOTH pipelines** (no LLM, deterministic)
    3. Runs POC pipeline: ModularEnrichedHybridLLM
    4. Runs production pipeline: HybridRetriever
    5. For each needle question:
       - Get results from both
       - **CRITICAL: Compare CONTENT TEXT, not chunk IDs**
         - POC IDs: `doc_id:heading:idx` (strategy-generated)
         - Production IDs: `doc_id_i_uuid8` (UUID-based, never match)
         - Compare: `{r['content'] for r in results}` as sets
       - Report MATCH (same content) or DIFF (with details)
    6. Summary: X/20 queries have matching content (target: ≥18/20 = 90%)

  **This is the CRITICAL test the user requested.**

  **Must NOT do**:
  - Don't compare chunk IDs (they will NEVER match due to UUID)
  - Don't mock HybridRetriever internals
  - Don't use LLM rewriting (use_rewrite=False for determinism)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Integration work, POC + production code paths
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (final)
  - **Blocks**: None
  - **Blocked By**: Tasks 3, 5, 6

  **References**:
  - `poc/modular_retrieval_pipeline/test_exact_replication.py` - HOW to mock query rewriting (lines 27-35)
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid_llm.py` - POC pipeline to compare against
  - `src/plm/search/retriever.py:HybridRetriever` - Production pipeline
  - `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` - Test corpus
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Test questions

  **Acceptance Criteria**:

  **Agent-Executed QA Scenarios:**

  ```
  Scenario: Comparison script runs end-to-end
    Tool: Bash
    Preconditions: All previous tasks complete
    Steps:
      1. uv run python scripts/compare_with_poc.py
      2. Assert: Script completes without error
      3. Assert: Output shows "X/20 queries match"
    Expected Result: Comparison completes with match report
    Evidence: Script output captured to .sisyphus/evidence/task-7-comparison.txt

  Scenario: Verify CONTENT matches POC (not IDs - IDs use UUID)
    Tool: Bash
    Steps:
      1. Run comparison script
      2. For a known query (e.g., "What is the Topology Manager?"):
         - Get content sets from both: {r['content'] for r in results}
         - Assert: production_contents == poc_contents (as sets)
    Expected Result: Same content retrieved (IDs will differ due to UUID)
    Evidence: Content comparison output

  Scenario: Report explains content differences
    Tool: Bash
    Steps:
      1. If any query shows DIFF:
         - Assert: Output shows which content chunks differ
         - Assert: Possible explanation logged (chunking strategy, enrichment)
    Expected Result: Differences explained with context
    Evidence: Diff explanation captured

  Scenario: Achieve ≥90% match rate
    Tool: Bash
    Steps:
      1. Run comparison script
      2. Assert: Match rate ≥ 18/20 (90%)
      3. If < 90%: Log warning with failure analysis
    Expected Result: High content match rate
    Evidence: Summary statistics
  ```

  **Commit**: YES
  - Message: `feat(scripts): add POC comparison test for production pipeline`
  - Files: `scripts/compare_with_poc.py`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(search): port QueryRewriter from POC using shared LLM` | query_rewriter.py |
| 2 | `feat(search): add optional query rewriting to retrieve()` | retriever.py |
| 3 | `feat(search): add FastAPI service with watcher` | service/*.py |
| 4 | `feat(search): add service dependencies` | pyproject.toml |
| 5 | `feat(search): add CLI with --no-rewrite` | cli.py |
| 6 | `feat(docker): add pipeline Docker setup` | docker/* |
| 7 | `feat(scripts): add POC comparison test` | compare_with_poc.py |

---

## Success Criteria

### Verification Commands
```bash
# 1. Verify QueryRewriter uses shared LLM
grep "from plm.shared.llm" src/plm/search/components/query_rewriter.py

# 2. Verify HybridRetriever has use_rewrite param
grep "use_rewrite" src/plm/search/retriever.py

# 3. Start service
docker compose -f docker/docker-compose.pipeline.yml up -d

# 4. Query WITH rewriting
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Topology Manager?", "use_rewrite": true}'

# 5. Query WITHOUT rewriting
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the Topology Manager?", "use_rewrite": false}'

# 6. CLI with --no-rewrite
uv run plm-query "What is the Topology Manager?" --no-rewrite

# 7. Run comparison with POC
uv run python scripts/compare_with_poc.py
```

### Final Checklist
- [ ] QueryRewriter uses `src/plm/shared/llm/base.py:call_llm()`
- [ ] HybridRetriever.retrieve() has `use_rewrite` parameter
- [ ] Service accepts `use_rewrite` in /query endpoint
- [ ] CLI has `--no-rewrite` flag
- [ ] POC comparison script shows ≥90% content matches (not IDs)
- [ ] Directory watcher handles failures (failed/ directory)
- [ ] Docker setup persists index
