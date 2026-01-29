# Modular Pipeline Independence: Logger + LLM Provider

## TL;DR

> **Quick Summary**: Make `modular_retrieval_pipeline` fully independent from `chunking_benchmark_v2` by copying the logger, LLM provider, and adding extensive logging throughout all components.
> 
> **Deliverables**: 
> - `utils/logger.py` - BenchmarkLogger (copied from chunking_benchmark_v2)
> - `utils/llm_provider.py` - AnthropicProvider + call_llm (simplified)
> - Modified `query_rewriter.py` - Local imports, no sys.path hack
> - Logging added to all 10 component files
> 
> **Estimated Effort**: Medium (6 tasks, ~4-6 hours)
> **Parallel Execution**: YES - Waves 2 and 3 can parallel
> **Critical Path**: Logger → LLM Provider → Query Rewriter → Component Logging

---

## Context

### Original Request
Make the modular_retrieval_pipeline fully independent by:
1. Adding logger implementation (copy BenchmarkLogger)
2. Adding extensive logs using info/debug/trace for debugging
3. Removing dependency on chunking_benchmark_v2

### Research Findings

**Current State**:
- 1 production file with external dependency: `components/query_rewriter.py`
- 0 logging in any component files
- 10 component files need logging added

**Source Files to Copy**:
| File | Lines | Purpose |
|------|-------|---------|
| `chunking_benchmark_v2/logger.py` | ~385 | BenchmarkLogger class |
| `chunking_benchmark_v2/enrichment/provider.py` | ~220 | AnthropicProvider (Anthropic only) |
| `chunking_benchmark_v2/retrieval/query_rewrite.py` | ~50 | QUERY_REWRITE_PROMPT + function |

### Metis Review

**Guardrails Applied**:
- No new dependencies (logger stdlib-only, LLM provider needs only httpx)
- Preserve existing function signatures
- No changes to Component protocol (base.py, types.py)
- Do not add logging to test files
- Do not refactor component logic - only ADD logging

**Defaults Applied**:
- Logger: Global singleton pattern (like chunking_benchmark_v2)
- LLM Provider: Anthropic only (exclude OllamaProvider - not used)
- Auth path: Keep hardcoded `~/.local/share/opencode/auth.json`
- Error handling: Silent fallback (return original query on failure)

---

## Work Objectives

### Core Objective
Remove all imports from `chunking_benchmark_v2` in production code and add comprehensive logging for debugging.

### Concrete Deliverables
- `poc/modular_retrieval_pipeline/utils/__init__.py`
- `poc/modular_retrieval_pipeline/utils/logger.py`
- `poc/modular_retrieval_pipeline/utils/llm_provider.py`
- Modified `poc/modular_retrieval_pipeline/components/query_rewriter.py`
- Logging added to 10 component files

### Definition of Done
- [x] No `chunking_benchmark_v2` imports in production code (grep verification)
- [x] Logger works (Python one-liner test)
- [x] LLM provider works (Python one-liner test)
- [x] Query rewriter works end-to-end (Python one-liner test)
- [x] Each component has >= 3 logger calls

### Must Have
- BenchmarkLogger with TRACE/DEBUG/INFO/WARN/ERROR levels
- AnthropicProvider with OAuth token refresh
- `call_llm()` wrapper function
- Logging in all 10 components (init, process entry, process exit minimum)

### Must NOT Have (Guardrails)
- Do NOT add OllamaProvider (not used in production)
- Do NOT modify base.py or types.py
- Do NOT add logging to test files (test_*.py, benchmark.py)
- Do NOT add new dependencies beyond httpx
- Do NOT change component behavior (only add logging)
- Do NOT add metrics/timing instrumentation
- Do NOT add retry logic to LLM provider

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest available)
- **User wants tests**: NO (manual verification procedures instead)
- **QA approach**: Automated bash/Python verification commands

### Automated Verification (ALL agent-executable)

**For all tasks**: Use bash Python one-liners and grep commands that agents can execute directly.

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
└── Task 1: Create utils/logger.py (no dependencies)

Wave 2 (After Wave 1):
├── Task 2: Create utils/llm_provider.py (depends on logger)
└── Task 3: Modify query_rewriter.py (depends on logger, llm_provider)

Wave 3 (After Wave 2):
├── Task 4: Add logging to extraction components (keyword, entity, content)
├── Task 5: Add logging to query components (expander, embedding, bm25)
└── Task 6: Add logging to scoring components (similarity, rrf, reranker)

Critical Path: Task 1 → Task 2 → Task 3
Parallel Speedup: Tasks 4, 5, 6 run in parallel after Task 3
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5, 6 | None |
| 2 | 1 | 3 | None |
| 3 | 1, 2 | 4, 5, 6 | None |
| 4 | 3 | None | 5, 6 |
| 5 | 3 | None | 4, 6 |
| 6 | 3 | None | 4, 5 |

---

## TODOs

- [x] 1. Create utils/logger.py (Copy BenchmarkLogger)

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/utils/` directory
  - Create `poc/modular_retrieval_pipeline/utils/__init__.py` with exports
  - Copy `BenchmarkLogger` class from `poc/chunking_benchmark_v2/logger.py`
  - Copy `get_logger()` and `set_logger()` functions
  - Keep TRACE/DEBUG/INFO/WARN/ERROR levels

  **Must NOT do**:
  - Add any new dependencies
  - Modify BenchmarkLogger behavior
  - Add new log levels

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Straightforward file copy with minor path adjustments
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundation for all other tasks)
  - **Parallel Group**: Wave 1
  - **Blocks**: Tasks 2, 3, 4, 5, 6
  - **Blocked By**: None

  **References**:
  - `poc/chunking_benchmark_v2/logger.py` - FULL FILE (copy lines 1-385)
  - Pattern: Copy verbatim, only change is removing any external imports

  **Acceptance Criteria**:
  ```bash
  # Verify file exists
  test -f poc/modular_retrieval_pipeline/utils/logger.py && echo "OK" || echo "MISSING"
  
  # Verify logger works
  cd poc/modular_retrieval_pipeline && python -c "
  from utils.logger import BenchmarkLogger, get_logger, set_logger
  logger = BenchmarkLogger(console=True, min_level='DEBUG')
  set_logger(logger)
  logger.info('INFO test')
  logger.debug('DEBUG test')
  logger.warn('WARN test')
  print('SUCCESS')
  "
  # Assert: Outputs log messages and "SUCCESS"
  ```

  **Commit**: YES
  - Message: `feat(modular): add logger utility (copy from chunking_benchmark_v2)`
  - Files: `utils/__init__.py`, `utils/logger.py`

---

- [x] 2. Create utils/llm_provider.py (AnthropicProvider + call_llm)

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/utils/llm_provider.py`
  - Copy `AnthropicProvider` class from `poc/chunking_benchmark_v2/enrichment/provider.py`
  - Copy `call_llm()` function
  - Copy `get_provider()` function (simplified for Anthropic only)
  - Update imports to use local logger: `from .logger import get_logger`
  - **EXCLUDE** OllamaProvider (not used in production)

  **Must NOT do**:
  - Add OllamaProvider
  - Add retry logic
  - Make auth path configurable

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: File copy with import adjustments
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/provider.py:78-294` - AnthropicProvider class
  - `poc/chunking_benchmark_v2/enrichment/provider.py:297-321` - get_provider, call_llm
  - Pattern: Copy AnthropicProvider + call_llm, exclude OllamaProvider

  **Acceptance Criteria**:
  ```bash
  # Verify file exists
  test -f poc/modular_retrieval_pipeline/utils/llm_provider.py && echo "OK" || echo "MISSING"
  
  # Verify no OllamaProvider
  grep -q "OllamaProvider" poc/modular_retrieval_pipeline/utils/llm_provider.py && echo "FAIL: OllamaProvider found" || echo "OK: No OllamaProvider"
  
  # Verify imports local logger
  grep -q "from .logger import" poc/modular_retrieval_pipeline/utils/llm_provider.py && echo "OK" || echo "FAIL: Wrong import"
  
  # Verify LLM call works (requires auth)
  cd poc/modular_retrieval_pipeline && python -c "
  from utils.llm_provider import call_llm
  result = call_llm('Say hello in 3 words', model='claude-haiku', timeout=10)
  print(f'Response: {result[:50]}...' if result else 'EMPTY')
  print('SUCCESS' if result else 'FAIL')
  "
  # Assert: Outputs response and "SUCCESS"
  ```

  **Commit**: YES
  - Message: `feat(modular): add LLM provider utility (Anthropic only)`
  - Files: `utils/llm_provider.py`, `utils/__init__.py`

---

- [x] 3. Modify query_rewriter.py to use local imports

  **What to do**:
  - Remove `sys.path.insert(...)` hack (line 28)
  - Remove `from retrieval.query_rewrite import rewrite_query` (line 29)
  - Add import: `from ..utils.llm_provider import call_llm`
  - Add import: `from ..utils.logger import get_logger`
  - Copy `QUERY_REWRITE_PROMPT` from `poc/chunking_benchmark_v2/retrieval/query_rewrite.py`
  - Inline the `rewrite_query()` logic directly in the file (as private function `_rewrite_query`)
  - Update `QueryRewriter.process()` to call `_rewrite_query()`
  - Add logging: DEBUG for rewrite attempts, INFO for success/fallback

  **Must NOT do**:
  - Change the `QueryRewriter` class interface
  - Modify timeout/fallback behavior
  - Make prompt configurable

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Critical component, needs careful refactoring
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2
  - **Blocks**: Tasks 4, 5, 6
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/components/query_rewriter.py` - Current file to modify
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:19-35` - QUERY_REWRITE_PROMPT
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:38-86` - rewrite_query() logic

  **Acceptance Criteria**:
  ```bash
  # Verify no sys.path hack
  grep -q "sys.path.insert" poc/modular_retrieval_pipeline/components/query_rewriter.py && echo "FAIL: sys.path hack found" || echo "OK"
  
  # Verify no chunking_benchmark_v2 import
  grep -q "chunking_benchmark_v2\|from retrieval" poc/modular_retrieval_pipeline/components/query_rewriter.py && echo "FAIL: External import found" || echo "OK"
  
  # Verify local imports
  grep -q "from ..utils" poc/modular_retrieval_pipeline/components/query_rewriter.py && echo "OK" || echo "FAIL: Missing local imports"
  
  # Verify query rewriter works end-to-end
  cd poc/modular_retrieval_pipeline && python -c "
  from components.query_rewriter import QueryRewriter
  from types import Query
  rewriter = QueryRewriter(timeout=5.0)
  result = rewriter.process(Query('why does my token expire'))
  print(f'Original: {result.original.text}')
  print(f'Rewritten: {result.rewritten}')
  print('SUCCESS' if result.rewritten else 'FAIL')
  "
  # Assert: Outputs original and rewritten query
  ```

  **Commit**: YES
  - Message: `refactor(modular): make query_rewriter independent from chunking_benchmark_v2`
  - Files: `components/query_rewriter.py`

---

- [x] 4. Add logging to extraction components

  **What to do**:
  - Add logging to `keyword_extractor.py`:
    - DEBUG: process() entry with content length
    - TRACE: YAKE extraction details
    - DEBUG: process() exit with keywords found
  - Add logging to `entity_extractor.py`:
    - DEBUG: process() entry with content length
    - TRACE: spaCy extraction details, entities by type
    - DEBUG: process() exit with entity count
  - Add logging to `content_enricher.py`:
    - DEBUG: process() entry
    - TRACE: Prefix parts construction
    - DEBUG: process() exit with enriched content length

  **Pattern for each file**:
  ```python
  from ..utils.logger import get_logger
  
  class Component:
      def __init__(self, ...):
          self._log = get_logger()
          self._log.debug(f"[{self.__class__.__name__}] Initialized with ...")
      
      def process(self, data):
          self._log.debug(f"[{self.__class__.__name__}] Processing ...")
          # ... existing logic ...
          self._log.trace(f"[{self.__class__.__name__}] Intermediate: ...")
          # ... more logic ...
          self._log.debug(f"[{self.__class__.__name__}] Completed: ...")
          return result
  ```

  **Must NOT do**:
  - Change component behavior
  - Add metrics/timing
  - Modify function signatures

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical logging additions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 5, 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py`
  - `poc/modular_retrieval_pipeline/components/entity_extractor.py`
  - `poc/modular_retrieval_pipeline/components/content_enricher.py`
  - Logging pattern: `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` (trace log style)

  **Acceptance Criteria**:
  ```bash
  # Verify logging in keyword_extractor
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/keyword_extractor.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in entity_extractor
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/entity_extractor.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in content_enricher
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/content_enricher.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  ```

  **Commit**: YES
  - Message: `feat(modular): add logging to extraction components`
  - Files: `keyword_extractor.py`, `entity_extractor.py`, `content_enricher.py`

---

- [x] 5. Add logging to query/encoding components

  **What to do**:
  - Add logging to `query_expander.py`:
    - DEBUG: process() entry with query text
    - TRACE: Expansion terms matched
    - DEBUG: process() exit with expanded query (or "no expansion")
  - Add logging to `embedding_encoder.py`:
    - DEBUG: encode() entry with text count
    - TRACE: Embedding dimensions
    - DEBUG: encode() exit with shape
  - Add logging to `bm25_scorer.py`:
    - DEBUG: index() entry with document count
    - DEBUG: score() entry with query
    - TRACE: Top-k scores
    - DEBUG: score() exit

  **Must NOT do**:
  - Change component behavior
  - Add metrics/timing
  - Modify function signatures

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical logging additions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 6)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:
  - `poc/modular_retrieval_pipeline/components/query_expander.py`
  - `poc/modular_retrieval_pipeline/components/embedding_encoder.py`
  - `poc/modular_retrieval_pipeline/components/bm25_scorer.py`

  **Acceptance Criteria**:
  ```bash
  # Verify logging in query_expander
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/query_expander.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in embedding_encoder
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/embedding_encoder.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in bm25_scorer
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/bm25_scorer.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  ```

  **Commit**: YES
  - Message: `feat(modular): add logging to query/encoding components`
  - Files: `query_expander.py`, `embedding_encoder.py`, `bm25_scorer.py`

---

- [x] 6. Add logging to scoring/fusion components

  **What to do**:
  - Add logging to `similarity_scorer.py`:
    - DEBUG: score() entry with query
    - TRACE: Similarity computation details
    - DEBUG: score() exit with top scores
  - Add logging to `rrf_fuser.py`:
    - DEBUG: fuse() entry with result sets
    - TRACE: RRF score calculations
    - DEBUG: fuse() exit with final ranking
  - Add logging to `reranker.py`:
    - DEBUG: rerank() entry with candidates
    - TRACE: Reranking scores
    - DEBUG: rerank() exit with reordered results

  **Must NOT do**:
  - Change component behavior
  - Add metrics/timing
  - Modify function signatures

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Mechanical logging additions
  - **Skills**: []

  **Parallelization**:
  - **Can Run In Parallel**: YES (with Tasks 4, 5)
  - **Parallel Group**: Wave 3
  - **Blocks**: None
  - **Blocked By**: Task 3

  **References**:
  - `poc/modular_retrieval_pipeline/components/similarity_scorer.py`
  - `poc/modular_retrieval_pipeline/components/rrf_fuser.py`
  - `poc/modular_retrieval_pipeline/components/reranker.py`

  **Acceptance Criteria**:
  ```bash
  # Verify logging in similarity_scorer
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/similarity_scorer.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in rrf_fuser
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/rrf_fuser.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  
  # Verify logging in reranker
  count=$(grep -c "self._log\." poc/modular_retrieval_pipeline/components/reranker.py)
  [ "$count" -ge 3 ] && echo "OK: $count log calls" || echo "FAIL: Only $count log calls"
  ```

  **Commit**: YES
  - Message: `feat(modular): add logging to scoring/fusion components`
  - Files: `similarity_scorer.py`, `rrf_fuser.py`, `reranker.py`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(modular): add logger utility (copy from chunking_benchmark_v2)` | utils/__init__.py, utils/logger.py |
| 2 | `feat(modular): add LLM provider utility (Anthropic only)` | utils/llm_provider.py |
| 3 | `refactor(modular): make query_rewriter independent from chunking_benchmark_v2` | components/query_rewriter.py |
| 4 | `feat(modular): add logging to extraction components` | keyword_extractor.py, entity_extractor.py, content_enricher.py |
| 5 | `feat(modular): add logging to query/encoding components` | query_expander.py, embedding_encoder.py, bm25_scorer.py |
| 6 | `feat(modular): add logging to scoring/fusion components` | similarity_scorer.py, rrf_fuser.py, reranker.py |

---

## Success Criteria

### Final Verification Commands
```bash
# 1. Independence: No chunking_benchmark_v2 imports in production code
grep -r "chunking_benchmark_v2\|from retrieval\." poc/modular_retrieval_pipeline/components/ poc/modular_retrieval_pipeline/utils/ 2>/dev/null | grep -v "__pycache__"
# Expected: No output (exit code 1)

# 2. Logger works
cd poc/modular_retrieval_pipeline && python -c "
from utils.logger import BenchmarkLogger, get_logger, set_logger
logger = BenchmarkLogger(console=True, min_level='TRACE')
set_logger(logger)
logger.trace('TRACE')
logger.debug('DEBUG')
logger.info('INFO')
print('Logger OK')
"

# 3. LLM Provider works
cd poc/modular_retrieval_pipeline && python -c "
from utils.llm_provider import call_llm
result = call_llm('Say OK', model='claude-haiku', timeout=10)
print('LLM Provider OK' if result else 'LLM Provider FAIL')
"

# 4. Query Rewriter works end-to-end
cd poc/modular_retrieval_pipeline && python -c "
from components.query_rewriter import QueryRewriter
from types import Query
rewriter = QueryRewriter(timeout=5.0)
result = rewriter.process(Query('why does token expire'))
print('Query Rewriter OK' if result.rewritten else 'Query Rewriter FAIL')
"

# 5. All components have logging (>=3 log calls each)
for f in poc/modular_retrieval_pipeline/components/*.py; do
  name=$(basename "$f")
  count=$(grep -c "self._log\." "$f" 2>/dev/null || echo 0)
  [ "$count" -ge 3 ] && status="OK" || status="FAIL"
  echo "$name: $count log calls ($status)"
done
```

### Final Checklist
- [x] `utils/logger.py` exists and works
- [x] `utils/llm_provider.py` exists (Anthropic only, no OllamaProvider)
- [x] `query_rewriter.py` has no sys.path hack
- [x] `query_rewriter.py` uses local imports only
- [x] All 9 other components have >= 3 logger calls each
- [x] No `chunking_benchmark_v2` imports in production code
- [x] All acceptance criteria pass
