## Pipeline Service Implementation - Learnings

**Date**: 2026-02-18
**Plan**: pipeline-service
**Status**: Complete (7/7 tasks)

---

### What Was Implemented

This plan ported the QueryRewriter from POC to production and built a complete search service with optional query rewriting.

#### Components Created/Modified:

1. **QueryRewriter** (`src/plm/search/components/query_rewriter.py`)
   - Ported from POC with shared LLM module integration
   - Uses `from plm.shared.llm import call_llm`
   - Returns `RewrittenQuery` type from `plm.search.types`
   - Preserves original QUERY_REWRITE_PROMPT exactly

2. **HybridRetriever Updates** (`src/plm/search/retriever.py`)
   - Added `use_rewrite: bool = False` parameter to `retrieve()`
   - Added `rewrite_timeout: float = 5.0` to `__init__`
   - Lazy-initializes QueryRewriter only when first needed
   - Maintains backward compatibility (default=False)

3. **FastAPI Service** (`src/plm/search/service/`)
   - `app.py`: FastAPI with `/query`, `/health`, `/status` endpoints
   - `watcher.py`: Directory watcher with robust failure handling
   - Supports `use_rewrite` flag in query requests

4. **CLI Tool** (`src/plm/search/service/cli.py`)
   - `plm-query` command with `--no-rewrite` flag
   - Entry point in pyproject.toml

5. **Dependencies** (pyproject.toml)
   - fastapi>=0.100.0
   - uvicorn[standard]>=0.23.0
   - watchdog>=3.0.0
   - click>=8.0.0

6. **Docker Setup**
   - `docker/docker-compose.pipeline.yml`
   - `docker/Dockerfile.pipeline`
   - Persistent volume for index

7. **POC Comparison** (`scripts/compare_with_poc.py`)
   - Compares production vs POC pipeline results
   - Compares CONTENT (not IDs, which differ by design)
   - Target: >=90% content match rate

---

### Key Patterns Learned

#### Shared LLM Integration
```python
from plm.shared.llm import call_llm

# Usage identical to POC's call_llm
rewritten = call_llm(
    prompt, 
    model="claude-haiku", 
    timeout=int(self.timeout)
)
```

#### Lazy Initialization Pattern
```python
# Store as None initially
self._query_rewriter: QueryRewriter | None = None

# Create only when first needed
if self._query_rewriter is None:
    self._query_rewriter = QueryRewriter(timeout=self.rewrite_timeout)
```

#### Backward Compatibility
```python
# Default to False to maintain existing behavior
def retrieve(self, query: str, k: int = 5, use_rewrite: bool = False) -> list[dict]:
```

---

### Verification Commands

```bash
# Test QueryRewriter
uv run python -c "from plm.search.components.query_rewriter import QueryRewriter; print('OK')"

# Test CLI
uv run plm-query --help

# Start service
docker compose -f docker/docker-compose.search.yml up -d

# Query with/without rewriting
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "test", "k": 5, "use_rewrite": true}'

# Compare with POC
uv run python scripts/compare_with_poc.py
```

---

### Commit History

1. `feat(search): port QueryRewriter from POC using shared LLM module`
2. `feat(search): add optional query rewriting to HybridRetriever.retrieve()`

---

### Notes

- All work was already implemented before this session started
- This session verified existing implementation and marked tasks complete
- Plan had 7 tasks, but only Tasks 1-2 needed commits in this session
- Tasks 3-7 were pre-existing from previous work
