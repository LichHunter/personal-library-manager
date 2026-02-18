# Enriched Hybrid Strategy (No LLM)

## TL;DR

> **Quick Summary**: Create a variant of `enriched_hybrid_llm (modular)` that skips LLM-based query rewriting, then benchmark and document results.
> 
> **Deliverables**:
> - New orchestrator: `modular_enriched_hybrid.py`
> - Updated benchmark with new strategy
> - Results added to `BENCHMARK_RESULTS.md`
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential (each task depends on previous)
> **Critical Path**: Task 1 (create orchestrator) -> Task 2 (update benchmark) -> Task 3 (run & document)

---

## Context

### Original Request
Create new strategy identical to `enriched_hybrid_llm (modular)` but without LLM query rewriting. Run benchmark with same data and add results to documentation.

### Interview Summary
**Key Discussions**:
- Strategy name: `enriched_hybrid` (drop "_llm" suffix)
- Keep QueryExpander (rule-based, not LLM)
- Use same benchmark data: 200 K8s docs, 20 questions

**Research Findings**:
- Current flow: Query -> QueryRewriter (Claude Haiku) -> QueryExpander -> Search -> RRF
- New flow: Query -> QueryExpander -> Search -> RRF (skip rewriter)
- Rewriter invocation is isolated in lines 312-330

### Metis Review
**Identified Gaps** (addressed):
- `get_index_stats()` also references `rewrite_timeout` - will remove
- Class docstring mentions LLM rewriting - will update
- Benchmark report is hardcoded for 2 strategies - will create separate report function
- No `RewrittenQuery` type needed - will simplify imports

---

## Work Objectives

### Core Objective
Create and benchmark a non-LLM variant of the hybrid retrieval strategy to measure the impact of query rewriting on accuracy.

### Concrete Deliverables
- `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py` - New orchestrator class
- Updated `poc/modular_retrieval_pipeline/benchmark.py` - New benchmark function
- Updated `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md` - New results row

### Definition of Done
- [ ] `python -c "from poc.modular_retrieval_pipeline.modular_enriched_hybrid import ModularEnrichedHybrid"` succeeds
- [ ] `grep -c "QueryRewriter" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py` returns 0
- [ ] Benchmark runs and produces accuracy percentage
- [ ] `BENCHMARK_RESULTS.md` contains row for `enriched_hybrid (modular, no LLM)`

### Must Have
- Identical retrieval logic to `enriched_hybrid_llm` except query rewriting
- Same interface: `set_embedder()`, `index()`, `retrieve()`
- Benchmark with same 20 questions and 200 documents
- Results documented with components listed

### Must NOT Have (Guardrails)
- NO changes to existing `ModularEnrichedHybridLLM` class
- NO changes to baseline benchmark function
- NO changes to existing test files
- NO new dependencies
- NO LLM calls in new strategy

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (existing benchmark system)
- **User wants tests**: Manual verification via benchmark
- **Framework**: Built-in benchmark runner

### Automated Verification

Each TODO includes EXECUTABLE verification procedures:

**For new orchestrator (using Bash)**:
```bash
# Import test
python -c "from poc.modular_retrieval_pipeline.modular_enriched_hybrid import ModularEnrichedHybrid; print('OK')"
# Assert: Output is "OK"

# No LLM references
grep -c "QueryRewriter\|rewrite_timeout\|RewrittenQuery" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py
# Assert: Returns 0
```

**For benchmark (using Bash)**:
```bash
cd poc/modular_retrieval_pipeline && python benchmark.py 2>&1 | grep -E "accuracy|VERDICT"
# Assert: Contains accuracy percentages for all strategies
```

---

## Execution Strategy

### Sequential Execution (No Parallelization)

```
Task 1: Create modular_enriched_hybrid.py
    |
    v
Task 2: Update benchmark.py with new function
    |
    v
Task 3: Run benchmark and update documentation
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1 | None | 2, 3 |
| 2 | 1 | 3 |
| 3 | 1, 2 | None |

---

## TODOs

- [x] 1. Create ModularEnrichedHybrid orchestrator

  **What to do**:
  - Copy `modular_enriched_hybrid_llm.py` to `modular_enriched_hybrid.py`
  - Rename class `ModularEnrichedHybridLLM` -> `ModularEnrichedHybrid`
  - Remove imports: `QueryRewriter`, `RewrittenQuery`
  - Remove `rewrite_timeout` parameter from `__init__`
  - Remove `self._query_rewriter` initialization (line 121)
  - Remove `self._rewrite_time_s` stat (line 135)
  - In `retrieve()`: Remove rewrite block (lines 312-330), use `query` directly instead of `rewritten_query`
  - In `get_index_stats()`: Remove `"rewrite_timeout_s"` entry
  - Update class docstring to remove LLM rewriting references

  **Must NOT do**:
  - Modify existing `modular_enriched_hybrid_llm.py`
  - Change any retrieval logic (RRF, weights, expansion)
  - Remove QueryExpander (it's rule-based, not LLM)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file creation with straightforward copy-and-edit pattern
  - **Skills**: `[]`
    - No special skills needed - simple file editing

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/modular_enriched_hybrid_llm.py:1-485` - Source file to copy from (entire file)
  
  **Lines to Remove/Modify** (in new file):
  - Line 36: Remove `from .components.query_rewriter import QueryRewriter`
  - Line 39: Change to `from .types import Query` (remove `RewrittenQuery`)
  - Lines 59-83: Update docstring to remove LLM references
  - Lines 91, 100, 106: Remove `rewrite_timeout` parameter and attribute
  - Line 121: Remove `self._query_rewriter = QueryRewriter(timeout=rewrite_timeout)`
  - Line 135: Remove `self._rewrite_time_s = 0.0`
  - Lines 312-330: Remove entire rewrite block
  - Line 334: Change `self._expand_query_direct(rewritten_query)` to `self._expand_query_direct(query)`
  - Line 483: Remove `"rewrite_timeout_s": self.rewrite_timeout,`

  **Acceptance Criteria**:

  ```bash
  # 1. File exists
  test -f poc/modular_retrieval_pipeline/modular_enriched_hybrid.py && echo "EXISTS"
  # Assert: Output is "EXISTS"

  # 2. Class can be imported
  python -c "from poc.modular_retrieval_pipeline.modular_enriched_hybrid import ModularEnrichedHybrid; print('IMPORT_OK')"
  # Assert: Output is "IMPORT_OK"

  # 3. No QueryRewriter references
  grep -c "QueryRewriter" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py || echo "0"
  # Assert: Output is "0"

  # 4. No rewrite_timeout references
  grep -c "rewrite_timeout" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py || echo "0"
  # Assert: Output is "0"

  # 5. No RewrittenQuery references
  grep -c "RewrittenQuery" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py || echo "0"
  # Assert: Output is "0"

  # 6. Class name is correct
  grep -c "class ModularEnrichedHybrid:" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py
  # Assert: Output is "1"
  ```

  **Commit**: YES
  - Message: `feat(retrieval): add enriched hybrid strategy without LLM rewriting`
  - Files: `poc/modular_retrieval_pipeline/modular_enriched_hybrid.py`
  - Pre-commit: Import test passes

---

- [x] 2. Update benchmark.py with new strategy

  **What to do**:
  - Add import for `ModularEnrichedHybrid` from new file
  - Create `run_modular_no_llm_benchmark()` function (copy from `run_modular_benchmark`, change class)
  - Update `main()` to run all 3 strategies
  - Update `generate_report()` to handle 3 strategies OR create separate comparison

  **Must NOT do**:
  - Modify `run_baseline_benchmark()` logic
  - Modify `run_modular_benchmark()` logic
  - Change test data loading

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Adding function following existing pattern
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:191-281` - `run_modular_benchmark()` function to copy pattern from
  - `poc/modular_retrieval_pipeline/benchmark.py:284-344` - `generate_report()` to update

  **Import to Add**:
  ```python
  from .modular_enriched_hybrid import ModularEnrichedHybrid
  ```

  **Function to Create** (based on existing pattern):
  ```python
  def run_modular_no_llm_benchmark(questions, needle_doc_id, chunks, documents, embedder):
      """Run benchmark with modular enriched hybrid strategy (no LLM rewriting)."""
      strategy = ModularEnrichedHybrid(debug=False)  # Note: no rewrite_timeout param
      strategy.set_embedder(embedder)
      # ... rest follows run_modular_benchmark pattern
  ```

  **Acceptance Criteria**:

  ```bash
  # 1. Import exists
  grep -c "from .modular_enriched_hybrid import ModularEnrichedHybrid" poc/modular_retrieval_pipeline/benchmark.py
  # Assert: Output is "1"

  # 2. New benchmark function exists
  grep -c "def run_modular_no_llm_benchmark" poc/modular_retrieval_pipeline/benchmark.py
  # Assert: Output is "1"

  # 3. Benchmark can run (smoke test with --help or quick check)
  cd poc/modular_retrieval_pipeline && python -c "import benchmark; print('BENCHMARK_OK')"
  # Assert: Output is "BENCHMARK_OK"
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add no-LLM strategy to benchmark comparison`
  - Files: `poc/modular_retrieval_pipeline/benchmark.py`
  - Pre-commit: Import test passes

---

- [x] 3. Run benchmark and update documentation

  **What to do**:
  - Run full benchmark: `cd poc/modular_retrieval_pipeline && python benchmark.py`
  - Record accuracy, latency, memory for new strategy
  - Add new row to `BENCHMARK_RESULTS.md` table
  - Add new section describing the strategy flow

  **Must NOT do**:
  - Modify existing strategy descriptions
  - Change table format
  - Fabricate results (must run actual benchmark)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Run command and edit markdown
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final task)
  - **Blocks**: None
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md:5-8` - Table format to follow
  - `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md:33-53` - Flow description format

  **New Row Format** (for table):
  ```markdown
  | enriched_hybrid (modular, no LLM) | KeywordExtractor, EntityExtractor, ContentEnricher, QueryExpander, EmbeddingEncoder, BM25Scorer, SimilarityScorer, RRFFuser | `poc/chunking_benchmark_v2/corpus/kubernetes/` (200 files) | `poc/chunking_benchmark_v2/corpus/needle_questions.json` (20 questions) | XX.X% (N/20) |
  ```

  **New Section Format** (after existing sections):
  ```markdown
  ## enriched_hybrid (modular, no LLM)

  **Flow:**

  1. **Indexing:** (same as enriched_hybrid_llm modular)
  
  2. **Retrieval:**
     - User query -> QueryExpander (domain terms) -> Expanded query (if matched)
     - Expanded query -> SimilarityScorer (cosine similarity) -> Semantic results
     - Expanded query -> BM25Scorer (lexical matching) -> BM25 results
     - RRFFuser merges results with adaptive weights
     - Returns top-k results
     
  **Key Difference**: No LLM-based query rewriting. Uses original user query directly.
  ```

  **Acceptance Criteria**:

  ```bash
  # 1. Benchmark completed (results file updated)
  test -f poc/modular_retrieval_pipeline/benchmark_results.json && echo "RESULTS_EXIST"
  # Assert: Output is "RESULTS_EXIST"

  # 2. New strategy in results
  grep -c "no.LLM\|no_llm\|ModularEnrichedHybrid" poc/modular_retrieval_pipeline/benchmark_results.json || echo "CHECK_MANUALLY"
  # Assert: Contains reference to new strategy

  # 3. Documentation updated
  grep -c "enriched_hybrid.*no LLM" poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md
  # Assert: Output is at least "1"

  # 4. New section exists
  grep -c "## enriched_hybrid (modular, no LLM)" poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md
  # Assert: Output is "1"
  ```

  **Commit**: YES
  - Message: `docs(benchmark): add enriched_hybrid (no LLM) benchmark results`
  - Files: `poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md`, `poc/modular_retrieval_pipeline/benchmark_results.json`
  - Pre-commit: None required

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(retrieval): add enriched hybrid strategy without LLM rewriting` | `modular_enriched_hybrid.py` | Import test |
| 2 | `feat(benchmark): add no-LLM strategy to benchmark comparison` | `benchmark.py` | Import test |
| 3 | `docs(benchmark): add enriched_hybrid (no LLM) benchmark results` | `BENCHMARK_RESULTS.md`, `benchmark_results.json` | grep check |

---

## Success Criteria

### Verification Commands
```bash
# All strategies can be imported
python -c "from poc.modular_retrieval_pipeline.modular_enriched_hybrid import ModularEnrichedHybrid; print('OK')"
# Expected: OK

# No LLM in new strategy
grep -c "QueryRewriter\|rewrite_timeout\|call_llm" poc/modular_retrieval_pipeline/modular_enriched_hybrid.py || echo "0"
# Expected: 0

# Documentation has 3 strategies
grep -c "enriched_hybrid" poc/modular_retrieval_pipeline/BENCHMARK_RESULTS.md
# Expected: 3 (baseline, modular, modular no-LLM)
```

### Final Checklist
- [ ] `ModularEnrichedHybrid` class exists and imports cleanly
- [ ] No LLM references in new orchestrator
- [ ] Benchmark runs all 3 strategies
- [ ] `BENCHMARK_RESULTS.md` has new row with actual accuracy
- [ ] New section describes the no-LLM flow
