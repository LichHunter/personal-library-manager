# BMX Optimal Strategy Investigation

## Context

### Original Request
Investigate why BMX (entropy-weighted BM25 variant) failed catastrophically at 41.5% coverage vs BM25's 83.0%, and design strategies where BMX performs optimally.

### Interview Summary
**Key Discussions**:
- Root cause: Custom `DOMAIN_EXPANSIONS` applied BEFORE BMX conflicted with BMX's native entropy-weighted mechanisms
- BMX has built-in WQA (Weighted Query Augmentation) feature via `search_weighted()` that was never used
- Paper shows BMX+WQA achieves +5.1% over BM25 on BEIR benchmark
- User decided to test ALL THREE proposed strategies with FULL 120-query benchmark

**Research Findings**:
- BMX uses SAME tokenization as BM25 (whitespace + stemmer + stopwords) - difference is scoring formula
- BMX adds entropy-weighted similarity boost: `+ beta * E(qi) * S(Q,D)`
- `search_weighted()` API: `bmx.search_weighted([query1, query2], query_weights=[0.7, 0.3], top_k=k)`
- BMX excels on long documents (+39% on LoCo) and real-world queries (+16% on BRIGHT)

### Metis Review
**Identified Gaps** (addressed):
- WQA query generation strategy: Using single LLM-rewritten query with weights [0.7, 0.3]
- BMX-Semantic isolation: Clean BMX (no expansion) + clean embeddings (same enrichment)
- Baseline comparison: Include BM25 run for same-session comparison
- API verification: Added sanity check task before full benchmark

---

## Work Objectives

### Core Objective
Test three BMX strategies (BMX-Pure, BMX-WQA, BMX-Semantic) to determine optimal BMX usage for the personal library manager's retrieval system.

### Concrete Deliverables
- `poc/chunking_benchmark_v2/retrieval/bmx_pure.py` - Raw BMX, no query expansion
- `poc/chunking_benchmark_v2/retrieval/bmx_wqa.py` - BMX with Weighted Query Augmentation
- `poc/chunking_benchmark_v2/retrieval/bmx_semantic.py` - BMX + semantic embeddings, RRF fusion
- Updated `poc/chunking_benchmark_v2/retrieval/__init__.py` with new strategy registrations
- Benchmark results comparing all three strategies

### Definition of Done
- [x] All three strategies implemented and registered
- [x] Sanity check (5 queries) passes for each strategy
- [x] Full benchmark (120 queries) completed with `--trace` flag
- [x] Results documented with per-dimension breakdown

### Must Have
- NO `DOMAIN_EXPANSIONS` or `expand_query()` in any new BMX strategy
- Use baguetter's native `BMXSparseIndex` with default configuration
- BMX-WQA must use `search_weighted()` method
- All strategies must work with existing benchmark runner

### Must NOT Have (Guardrails)
- Must NOT modify existing `enriched_hybrid_bmx.py` (keep as reference for what NOT to do)
- Must NOT apply custom query expansion BEFORE BMX search
- Must NOT mix enrichment strategies between variants (isolate variables)
- Must NOT skip the `--trace` flag (need debugging output for analysis)
- Must NOT use different embedders between strategies (use BGE-base for all)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (benchmark runner exists)
- **User wants tests**: Manual verification via benchmark runs
- **Framework**: Benchmark-based verification (run_benchmark.py)

### Manual QA Approach

Each TODO includes benchmark verification:

**Verification Tool**: Python benchmark runner
**Procedure**: Run benchmark commands, verify output coverage percentages
**Evidence**: Terminal output with coverage metrics per dimension

---

## Task Flow

```
Task 0 (API Verify) → Task 1 (BMX-Pure) → Task 2 (BMX-WQA) → Task 3 (BMX-Semantic) → Task 4 (Register) → Task 5 (Sanity) → Task 6 (Full Benchmark) → Task 7 (Analysis)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| A | 1, 2, 3 | Independent strategy implementations after Task 0 |

| Task | Depends On | Reason |
|------|------------|--------|
| 1, 2, 3 | 0 | Need API verification before implementing |
| 4 | 1, 2, 3 | Need all strategies before registration |
| 5 | 4 | Need registration before sanity check |
| 6 | 5 | Need sanity check before full benchmark |
| 7 | 6 | Need benchmark results before analysis |

---

## TODOs

- [x] 0. Verify BMX `search_weighted()` API

  **What to do**:
  - Create quick test script to verify `search_weighted()` signature
  - Confirm it accepts `[query1, query2]` and `query_weights=[w1, w2]`
  - Test with a simple example to ensure it works as expected

  **Must NOT do**:
  - Do not create production code yet
  - Do not run full benchmark

  **Parallelizable**: NO (blocking - must complete before Tasks 1-3)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py:7` - Shows `from baguetter.indices import BMXSparseIndex` import pattern
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py:174-176` - Shows BMX initialization and `add_many()` pattern
  - BMX paper Section 3.2 - Describes WQA formula: `score(D, Q, Q^A) = score(D, Q) + sum(wi * score(D, Qi^A))`
  - Baguetter docs: `search_weighted(queries, query_weights, top_k)` method signature

  **Acceptance Criteria**:
  - [ ] Test script runs without error
  - [ ] `search_weighted()` accepts list of queries and list of weights
  - [ ] Returns results with scores
  - [ ] Terminal output confirms: "BMX search_weighted API verified"

  **Commit**: NO (temporary test script)

---

- [x] 1. Implement BMX-Pure Strategy

  **What to do**:
  - Create `poc/chunking_benchmark_v2/retrieval/bmx_pure.py`
  - Use raw BMX without ANY query expansion
  - Pass original query directly to `bmx.search()`
  - Implement hybrid fusion with semantic using RRF (same pattern as enriched_hybrid.py)
  - Use baguetter's built-in preprocessing (stemming, stopwords)

  **Must NOT do**:
  - NO `DOMAIN_EXPANSIONS` dictionary
  - NO `expand_query()` function
  - NO custom tokenization (let BMX handle it)

  **Parallelizable**: YES (with Tasks 2, 3 after Task 0)

  **References**:

  **Pattern References** (existing code to follow):
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:146-178` - Index method pattern: enrichment + embedding + BM25 setup
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:180-290` - Retrieve method pattern: semantic scores, BM25 scores, RRF fusion
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py:173-177` - BMX index initialization pattern: `BMXSparseIndex()` + `add_many()`

  **API/Type References** (contracts to implement against):
  - `poc/chunking_benchmark_v2/retrieval/base.py:RetrievalStrategy` - Base class to inherit
  - `poc/chunking_benchmark_v2/retrieval/base.py:EmbedderMixin` - Mixin for embedding functionality
  - `poc/chunking_benchmark_v2/strategies/__init__.py:Chunk` - Chunk type for input/output

  **Key Implementation Details**:
  - Class name: `BMXPureRetrieval`
  - Default name: `"bmx_pure"`
  - BMX search: `self.bmx.search(query, top_k=len(self.chunks))` - NO expansion
  - Semantic search: Same as enriched_hybrid.py (encode_query, dot product)
  - RRF fusion: Same weights as enriched_hybrid.py (1.0/1.0, rrf_k=60)

  **Acceptance Criteria**:
  - [ ] File created at `poc/chunking_benchmark_v2/retrieval/bmx_pure.py`
  - [ ] Class `BMXPureRetrieval` inherits from `RetrievalStrategy, EmbedderMixin`
  - [ ] NO `DOMAIN_EXPANSIONS` or `expand_query()` present in file
  - [ ] `retrieve()` method passes raw query to BMX
  - [ ] Python import check: `python -c "from retrieval.bmx_pure import BMXPureRetrieval; print('OK')"`

  **Commit**: NO (groups with Task 4)

---

- [x] 2. Implement BMX-WQA Strategy

  **What to do**:
  - Create `poc/chunking_benchmark_v2/retrieval/bmx_wqa.py`
  - Use Claude Haiku to generate ONE rewritten query (reuse `query_rewrite.py`)
  - Use `bmx.search_weighted([original, rewritten], query_weights=[0.7, 0.3], top_k=k)`
  - Implement hybrid fusion with semantic using RRF
  - Handle edge case: if LLM returns empty, fall back to original query only

  **Must NOT do**:
  - NO `DOMAIN_EXPANSIONS` dictionary
  - NO custom query expansion - only LLM rewriting
  - NO multiple LLM calls per query (keep latency reasonable)

  **Parallelizable**: YES (with Tasks 1, 3 after Task 0)

  **References**:

  **Pattern References** (existing code to follow):
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:188-202` - LLM query rewriting pattern: `rewrite_query()` call and timing
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:38-86` - `rewrite_query()` function: takes query, timeout, debug; returns rewritten or original
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py:173-177` - BMX index setup pattern

  **API/Type References**:
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:rewrite_query` - Function signature: `rewrite_query(query: str, timeout: float = 5.0, debug: bool = False) -> str`
  - BMX `search_weighted()`: `bmx.search_weighted(queries: list[str], query_weights: list[float], top_k: int)`

  **Key Implementation Details**:
  - Class name: `BMXWQARetrieval`
  - Default name: `"bmx_wqa"`
  - WQA weights: `[0.7, 0.3]` for `[original, rewritten]` (original gets higher weight)
  - Fallback: If `rewritten == original` or empty, use `bmx.search(original, top_k=k)` instead
  - Track `_rewrite_time_s` for latency analysis

  **Acceptance Criteria**:
  - [ ] File created at `poc/chunking_benchmark_v2/retrieval/bmx_wqa.py`
  - [ ] Uses `from .query_rewrite import rewrite_query`
  - [ ] Uses `bmx.search_weighted()` with weights [0.7, 0.3]
  - [ ] NO `DOMAIN_EXPANSIONS` or `expand_query()` present in file
  - [ ] Python import check: `python -c "from retrieval.bmx_wqa import BMXWQARetrieval; print('OK')"`

  **Commit**: NO (groups with Task 4)

---

- [x] 3. Implement BMX-Semantic Strategy

  **What to do**:
  - Create `poc/chunking_benchmark_v2/retrieval/bmx_semantic.py`
  - Use clean BMX (no expansion) for sparse retrieval
  - Use clean semantic embeddings (same enrichment as other strategies)
  - Apply RRF fusion with equal weights (1.0/1.0)
  - This tests if clean BMX + clean semantic produces better fusion than our broken implementation

  **Must NOT do**:
  - NO `DOMAIN_EXPANSIONS` dictionary
  - NO query expansion of any kind
  - NO different enrichment strategy (use same FastEnricher)

  **Parallelizable**: YES (with Tasks 1, 2 after Task 0)

  **References**:

  **Pattern References** (existing code to follow):
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:146-290` - Full hybrid implementation pattern (this is the gold standard)
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_bmx.py:250-286` - RRF fusion calculation pattern

  **Key Implementation Details**:
  - Class name: `BMXSemanticRetrieval`
  - Default name: `"bmx_semantic"`
  - This is essentially `enriched_hybrid.py` but with BMX instead of BM25Okapi
  - RRF weights: Equal (1.0/1.0) - no boosting
  - Purpose: Isolate whether BMX vs BM25 makes a difference when both are used cleanly

  **Acceptance Criteria**:
  - [ ] File created at `poc/chunking_benchmark_v2/retrieval/bmx_semantic.py`
  - [ ] Uses `BMXSparseIndex` from baguetter
  - [ ] Uses same `FastEnricher` pattern as enriched_hybrid.py
  - [ ] NO `DOMAIN_EXPANSIONS` or `expand_query()` present in file
  - [ ] RRF fusion uses equal weights (1.0/1.0)
  - [ ] Python import check: `python -c "from retrieval.bmx_semantic import BMXSemanticRetrieval; print('OK')"`

  **Commit**: NO (groups with Task 4)

---

- [x] 4. Register New Strategies in __init__.py

  **What to do**:
  - Add imports for all three new strategy classes
  - Add entries to `RETRIEVAL_STRATEGIES` dict
  - Add to `__all__` list

  **Must NOT do**:
  - Do not remove existing strategies
  - Do not modify strategy class names

  **Parallelizable**: NO (depends on 1, 2, 3)

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/retrieval/__init__.py:25-26` - Import pattern: `from .enriched_hybrid_bmx import EnrichedHybridBMXRetrieval`
  - `poc/chunking_benchmark_v2/retrieval/__init__.py:44-46` - Registration pattern: `"strategy_name": StrategyClass,`
  - `poc/chunking_benchmark_v2/retrieval/__init__.py:110-111` - __all__ pattern

  **Exact Changes**:
  ```python
  # Add imports after line 26:
  from .bmx_pure import BMXPureRetrieval
  from .bmx_wqa import BMXWQARetrieval
  from .bmx_semantic import BMXSemanticRetrieval

  # Add to RETRIEVAL_STRATEGIES after line 46:
  "bmx_pure": BMXPureRetrieval,
  "bmx_wqa": BMXWQARetrieval,
  "bmx_semantic": BMXSemanticRetrieval,

  # Add to __all__ after line 111:
  "BMXPureRetrieval",
  "BMXWQARetrieval",
  "BMXSemanticRetrieval",
  ```

  **Acceptance Criteria**:
  - [ ] All three imports added
  - [ ] All three strategies registered in `RETRIEVAL_STRATEGIES`
  - [ ] All three class names in `__all__`
  - [ ] Python check: `python -c "from retrieval import BMXPureRetrieval, BMXWQARetrieval, BMXSemanticRetrieval; print('OK')"`

  **Commit**: YES
  - Message: `feat(poc): add three BMX retrieval strategies (pure, wqa, semantic)`
  - Files: `poc/chunking_benchmark_v2/retrieval/bmx_pure.py`, `poc/chunking_benchmark_v2/retrieval/bmx_wqa.py`, `poc/chunking_benchmark_v2/retrieval/bmx_semantic.py`, `poc/chunking_benchmark_v2/retrieval/__init__.py`
  - Pre-commit: `python -c "from retrieval import BMXPureRetrieval, BMXWQARetrieval, BMXSemanticRetrieval"`

---

- [x] 5. Run Sanity Check (5 Queries Per Strategy)

  **What to do**:
  - Run quick benchmark with 5 queries to verify strategies don't crash
  - Use `--trace` flag for debugging output
  - Verify each strategy produces reasonable results (not 0% or 100%)

  **Must NOT do**:
  - Do not run full 120-query benchmark yet
  - Do not proceed if any strategy crashes

  **Parallelizable**: NO (depends on 4)

  **References**:
  - `poc/chunking_benchmark_v2/README.md` - Shows benchmark command pattern
  - `poc/chunking_benchmark_v2/run_benchmark.py` - Benchmark runner

  **Commands to Run**:
  ```bash
  cd /home/fujin/Code/personal-library-manager
  nix develop
  cd poc/chunking_benchmark_v2
  source .venv/bin/activate

  # Quick sanity check - 5 queries each
  python run_benchmark.py --strategies bmx_pure --trace --limit 5
  python run_benchmark.py --strategies bmx_wqa --trace --limit 5
  python run_benchmark.py --strategies bmx_semantic --trace --limit 5
  ```

  **Acceptance Criteria**:
  - [ ] `bmx_pure` completes without crash, coverage > 0%
  - [ ] `bmx_wqa` completes without crash, coverage > 0%
  - [ ] `bmx_semantic` completes without crash, coverage > 0%
  - [ ] No Python exceptions in output
  - [ ] All three strategies produce ranked results

  **Commit**: NO (diagnostic step)

---

- [x] 6. Run Full Benchmark (120 Queries)

  **What to do**:
  - Run full benchmark on all three new strategies
  - Include `enriched_hybrid` (BM25 baseline) for same-session comparison
  - Use `--trace` flag for detailed logging
  - Capture per-dimension results (original, synonym, problem, casual, contextual, negation)

  **Must NOT do**:
  - Do not skip any query dimension
  - Do not run without `--trace` flag

  **Parallelizable**: NO (depends on 5)

  **References**:
  - `poc/chunking_benchmark_v2/README.md` - Shows expected result format
  - `poc/chunking_benchmark_v2/corpus/ground_truth_realistic.json` - Contains 20 queries x 6 dimensions = 120 total

  **Commands to Run**:
  ```bash
  cd /home/fujin/Code/personal-library-manager/poc/chunking_benchmark_v2
  source .venv/bin/activate

  # Full benchmark - all BMX strategies + BM25 baseline
  python run_benchmark.py --strategies bmx_pure,bmx_wqa,bmx_semantic,enriched_hybrid --trace
  ```

  **Expected Output Format**:
  ```
  | Strategy | Original | Synonym | Problem | Casual | Contextual | Negation | Latency |
  |----------|----------|---------|---------|--------|------------|----------|---------|
  | bmx_pure | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | ~XXms |
  | bmx_wqa | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | ~XXms |
  | bmx_semantic | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | XX.X% | ~XXms |
  | enriched_hybrid | 83.0% | 66.0% | 64.2% | 71.7% | 69.8% | 52.8% | ~15ms |
  ```

  **Acceptance Criteria**:
  - [ ] All 4 strategies complete full 120 queries
  - [ ] Results table captured with per-dimension breakdown
  - [ ] Latency numbers recorded for each strategy
  - [ ] No crashes or exceptions during benchmark

  **Commit**: NO (results only)

---

- [x] 7. Analyze Results and Document Findings

  **What to do**:
  - Compare BMX strategies against baseline (BM25 83.0%) and best (LLM 88.7%)
  - Determine if BMX-WQA reaches target of 85%+
  - Document which strategy (if any) should replace current best
  - Update `poc/chunking_benchmark_v2/README.md` with findings

  **Must NOT do**:
  - Do not recommend BMX if it doesn't beat or match BM25 baseline
  - Do not update README without actual benchmark data

  **Parallelizable**: NO (depends on 6)

  **References**:
  - `poc/chunking_benchmark_v2/README.md:17-25` - Current results table to update
  - Draft file: `.sisyphus/drafts/bmx-optimal-strategy.md:200-209` - Success criteria

  **Analysis Questions**:
  1. Does BMX-Pure beat BM25 baseline (83.0%)? If yes, our previous failure was due to expansion conflict.
  2. Does BMX-WQA reach 85%+? If yes, BMX is viable with proper WQA usage.
  3. Does BMX-Semantic beat BMX-Pure? If yes, hybrid fusion still helps.
  4. Is BMX-WQA latency acceptable (~960ms like enriched_hybrid_llm)?

  **Decision Matrix**:
  | BMX-WQA Coverage | Recommendation |
  |------------------|----------------|
  | >= 85% | BMX-WQA is viable alternative to enriched_hybrid_llm |
  | 80-85% | BMX works but not better than current best |
  | < 80% | BMX unsuitable for this corpus size (51 chunks) |

  **Acceptance Criteria**:
  - [ ] Results table added to README with all 3 BMX strategies
  - [ ] Key findings section documents comparison to baselines
  - [ ] Clear recommendation for or against BMX usage
  - [ ] Root cause of original failure confirmed or refuted

  **Commit**: YES
  - Message: `docs(poc): add BMX strategy benchmark results and analysis`
  - Files: `poc/chunking_benchmark_v2/README.md`
  - Pre-commit: None (documentation only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 4 | `feat(poc): add three BMX retrieval strategies (pure, wqa, semantic)` | bmx_pure.py, bmx_wqa.py, bmx_semantic.py, __init__.py | Python import check |
| 7 | `docs(poc): add BMX strategy benchmark results and analysis` | README.md | None |

---

## Success Criteria

### Verification Commands
```bash
# After Task 4:
cd poc/chunking_benchmark_v2
python -c "from retrieval import BMXPureRetrieval, BMXWQARetrieval, BMXSemanticRetrieval; print('All strategies imported OK')"

# After Task 6:
# Verify results file exists
ls -la results/

# Check for expected coverage format in output
```

### Final Checklist
- [x] All "Must Have" present: 3 strategies implemented, no DOMAIN_EXPANSIONS
- [x] All "Must NOT Have" absent: existing BMX file untouched, no custom expansion
- [x] Full benchmark completed with trace output
- [x] Results documented with clear recommendation
