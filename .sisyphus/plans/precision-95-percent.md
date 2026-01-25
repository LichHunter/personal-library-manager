# Precision 95%+ Target: BMX Upgrade + LLM Query Rewriting

## Context

### Original Request
Achieve 95%+ coverage on RAG chunking benchmark by upgrading BM25 to BMX algorithm (testing Oracle's recommendation), then adding LLM query rewriting with Claude Haiku.

### Interview Summary
**Key Discussions**:
- User wants BMX first, then LLM: Test BMX upgrade before LLM integration
- Rationale: Empirical validation of Oracle's recommendation ("BM25 isn't the bottleneck")
- Current state: 83.0% coverage (44/53 facts) with weighted RRF + query expansion

**Research Findings**:
- BMX library: `pip install baguetter` (mixedbread-ai, 202 stars, Apache 2.0)
- API: `BMXSparseIndex.add_many(keys, values)` then `idx.search(query, top_k)`
- Paper: https://arxiv.org/abs/2408.06643 (Aug 2024)
- Existing LLM: `call_llm(prompt, model="claude-haiku")` ready in `enrichment/provider.py`

### Metis Review
**Identified Gaps** (addressed):
- API incompatibility: BMX returns top-k only, not full-corpus scores for RRF fusion
- Tokenization: BMX has built-in tokenizer, may differ from current `.lower().split()`
- Score normalization: BMX scores may be on different scale than BM25

**Mitigation**: Create new strategy file (don't modify existing), investigate API first, keep BM25 as fallback

---

## Work Objectives

### Core Objective
Improve RAG benchmark coverage from 83% to 95%+ by (1) upgrading BM25 to BMX and (2) adding LLM query rewriting, while validating Oracle's recommendation through empirical testing.

### Concrete Deliverables
- `retrieval/enriched_hybrid_bmx.py` - New strategy with BMX instead of BM25
- `retrieval/query_rewrite.py` - LLM query rewriting wrapper
- Benchmark results comparing BM25 vs BMX vs BMX+LLM
- Updated `PRECISION_RESEARCH.md` with findings

### Definition of Done
- [ ] BMX strategy registered and runnable: `python run_benchmark.py --strategies enriched_hybrid_bmx`
- [ ] LLM query rewriting integrated and tested
- [ ] Benchmark shows coverage >= 95% (50+/53 facts) OR documents why unreachable
- [ ] Side-by-side comparison table: BM25 vs BMX vs BMX+LLM

### Must Have
- Empirical comparison of BMX vs BM25 (validate Oracle's advice)
- Preserve existing `enriched_hybrid.py` as baseline
- Query expansion (`expand_query()`) remains functional
- Weighted RRF behavior preserved or improved

### Must NOT Have (Guardrails)
- Do NOT modify `run_benchmark.py` or evaluation logic
- Do NOT change ground truth or benchmark queries
- Do NOT tune RRF parameters in Phase 1 (isolate BMX impact)
- Do NOT combine Phase 1 and Phase 2 changes (measure separately)
- Do NOT remove existing BM25 code (keep as fallback)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (existing benchmark framework)
- **User wants tests**: Manual benchmark verification
- **Framework**: Python benchmark runner (`run_benchmark.py`)

### Verification Approach

Each phase includes benchmark-based verification:

**Phase 1 (BMX)**:
- Command: `python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_bmx --trace`
- Compare: Coverage delta vs baseline (83%)
- Log: Index time, query latency, memory usage

**Phase 2 (LLM Rewriting)**:
- Command: `python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_bmx_llm --trace`
- Compare: Coverage delta vs Phase 1 result
- Log: LLM latency per query, token usage

---

## Task Flow

```
Task 0 (Install baguetter)
    |
    v
Task 1 (API Investigation) --> Decision: Can we do RRF?
    |                              |
    v                              v (if no)
Task 2 (Create BMX strategy)   Task 2-alt (Use BMX hybrid)
    |
    v
Task 3 (Run Phase 1 benchmark)
    |
    v
Task 4 (Analyze BMX results) --> Decision: Continue to Phase 2?
    |
    v
Task 5 (Implement query rewriting)
    |
    v
Task 6 (Run Phase 2 benchmark)
    |
    v
Task 7 (Document results)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| Sequential | All | Each task depends on previous results |

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | 0 | Need baguetter installed to investigate API |
| 2 | 1 | Need API knowledge to implement correctly |
| 3 | 2 | Need strategy registered to run benchmark |
| 4 | 3 | Need benchmark results to analyze |
| 5 | 4 | Decision point based on Phase 1 results |
| 6 | 5 | Need query rewriting implemented |
| 7 | 6 | Need all results to document |

---

## TODOs

- [x] 0. Install baguetter and verify environment

  **What to do**:
  - Add baguetter to POC dependencies
  - Run `uv add baguetter` in `poc/chunking_benchmark_v2/`
  - Verify import works: `from baguetter.indices import BMXSparseIndex`

  **Must NOT do**:
  - Install globally (use POC venv only)
  - Modify other dependencies

  **Parallelizable**: NO (blocking for all subsequent tasks)

  **References**:
  - `poc/chunking_benchmark_v2/pyproject.toml` - Add baguetter dependency here
  - https://github.com/mixedbread-ai/baguetter - Official repo for version info

  **Acceptance Criteria**:
  - [ ] `uv add baguetter` completes without errors
  - [ ] Python REPL: `from baguetter.indices import BMXSparseIndex` → no ImportError
  - [ ] `BMXSparseIndex()` instantiates without error

  **Commit**: YES
  - Message: `chore(benchmark): add baguetter dependency for BMX`
  - Files: `poc/chunking_benchmark_v2/pyproject.toml`, `poc/chunking_benchmark_v2/uv.lock`

---

- [x] 1. Investigate BMX API for RRF compatibility

  **What to do**:
  - Create test script to understand BMX behavior
  - Test if BMX can return scores for ALL documents (not just top-k)
  - Check if BMX scores are comparable in scale to BM25 scores
  - Document findings for implementation task

  **Must NOT do**:
  - Start implementation before understanding API
  - Assume API matches our needs without testing

  **Parallelizable**: NO (depends on Task 0)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:232-244` - Current BM25 score retrieval pattern
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:246-267` - RRF fusion logic
  - https://github.com/mixedbread-ai/baguetter/blob/main/baguetter/indices/sparse/base.py - BMX base class (check for `get_scores` equivalent)

  **Test Script**:
  ```python
  from baguetter.indices import BMXSparseIndex
  
  idx = BMXSparseIndex()
  docs = ["doc one content", "doc two about cats", "doc three mentions dogs"]
  idx.add_many(["d1", "d2", "d3"], docs)
  
  # Test 1: What does search return?
  results = idx.search("cats", top_k=10)
  print(f"Type: {type(results)}")
  print(f"Keys: {results.keys}")
  print(f"Scores: {results.scores}")
  
  # Test 2: Can we get scores for ALL docs?
  results_all = idx.search("cats", top_k=len(docs))
  print(f"All scores: {results_all.scores}")
  
  # Test 3: What's the score scale?
  print(f"Score range: {min(results_all.scores)} to {max(results_all.scores)}")
  ```

  **Acceptance Criteria**:
  - [ ] Test script runs without errors
  - [ ] Document answer to: "Can BMX return scores for all documents?"
  - [ ] Document answer to: "What is BMX score scale vs BM25?"
  - [ ] Document answer to: "How to adapt RRF fusion for BMX?"

  **Commit**: NO (research task, findings go to PRECISION_RESEARCH.md)

---

- [ ] 2. Create BMX-based hybrid retrieval strategy

  **What to do**:
  - Copy `enriched_hybrid.py` to `enriched_hybrid_bmx.py`
  - Replace `from rank_bm25 import BM25Okapi` with `from baguetter.indices import BMXSparseIndex`
  - Adapt indexing: `idx.add_many(chunk_ids, enriched_contents)`
  - Adapt scoring: based on Task 1 findings
  - Preserve `expand_query()` function (copy it)
  - Preserve weighted RRF logic
  - Register in `retrieval/__init__.py`

  **Must NOT do**:
  - Modify original `enriched_hybrid.py`
  - Change RRF parameters (rrf_k, weights)
  - Remove query expansion functionality

  **Parallelizable**: NO (depends on Task 1)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py` - Source to copy and modify
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:21-41` - DOMAIN_EXPANSIONS dict (copy as-is)
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:44-77` - expand_query function (copy as-is)
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:80-175` - Index method (modify for BMX)
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:177-318` - Retrieve method (modify for BMX)
  - `poc/chunking_benchmark_v2/retrieval/__init__.py` - Register new strategy

  **Implementation Pattern**:
  ```python
  # In enriched_hybrid_bmx.py
  from baguetter.indices import BMXSparseIndex
  
  class EnrichedHybridBMXRetrieval(RetrievalStrategy, EmbedderMixin):
      def __init__(self, ...):
          # ... same as EnrichedHybridRetrieval
          self.bmx: Optional[BMXSparseIndex] = None
          self._chunk_id_to_idx: dict[str, int] = {}  # For score lookup
      
      def index(self, chunks, ...):
          # ... enrichment same as before
          self.bmx = BMXSparseIndex()
          chunk_ids = [chunk.id for chunk in chunks]
          self.bmx.add_many(chunk_ids, enriched_contents)
          self._chunk_id_to_idx = {cid: i for i, cid in enumerate(chunk_ids)}
      
      def retrieve(self, query, k=5):
          # ... based on Task 1 findings
  ```

  **Acceptance Criteria**:
  - [ ] File `retrieval/enriched_hybrid_bmx.py` exists
  - [ ] Class `EnrichedHybridBMXRetrieval` defined
  - [ ] BMX indexing works: no errors on `index()` call
  - [ ] BMX retrieval works: returns list of Chunk objects
  - [ ] Strategy registered: `from retrieval import RETRIEVAL_STRATEGIES; "enriched_hybrid_bmx" in RETRIEVAL_STRATEGIES`
  - [ ] Query expansion preserved: `expand_query("database stack")` returns expanded query

  **Commit**: YES
  - Message: `feat(benchmark): add BMX-based hybrid retrieval strategy`
  - Files: `retrieval/enriched_hybrid_bmx.py`, `retrieval/__init__.py`
  - Pre-commit: Import test passes

---

- [ ] 3. Run Phase 1 benchmark (BMX vs BM25)

  **What to do**:
  - Run baseline benchmark with existing BM25 strategy (for fresh comparison)
  - Run benchmark with new BMX strategy
  - Capture detailed metrics: coverage, latency, index time
  - Generate comparison table

  **Must NOT do**:
  - Modify benchmark runner
  - Change config parameters
  - Skip baseline comparison

  **Parallelizable**: NO (depends on Task 2)

  **References**:
  - `poc/chunking_benchmark_v2/run_benchmark.py` - Benchmark runner (read-only)
  - `poc/chunking_benchmark_v2/config_fast_enrichment.yaml` - Config to use
  - `poc/chunking_benchmark_v2/results/` - Results directory

  **Commands**:
  ```bash
  cd poc/chunking_benchmark_v2
  
  # Baseline (BM25)
  python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_fast --trace > results/phase1_bm25.log 2>&1
  
  # BMX
  python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_bmx --trace > results/phase1_bmx.log 2>&1
  ```

  **Acceptance Criteria**:
  - [ ] Both benchmark runs complete without errors
  - [ ] Results JSON files generated in `results/` directory
  - [ ] Coverage numbers extracted for both strategies
  - [ ] Comparison table created:
    | Metric | BM25 | BMX | Delta |
    |--------|------|-----|-------|
    | Coverage (original) | 83% | ? | ? |
    | Index time | ? | ? | ? |
    | Query latency | ? | ? | ? |

  **Commit**: NO (results are ephemeral, findings go to documentation)

---

- [ ] 4. Analyze Phase 1 results and decide on Phase 2

  **What to do**:
  - Compare BMX vs BM25 coverage
  - Document findings in PRECISION_RESEARCH.md
  - Determine if BMX improves coverage (validates/invalidates Oracle)
  - Decision: Proceed to Phase 2 (LLM) or stop if target reached

  **Must NOT do**:
  - Skip documentation of findings
  - Proceed to Phase 2 without recording Phase 1 results

  **Parallelizable**: NO (depends on Task 3)

  **References**:
  - `poc/chunking_benchmark_v2/PRECISION_RESEARCH.md:1528-1648` - Phase 2 research section (add results here)
  - Task 3 results

  **Decision Criteria**:
  - If BMX coverage >= 95%: STOP (target reached)
  - If BMX coverage >= 85%: Proceed to Phase 2 (LLM can close gap)
  - If BMX coverage < 83%: BMX regression - consider reverting, still try LLM
  - If BMX coverage == 83% (+/- 1%): Oracle was right, proceed to Phase 2

  **Acceptance Criteria**:
  - [ ] PRECISION_RESEARCH.md updated with Phase 1 results
  - [ ] Oracle validation documented: "Oracle was [correct/incorrect] because..."
  - [ ] Decision documented: Proceed to Phase 2 [YES/NO] because...

  **Commit**: YES
  - Message: `docs(benchmark): add BMX vs BM25 comparison results`
  - Files: `poc/chunking_benchmark_v2/PRECISION_RESEARCH.md`

---

- [ ] 5. Implement LLM query rewriting

  **What to do**:
  - Create `retrieval/query_rewrite.py` with rewriting logic
  - Use `call_llm(prompt, "claude-haiku")` from existing provider
  - Create prompt that converts user queries to documentation-aligned queries
  - Integrate into BMX strategy (or create combined strategy)
  - Handle edge cases: empty response, timeout, rate limits

  **Must NOT do**:
  - Use expensive models (stick to Haiku)
  - Add rewriting to original BM25 strategy (keep separate)
  - Make LLM calls blocking without timeout

  **Parallelizable**: NO (depends on Task 4 decision)

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/provider.py` - AnthropicProvider, call_llm function
  - `poc/chunking_benchmark_v2/PRECISION_RESEARCH.md:1577-1597` - Query rewrite prompt design
  - `poc/chunking_benchmark_v2/retrieval/hyde.py` - Existing LLM integration pattern (uses Ollama, adapt for Anthropic)

  **Implementation Pattern**:
  ```python
  # query_rewrite.py
  from enrichment.provider import call_llm
  
  QUERY_REWRITE_PROMPT = """Rewrite this user question as a direct documentation lookup query.
  Convert problem descriptions to feature questions.
  Expand abbreviations and technical jargon.
  
  User question: {query}
  
  Rewritten query (one line):"""
  
  def rewrite_query(query: str, timeout: float = 5.0) -> str:
      """Rewrite query using Claude Haiku for better retrieval."""
      try:
          rewritten = call_llm(
              QUERY_REWRITE_PROMPT.format(query=query),
              model="claude-haiku"
          )
          return rewritten.strip() if rewritten else query
      except Exception:
          return query  # Fallback to original on error
  ```

  **Acceptance Criteria**:
  - [ ] File `retrieval/query_rewrite.py` exists with `rewrite_query()` function
  - [ ] Rewriting works: `rewrite_query("Why can't I schedule every 30 seconds?")` returns documentation-style query
  - [ ] Fallback works: Returns original query on error/timeout
  - [ ] Integration with BMX strategy: creates `enriched_hybrid_bmx_llm` variant OR modifies existing
  - [ ] Latency acceptable: < 500ms per rewrite (Haiku is fast)

  **Commit**: YES
  - Message: `feat(benchmark): add LLM query rewriting with Claude Haiku`
  - Files: `retrieval/query_rewrite.py`, updated strategy file

---

- [ ] 6. Run Phase 2 benchmark (BMX + LLM rewriting)

  **What to do**:
  - Run benchmark with combined BMX + LLM rewriting
  - Compare to Phase 1 (BMX only) results
  - Measure LLM impact: coverage improvement, latency cost
  - Generate final comparison table

  **Must NOT do**:
  - Skip comparison with Phase 1
  - Ignore latency impact

  **Parallelizable**: NO (depends on Task 5)

  **References**:
  - Task 3 results (Phase 1 baseline)
  - `poc/chunking_benchmark_v2/run_benchmark.py`

  **Commands**:
  ```bash
  cd poc/chunking_benchmark_v2
  
  # BMX + LLM
  python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_bmx_llm --trace > results/phase2_bmx_llm.log 2>&1
  ```

  **Acceptance Criteria**:
  - [ ] Benchmark completes without errors
  - [ ] Coverage number extracted
  - [ ] Final comparison table:
    | Metric | BM25 | BMX | BMX+LLM | Target |
    |--------|------|-----|---------|--------|
    | Coverage | 83% | ? | ? | 95% |
    | Latency | ? | ? | ? | <700ms |

  **Commit**: NO (results are ephemeral)

---

- [ ] 7. Document final results and conclusions

  **What to do**:
  - Update PRECISION_RESEARCH.md with complete findings
  - Document Oracle validation conclusion
  - Create summary table of all approaches tested
  - Document remaining gaps (if any) and potential next steps
  - Update poc/chunking_benchmark_v2/README.md with new strategies

  **Must NOT do**:
  - Leave undocumented findings
  - Skip Oracle validation conclusion

  **Parallelizable**: NO (depends on Task 6)

  **References**:
  - `poc/chunking_benchmark_v2/PRECISION_RESEARCH.md` - Main documentation file
  - `poc/chunking_benchmark_v2/README.md` - Update with new strategies
  - All previous task results

  **Documentation Structure**:
  ```markdown
  ## Phase 3: BMX + LLM Integration Results (2026-01-25)
  
  ### Oracle Validation
  Oracle recommended skipping BMX. Empirical results:
  - BMX coverage: X% (delta from BM25: Y%)
  - Conclusion: Oracle was [correct/incorrect] because...
  
  ### Final Results
  | Strategy | Coverage | Latency | Notes |
  |----------|----------|---------|-------|
  | BM25 + expansion | 83% | ~7ms | Baseline |
  | BMX + expansion | ?% | ?ms | Phase 1 |
  | BMX + LLM rewrite | ?% | ?ms | Phase 2 |
  
  ### Conclusion
  [Did we reach 95%? Why/why not? What would be needed?]
  ```

  **Acceptance Criteria**:
  - [ ] PRECISION_RESEARCH.md has complete Phase 3 section
  - [ ] Oracle validation conclusion documented
  - [ ] Final comparison table complete
  - [ ] Remaining gaps documented (if target not reached)
  - [ ] README.md updated with new strategies

  **Commit**: YES
  - Message: `docs(benchmark): complete precision investigation with BMX and LLM results`
  - Files: `PRECISION_RESEARCH.md`, `README.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 0 | `chore(benchmark): add baguetter dependency for BMX` | pyproject.toml, uv.lock | import test |
| 2 | `feat(benchmark): add BMX-based hybrid retrieval strategy` | enriched_hybrid_bmx.py, __init__.py | import test |
| 4 | `docs(benchmark): add BMX vs BM25 comparison results` | PRECISION_RESEARCH.md | - |
| 5 | `feat(benchmark): add LLM query rewriting with Claude Haiku` | query_rewrite.py, strategy | import test |
| 7 | `docs(benchmark): complete precision investigation with BMX and LLM results` | PRECISION_RESEARCH.md, README.md | - |

---

## Success Criteria

### Verification Commands
```bash
# After Task 0: Verify baguetter installation
python -c "from baguetter.indices import BMXSparseIndex; print('OK')"

# After Task 2: Verify strategy registration
python -c "from retrieval import RETRIEVAL_STRATEGIES; assert 'enriched_hybrid_bmx' in RETRIEVAL_STRATEGIES; print('OK')"

# After Task 5: Verify query rewriting
python -c "from retrieval.query_rewrite import rewrite_query; print(rewrite_query('Why cant I schedule every 30 seconds?'))"

# Final: Run full benchmark
python run_benchmark.py --config config_fast_enrichment.yaml --strategies enriched_hybrid_bmx_llm
```

### Final Checklist
- [ ] BMX strategy implemented and working
- [ ] LLM query rewriting integrated
- [ ] Coverage >= 95% OR documented why unreachable
- [ ] Oracle recommendation validated with empirical data
- [ ] All findings documented in PRECISION_RESEARCH.md
- [ ] No regression in existing BM25 strategy (preserved as fallback)
