# Chunking Benchmark V2 Precision Investigation - Deep Dive

## Context

### Original Request
Investigate why chunking benchmark v2 loses approximately 25% precision (measured as key facts coverage). The goal is NOT just documentation - it's to **close the precision gap and achieve 95%+ coverage** (ideally 98-100%).

### Updated Requirements (from user clarification)
1. **Root cause discovery**: Find WHERE and WHY precision is lost
2. **Solution testing**: Don't just research - actually TEST solutions
3. **Oracle consultation**: At each major decision point
4. **Data trail**: Maintain research file for context continuity across sessions
5. **Target**: 95%+ precision (ideally 98-100%)
6. **No shortcuts**: Test all solutions on realistic corpus with human queries
7. **End state**: Detailed root cause + tested solutions + BEST WORKING solution

### Interview Summary
**Key Discussions**:
- Best strategy (enriched_hybrid_fast) achieves 77.4% coverage, missing ~22.6% of key facts
- Worst degradation: negation queries (-26.5%), problem queries (-22.2%)
- Hypothesis 1 (chunk boundaries) already DISPROVEN - facts exist in chunks but wrong chunks retrieved
- Primary issue: RETRIEVAL RANKING - correct chunks not in top-5
- User wants FULL investigation with SOLUTION IMPLEMENTATION
- User wants ORACLE consultation at each decision point
- User wants testing on realistic corpus (20 queries, 53 facts, 5 documents)

**Research Findings**:
- Logger infrastructure exists with TRACE level support
- FastEnricher uses YAKE (keywords) + spaCy (NER)
- EnrichedHybridRetrieval combines BM25 + semantic with RRF
- Ground truth: 20 queries with 53 total key facts
- Key missed facts: `TTL: 1 hour` (0%), `RPO/RTO` (0% on most dimensions)
- Negation queries lose 26.5%, problem queries lose 22.2%

### Success Criteria
- **Minimum**: 95% key facts coverage on original queries
- **Target**: 98-100% coverage on original queries
- **Must maintain**: Reasonable performance on variant queries (synonym, problem, casual, contextual, negation)

---

## Work Objectives

### Core Objective
Achieve 95%+ precision (key facts coverage) through systematic investigation, hypothesis testing, and solution implementation.

### Concrete Deliverables
1. Root cause analysis with evidence for each precision loss source
2. Tested solutions with measured impact
3. Final implementation achieving 95%+ precision
4. Complete documentation in PRECISION_RESEARCH.md

### Definition of Done
- [ ] Achieved 95%+ coverage on original queries (verified by benchmark)
- [ ] Root causes documented with evidence
- [ ] All tested solutions documented with results
- [ ] Best solution implemented and validated
- [ ] PRECISION_RESEARCH.md contains complete investigation trail

### Must Have
- Oracle consultation at each major decision point
- Testing on full realistic corpus (all 20 queries, 53 facts)
- Evidence-based decision making
- Context preservation in PRECISION_RESEARCH.md

### Must NOT Have (Guardrails)
- DO NOT stop at documentation without testing solutions
- DO NOT accept solutions without benchmark validation
- DO NOT skip Oracle consultation on major decisions
- DO NOT compromise on testing thoroughness

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Framework**: benchmark run with exact_match metric
- **Corpus**: realistic_documents (5 docs, ~3400 words each)
- **Queries**: ground_truth_realistic.json (20 queries, 53 facts)
- **Success**: 95%+ coverage = 50+ facts found out of 53

### Benchmark Command
```bash
cd poc/chunking_benchmark_v2
python run_benchmark.py --config [config_file] --trace
# Check results/[timestamp]/summary.json for coverage
```

---

## Task Flow

```
Phase 1: DIAGNOSIS
  Task 1 (Logging) --> Task 2 (Baseline) --> Task 3 (Root Cause)
                                                    |
                                                    v
                                           [Oracle: Confirm diagnosis]
                                                    |
                                                    v
Phase 2: SOLUTION EXPLORATION
  Task 4 (Research) --> Task 5 (Hypothesis) --> [Oracle: Prioritize]
                                                    |
                                                    v
Phase 3: SOLUTION TESTING
  Task 6 (Implement & Test) --> Task 7 (Iterate) --> ... --> [Oracle: Validate]
                                                               |
                                                               v
Phase 4: FINALIZATION
  Task 8 (Best Solution) --> Task 9 (Documentation)
```

## Parallelization

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | - | First task |
| 2 | 1 | Need logging |
| 3 | 2 | Need baseline data |
| 4 | 3 | Need root cause confirmed |
| 5 | 4 | Need research findings |
| 6+ | 5 | Iterative testing |

---

## TODOs

### PHASE 1: DIAGNOSIS

- [x] 1. Setup Logging Infrastructure

  **What to do**:
  - Add comprehensive logging to retrieval path
  - Log: query -> BM25 scores -> semantic scores -> RRF calculation -> final ranking
  - Log: for each key fact, which chunk contains it and what rank that chunk got
  - Enable with `--trace` flag

  **Must NOT do**:
  - Change retrieval logic yet
  - Skip any logging points

  **Parallelizable**: NO (first task)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid.py:50-54` - Existing trace pattern
  - `poc/chunking_benchmark_v2/run_benchmark.py:185-222` - evaluate_single_query
  - `poc/chunking_benchmark_v2/logger.py` - Logger infrastructure

  **Logging Points**:
  1. `enriched_hybrid.py:retrieve()`:
     - Log query text
     - Log enrichment prefix for query (if any)
     - Log top-10 BM25 scores with chunk IDs and content preview
     - Log top-10 semantic scores with chunk IDs
     - Log RRF calculation: sem_rank + bm25_rank -> rrf_score
     - Log final top-k with dominant signal (BM25 vs semantic)
  
  2. `run_benchmark.py:evaluate_single_query()`:
     - For each key fact:
       - Log fact text
       - Log if found in retrieved chunks
       - If NOT found: find which chunk contains it and log its rank

  **Acceptance Criteria**:
  - [ ] `python run_benchmark.py --config config_fast_enrichment.yaml --trace --dry-run` works
  - [ ] TRACE output shows all logging points listed above

  **Commit**: YES
  - Message: `feat(benchmark): add comprehensive trace logging for precision investigation`
  - Files: `retrieval/enriched_hybrid.py`, `run_benchmark.py`

---

- [x] 2. Run Baseline Benchmark with Full Logging

  **What to do**:
  - Clear enrichment cache
  - Run benchmark with trace logging
  - Save results and logs
  - Record baseline metrics

  **Acceptance Criteria**:
  - [ ] `rm -rf enrichment_cache/*`
  - [ ] `python run_benchmark.py --config config_fast_enrichment.yaml --trace 2>&1 | tee baseline_investigation.log`
  - [ ] Results saved to `results/[timestamp]/`
  - [ ] Baseline coverage recorded: enriched_hybrid_fast original = X%

  **Commit**: NO (data only)

---

- [x] 3. Root Cause Analysis - Identify ALL Precision Loss Sources

  **What to do**:
  - Parse logs to identify EVERY missed fact
  - For each missed fact, determine:
    - Which chunk contains it (grep through enriched content)
    - What rank that chunk got (from trace logs)
    - WHY it ranked low (BM25 vs semantic disagreement? Both low?)
  - Categorize root causes:
    - BM25 failure (keywords don't match)
    - Semantic failure (embedding mismatch)
    - Both fail (fundamental mismatch)
    - RRF combination issue
  - Identify patterns (query types that fail, fact types that fail)

  **Must NOT do**:
  - Implement solutions yet
  - Skip any missed facts

  **References**:
  - `poc/chunking_benchmark_v2/corpus/ground_truth_realistic.json` - All facts
  - `baseline_investigation.log` - Trace output
  - `results/[latest]/benchmark_results.json` - Per-query results

  **Analysis Template** (for each missed fact):
  ```
  Fact: "TTL: 1 hour"
  Query: realistic_013 (original)
  Chunk containing fact: arch_overview_fix_6
  Chunk rank: 15th (not in top-5)
  BM25 rank: 12th, score: 2.3
  Semantic rank: 18th, score: 0.72
  Root cause: BOTH FAIL - query terms "Redis cache TTL" don't match chunk keywords
  ```

  **Acceptance Criteria**:
  - [ ] All 12-13 missed facts analyzed (53 * 22.6% = ~12)
  - [ ] Root cause categorized for each
  - [ ] Pattern summary documented
  - [ ] Updated PRECISION_RESEARCH.md with Finding 3: Root Cause Analysis

  **Commit**: YES
  - Message: `docs(benchmark): add root cause analysis for precision loss`
  - Files: `PRECISION_RESEARCH.md`

---

- [ ] 3.1. ORACLE CONSULTATION - Validate Root Cause Analysis

  **What to do**:
  - Present root cause findings to Oracle
  - Ask Oracle to validate diagnosis
  - Ask Oracle for solution prioritization

  **Prompt for Oracle**:
  ```
  I've completed root cause analysis for the 25% precision loss in our RAG benchmark.

  FINDINGS:
  [Insert root cause summary from Task 3]

  QUESTIONS:
  1. Does this diagnosis make sense? Am I missing any root causes?
  2. Which root cause should I address first for maximum impact?
  3. What solutions would you recommend trying?
  ```

  **Acceptance Criteria**:
  - [ ] Oracle validates or corrects diagnosis
  - [ ] Oracle provides prioritized solution list
  - [ ] Oracle feedback recorded in PRECISION_RESEARCH.md

  **Commit**: YES
  - Message: `docs(benchmark): add oracle feedback on root cause analysis`
  - Files: `PRECISION_RESEARCH.md`

---

### PHASE 2: SOLUTION EXPLORATION

- [ ] 4. Research Potential Solutions

  **What to do**:
  - Based on root causes, research targeted solutions
  - For each root cause category, find 2-3 potential fixes
  - Document with expected impact and implementation complexity

  **Research Areas** (adjust based on root causes found):
  - BM25 failures: Query expansion, synonym injection, better tokenization
  - Semantic failures: Query embedding enhancement, late interaction
  - RRF issues: Adjust k parameter, weight tuning, alternative fusion
  - Enrichment issues: Better keywords, contextual enrichment, hypothetical document embedding

  **References** (for web search):
  - "RAG hybrid retrieval precision" 
  - "BM25 query expansion techniques"
  - "Semantic search negative queries"
  - "Reciprocal rank fusion optimization"

  **Acceptance Criteria**:
  - [ ] 5+ potential solutions documented
  - [ ] Each solution has: description, expected impact, complexity, relevant root cause
  - [ ] Solutions ranked by expected impact

  **Commit**: YES
  - Message: `docs(benchmark): add solution research findings`
  - Files: `PRECISION_RESEARCH.md`

---

- [ ] 5. Formulate Testable Hypotheses

  **What to do**:
  - Convert solutions to testable hypotheses
  - Define test procedure for each
  - Estimate expected improvement

  **Hypothesis Template**:
  ```
  Hypothesis: [Solution X] will improve coverage by [Y%]
  Root cause addressed: [Which root cause]
  Test procedure: [Specific code change and benchmark config]
  Success criteria: Coverage >= [target]%
  ```

  **Acceptance Criteria**:
  - [ ] 3-5 hypotheses formulated
  - [ ] Each has clear test procedure
  - [ ] Prioritized by expected impact

  **Commit**: YES
  - Message: `docs(benchmark): add testable hypotheses for precision improvement`
  - Files: `PRECISION_RESEARCH.md`

---

- [ ] 5.1. ORACLE CONSULTATION - Validate Solution Strategy

  **What to do**:
  - Present hypotheses to Oracle
  - Get validation on testing order
  - Get implementation guidance

  **Prompt for Oracle**:
  ```
  I've formulated the following hypotheses to address precision loss:

  [Insert hypotheses from Task 5]

  QUESTIONS:
  1. Is this the right testing order?
  2. Are there any solutions I should add or remove?
  3. What's the best implementation approach for hypothesis #1?
  ```

  **Acceptance Criteria**:
  - [ ] Oracle validates testing order
  - [ ] Oracle provides implementation guidance
  - [ ] Oracle feedback recorded

  **Commit**: YES
  - Message: `docs(benchmark): add oracle feedback on solution strategy`
  - Files: `PRECISION_RESEARCH.md`

---

### PHASE 3: SOLUTION TESTING (ITERATIVE)

- [ ] 6. Implement and Test Solution #1

  **What to do**:
  - Implement first hypothesis solution
  - Run full benchmark on realistic corpus
  - Record results
  - Compare to baseline

  **Implementation Notes**:
  - Create backup of modified files
  - Use feature flag or separate config if possible
  - Document exact changes made

  **Acceptance Criteria**:
  - [ ] Solution implemented
  - [ ] Benchmark run: `python run_benchmark.py --config config_[solution1].yaml --trace`
  - [ ] Results recorded:
    - Coverage: X% (baseline was Y%)
    - Change: +/- Z%
  - [ ] If improved, keep changes; if not, revert
  - [ ] Finding documented in PRECISION_RESEARCH.md

  **Commit**: YES (if solution improves precision)
  - Message: `feat(benchmark): implement [solution name] - coverage now X%`
  - Files: [depends on solution]

---

- [ ] 7. Iterate: Test Remaining Solutions

  **What to do**:
  - Repeat Task 6 for each remaining hypothesis
  - Stack improvements if they're additive
  - Track cumulative progress toward 95% target

  **Progress Tracking**:
  ```
  Baseline: 77.4%
  After Solution 1: X%
  After Solution 2: Y%
  ...
  Current: Z% (target: 95%+)
  ```

  **When to Stop**:
  - Achieved 95%+ coverage, OR
  - Tested all hypotheses, OR
  - Oracle advises different approach

  **Acceptance Criteria**:
  - [ ] All hypotheses tested
  - [ ] Progress tracked for each
  - [ ] Best combination identified

  **Commit**: YES (for each successful improvement)

---

- [ ] 7.1. ORACLE CONSULTATION - Mid-Investigation Check

  **What to do**:
  - Report progress to Oracle
  - Get guidance on next steps
  - Decide if more solutions needed

  **Prompt for Oracle**:
  ```
  PROGRESS REPORT:
  - Baseline: 77.4% coverage
  - Current: X% coverage
  - Target: 95%+
  
  Solutions tested:
  [List with results]

  QUESTIONS:
  1. Should I continue with current approach?
  2. Are there other solutions to try?
  3. If we can't reach 95%, what's acceptable?
  ```

  **Acceptance Criteria**:
  - [ ] Oracle provides guidance on continuation
  - [ ] Next steps clear

---

### PHASE 4: FINALIZATION

- [ ] 8. Finalize Best Solution

  **What to do**:
  - Identify best solution or combination
  - Ensure it's properly implemented
  - Run final validation benchmark
  - Verify on ALL query dimensions (not just original)

  **Final Validation**:
  ```bash
  python run_benchmark.py --config config_final.yaml --trace
  ```

  **Acceptance Criteria**:
  - [ ] Final coverage on original queries: X% (should be 95%+)
  - [ ] Coverage on all dimensions documented:
    | Dimension | Before | After | Change |
    |-----------|--------|-------|--------|
    | original | 77.4% | X% | +Y% |
    | synonym | 71.7% | X% | +Y% |
    | problem | 62.3% | X% | +Y% |
    | casual | 64.2% | X% | +Y% |
    | contextual | 67.9% | X% | +Y% |
    | negation | 54.7% | X% | +Y% |

  **Commit**: YES
  - Message: `feat(benchmark): final precision solution - X% coverage achieved`
  - Files: [all modified files]

---

- [ ] 8.1. ORACLE CONSULTATION - Final Validation

  **What to do**:
  - Present final results to Oracle
  - Get sign-off on solution
  - Get recommendations for production use

  **Prompt for Oracle**:
  ```
  FINAL RESULTS:
  - Achieved X% coverage on original queries
  - Solution: [description]
  - Trade-offs: [any negatives]

  QUESTIONS:
  1. Is this solution acceptable for production?
  2. Any concerns about the approach?
  3. Recommendations for monitoring/maintenance?
  ```

  **Acceptance Criteria**:
  - [ ] Oracle approves solution
  - [ ] Recommendations documented

---

- [ ] 9. Complete Documentation

  **What to do**:
  - Update PRECISION_RESEARCH.md with complete investigation
  - Document:
    - Full root cause analysis
    - All solutions tested with results
    - Final solution details
    - Implementation guide
    - Future recommendations

  **Documentation Structure**:
  ```markdown
  ## Investigation Summary
  - Started: [date]
  - Completed: [date]
  - Baseline: 77.4%
  - Final: X%
  - Improvement: +Y%

  ## Root Causes Identified
  [From Task 3]

  ## Solutions Tested
  | Solution | Result | Impact |
  |----------|--------|--------|
  | ... | ... | ... |

  ## Final Solution
  [Detailed description]

  ## Implementation Guide
  [How to apply the solution]

  ## Lessons Learned
  [What worked, what didn't]
  ```

  **Acceptance Criteria**:
  - [ ] PRECISION_RESEARCH.md fully updated
  - [ ] All findings documented
  - [ ] Reproducible by future investigators

  **Commit**: YES
  - Message: `docs(benchmark): complete precision investigation documentation`
  - Files: `PRECISION_RESEARCH.md`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 1 | `feat(benchmark): add comprehensive trace logging` | retrieval code |
| 3 | `docs(benchmark): add root cause analysis` | PRECISION_RESEARCH.md |
| 3.1 | `docs(benchmark): add oracle feedback on diagnosis` | PRECISION_RESEARCH.md |
| 4 | `docs(benchmark): add solution research` | PRECISION_RESEARCH.md |
| 5 | `docs(benchmark): add testable hypotheses` | PRECISION_RESEARCH.md |
| 6+ | `feat(benchmark): implement [solution]` | various |
| 8 | `feat(benchmark): final solution` | various |
| 9 | `docs(benchmark): complete documentation` | PRECISION_RESEARCH.md |

---

## Success Criteria

### Verification Commands
```bash
# Final benchmark validation
python run_benchmark.py --config config_final.yaml --trace

# Check coverage
jq '.best_configurations | to_entries[] | select(.key | contains("enriched")) | .value.coverage' results/[latest]/summary.json
# Expected: >= 0.95 (95%+)

# Check documentation completeness
grep -c "## " PRECISION_RESEARCH.md
# Expected: More sections than before
```

### Final Checklist
- [ ] Achieved 95%+ coverage on original queries
- [ ] Root causes documented with evidence
- [ ] All solutions tested and documented
- [ ] Best solution implemented and validated
- [ ] Oracle consultations recorded
- [ ] PRECISION_RESEARCH.md complete for future reference

---

## Context Preservation

**IMPORTANT**: This investigation may span multiple sessions. Maintain context by:

1. **Always update PRECISION_RESEARCH.md** after each task
2. **Include session ID** in findings
3. **Document current state** at end of each session:
   - Current task number
   - Current coverage level
   - Next steps
4. **New sessions start by reading** PRECISION_RESEARCH.md

**Session End Protocol**:
```markdown
## Session End: [timestamp]
- Completed tasks: [list]
- Current coverage: X%
- Current task: [number]
- Next steps: [what to do next]
- Blockers: [if any]
```

---

## Estimated Time

| Phase | Tasks | Estimate |
|-------|-------|----------|
| Phase 1: Diagnosis | 1-3.1 | 4-6 hours |
| Phase 2: Solution Exploration | 4-5.1 | 3-4 hours |
| Phase 3: Solution Testing | 6-7.1 | 6-10 hours (iterative) |
| Phase 4: Finalization | 8-9 | 2-3 hours |
| **Total** | | **15-23 hours** |

Note: This is an iterative investigation. Time depends on how quickly we find solutions that work.
