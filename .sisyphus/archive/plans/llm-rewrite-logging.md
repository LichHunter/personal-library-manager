# LLM Rewrite Logging - Capture Query Transformations

## TL;DR

> **Quick Summary**: Enable debug logging in needle-haystack benchmark to capture how LLM rewrites queries, then analyze the transformations on failing questions.
> 
> **Deliverables**:
> - Benchmark run with rewrite logging enabled
> - Log file with original → rewritten query pairs
> 
> **Estimated Effort**: Quick
> **Parallel Execution**: NO - sequential

---

## Context

### Original Request
Log how LLM rewrites queries during benchmark to understand if rewrites help or hurt retrieval.

### Key Finding
The benchmark already has debug logging infrastructure - just needs `debug=True` flag.

When enabled, logs show:
```
[enriched-hybrid-llm] ORIGINAL_QUERY: {query}
[enriched-hybrid-llm] REWRITTEN_QUERY: {rewritten_query}
```

---

## Work Objectives

### Core Objective
Capture LLM query rewrites during benchmark run on adversarial questions to analyze transformation quality.

### Concrete Deliverables
- `rewrite_log.txt` - Full benchmark output with rewrite logging
- Updated `results/needle_questions_adversarial_retrieval.json` - Benchmark results

### Must Have
- Debug logging enabled (`debug=True`)
- Run on adversarial questions (20 questions, known 65% pass rate)
- Capture both ORIGINAL and REWRITTEN queries

### Must NOT Have
- No changes to rewrite logic
- No new files beyond log output

---

## TODOs

- [x] 1. Enable debug logging in benchmark

  **What to do**:
  - Edit `poc/chunking_benchmark_v2/benchmark_needle_haystack.py` line 246
  - Change: `strategy = create_retrieval_strategy("enriched_hybrid_llm")`
  - To: `strategy = create_retrieval_strategy("enriched_hybrid_llm", debug=True)`

  **Acceptance Criteria**:
  - [x] Line 246 contains `debug=True`

  **Commit**: NO (temporary change for investigation)

---

- [x] 2. Run benchmark on adversarial questions with logging

  **What to do**:
  ```bash
  cd poc/chunking_benchmark_v2
  python benchmark_needle_haystack.py --questions corpus/needle_questions_adversarial.json --run-benchmark 2>&1 | tee rewrite_log.txt
  ```

  **References**:
  - `corpus/needle_questions_adversarial.json` - 20 hard questions
  - Categories: VERSION (40% pass), COMPARISON (100%), NEGATION (60%), VOCABULARY (60%)

  **Acceptance Criteria**:
  - [x] Benchmark completes
  - [x] `rewrite_log.txt` contains ORIGINAL_QUERY and REWRITTEN_QUERY lines
  - [x] Results saved to `results/needle_questions_adversarial_retrieval.json`

  **Commit**: NO

---

- [x] 3. Parse logs and extract rewrite pairs

  **What to do**:
  - Read `rewrite_log.txt`
  - Extract all ORIGINAL_QUERY → REWRITTEN_QUERY pairs
  - Match with hit/miss results from JSON
  - Present analysis of rewrite quality on failures

  **Acceptance Criteria**:
  - [x] List of all 20 query transformations
  - [x] Correlation with pass/fail status
  - [x] Identify patterns in failed rewrites

  **Commit**: NO

---

## Success Criteria

After completion:
- We can see exactly how each query was transformed
- We can analyze if rewrites on VOCABULARY category (60% pass) are appropriate
- We have data to decide if rewrite prompt needs improvement
