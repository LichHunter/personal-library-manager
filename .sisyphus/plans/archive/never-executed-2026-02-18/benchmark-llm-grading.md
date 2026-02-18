# Benchmark LLM Grading Enhancement

## TL;DR

> **Quick Summary**: Add proper retrieval evaluation metrics to benchmark.py - LLM grading (Sonnet rates quality 1-10), rank-based metrics (Hit@1, Hit@5, MRR), and composite total score.
> 
> **Deliverables**:
> - New `RetrievalGrader` class in `components/retrieval_grader.py` (follows QueryRewriter pattern)
> - New `RetrievalMetrics` class in `utils/metrics.py` (rank, total_score, MRR calculations)
> - Modified `run_modular_no_llm_benchmark()` function using these classes
> - Updated results JSON structure with all new fields
> 
> **Estimated Effort**: Medium (2-3 hours)
> **Parallel Execution**: NO - sequential tasks
> **Critical Path**: Task 1 → Task 2 → Task 3 → Task 4

---

## Context

### Original Request
User wants to improve benchmark evaluation beyond simple "needle found in top-5" check:
- Use Claude Sonnet to grade if retrieved chunks would help solve the user's problem
- Add Hit@1, Hit@5, MRR metrics
- Add composite total score that combines quality + ranking

### Interview Summary
**Key Discussions**:
- Sonnet should receive: Question + Retrieved Chunks + Expected Answer
- Keep both needle metrics AND LLM grading (comprehensive evaluation)
- Always run grading (no flag needed)
- Total score: `llm_grade × position_weight` where position 1 gets full credit

**Research Findings**:
- LLM calls use `call_llm(prompt, model="claude-sonnet", timeout=30)` from `utils/llm_provider.py`
- Current benchmark at lines 331-443 in `benchmark.py`
- Questions JSON has `expected_answer` field

### Metis Review
**Identified Gaps** (addressed):
- Edge case: LLM returns non-numeric grade → parse with regex, fallback to null
- Edge case: Grade outside 1-10 → clamp to valid range
- Failure handling: Set `llm_grade: null`, continue execution
- Rank when not found: Use `null` (not a fake number like 6)

---

## Work Objectives

### Core Objective
Enhance `run_modular_no_llm_benchmark()` to evaluate retrieval quality using LLM grading and standard IR metrics.

### Concrete Deliverables
- `RetrievalGrader` class in `components/retrieval_grader.py` (follows QueryRewriter pattern)
- `RetrievalMetrics` class in `utils/metrics.py` (rank, MRR, total_score calculations)
- Updated `run_modular_no_llm_benchmark()` using these classes
- New results JSON structure with 8 new fields per question

### Definition of Done
- [x] `python benchmark.py --strategy modular-no-llm --quick` completes with grading
- [x] Results JSON contains: `llm_grade`, `llm_reasoning`, `rank`, `hit_at_1`, `hit_at_5`, `total_score`
- [x] Aggregate metrics contain: `hit_at_1_rate`, `hit_at_5_rate`, `mrr`, `avg_llm_grade`, `avg_total_score`
- [x] Pass rate metrics: `pass_rate_8`, `pass_rate_7`, `pass_rate_6_5`
- [x] LLM failures produce `null` grades, benchmark continues

### Must Have
- LLM grading with Sonnet (1-10 scale)
- Hit@1, Hit@5 boolean per question
- MRR calculation
- Total score = llm_grade × position_weight
- **Pass rates**: Multiple thresholds (>=8.0 excellent, >=7.0 good, >=6.5 acceptable)
- **Debug/trace logging**: All grading details logged (question, chunks, grade, reasoning)
- Graceful error handling

### Must NOT Have (Guardrails)
- Do NOT modify `run_baseline_benchmark()` or `run_modular_benchmark()`
- Do NOT add CLI flags (always run grading)
- Do NOT change retrieval logic or k=5
- Do NOT cache LLM grades
- Do NOT parse/analyze reasoning beyond storing raw string
- Do NOT add parallel grading (keep simple, sequential)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO formal test framework in POC
- **User wants tests**: Manual verification via benchmark run
- **QA approach**: Run benchmark, inspect JSON output with jq

### Automated Verification

**After each task, verify with:**
```bash
# Run quick benchmark (5 questions)
cd /home/fujin/Code/personal-library-manager
python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --quick

# Inspect results
jq '.results[0]' poc/modular_retrieval_pipeline/benchmark_results.json
jq '{hit_at_1_rate, hit_at_5_rate, mrr, avg_llm_grade, avg_total_score, pass_rate_8, pass_rate_7, pass_rate_6_5}' poc/modular_retrieval_pipeline/benchmark_results.json
```

---

## TODOs

- [x] 1. Create `RetrievalGrader` class in `components/retrieval_grader.py`

  **What to do**:
  - Create new file `components/retrieval_grader.py` following `QueryRewriter` pattern
  - Class `RetrievalGrader` with:
    - `__init__(self, timeout: float = 30.0)` - configurable timeout
    - `grade(self, question: str, expected_answer: str, chunks: list[dict]) -> GradeResult`
    - Returns dataclass `GradeResult(grade: int | None, reasoning: str | None, latency_ms: float)`
  - Use `call_llm(prompt, model="claude-sonnet", timeout=int(self.timeout))` - same pattern as QueryRewriter
  - Parse JSON response, handle errors gracefully (return None on failure)
  - **Add debug/trace logging** (same pattern as QueryRewriter):
    ```python
    self._log.debug(f"[retrieval-grader] Grading question: {question[:50]}...")
    self._log.trace(f"[retrieval-grader] Expected: {expected_answer[:100]}...")
    self._log.trace(f"[retrieval-grader] Chunks ({len(chunks)}): {[c.get('doc_id', 'unknown') for c in chunks]}")
    self._log.debug(f"[retrieval-grader] SUCCESS in {elapsed:.3f}s: grade={grade}")
    ```

  **Must NOT do**:
  - Do NOT cache grades
  - Do NOT retry on failure (return None, let caller continue)
  - Do NOT implement Component protocol (this is a standalone utility, not pipeline component)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Reason**: Single class, follows existing QueryRewriter pattern exactly

  **References**:
  - `poc/modular_retrieval_pipeline/components/query_rewriter.py` - **PRIMARY PATTERN TO FOLLOW**
    - Lines 47-71: Class structure with docstring
    - Lines 73-82: `__init__` with timeout and logger
    - Lines 103-129: `_rewrite_query()` with timing, try/except, logging
    - Line 107-108: `call_llm(prompt, model="claude-haiku", timeout=int(self.timeout))`
  - `poc/modular_retrieval_pipeline/utils/llm_provider.py` - `call_llm()` interface
  - `poc/modular_retrieval_pipeline/utils/logger.py` - `get_logger()` for logging

  **Grading Prompt Template** (module constant like QUERY_REWRITE_PROMPT):
  ```python
  GRADING_PROMPT = """You are an impartial judge evaluating retrieval quality for a documentation search system.

USER QUESTION: {question}

EXPECTED ANSWER (ground truth): {expected_answer}

RETRIEVED CHUNKS:
---
{chunks_text}
---

TASK: Determine if the retrieved chunks contain sufficient information to answer the user's question correctly.

EVALUATION CRITERIA:
1. Does the retrieved content contain the key facts from the expected answer?
2. Would a user be able to solve their problem using ONLY these chunks?
3. Ignore style, formatting, or verbosity - focus on factual completeness.

SCORING GUIDE:
- 10: PERFECT - Chunks contain the complete answer with all necessary details
- 8-9: EXCELLENT - Chunks contain the core answer, minor details may be missing
- 6-7: GOOD - Chunks contain most relevant information, user could likely solve their problem
- 4-5: PARTIAL - Chunks have some relevant info but missing key details needed to fully answer
- 2-3: POOR - Chunks are tangentially related but don't address the actual question
- 1: IRRELEVANT - Chunks have no useful information for this question

IMPORTANT:
- Compare chunks against the EXPECTED ANSWER to verify factual coverage
- A chunk mentioning the topic is NOT enough - it must contain actionable information
- Grade based on whether the user could SOLVE THEIR PROBLEM, not just "learn something"

First, analyze what key facts from the expected answer appear in the chunks.
Then provide your grade.

Respond with ONLY a JSON object (no markdown, no extra text):
{{"grade": <integer 1-10>, "reasoning": "<which key facts are present/missing>"}}"""
  ```
  
  **Class Structure:**
  ```python
  from dataclasses import dataclass
  from typing import Optional
  import json
  import time
  
  from ..utils.llm_provider import call_llm
  from ..utils.logger import get_logger
  
  @dataclass
  class GradeResult:
      grade: Optional[int]  # 1-10 or None on failure
      reasoning: Optional[str]
      latency_ms: float
  
  class RetrievalGrader:
      """Grades retrieval quality using Claude Sonnet.
      
      Follows the same pattern as QueryRewriter:
      - Stateless (no stored LLM client)
      - Configurable timeout
      - Graceful error handling (returns None on failure)
      """
      
      def __init__(self, timeout: float = 30.0):
          self.timeout = timeout
          self._log = get_logger()
      
      def grade(self, question: str, expected_answer: str, chunks: list[dict]) -> GradeResult:
          # Format chunks, call LLM, parse JSON, handle errors
          ...
  ```

  **Acceptance Criteria**:
  ```bash
  # Class exists and is importable
  python -c "from poc.modular_retrieval_pipeline.components.retrieval_grader import RetrievalGrader, GradeResult; print('OK')"
  
  # Verify follows same import pattern as QueryRewriter
  python -c "from poc.modular_retrieval_pipeline.components.retrieval_grader import RetrievalGrader; g = RetrievalGrader(timeout=5.0); print(f'timeout={g.timeout}')"
  ```

  **Commit**: NO (group with Task 2)

---

- [x] 2. Create `RetrievalMetrics` class in `utils/metrics.py`

  **What to do**:
  - Create new file `utils/metrics.py` with `RetrievalMetrics` class
  - Class contains all metric calculation methods (stateless, can be used as static methods or instance)
  - Methods:
    - `calculate_rank(retrieved_chunks: list[dict], needle_doc_id: str) -> int | None`
      - Returns 1-5 if found, None if not in top-k
    - `calculate_total_score(llm_grade: int | None, rank: int | None) -> float | None`
      - Rank 1: weight = 1.0 (full credit)
      - Rank 2-3: weight = 0.95 (small penalty)
      - Rank 4-5: weight = 0.85 (moderate penalty)
      - Not found (None): weight = 0.6 (significant penalty)
      - If llm_grade is None: return None
    - `calculate_mrr(results: list[dict]) -> float`
      - Mean Reciprocal Rank: `sum(1/rank for each found) / total_questions`
    - `calculate_pass_rates(results: list[dict], thresholds: list[float]) -> dict[float, float]`
      - Returns dict of threshold → pass_rate percentage

  **Must NOT do**:
  - Do NOT change existing accuracy calculation in benchmark.py
  - Do NOT add caching (these are pure calculations)

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - **Reason**: Simple math class, no external dependencies

  **References**:
  - MRR formula: `sum(1/rank for each found) / total_questions`
  - `poc/modular_retrieval_pipeline/utils/logger.py` - pattern for utils module structure

  **Class Structure:**
  ```python
  """Retrieval evaluation metrics.
  
  Provides standard IR metrics for evaluating retrieval quality:
  - Rank calculation
  - Hit@k metrics
  - Mean Reciprocal Rank (MRR)
  - Total score (combines LLM grade with position)
  """
  
  from typing import Optional
  
  
  class RetrievalMetrics:
      """Calculator for retrieval evaluation metrics.
      
      All methods are stateless and can be used as static methods.
      Instance creation allows for future configuration if needed.
      
      Position weights for total_score:
      - Rank 1:     1.0  (full credit - best possible)
      - Rank 2-3:   0.95 (small penalty)
      - Rank 4-5:   0.85 (moderate penalty)
      - Not found:  0.6  (significant penalty)
      """
      
      POSITION_WEIGHTS = {
          1: 1.0,
          2: 0.95,
          3: 0.95,
          4: 0.85,
          5: 0.85,
          None: 0.6,  # Not found
      }
      
      def calculate_rank(self, retrieved_chunks: list[dict], needle_doc_id: str) -> Optional[int]:
          """Find rank of needle document in retrieved chunks (1-indexed)."""
          for i, chunk in enumerate(retrieved_chunks):
              if chunk.get("doc_id") == needle_doc_id:
                  return i + 1  # 1-indexed rank
          return None
      
      def calculate_total_score(self, llm_grade: Optional[int], rank: Optional[int]) -> Optional[float]:
          """Calculate total score = llm_grade × position_weight."""
          if llm_grade is None:
              return None
          weight = self.POSITION_WEIGHTS.get(rank, self.POSITION_WEIGHTS[None])
          return llm_grade * weight
      
      def calculate_mrr(self, results: list[dict]) -> float:
          """Calculate Mean Reciprocal Rank."""
          reciprocal_ranks = []
          for r in results:
              rank = r.get("rank")
              if rank is not None:
                  reciprocal_ranks.append(1.0 / rank)
              else:
                  reciprocal_ranks.append(0.0)
          return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
      
      def calculate_pass_rates(self, results: list[dict], thresholds: list[float]) -> dict[float, float]:
          """Calculate pass rates at multiple thresholds."""
          valid_scores = [r for r in results if r.get("total_score") is not None]
          total = len(results)
          rates = {}
          for threshold in thresholds:
              passed = len([r for r in valid_scores if r["total_score"] >= threshold])
              rates[threshold] = (passed / total * 100) if total > 0 else 0.0
          return rates
  ```

  **Acceptance Criteria**:
  ```bash
  # Class exists and is importable
  python -c "from poc.modular_retrieval_pipeline.utils.metrics import RetrievalMetrics; print('OK')"
  
  # Test basic calculations
  python -c "
from poc.modular_retrieval_pipeline.utils.metrics import RetrievalMetrics
m = RetrievalMetrics()

# Test rank calculation
chunks = [{'doc_id': 'a'}, {'doc_id': 'b'}, {'doc_id': 'c'}]
assert m.calculate_rank(chunks, 'b') == 2
assert m.calculate_rank(chunks, 'z') is None

# Test total score
assert m.calculate_total_score(10, 1) == 10.0
assert m.calculate_total_score(10, 3) == 9.5
assert m.calculate_total_score(10, None) == 6.0
assert m.calculate_total_score(None, 1) is None

print('All tests passed!')
"
  ```

  **Commit**: NO (group with Task 3)

---

- [x] 3. Modify `run_modular_no_llm_benchmark()` to use new classes

  **What to do**:
  - Add imports at top of benchmark.py:
    ```python
    from components.retrieval_grader import RetrievalGrader, GradeResult
    from utils.metrics import RetrievalMetrics
    ```
  - Initialize classes at start of function:
    ```python
    grader = RetrievalGrader(timeout=30.0)
    metrics = RetrievalMetrics()
    ```
  - After retrieving chunks (line ~410):
    - Calculate rank using `metrics.calculate_rank(retrieved, needle_doc_id)`
    - Call `grader.grade(question, expected_answer, retrieved)` 
    - Calculate `total_score` using `metrics.calculate_total_score(grade_result.grade, rank)`
  - Update result dict to include all new fields:
    ```python
    result = {
        "question_id": q["id"],
        "question": q["question"],
        "expected_answer": q.get("expected_answer", ""),
        "needle_found": needle_found,
        "rank": rank,  # int 1-5 or None
        "hit_at_1": rank == 1,
        "hit_at_5": rank is not None and rank <= 5,
        "llm_grade": grade_result.grade,
        "llm_reasoning": grade_result.reasoning,
        "total_score": total_score,
        "latency_ms": round(latency, 1),
        "grading_latency_ms": grade_result.latency_ms,
    }
    ```
  - Update aggregate metrics calculation at end using RetrievalMetrics:
    ```python
    # Existing
    accuracy = sum(1 for r in results if r["needle_found"]) / len(results) * 100
    
    # New metrics using RetrievalMetrics class
    hit_at_1_rate = sum(1 for r in results if r["hit_at_1"]) / len(results) * 100
    hit_at_5_rate = sum(1 for r in results if r["hit_at_5"]) / len(results) * 100
    mrr = metrics.calculate_mrr(results)
    
    grades = [r["llm_grade"] for r in results if r["llm_grade"] is not None]
    avg_llm_grade = sum(grades) / len(grades) if grades else None
    scores = [r["total_score"] for r in results if r["total_score"] is not None]
    avg_total_score = sum(scores) / len(scores) if scores else None
    
    # Pass rates at multiple thresholds
    pass_rates = metrics.calculate_pass_rates(results, [8.0, 7.0, 6.5])
    pass_rate_8 = pass_rates[8.0]
    pass_rate_7 = pass_rates[7.0]
    pass_rate_6_5 = pass_rates[6.5]
    ```
  - Update return dict to include new aggregates including all pass rates

  **Must NOT do**:
  - Do NOT modify other benchmark functions
  - Do NOT change retrieval k=5
  - Do NOT remove existing metrics

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
  - **Skills**: `[]`
  - **Reason**: Modifying existing function, need to integrate new classes carefully

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:331-443` - Current function
  - `poc/modular_retrieval_pipeline/benchmark.py:408-432` - Per-question loop
  - `poc/modular_retrieval_pipeline/components/retrieval_grader.py` - RetrievalGrader (Task 1)
  - `poc/modular_retrieval_pipeline/utils/metrics.py` - RetrievalMetrics (Task 2)

  **Acceptance Criteria**:
  ```bash
  # Run benchmark and check new fields exist
  cd /home/fujin/Code/personal-library-manager
  python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --quick
  
  # Verify per-question fields
  jq '.results[0] | keys' poc/modular_retrieval_pipeline/benchmark_results.json
  # Should include: llm_grade, llm_reasoning, rank, hit_at_1, hit_at_5, total_score
  
  # Verify aggregate fields including pass rates
  jq '{accuracy, hit_at_1_rate, hit_at_5_rate, mrr, avg_llm_grade, avg_total_score, pass_rate_8, pass_rate_7, pass_rate_6_5}' poc/modular_retrieval_pipeline/benchmark_results.json
  # All fields should be numbers (or null for avg if all grades failed)
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add LLM grading and rank-based metrics`
  - Files: 
    - `poc/modular_retrieval_pipeline/components/retrieval_grader.py` (new)
    - `poc/modular_retrieval_pipeline/utils/metrics.py` (new)
    - `poc/modular_retrieval_pipeline/benchmark.py`

---

- [x] 4. Update logging to show new metrics

  **What to do**:
  - Update per-question log line to show grade, rank, and total score:
    ```python
    # Before: [1/20] ✓ (45ms) Question text...
    # After:  [1/20] ✓ R1 G8 T8.0 (45ms) Question text...
    # Where R1 = rank 1, G8 = grade 8, T8.0 = total score
    # Use ✓ if total_score >= 7.0 (pass), ✗ otherwise
    ```
  - Add debug/trace logging for each question:
    ```python
    logger.trace(f"[q:{q['id']}] Retrieved {len(retrieved)} chunks from docs: {doc_ids}")
    logger.trace(f"[q:{q['id']}] Needle doc '{needle_doc_id}' at rank: {rank}")
    logger.debug(f"[q:{q['id']}] Grade={llm_grade}, Reasoning: {reasoning}")
    logger.debug(f"[q:{q['id']}] Total score: {total_score} (grade={llm_grade} × weight={weight})")
    ```
  - Add new metric logs at end:
    ```python
    logger.metric("hit_at_1_rate", hit_at_1_rate, "%")
    logger.metric("hit_at_5_rate", hit_at_5_rate, "%")
    logger.metric("mrr", mrr)
    logger.metric("avg_llm_grade", avg_llm_grade)
    logger.metric("avg_total_score", avg_total_score)
    logger.metric("pass_rate_8", pass_rate_8, "%")    # Excellent (>=8.0)
    logger.metric("pass_rate_7", pass_rate_7, "%")    # Good (>=7.0) - PRIMARY
    logger.metric("pass_rate_6_5", pass_rate_6_5, "%")  # Acceptable (>=6.5)
    ```

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:425-428` - Current log format
  - `poc/modular_retrieval_pipeline/utils/logger.py` - Logger methods

  **Acceptance Criteria**:
  ```bash
  # Run and check output shows new metrics including all pass rates
  python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --quick 2>&1 | grep -E "(hit_at_1|mrr|avg_llm_grade|pass_rate)"
  # Should show: pass_rate_8, pass_rate_7, pass_rate_6_5
  ```

  **Commit**: YES
  - Message: `feat(benchmark): add logging for LLM grade and rank metrics`
  - Files: `poc/modular_retrieval_pipeline/benchmark.py`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 3 | `feat(benchmark): add LLM grading and rank-based metrics` | components/retrieval_grader.py, utils/metrics.py, benchmark.py |
| 4 | `feat(benchmark): add logging for LLM grade and rank metrics` | benchmark.py |

---

## Success Criteria

### Verification Commands
```bash
# Full benchmark run
cd /home/fujin/Code/personal-library-manager
python poc/modular_retrieval_pipeline/benchmark.py --strategy modular-no-llm --quick

# Check all new fields present
jq '.results[0] | {llm_grade, llm_reasoning, rank, hit_at_1, hit_at_5, total_score}' \
  poc/modular_retrieval_pipeline/benchmark_results.json

# Check aggregates including all pass rates
jq '{accuracy, hit_at_1_rate, hit_at_5_rate, mrr, avg_llm_grade, avg_total_score, pass_rate_8, pass_rate_7, pass_rate_6_5}' \
  poc/modular_retrieval_pipeline/benchmark_results.json

# Verify grades are in range 1-10 or null
jq '[.results[].llm_grade] | map(select(. != null)) | all(. >= 1 and . <= 10)' \
  poc/modular_retrieval_pipeline/benchmark_results.json
# Should output: true
```

### Final Checklist
- [x] `RetrievalGrader` class created in `components/retrieval_grader.py`
- [x] `RetrievalMetrics` class created in `utils/metrics.py`
- [x] All new fields present in results JSON
- [x] LLM grading works (grades 1-10 or null on failure)
- [x] Hit@1, Hit@5, MRR calculated correctly
- [x] Total score formula applied (graduated weights: 1.0, 0.95, 0.85, 0.6)
- [x] Pass rates at three thresholds (8.0, 7.0, 6.5)
- [x] Debug/trace logging for all grading details
- [x] Existing accuracy metric unchanged
- [x] Graceful handling of LLM failures
