# Manual Testing Interface for RAG Benchmark Validation

## Context

### Original Request
Create a manual testing interface to validate the RAG benchmark results (88.7% for `enriched_hybrid_llm` strategy). The goal is to build confidence in benchmark accuracy or discover testing methodology flaws through agent-driven qualitative testing.

### Interview Summary
**Key Discussions**:
- Question generation: Agent-driven (reads corpus, generates human-like questions)
- Strategies to test: Only `enriched_hybrid_llm` (the best performer at 88.7%)
- Output format: Markdown report with comparison tables
- Grading: Score 1-10 with expected answer and explanation
- Query count: Agent decides based on document coverage

**Research Findings**:
- Current benchmark: 5 CloudFlow docs, 20 queries × 6 variants, 53 key facts
- Best strategy (`enriched_hybrid_llm`): BM25 + semantic + LLM query rewriting
- Existing evaluation: binary (fact found/not found in retrieved chunks)
- Gap: No qualitative assessment of retrieval usefulness

### Metis Review
**Identified Gaps** (addressed):
1. **No grading rubric defined** → Added explicit 1-10 rubric in prompts
2. **LLM inconsistency risk** → Use temperature=0 for deterministic grading
3. **Scope creep risk** → Locked to single strategy, no comparison
4. **Missing success threshold** → Added VALIDATED/INCONCLUSIVE/INVALIDATED verdict
5. **Question generation bias** → Agent sees style guide but generates new questions
6. **Edge case: question with no answer** → Agent must verify answer exists first

---

## Work Objectives

### Core Objective
Build a CLI tool that uses Claude agents to generate human-like questions, execute retrieval, grade results qualitatively (1-10), and produce a markdown report validating or invalidating the 88.7% benchmark claim.

### Concrete Deliverables
- `poc/chunking_benchmark_v2/manual_test.py` - Single-file CLI tool
- `poc/chunking_benchmark_v2/results/manual_test_<timestamp>.md` - Output report

### Definition of Done
- [ ] `python manual_test.py` runs without errors
- [ ] Generates questions covering all 5 documents
- [ ] Each question graded 1-10 with grounded explanation
- [ ] Report contains aggregate score and validation verdict
- [ ] Runtime < 5 minutes for default run

### Must Have
- Agent reads corpus to establish ground truth before generating questions
- Questions follow realistic patterns (synonym, problem, casual, contextual, negation)
- Explicit grading rubric (1-10 scale with defined criteria)
- Comparison of manual scores vs benchmark claims
- VALIDATED / INCONCLUSIVE / INVALIDATED conclusion

### Must NOT Have (Guardrails)
- **NO multi-strategy comparison** - Only test `enriched_hybrid_llm`
- **NO interactive/REPL mode** - Single-shot CLI execution
- **NO new dependencies** - Use existing packages from pyproject.toml
- **NO modification of existing benchmark code** - New file only
- **NO caching or persistence** - Each run is independent
- **NO more than 15 questions per run** - Keep focused
- **NO automated "fix" suggestions** - This is validation only
- **NO parameter tuning** - Use fixed k=5 (same as benchmark)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: NO (new tool being created)
- **User wants tests**: NO (validation tool - manual QA sufficient)
- **Framework**: N/A
- **QA approach**: Manual verification

### Manual QA Procedures

Each TODO includes detailed verification steps with specific commands and expected outputs.

---

## Task Flow

```
Task 1 (Grading Rubric) → Task 2 (Core Script) → Task 3 (Question Generation)
                                       ↓
                         Task 4 (Retrieval Integration)
                                       ↓
                         Task 5 (Grading Logic)
                                       ↓
                         Task 6 (Report Generation)
                                       ↓
                         Task 7 (End-to-End Validation)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| Sequential | All | Each task depends on previous |

| Task | Depends On | Reason |
|------|------------|--------|
| 2 | 1 | Script uses rubric |
| 3 | 2 | Question gen needs script scaffold |
| 4 | 3 | Retrieval needs questions |
| 5 | 4 | Grading needs retrieval results |
| 6 | 5 | Report needs grading |
| 7 | 6 | Validation needs full flow |

---

## TODOs

- [x] 1. Define Grading Rubric and Validation Thresholds

  **What to do**:
  - Create explicit 1-10 grading criteria that will be embedded in the grading prompt
  - Define aggregate score thresholds for VALIDATED/INCONCLUSIVE/INVALIDATED
  - Document the rubric in a docstring within the script

  **Grading Rubric (to be embedded in prompt)**:
  ```
  10: Perfect - All requested information retrieved, directly answers question
  9: Excellent - Complete answer with minor irrelevant content
  8: Very Good - Answer present but buried in some noise
  7: Good - Core answer present, missing some supporting details
  6: Adequate - Partial answer, enough to be useful
  5: Borderline - Some relevant info but misses key point
  4: Poor - Tangentially related content only
  3: Very Poor - Mostly irrelevant, hint of topic
  2: Bad - Almost entirely irrelevant
  1: Failed - No relevant content retrieved
  ```

  **Validation Thresholds**:
  - VALIDATED: Average score >= 7.5 (88.7% benchmark is trustworthy)
  - INCONCLUSIVE: Average score 5.5-7.4 (benchmark may be optimistic)
  - INVALIDATED: Average score < 5.5 (benchmark is misleading)

  **Must NOT do**:
  - Don't make rubric subjective ("good enough")
  - Don't create complex multi-dimensional scoring

  **Parallelizable**: NO (foundational)

  **References**:
  - `poc/chunking_benchmark_v2/run_benchmark.py:152-175` - Existing match functions (exact_match, fuzzy_match)
  - `poc/chunking_benchmark_v2/corpus/ground_truth_realistic.json` - Key facts structure
  - `poc/chunking_benchmark_v2/README.md` - Benchmark results (88.7% target)

  **Acceptance Criteria**:
  - [ ] Rubric documented as Python constant/docstring
  - [ ] Thresholds defined as constants
  - [ ] Run: `grep -c "Score.*:" manual_test.py` → Shows 10 (one per score level)

  **Commit**: NO (groups with 2)

---

- [x] 2. Create Script Scaffold with CLI and Document Loading

  **What to do**:
  - Create `manual_test.py` with argparse CLI (--questions N, --output PATH)
  - Load corpus documents using existing benchmark patterns
  - Initialize `enriched_hybrid_llm` strategy
  - Add timing wrapper for overall execution

  **CLI Interface**:
  ```
  python manual_test.py                    # Default: agent decides question count
  python manual_test.py --questions 10     # Specific count
  python manual_test.py --output report.md # Custom output path
  ```

  **Must NOT do**:
  - Don't add interactive prompts
  - Don't add --strategy flag (locked to enriched_hybrid_llm)

  **Parallelizable**: NO (depends on 1)

  **References**:
  - `poc/chunking_benchmark_v2/run_benchmark.py:111-145` - Corpus loading pattern (load_corpus, load_queries)
  - `poc/chunking_benchmark_v2/retrieval/__init__.py:67-90` - create_retrieval_strategy()
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:81-114` - Strategy initialization
  - `poc/chunking_benchmark_v2/config_realistic.yaml` - Embedder config (BAAI/bge-base-en-v1.5)

  **Acceptance Criteria**:
  - [ ] Run: `python manual_test.py --help` → Shows usage
  - [ ] Run: `python manual_test.py --questions 1 2>&1 | head -5` → Shows "Loading corpus..."
  - [ ] Script exits cleanly (no hanging)

  **Commit**: YES
  - Message: `feat(benchmark): add manual testing script scaffold`
  - Files: `poc/chunking_benchmark_v2/manual_test.py`
  - Pre-commit: `python manual_test.py --help`

---

- [x] 3. Implement Agent Question Generation

  **What to do**:
  - Create `generate_questions()` function using Claude
  - Agent reads corpus documents (provided in prompt)
  - Agent generates questions following existing query patterns
  - Each question includes: query text, expected answer (grounded), source document
  - Agent verifies answer exists in corpus before including question

  **Prompt Structure**:
  ```
  You are testing a RAG retrieval system. Read these documents and generate {N} test questions.
  
  DOCUMENTS:
  {document contents}
  
  STYLE GUIDE (from existing queries):
  {sample queries from ground_truth_realistic.json}
  
  For each question, output JSON:
  {"query": "...", "expected_answer": "...", "source_doc": "...", "difficulty": "easy|medium|hard"}
  
  Rules:
  - Questions must have answers IN the documents (verify before including)
  - Mix difficulty levels
  - Cover all 5 documents
  - Use realistic human phrasing (casual, problem-oriented, technical)
  ```

  **Must NOT do**:
  - Don't generate questions about topics not in corpus
  - Don't let agent decide validation thresholds

  **Parallelizable**: NO (depends on 2)

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/provider.py:300-321` - call_llm() function
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py:15-50` - Prompt structure pattern
  - `poc/chunking_benchmark_v2/corpus/ground_truth_realistic.json:7-27` - Query pattern examples
  - `poc/chunking_benchmark_v2/corpus/realistic_documents/*.md` - Document content

  **Acceptance Criteria**:
  - [ ] Run: `python -c "from manual_test import generate_questions; print(generate_questions.__doc__)"` → Shows docstring
  - [ ] Run snippet that generates 2 questions → Returns list of dicts with required fields
  - [ ] Each question's `expected_answer` can be found in source document

  **Commit**: NO (groups with 4)

---

- [x] 4. Implement Retrieval Integration

  **What to do**:
  - Create `run_retrieval()` function that executes enriched_hybrid_llm
  - Index documents once, then query for each question
  - Return retrieved chunks with scores and content
  - Capture latency per query

  **Function Signature**:
  ```python
  def run_retrieval(
      strategy: EnrichedHybridLLMRetrieval,
      query: str,
      k: int = 5
  ) -> dict:
      """
      Returns:
          {
              "query": str,
              "chunks": [{"content": str, "score": float, "doc_id": str}],
              "latency_ms": float
          }
      """
  ```

  **Must NOT do**:
  - Don't modify retrieval strategy parameters
  - Don't add caching

  **Parallelizable**: NO (depends on 3)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:181-339` - retrieve() method
  - `poc/chunking_benchmark_v2/run_benchmark.py:185-245` - evaluate_single_query pattern
  - `poc/chunking_benchmark_v2/retrieval/base.py:130-133` - retrieve() interface

  **Acceptance Criteria**:
  - [ ] Run retrieval on test query → Returns dict with chunks, latency
  - [ ] `len(result["chunks"]) == 5` (k=5 default)
  - [ ] Each chunk has content, score, doc_id fields

  **Commit**: YES
  - Message: `feat(benchmark): add question generation and retrieval for manual testing`
  - Files: `poc/chunking_benchmark_v2/manual_test.py`
  - Pre-commit: `python -c "import manual_test"`

---

- [x] 5. Implement Agent Grading Logic

  **What to do**:
  - Create `grade_result()` function using Claude Sonnet (higher quality for judgment)
  - Prompt includes: question, expected answer, retrieved chunks, grading rubric
  - Returns: score (1-10), explanation, verdict (PASS/PARTIAL/FAIL)
  - Use temperature=0 for deterministic grading

  **Prompt Structure**:
  ```
  You are grading a RAG retrieval system's response.
  
  QUESTION: {query}
  EXPECTED ANSWER: {expected_answer}
  
  RETRIEVED CONTENT:
  {chunk1_content}
  ---
  {chunk2_content}
  ...
  
  GRADING RUBRIC:
  10: Perfect - All requested information retrieved, directly answers question
  ...
  1: Failed - No relevant content retrieved
  
  Grade this retrieval. Output JSON:
  {"score": N, "explanation": "...", "verdict": "PASS|PARTIAL|FAIL"}
  
  PASS = score >= 7, PARTIAL = score 4-6, FAIL = score <= 3
  ```

  **Must NOT do**:
  - Don't let agent modify rubric
  - Don't grade without showing expected answer (avoid hallucination)

  **Parallelizable**: NO (depends on 4)

  **References**:
  - `poc/chunking_benchmark_v2/enrichment/provider.py:300-321` - call_llm() with model parameter
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py` - Prompt structure with fallback

  **Acceptance Criteria**:
  - [ ] Grade a known-good retrieval → Score 8-10
  - [ ] Grade a known-bad retrieval (wrong topic) → Score 1-3
  - [ ] Output includes explanation string
  - [ ] Deterministic: same input → same score (run 3x, check consistency)

  **Commit**: NO (groups with 6)

---

- [x] 6. Implement Markdown Report Generation

  **What to do**:
  - Create `generate_report()` function
  - Output sections: Summary, Results Table, Detailed Findings, Conclusion
  - Include aggregate score and comparison to benchmark (88.7%)
  - Add VALIDATED/INCONCLUSIVE/INVALIDATED verdict

  **Report Structure**:
  ```markdown
  # Manual Testing Report
  
  Generated: {timestamp}
  Strategy: enriched_hybrid_llm
  Questions: {N}
  
  ## Summary
  - Average Score: X.X / 10
  - Benchmark Claim: 88.7% coverage
  - Validation: VALIDATED / INCONCLUSIVE / INVALIDATED
  
  ## Results
  
  | # | Question (truncated) | Score | Verdict | Source |
  |---|---------------------|-------|---------|--------|
  | 1 | "What is the API..." | 8 | PASS | api_reference |
  ...
  
  ## Detailed Findings
  
  ### Question 1: "What is the API rate limit?"
  **Expected**: 100 requests per minute per authenticated user
  **Score**: 8/10 (PASS)
  **Explanation**: Retrieved chunk contains exact rate limit info...
  **Retrieved Chunks**:
  1. [api_reference] "CloudFlow enforces..." (truncated to 200 chars)
  ...
  
  ## Conclusion
  {validation verdict with reasoning}
  ```

  **Must NOT do**:
  - Don't include full chunk content (truncate to 200 chars)
  - Don't add charts or HTML

  **Parallelizable**: NO (depends on 5)

  **References**:
  - `poc/chunking_benchmark_v2/run_benchmark.py:652-708` - generate_summary() pattern
  - `poc/chunking_benchmark_v2/README.md` - Table formatting style

  **Acceptance Criteria**:
  - [ ] Run: `python manual_test.py --questions 3` → Creates .md file
  - [ ] Report contains all sections (Summary, Results, Detailed, Conclusion)
  - [ ] Report is valid markdown (no broken tables)
  - [ ] Conclusion contains one of: VALIDATED, INCONCLUSIVE, INVALIDATED

  **Commit**: YES
  - Message: `feat(benchmark): complete manual testing tool with grading and report`
  - Files: `poc/chunking_benchmark_v2/manual_test.py`
  - Pre-commit: `python manual_test.py --questions 2`

---

- [ ] 7. End-to-End Validation Run

  **What to do**:
  - Run full manual test with agent-decided question count
  - Review generated report for sanity
  - Compare results to benchmark claims
  - Document findings

  **Must NOT do**:
  - Don't modify tool based on initial results (that's scope creep)

  **Parallelizable**: NO (final validation)

  **References**:
  - `poc/chunking_benchmark_v2/README.md` - Benchmark claims (88.7%)
  - Generated report from Task 6

  **Acceptance Criteria**:
  - [ ] Run: `python manual_test.py` (no args) → Completes successfully
  - [ ] Runtime < 5 minutes
  - [ ] Report saved to `results/manual_test_<timestamp>.md`
  - [ ] Report conclusion is one of: VALIDATED, INCONCLUSIVE, INVALIDATED
  - [ ] No crashes or unhandled errors

  **Manual Execution Verification**:
  - [ ] Using terminal:
    - Command: `cd poc/chunking_benchmark_v2 && source .venv/bin/activate && python manual_test.py`
    - Expected output contains: "Manual Testing Report"
    - Expected output contains: "Average Score:"
    - Expected output contains: "Validation:"
  - [ ] Open generated report in editor, verify:
    - Tables render correctly
    - No empty sections
    - Scores are within 1-10 range

  **Commit**: YES
  - Message: `docs(benchmark): add manual testing validation results`
  - Files: `poc/chunking_benchmark_v2/results/manual_test_*.md`
  - Pre-commit: N/A (documentation)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 2 | `feat(benchmark): add manual testing script scaffold` | manual_test.py | `python manual_test.py --help` |
| 4 | `feat(benchmark): add question generation and retrieval for manual testing` | manual_test.py | `python -c "import manual_test"` |
| 6 | `feat(benchmark): complete manual testing tool with grading and report` | manual_test.py | `python manual_test.py --questions 2` |
| 7 | `docs(benchmark): add manual testing validation results` | results/*.md | N/A |

---

## Success Criteria

### Verification Commands
```bash
# Script runs without error
cd poc/chunking_benchmark_v2 && source .venv/bin/activate
python manual_test.py --questions 5

# Report generated
ls results/manual_test_*.md

# Report contains conclusion
grep -E "(VALIDATED|INCONCLUSIVE|INVALIDATED)" results/manual_test_*.md
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Report contains validation verdict
- [ ] Runtime under 5 minutes for default run
- [ ] No new dependencies added
