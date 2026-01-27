# Realistic Questions Benchmark - Fix & Execute

## Context

### Problem
The `benchmark_realistic_questions.py` script uses `anthropic.Anthropic()` directly, which requires `ANTHROPIC_API_KEY` environment variable. However, the project has an existing `call_llm()` function in `enrichment/provider.py` that uses OAuth authentication via `~/.local/share/opencode/auth.json`.

### Root Cause
I implemented `transform_question()` using the wrong authentication pattern. Should have followed existing codebase patterns.

### Solution
Replace `anthropic.Anthropic()` calls with `call_llm()` from `enrichment.provider`.

---

## Work Objectives

### Core Objective
Fix the authentication issue and execute the full benchmark pipeline end-to-end.

### Concrete Deliverables
1. Fixed `transform_question()` using `call_llm()`
2. Verified `--validate-mapping` works
3. Verified `--test-prompt` works (with real LLM calls)
4. Generated `corpus/realistic_questions.json` (200 questions)
5. Executed `--run-benchmark` against full corpus
6. Generated `results/realistic_<timestamp>/benchmark_report.md`

### Definition of Done
- [ ] All 6 CLI commands execute successfully
- [ ] 200 realistic questions generated
- [ ] Retrieval benchmark completed
- [ ] Failure report generated

---

## TODOs

- [ ] 1. Fix transform_question() to use call_llm

  **What to do**:
  - Remove `import anthropic` line
  - Add `from enrichment.provider import call_llm` import
  - Replace the `client.messages.create()` call with `call_llm(prompt, model="claude-haiku", timeout=30)`
  - Parse the response string (call_llm returns string, not object)

  **Code change**:
  ```python
  # OLD (line 17):
  import anthropic
  
  # NEW:
  from enrichment.provider import call_llm
  
  # OLD (lines 152-163):
  client = anthropic.Anthropic()
  prompt = TRANSFORMATION_PROMPT_V2.format(original_question=original_question)
  
  for attempt in range(max_retries):
      try:
          response = client.messages.create(
              model="claude-3-5-haiku-latest",
              max_tokens=256,
              messages=[{"role": "user", "content": prompt}],
          )
          content = response.content[0].text.strip()
  
  # NEW:
  prompt = TRANSFORMATION_PROMPT_V2.format(original_question=original_question)
  
  for attempt in range(max_retries):
      try:
          content = call_llm(prompt, model="claude-haiku", timeout=30)
          if not content:
              print(f"[transform_question] Attempt {attempt + 1}: Empty response")
              continue
          content = content.strip()
  ```

  **References**:
  - `enrichment/provider.py:319-321` - `call_llm()` function signature
  - `retrieval/query_rewrite.py:66` - Example usage: `call_llm(prompt, model="claude-haiku", timeout=5)`
  - `benchmark_needle_haystack.py:106` - Example usage: `call_llm(prompt, model="claude-sonnet", timeout=120)`

  **Acceptance Criteria**:
  - [ ] `import anthropic` removed
  - [ ] `from enrichment.provider import call_llm` added
  - [ ] `transform_question()` uses `call_llm()`
  - [ ] Script imports without errors: `python -c "from benchmark_realistic_questions import transform_question"`

  **Commit**: YES
  - Message: `fix(benchmark): use call_llm for proper OAuth authentication`

---

- [ ] 2. Run --validate-mapping

  **What to do**:
  - Run `python benchmark_realistic_questions.py --validate-mapping`
  - Verify output shows ~97% coverage

  **Acceptance Criteria**:
  - [ ] Command executes without errors
  - [ ] Output shows: "Questions with matching docs: ~2500 (97%+)"

  **Commit**: NO (verification only)

---

- [ ] 3. Run --test-prompt

  **What to do**:
  - Run `python benchmark_realistic_questions.py --test-prompt`
  - Verify transformations work with real LLM calls
  - Expect ≥80% quality pass rate

  **Acceptance Criteria**:
  - [ ] Command executes without auth errors
  - [ ] 20 samples transformed
  - [ ] Quality pass rate shown
  - [ ] Output format matches expected (Q1, Q2, Quality score)

  **Commit**: NO (verification only)

---

- [ ] 4. Run --generate 200

  **What to do**:
  - Run `python benchmark_realistic_questions.py --generate 200`
  - Wait ~10-15 minutes for 200 Haiku calls
  - Verify `corpus/realistic_questions.json` created

  **Acceptance Criteria**:
  - [ ] Command executes without errors
  - [ ] `corpus/realistic_questions.json` exists
  - [ ] JSON contains 200 questions
  - [ ] Metadata shows total and high_quality counts

  **Commit**: YES (include generated JSON)
  - Message: `data(benchmark): generate 200 realistic questions from kubefix`

---

- [ ] 5. Run --run-benchmark

  **What to do**:
  - Run `python benchmark_realistic_questions.py --run-benchmark`
  - Wait ~30 minutes for full corpus indexing and 400 queries
  - Verify results saved to timestamped folder

  **Acceptance Criteria**:
  - [ ] Loads 1,569 documents
  - [ ] Creates ~8000+ chunks
  - [ ] Runs 400 queries (200 × 2 variants)
  - [ ] Creates `results/realistic_<timestamp>/retrieval_results.json`
  - [ ] Summary shows Hit@1, Hit@5, MRR

  **Commit**: YES (include results)
  - Message: `results(benchmark): realistic questions retrieval results`

---

- [ ] 6. Run --report

  **What to do**:
  - Run `python benchmark_realistic_questions.py --report results/realistic_<timestamp>`
  - Verify report generated in same folder

  **Acceptance Criteria**:
  - [ ] Command executes without errors
  - [ ] `results/realistic_<timestamp>/benchmark_report.md` created
  - [ ] Report contains: Summary, Q1 vs Q2, Failure Analysis, Worst Failures

  **Commit**: YES (include report)
  - Message: `docs(benchmark): realistic questions failure analysis report`

---

## Success Criteria

### Verification Commands
```bash
cd poc/chunking_benchmark_v2

# Task 1: Verify fix
python -c "from benchmark_realistic_questions import transform_question; print('OK')"

# Task 2: Validate mapping
python benchmark_realistic_questions.py --validate-mapping

# Task 3: Test prompt quality
python benchmark_realistic_questions.py --test-prompt

# Task 4: Generate questions
python benchmark_realistic_questions.py --generate 200

# Task 5: Run benchmark
python benchmark_realistic_questions.py --run-benchmark

# Task 6: Generate report
python benchmark_realistic_questions.py --report results/realistic_<timestamp>
```

### Final Checklist
- [ ] No `anthropic` import in script
- [ ] Uses `call_llm` from `enrichment.provider`
- [ ] 200 questions generated with ≥80% quality
- [ ] Retrieval benchmark completed
- [ ] Report generated with failure analysis
