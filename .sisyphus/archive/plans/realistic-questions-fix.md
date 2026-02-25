# Realistic Questions Benchmark - Fix & Execute

## Context

### Problem
1. The script uses `anthropic.Anthropic()` directly - needs `call_llm()` from `enrichment.provider`
2. **The prompt was NEVER actually tested** - all previous runs failed with auth errors (0/20 success)
3. No questions were generated - `corpus/realistic_questions.json` doesn't exist

### What Needs to Happen
1. Fix authentication to use `call_llm()`
2. **Test the prompt and iterate until quality is acceptable** (≥80% pass rate)
3. Generate 200 questions
4. Run retrieval benchmark
5. Generate failure report

---

## TODOs

- [ ] 1. Fix transform_question() to use call_llm

  **What to do**:
  - Remove `import anthropic` (line 17)
  - Add `from enrichment.provider import call_llm`
  - Replace `client.messages.create()` with `call_llm(prompt, model="claude-haiku", timeout=30)`
  - Handle string response (call_llm returns string, not object)

  **Exact code changes**:
  
  ```python
  # Line 17 - REMOVE:
  import anthropic
  
  # Line 17 - ADD:
  from enrichment.provider import call_llm
  
  # Lines 152-163 - REPLACE entire block:
  # OLD:
  client = anthropic.Anthropic()  # Uses ANTHROPIC_API_KEY env var
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
              print(f"[transform_question] Attempt {attempt + 1}: Empty response from LLM")
              if attempt < max_retries - 1:
                  time.sleep(1)
              continue
          
          content = content.strip()
  ```

  **References**:
  - `enrichment/provider.py:319-321` - `call_llm(prompt, model, timeout)` signature
  - `retrieval/query_rewrite.py:66` - Usage example: `call_llm(prompt, model="claude-haiku", timeout=5)`

  **Acceptance Criteria**:
  - [ ] Script imports without errors: `python -c "from benchmark_realistic_questions import transform_question"`
  - [ ] No `anthropic` in imports

  **Commit**: YES
  - Message: `fix(benchmark): use call_llm for OAuth authentication`

---

- [ ] 2. Test prompt quality and iterate (CRITICAL - WAS NEVER DONE)

  **What to do**:
  - Run `python benchmark_realistic_questions.py --test-prompt`
  - Review the 20 transformations
  - Check quality pass rate (target: ≥80%)
  - **If quality is low (<80%)**: Analyze common issues and adjust `TRANSFORMATION_PROMPT_V2`
  - Re-run until ≥80% pass rate achieved

  **Quality heuristics** (already implemented):
  1. Originality: <70% word overlap with original
  2. Phrasing: Starts with problem language (how, why, my, can, etc.)
  3. Conciseness: <120 characters
  4. Realism: Contains symptoms (error, can't, doesn't, etc.)
  5. Not "What is" pattern

  **Potential prompt adjustments if quality is low**:
  - Add more examples for failing patterns
  - Strengthen instructions for conciseness
  - Add negative examples ("Don't do this...")
  - Adjust character limit guidance

  **Acceptance Criteria**:
  - [ ] `--test-prompt` runs without auth errors
  - [ ] 20 samples successfully transformed
  - [ ] Quality pass rate ≥80% (16/20 passing)
  - [ ] If needed: prompt adjusted and re-tested

  **Commit**: YES (if prompt was modified)
  - Message: `feat(benchmark): iterate prompt for ≥80% quality pass rate`

---

- [ ] 3. Generate 200 realistic questions

  **What to do**:
  - Run `python benchmark_realistic_questions.py --generate 200`
  - Wait ~10-15 minutes for 200 Haiku calls
  - Verify output file created

  **Acceptance Criteria**:
  - [ ] `corpus/realistic_questions.json` exists
  - [ ] Contains 200 questions
  - [ ] Metadata shows high_quality count (expect ≥160 at 80% rate)

  **Commit**: YES
  - Message: `data(benchmark): generate 200 realistic questions`
  - Files: `corpus/realistic_questions.json`

---

- [ ] 4. Run retrieval benchmark

  **What to do**:
  - Run `python benchmark_realistic_questions.py --run-benchmark`
  - Wait ~30 minutes for indexing + 400 queries

  **Acceptance Criteria**:
  - [ ] Loads 1,569 documents
  - [ ] Creates timestamped results folder
  - [ ] `results/realistic_<timestamp>/retrieval_results.json` exists
  - [ ] Summary shows Hit@1, Hit@5, MRR

  **Commit**: YES
  - Message: `results(benchmark): realistic questions retrieval benchmark`

---

- [ ] 5. Generate failure report

  **What to do**:
  - Run `python benchmark_realistic_questions.py --report results/realistic_<timestamp>`

  **Acceptance Criteria**:
  - [ ] `results/realistic_<timestamp>/benchmark_report.md` created
  - [ ] Report contains: Summary, Q1 vs Q2, Failure Analysis, Worst Failures

  **Commit**: YES
  - Message: `docs(benchmark): failure analysis report`

---

## Post-Completion Cleanup

After all tasks complete successfully:
- [ ] Remove `--test-prompt` CLI flag (no longer needed after prompt is validated)
- [ ] Update notepad with final learnings

---

## Success Criteria

```bash
cd poc/chunking_benchmark_v2

# Verify files exist
ls -la corpus/realistic_questions.json
ls -la results/realistic_*/retrieval_results.json
ls -la results/realistic_*/benchmark_report.md

# Verify question count
python -c "import json; d=json.load(open('corpus/realistic_questions.json')); print(f'Questions: {len(d[\"questions\"])}, High quality: {d[\"metadata\"][\"high_quality\"]}')"
```

### Final Checklist
- [ ] Auth fixed (uses call_llm, not anthropic.Anthropic)
- [ ] Prompt tested and validated (≥80% quality)
- [ ] 200 questions generated
- [ ] Retrieval benchmark completed  
- [ ] Failure report generated
