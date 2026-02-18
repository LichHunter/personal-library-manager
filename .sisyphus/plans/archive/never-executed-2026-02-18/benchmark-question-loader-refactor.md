# Benchmark Question Loader Refactor

## TL;DR

> **Quick Summary**: Create a new question loader component with a well-defined JSON schema to fix the benchmark bug where "informed" questions were never actually tested. Convert informed_questions.json to the new format.
> 
> **Deliverables**:
> - `poc/modular_retrieval_pipeline/components/question_loader.py` - New loader component
> - `poc/modular_retrieval_pipeline/corpus/informed_questions.json` - Converted question file
> - `poc/modular_retrieval_pipeline/corpus/README.md` - Schema documentation
> - Updated `benchmark.py` to use new loader
> 
> **Estimated Effort**: Short (1-4 hours)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 → Task 3 → Task 4 → Task 5

---

## Context

### Original Request
Fix the benchmark question loading system by creating a new component with a well-defined JSON structure. The current benchmark.py hardcodes field extraction (`realistic_q1`, `realistic_q2`), causing the "informed" benchmark to run the same realistic questions instead of technical terminology queries.

### Interview Summary
**Key Discussions**:
- Root cause: benchmark.py extracts from `realistic_q1`/`realistic_q2` fields regardless of input file
- informed_questions.json has same `realistic_q1`/`realistic_q2` values as realistic file
- Need a clean, documented JSON schema for question files

**Decisions Made**:
- Format support: **New format ONLY** (clean break, no backward compatibility)
- Metadata: **PRESERVE** in optional fields (define explicit schema)
- Files to convert: **informed_questions.json ONLY** (minimal scope)
- Low-quality questions: **EXCLUDE** from converted file

### Metis Review
**Identified Gaps** (addressed):
- Current 3 different formats need unification → Defined single canonical schema
- Metadata fields need explicit handling → Optional metadata fields in schema
- Backward compatibility question → Decided: new format only
- Edge cases (quality_pass=false) → Decided: exclude during conversion

---

## Work Objectives

### Core Objective
Create a flexible question loader component with a documented JSON schema that properly separates "realistic" (natural language) and "informed" (technical terminology) questions.

### Concrete Deliverables
- `poc/modular_retrieval_pipeline/components/question_loader.py`
- `poc/modular_retrieval_pipeline/corpus/informed_questions.json`
- `poc/modular_retrieval_pipeline/corpus/README.md`
- Updated `poc/modular_retrieval_pipeline/benchmark.py`

### Definition of Done
- [x] `python poc/modular_retrieval_pipeline/benchmark.py --questions poc/modular_retrieval_pipeline/corpus/informed_questions.json --strategy modular-no-llm --limit 5` runs successfully
- [x] Converted file contains 50 questions (25 topics × 2 variants)
- [x] JSON schema is documented in corpus/README.md
- [x] Loader correctly extracts `question`, `expected_answer`, `doc_id` fields

### Must Have
- New loader component with explicit field contracts
- Converted informed_questions.json with TRUE informed queries (technical terminology)
- Schema documentation

### Must NOT Have (Guardrails)
- NO JSON Schema runtime validation library (keep it simple)
- NO auto-detecting question format at runtime
- NO refactoring of `run_modular_no_llm_benchmark()` function
- NO CLI argument to select loader behavior
- NO abstract Question class hierarchy
- NO unit tests (unless explicitly requested later)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest in project)
- **User wants tests**: NO (manual-only for this task)
- **Framework**: N/A
- **QA approach**: Manual verification with automated commands

### Automated Verification (NO User Intervention)

Each TODO includes EXECUTABLE verification that agents can run directly:

**Verification Tools**:
| Type | Tool | Method |
|------|------|--------|
| File existence | Bash `ls` | Check files exist |
| JSON validity | Bash `python -c` | Parse and validate structure |
| Question count | Bash `python -c` | Assert count matches expected |
| Benchmark run | Bash `python benchmark.py` | Exit code 0 + output contains metrics |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Create question_loader.py component
└── Task 2: Create corpus/ directory + README with schema docs

Wave 2 (After Wave 1):
├── Task 3: Convert informed_questions.json to new format
└── (depends on Task 1 for schema definition)

Wave 3 (After Wave 2):
├── Task 4: Update benchmark.py to use new loader
└── (depends on Task 1 + Task 3)

Wave 4 (After Wave 3):
└── Task 5: Verify end-to-end benchmark run
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 3, 4 | 2 |
| 2 | None | 3 | 1 |
| 3 | 1, 2 | 4, 5 | None |
| 4 | 1, 3 | 5 | None |
| 5 | 4 | None | None (final) |

### Agent Dispatch Summary

| Wave | Tasks | Recommended Dispatch |
|------|-------|---------------------|
| 1 | 1, 2 | Parallel: `category="quick"` for each |
| 2 | 3 | Sequential after Wave 1 |
| 3 | 4 | Sequential after Task 3 |
| 4 | 5 | Final verification |

---

## TODOs

- [x] 1. Create question_loader.py component

  **What to do**:
  - Create `poc/modular_retrieval_pipeline/components/question_loader.py`
  - Implement `load_questions(filepath: str) -> list[dict]` function
  - Define explicit field contracts in docstring
  - Required fields: `question`, `expected_answer`, `doc_id`
  - Optional fields: `id`, `difficulty`, `type`, `section`, `quality_score`, `source`
  - Validate required fields exist, raise clear error if missing
  - Return flat list of question dicts (no nested variants)

  **Must NOT do**:
  - Do NOT add JSON Schema validation library
  - Do NOT support old format (realistic_q1/q2)
  - Do NOT create abstract classes

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single file creation with straightforward logic
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 2)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/components/keyword_extractor.py:1-24` - Docstring style with explicit field contracts
  - `poc/modular_retrieval_pipeline/components/query_expander.py` - Simple component pattern

  **API/Type References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:83-134` - Current load_questions() to understand expected output format

  **Why Each Reference Matters**:
  - `keyword_extractor.py` shows how to document field contracts in docstrings
  - `benchmark.py` shows what format the caller expects

  **Acceptance Criteria**:

  ```bash
  # Verify file created
  ls poc/modular_retrieval_pipeline/components/question_loader.py
  # Assert: File exists

  # Verify module imports without error
  python -c "from poc.modular_retrieval_pipeline.components.question_loader import load_questions; print('OK')"
  # Assert: Output is "OK"

  # Verify function signature
  python -c "
  from poc.modular_retrieval_pipeline.components.question_loader import load_questions
  import inspect
  sig = inspect.signature(load_questions)
  assert 'filepath' in [p.name for p in sig.parameters.values()], 'Missing filepath param'
  print('Signature OK')
  "
  # Assert: Output is "Signature OK"
  ```

  **Evidence to Capture**:
  - [ ] Terminal output from verification commands

  **Commit**: YES
  - Message: `feat(benchmark): add question_loader component with defined schema`
  - Files: `poc/modular_retrieval_pipeline/components/question_loader.py`
  - Pre-commit: N/A

---

- [x] 2. Create corpus directory and schema documentation

  **What to do**:
  - Create directory `poc/modular_retrieval_pipeline/corpus/`
  - Create `poc/modular_retrieval_pipeline/corpus/README.md` with:
    - JSON schema documentation
    - Required fields with types and descriptions
    - Optional fields with types and descriptions
    - Example question entry
    - Example complete file structure

  **Must NOT do**:
  - Do NOT create JSON Schema file (just markdown docs)
  - Do NOT add complex validation rules

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Directory creation + documentation writing
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 1 (with Task 1)
  - **Blocks**: Task 3
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json` - Current format to understand field mapping
  - `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json` - Source of informed questions

  **Documentation References**:
  - `poc/chunking_benchmark_v2/README.md` - Example of benchmark documentation style

  **Why Each Reference Matters**:
  - `realistic_questions.json` shows current field names to document mapping FROM
  - `README.md` shows documentation style used in this project

  **Acceptance Criteria**:

  ```bash
  # Verify directory exists
  ls -d poc/modular_retrieval_pipeline/corpus/
  # Assert: Directory exists

  # Verify README exists and has content
  test -s poc/modular_retrieval_pipeline/corpus/README.md && echo "README exists with content"
  # Assert: Output contains "README exists with content"

  # Verify README contains required sections
  grep -q "question" poc/modular_retrieval_pipeline/corpus/README.md && \
  grep -q "expected_answer" poc/modular_retrieval_pipeline/corpus/README.md && \
  grep -q "doc_id" poc/modular_retrieval_pipeline/corpus/README.md && \
  echo "Schema fields documented"
  # Assert: Output is "Schema fields documented"
  ```

  **Evidence to Capture**:
  - [ ] Terminal output from verification commands

  **Commit**: YES (group with Task 1)
  - Message: `feat(benchmark): add question_loader component with defined schema`
  - Files: `poc/modular_retrieval_pipeline/corpus/README.md`
  - Pre-commit: N/A

---

- [x] 3. Convert informed_questions.json to new format

  **What to do**:
  - Read `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json`
  - For each question entry with `quality_pass: true` (or missing = assume true):
    - Create entry with `question` = `original_instruction` (the INFORMED/technical query)
    - Set `expected_answer` = `original_instruction`
    - Set `doc_id` = existing `doc_id`
    - Preserve optional metadata: `difficulty`, `type`, `quality_score`, `source`
    - Add `variant` field: "informed_q1" or "informed_q2" for traceability
  - For EACH original question, create TWO entries:
    - One with question text being the technical/informed version
    - Need to generate informed variants from original_instruction
  - Save to `poc/modular_retrieval_pipeline/corpus/informed_questions.json`
  - Validate: 50 total entries (25 questions × 2 variants)

  **CRITICAL**: The current `informed_questions.json` does NOT have separate informed_q1/q2 fields!
  - The `original_instruction` IS the informed question
  - Need to create 2 variants by:
    - variant 1: Use `original_instruction` directly
    - variant 2: Rephrase `original_instruction` slightly (or use same if acceptable)

  **Must NOT do**:
  - Do NOT include questions with `quality_pass: false`
  - Do NOT copy `realistic_q1`/`realistic_q2` values (those are NOT informed!)
  - Do NOT modify the original file in chunking_benchmark_v2/

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Data transformation task
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (Sequential)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Tasks 1, 2

  **References**:

  **Pattern References**:
  - `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json` - Source file with `original_instruction` field

  **Schema References**:
  - `poc/modular_retrieval_pipeline/corpus/README.md` - Target schema (created in Task 2)

  **Why Each Reference Matters**:
  - Source file shows current structure with `original_instruction` containing technical queries
  - Target schema defines required output format

  **Acceptance Criteria**:

  ```bash
  # Verify file created
  ls poc/modular_retrieval_pipeline/corpus/informed_questions.json
  # Assert: File exists

  # Verify valid JSON
  python -c "import json; json.load(open('poc/modular_retrieval_pipeline/corpus/informed_questions.json'))" && echo "Valid JSON"
  # Assert: Output is "Valid JSON"

  # Verify question count (should be 50 = 25 questions × 2 variants)
  python -c "
  import json
  with open('poc/modular_retrieval_pipeline/corpus/informed_questions.json') as f:
      data = json.load(f)
  count = len(data['questions'])
  assert count == 50, f'Expected 50 questions, got {count}'
  print(f'Question count OK: {count}')
  "
  # Assert: Output is "Question count OK: 50"

  # Verify required fields present in first question
  python -c "
  import json
  with open('poc/modular_retrieval_pipeline/corpus/informed_questions.json') as f:
      data = json.load(f)
  q = data['questions'][0]
  assert 'question' in q, 'Missing question field'
  assert 'expected_answer' in q, 'Missing expected_answer field'
  assert 'doc_id' in q, 'Missing doc_id field'
  print('Required fields present')
  "
  # Assert: Output is "Required fields present"

  # Verify questions are INFORMED (contain technical terms), not realistic
  python -c "
  import json
  with open('poc/modular_retrieval_pipeline/corpus/informed_questions.json') as f:
      data = json.load(f)
  # First question should be about 'Infrastructure Provider' or 'Gateway API' (informed)
  # NOT 'load balancers and network configurations' (realistic)
  q = data['questions'][0]['question'].lower()
  assert 'load balancer' not in q or 'gateway' in q, f'Question looks realistic, not informed: {q}'
  print('Questions are informed (technical terminology)')
  "
  # Assert: Output is "Questions are informed (technical terminology)"
  ```

  **Evidence to Capture**:
  - [ ] Terminal output from verification commands
  - [ ] Sample of first 3 questions showing technical terminology

  **Commit**: YES
  - Message: `feat(benchmark): add informed questions in new schema format`
  - Files: `poc/modular_retrieval_pipeline/corpus/informed_questions.json`
  - Pre-commit: N/A

---

- [x] 4. Update benchmark.py to use new question_loader

  **What to do**:
  - Import `load_questions` from `components.question_loader`
  - Modify `load_questions()` function in benchmark.py OR replace its usage
  - When loading from `corpus/` directory, use new loader
  - Ensure returned format matches what `run_modular_no_llm_benchmark()` expects:
    ```python
    {
        "id": str,
        "question": str,
        "expected_answer": str,
        "doc_id": str
    }
    ```
  - Keep old `load_questions()` working for backward compatibility with needle format (optional)

  **Must NOT do**:
  - Do NOT refactor `run_modular_no_llm_benchmark()` function
  - Do NOT change the benchmark's core logic
  - Do NOT add CLI arguments for format selection

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Small modification to existing file
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 3 (Sequential)
  - **Blocks**: Task 5
  - **Blocked By**: Tasks 1, 3

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/benchmark.py:83-134` - Current load_questions() function to modify
  - `poc/modular_retrieval_pipeline/benchmark.py:220-250` - Where questions are used (check expected format)

  **API/Type References**:
  - `poc/modular_retrieval_pipeline/components/question_loader.py` - New loader to import (created in Task 1)

  **Why Each Reference Matters**:
  - Current `load_questions()` shows format that callers expect
  - Usage site shows exact dict keys needed

  **Acceptance Criteria**:

  ```bash
  # Verify benchmark.py imports new loader
  grep -q "from.*question_loader import" poc/modular_retrieval_pipeline/benchmark.py || \
  grep -q "question_loader" poc/modular_retrieval_pipeline/benchmark.py && \
  echo "Import found"
  # Assert: Output is "Import found"

  # Verify benchmark runs with new questions file (dry run with limit)
  cd /home/fujin/Code/personal-library-manager && \
  python poc/modular_retrieval_pipeline/benchmark.py \
    --questions poc/modular_retrieval_pipeline/corpus/informed_questions.json \
    --strategy modular-no-llm \
    --limit 2 2>&1 | head -20
  # Assert: No import errors, shows "Running benchmark" or similar
  ```

  **Evidence to Capture**:
  - [ ] Terminal output from benchmark dry run

  **Commit**: YES
  - Message: `refactor(benchmark): integrate question_loader component`
  - Files: `poc/modular_retrieval_pipeline/benchmark.py`
  - Pre-commit: N/A

---

- [x] 5. End-to-end verification: Run full benchmark with informed questions

  **What to do**:
  - Run complete benchmark with informed questions
  - Verify metrics are calculated
  - Compare Hit@5 between realistic and informed to validate the fix
  - Expected: informed questions should show DIFFERENT (likely better) results than realistic

  **Must NOT do**:
  - Do NOT modify any code
  - This is verification only

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Command execution and validation only
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 4 (Final)
  - **Blocks**: None
  - **Blocked By**: Task 4

  **References**:

  **Pattern References**:
  - `poc/modular_retrieval_pipeline/benchmark_results_realistic.json` - Baseline to compare against (36% Hit@5)

  **Why Each Reference Matters**:
  - Baseline results let us verify informed questions produce different results

  **Acceptance Criteria**:

  ```bash
  # Run full benchmark (all 50 questions)
  cd /home/fujin/Code/personal-library-manager && \
  python poc/modular_retrieval_pipeline/benchmark.py \
    --questions poc/modular_retrieval_pipeline/corpus/informed_questions.json \
    --strategy modular-no-llm \
    --output poc/modular_retrieval_pipeline/results/true_informed_benchmark_$(date +%Y%m%d_%H%M%S).json
  # Assert: Exit code 0

  # Verify output file created
  ls poc/modular_retrieval_pipeline/results/true_informed_benchmark_*.json | tail -1
  # Assert: File exists

  # Extract and display metrics
  python -c "
  import json
  import glob
  files = sorted(glob.glob('poc/modular_retrieval_pipeline/results/true_informed_benchmark_*.json'))
  latest = files[-1]
  with open(latest) as f:
      data = json.load(f)
  print(f'Hit@5: {data.get(\"hit_at_5_rate\", \"N/A\")}%')
  print(f'Hit@1: {data.get(\"hit_at_1_rate\", \"N/A\")}%')
  print(f'MRR: {data.get(\"mrr\", \"N/A\")}')
  print(f'Questions: {data.get(\"total_questions\", \"N/A\")}')
  "
  # Assert: Metrics are displayed, Hit@5 should differ from baseline 36%
  ```

  **Evidence to Capture**:
  - [ ] Benchmark output showing metrics
  - [ ] Comparison with baseline (36% Hit@5 for realistic)

  **Commit**: NO (verification only)

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 + 2 | `feat(benchmark): add question_loader component with defined schema` | question_loader.py, corpus/README.md | Module imports OK |
| 3 | `feat(benchmark): add informed questions in new schema format` | corpus/informed_questions.json | 50 questions, valid JSON |
| 4 | `refactor(benchmark): integrate question_loader component` | benchmark.py | Benchmark runs without error |

---

## Success Criteria

### Verification Commands
```bash
# Full verification sequence
cd /home/fujin/Code/personal-library-manager

# 1. Check all files exist
ls poc/modular_retrieval_pipeline/components/question_loader.py
ls poc/modular_retrieval_pipeline/corpus/README.md
ls poc/modular_retrieval_pipeline/corpus/informed_questions.json

# 2. Verify question count
python -c "import json; d=json.load(open('poc/modular_retrieval_pipeline/corpus/informed_questions.json')); print(len(d['questions']))"
# Expected: 50

# 3. Run benchmark
python poc/modular_retrieval_pipeline/benchmark.py \
  --questions poc/modular_retrieval_pipeline/corpus/informed_questions.json \
  --strategy modular-no-llm \
  --limit 5
# Expected: Exit 0, shows accuracy metrics
```

### Final Checklist
- [ ] question_loader.py created with documented schema
- [ ] corpus/README.md documents JSON structure
- [ ] informed_questions.json has 50 questions with technical terminology
- [ ] benchmark.py uses new loader
- [ ] Benchmark runs successfully with new file
- [ ] Results differ from baseline 36% (proving informed questions are actually used)
