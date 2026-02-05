# Informed Questions Benchmark Run

## TL;DR

> **Quick Summary**: Create informed-user questions JSON file (50 questions) and run modular-no-llm benchmark to measure retrieval improvement when users use proper Kubernetes terminology.
> 
> **Deliverables**:
> - `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json` (25 questions × 2 variants)
> - `poc/modular_retrieval_pipeline/results/informed_benchmark_<timestamp>.json` (benchmark results)
> - Summary metrics comparing informed vs realistic (36% baseline)
> 
> **Estimated Effort**: Quick (15-20 minutes total)
> **Parallel Execution**: NO - sequential tasks
> **Critical Path**: Create JSON → Verify Redis → Run Benchmark → Report Results

---

## Context

### Original Request
Create a question list with 50 informed-user questions and run benchmark tests on modular-no-llm hybrid retrieval with full Kubernetes data.

### Interview Summary
**Key Discussions**:
- Previous realistic benchmark showed 36% Hit@5 due to vocabulary mismatch
- Transformed 25 questions into "informed" style using proper K8s terminology
- Questions saved in draft: `.sisyphus/drafts/informed-questions-transform.md`

**Research Findings**:
- Benchmark script expects `realistic_q1`/`realistic_q2` field names (NOT `informed_q1`/`informed_q2`)
- Full corpus: 1,569 docs → 7,269 chunks
- Redis cache available for fast indexing

### Metis Review
**Identified Gaps** (addressed):
- Field name mismatch: JSON must use `realistic_q1`/`realistic_q2` for benchmark compatibility
- Duplicate doc_id: Q004 and Q023 both target same doc (documented, acceptable)
- Output file preservation: Use timestamped output path

---

## Work Objectives

### Core Objective
Measure retrieval accuracy improvement when users ask questions using proper Kubernetes terminology instead of natural language problem descriptions.

### Concrete Deliverables
1. `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json`
2. `poc/modular_retrieval_pipeline/results/informed_benchmark_<timestamp>.json`

### Definition of Done
- [x] JSON file created with 25 questions, each having `realistic_q1` and `realistic_q2` fields
- [x] Benchmark completes successfully with `modular-no-llm` strategy
- [x] Results JSON contains Hit@1, Hit@5, MRR metrics
- [x] Results documented for comparison against 36% realistic baseline

### Must Have
- JSON file with correct field names (`realistic_q1`, `realistic_q2`)
- All 25 questions from the draft included
- Each question has valid `doc_id` pointing to existing corpus document
- Benchmark run captures all 50 question variants

### Must NOT Have (Guardrails)
- Do NOT modify benchmark.py script
- Do NOT run other strategies (baseline, modular-llm)
- Do NOT add extra questions beyond the 25 defined
- Do NOT create visualizations or graphs
- Do NOT analyze individual question failures (just capture metrics)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (benchmark.py exists)
- **User wants tests**: Manual verification (benchmark IS the test)
- **Framework**: N/A - benchmark script provides verification

### Automated Verification (Agent-Executable)

**For each task, verification commands are embedded in acceptance criteria.**

---

## Execution Strategy

### Sequential Execution

```
Task 1: Create JSON file
    ↓
Task 2: Verify prerequisites (Redis, JSON validity)
    ↓
Task 3: Run benchmark
    ↓
Task 4: Report results
```

### Dependency Matrix

| Task | Depends On | Blocks |
|------|------------|--------|
| 1. Create JSON | None | 2, 3 |
| 2. Verify prereqs | 1 | 3 |
| 3. Run benchmark | 1, 2 | 4 |
| 4. Report results | 3 | None |

---

## TODOs

- [x] 1. Create informed_questions.json file

  **What to do**:
  - Extract JSON from `.sisyphus/drafts/informed-questions-transform.md`
  - **CRITICAL**: Rename field names from `informed_q1`/`informed_q2` to `realistic_q1`/`realistic_q2`
  - Save to `poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json`
  - The "informed" questions become the new `realistic_q1`/`realistic_q2` values

  **Must NOT do**:
  - Do NOT keep both informed and realistic fields (only the informed versions, renamed)
  - Do NOT modify original realistic_questions.json

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Simple file creation with field renaming

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Tasks 2, 3
  - **Blocked By**: None

  **References**:
  - `.sisyphus/drafts/informed-questions-transform.md` - Source JSON with questions
  - `poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json` - Reference format

  **Acceptance Criteria**:

  ```bash
  # Verify file exists
  test -f poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json && echo "PASS: file exists"

  # Verify JSON structure
  python3 -c "
  import json
  with open('poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json') as f:
      data = json.load(f)
  assert 'questions' in data, 'Missing questions array'
  assert len(data['questions']) == 25, f'Expected 25 questions, got {len(data[\"questions\"])}'
  assert all('realistic_q1' in q for q in data['questions']), 'Missing realistic_q1 field'
  assert all('realistic_q2' in q for q in data['questions']), 'Missing realistic_q2 field'
  assert all('doc_id' in q for q in data['questions']), 'Missing doc_id field'
  print(f'PASS: {len(data[\"questions\"])} questions with correct fields')
  "
  ```

  **Commit**: NO

---

- [x] 2. Verify prerequisites

  **What to do**:
  - Check Redis cache is running
  - Verify all doc_ids in JSON exist in corpus
  - Create results directory if needed

  **Must NOT do**:
  - Do NOT start Redis if not running (just report)
  - Do NOT modify corpus files

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Simple verification commands

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 3
  - **Blocked By**: Task 1

  **References**:
  - `poc/modular_retrieval_pipeline/docker-compose.yml` - Redis configuration

  **Acceptance Criteria**:

  ```bash
  # Verify Redis is running
  docker exec plm-redis-cache redis-cli PING | grep -q PONG && echo "PASS: Redis running" || echo "FAIL: Redis not running"

  # Create results directory
  mkdir -p poc/modular_retrieval_pipeline/results

  # Verify doc_ids exist in corpus
  python3 -c "
  import json
  import os

  with open('poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json') as f:
      data = json.load(f)

  corpus_dir = 'poc/chunking_benchmark_v2/corpus/kubernetes'
  missing = []
  for q in data['questions']:
      doc_id = q['doc_id']
      doc_path = os.path.join(corpus_dir, f'{doc_id}.md')
      if not os.path.exists(doc_path):
          missing.append(doc_id)

  if missing:
      print(f'FAIL: Missing docs: {missing}')
  else:
      print(f'PASS: All {len(data[\"questions\"])} doc_ids exist in corpus')
  "
  ```

  **Commit**: NO

---

- [x] 3. Run benchmark

  **What to do**:
  - Run benchmark with `modular-no-llm` strategy
  - Use informed_questions.json as input
  - Output to timestamped results file
  - Allow full 50 questions (no --limit flag)

  **Must NOT do**:
  - Do NOT use --quick flag
  - Do NOT use --limit flag
  - Do NOT run other strategies

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Single command execution

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: Task 4
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark.py` - Benchmark script
  - `poc/modular_retrieval_pipeline/README.md` - CLI options

  **Acceptance Criteria**:

  ```bash
  # Run benchmark (this takes ~2-5 minutes with warm cache)
  cd /home/fujin/Code/personal-library-manager
  
  python poc/modular_retrieval_pipeline/benchmark.py \
    --strategy modular-no-llm \
    --questions poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json \
    --output poc/modular_retrieval_pipeline/results/informed_benchmark_$(date +%Y%m%d_%H%M%S).json
  
  # Verify results file was created
  ls -la poc/modular_retrieval_pipeline/results/informed_benchmark_*.json | tail -1
  ```

  **Commit**: NO

---

- [x] 4. Report results and compare to baseline

  **What to do**:
  - Extract key metrics from results JSON
  - Compare against realistic baseline (36% Hit@5)
  - Print summary table

  **Must NOT do**:
  - Do NOT create visualization files
  - Do NOT modify results JSON

  **Recommended Agent Profile**:
  - **Category**: `quick`
  - **Skills**: `[]`
  - Reason: Data extraction and reporting

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Blocks**: None (final task)
  - **Blocked By**: Task 3

  **References**:
  - `poc/modular_retrieval_pipeline/benchmark_results_realistic.json` - Baseline results (36%)

  **Acceptance Criteria**:

  ```bash
  # Extract and compare results
  python3 << 'EOF'
  import json
  import glob

  # Get latest informed benchmark results
  results_files = sorted(glob.glob('poc/modular_retrieval_pipeline/results/informed_benchmark_*.json'))
  if not results_files:
      print("FAIL: No results file found")
      exit(1)

  with open(results_files[-1]) as f:
      informed = json.load(f)

  # Baseline comparison
  baseline_hit5 = 36.0  # From realistic benchmark
  baseline_hit1 = 24.0
  baseline_mrr = 0.28

  informed_hit5 = informed.get('hit_at_5_rate', 0)
  informed_hit1 = informed.get('hit_at_1_rate', 0)
  informed_mrr = informed.get('mrr', 0)

  print("\n" + "="*60)
  print("BENCHMARK COMPARISON: Informed vs Realistic Questions")
  print("="*60)
  print(f"\n{'Metric':<20} {'Realistic':<15} {'Informed':<15} {'Delta':<15}")
  print("-"*60)
  print(f"{'Hit@5':<20} {baseline_hit5:<15.1f} {informed_hit5:<15.1f} {informed_hit5 - baseline_hit5:+.1f}")
  print(f"{'Hit@1':<20} {baseline_hit1:<15.1f} {informed_hit1:<15.1f} {informed_hit1 - baseline_hit1:+.1f}")
  print(f"{'MRR':<20} {baseline_mrr:<15.3f} {informed_mrr:<15.3f} {informed_mrr - baseline_mrr:+.3f}")
  print("-"*60)

  improvement = informed_hit5 - baseline_hit5
  if improvement >= 30:
      print(f"\nRESULT: SIGNIFICANT IMPROVEMENT (+{improvement:.1f}%)")
  elif improvement >= 15:
      print(f"\nRESULT: MODERATE IMPROVEMENT (+{improvement:.1f}%)")
  elif improvement > 0:
      print(f"\nRESULT: SLIGHT IMPROVEMENT (+{improvement:.1f}%)")
  else:
      print(f"\nRESULT: NO IMPROVEMENT ({improvement:.1f}%)")

  print(f"\nResults saved to: {results_files[-1]}")
  EOF
  ```

  **Commit**: NO

---

## Commit Strategy

No commits required for this plan. This is a benchmark run that produces JSON output files.

---

## Success Criteria

### Verification Commands
```bash
# Verify informed_questions.json exists and is valid
python3 -c "import json; d=json.load(open('poc/chunking_benchmark_v2/corpus/kubernetes/informed_questions.json')); print(f'{len(d[\"questions\"])} questions')"

# Verify benchmark results exist
ls poc/modular_retrieval_pipeline/results/informed_benchmark_*.json
```

### Final Checklist
- [x] JSON file created with 25 questions
- [x] All questions have `realistic_q1`, `realistic_q2`, `doc_id` fields
- [x] Benchmark completed for all 50 question variants
- [x] Results JSON contains Hit@1, Hit@5, MRR metrics
- [x] Comparison report generated showing improvement over 36% baseline
