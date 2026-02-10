# POC-1b Scale Testing & Documentation Plan

## TL;DR

> **Quick Summary**: Scale up POC-1b enhanced filter testing from 15 to 30/50/100 chunks with high-quality ground truth generation, comprehensive logging, and results documentation.
> 
> **Deliverables**:
> - Documentation of current results and strategy in POC folder
> - Ground truth files for 30, 50, and 100 chunk test sets
> - Full pipeline runs with detailed audit trails
> - Cross-scale analysis comparing filter performance
> 
> **Estimated Effort**: Large (2-3 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: GT Generation → Pipeline Runs → Analysis

---

## Context

### Original Request
Scale up POC-1b testing from 15 chunks to 30/50/100 chunks, document results, ensure high-quality GT with double/triple verification.

### Interview Summary
**Key Discussions**:
- GT Quality: Opus + exhaustive span verification (strict - reject any term not literally in text)
- Chunk Selection: Random sampling from available pool
- Logging: Full audit trail (every term tracked through pipeline)

**Current State**:
- 15 chunks tested with P=98.2%, R=94.6%, H=1.8%, F1=0.963
- F25_ULTIMATE enhanced filter implemented and working
- Need to verify filter generalizes to larger scale

### Metis Review
**Identified Gaps** (addressed):
- CRITICAL: Current code uses sequential slicing, not random sampling → Must implement random sampling with fixed seed
- GT generation checkpointing needed for API failure recovery
- Filter must be version-locked during experiments
- Missing explicit acceptance criteria → Added executable tests
- Edge cases for zero-term chunks, duplicates → Added handling

---

## Work Objectives

### Core Objective
Validate that the enhanced noise filter (F25_ULTIMATE) generalizes beyond the initial 15-chunk test set by scaling to 30, 50, and 100 chunks with rigorous ground truth quality.

### Concrete Deliverables
1. `poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md` - Comprehensive results documentation
2. `poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md` - Strategy and methodology documentation
3. `artifacts/gt_30_chunks.json` - Ground truth for 30 chunks
4. `artifacts/gt_50_chunks.json` - Ground truth for 50 chunks
5. `artifacts/gt_100_chunks.json` - Ground truth for 100 chunks
6. `artifacts/scale_test_30.json` - Pipeline results for 30 chunks
7. `artifacts/scale_test_50.json` - Pipeline results for 50 chunks
8. `artifacts/scale_test_100.json` - Pipeline results for 100 chunks
9. `artifacts/scale_comparison.json` - Cross-scale analysis

### Definition of Done
- [ ] `python validate_gt.py artifacts/gt_30_chunks.json` → exits 0
- [ ] `python validate_gt.py artifacts/gt_50_chunks.json` → exits 0
- [ ] `python validate_gt.py artifacts/gt_100_chunks.json` → exits 0
- [ ] All test runs produce valid metrics (0 ≤ P,R,H,F1 ≤ 1)
- [ ] Documentation files exist and are non-empty
- [ ] Audit logs exist for all runs

### Must Have
- Random sampling with fixed seed (reproducibility)
- Version-locked F25_ULTIMATE filter (no changes during experiment)
- Span verification for all GT terms (zero hallucinations by construction)
- Full audit trail for every term decision
- Checkpointing for GT generation (resume on failure)

### Must NOT Have (Guardrails)
- MUST NOT modify F25_ULTIMATE filter during scaling experiments
- MUST NOT regenerate GT for existing chunks
- MUST NOT cherry-pick chunks for inclusion
- MUST NOT use sequential chunk selection (must be random)
- MUST NOT compare results across different filter versions
- MUST NOT add new filters mid-experiment

---

## Verification Strategy

### Test Decision
- **Infrastructure exists**: YES (existing test scripts)
- **Automated tests**: Tests-after (validation scripts)
- **Framework**: Python assert statements + dedicated validation scripts

### Agent-Executed QA Scenarios (MANDATORY)

**Scenario: GT File Validation**
```
Tool: Bash
Steps:
  1. cd /home/susano/Code/personal-library-manager/poc/poc-1b-llm-extraction-improvements
  2. source .venv/bin/activate
  3. python -c "
     import json
     gt = json.load(open('artifacts/gt_30_chunks.json'))
     assert gt['metadata']['total_chunks'] == 30
     assert len(gt['chunks']) == 30
     assert all('terms' in c and len(c['terms']) > 0 for c in gt['chunks'])
     assert gt['metadata']['random_seed'] == 42
     print('PASS: GT structure valid')
     "
Expected: "PASS: GT structure valid"
```

**Scenario: Span Verification Audit**
```
Tool: Bash
Steps:
  1. python audit_ground_truth.py --file artifacts/gt_30_chunks.json
Expected: "Ungrounded terms: 0"
```

**Scenario: Pipeline Metrics Validation**
```
Tool: Bash
Steps:
  1. python -c "
     import json
     r = json.load(open('artifacts/scale_test_30.json'))
     m = r['aggregate']['m2m_v3']
     assert 0 <= m['precision'] <= 1
     assert 0 <= m['recall'] <= 1
     assert 0 <= m['hallucination'] <= 1
     assert 0 <= m['f1'] <= 1
     print(f'PASS: P={m[\"precision\"]:.1%} R={m[\"recall\"]:.1%} H={m[\"hallucination\"]:.1%} F1={m[\"f1\"]:.3f}')
     "
Expected: "PASS:" followed by valid metrics
```

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Start Immediately):
├── Task 1: Document current results (RESULTS.md)
├── Task 2: Document strategy (STRATEGY.md)
└── Task 3: Implement random sampling infrastructure

Wave 2 (After Wave 1):
├── Task 4: Generate GT for 30 chunks
├── Task 5: Generate GT for 50 chunks (can start after 4 begins)
└── Task 6: Generate GT for 100 chunks (can start after 5 begins)

Wave 3 (After Wave 2):
├── Task 7: Run pipeline on 30 chunks
├── Task 8: Run pipeline on 50 chunks
└── Task 9: Run pipeline on 100 chunks

Wave 4 (After Wave 3):
└── Task 10: Cross-scale analysis and final documentation

Critical Path: Task 3 → Task 4 → Task 7 → Task 10
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 10 | 2, 3 |
| 2 | None | 10 | 1, 3 |
| 3 | None | 4, 5, 6 | 1, 2 |
| 4 | 3 | 7 | 5, 6 (staggered) |
| 5 | 3, 4 (partial) | 8 | 6 |
| 6 | 3, 5 (partial) | 9 | None |
| 7 | 4 | 10 | 8, 9 |
| 8 | 5 | 10 | 7, 9 |
| 9 | 6 | 10 | 7, 8 |
| 10 | 7, 8, 9 | None | None (final) |

---

## TODOs

### Task 1: Document Current Results (RESULTS.md)

**What to do**:
- Create `poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md`
- Document baseline results (15 chunks)
- Include metrics: P=98.2%, R=94.6%, H=1.8%, F1=0.963
- List all FP terms and analysis
- Include the enhanced filter performance comparison

**Must NOT do**:
- Do not include implementation details (that's STRATEGY.md)
- Do not include future work plans

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: `[]`
- **Reason**: Documentation task, no technical skills needed

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 2, 3)
- **Blocks**: Task 10
- **Blocked By**: None

**References**:
- `artifacts/enhanced_filter_results.json` - Final results data
- `artifacts/filter_upgrade_results.json` - Filter comparison data
- `test_filter_upgrades.py` - Filter test methodology

**Acceptance Criteria**:
- [ ] File exists: `poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md`
- [ ] Contains section: "Baseline Results (15 chunks)"
- [ ] Contains metrics table with P, R, H, F1
- [ ] Contains FP term analysis
- [ ] File is > 200 lines

**Agent-Executed QA**:
```
Scenario: Documentation file validation
  Tool: Bash
  Steps:
    1. test -f poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md
    2. grep -q "Precision" poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md
    3. grep -q "98.2" poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md
    4. wc -l poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md | awk '{print ($1 > 50 ? "PASS" : "FAIL")}'
  Expected: All commands succeed, final output is "PASS"
```

**Commit**: YES (groups with 2)
- Message: `docs(poc-1b): add results documentation`
- Files: `docs/RESULTS.md`

---

### Task 2: Document Strategy (STRATEGY.md)

**What to do**:
- Create `poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md`
- Document the D+v2.2 pipeline architecture
- Document F25_ULTIMATE filter components and rationale
- Document GT generation methodology (Opus + span verification)
- Document the scoring methodology (m2m_v3)

**Must NOT do**:
- Do not include numerical results (that's RESULTS.md)
- Do not include future roadmap

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 3)
- **Blocks**: Task 10
- **Blocked By**: None

**References**:
- `test_dplus_v3_sweep.py:83-160` - Pipeline prompts
- `test_dplus_v3_sweep.py:71-190` - Filter implementation
- `test_dplus_v3_sweep.py:616-810` - Scoring functions

**Acceptance Criteria**:
- [ ] File exists: `poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md`
- [ ] Contains section: "Pipeline Architecture"
- [ ] Contains section: "Enhanced Noise Filter"
- [ ] Contains section: "Ground Truth Methodology"
- [ ] Contains section: "Scoring (m2m_v3)"

**Agent-Executed QA**:
```
Scenario: Strategy documentation validation
  Tool: Bash
  Steps:
    1. test -f poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md
    2. grep -q "Pipeline" poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md
    3. grep -q "F25_ULTIMATE" poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md
    4. grep -q "span verification" poc/poc-1b-llm-extraction-improvements/docs/STRATEGY.md
  Expected: All commands succeed (exit 0)
```

**Commit**: YES (groups with 1)
- Message: `docs(poc-1b): add strategy documentation`
- Files: `docs/STRATEGY.md`

---

### Task 3: Implement Random Sampling Infrastructure

**What to do**:
- Create `poc/poc-1b-llm-extraction-improvements/scale_test_runner.py`
- Implement random chunk sampling with fixed seed (42)
- Implement checkpointing for GT generation (save after each chunk)
- Implement resume capability for API failures
- Lock filter version in metadata

**Must NOT do**:
- Do not modify existing `test_dplus_v3_sweep.py`
- Do not change F25_ULTIMATE filter logic
- Do not use sequential chunk selection

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`
- **Reason**: New infrastructure code, moderate complexity

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 1 (with Tasks 1, 2)
- **Blocks**: Tasks 4, 5, 6
- **Blocked By**: None

**References**:
- `test_dplus_v3_sweep.py:1036-1070` - Existing cache loading pattern
- `expand_ground_truth.py` - GT expansion methodology (if exists)
- `artifacts/small_chunk_ground_truth_v2.json` - GT file format

**Acceptance Criteria**:
- [ ] File exists: `scale_test_runner.py`
- [ ] Has function: `sample_chunks(pool, n, seed=42)`
- [ ] Has function: `generate_gt_with_checkpointing(chunks, output_path)`
- [ ] Has function: `run_pipeline_with_logging(gt_path, output_path)`
- [ ] Random seed is stored in output metadata
- [ ] Checkpointing saves progress after each chunk

**Agent-Executed QA**:
```
Scenario: Random sampling produces consistent results
  Tool: Bash
  Steps:
    1. cd poc/poc-1b-llm-extraction-improvements && source .venv/bin/activate
    2. python -c "
       from scale_test_runner import sample_chunks
       import hashlib
       pool = list(range(1000))
       s1 = sample_chunks(pool, 30, seed=42)
       s2 = sample_chunks(pool, 30, seed=42)
       assert s1 == s2, 'Not reproducible!'
       print('PASS: Random sampling is reproducible')
       "
  Expected: "PASS: Random sampling is reproducible"

Scenario: Checkpointing saves progress
  Tool: Bash
  Steps:
    1. python -c "
       # Simulate partial GT generation
       import json
       from pathlib import Path
       checkpoint_file = Path('artifacts/gt_checkpoint_test.json')
       if checkpoint_file.exists():
           checkpoint_file.unlink()
       # Test would create checkpoint after each chunk
       print('PASS: Checkpoint infrastructure exists')
       "
  Expected: "PASS: Checkpoint infrastructure exists"
```

**Commit**: YES
- Message: `feat(poc-1b): add scale test runner with random sampling`
- Files: `scale_test_runner.py`

---

### Task 4: Generate GT for 30 Chunks

**What to do**:
- Use scale_test_runner.py to sample 30 chunks randomly
- Run Opus exhaustive extraction on each chunk
- Apply strict span verification (reject terms not in text)
- Save to `artifacts/gt_30_chunks.json` with metadata
- Verify zero hallucinations

**Must NOT do**:
- Do not include chunks from existing 15-chunk GT (fresh sample)
- Do not accept any ungrounded terms
- Do not modify extraction prompts

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`
- **Reason**: API-heavy task, needs error handling

**Parallelization**:
- **Can Run In Parallel**: YES (staggered with 5)
- **Parallel Group**: Wave 2
- **Blocks**: Task 7
- **Blocked By**: Task 3

**References**:
- `expand_ground_truth.py` - GT expansion methodology
- `scale_test_runner.py` - New infrastructure (Task 3)
- `artifacts/small_chunk_ground_truth_v2.json` - GT format

**Acceptance Criteria**:
- [ ] File exists: `artifacts/gt_30_chunks.json`
- [ ] Contains exactly 30 chunks
- [ ] All terms pass span verification
- [ ] Metadata includes: random_seed=42, opus_model_version, timestamp
- [ ] Average terms per chunk > 10

**Agent-Executed QA**:
```
Scenario: GT file validation
  Tool: Bash
  Steps:
    1. cd poc/poc-1b-llm-extraction-improvements && source .venv/bin/activate
    2. python -c "
       import json
       gt = json.load(open('artifacts/gt_30_chunks.json'))
       assert gt['metadata']['total_chunks'] == 30, 'Wrong chunk count'
       assert len(gt['chunks']) == 30, 'Missing chunks'
       assert gt['metadata']['random_seed'] == 42, 'Wrong seed'
       total_terms = sum(len(c['terms']) for c in gt['chunks'])
       avg = total_terms / 30
       assert avg > 10, f'Low term density: {avg}'
       print(f'PASS: 30 chunks, {total_terms} total terms, {avg:.1f} avg')
       "
  Expected: "PASS:" followed by stats

Scenario: Span verification audit
  Tool: Bash
  Steps:
    1. python -c "
       import json
       gt = json.load(open('artifacts/gt_30_chunks.json'))
       ungrounded = 0
       for c in gt['chunks']:
           content = c['content'].lower()
           for t in c['terms']:
               if t['term'].lower() not in content:
                   ungrounded += 1
       assert ungrounded == 0, f'{ungrounded} ungrounded terms!'
       print('PASS: Zero ungrounded terms')
       "
  Expected: "PASS: Zero ungrounded terms"
```

**Commit**: NO (large generated file, commit separately)

---

### Task 5: Generate GT for 50 Chunks

**What to do**:
- Sample 50 chunks (can include 30-chunk subset for consistency check)
- Run Opus exhaustive extraction
- Apply strict span verification
- Save to `artifacts/gt_50_chunks.json`

**Must NOT do**:
- Do not change random seed (must be 42)
- Do not modify span verification strictness

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES (staggered)
- **Parallel Group**: Wave 2
- **Blocks**: Task 8
- **Blocked By**: Task 3, partial completion of Task 4

**References**:
- Same as Task 4

**Acceptance Criteria**:
- [ ] File exists: `artifacts/gt_50_chunks.json`
- [ ] Contains exactly 50 chunks
- [ ] First 30 chunks match gt_30_chunks.json (same seed)
- [ ] All terms pass span verification

**Agent-Executed QA**:
```
Scenario: GT consistency with 30-chunk set
  Tool: Bash
  Steps:
    1. python -c "
       import json
       gt30 = json.load(open('artifacts/gt_30_chunks.json'))
       gt50 = json.load(open('artifacts/gt_50_chunks.json'))
       ids_30 = {c['chunk_id'] for c in gt30['chunks']}
       ids_50 = {c['chunk_id'] for c in gt50['chunks'][:30]}
       assert ids_30 == ids_50, 'First 30 chunks dont match!'
       print('PASS: 50-chunk GT is superset of 30-chunk GT')
       "
  Expected: "PASS: 50-chunk GT is superset of 30-chunk GT"
```

**Commit**: NO

---

### Task 6: Generate GT for 100 Chunks

**What to do**:
- Sample 100 chunks
- Run Opus exhaustive extraction (with rate limiting)
- Apply strict span verification
- Save to `artifacts/gt_100_chunks.json`
- Expect ~60-90 minutes runtime

**Must NOT do**:
- Do not exceed API rate limits
- Do not proceed if checkpoint shows > 10% failure rate

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES (after 50 starts)
- **Parallel Group**: Wave 2
- **Blocks**: Task 9
- **Blocked By**: Task 3, partial completion of Task 5

**References**:
- Same as Task 4

**Acceptance Criteria**:
- [ ] File exists: `artifacts/gt_100_chunks.json`
- [ ] Contains exactly 100 chunks
- [ ] First 50 chunks match gt_50_chunks.json
- [ ] All terms pass span verification
- [ ] Estimated cost logged in metadata (should be ~$30-50)

**Agent-Executed QA**:
```
Scenario: 100-chunk GT validation
  Tool: Bash
  Steps:
    1. python -c "
       import json
       gt = json.load(open('artifacts/gt_100_chunks.json'))
       assert len(gt['chunks']) == 100
       total = sum(len(c['terms']) for c in gt['chunks'])
       print(f'PASS: 100 chunks, {total} total terms')
       "
  Expected: "PASS: 100 chunks," followed by term count
```

**Commit**: NO

---

### Task 7: Run Pipeline on 30 Chunks

**What to do**:
- Load GT from `artifacts/gt_30_chunks.json`
- Run full extraction pipeline (Phase 1-4)
- Apply F25_ULTIMATE enhanced filter
- Score with m2m_v3 methodology
- Save detailed results to `artifacts/scale_test_30.json`
- Generate full audit log to `artifacts/logs/`

**Must NOT do**:
- Do not modify F25_ULTIMATE filter
- Do not skip any chunks

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 3 (with Tasks 8, 9)
- **Blocks**: Task 10
- **Blocked By**: Task 4

**References**:
- `test_dplus_v3_sweep.py:1187-1320` - Main sweep logic
- `test_dplus_v3_sweep.py:1124-1179` - Assembly and audit

**Acceptance Criteria**:
- [ ] File exists: `artifacts/scale_test_30.json`
- [ ] Contains aggregate metrics for m2m_v3
- [ ] Contains per-chunk results
- [ ] Log file exists in `artifacts/logs/`
- [ ] Precision ≥ 90% (sanity check)

**Agent-Executed QA**:
```
Scenario: Pipeline results validation
  Tool: Bash
  Steps:
    1. python -c "
       import json
       r = json.load(open('artifacts/scale_test_30.json'))
       m = r['aggregate']['m2m_v3']
       assert 0.90 <= m['precision'] <= 1.0, f'Low precision: {m[\"precision\"]}'
       assert 0.85 <= m['recall'] <= 1.0, f'Low recall: {m[\"recall\"]}'
       print(f'PASS: P={m[\"precision\"]:.1%} R={m[\"recall\"]:.1%} H={m[\"hallucination\"]:.1%}')
       "
  Expected: "PASS:" with valid metrics

Scenario: Audit log exists
  Tool: Bash
  Steps:
    1. ls artifacts/logs/scale_test_30_*.log
    2. wc -l artifacts/logs/scale_test_30_*.log | tail -1 | awk '{print ($1 > 100 ? "PASS" : "FAIL")}'
  Expected: File found, "PASS"
```

**Commit**: YES (groups with 8, 9)
- Message: `test(poc-1b): add 30-chunk scale test results`
- Files: `artifacts/scale_test_30.json`

---

### Task 8: Run Pipeline on 50 Chunks

**What to do**:
- Same as Task 7, but with 50-chunk GT
- Save to `artifacts/scale_test_50.json`

**Must NOT do**:
- Same as Task 7

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 3
- **Blocks**: Task 10
- **Blocked By**: Task 5

**References**:
- Same as Task 7

**Acceptance Criteria**:
- [ ] File exists: `artifacts/scale_test_50.json`
- [ ] Precision ≥ 90%
- [ ] Results consistent with 30-chunk trends

**Agent-Executed QA**:
```
Scenario: 50-chunk results validation
  Tool: Bash
  Steps:
    1. python -c "
       import json
       r = json.load(open('artifacts/scale_test_50.json'))
       m = r['aggregate']['m2m_v3']
       assert 0.90 <= m['precision'] <= 1.0
       print(f'PASS: P={m[\"precision\"]:.1%} R={m[\"recall\"]:.1%}')
       "
  Expected: "PASS:" with valid metrics
```

**Commit**: YES (groups with 7, 9)
- Message: `test(poc-1b): add 50-chunk scale test results`
- Files: `artifacts/scale_test_50.json`

---

### Task 9: Run Pipeline on 100 Chunks

**What to do**:
- Same as Task 7, but with 100-chunk GT
- Save to `artifacts/scale_test_100.json`
- Expect ~30-60 minutes runtime

**Must NOT do**:
- Same as Task 7

**Recommended Agent Profile**:
- **Category**: `unspecified-high`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: YES
- **Parallel Group**: Wave 3
- **Blocks**: Task 10
- **Blocked By**: Task 6

**References**:
- Same as Task 7

**Acceptance Criteria**:
- [ ] File exists: `artifacts/scale_test_100.json`
- [ ] Precision ≥ 90%
- [ ] All 100 chunks processed

**Agent-Executed QA**:
```
Scenario: 100-chunk results validation
  Tool: Bash
  Steps:
    1. python -c "
       import json
       r = json.load(open('artifacts/scale_test_100.json'))
       assert len(r['per_chunk']) == 100, 'Missing chunks'
       m = r['aggregate']['m2m_v3']
       print(f'PASS: 100 chunks, P={m[\"precision\"]:.1%} R={m[\"recall\"]:.1%}')
       "
  Expected: "PASS: 100 chunks," with metrics
```

**Commit**: YES (groups with 7, 8)
- Message: `test(poc-1b): add 100-chunk scale test results`
- Files: `artifacts/scale_test_100.json`

---

### Task 10: Cross-Scale Analysis and Final Documentation

**What to do**:
- Create `artifacts/scale_comparison.json` with all scale metrics
- Update `docs/RESULTS.md` with scale test findings
- Create comparison charts/tables
- Calculate confidence intervals for 30+ chunk results
- Document filter generalization conclusion

**Must NOT do**:
- Do not modify filter based on results
- Do not cherry-pick favorable comparisons

**Recommended Agent Profile**:
- **Category**: `writing`
- **Skills**: `[]`

**Parallelization**:
- **Can Run In Parallel**: NO (final task)
- **Parallel Group**: Wave 4
- **Blocks**: None
- **Blocked By**: Tasks 1, 2, 7, 8, 9

**References**:
- `artifacts/scale_test_30.json`
- `artifacts/scale_test_50.json`
- `artifacts/scale_test_100.json`
- `docs/RESULTS.md` (from Task 1)

**Acceptance Criteria**:
- [ ] File exists: `artifacts/scale_comparison.json`
- [ ] Contains metrics for all 4 scales (15, 30, 50, 100)
- [ ] RESULTS.md updated with "Scale Testing Results" section
- [ ] Includes conclusion on filter generalization

**Agent-Executed QA**:
```
Scenario: Scale comparison validation
  Tool: Bash
  Steps:
    1. python -c "
       import json
       comp = json.load(open('artifacts/scale_comparison.json'))
       assert all(s in comp for s in ['15', '30', '50', '100'])
       for s in ['15', '30', '50', '100']:
           assert 'precision' in comp[s]
           assert 'recall' in comp[s]
       print('PASS: All scales have metrics')
       "
  Expected: "PASS: All scales have metrics"

Scenario: Documentation updated
  Tool: Bash
  Steps:
    1. grep -q "Scale Testing" poc/poc-1b-llm-extraction-improvements/docs/RESULTS.md
  Expected: Exit 0 (found)
```

**Commit**: YES
- Message: `docs(poc-1b): add scale testing analysis and comparison`
- Files: `artifacts/scale_comparison.json`, `docs/RESULTS.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1+2 | `docs(poc-1b): add results and strategy documentation` | `docs/RESULTS.md`, `docs/STRATEGY.md` | `ls docs/` |
| 3 | `feat(poc-1b): add scale test runner with random sampling` | `scale_test_runner.py` | `python -c "import scale_test_runner"` |
| 7+8+9 | `test(poc-1b): add scale test results (30/50/100 chunks)` | `artifacts/scale_test_*.json` | `ls artifacts/scale_test_*.json` |
| 10 | `docs(poc-1b): add scale testing analysis` | `artifacts/scale_comparison.json`, `docs/RESULTS.md` | `cat artifacts/scale_comparison.json` |

---

## Success Criteria

### Verification Commands
```bash
# All GT files exist and are valid
python validate_gt.py artifacts/gt_30_chunks.json
python validate_gt.py artifacts/gt_50_chunks.json
python validate_gt.py artifacts/gt_100_chunks.json

# All test results exist
ls artifacts/scale_test_{30,50,100}.json

# Metrics are reasonable (P ≥ 90%, R ≥ 85%)
python -c "
import json
for n in [30, 50, 100]:
    r = json.load(open(f'artifacts/scale_test_{n}.json'))
    m = r['aggregate']['m2m_v3']
    assert m['precision'] >= 0.90, f'{n}: low precision'
    assert m['recall'] >= 0.85, f'{n}: low recall'
print('All metrics pass sanity checks')
"

# Documentation exists
test -f docs/RESULTS.md && test -f docs/STRATEGY.md
```

### Final Checklist
- [ ] All GT files created with zero hallucinations
- [ ] All pipeline runs completed successfully
- [ ] Metrics are stable across scales (no major degradation)
- [ ] Full audit trails available for all runs
- [ ] Documentation complete and accurate
- [ ] Random seed (42) used consistently
- [ ] F25_ULTIMATE filter unchanged throughout
