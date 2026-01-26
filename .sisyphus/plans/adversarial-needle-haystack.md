# Adversarial Needle-in-Haystack Benchmark

## Context

### Original Request
Create a second needle-in-haystack benchmark with **20 adversarially hard questions** targeting known retrieval weaknesses. Questions should be designed to stress-test the `enriched_hybrid_llm` strategy while remaining **fair** (all answers must be retrievable from the document).

### Interview Summary
**Key Decisions**:
- Same needle document: `tasks_administer-cluster_topology-manager.md` (fair comparison to prior 90% test)
- Balanced question distribution: 5 version lookups, 5 comparisons, 5 negations, 5 vocabulary mismatches
- Expected pass rate: 50-70% (down from 90% human-style)
- Naming: `_adversarial` suffix on all output files
- Reuse existing `benchmark_needle_haystack.py` infrastructure

**Prior Test Results (Baseline)**:
- Human-style test: 90% pass rate (18/20), avg 8.45/10
- Failures: Q6 (v1.27 lookup - 4/10), Q12 (K8s 1.32 - 2/10)
- Both failures were VERSION LOOKUPS

**Root Causes from Edge Case Testing**:
- EMBEDDING_BLIND: 87% (semantic search misses relevant chunks)
- VOCABULARY_MISMATCH: 40% (query terms don't match doc terms)
- NEGATION_BLIND: 33% ("don't do X" framing weak)

### Metis Review
**Identified Gaps** (addressed in plan):
- Add pilot phase to validate question difficulty calibration
- Pre-verify all answers exist in retrievable chunks
- Document which questions target which weakness category
- Include category breakdown in final report (not just overall pass rate)
- Maintain identical chunking parameters for fair comparison

**Key Risk**: Vocabulary mismatch questions may be too artificial. Mitigation: Include mix of moderate and extreme mismatches.

---

## Work Objectives

### Core Objective
Create and run an adversarial needle-in-haystack benchmark that stress-tests `enriched_hybrid_llm` retrieval with questions targeting known weaknesses (version lookups, comparisons, negations, vocabulary mismatches).

### Concrete Deliverables
- `poc/chunking_benchmark_v2/corpus/needle_questions_adversarial.json` - 20 adversarial questions
- `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_retrieval.json` - Retrieval results
- `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_graded.md` - Manual grading
- `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_report.md` - Final report with category analysis

### Definition of Done
- [x] 20 adversarial questions generated with balanced distribution (5+5+5+5)
- [x] Each question tagged with weakness category (VERSION, COMPARISON, NEGATION, VOCABULARY)
- [x] All 20 retrievals complete with chunks captured
- [x] All 20 questions manually graded with scores and reasoning
- [x] Final report includes:
  - Overall pass rate and comparison to prior 90% test
  - Category breakdown (pass rate per question type)
  - Specific failure analysis
  - Recommendations for strategy improvement

### Must Have
- Balanced distribution: 5 version lookups, 5 comparisons, 5 negations, 5 vocabulary mismatches
- Same needle document as prior test (`tasks_administer-cluster_topology-manager.md`)
- Same chunking parameters (MarkdownSemanticStrategy, target=400, min=50, max=800)
- Same grading rubric (1-10 scale)
- Category tags on each question for analysis

### Must NOT Have (Guardrails)
- Do NOT modify retrieval strategy or chunking parameters
- Do NOT add questions with answers NOT in the document
- Do NOT use artificially impossible vocabulary mismatches
- Do NOT change the grading rubric to be more lenient
- Do NOT include multi-hop questions (not in confirmed distribution)
- Do NOT modify existing `benchmark_needle_haystack.py` beyond question loading
- Do NOT add new dependencies

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (existing benchmark script and infrastructure)
- **User wants tests**: Manual verification (this IS a benchmark)
- **Framework**: Python + existing infrastructure

### Manual Execution Verification
Each task includes specific verification commands and expected outputs.

---

## Task Flow

```
Task 1 (Pilot: 4 questions) → Task 2 (Full: 20 questions) → Task 3 (Run benchmark)
                                                                     ↓
                            Task 5 (Final report) ← Task 4 (Manual grading)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| None | All sequential | Each task depends on previous output |

---

## TODOs

- [x] 1. Pilot: Create and verify 4 adversarial questions (1 per category)

  **What to do**:
  - Create 4 pilot questions (1 version lookup, 1 comparison, 1 negation, 1 vocabulary mismatch)
  - Run quick retrieval test for each to verify answers are retrievable
  - Validate difficulty is calibrated (expected: 1-2 pass, 2-3 fail)
  - Save to `corpus/needle_questions_adversarial_pilot.json`

  **Pilot Questions (Verified in Document)**:
  
  | Category | Question | Expected Answer | Doc Line |
  |----------|----------|-----------------|----------|
  | VERSION | "What's the minimum kubernetes version requirement for topology manager?" | v1.18 | Line 10 |
  | COMPARISON | "How does the restricted policy differ from single-numa-node when a pod can't get preferred affinity?" | restricted rejects ANY non-preferred, single-numa-node rejects only if >1 NUMA needed | Lines 193-217 |
  | NEGATION | "Why is using more than 8 NUMA nodes not recommended?" | State explosion when enumerating NUMA affinities | Lines 266-271 |
  | VOCABULARY | "How do I configure CPU placement policy in kubelet?" | Use --topology-manager-policy flag | Line 166 |

  **Must NOT do**:
  - Do not create all 20 questions yet
  - Do not skip verification step

  **Parallelizable**: NO (needed to calibrate Task 2)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Question format from prior test
  - `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/tasks_administer-cluster_topology-manager.md` - Needle document
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py` - Benchmark script

  **Acceptance Criteria**:
  - [ ] `corpus/needle_questions_adversarial_pilot.json` exists with 4 questions
  - [ ] Each question has: `id`, `question`, `expected_answer`, `difficulty`, `category`, `doc_line`
  - [ ] Quick retrieval test shows answers ARE retrievable (even if not top-1)
  - [ ] Pilot pass rate between 25-75% (1-3 out of 4)
  - [ ] If pass rate is 100% or 0%, recalibrate question difficulty

  **Commit**: NO (pilot only, commit with full set in Task 2)

---

- [x] 2. Generate full set of 20 adversarial questions

  **What to do**:
  - Based on pilot results, generate remaining 16 questions (4 per category)
  - Merge pilot questions into full set
  - Each question tagged with category: `VERSION`, `COMPARISON`, `NEGATION`, `VOCABULARY`
  - Save to `corpus/needle_questions_adversarial.json`

  **Question Distribution (20 total)**:

  ### VERSION LOOKUP (5 questions) - Target EMBEDDING_BLIND
  All answers are specific version numbers/dates that semantic search struggles with:
  
  | ID | Question | Expected Answer | Doc Line | Difficulty |
  |----|----------|-----------------|----------|------------|
  | adv_v01 | "What's the minimum kubernetes version requirement for topology manager?" | v1.18 | Line 10 | medium |
  | adv_v02 | "Which Kubernetes release made Topology Manager GA/stable?" | v1.27 | Line 16 | hard |
  | adv_v03 | "When did the prefer-closest-numa-nodes option become generally available?" | Kubernetes 1.32 | Line 239 | hard |
  | adv_v04 | "In what k8s version did max-allowable-numa-nodes become GA?" | Kubernetes 1.35 | Line 258 | hard |
  | adv_v05 | "What's the default limit on NUMA nodes before kubelet refuses to start with topology manager?" | 8 | Line 264, 391 | medium |

  ### COMPARISON (5 questions) - Target multi-chunk retrieval
  Answers require synthesizing information about 2+ concepts:
  
  | ID | Question | Expected Answer | Doc Lines | Difficulty |
  |----|----------|-----------------|-----------|------------|
  | adv_c01 | "How does restricted policy differ from single-numa-node when pod can't get preferred affinity?" | restricted rejects any non-preferred; single-numa-node only rejects if >1 NUMA needed | 193-217 | hard |
  | adv_c02 | "What's the key difference between container scope and pod scope for topology alignment?" | container=individual alignment per container, no grouping; pod=groups all containers to common NUMA set | 118-151 | medium |
  | adv_c03 | "Compare what happens with none policy vs best-effort policy when NUMA affinity can't be satisfied" | none=no alignment attempted; best-effort=stores non-preferred hint, admits pod anyway | 179-191 | medium |
  | adv_c04 | "How does topology manager behavior differ for Guaranteed QoS pods with integer CPU vs fractional CPU?" | integer CPU gets topology hints from CPU Manager; fractional CPU gets default hint only | 329-376 | hard |
  | adv_c05 | "What's the difference between TopologyManagerPolicyBetaOptions and TopologyManagerPolicyAlphaOptions feature gates?" | Beta=enabled by default, Alpha=disabled by default; both control policy option visibility | 232-234 | medium |

  ### NEGATION (5 questions) - Target NEGATION_BLIND
  Questions framed as "what NOT to do" or "what can go wrong":
  
  | ID | Question | Expected Answer | Doc Lines | Difficulty |
  |----|----------|-----------------|-----------|------------|
  | adv_n01 | "Why is using more than 8 NUMA nodes not recommended with topology manager?" | State explosion when enumerating NUMA affinities; use max-allowable-numa-nodes at own risk | 266-271 | medium |
  | adv_n02 | "What happens to a pod that fails topology affinity check with restricted policy? Can it be rescheduled?" | Pod enters Terminated state; scheduler will NOT reschedule; need ReplicaSet/Deployment | 198-204 | medium |
  | adv_n03 | "Why can't the Kubernetes scheduler prevent pods from failing on nodes due to topology?" | Scheduler is not topology-aware; this is a known limitation | 396-397 | hard |
  | adv_n04 | "What's wrong with using container scope for latency-sensitive applications?" | Containers may end up on different NUMA nodes since there's no grouping | 119-121 | hard |
  | adv_n05 | "When does single-numa-node policy reject a pod that would be admitted by restricted?" | When pod needs resources from exactly 2+ NUMA nodes; restricted accepts any preferred, single-numa-node requires exactly 1 | 153-158, 209-217 | hard |

  ### VOCABULARY MISMATCH (5 questions) - Target VOCABULARY_MISMATCH
  Intentionally use synonyms/jargon that differs from document terms:
  
  | ID | Question | Expected Answer | Doc Term | Mismatch Term | Difficulty |
  |----|----------|-----------------|----------|---------------|------------|
  | adv_m01 | "How do I configure CPU placement policy in kubelet?" | --topology-manager-policy flag | "topology manager policy" | "CPU placement policy" | medium |
  | adv_m02 | "How do I enable NUMA awareness on Windows k8s nodes?" | Enable WindowsCPUAndMemoryAffinity feature gate | "Topology Manager support" | "NUMA awareness" | hard |
  | adv_m03 | "How does k8s coordinate resource co-location across multi-socket servers?" | Topology Manager acts as source of truth for CPU Manager and Device Manager | "topology aligned resource allocation" | "resource co-location" | hard |
  | adv_m04 | "What kubelet setting controls the granularity of resource alignment?" | topologyManagerScope (container or pod) | "scope" | "granularity of resource alignment" | hard |
  | adv_m05 | "How do I optimize inter-process communication latency for pods?" | Use pod scope with single-numa-node policy to eliminate inter-NUMA overhead | "latency-critical execution", "inter-NUMA communication overhead" | "inter-process communication latency" | hard |

  **Must NOT do**:
  - Do NOT copy exact phrases from documentation in questions
  - Do NOT create questions with answers not in the document
  - Do NOT make vocabulary mismatches impossibly obscure

  **Parallelizable**: NO (needed for Task 3)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json:1-170` - Question format
  - `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/tasks_administer-cluster_topology-manager.md` - Needle document
  - `poc/chunking_benchmark_v2/results/needle_haystack_graded.md` - Prior test results for Q6, Q12 failures

  **Acceptance Criteria**:
  - [ ] `corpus/needle_questions_adversarial.json` exists with exactly 20 questions
  - [ ] 5 questions per category: VERSION, COMPARISON, NEGATION, VOCABULARY
  - [ ] Each question has: `id`, `question`, `expected_answer`, `difficulty`, `category`, `doc_line`
  - [ ] All expected answers verified to exist in needle document
  - [ ] Questions use natural language (not copy-paste from doc)

  **Commit**: YES
  - Message: `feat(benchmark): add 20 adversarial needle-haystack questions`
  - Files: `corpus/needle_questions_adversarial.json`

---

- [x] 3. Run benchmark: retrieve for all 20 adversarial questions

  **What to do**:
  - Use existing indexed corpus (200 K8s docs, same as prior test)
  - For each of 20 adversarial questions:
    - Call `strategy.retrieve(question, k=5)`
    - Capture top-5 chunks with content
    - Capture latency
    - Flag if needle document chunks appear in results
  - Save results to `results/needle_haystack_adversarial_retrieval.json`

  **CRITICAL**: Use IDENTICAL parameters to prior test:
  - Chunking: `MarkdownSemanticStrategy(target=400, min=50, max=800)`
  - Strategy: `enriched_hybrid_llm`
  - k=5 results per query

  **Must NOT do**:
  - Do NOT change chunking or retrieval parameters
  - Do NOT modify the strategy
  - Do NOT use different k value

  **Parallelizable**: NO (needed for Task 4)

  **References**:
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py:run_benchmark()` - Benchmark runner
  - `poc/chunking_benchmark_v2/results/needle_haystack_retrieval.json` - Prior test results format
  - `poc/chunking_benchmark_v2/corpus/needle_questions_adversarial.json` - Questions to query

  **Acceptance Criteria**:
  - [ ] Run: `python benchmark_needle_haystack.py --questions corpus/needle_questions_adversarial.json --run-benchmark`
  - [ ] Output shows: "Completed 20/20 queries"
  - [ ] `results/needle_haystack_adversarial_retrieval.json` exists with structure:
    ```json
    {
      "config": {"strategy": "enriched_hybrid_llm", "...": "..."},
      "results": [
        {
          "question_id": "adv_v01",
          "question": "...",
          "expected_answer": "...",
          "category": "VERSION",
          "retrieved_chunks": [...],
          "needle_found": true/false,
          "latency_ms": 150
        }
      ]
    }
    ```
  - [ ] All 20 questions have retrieved_chunks (non-empty)

  **Commit**: YES
  - Message: `feat(benchmark): run adversarial needle-haystack retrieval`
  - Files: `results/needle_haystack_adversarial_retrieval.json`

---

- [x] 4. Manual grading by executing agent

  **What to do**:
  The executing agent (Sisyphus) will **manually read and grade** each of the 20 retrieval results.

  **Grading Process**:
  1. Read `results/needle_haystack_adversarial_retrieval.json`
  2. For each of 20 questions:
     - Read the question and expected answer
     - Read the category tag (VERSION, COMPARISON, NEGATION, VOCABULARY)
     - Read ALL 5 retrieved chunks carefully
     - Determine: Does the retrieved content contain the expected answer?
     - Assign score 1-10 using rubric below
     - Write reasoning explaining the score
     - Note if failure is due to: wrong chunks retrieved, right chunk but answer truncated, or vocabulary mismatch not bridged
  3. Save graded results to `results/needle_haystack_adversarial_graded.md`

  **Grading Rubric** (SAME as prior test):
  - **9-10**: Expected answer found verbatim or nearly verbatim in retrieved chunks
  - **7-8**: Answer concept present, slightly different wording
  - **5-6**: Partial answer, missing key details
  - **3-4**: Tangentially related content only
  - **1-2**: Completely irrelevant content, needle not found

  **Output format** (`results/needle_haystack_adversarial_graded.md`):
  ```markdown
  # Adversarial Needle-in-Haystack Grading Results
  
  ## Summary
  - Total Questions: 20
  - Average Score: X.X/10
  - Pass Rate (>=7): Y/20 (Z%)
  - Expected Pass Rate: 50-70%
  
  ## Results by Category
  | Category | Questions | Passed | Pass Rate | Avg Score |
  |----------|-----------|--------|-----------|-----------|
  | VERSION | 5 | ? | ?% | ?/10 |
  | COMPARISON | 5 | ? | ?% | ?/10 |
  | NEGATION | 5 | ? | ?% | ?/10 |
  | VOCABULARY | 5 | ? | ?% | ?/10 |
  
  ## Detailed Grades
  
  ### adv_v01 [VERSION]: {question}
  **Expected Answer**: {expected_answer}
  **Score**: X/10
  **Reasoning**: {why this score - what was found or not found}
  **Failure Analysis**: {if score <7: EMBEDDING_BLIND / VOCABULARY_MISMATCH / CHUNKING_ISSUE / OTHER}
  
  [repeat for all 20]
  ```

  **Must NOT do**:
  - Do NOT automate grading via LLM API calls
  - Do NOT skip any questions
  - Do NOT grade more leniently because questions are adversarial

  **Parallelizable**: NO (needed for Task 5)

  **References**:
  - `poc/chunking_benchmark_v2/results/needle_haystack_graded.md` - Prior test grading format
  - `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_retrieval.json` - Input

  **Acceptance Criteria**:
  - [ ] Agent reads all 20 questions and their retrieved chunks
  - [ ] `results/needle_haystack_adversarial_graded.md` exists with all 20 grades
  - [ ] Each question has: score (1-10), reasoning (non-empty), failure analysis (if score <7)
  - [ ] Summary section shows: average score, pass rate, results by category
  - [ ] Category breakdown table completed

  **Commit**: YES
  - Message: `feat(benchmark): manual grading for adversarial needle-haystack`
  - Files: `results/needle_haystack_adversarial_graded.md`

---

- [x] 5. Generate comprehensive final report

  **What to do**:
  Create `results/needle_haystack_adversarial_report.md` with:
  - Executive summary (verdict, pass rate, comparison to prior 90% test)
  - Category analysis (which question types failed most)
  - Failure pattern analysis (EMBEDDING_BLIND, VOCABULARY_MISMATCH, etc.)
  - Comparison to prior test (what changed, what stayed the same)
  - Recommendations for strategy improvement
  - Full per-question breakdown

  **Report structure**:
  ```markdown
  # Adversarial Needle-in-Haystack Benchmark Report
  
  ## Executive Summary
  - **Verdict**: EXPECTED / BETTER_THAN_EXPECTED / WORSE_THAN_EXPECTED
  - **Pass Rate**: X/20 (Y%) vs Prior Test: 18/20 (90%)
  - **Average Score**: Z/10 vs Prior Test: 8.45/10
  - **Expected Range**: 50-70%
  
  ## Key Findings
  1. [Most impactful finding]
  2. [Second finding]
  3. [Third finding]
  
  ## Category Analysis
  
  | Category | Pass Rate | Avg Score | Worst Question | Root Cause |
  |----------|-----------|-----------|----------------|------------|
  | VERSION | ?% | ?/10 | adv_v?? | EMBEDDING_BLIND |
  | COMPARISON | ?% | ?/10 | adv_c?? | ? |
  | NEGATION | ?% | ?/10 | adv_n?? | ? |
  | VOCABULARY | ?% | ?/10 | adv_m?? | VOCABULARY_MISMATCH |
  
  ## Comparison to Prior Test (90% Human-Style)
  
  | Metric | Prior Test | Adversarial Test | Change |
  |--------|------------|------------------|--------|
  | Pass Rate | 90% (18/20) | ?% (?/20) | -?% |
  | Avg Score | 8.45/10 | ?/10 | -? |
  | Version Lookup Pass | 60% (3/5) | ?% | ? |
  
  ## Failure Analysis
  
  ### By Root Cause
  | Root Cause | Count | % of Failures | Example |
  |------------|-------|---------------|---------|
  | EMBEDDING_BLIND | ? | ?% | adv_v02 |
  | VOCABULARY_MISMATCH | ? | ?% | adv_m02 |
  | NEGATION_BLIND | ? | ?% | adv_n03 |
  | CHUNKING_ISSUE | ? | ?% | ? |
  
  ## Recommendations
  1. [Specific improvement for EMBEDDING_BLIND]
  2. [Specific improvement for VOCABULARY_MISMATCH]
  3. [Specific improvement for NEGATION_BLIND]
  
  ## Detailed Results
  [All 20 questions with full breakdown from grading]
  
  ## Conclusion
  [Summary of what we learned about enriched_hybrid_llm weaknesses]
  ```

  **Must NOT do**:
  - Do NOT add charts/visualizations (markdown only)
  - Do NOT skip failure analysis
  - Do NOT omit comparison to prior test

  **Parallelizable**: NO (final task)

  **References**:
  - `poc/chunking_benchmark_v2/results/needle_haystack_report.md` - Prior test report format
  - `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_graded.md` - Grading input
  - `poc/chunking_benchmark_v2/corpus/ALL_TEST_QUESTIONS.md` - Root cause patterns

  **Acceptance Criteria**:
  - [ ] `results/needle_haystack_adversarial_report.md` exists
  - [ ] Report contains: Executive Summary, Category Analysis, Comparison to Prior Test, Failure Analysis, Recommendations
  - [ ] Verdict is one of: EXPECTED (50-70%), BETTER_THAN_EXPECTED (>70%), WORSE_THAN_EXPECTED (<50%)
  - [ ] All 20 questions included in detailed results
  - [ ] Recommendations are specific and actionable

  **Commit**: YES
  - Message: `feat(benchmark): final adversarial needle-haystack report`
  - Files: `results/needle_haystack_adversarial_report.md`

---

- [x] 6. Verify all deliverables and commit

  **What to do**:
  - Verify all files exist and are complete
  - Review final report for accuracy
  - Ensure all commits are made
  - Update `corpus/ALL_TEST_QUESTIONS.md` with adversarial questions

  **Expected files**:
  - `corpus/needle_questions_adversarial.json` - 20 adversarial questions
  - `results/needle_haystack_adversarial_retrieval.json` - Retrieval results
  - `results/needle_haystack_adversarial_graded.md` - Manual grades
  - `results/needle_haystack_adversarial_report.md` - Final report

  **Parallelizable**: NO (final verification)

  **Acceptance Criteria**:
  - [ ] All 4 files exist and are complete
  - [ ] Report shows clear verdict
  - [ ] All changes committed
  - [ ] `ALL_TEST_QUESTIONS.md` updated with adversarial question section

  **Commit**: YES
  - Message: `feat(benchmark): complete adversarial needle-haystack benchmark`
  - Files: All benchmark files, `corpus/ALL_TEST_QUESTIONS.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | (no commit - pilot only) | - | Pilot validates difficulty |
| 2 | `feat(benchmark): add 20 adversarial needle-haystack questions` | `corpus/needle_questions_adversarial.json` | 20 questions, 5 per category |
| 3 | `feat(benchmark): run adversarial needle-haystack retrieval` | `results/needle_haystack_adversarial_retrieval.json` | 20 results captured |
| 4 | `feat(benchmark): manual grading for adversarial needle-haystack` | `results/needle_haystack_adversarial_graded.md` | 20 grades with reasoning |
| 5 | `feat(benchmark): final adversarial needle-haystack report` | `results/needle_haystack_adversarial_report.md` | Report with verdict |
| 6 | `feat(benchmark): complete adversarial needle-haystack benchmark` | All files + `ALL_TEST_QUESTIONS.md` | All deliverables present |

---

## Success Criteria

### Verification Commands
```bash
cd poc/chunking_benchmark_v2

# Run full benchmark with adversarial questions
python benchmark_needle_haystack.py --questions corpus/needle_questions_adversarial.json --run-benchmark

# Verify outputs
ls -la corpus/needle_questions_adversarial.json
ls -la results/needle_haystack_adversarial_*.json results/needle_haystack_adversarial_*.md

# Check report
head -80 results/needle_haystack_adversarial_report.md
```

### Final Checklist
- [x] 20 adversarial questions generated (5 per category)
- [x] All 20 retrievals captured
- [x] All 20 questions manually graded with reasoning
- [x] Category breakdown shows per-type pass rates
- [x] Comparison to prior 90% test included
- [x] Failure analysis identifies root causes
- [x] Recommendations for improvement provided
- [x] Final verdict: EXPECTED (50-70%), BETTER, or WORSE

### Expected Outcomes
Based on prior test failures (version lookups at 50% fail rate):
- VERSION questions: Expected 2-3/5 pass (40-60%)
- COMPARISON questions: Expected 3-4/5 pass (60-80%)
- NEGATION questions: Expected 2-3/5 pass (40-60%)
- VOCABULARY questions: Expected 2-3/5 pass (40-60%)
- **Overall Expected**: 10-14/20 pass (50-70%)
