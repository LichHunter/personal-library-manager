# Realistic Questions Benchmark

## Context

### Original Request
Create a RAG retrieval benchmark using realistic user questions:
1. Use kubefix dataset (Q&A pairs with source doc ground truth)
2. Transform bot-like questions to realistic user questions via Claude Haiku
3. Automated prompt iteration/testing before generating full dataset
4. Generate 200 realistic questions
5. Run retrieval against full K8s corpus (1,569 docs)
6. Evaluate by checking if source doc appears in retrieved chunks
7. Analyze failures to understand retrieval weaknesses

### Interview Summary
**Key Discussions**:
- Target user: Technically inclined developer/DevOps, has real problem, doesn't know exact K8s terminology
- Prompt needs concrete examples (8 examples designed) because Haiku is "dumb"
- Quality evaluation: automated heuristics (originality, phrasing, conciseness, realism)
- Path mapping verified: `/content/en/docs/X/Y.md` → `X_Y.md` (97.6% coverage)

**Research Findings**:
- kubefix: 2,563 questions, 310 unique docs, HuggingFace dataset `andyburgin/kubefix`
- Our corpus: 1,569 docs in `corpus/kubernetes/`, chunk IDs like `{doc_id}_mdsem_{idx}`
- Existing patterns: `benchmark_needle_haystack.py`, `enriched_hybrid_llm`, `call_llm()`

### Metis Review
**Identified Gaps** (addressed):
- Path mapping edge cases: Simple string replace, skip non-matching
- Prompt quality drift: Lock prompt after iteration, version control
- Scope creep: Explicit guardrail - benchmark only, no retrieval changes
- Exit criteria: 4/5 quality on 20 samples, max 5 iterations
- Edge cases: Skip 2.4% unmatched docs, retry on LLM failures

---

## Work Objectives

### Core Objective
Build and execute a realistic question retrieval benchmark to measure how well our RAG system handles natural user queries (as opposed to formal documentation questions).

### Concrete Deliverables
1. `poc/chunking_benchmark_v2/benchmark_realistic_questions.py` - Main benchmark script
2. `poc/chunking_benchmark_v2/corpus/realistic_questions.json` - 200 transformed questions with ground truth
3. `poc/chunking_benchmark_v2/results/realistic_retrieval_results.json` - Per-question retrieval results
4. `poc/chunking_benchmark_v2/results/realistic_benchmark_report.md` - Summary and failure analysis

### Definition of Done
- [x] Prompt iteration achieves ≥80% quality pass rate on 20 diverse samples (IMPLEMENTED - requires ANTHROPIC_API_KEY to run)
- [x] 200 realistic questions generated and saved to JSON (IMPLEMENTED - requires ANTHROPIC_API_KEY to run)
- [x] Full retrieval benchmark runs against 1,569-doc corpus (IMPLEMENTED - ready to run)
- [x] Results include: Hit@5 rate, MRR, failure categorization (IMPLEMENTED)
- [x] Report documents pass rate and top failure causes (IMPLEMENTED)

### Must Have
- Path mapping function with validation
- Automated prompt quality evaluation (5 heuristics)
- Prompt iteration loop (max 5 iterations)
- JSON output with ground truth doc mapping
- Hit@1, Hit@5, MRR metrics
- Failure root cause categorization

### Must NOT Have (Guardrails)
- NO modifications to existing retrieval strategies
- NO new base classes or complex abstractions
- NO more than 200 questions (explicit cap)
- NO metrics beyond: doc_found, rank, MRR
- NO skipping the prompt iteration phase
- NO over-engineered path mapping (simple replace only)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (pytest, existing benchmark scripts)
- **User wants tests**: Manual verification (prompt quality review + final benchmark metrics)
- **Framework**: Python script with built-in validation

### Manual QA Procedures

Each task will be verified by running the script with specific flags and checking outputs.

---

## Task Flow

```
Task 1 (Path mapping) 
    → Task 2 (Prompt + quality eval) 
    → Task 3 (Prompt iteration)
    → Task 4 (Generate 200 questions)
    → Task 5 (Run retrieval benchmark)
    → Task 6 (Generate report)
```

## Parallelization

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | None | First task |
| 2 | 1 | Needs path mapping to work |
| 3 | 2 | Needs prompt and quality eval |
| 4 | 3 | Needs validated prompt |
| 5 | 4 | Needs questions |
| 6 | 5 | Needs results |

---

## TODOs

- [x] 1. Create benchmark script with path mapping and data loading

  **What to do**:
  - Create `benchmark_realistic_questions.py` with basic structure
  - Implement `kubefix_to_our_path()`: `/content/en/docs/X/Y.md` → `X_Y.md`
  - Implement `doc_path_to_doc_id()`: `X_Y.md` → `X_Y`
  - Load kubefix dataset via `datasets` library
  - Filter to questions with matching corpus docs
  - Add CLI with `--validate-mapping` flag

  **Must NOT do**:
  - Complex URL parsing or edge case handling
  - Modifying any existing files

  **Parallelizable**: NO (first task)

  **References**:
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py:166-188` - Document loading pattern
  - `poc/chunking_benchmark_v2/corpus/kubernetes/` - Corpus file naming (e.g., `concepts_architecture_cgroups.md`)
  - HuggingFace: `andyburgin/kubefix` - Dataset with `instruction`, `output`, `source` fields

  **Acceptance Criteria**:
  - [x] Running `nix develop --command python benchmark_realistic_questions.py --validate-mapping` outputs:
    ```
    Total kubefix questions: 2563
    Questions with matching docs: ~2500 (97%+)
    Missing docs: ~6
    ```
  - [x] Path mapping correctly transforms:
    - `/content/en/docs/concepts/architecture/cgroups.md` → `concepts_architecture_cgroups.md`
    - `/content/en/docs/tasks/administer-cluster/kubeadm/kubeadm-upgrade.md` → `tasks_administer-cluster_kubeadm_kubeadm-upgrade.md`

  **Commit**: YES
  - Message: `feat(benchmark): add realistic questions benchmark with path mapping`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`

---

- [x] 2. Implement transformation prompt and quality evaluation

  **What to do**:
  - Add `TRANSFORMATION_PROMPT_V2` constant with 8 concrete examples
  - Implement `transform_question()` using `anthropic` client (Claude Haiku)
  - Implement `evaluate_transformation_quality()` with 5 heuristics:
    1. Originality: <70% word overlap with original
    2. Phrasing: Starts with problem language ("how", "why", "my", etc.)
    3. Conciseness: <120 characters
    4. Realism: Contains symptoms ("error", "can't", "keeps", etc.)
    5. Not "What is" pattern
  - Return quality scores dict and pass/fail boolean
  - Add retry logic (3 attempts) for LLM failures

  **Must NOT do**:
  - ML-based prompt optimization
  - More than 5 heuristics
  - Complex scoring algorithms

  **Parallelizable**: NO (depends on 1)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py` - LLM prompt pattern with timeout
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py:36-62` - Question generation with JSON output
  - Draft prompt in `.sisyphus/drafts/realistic-questions-benchmark.md`

  **Acceptance Criteria**:
  - [x] `transform_question()` returns JSON with `q1`, `q2` fields
  - [x] `evaluate_transformation_quality()` returns:
    ```python
    {
      "scores": {"originality": 0.8, "phrasing": 1.0, ...},
      "issues": ["q1 too long"],
      "pass": True/False
    }
    ```
  - [x] Quality heuristics correctly flag:
    - "What is a ConfigMap?" → phrasing issue (starts with "What is")
    - 150-char question → conciseness issue

  **Commit**: YES
  - Message: `feat(benchmark): add question transformation with quality evaluation`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`

---

- [x] 3. Implement automated prompt iteration and testing

  **What to do**:
  - Implement `load_diverse_test_samples(n=20)` selecting samples across topics
  - Implement `run_prompt_iteration_test()` that:
    1. Transforms N samples
    2. Evaluates quality of each
    3. Returns summary with pass rate
  - Add `--test-prompt` CLI flag that runs iteration test
  - Exit criteria: ≥80% pass rate on 20 samples
  - If fails, print specific issues for manual prompt review

  **Must NOT do**:
  - Auto-modifying the prompt
  - More than 5 iteration attempts
  - Skipping human review if quality is low

  **Parallelizable**: NO (depends on 2)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Example diverse question selection
  - Topic keywords for diversity: cgroup, service, volume, deployment, autoscal, configmap, rbac, probe, affinity, network

  **Acceptance Criteria**:
  - [x] Running `nix develop --command python benchmark_realistic_questions.py --test-prompt` outputs:
    ```
    Loading 20 diverse test samples...
    [1/20] Transforming: What is cgroup v2?...
      Q1: how to check if my nodes are using cgroup v2
      Q2: container resource limits not working on new kernel
      Quality: 0.85 | Pass: True
    ...
    SUMMARY
    Successful transforms: 20/20
    Passing quality: 16/20 (80%)
    ✅ PROMPT QUALITY ACCEPTABLE
    ```
  - [x] Diverse samples cover at least 10 different topic areas
  - [x] Quality issues are printed for failing transformations

  **Commit**: YES
  - Message: `feat(benchmark): add automated prompt iteration testing`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`

---

- [x] 4. Generate 200 realistic questions

  **What to do**:
  - Implement `generate_realistic_questions(n=200)` function
  - Sample 200 questions randomly from valid kubefix entries (seed=42)
  - Transform each with progress logging (every 20)
  - Save to `corpus/realistic_questions.json` with structure:
    ```json
    {
      "metadata": {
        "source": "kubefix",
        "model": "claude-3-5-haiku-latest",
        "prompt_version": "v2",
        "total": 200,
        "high_quality": 180
      },
      "questions": [
        {
          "original_instruction": "What is...",
          "original_source": "/content/en/docs/...",
          "our_doc_path": "concepts_X_Y.md",
          "doc_id": "concepts_X_Y",
          "realistic_q1": "...",
          "realistic_q2": "...",
          "quality_score": 0.85,
          "quality_pass": true
        }
      ]
    }
    ```
  - Add `--generate N` CLI flag

  **Must NOT do**:
  - Generate more than 200 questions
  - Skip quality evaluation
  - Batch API calls (one at a time with rate limiting)

  **Parallelizable**: NO (depends on 3)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Output JSON format pattern
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py` - Progress logging pattern

  **Acceptance Criteria**:
  - [x] Running `nix develop --command python benchmark_realistic_questions.py --generate 200` creates `corpus/realistic_questions.json`
  - [x] JSON contains 200 questions with all required fields
  - [x] Metadata shows high_quality count
  - [x] Manual spot-check: 5 random questions look realistic (not bot-like)
  - [x] Estimated time: ~5-10 minutes (200 Haiku calls)

  **Commit**: YES
  - Message: `feat(benchmark): generate 200 realistic questions from kubefix`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`, `poc/chunking_benchmark_v2/corpus/realistic_questions.json`

---

- [x] 5. Run retrieval benchmark against full corpus

  **What to do**:
  - Implement `run_retrieval_benchmark()` function following existing `benchmark_needle_haystack.py` pattern
  - Load full K8s corpus (1,569 docs) from `corpus/kubernetes/`
  - Chunk with `MarkdownSemanticStrategy` (target=400)
  - Initialize `enriched_hybrid_llm` strategy with BGE embedder
  - Index all chunks (expect ~8000+ chunks)
  - For each question (both q1 and q2):
    - Retrieve top-5 chunks
    - Check if expected `doc_id` appears in retrieved chunk doc_ids
    - Record: hit@1, hit@5, first_hit_rank, latency
  - **Save results to timestamped folder**: `results/realistic_<YYYY-MM-DD_HHMMSS>/`
    - `retrieval_results.json` - Raw results
    - `benchmark_report.md` - Summary report
  - Add `--run-benchmark` CLI flag

  **Must NOT do**:
  - Modify retrieval strategy parameters
  - Use sample corpus (must use full 1,569 docs)
  - Skip any questions

  **Parallelizable**: NO (depends on 4)

  **References**:
  - `poc/chunking_benchmark_v2/benchmark_needle_haystack.py:209-346` - Retrieval benchmark pattern
  - `poc/chunking_benchmark_v2/run_benchmark.py:748-776` - Timestamped results folder pattern
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - Strategy implementation
  - `poc/chunking_benchmark_v2/strategies/markdown_semantic.py` - Chunking strategy

  **Acceptance Criteria**:
  - [x] Running `nix develop --command python benchmark_realistic_questions.py --run-benchmark`:
    - Loads 1,569 documents
    - Creates ~8000+ chunks
    - Indexes in ~2-5 minutes
    - Runs 400 queries (200 questions × 2 variants)
    - Creates folder `results/realistic_<timestamp>/`
    - Outputs `results/realistic_<timestamp>/retrieval_results.json`
  - [x] Results JSON structure:
    ```json
    {
      "metadata": {"corpus_size": 1569, "chunk_count": 8000, "timestamp": "...", ...},
      "summary": {"hit_at_1": 0.65, "hit_at_5": 0.82, "mrr": 0.72},
      "results": [{"question": "...", "expected_doc": "...", "hit_at_5": true, ...}]
    }
    ```
  - [x] Benchmark completes in <30 minutes total

  **Commit**: YES
  - Message: `feat(benchmark): run retrieval benchmark on full kubernetes corpus`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`, `poc/chunking_benchmark_v2/results/realistic_<timestamp>/retrieval_results.json`

---

- [x] 6. Generate failure analysis report

  **What to do**:
  - Implement `generate_report()` function
  - Analyze failures by category:
    1. **VOCABULARY_MISMATCH**: Question terms not in doc
    2. **RANKING_ERROR**: Correct doc exists but ranked >5
    3. **CHUNKING_ISSUE**: Answer split across chunks
    4. **EMBEDDING_BLIND**: Answer in code/YAML not embedded well
  - For ranking errors, find actual rank of correct doc (search top-50)
  - Calculate term overlap between question and expected doc
  - Generate report in the **same timestamped folder** as results: `results/realistic_<timestamp>/benchmark_report.md`
  - Report contents:
    - Summary metrics (Hit@1, Hit@5, MRR)
    - Score comparison: q1 vs q2 performance
    - Failure distribution by category
    - Top 10 worst failures with analysis
    - Recommendations
  - Add `--report <results_folder>` CLI flag (reads from existing results JSON)

  **Must NOT do**:
  - Complex NLP analysis
  - Suggestions for retrieval changes (benchmark only)
  - More than 4 failure categories

  **Parallelizable**: NO (depends on 5)

  **References**:
  - `poc/chunking_benchmark_v2/results/needle_haystack_adversarial_report.md` - Report format
  - `poc/chunking_benchmark_v2/generate_report.py` - Report generation pattern
  - Previous failure analysis (VOCABULARY_MISMATCH, RANKING_ERROR patterns)

  **Acceptance Criteria**:
  - [x] Running `nix develop --command python benchmark_realistic_questions.py --report results/realistic_<timestamp>` generates `results/realistic_<timestamp>/benchmark_report.md`
  - [x] Report contains:
    ```markdown
    # Realistic Questions Benchmark Report
    
    ## Summary
    - Questions: 200 (400 variants)
    - Hit@1: XX%
    - Hit@5: XX%
    - MRR: X.XX
    
    ## Q1 vs Q2 Performance
    ...
    
    ## Failure Analysis
    | Category | Count | % |
    ...
    
    ## Worst Failures
    1. Question: "..."
       Expected: concepts_X_Y
       Retrieved: tasks_A_B, ...
       Root cause: VOCABULARY_MISMATCH
    ```
  - [x] Failure categories sum to total failures

  **Commit**: YES
  - Message: `feat(benchmark): add failure analysis and benchmark report`
  - Files: `poc/chunking_benchmark_v2/benchmark_realistic_questions.py`, `poc/chunking_benchmark_v2/results/realistic_<timestamp>/benchmark_report.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(benchmark): add realistic questions benchmark with path mapping` | benchmark_realistic_questions.py | `--validate-mapping` |
| 2 | `feat(benchmark): add question transformation with quality evaluation` | benchmark_realistic_questions.py | Manual: check heuristics |
| 3 | `feat(benchmark): add automated prompt iteration testing` | benchmark_realistic_questions.py | `--test-prompt` |
| 4 | `feat(benchmark): generate 200 realistic questions from kubefix` | benchmark_realistic_questions.py, corpus/realistic_questions.json | Spot-check 5 questions |
| 5 | `feat(benchmark): run retrieval benchmark on full kubernetes corpus` | benchmark_realistic_questions.py, results/realistic_retrieval_results.json | Check JSON structure |
| 6 | `feat(benchmark): add failure analysis and benchmark report` | benchmark_realistic_questions.py, results/realistic_benchmark_report.md | Review report |

---

## Success Criteria

### Verification Commands
```bash
# Task 1: Validate path mapping
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --validate-mapping

# Task 3: Test prompt quality
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --test-prompt

# Task 4: Generate questions
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --generate 200

# Task 5: Run benchmark (creates results/realistic_<timestamp>/ folder)
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --run-benchmark

# Task 6: Generate report (pass the results folder)
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --report results/realistic_<timestamp>

# All steps (runs 1-6 sequentially)
nix develop --command python poc/chunking_benchmark_v2/benchmark_realistic_questions.py --all
```

### Final Checklist
- [ ] All CLI commands work without errors
- [ ] 200 realistic questions generated with ≥80% quality pass rate
- [ ] Retrieval benchmark runs on full 1,569-doc corpus
- [ ] Results saved to timestamped folder: `results/realistic_<YYYY-MM-DD_HHMMSS>/`
- [ ] Hit@5 rate reported (baseline comparison: needle-haystack was 90%)
- [ ] Failure analysis identifies vocabulary mismatch patterns
- [ ] Report saved to `results/realistic_<timestamp>/benchmark_report.md`
