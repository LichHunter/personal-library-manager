# Needle-in-Haystack Benchmark for enriched_hybrid_llm

## Context

### Original Request
Create a "needle-in-haystack" benchmark to test the `enriched_hybrid_llm` retrieval strategy on 200 Kubernetes documentation files. Generate 20 questions from a single "needle" document using Sonnet/Haiku with full document context, run retrieval benchmark, and have the **executing agent (Sisyphus) manually grade results** and produce a comprehensive report.

### Interview Summary
**Key Discussions**:
- Use existing 200 K8s docs from `corpus/kubernetes_sample_200/` (already created, ~2.2MB)
- Select ONE "needle" document, generate 20 questions with full doc in LLM context
- Index all 200 docs with enriched_hybrid_llm strategy
- Query with 20 questions, collect retrieved chunks
- **Executing agent (Sisyphus) manually grades each retrieval** by reading questions, expected answers, and retrieved chunks
- Agent produces final report with grades and reasoning

**Research Findings**:
- `enriched_hybrid_llm` achieves 88.7% coverage, 93% on baseline questions
- Indexing is FREE (local BGE + YAKE/spaCy), ~$0.0015/query for Haiku rewrite
- Existing `test_gems.py` and `manual_test.py` have reusable patterns
- 200 docs → estimated ~880 chunks (needs verification)

### Metis Review
**Identified Gaps** (addressed):
- Grading approach: **Manual grading by executing agent** (not automated LLM calls from code)
- Needle selection: Use manual selection of medium-complexity doc (~5-10KB)
- Question scope: All 20 questions from needle doc only (cleaner signal)
- Success threshold: Use existing 7.5/10 for VALIDATED

---

## Work Objectives

### Core Objective
Build and run a needle-in-haystack benchmark that tests whether `enriched_hybrid_llm` can find specific content from 1 document buried among 200 K8s docs.

### Concrete Deliverables
- `poc/chunking_benchmark_v2/benchmark_needle_haystack.py` - Main benchmark script
- `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Generated questions + expected answers
- `poc/chunking_benchmark_v2/results/needle_haystack_report.md` - Final graded report

### Definition of Done
- [x] 200 K8s docs indexed successfully with enriched_hybrid_llm
- [x] 20 questions generated from needle document
- [x] All 20 retrievals complete with chunks captured
- [x] All 20 gradings complete with scores and reasoning
- [x] Final report shows aggregate metrics + per-question breakdown
- [x] Command `python benchmark_needle_haystack.py` runs end-to-end

### Must Have
- Single needle document selection (manual or via flag)
- 20 questions generated with FULL needle doc in context
- Sonnet for question generation (better quality than Haiku)
- **Manual grading by executing agent** (reads output, grades each question 1-10)
- Comprehensive markdown report with per-question grades

### Must NOT Have (Guardrails)
- Do NOT modify existing `test_gems.py`, `manual_test.py`, or `enriched_hybrid_llm.py`
- Do NOT compare multiple strategies (single strategy benchmark only)
- Do NOT add new dependencies to pyproject.toml
- Do NOT generate more than 20 questions
- Do NOT add visualization/charts (markdown only)
- Do NOT create database storage (file-based only)
- Do NOT parallelize question generation (sequential for reproducibility)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES (existing test files, provider.py)
- **User wants tests**: Manual verification (this IS a test/benchmark)
- **Framework**: Python + existing infrastructure

### Manual Execution Verification

Each task includes verification commands and expected outputs.

---

## Task Flow

```
Task 1 (Select Needle) → Task 2 (Generate Questions) → Task 3 (Run Benchmark)
                                                              ↓
                      Task 5 (Report) ← Task 4 (Manual Grading by Agent)
```

## Parallelization

| Group | Tasks | Reason |
|-------|-------|--------|
| None | All sequential | Each task depends on previous output |

---

## TODOs

- [x] 1. Select needle document and validate

  **What to do**:
  - List all 200 docs in `corpus/kubernetes_sample_200/`
  - Select ONE document of medium complexity (5-10KB, good semantic density)
  - Validate: document has ≥2000 words (enough for 20 questions)
  - Save selection to `corpus/needle_selection.json` with doc ID, path, word count

  **Recommended needle criteria**:
  - Not too short (<2KB) - not enough content for 20 questions
  - Not too long (>20KB) - might exceed context limits
  - Good semantic variety - multiple topics/sections
  - Example candidates: `concepts_*.md`, `tasks_*.md` (avoid `reference_*` which are often short)

  **Must NOT do**:
  - Do not select multiple needles (single needle only)
  - Do not select `_index.md` files (too short)

  **Parallelizable**: NO (needed for Task 2)

  **References**:
  - `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` - All 200 docs
  - `poc/chunking_benchmark_v2/test_gems.py:48-75` - load_corpus() pattern

  **Acceptance Criteria**:
  - [ ] `corpus/needle_selection.json` exists with structure:
    ```json
    {
      "doc_id": "...",
      "filename": "...",
      "word_count": 2500,
      "char_count": 15000,
      "reason": "Selected for medium complexity and semantic variety"
    }
    ```
  - [ ] Word count ≥ 2000
  - [ ] File size between 5KB and 20KB

  **Commit**: YES
  - Message: `feat(benchmark): select needle document for haystack test`
  - Files: `corpus/needle_selection.json`

---

- [x] 2. Generate 20 REALISTIC HUMAN-STYLE questions from needle document

  **What to do**:
  - Create `benchmark_needle_haystack.py` with question generation function
  - Load needle document (full content)
  - Agent generates 20 questions that REAL USERS would ask
  - Prompt MUST include full document content (not summary)
  - Prompt MUST require expected answer for each question
  - Save to `corpus/needle_questions.json`

  **CRITICAL: Question Generation Strategy - HUMAN-LIKE QUERIES**
  
  Questions must simulate how REAL USERS search, NOT how documentation is written:
  
  | Bad (Doc-like) | Good (Human-like) |
  |----------------|-------------------|
  | "What is the default Topology Manager scope?" | "How do I set topology scope in kubelet?" |
  | "What are the four policies?" | "Which topology policy should I use for low latency apps?" |
  | "What is the max NUMA nodes?" | "Getting error about too many NUMA nodes, what's the limit?" |
  | "What feature gate enables Windows support?" | "How to enable topology manager on Windows nodes?" |
  
  **Question types to include (mix of all):**
  
  1. **Problem-based** (5 questions): User has an issue, needs solution
     - "My pod keeps failing with topology affinity error, why?"
     - "Containers in my pod are on different NUMA nodes, how to fix?"
  
  2. **How-to** (5 questions): User wants to accomplish something
     - "How do I force all containers onto a single NUMA node?"
     - "How to configure kubelet for topology-aware scheduling?"
  
  3. **Conceptual** (5 questions): User wants to understand something
     - "What's the difference between restricted and best-effort policies?"
     - "Why would I use pod scope vs container scope?"
  
  4. **Specific fact lookup** (5 questions): User needs a specific value/name
     - "What flag sets the topology manager policy?"
     - "Which K8s version made topology manager stable?"
  
  **Vocabulary transformation rules:**
  - Use synonyms: "CPU pinning" instead of "CPU isolation"
  - Use abbreviations: "k8s", "NUMA", "QoS"
  - Use problem descriptions: "pod rejected" instead of "admission failure"
  - Use casual language: "how do I", "what's the", "why does"
  - Include typos/variations: "numa node" vs "NUMA Node"

  **Must NOT do**:
  - Do NOT copy exact phrases from the documentation
  - Do NOT use formal documentation language
  - Do NOT make questions too easy by using exact doc terminology

  **Parallelizable**: NO (needed for Task 3)

  **References**:
  - `poc/chunking_benchmark_v2/manual_test.py:150-250` - generate_questions() pattern
  - `poc/chunking_benchmark_v2/enrichment/provider.py:call_llm()` - LLM interface
  - `poc/chunking_benchmark_v2/corpus/needle_selection.json` - Selected needle doc

  **Acceptance Criteria**:
  - [ ] `corpus/needle_questions.json` exists with 20 questions
  - [ ] Each question has: `id`, `question`, `expected_answer`, `difficulty`
  - [ ] Run: `python benchmark_needle_haystack.py --generate-questions`
  - [ ] Output: "Generated 20 questions for needle document: {doc_id}"
  - [ ] Manual verification: Questions are specific and answerable from doc

  **Commit**: YES
  - Message: `feat(benchmark): add question generation for needle-haystack test`
  - Files: `benchmark_needle_haystack.py`, `corpus/needle_questions.json`

---

- [x] 3. Run benchmark: index 200 docs and retrieve for all questions

  **What to do**:
  - Add indexing function to `benchmark_needle_haystack.py`
  - Load ALL 200 docs from `corpus/kubernetes_sample_200/`
  - Chunk with `MarkdownSemanticStrategy(target=400, min=50, max=800)`
  - Index with `enriched_hybrid_llm` strategy
  - For each of 20 questions:
    - Call `strategy.retrieve(question, k=5)`
    - Capture top-5 chunks with content
    - Capture latency
  - Save results to `results/needle_haystack_retrieval.json`

  **Must NOT do**:
  - Do not use FixedSizeStrategy (must use MarkdownSemanticStrategy)
  - Do not change enriched_hybrid_llm parameters
  - Do not skip caching (use EnrichmentCache)

  **Parallelizable**: NO (needed for Task 4)

  **References**:
  - `poc/chunking_benchmark_v2/test_gems.py:78-86` - chunk_documents() with MarkdownSemanticStrategy
  - `poc/chunking_benchmark_v2/test_gems.py:89-140` - run_retrieval() pattern
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - Strategy implementation
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Questions to query

  **Acceptance Criteria**:
  - [ ] Run: `python benchmark_needle_haystack.py --run-benchmark`
  - [ ] Output shows: "Indexed {N} chunks from 200 documents"
  - [ ] Output shows: "Completed 20/20 queries"
  - [ ] `results/needle_haystack_retrieval.json` exists with structure:
    ```json
    {
      "index_stats": {"num_chunks": 880, "...": "..."},
      "results": [
        {
          "question_id": "q_001",
          "question": "...",
          "expected_answer": "...",
          "retrieved_chunks": ["chunk content 1", "..."],
          "latency_ms": 150
        }
      ]
    }
    ```
  - [ ] All 20 questions have retrieved_chunks (non-empty)

  **Commit**: YES
  - Message: `feat(benchmark): add indexing and retrieval for needle-haystack`
  - Files: `benchmark_needle_haystack.py`, `results/needle_haystack_retrieval.json`

---

- [x] 4. **MANUAL GRADING BY EXECUTING AGENT** - Grade all retrieval results

  **What to do**:
  The executing agent (Sisyphus) will **manually read and grade** each of the 20 retrieval results.
  
  **This is NOT automated code** - the agent performs grading by:
  1. Reading `results/needle_haystack_retrieval.json`
  2. For each of 20 questions:
     - Read the question
     - Read the expected answer
     - Read ALL 5 retrieved chunks carefully
     - Determine: Does the retrieved content contain the expected answer?
     - Assign score 1-10 using rubric below
     - Write reasoning explaining the score
  3. Save graded results to `results/needle_haystack_graded.md`

  **Grading Rubric** (agent uses this to score):
  - **9-10**: Expected answer found verbatim or nearly verbatim in retrieved chunks
  - **7-8**: Answer concept present, slightly different wording
  - **5-6**: Partial answer, missing key details
  - **3-4**: Tangentially related content only
  - **1-2**: Completely irrelevant content, needle not found

  **Output format** (`results/needle_haystack_graded.md`):
  ```markdown
  # Needle-in-Haystack Grading Results
  
  ## Summary
  - Total Questions: 20
  - Average Score: X.X/10
  - Pass Rate (≥7): Y/20 (Z%)
  
  ## Detailed Grades
  
  ### Q1: {question}
  **Expected Answer**: {expected_answer}
  **Score**: X/10
  **Reasoning**: {why this score - what was found or not found}
  **Retrieved Chunk Excerpts**: (key relevant portions)
  
  [repeat for all 20]
  ```

  **Must NOT do**:
  - Do NOT automate grading via LLM API calls in code
  - Do NOT skip any questions
  - Do NOT assign scores without reading chunks

  **Parallelizable**: NO (needed for Task 5)

  **References**:
  - `poc/chunking_benchmark_v2/results/needle_haystack_retrieval.json` - Input: questions + retrieved chunks
  - `poc/chunking_benchmark_v2/manual_test.py:50-100` - GRADING_RUBRIC reference
  - `poc/chunking_benchmark_v2/corpus/needle_questions.json` - Original questions with expected answers

  **Acceptance Criteria**:
  - [ ] Agent reads all 20 questions and their retrieved chunks
  - [ ] `results/needle_haystack_graded.md` exists with all 20 grades
  - [ ] Each question has: score (1-10), reasoning (non-empty)
  - [ ] Summary section shows: average score, pass rate, total questions
  - [ ] Reasoning explains what was found/not found in chunks

  **Commit**: YES
  - Message: `feat(benchmark): manual grading results for needle-haystack`
  - Files: `results/needle_haystack_graded.md`

---

- [x] 5. Generate comprehensive report

  **What to do**:
  - The graded results are already in `results/needle_haystack_graded.md` from Task 4
  - Agent consolidates into final `results/needle_haystack_report.md` with:
    - Executive summary (pass rate, average score, verdict)
    - Needle document info (ID, title, word count)
    - Index statistics (chunk count, enrichment time)
    - Per-question breakdown (all 20 questions with full details)
    - Aggregate metrics
    - Conclusion and recommendations

  **Report structure**:
  ```markdown
  # Needle-in-Haystack Benchmark Report
  
  ## Executive Summary
  - **Verdict**: VALIDATED / INCONCLUSIVE / INVALIDATED
  - **Pass Rate**: X/20 (Y%)
  - **Average Score**: Z/10
  
  ## Configuration
  - Strategy: enriched_hybrid_llm
  - Documents: 200 Kubernetes docs
  - Chunks: N total
  - Needle: {doc_id}
  
  ## Results by Question
  
  ### Q1: {question}
  **Expected**: {expected_answer}
  **Score**: X/10
  **Reasoning**: {agent_reasoning}
  **Retrieved Chunks**: (truncated)
  
  [... repeat for all 20 ...]
  
  ## Aggregate Metrics
  - Score distribution histogram
  - Latency statistics
  - Pass/fail breakdown
  
  ## Conclusion
  ```

  **Must NOT do**:
  - Do not add charts/visualizations (markdown only)
  - Do not truncate reasoning
  - Do not skip failed questions in report

  **Parallelizable**: NO (final task)

  **References**:
  - `poc/chunking_benchmark_v2/manual_test.py:400-500` - Report generation pattern
  - `poc/chunking_benchmark_v2/results/FINAL_RESULTS/README.md` - Report style
  - `poc/chunking_benchmark_v2/results/needle_haystack_graded.md` - Input: graded results from Task 4

  **Acceptance Criteria**:
  - [ ] `results/needle_haystack_report.md` exists
  - [ ] Report contains: Executive Summary, Configuration, all 20 Q&A, Aggregate Metrics, Conclusion
  - [ ] Verdict is one of: VALIDATED (≥75% pass), INCONCLUSIVE (50-74%), INVALIDATED (<50%)
  - [ ] Report is readable and complete

  **Commit**: YES
  - Message: `feat(benchmark): final needle-haystack report`
  - Files: `results/needle_haystack_report.md`

---

- [x] 6. Verify all deliverables and commit

  **What to do**:
  - Verify all files exist and are complete
  - Review final report for accuracy
  - Commit all benchmark artifacts

  **Expected files**:
  - `corpus/needle_selection.json` - Selected needle doc info
  - `corpus/needle_questions.json` - 20 generated questions
  - `benchmark_needle_haystack.py` - Benchmark script (Tasks 1-3)
  - `results/needle_haystack_retrieval.json` - Retrieval results
  - `results/needle_haystack_graded.md` - Manual grades from Task 4
  - `results/needle_haystack_report.md` - Final report from Task 5

  **Parallelizable**: NO (final verification)

  **Acceptance Criteria**:
  - [ ] All 6 files exist
  - [ ] Report shows clear VALIDATED/INCONCLUSIVE/INVALIDATED verdict
  - [ ] All changes committed

  **Commit**: YES
  - Message: `feat(benchmark): complete needle-haystack benchmark`
  - Files: All benchmark files

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(benchmark): select needle document for haystack test` | `corpus/needle_selection.json` | File exists, word_count ≥ 2000 |
| 2 | `feat(benchmark): add question generation for needle-haystack test` | `benchmark_needle_haystack.py`, `corpus/needle_questions.json` | 20 questions generated |
| 3 | `feat(benchmark): add indexing and retrieval for needle-haystack` | `benchmark_needle_haystack.py`, `results/needle_haystack_retrieval.json` | 20 results captured |
| 4 | `feat(benchmark): manual grading results for needle-haystack` | `results/needle_haystack_graded.md` | 20 grades with reasoning |
| 5 | `feat(benchmark): final needle-haystack report` | `results/needle_haystack_report.md` | Report with verdict |
| 6 | `feat(benchmark): complete needle-haystack benchmark` | All files | All deliverables present |

---

## Success Criteria

### Verification Commands
```bash
cd poc/chunking_benchmark_v2

# Full benchmark
python benchmark_needle_haystack.py --all

# Verify outputs
ls -la corpus/needle_selection.json corpus/needle_questions.json
ls -la results/needle_haystack_*.json results/needle_haystack_report.md

# Check report
head -50 results/needle_haystack_report.md
```

### Final Checklist
- [x] All 6 tasks completed
- [x] 200 docs indexed successfully
- [x] 20 questions generated from needle
- [x] All 20 retrievals captured
- [x] All 20 questions manually graded by agent with reasoning
- [x] Final report generated with verdict (VALIDATED/INCONCLUSIVE/INVALIDATED)
- [x] No modifications to existing test_gems.py or manual_test.py
