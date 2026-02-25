# Strategy v5: Cost-Effective NER Pipeline

## TL;DR

> **Quick Summary**: Replace expensive 2-Sonnet extraction with 3-Haiku + NER model, route by confidence tiers (HIGH/MEDIUM/LOW), and collect statistics on rejected LOW confidence terms to inform future Opus decision.
> 
> **Deliverables**:
> - `strategy_v5` preset in `hybrid_ner.py`
> - 3 Haiku extraction functions with diversified prompts
> - NER integration as +1 vote source (GLiNER zero-shot)
> - Confidence tier routing (HIGH → keep, MEDIUM → Sonnet, LOW → reject + log)
> - LOW confidence statistics logging (`low_confidence_stats.jsonl`)
> - Validation-first: 50-doc comparison before full implementation
> 
> **Estimated Effort**: Medium (2-3 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Task 1 (validation) → Task 2-4 (parallel) → Task 5-6 (parallel) → Task 7-8 (sequential)

---

## Context

### Original Request
User wants to reduce extraction costs by replacing 2 Sonnet calls with 3 Haiku calls + NER model, while maintaining quality through smart validation routing. Defer Opus decision by collecting statistics on rejected LOW confidence terms.

### Interview Summary
**Key Discussions**:
- Current architecture: 2 Sonnet + 1 Haiku extraction (~$0.06/doc)
- Proposed: 3 Haiku + NER as +1 vote (~$0.026/doc, ~57% savings)
- NER terms should NOT auto-keep (Oracle identified this as critical flaw)
- LOW confidence terms: reject but log statistics for later Opus decision
- Skip Sonnet validation if entity_ratio >= 0.7 (saves ~40% Sonnet calls)

**Research Findings**:
- Haiku recall risk: 2-5% worse than Sonnet on contextual disambiguation
- NER (BERTOverflow): 79% F1, but 40-60% FN on unseen entities
- Confidence aggregation: 0.85 HIGH, 0.70 MEDIUM industry standard
- Hybrid best practice: LLM-primary with NER validation, not reverse

### Metis Review
**Identified Gaps** (addressed):
- Validate "3 Haiku ≈ 2 Sonnet" assumption BEFORE full implementation
- Pin NER model version (no floating versions)
- Define quality floor (precision/recall minimums)
- Include rollback plan (feature flag)
- Edge cases: empty extraction, 3-way disagreement, NER adding terms

---

## Work Objectives

### Core Objective
Reduce extraction cost by ~57% while maintaining P >= 90%, R >= 88% through smart confidence routing and deferred expensive validation.

### Concrete Deliverables
- `strategy_v5` preset in `STRATEGY_PRESETS` dict
- `_extract_haiku_fewshot()` function (reusing retrieval logic with Haiku)
- `_extract_haiku_taxonomy()` function (reusing exhaustive logic with Haiku)
- `_extract_ner_gliner()` function (GLiNER zero-shot extraction)
- Modified `extract_hybrid()` with confidence tier routing
- `_log_low_confidence_stats()` function for JSONL output
- `low_confidence_stats.jsonl` output file per benchmark run

### Definition of Done
- [ ] `python benchmark_comparison.py --strategy strategy_v5 --n-docs 10` completes
- [ ] Precision >= 90%, Recall >= 88% on 10-doc benchmark
- [ ] `artifacts/results/low_confidence_stats.jsonl` contains valid entries
- [ ] Cost per doc reduced vs v4.3 (verify via token counting)

### Must Have
- Validation-first: 50-doc comparison before full implementation
- NER = +1 vote source (not auto-keep)
- Confidence tiers: HIGH/MEDIUM/LOW with defined routing
- Statistics logging for LOW confidence rejected terms
- Feature flag to revert to v4.3 if quality drops

### Must NOT Have (Guardrails)
- NO Opus integration (deferred - logging only)
- NO multiple NER model evaluation (GLiNER only for v1)
- NO Haiku prompt optimization (use existing prompts unchanged)
- NO statistics dashboard/visualization (JSONL + grep/jq only)
- NO threshold tuning infrastructure (ship with fixed values, tune later)
- NO extensive error handling beyond basic try/except
- NO abstraction "for future NER models"

---

## Verification Strategy

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
> ALL verification is executed by the agent using tools.

### Test Decision
- **Infrastructure exists**: YES (pytest in place)
- **Automated tests**: Tests-after (validate results first, then add regression tests)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY - ALL tasks)

**Verification Tool by Deliverable Type:**

| Type | Tool | How Agent Verifies |
|------|------|-------------------|
| **Python functions** | Bash (pytest) | Run test file, assert pass |
| **Benchmark results** | Bash (python + jq) | Run benchmark, parse JSON output |
| **Statistics file** | Bash (jq) | Validate JSONL schema |
| **Cost comparison** | Bash (python) | Calculate token counts |

---

## Execution Strategy

### Parallel Execution Waves

```
Wave 1 (Validation First - BLOCKING):
└── Task 1: Validate 3 Haiku ≈ 2 Sonnet assumption (50 docs)

Wave 2 (After validation passes):
├── Task 2: Add GLiNER NER extraction function
├── Task 3: Add Haiku few-shot extraction function
└── Task 4: Add Haiku taxonomy extraction function

Wave 3 (After Wave 2):
├── Task 5: Implement confidence tier routing
└── Task 6: Implement LOW confidence statistics logging

Wave 4 (Integration):
├── Task 7: Create strategy_v5 preset
└── Task 8: Benchmark and validate results

Critical Path: Task 1 → Tasks 2-4 → Tasks 5-6 → Tasks 7-8
```

### Dependency Matrix

| Task | Depends On | Blocks | Can Parallelize With |
|------|------------|--------|---------------------|
| 1 | None | 2, 3, 4, 5, 6, 7, 8 | None (validation gate) |
| 2 | 1 | 5, 7 | 3, 4 |
| 3 | 1 | 5, 7 | 2, 4 |
| 4 | 1 | 5, 7 | 2, 3 |
| 5 | 2, 3, 4 | 7 | 6 |
| 6 | 2, 3, 4 | 7 | 5 |
| 7 | 5, 6 | 8 | None |
| 8 | 7 | None | None |

---

## TODOs

- [ ] 1. Validate 3 Haiku vs 2 Sonnet Extraction (BLOCKING)

  **What to do**:
  - Create temporary test script `validate_haiku_recall.py`
  - Run current pipeline (2 Sonnet + 1 Haiku) on 50 docs, save extractions
  - Run modified pipeline (3 Haiku, no Sonnet) on same 50 docs, save extractions
  - Compare: count terms found by Sonnet but not by 3x Haiku
  - Calculate recall delta
  - Decision gate: If recall gap > 3%, STOP and reconsider architecture

  **Must NOT do**:
  - Don't modify production code yet
  - Don't optimize prompts (use existing unchanged)
  - Don't run on full test set

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Validation experiment requiring careful comparison methodology
  - **Skills**: `[]`
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (blocking gate)
  - **Blocks**: All other tasks
  - **Blocked By**: None

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:1272-1275` — Current extraction calls (retrieval_sonnet, exhaustive_sonnet, haiku_simple)
  - `benchmark_comparison.py:main()` — How benchmarks are run

  **API/Type References**:
  - `hybrid_ner.py:_extract_retrieval_fixed()` — Sonnet retrieval extraction
  - `hybrid_ner.py:_extract_exhaustive_sonnet()` — Sonnet exhaustive extraction
  - `hybrid_ner.py:_extract_haiku_simple()` — Haiku extraction

  **Acceptance Criteria**:

  - [ ] `validate_haiku_recall.py` script created and runs successfully
  - [ ] 50-doc comparison completed with results saved
  - [ ] Recall delta calculated and documented
  - [ ] Decision documented: PROCEED if gap <= 3%, STOP if gap > 3%

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Validation script runs successfully
    Tool: Bash
    Preconditions: .venv activated, API keys set
    Steps:
      1. cd poc/poc-1c-scalable-ner
      2. python validate_haiku_recall.py --n-docs 50 --seed 42
      3. Assert: exit code 0
      4. Assert: output contains "Recall delta:"
      5. Assert: output contains "Decision:"
    Expected Result: Script completes with clear decision
    Evidence: Terminal output captured

  Scenario: Recall gap is acceptable
    Tool: Bash
    Preconditions: Validation script completed
    Steps:
      1. Parse output for "Recall delta: X%"
      2. Assert: X <= 3.0
    Expected Result: Recall gap within acceptable range
    Evidence: Recall delta value logged
  ```

  **Commit**: NO (temporary validation script)

---

- [ ] 2. Add GLiNER NER Extraction Function

  **What to do**:
  - Install `gliner` package (add to pyproject.toml)
  - Create `_extract_ner_gliner()` function in `hybrid_ner.py`
  - Use GLiNER with labels: ["library", "framework", "programming language", "tool", "API", "data type"]
  - Return list of extracted terms (same format as Haiku extractors)
  - Handle model loading (cache globally to avoid reload per doc)

  **Must NOT do**:
  - Don't evaluate other NER models
  - Don't fine-tune GLiNER
  - Don't auto-keep NER terms (they're just +1 vote)

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Single function addition with clear pattern to follow
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 3, 4)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:294-315` — `_extract_haiku_simple()` function pattern
  - `hybrid_ner.py:456-504` — LLM extraction function pattern

  **External References**:
  - GLiNER: https://github.com/urchade/GLiNER
  - Usage: `model.predict_entities(text, labels, threshold=0.5)`

  **Acceptance Criteria**:

  - [ ] `gliner` added to pyproject.toml dependencies
  - [ ] `_extract_ner_gliner(doc: dict) -> list[str]` function exists
  - [ ] Function returns list of entity strings
  - [ ] Model is cached globally (not reloaded per call)
  - [ ] Threshold set to 0.5 (adjustable via constant)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: GLiNER extraction works
    Tool: Bash
    Preconditions: .venv activated, gliner installed
    Steps:
      1. python -c "from hybrid_ner import _extract_ner_gliner; print(_extract_ner_gliner({'text': 'I use React and Python for web development'}))"
      2. Assert: output contains "React" or "Python"
      3. Assert: exit code 0
    Expected Result: Returns list of extracted entities
    Evidence: Terminal output

  Scenario: Model caching works
    Tool: Bash
    Steps:
      1. python -c "
         from hybrid_ner import _extract_ner_gliner
         import time
         t1 = time.time()
         _extract_ner_gliner({'text': 'test'})
         t2 = time.time()
         _extract_ner_gliner({'text': 'test'})
         t3 = time.time()
         print(f'First call: {t2-t1:.2f}s, Second call: {t3-t2:.2f}s')
         assert t3-t2 < 0.5, 'Second call should be fast (cached)'
         "
    Expected Result: Second call is significantly faster
    Evidence: Timing output
  ```

  **Commit**: YES
  - Message: `feat(ner): add GLiNER extraction function`
  - Files: `hybrid_ner.py`, `pyproject.toml`
  - Pre-commit: `python -c "from hybrid_ner import _extract_ner_gliner"`

---

- [ ] 3. Add Haiku Few-Shot Extraction Function

  **What to do**:
  - Create `_extract_haiku_fewshot()` function
  - Reuse `RETRIEVAL_PROMPT_TEMPLATE` from existing code
  - Change model from Sonnet to Haiku
  - Keep same FAISS retrieval logic for few-shot examples

  **Must NOT do**:
  - Don't modify the prompt template
  - Don't change retrieval logic
  - Don't optimize for Haiku

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Copy existing function, change model parameter
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 4)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:317-343` — `RETRIEVAL_PROMPT_TEMPLATE`
  - `hybrid_ner.py:456-504` — `_extract_retrieval_fixed()` implementation
  - `retrieval_ner.py:safe_retrieve()` — FAISS retrieval function

  **Acceptance Criteria**:

  - [ ] `_extract_haiku_fewshot(doc, train_docs, index, model) -> tuple[list[str], str]` exists
  - [ ] Uses `claude-haiku-4-5-20250514` (or current Haiku model)
  - [ ] Reuses exact same prompt template as retrieval_sonnet
  - [ ] Returns same format as other extraction functions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Haiku few-shot extraction works
    Tool: Bash
    Preconditions: .venv activated, FAISS index exists
    Steps:
      1. python -c "
         from hybrid_ner import _extract_haiku_fewshot, load_artifacts
         train_docs, index, model, _, _ = load_artifacts()
         doc = train_docs[0]
         terms, raw = _extract_haiku_fewshot(doc, train_docs, index, model)
         print(f'Extracted {len(terms)} terms')
         assert len(terms) > 0
         "
    Expected Result: Returns non-empty list of terms
    Evidence: Term count output
  ```

  **Commit**: YES (groups with Task 4)
  - Message: `feat(extraction): add Haiku few-shot and taxonomy extractors`
  - Files: `hybrid_ner.py`

---

- [ ] 4. Add Haiku Taxonomy Extraction Function

  **What to do**:
  - Create `_extract_haiku_taxonomy()` function
  - Reuse `EXHAUSTIVE_PROMPT` from existing code
  - Change model from Sonnet to Haiku
  - Keep exact same prompt (no simplification)

  **Must NOT do**:
  - Don't simplify the prompt for Haiku
  - Don't modify taxonomy categories
  - Don't add new logic

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Copy existing function, change model parameter
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 2 (with Tasks 2, 3)
  - **Blocks**: Tasks 5, 7
  - **Blocked By**: Task 1

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:242-292` — `EXHAUSTIVE_PROMPT`
  - `hybrid_ner.py:506-530` — `_extract_exhaustive_sonnet()` implementation

  **Acceptance Criteria**:

  - [ ] `_extract_haiku_taxonomy(doc: dict) -> tuple[list[str], str]` exists
  - [ ] Uses Haiku model
  - [ ] Reuses exact same prompt as exhaustive_sonnet
  - [ ] Returns same format as other extraction functions

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Haiku taxonomy extraction works
    Tool: Bash
    Steps:
      1. python -c "
         from hybrid_ner import _extract_haiku_taxonomy
         terms, raw = _extract_haiku_taxonomy({'text': 'Using React with TypeScript and PostgreSQL database'})
         print(f'Extracted: {terms}')
         assert 'React' in terms or 'TypeScript' in terms or 'PostgreSQL' in terms
         "
    Expected Result: Extracts known technology terms
    Evidence: Extracted terms output
  ```

  **Commit**: YES (groups with Task 3)
  - Message: `feat(extraction): add Haiku few-shot and taxonomy extractors`
  - Files: `hybrid_ner.py`

---

- [ ] 5. Implement Confidence Tier Routing

  **What to do**:
  - Modify `extract_hybrid()` to use 4 sources: haiku_fewshot, haiku_taxonomy, haiku_simple, ner_gliner
  - Add confidence scoring logic:
    - HIGH: vote_count >= 3 OR entity_ratio >= 0.8 OR structural_pattern
    - MEDIUM: vote_count == 2 OR entity_ratio in [0.5, 0.8)
    - LOW: everything else
  - Route:
    - HIGH → add to final output (no validation)
    - MEDIUM → existing Sonnet validation (skip if entity_ratio >= 0.7)
    - LOW → reject + call `_log_low_confidence_stats()`

  **Must NOT do**:
  - Don't add Opus handling
  - Don't modify Sonnet validation logic (reuse existing)
  - Don't tune thresholds (use fixed values)

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Core logic change requiring careful integration
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 6)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 4

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:1293-1328` — Current confidence routing logic
  - `hybrid_ner.py:1301-1322` — Current source_count based routing

  **API/Type References**:
  - `StrategyConfig` dataclass — Add new fields for thresholds

  **Acceptance Criteria**:

  - [ ] `StrategyConfig` has fields: `high_confidence_vote_threshold`, `high_entity_ratio_threshold`, `medium_entity_ratio_threshold`
  - [ ] Terms with vote_count >= 3 skip validation
  - [ ] Terms with entity_ratio >= 0.8 skip validation
  - [ ] Terms with entity_ratio >= 0.7 skip Sonnet validation (optimization)
  - [ ] LOW confidence terms are rejected and logged (not included in output)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: HIGH confidence terms skip validation
    Tool: Bash
    Steps:
      1. Run benchmark with debug logging enabled
      2. Find a term with vote_count >= 3
      3. Assert: term did NOT go through Sonnet validation
      4. Assert: term IS in final output
    Expected Result: High confidence terms bypass validation
    Evidence: Debug log showing routing decision

  Scenario: MEDIUM terms go to Sonnet
    Tool: Bash
    Steps:
      1. Run benchmark with debug logging
      2. Find term with vote_count == 2 and entity_ratio < 0.7
      3. Assert: term went through Sonnet validation
    Expected Result: Medium terms validated
    Evidence: Debug log showing Sonnet call

  Scenario: LOW terms are rejected and logged
    Tool: Bash
    Steps:
      1. Run benchmark
      2. Check low_confidence_stats.jsonl exists
      3. Parse JSONL for rejected terms
      4. Assert: none of these terms in final output
    Expected Result: Low terms excluded but logged
    Evidence: JSONL file contents
  ```

  **Commit**: YES
  - Message: `feat(routing): implement confidence tier routing (HIGH/MEDIUM/LOW)`
  - Files: `hybrid_ner.py`

---

- [ ] 6. Implement LOW Confidence Statistics Logging

  **What to do**:
  - Create `_log_low_confidence_stats()` function
  - Log to `artifacts/results/low_confidence_stats.jsonl`
  - Schema per line:
    ```json
    {
      "term": "string",
      "doc_id": "string", 
      "vote_count": 1,
      "entity_ratio": 0.32,
      "sources": ["haiku_simple"],
      "in_gt": true,
      "timestamp": "2026-02-10T12:00:00Z"
    }
    ```
  - Append to file (don't overwrite between runs)
  - Add `--clear-stats` flag to benchmark CLI to reset file

  **Must NOT do**:
  - Don't build visualization
  - Don't add aggregation/summary logic
  - Don't create database storage

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple file I/O function
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: YES
  - **Parallel Group**: Wave 3 (with Task 5)
  - **Blocks**: Task 7
  - **Blocked By**: Tasks 2, 3, 4

  **References**:

  **Pattern References**:
  - `benchmark_comparison.py:save_results()` — How results are saved to JSON

  **Acceptance Criteria**:

  - [ ] `_log_low_confidence_stats(term, doc_id, vote_count, entity_ratio, sources, in_gt)` function exists
  - [ ] Writes valid JSONL to `artifacts/results/low_confidence_stats.jsonl`
  - [ ] Each line is valid JSON matching schema
  - [ ] File is appended to (not overwritten) across runs
  - [ ] `--clear-stats` flag added to CLI

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Statistics file is valid JSONL
    Tool: Bash
    Steps:
      1. Run benchmark with strategy_v5
      2. cat artifacts/results/low_confidence_stats.jsonl | head -5
      3. For each line: jq . (validate JSON)
      4. Assert: all lines parse successfully
    Expected Result: Valid JSONL file
    Evidence: jq output

  Scenario: Schema is correct
    Tool: Bash
    Steps:
      1. jq -r 'keys' artifacts/results/low_confidence_stats.jsonl | head -1
      2. Assert: contains ["doc_id", "entity_ratio", "in_gt", "sources", "term", "timestamp", "vote_count"]
    Expected Result: All required fields present
    Evidence: Field list output

  Scenario: Clear stats flag works
    Tool: Bash
    Steps:
      1. python benchmark_comparison.py --clear-stats
      2. Assert: low_confidence_stats.jsonl is empty or deleted
    Expected Result: Stats file cleared
    Evidence: File size or absence
  ```

  **Commit**: YES
  - Message: `feat(stats): add LOW confidence statistics logging`
  - Files: `hybrid_ner.py`, `benchmark_comparison.py`

---

- [ ] 7. Create strategy_v5 Preset

  **What to do**:
  - Add `strategy_v5` to `STRATEGY_PRESETS` dict
  - Configure:
    - `use_haiku_extraction: True` (new flag)
    - `use_ner_extraction: True` (new flag)
    - `high_confidence_vote_threshold: 3`
    - `high_entity_ratio_threshold: 0.8`
    - `medium_entity_ratio_threshold: 0.5`
    - `skip_validation_entity_ratio: 0.7`
    - `log_low_confidence: True`
  - Remove `common_word_seed_list` dependency (no vocabulary list!)
  - Keep existing filtering logic (stop words, gerunds, etc.)

  **Must NOT do**:
  - Don't add Opus configuration
  - Don't add prompt customization options
  - Don't create multiple v5 variants yet

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Configuration addition
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (Wave 4)
  - **Blocks**: Task 8
  - **Blocked By**: Tasks 5, 6

  **References**:

  **Pattern References**:
  - `hybrid_ner.py:161-181` — `strategy_v4_3` preset structure
  - `hybrid_ner.py:35-66` — `StrategyConfig` dataclass

  **Acceptance Criteria**:

  - [ ] `strategy_v5` exists in `STRATEGY_PRESETS`
  - [ ] `StrategyConfig` has new fields for v5 settings
  - [ ] NO `common_word_seed_list` in v5 config
  - [ ] `get_strategy_config("strategy_v5")` returns valid config

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: strategy_v5 is valid
    Tool: Bash
    Steps:
      1. python -c "from hybrid_ner import get_strategy_config; cfg = get_strategy_config('strategy_v5'); print(cfg)"
      2. Assert: exit code 0
      3. Assert: output shows use_haiku_extraction=True
    Expected Result: Config loads successfully
    Evidence: Config output
  ```

  **Commit**: YES
  - Message: `feat(strategy): add strategy_v5 preset with Haiku+NER extraction`
  - Files: `hybrid_ner.py`

---

- [ ] 8. Benchmark and Validate Results

  **What to do**:
  - Run full benchmark: `python benchmark_comparison.py --strategy strategy_v5 --n-docs 10 --seed 42`
  - Compare results to v4.3 baseline
  - Verify:
    - Precision >= 90% (quality floor)
    - Recall >= 88% (quality floor)
    - Cost reduction achieved (via token counting)
  - Document results in `artifacts/results/strategy_v5_benchmark.md`
  - If quality floor NOT met: document issues, do NOT merge

  **Must NOT do**:
  - Don't tune thresholds to hit targets (that's a separate task)
  - Don't run on full test set yet
  - Don't deploy if quality floor not met

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Running existing benchmark, documenting results
  - **Skills**: `[]`

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Sequential (final)
  - **Blocks**: None
  - **Blocked By**: Task 7

  **References**:

  **Pattern References**:
  - `benchmark_comparison.py` — Benchmark CLI
  - `artifacts/results/hybrid_strategy_v4_3_results.json` — Baseline results format

  **Test References**:
  - v4.3 baseline: P=92.8%, R=91.9%, H=7.2%

  **Acceptance Criteria**:

  - [ ] Benchmark completes without errors
  - [ ] Precision >= 90% (vs 92.8% baseline, max 2.8% drop)
  - [ ] Recall >= 88% (vs 91.9% baseline, max 3.9% drop)
  - [ ] `low_confidence_stats.jsonl` has entries
  - [ ] Results documented in markdown file
  - [ ] Cost comparison documented (tokens used)

  **Agent-Executed QA Scenarios**:

  ```
  Scenario: Benchmark runs successfully
    Tool: Bash
    Preconditions: All previous tasks complete
    Steps:
      1. cd poc/poc-1c-scalable-ner
      2. python benchmark_comparison.py --strategy strategy_v5 --n-docs 10 --seed 42
      3. Assert: exit code 0
      4. Assert: output contains "Precision:" and "Recall:"
    Expected Result: Benchmark completes with metrics
    Evidence: Terminal output with P/R/H values

  Scenario: Quality floor met
    Tool: Bash
    Steps:
      1. Parse benchmark output for precision and recall
      2. Assert: precision >= 0.90
      3. Assert: recall >= 0.88
    Expected Result: Metrics meet quality floor
    Evidence: P/R values logged

  Scenario: Statistics file populated
    Tool: Bash
    Steps:
      1. wc -l artifacts/results/low_confidence_stats.jsonl
      2. Assert: line count > 0
    Expected Result: LOW confidence terms were logged
    Evidence: Line count output

  Scenario: Cost reduction achieved
    Tool: Bash
    Steps:
      1. Compare token usage in v5 vs v4.3 results
      2. Calculate cost difference
      3. Assert: v5 cost < v4.3 cost
    Expected Result: Cost savings documented
    Evidence: Cost comparison in results markdown
  ```

  **Commit**: YES
  - Message: `docs(benchmark): add strategy_v5 benchmark results`
  - Files: `artifacts/results/strategy_v5_benchmark.md`, `artifacts/results/strategy_v5_results.json`

---

## Commit Strategy

| After Task | Message | Files |
|------------|---------|-------|
| 2 | `feat(ner): add GLiNER extraction function` | hybrid_ner.py, pyproject.toml |
| 3+4 | `feat(extraction): add Haiku few-shot and taxonomy extractors` | hybrid_ner.py |
| 5 | `feat(routing): implement confidence tier routing` | hybrid_ner.py |
| 6 | `feat(stats): add LOW confidence statistics logging` | hybrid_ner.py, benchmark_comparison.py |
| 7 | `feat(strategy): add strategy_v5 preset` | hybrid_ner.py |
| 8 | `docs(benchmark): add strategy_v5 benchmark results` | artifacts/results/*.md |

---

## Success Criteria

### Verification Commands
```bash
# Run benchmark
cd poc/poc-1c-scalable-ner
source .venv/bin/activate
python benchmark_comparison.py --strategy strategy_v5 --n-docs 10 --seed 42

# Expected output includes:
# Precision: >= 0.90
# Recall: >= 0.88
# low_confidence_stats.jsonl created

# Validate stats file
jq . artifacts/results/low_confidence_stats.jsonl | head -20

# Compare to baseline
cat artifacts/results/hybrid_strategy_v4_3_results.json | jq '.precision, .recall'
```

### Final Checklist
- [ ] Task 1 validation passed (recall gap <= 3%)
- [ ] All "Must Have" features present
- [ ] All "Must NOT Have" exclusions respected
- [ ] Precision >= 90%, Recall >= 88%
- [ ] LOW confidence stats logging works
- [ ] Cost reduction documented
- [ ] No vocabulary list dependency in v5

---

## Rollback Plan

If quality floor is not met after Task 8:

1. **Do NOT merge** strategy_v5 changes
2. **Document** specific failure mode (which terms are being missed/hallucinated)
3. **Options**:
   - Keep 1 Sonnet extractor + 2 Haiku (hybrid approach)
   - Tune thresholds (separate task)
   - Improve Haiku prompts (separate task)
4. **Revert** to strategy_v4_3 for production use

Feature flag approach: `strategy_v5` is a new preset, not replacing v4_3. Can switch back anytime via `--strategy strategy_v4_3`.
