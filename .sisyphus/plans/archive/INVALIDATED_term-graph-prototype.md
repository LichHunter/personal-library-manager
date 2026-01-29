# Term Graph Prototype: Query Enrichment for Vocabulary Mismatch

## Context

### Original Request
Build a term graph prototype to validate that query enrichment can improve retrieval performance on vocabulary mismatch failures. Test the hypothesis that adding canonical K8s terms to user queries bridges the gap between user language and documentation terminology.

### Interview Summary
**Key Discussions**:
- Analyzed 20 failed retrieval questions manually
- Found 70% (14/20) would benefit from term graph enrichment
- Core pattern: Users describe BEHAVIOR, docs describe MECHANISM
- Created 15 graph entries with 76 user synonyms across 7 domains

**Research Findings**:
- Existing retrieval pipeline has clear integration point at `enriched_hybrid_llm.py`
- Pattern to follow: `expand_query()` function
- SKOS-based model recommended but flat JSON sufficient for prototype

### Metis Review
**Identified Gaps** (addressed):
- Question set selection: Using 20 analyzed questions (targeted validation)
- Enrichment strategy: Additive (graph terms added to LLM-rewritten query)
- Success threshold: +10 percentage points minimum
- Edge cases: Multi-entry match, synonym already present, empty match, expansion limits

---

## Work Objectives

### Core Objective
Validate that a term graph can improve Hit@5 by at least 10 percentage points on the 20 analyzed vocabulary mismatch questions.

### Concrete Deliverables
- `poc/chunking_benchmark_v2/term_graph.json` - Term graph data file (15 entries, 76 synonyms)
- `poc/chunking_benchmark_v2/retrieval/term_graph.py` - Graph loading and enrichment module
- Modified `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py` - Integration
- `poc/chunking_benchmark_v2/results/term_graph_validation/` - Benchmark results with comparison

### Definition of Done
- [ ] Baseline Hit@5 recorded for 20 questions WITHOUT graph enrichment
- [ ] Graph enrichment Hit@5 recorded for same 20 questions
- [ ] Improvement is at least +10 percentage points
- [ ] No regression: questions that passed before still pass
- [ ] Per-question log shows which improved/regressed/unchanged
- [ ] Enrichment audit log shows: original → rewritten → enriched for each query

### Must Have
- Term graph with 15 entries from validation analysis
- Case-insensitive matching with word boundaries
- Additive enrichment (graph terms added to LLM-rewritten query)
- Expansion limit: max 5 terms per entry, 10 terms total
- Debug logging of enrichment decisions
- Before/after comparison on same 20 questions

### Must NOT Have (Guardrails)
- NO hierarchy (broader/narrower) - flat synonyms only
- NO weights or confidence scores
- NO modification to `benchmark_realistic_questions.py` core logic
- NO database or external storage - JSON file only
- NO automatic graph construction from documents
- NO fuzzy matching or stemming
- NO touching existing `DOMAIN_EXPANSIONS` dictionary
- NO changes to the 200-question full benchmark (yet)

---

## Verification Strategy (MANDATORY)

### Test Decision
- **Infrastructure exists**: YES - pytest available, but this is a benchmark POC
- **User wants tests**: Manual verification via benchmark comparison
- **Framework**: Benchmark-driven validation (before/after Hit@5)

### Verification Approach

Each task includes manual verification via benchmark output and file inspection.

**Evidence Required:**
- Benchmark output showing Hit@5 before and after
- Per-question results table
- Sample enrichment logs showing term additions

---

## Task Flow

```
Task 1 (Data) → Task 2 (Implementation) → Task 3 (Integration) → Task 4 (Benchmark) → Task 5 (Analysis)
```

## Parallelization

| Task | Depends On | Reason |
|------|------------|--------|
| 1 | None | Data preparation standalone |
| 2 | 1 | Needs graph data to load |
| 3 | 2 | Needs enrichment function |
| 4 | 3 | Needs integrated pipeline |
| 5 | 4 | Needs benchmark results |

---

## TODOs

- [x] 1. Create term graph JSON data file

  **What to do**:
  - Create `poc/chunking_benchmark_v2/term_graph.json`
  - Convert YAML from `term-graph-validation.md:118-292` to JSON format
  - Structure: `{"entries": [{"canonical": "...", "synonyms": ["...", "..."]}]}`
  - Include all 15 entries across 7 domains
  - Verify 76 total synonyms are captured

  **Must NOT do**:
  - Do not add hierarchy (broader/narrower)
  - Do not add weights or confidence scores
  - Do not add domain tags (keep it flat)

  **Parallelizable**: NO (first task)

  **References**:
  - `.sisyphus/drafts/term-graph-validation.md:118-292` - YAML source data for all 15 entries
  - `poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json` - Example JSON structure in codebase

  **Acceptance Criteria**:
  - [ ] File exists at `poc/chunking_benchmark_v2/term_graph.json`
  - [ ] Valid JSON (parse with `python -c "import json; json.load(open('term_graph.json'))"`)
  - [ ] Contains exactly 15 entries
  - [ ] Total synonym count is 76 (verify with jq or Python)
  - [ ] Each entry has `canonical` (string) and `synonyms` (list of strings)

  **Commit**: YES
  - Message: `feat(retrieval): add term graph seed data for query enrichment`
  - Files: `poc/chunking_benchmark_v2/term_graph.json`

---

- [x] 2. Implement term graph enrichment module

  **What to do**:
  - Create `poc/chunking_benchmark_v2/retrieval/term_graph.py`
  - Implement `load_term_graph(path: str) -> dict` - loads JSON, returns graph dict
  - Implement `enrich_from_graph(query: str, graph: dict, max_terms: int = 10) -> tuple[str, list[str]]`
    - Returns (enriched_query, list_of_added_terms) for logging
  - Use case-insensitive matching with word boundaries (regex `\b...\b`)
  - If synonym found in query, add canonical term + other synonyms (up to 5 per entry)
  - If multiple entries match, add from all (cap at max_terms total)
  - If synonym already in query, skip it (no duplicates)
  - If no matches, return original query unchanged

  **Must NOT do**:
  - Do not implement fuzzy matching or stemming
  - Do not add caching
  - Do not modify any existing files in this task

  **Parallelizable**: NO (depends on task 1)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:45-78` - `expand_query()` pattern to follow
  - `poc/chunking_benchmark_v2/retrieval/query_rewrite.py` - Existing query manipulation patterns
  - Python `re` module for word boundary matching: `r'\b' + re.escape(term) + r'\b'`

  **Acceptance Criteria**:
  - [ ] File exists at `poc/chunking_benchmark_v2/retrieval/term_graph.py`
  - [ ] `load_term_graph()` successfully loads `term_graph.json`
  - [ ] Manual test in Python REPL:
    ```python
    from retrieval.term_graph import load_term_graph, enrich_from_graph
    graph = load_term_graph("term_graph.json")
    result, added = enrich_from_graph("how to check permission", graph)
    assert "SubjectAccessReview" in result
    assert "SubjectAccessReview" in added
    ```
  - [ ] Empty match returns original query: `enrich_from_graph("unrelated query", graph)` returns original
  - [ ] Word boundary works: "permission" matches, "permissions" matches, "permissioned" does NOT match

  **Commit**: YES
  - Message: `feat(retrieval): implement term graph enrichment module`
  - Files: `poc/chunking_benchmark_v2/retrieval/term_graph.py`

---

- [x] 3. Integrate term graph into retrieval pipeline

  **What to do**:
  - Modify `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py`
  - Add import: `from .term_graph import load_term_graph, enrich_from_graph`
  - Load graph once at module level or in `__init__` (avoid reloading per query)
  - In `retrieve()` method, AFTER `rewrite_query()` call (around line 193), add:
    ```python
    enriched_query, added_terms = enrich_from_graph(rewritten_query, self.term_graph)
    if added_terms:
        logger.debug(f"Term graph added: {added_terms}")
    ```
  - Use `enriched_query` for subsequent retrieval steps
  - Add optional `use_term_graph: bool = True` parameter to enable/disable for A/B testing

  **Must NOT do**:
  - Do not modify `DOMAIN_EXPANSIONS` dictionary
  - Do not change BM25 or semantic retrieval logic
  - Do not modify RRF fusion parameters

  **Parallelizable**: NO (depends on task 2)

  **References**:
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:193-210` - Integration point after `rewrite_query()`
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:45-78` - `expand_query()` shows similar pattern
  - `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py:1-30` - Existing imports

  **Acceptance Criteria**:
  - [ ] Import statement added at top of file
  - [ ] Graph loaded (once, not per query)
  - [ ] `retrieve()` method calls `enrich_from_graph()` after `rewrite_query()`
  - [ ] Debug logging shows added terms when `logging.DEBUG` enabled
  - [ ] `use_term_graph=False` disables enrichment (for baseline comparison)
  - [ ] Manual test: Run retrieval on one question, verify enrichment in logs

  **Commit**: YES
  - Message: `feat(retrieval): integrate term graph enrichment into hybrid retrieval`
  - Files: `poc/chunking_benchmark_v2/retrieval/enriched_hybrid_llm.py`

---

- [x] 4. Run baseline and enriched benchmarks

  **What to do**:
  - Create `poc/chunking_benchmark_v2/benchmark_term_graph.py` - dedicated script for this validation
  - Extract the 20 analyzed questions from `realistic_questions.json` (filter by specific question texts)
  - Run benchmark twice:
    1. Baseline: `use_term_graph=False` 
    2. Enriched: `use_term_graph=True`
  - Save results to `results/term_graph_validation/`:
    - `baseline_results.json` - per-question results without enrichment
    - `enriched_results.json` - per-question results with enrichment
    - `comparison.json` - side-by-side with improvement delta
    - `enrichment_log.json` - for each query: original → rewritten → enriched
  - Use same random seed for reproducibility

  **Must NOT do**:
  - Do not modify `benchmark_realistic_questions.py`
  - Do not run on all 200 questions (only the 20 analyzed)
  - Do not change chunking or indexing

  **Parallelizable**: NO (depends on task 3)

  **References**:
  - `poc/chunking_benchmark_v2/benchmark_realistic_questions.py:689-892` - Benchmark runner pattern
  - `poc/chunking_benchmark_v2/corpus/kubernetes/realistic_questions.json` - Source questions
  - `.sisyphus/drafts/term-graph-validation.md:31-74` - The 20 specific questions to test

  **Acceptance Criteria**:
  - [ ] Script `benchmark_term_graph.py` exists and runs without error
  - [ ] Baseline results saved to `results/term_graph_validation/baseline_results.json`
  - [ ] Enriched results saved to `results/term_graph_validation/enriched_results.json`
  - [ ] `comparison.json` shows per-question: baseline_hit, enriched_hit, improved/regressed/unchanged
  - [ ] `enrichment_log.json` shows for each query: original, rewritten, enriched, added_terms
  - [ ] Both runs use identical index (no re-indexing between runs)

  **Commit**: YES
  - Message: `feat(benchmark): add term graph validation benchmark script`
  - Files: `poc/chunking_benchmark_v2/benchmark_term_graph.py`, `poc/chunking_benchmark_v2/results/term_graph_validation/`

---

- [ ] 5. Analyze results and document findings

  **What to do**:
  - Calculate Hit@5 for baseline and enriched
  - Determine if +10 percentage point threshold is met
  - Create `results/term_graph_validation/ANALYSIS.md` with:
    - Summary table: baseline vs enriched Hit@5
    - Per-question breakdown (20 rows)
    - Questions that improved (expected: ~13-14)
    - Questions that regressed (expected: 0)
    - Questions unchanged
    - Sample enrichment examples (3-5 interesting cases)
    - Conclusion: validated / not validated
    - Next steps if validated

  **Must NOT do**:
  - Do not draw conclusions beyond the 20-question sample
  - Do not claim production-readiness

  **Parallelizable**: NO (depends on task 4)

  **References**:
  - `results/term_graph_validation/comparison.json` - Source data for analysis
  - `results/term_graph_validation/enrichment_log.json` - Example enrichments
  - `.sisyphus/drafts/term-graph-validation.md:78-95` - Expected outcomes (70% would benefit)

  **Acceptance Criteria**:
  - [ ] `ANALYSIS.md` exists with all required sections
  - [ ] Hit@5 improvement is calculated and documented
  - [ ] Per-question table shows all 20 questions
  - [ ] At least 3 sample enrichment examples included
  - [ ] Clear conclusion: "Validated" if +10pp, "Not validated" otherwise
  - [ ] If validated, next steps section outlines path to full implementation

  **Commit**: YES
  - Message: `docs(benchmark): add term graph validation analysis`
  - Files: `poc/chunking_benchmark_v2/results/term_graph_validation/ANALYSIS.md`

---

## Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(retrieval): add term graph seed data for query enrichment` | `term_graph.json` | JSON parses, 15 entries |
| 2 | `feat(retrieval): implement term graph enrichment module` | `retrieval/term_graph.py` | REPL test passes |
| 3 | `feat(retrieval): integrate term graph enrichment into hybrid retrieval` | `retrieval/enriched_hybrid_llm.py` | Manual retrieval test |
| 4 | `feat(benchmark): add term graph validation benchmark script` | `benchmark_term_graph.py`, `results/` | Benchmark completes |
| 5 | `docs(benchmark): add term graph validation analysis` | `ANALYSIS.md` | All sections present |

---

## Success Criteria

### Verification Commands
```bash
cd poc/chunking_benchmark_v2

# Task 1: Verify JSON
python -c "import json; d=json.load(open('term_graph.json')); print(f'{len(d[\"entries\"])} entries')"
# Expected: 15 entries

# Task 2: Verify module
python -c "from retrieval.term_graph import load_term_graph, enrich_from_graph; g=load_term_graph('term_graph.json'); print(enrich_from_graph('check permission', g))"
# Expected: ('check permission SubjectAccessReview ...', ['SubjectAccessReview', ...])

# Task 4: Run benchmark
python benchmark_term_graph.py
# Expected: Creates results in results/term_graph_validation/

# Task 5: Verify analysis
cat results/term_graph_validation/ANALYSIS.md | head -50
# Expected: Shows Hit@5 comparison
```

### Final Checklist
- [ ] All "Must Have" present
- [ ] All "Must NOT Have" absent
- [ ] Baseline Hit@5 documented
- [ ] Enriched Hit@5 documented
- [ ] Improvement >= +10 percentage points (or documented why not)
- [ ] No regressions (0 questions went from pass to fail)
- [ ] Clear validation conclusion
