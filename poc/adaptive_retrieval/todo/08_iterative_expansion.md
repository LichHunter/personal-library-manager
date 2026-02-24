# TODO: Approach - Iterative Expansion (#04)

## Purpose

Implement and evaluate iterative expansion: start with chunks, expand to headings/documents if insufficient.

---

## Preparation

### Prerequisites
- [ ] Previous P1 approach tested and documented
- [ ] Results documented in `results/03_adaptive_classifier.md`

### Research Phase
- [ ] Read approach description: `04_iterative_expansion.md`
- [ ] Research sufficiency check methods
- [ ] Research expansion strategies
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement sufficiency check (heuristic or LLM-based)
- [ ] Implement expansion logic (chunk → heading → document)
- [ ] Configure expansion parameters
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run iterative expansion on ALL test queries
- [ ] Record expansion level for each query
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved
- [ ] Expansion rate (% queries that expanded)
- [ ] Average expansion level

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics + expansion rate
- [ ] Procedural queries: all metrics + expansion rate
- [ ] Explanatory queries: all metrics + expansion rate
- [ ] Comparison queries: all metrics + expansion rate
- [ ] Troubleshooting queries: all metrics + expansion rate

### Step 5: Compare to Baseline and Best Previous
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best previous approach
- [ ] Analyze expansion patterns

---

## Conclusion

### Create Result Document
- [ ] Create `results/04_iterative_expansion.md` with:
  - Approach summary (sufficiency check method)
  - Configuration details
  - Metrics table (vs baseline AND best previous)
  - Expansion rate analysis
  - Per-query-type breakdown
  - Sufficiency check accuracy

### Apply Success Criteria
From `evaluation_criteria.md`:
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best previous approach?
- [ ] Latency acceptable?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Expansion rate tracked and analyzed
- [ ] Comparison complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/04_iterative_expansion.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/04_iterative_expansion.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `09_recursive_retriever.md`
