# TODO: Approach - Sentence Window (#09)

## Purpose

Implement and evaluate sentence window retrieval: index at sentence level, return surrounding window.

---

## Preparation

### Prerequisites
- [ ] Previous P1 approach tested and documented
- [ ] Results documented in `results/08_recursive_retriever.md`

### Research Phase
- [ ] Read approach description: `09_sentence_window.md`
- [ ] Research sentence splitting methods
- [ ] Research window storage strategies
- [ ] Evaluate storage overhead implications
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement sentence-level splitting
- [ ] Implement window metadata storage
- [ ] Implement window expansion at retrieval
- [ ] Smoke test: verify basic functionality works
- [ ] Note: May require re-indexing

### Step 2: Test on Full Query Set
- [ ] Run sentence window retrieval on ALL test queries
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved
- [ ] Storage overhead (vs chunk-based)

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics
- [ ] Procedural queries: all metrics
- [ ] Explanatory queries: all metrics
- [ ] Comparison queries: all metrics
- [ ] Troubleshooting queries: all metrics

### Step 5: Compare to Baseline and Best Previous
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best previous approach
- [ ] Analyze storage/performance tradeoff

---

## Conclusion

### Create Result Document
- [ ] Create `results/09_sentence_window.md` with:
  - Approach summary
  - Configuration details (window size)
  - Metrics table (vs baseline AND best previous)
  - Storage overhead analysis
  - Per-query-type breakdown

### Apply Success Criteria
From `evaluation_criteria.md`:
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best previous approach?
- [ ] Storage overhead acceptable?
- [ ] Latency acceptable?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Storage overhead documented
- [ ] Comparison complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/09_sentence_window.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/09_sentence_window.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `11_p1_review.md`
