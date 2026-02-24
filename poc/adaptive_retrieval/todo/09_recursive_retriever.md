# TODO: Approach - Recursive Retriever (#08)

## Purpose

Implement and evaluate recursive retriever: follow references from chunks to parent content.

---

## Preparation

### Prerequisites
- [ ] Previous P1 approach tested and documented
- [ ] Results documented in `results/04_iterative_expansion.md`

### Research Phase
- [ ] Read approach description: `08_recursive_retriever.md`
- [ ] Research reference following patterns
- [ ] Research differences from parent-child
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement reference following from chunks
- [ ] Configure return level
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run recursive retrieval on ALL test queries
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics
- [ ] Procedural queries: all metrics
- [ ] Explanatory queries: all metrics
- [ ] Comparison queries: all metrics
- [ ] Troubleshooting queries: all metrics

### Step 5: Compare to Baseline and Best Previous
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best previous approach
- [ ] Compare specifically to parent-child (#15)

---

## Conclusion

### Create Result Document
- [ ] Create `results/08_recursive_retriever.md` with:
  - Approach summary
  - Configuration details
  - Metrics table (vs baseline AND best previous)
  - Comparison to parent-child approach
  - Per-query-type breakdown

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
- [ ] Comparison complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/08_recursive_retriever.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/08_recursive_retriever.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `10_sentence_window.md`
