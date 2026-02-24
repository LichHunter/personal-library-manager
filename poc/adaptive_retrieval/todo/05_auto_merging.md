# TODO: Approach - Auto-Merging (#02)

## Purpose

Implement and evaluate auto-merging: merge chunks to parent when majority of siblings are retrieved.

---

## Preparation

### Prerequisites
- [ ] Baseline measurement completed
- [ ] Previous approaches tested and documented
- [ ] Results documented in `results/15_parent_child.md`

### Research Phase
- [ ] Read approach description: `02_auto_merging.md`
- [ ] Research merge threshold strategies
- [ ] Research sibling chunk detection
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement sibling detection via heading_id
- [ ] Implement merge threshold logic
- [ ] Implement heading content retrieval for merged results
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run auto-merging retrieval on ALL test queries
- [ ] Record raw results for each query
- [ ] Track merge rate (how often merging occurs)

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved
- [ ] Merge rate (% of queries where merge occurred)

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics + merge rate
- [ ] Procedural queries: all metrics + merge rate
- [ ] Explanatory queries: all metrics + merge rate
- [ ] Comparison queries: all metrics + merge rate
- [ ] Troubleshooting queries: all metrics + merge rate

### Step 5: Compare to Baseline
- [ ] Calculate delta for each metric vs baseline
- [ ] Identify improvements and regressions
- [ ] Analyze merge behavior patterns

---

## Conclusion

### Create Result Document
- [ ] Create `results/02_auto_merging.md` with:
  - Approach summary (what was implemented)
  - Configuration used (merge threshold)
  - Metrics table (with baseline comparison)
  - Per-query-type breakdown
  - Merge rate analysis by query type
  - Improvements and regressions analysis

### Apply Success Criteria
From `evaluation_criteria.md`:
- [ ] Answer Success Rate improved ≥10%?
- [ ] No query type regressed >5%?
- [ ] Latency increase ≤500ms at p95?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Merge rate tracked and analyzed
- [ ] Comparison to baseline complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/02_auto_merging.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/02_auto_merging.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `06_p0_review.md`
