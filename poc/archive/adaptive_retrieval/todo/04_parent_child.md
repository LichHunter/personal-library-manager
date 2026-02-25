# TODO: Approach - Parent-Child (#15)

## Purpose

Implement and evaluate parent-child retrieval: search on chunks, return parent (heading) content.

---

## Preparation

### Prerequisites
- [ ] Baseline measurement completed
- [ ] Previous approach tested and documented
- [ ] Results documented in `results/01_reranking.md`

### Research Phase
- [ ] Read approach description: `15_parent_child.md`
- [ ] Research PLM's existing heading_id/doc_id structure
- [ ] Research how to retrieve heading content from chunks
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement parent content retrieval from chunk heading_ids
- [ ] Configure return level (heading vs document)
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run parent-child retrieval on ALL test queries
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved (expect increase)

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics
- [ ] Procedural queries: all metrics
- [ ] Explanatory queries: all metrics
- [ ] Comparison queries: all metrics
- [ ] Troubleshooting queries: all metrics

### Step 5: Compare to Baseline
- [ ] Calculate delta for each metric vs baseline
- [ ] Identify improvements and regressions
- [ ] Note context size changes

---

## Conclusion

### Create Result Document
- [ ] Create `results/15_parent_child.md` with:
  - Approach summary (what was implemented)
  - Configuration used (return level)
  - Metrics table (with baseline comparison)
  - Per-query-type breakdown
  - Context size analysis
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
- [ ] Comparison to baseline complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/15_parent_child.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/15_parent_child.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `05_auto_merging.md`
