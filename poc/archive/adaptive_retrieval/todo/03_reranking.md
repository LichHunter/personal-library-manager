# TODO: Approach - Reranking (#01)

## Purpose

Implement and evaluate cross-encoder reranking as post-processing step after retrieval.

---

## Preparation

### Prerequisites
- [ ] Baseline measurement completed (`02_baseline.md`)
- [ ] Test query set available
- [ ] Baseline results documented in `results/00_baseline.md`

### Research Phase
- [ ] Read approach description: `01_reranking.md`
- [ ] Research available reranking models
- [ ] Research integration patterns
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement reranking in PLM retrieval pipeline
- [ ] Configure model and parameters
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run reranking retrieval on ALL test queries
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

### Step 5: Compare to Baseline
- [ ] Calculate delta for each metric vs baseline
- [ ] Identify improvements and regressions
- [ ] Note which query types benefited most

---

## Conclusion

### Create Result Document
- [ ] Create `results/01_reranking.md` with:
  - Approach summary (what was implemented)
  - Configuration used
  - Metrics table (with baseline comparison)
  - Per-query-type breakdown
  - Improvements and regressions analysis
  - Latency impact analysis

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
- [ ] Results file created: `results/01_reranking.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/01_reranking.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `04_parent_child.md`
