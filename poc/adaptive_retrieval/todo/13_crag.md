# TODO: Approach - CRAG (#06)

## Purpose

Implement and evaluate Corrective RAG: evaluate retrieval quality and apply corrections.

---

## Preparation

### Prerequisites
- [ ] Previous P2 approach tested and documented
- [ ] Results documented in `results/05_multi_scale.md`

### Research Phase
- [ ] Read approach description: `06_crag.md`
- [ ] Research retrieval quality evaluation methods
- [ ] Research correction strategies
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement retrieval quality evaluator
- [ ] Implement correction logic (expand, reformulate)
- [ ] Configure evaluation thresholds
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run CRAG retrieval on ALL test queries
- [ ] Record evaluation results for each query
- [ ] Record correction actions taken
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Correction rate (% queries corrected)
- [ ] Evaluator accuracy

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics
- [ ] Procedural queries: all metrics
- [ ] Explanatory queries: all metrics
- [ ] Comparison queries: all metrics
- [ ] Troubleshooting queries: all metrics

### Step 5: Compare to Baseline and Best Previous
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best previous approach

---

## Conclusion

### Create Result Document
- [ ] Create `results/06_crag.md` with:
  - Approach summary
  - Configuration details
  - Metrics table (vs baseline AND best previous)
  - Correction rate analysis
  - Per-query-type breakdown

### Apply Success Criteria
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best previous approach?
- [ ] Latency acceptable?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Correction rate analyzed
- [ ] Comparison complete
- [ ] Decision documented with justification
- [ ] Results file created: `results/06_crag.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/06_crag.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification

**Next:** Read and execute `14_adaptive_k.md`
