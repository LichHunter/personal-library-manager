# TODO: Approach - Adaptive-k (#10)

## Purpose

Implement and evaluate adaptive-k: dynamically select number of documents to retrieve.

---

## Preparation

### Prerequisites
- [ ] Previous P2 approach tested and documented
- [ ] Results documented in `results/06_crag.md`

### Research Phase
- [ ] Read approach description: `10_adaptive_k.md`
- [ ] Research k-selection methods
- [ ] Research score-based cutoffs
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement k-prediction or score-based cutoff
- [ ] Configure parameters
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run adaptive-k retrieval on ALL test queries
- [ ] Record selected k for each query
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average k selected
- [ ] Token efficiency (quality per token)

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics + avg k
- [ ] Procedural queries: all metrics + avg k
- [ ] Explanatory queries: all metrics + avg k
- [ ] Comparison queries: all metrics + avg k
- [ ] Troubleshooting queries: all metrics + avg k

### Step 5: Compare to Baseline and Best Previous
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best previous approach

---

## Conclusion

### Create Result Document
- [ ] Create `results/10_adaptive_k.md` with:
  - Approach summary
  - Configuration details
  - Metrics table (vs baseline AND best previous)
  - K-selection analysis by query type
  - Token efficiency analysis

### Apply Success Criteria
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best previous approach?
- [ ] Token efficiency improved?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] K-selection patterns analyzed
- [ ] Comparison complete
- [ ] Decision documented with justification
- [ ] Results file created: `results/10_adaptive_k.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/10_adaptive_k.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification

**Next:** Read and execute `15_multi_query.md`
