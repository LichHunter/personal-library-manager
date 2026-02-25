# TODO: Approach - Multi-Query (#14)

## Purpose

Implement and evaluate multi-query retrieval: generate query variants, retrieve for each, merge results.

---

## Preparation

### Prerequisites
- [ ] Previous P2 approach tested and documented
- [ ] Results documented in `results/10_adaptive_k.md`

### Research Phase
- [ ] Read approach description: `14_multi_query.md`
- [ ] Research query variant generation
- [ ] Research result merging strategies
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement query variant generation (LLM or rules)
- [ ] Implement parallel retrieval for variants
- [ ] Implement result merging and deduplication
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run multi-query retrieval on ALL test queries
- [ ] Record generated variants
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Unique document rate (new docs from variants)
- [ ] LLM calls (if using LLM for variants)

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
- [ ] Create `results/14_multi_query.md` with:
  - Approach summary
  - Configuration details
  - Metrics table (vs baseline AND best previous)
  - Variant effectiveness analysis
  - Per-query-type breakdown
  - Cost analysis (LLM calls)

### Apply Success Criteria
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best previous approach?
- [ ] Latency acceptable?
- [ ] Cost justified by gains?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Comparison complete
- [ ] Decision documented with justification
- [ ] Results file created: `results/14_multi_query.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/14_multi_query.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification

**Next:** Read and execute `16_p2_review.md`
