# TODO: Approach - Adaptive-RAG Classifier (#03)

## Purpose

Implement and evaluate query complexity classifier that routes queries to appropriate retrieval granularity.

---

## Preparation

### Prerequisites
- [ ] P0 review completed (`06_p0_review.md`)
- [ ] Decision to continue to P1 documented
- [ ] Results documented in `results/phase2_p0_summary.md`

### Research Phase
- [ ] Read approach description: `03_adaptive_rag_classifier.md`
- [ ] Research classifier options (rule-based, trained, LLM)
- [ ] Research query complexity features
- [ ] Determine implementation approach

---

## Execution

### Step 1: Implement
- [ ] Implement query classifier (chosen approach)
- [ ] Implement routing logic based on classification
- [ ] Configure granularity retrieval for each class
- [ ] Smoke test: verify basic functionality works

### Step 2: Test on Full Query Set
- [ ] Run adaptive retrieval on ALL test queries
- [ ] Record classification for each query
- [ ] Record raw results for each query

### Step 3: Measure All Metrics
- [ ] MRR@10
- [ ] Recall@10
- [ ] Answer Success Rate
- [ ] Under-Retrieval Rate
- [ ] Latency (p50, p95, p99)
- [ ] Average tokens retrieved
- [ ] Classification accuracy vs oracle

### Step 4: Measure by Query Type
- [ ] Factoid queries: all metrics + classification accuracy
- [ ] Procedural queries: all metrics + classification accuracy
- [ ] Explanatory queries: all metrics + classification accuracy
- [ ] Comparison queries: all metrics + classification accuracy
- [ ] Troubleshooting queries: all metrics + classification accuracy

### Step 5: Compare to Baseline and Best P0
- [ ] Calculate delta for each metric vs baseline
- [ ] Calculate delta for each metric vs best P0 approach
- [ ] Analyze classification errors

---

## Conclusion

### Create Result Document
- [ ] Create `results/03_adaptive_classifier.md` with:
  - Approach summary (classifier type used)
  - Configuration details
  - Metrics table (vs baseline AND best P0)
  - Classification accuracy analysis
  - Per-query-type breakdown
  - Error analysis (misclassifications)

### Apply Success Criteria
From `evaluation_criteria.md`:
- [ ] Answer Success Rate improved ≥10% over baseline?
- [ ] Better than best P0 approach?
- [ ] Classification accuracy ≥70%?
- [ ] Latency acceptable?

### Decision
- [ ] Document decision: RECOMMEND / REJECT / NEEDS MODIFICATION
- [ ] Document justification for decision

### Verification Checklist
- [ ] Implementation complete and tested
- [ ] All metrics measured and documented
- [ ] Classification accuracy analyzed
- [ ] Comparison to baseline AND best P0 complete
- [ ] Success criteria evaluated
- [ ] Decision documented with justification
- [ ] Results file created: `results/03_adaptive_classifier.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Double-check `results/03_adaptive_classifier.md` is complete
2. [ ] Verify all metrics are documented
3. [ ] Verify decision is clearly stated with justification
4. [ ] Ensure implementation notes are captured for future reference

**Next:** Read and execute `08_iterative_expansion.md`
