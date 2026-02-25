# TODO: Baseline Measurement

## Purpose

Establish current PLM retrieval performance. This serves as the reference point for all approach comparisons.

---

## Preparation

### Prerequisites
- [ ] Test query set completed (`01_prepare_test_set.md`)
- [ ] PLM search service running
- [ ] Metrics calculation methods ready

### Reference Documents
- [ ] Read `evaluation_criteria.md` Section 2 (Baseline Definition)
- [ ] Read `evaluation_criteria.md` Section 3 (What We Must Test)

---

## Execution

### Step 1: Run Baseline Retrieval
- [ ] Run current PLM retrieval on ALL test queries
- [ ] Use default settings (no modifications)
- [ ] Record raw results for each query

### Step 2: Measure Retrieval Quality Metrics
- [ ] Calculate MRR@10
- [ ] Calculate Recall@10
- [ ] Calculate Precision@10

### Step 3: Measure Context Sufficiency Metrics
- [ ] For each query, generate answer using LLM with retrieved context
- [ ] Evaluate answers using LLM-as-judge
- [ ] Calculate Answer Success Rate
- [ ] Calculate Under-Retrieval Rate (queries that couldn't be answered)

### Step 4: Measure Efficiency Metrics
- [ ] Record latency for each query
- [ ] Calculate p50, p95, p99 latency
- [ ] Calculate average tokens retrieved per query

### Step 5: Measure by Query Type
- [ ] Calculate all metrics for Factoid queries
- [ ] Calculate all metrics for Procedural queries
- [ ] Calculate all metrics for Explanatory queries
- [ ] Calculate all metrics for Comparison queries
- [ ] Calculate all metrics for Troubleshooting queries

### Step 6: Identify Weak Points
- [ ] Which query types have lowest Answer Success Rate?
- [ ] Which query types have highest Under-Retrieval Rate?
- [ ] Where is the biggest gap vs oracle performance?

---

## Conclusion

### Deliverables
- [ ] Create `results/00_baseline.md` with:
  - Overall metrics table
  - Per-query-type breakdown
  - Latency distribution
  - Identified weak points
  - Comparison with oracle performance

### Baseline Metrics Summary Table
Document these values - they will be referenced by all approach tests:

| Metric | Value |
|--------|-------|
| MRR@10 | |
| Recall@10 | |
| Answer Success Rate | |
| Under-Retrieval Rate | |
| Latency p50 | |
| Latency p95 | |
| Avg Tokens Retrieved | |

### Verification Checklist
- [ ] All metrics calculated and documented
- [ ] Per-query-type breakdown complete
- [ ] Weak points identified
- [ ] Results file created in `results/00_baseline.md`

---

## Proceed to Next

Before proceeding:
1. [ ] Verify `results/00_baseline.md` is complete
2. [ ] Verify all baseline metrics are documented
3. [ ] Confirm test infrastructure is working correctly

**Next:** Read and execute `03_reranking.md`
