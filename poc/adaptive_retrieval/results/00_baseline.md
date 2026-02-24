# Baseline Measurement Results

**Date:** 2026-02-21 16:23:13
**Queries:** 229
**Labeled queries:** 65
**Configuration:** k=10, use_rewrite=False, judge=haiku

---

## Overall Metrics

| Metric | Value |
|--------|-------|
| MRR@10 (labeled) | 0.658 |
| Hit@5 (labeled) | 75.4% |
| Hit@10 (labeled) | 84.6% |
| Answer Success Rate | 92.1% |
| Avg Grade | 7.64/10 |
| Judge Failures | 0/229 |
| Latency p50 | 300ms |
| Latency p95 | 422ms |
| Latency p99 | 455ms |
| Avg Tokens Retrieved | 904 |
| Precision@10 (labeled) | 0.085 |

---

## Performance by Query Type

| Type | Count | MRR@10 | Hit@10 | Success Rate | Avg Grade |
|------|-------|--------|--------|--------------|-----------|
| explanatory | 53 | 0.615 | 76.9% | 71.7% | 6.25 |
| factoid | 54 | 0.635 | 82.4% | 96.3% | 7.98 |
| troubleshooting | 34 | 0.583 | 75.0% | 97.1% | 7.47 |
| procedural | 55 | 1.000 | 100.0% | 100.0% | 8.47 |
| comparison | 33 | 0.604 | 100.0% | 100.0% | 8.12 |

---

## Weak Points Identified

### Lowest Success Rates:

- **explanatory**: 71.7% success rate
- **factoid**: 96.3% success rate
- **troubleshooting**: 97.1% success rate

### Category Distribution:

- Partially Correct: 154 (67.2%)
- Correct: 57 (24.9%)
- Cannot Answer: 14 (6.1%)
- Incorrect: 4 (1.7%)

### Sample Failed Queries (Top 10):

- **adv_adv_n04** (explanatory): 3/10
  - Query: "What's wrong with using container scope for latency-sensitive applications?..."
  - Reasoning: The retrieved chunks do not contain any information about the drawbacks of using container scope for...

- **informed_q_016_q1** (factoid): 2/10
  - Query: "What is the purpose of the authentication flow in the aggregation layer?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of the authentication flow in ...

- **informed_q_021_q1** (factoid): 3/10
  - Query: "What is the purpose of the `kubectl logs pods/job-wq-2-7r7b2` command?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of the `kubectl logs pods/job-...

- **expl_gen_006** (explanatory): 2/10
  - Query: "What is the purpose of the kubelet?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of the kubelet. The chunks dis...

- **expl_gen_008** (explanatory): 2/10
  - Query: "What is the purpose of init containers?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of init containers in Kubernet...

- **expl_gen_012** (explanatory): 2/10
  - Query: "What is the purpose of namespaces?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of namespaces in Kubernetes. T...

- **expl_gen_014** (explanatory): 2/10
  - Query: "What is the purpose of ConfigMaps?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of ConfigMaps in Kubernetes. T...

- **expl_gen_016** (explanatory): 2/10
  - Query: "What is the purpose of kube-proxy?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of kube-proxy. The chunks are ...

- **expl_gen_018** (explanatory): 4/10
  - Query: "What is the purpose of admission controllers?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of admission controllers in Ku...

- **expl_gen_020** (explanatory): 2/10
  - Query: "What is the purpose of ReplicaSets?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of ReplicaSets in Kubernetes. ...


---

## Oracle Comparison

Oracle performance was established by testing all retrieval granularities (chunk, heading, document) and taking the best per-query result across all approaches.

| Metric | Baseline | Oracle | Gap |
|--------|----------|--------|-----|
| Answer Success Rate | 92.1% | 99.6% | +7.4% |

### Oracle vs Baseline by Query Type

| Type | Baseline ASR | Oracle ASR | Gap |
|------|-------------|------------|-----|
| explanatory | 71.7% | 98.1% | +26.4% |
| factoid | 96.3% | 100.0% | +3.7% |
| troubleshooting | 97.1% | 100.0% | +2.9% |
| procedural | 100.0% | 100.0% | +0.0% |
| comparison | 100.0% | 100.0% | +0.0% |

**Key insight:** The oracle improves 17 queries over baseline. 14 of 17 are explanatory queries, confirming this is the category with the largest retrieval gap. Only 1 query (adv_adv_n04) fails across ALL approaches (corpus gap).

### Optimal Granularity Distribution

| Granularity | Count | % |
|-------------|-------|---|
| Chunk | 123 | 53.9% |
| Reranked chunk | 65 | 28.5% |
| Merged | 22 | 9.6% |
| Document | 17 | 7.5% |
| Heading | 1 | 0.4% |

See `oracle_performance.md` for full details.

---

## Latency Distribution

| Percentile | Retrieval (ms) |
|------------|----------------|
| p50 | 300 |
| p95 | 422 |
| p99 | 455 |

---

## Next Steps

1. Test P0 approaches: reranking, parent-child, auto-merging
2. Focus on improving **explanatory** queries (lowest success rate)
3. Investigate common failure patterns in judge reasoning

---

*Generated by run_baseline.py on 2026-02-21T16:23:13.323102*