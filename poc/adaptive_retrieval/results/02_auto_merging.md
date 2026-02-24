# Auto-Merging Retrieval Results

**Date:** 2026-02-21 21:57:39
**Queries:** 229
**Labeled queries:** 65
**Configuration:** retrieve_k=10, merge_threshold=0.5, judge=haiku

---

## Approach Summary

Auto-merging retrieval: retrieve chunks, merge to parent heading if threshold met:
1. Retrieve top-10 chunks via hybrid search
2. Group chunks by heading_id
3. For each heading: count total sibling chunks
4. If (retrieved_count / total_siblings) >= 0.5 AND total_siblings > 1:
   - Replace with full heading content (all sibling chunks concatenated)
5. Otherwise: keep original individual chunks
6. Return mixed result set (merged headings + individual chunks)

---

## Overall Metrics

| Metric | Auto-Merging | Baseline | Delta |
|--------|-------------|----------|-------|
| MRR@10 (labeled) | 0.657 | 0.658 |  (-0.0) |
| Hit@5 (labeled) | 75.4% | 75.4% |  (+0.0) |
| **Answer Success Rate** | **95.2%** | **92.1%** | ** (+3.1)** |
| Avg Grade | 8.03/10 | 7.64/10 |  (+0.38) |
| Latency p50 (total) | 307ms | 300ms |  (+7)ms |
| Latency p95 (total) | 903ms | 422ms |  (+481)ms |
| Merge latency p50 | 3ms | — | — |
| Merge latency p95 | 4ms | — | — |
| Avg Tokens Retrieved | 1095 | 904 |  (+191) |
| Merge Rate | 92.6% | — | — |
| Avg Merges per Query | 2.58 | — | — |
| Avg Chunks Kept per Query | 6.79 | — | — |

---

## Performance by Query Type

| Type | Count | MRR@10 | Hit@5 | Success Rate | Avg Grade | Baseline SR | Delta |
|------|-------|--------|-------|--------------|-----------|-------------|-------|
| explanatory | 53 | 0.615 | 69.2% | 81.1% | 7.02 | 71.7% | +9.4% |
| troubleshooting | 34 | 0.583 | 75.0% | 97.1% | 7.82 | 97.1% | +0.0% |
| procedural | 55 | 1.000 | 100.0% | 100.0% | 8.64 | 100.0% | +0.0% |
| comparison | 33 | 0.604 | 75.0% | 100.0% | 8.39 | 100.0% | +0.0% |
| factoid | 54 | 0.633 | 73.5% | 100.0% | 8.30 | 96.3% | +3.7% |

---

## Success Criteria Evaluation

| Criterion | Threshold | Actual | Pass? |
|-----------|-----------|--------|-------|
| Answer Success Rate ≥+10% | +10.0% | +3.1% | ❌ |
| Latency increase ≤500ms (p95) | ≤500ms | +481ms | ✅ |
| No type regresses >5% | >-5% | All OK | ✅ |

---

## Category Distribution

- Partially Correct: 135 (59.0%)
- Correct: 83 (36.2%)
- Incorrect: 6 (2.6%)
- Cannot Answer: 5 (2.2%)

## Merge Rate Analysis by Query Type

| Type | Count | Merge Rate | Avg Merges/Query | Avg Chunks Kept | Avg Tokens |
|------|-------|------------|------------------|-----------------|------------|
| troubleshooting | 34 | 97.1% | 2.97 | 6.56 | 1198 |
| explanatory | 53 | 94.3% | 2.75 | 6.42 | 1129 |
| comparison | 33 | 90.9% | 2.58 | 6.45 | 1154 |
| procedural | 55 | 90.9% | 2.25 | 7.29 | 958 |
| factoid | 54 | 90.7% | 2.48 | 7.02 | 1099 |

**Merge Behavior Patterns:**

- **Troubleshooting queries merge most often (97.1%)** — diagnostic content tends to cluster under the same heading, so multiple retrieved chunks share a parent heading and trigger merges.
- **Explanatory queries have the second-highest merge rate (94.3%)** but also the lowest ASR (81.1%). This suggests merging doesn't help when the fundamental retrieval misses the right content — adding sibling context around irrelevant chunks adds noise, not signal.
- **Procedural queries keep the most chunks unmerged (7.29 avg)** — step-by-step content is often scattered across distinct headings, so fewer siblings appear together in results and merging triggers less aggressively.
- **Factoid queries merge least (90.7%)** yet achieve 100% ASR — factoid answers are typically contained in a single chunk, so merging is unnecessary. The marginal merging that does occur adds harmless context.
- **Overall insight:** High merge rates (>90% across all types) indicate the 0.5 threshold is too aggressive — it merges almost everything. A higher threshold (e.g., 0.7) might preserve chunk-level precision for factoid queries while still helping explanatory queries that genuinely need broader context.

---

## Sample Failed Queries (Top 10)

- **adv_adv_n04** (explanatory): 3/10
  - Query: "What's wrong with using container scope for latency-sensitive applications?..."
  - Reasoning: The retrieved chunks do not contain any information about the issues with using container scope for latency-sensitive ap...

- **expl_gen_006** (explanatory): 3/10
  - Query: "What is the purpose of the kubelet?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of the kubelet. The chunks discuss topics like Sta...

- **expl_gen_012** (explanatory): 2/10
  - Query: "What is the purpose of namespaces?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of namespaces in Kubernetes. The chunks discuss to...

- **expl_gen_016** (explanatory): 2/10
  - Query: "What is the purpose of kube-proxy?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of kube-proxy. The chunks discuss topics like cont...

- **expl_gen_020** (explanatory): 3/10
  - Query: "What is the purpose of ReplicaSets?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of ReplicaSets. The chunks discuss topics like con...

- **expl_gen_024** (explanatory): 4/10
  - Query: "What is the purpose of ServiceAccounts?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of ServiceAccounts in Kubernetes. The chunks discu...

- **expl_gen_032** (explanatory): 2/10
  - Query: "What is the purpose of node selectors?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of node selectors. The chunks discuss topics like ...

- **expl_gen_034** (explanatory): 2/10
  - Query: "What is the purpose of PodSecurityStandards?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of Pod Security Standards. The chunks discuss topi...

- **expl_gen_038** (explanatory): 2/10
  - Query: "What is the purpose of finalizers?..."
  - Reasoning: The retrieved chunks do not contain any information about the purpose of finalizers in Kubernetes. The chunks discuss to...

- **expl_gen_040** (explanatory): 4/10
  - Query: "What is the purpose of annotations in Kubernetes?..."
  - Reasoning: The retrieved chunks do not directly address the purpose of annotations in Kubernetes. The chunks provide some informati...


---

## Decision

**REJECT** — Marginal improvement doesn't justify changes.

Answer Success Rate delta: +3.1% (threshold: +10%)
Latency increase (p95): +481ms (threshold: ≤500ms)
Merge rate: 92.6% of queries

---

*Generated by run_auto_merging.py on 2026-02-21T21:57:39.466211*