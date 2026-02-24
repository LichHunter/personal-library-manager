# Evaluation Criteria for Adaptive Retrieval Approaches

## 1. Success Definition

### Primary Success Criterion

**An approach is successful if it improves Answer Success Rate without unacceptable latency increase.**

Specifically:
- **Answer Success Rate** must improve by ≥10% relative to baseline
- **Latency** must not increase by more than 500ms (p95)
- **No regression** on any query type category

### Secondary Success Criteria

- Reduces under-retrieval (queries that fail due to insufficient context)
- Does not significantly increase over-retrieval (wasted tokens)
- Maintains or improves MRR@10 for retrieval quality

---

## 2. Baseline Definition

### What is the Baseline?

**Current PLM retrieval pipeline:**
- Hybrid search (BM25 + semantic + RRF fusion)
- Returns top-10 chunks (~512 tokens each)
- No reranking
- No context expansion
- No granularity adaptation

### Baseline Metrics to Establish First

Before testing any approach, measure baseline on:

| Metric | Description | How to Measure |
|--------|-------------|----------------|
| MRR@10 | Retrieval ranking quality | Against labeled relevant chunks |
| Answer Success Rate | Can LLM answer with retrieved context? | LLM-as-judge on test queries |
| Average Tokens Retrieved | Context size | Count tokens in top-10 results |
| Latency (p50, p95) | Response time | Time from query to results |
| Under-Retrieval Rate | Failures due to missing context | % of "cannot answer" responses |

### Baseline Test Must Run On

- All query type categories (see Section 4)
- Minimum 200 queries total
- Results documented before any approach testing

---

## 3. What We Must Test

### 3.1 Retrieval Quality

**Question:** Does the approach find the right content?

| Test Point | Success Threshold | Failure Indicator |
|------------|-------------------|-------------------|
| MRR@10 | ≥ baseline | Drop >5% from baseline |
| Recall@10 | ≥ baseline | Drop >5% from baseline |
| Relevant chunk in results | ≥ baseline | Regression |

### 3.2 Context Sufficiency

**Question:** Is retrieved context enough to answer the query?

| Test Point | Success Threshold | Failure Indicator |
|------------|-------------------|-------------------|
| Answer Success Rate | ≥10% improvement | No improvement or regression |
| Under-Retrieval Rate | ≥20% reduction | No reduction |
| Context Completeness | Answers fully supported | Partial or unsupported answers |

### 3.3 Efficiency

**Question:** Is the approach practical for production?

| Test Point | Success Threshold | Failure Indicator |
|------------|-------------------|-------------------|
| Latency increase | ≤500ms at p95 | >1s increase |
| LLM calls added | ≤1 per query | >2 per query |
| Token efficiency | Improvement or neutral | >2x token increase with no quality gain |

### 3.4 Granularity Accuracy (For Adaptive Approaches)

**Question:** Does the approach pick the right context level?

| Test Point | Success Threshold | Failure Indicator |
|------------|-------------------|-------------------|
| Correct granularity prediction | ≥70% match with oracle | <50% match |
| Under-scope errors | ≤20% of queries | >30% of queries |
| Over-scope errors | ≤30% of queries | >50% of queries |

---

## 4. Query Type Categories

All approaches must be tested across these categories:

### Category A: Factoid Queries
- **Definition:** Single fact lookup, answer is 1-2 sentences
- **Examples:** "What is the default port for Redis?" / "What command lists pods?"
- **Expected optimal granularity:** Chunk
- **Test set size:** Minimum 50 queries

### Category B: Procedural Queries
- **Definition:** Step-by-step instructions, "how to" questions
- **Examples:** "How do I create a StatefulSet?" / "Steps to configure ingress"
- **Expected optimal granularity:** Heading/Section
- **Test set size:** Minimum 50 queries

### Category C: Explanatory Queries
- **Definition:** Conceptual understanding, "why" or "how does X work"
- **Examples:** "How does Kubernetes scheduling work?" / "Explain pod lifecycle"
- **Expected optimal granularity:** Heading or Document
- **Test set size:** Minimum 50 queries

### Category D: Comparison Queries
- **Definition:** Compare two or more concepts
- **Examples:** "Deployment vs StatefulSet" / "Difference between ConfigMap and Secret"
- **Expected optimal granularity:** Multiple Headings
- **Test set size:** Minimum 30 queries

### Category E: Troubleshooting Queries
- **Definition:** Diagnose or fix a problem
- **Examples:** "Why is my pod in CrashLoopBackOff?" / "How to debug OOMKilled"
- **Expected optimal granularity:** Heading + Related content
- **Test set size:** Minimum 30 queries

---

## 5. Test Data Requirements

### 5.1 Query Set

| Requirement | Specification |
|-------------|---------------|
| Total queries | Minimum 200 |
| Coverage | All 5 query categories |
| Source | Real user queries preferred, synthetic acceptable |
| Labeling | Each query must have: category, relevant chunks, optimal granularity |

### 5.2 Ground Truth Labels

Each test query must have:

| Label | Description | Required For |
|-------|-------------|--------------|
| `query_type` | Category A-E | All tests |
| `relevant_chunk_ids` | Chunks that contain the answer | Retrieval quality |
| `optimal_granularity` | Smallest sufficient context level | Granularity accuracy |
| `expected_answer` | Key points that must be in answer | Answer success |

### 5.3 Oracle Baseline

Before testing approaches, establish **oracle performance**:
- For each query, test all granularities (chunk, heading, document)
- Record which granularity produces correct answer
- This defines the theoretical maximum for adaptive approaches

---

## 6. Measurement Methodology

### 6.1 Answer Success Rate

**Method:** LLM-as-Judge

| Step | Description |
|------|-------------|
| 1 | Retrieve context using approach |
| 2 | Generate answer using LLM with retrieved context |
| 3 | Judge LLM evaluates: "Does the answer correctly address the query based on the context?" |
| 4 | Score: Correct / Partially Correct / Incorrect / Cannot Answer |

**Success = (Correct + Partially Correct) / Total**

### 6.2 Granularity Match

**Method:** Compare predicted vs oracle optimal

| Step | Description |
|------|-------------|
| 1 | Approach predicts/selects granularity |
| 2 | Compare to labeled optimal granularity |
| 3 | Score: Match / Under-scope / Over-scope |

### 6.3 Latency

**Method:** End-to-end timing

| Measurement | Description |
|-------------|-------------|
| Baseline latency | Query → chunks returned |
| Approach latency | Query → final context returned |
| Delta | Approach latency - Baseline latency |

Measure at p50, p95, p99.

---

## 7. Comparison Framework

### 7.1 Approach Comparison Matrix

| Dimension | Baseline | Approach A | Approach B | ... |
|-----------|----------|------------|------------|-----|
| **Quality** |
| MRR@10 | ? | ? | ? | |
| Answer Success Rate | ? | ? | ? | |
| **Efficiency** |
| Latency (p95) | ? | ? | ? | |
| Tokens Retrieved | ? | ? | ? | |
| **By Query Type** |
| Factoid Success | ? | ? | ? | |
| Procedural Success | ? | ? | ? | |
| Explanatory Success | ? | ? | ? | |
| Comparison Success | ? | ? | ? | |
| Troubleshooting Success | ? | ? | ? | |

### 7.2 Decision Criteria

**Approach is RECOMMENDED if:**
- Answer Success Rate improves ≥10% overall
- No query type category regresses >5%
- Latency increase ≤500ms at p95
- Implementation complexity is justified by gains

**Approach is REJECTED if:**
- Answer Success Rate does not improve
- Any query type category regresses >10%
- Latency increase >1s at p95
- Gains are marginal (<5%) but complexity is high

---

## 8. Test Execution Order

### Phase 1: Establish Baseline
1. Prepare test query set (200+ queries, labeled)
2. Run baseline PLM retrieval on all queries
3. Measure and document all baseline metrics
4. Establish oracle performance (test all granularities)

### Phase 2: Test P0 Approaches First
1. Reranking (#01)
2. Parent-Child (#15)
3. Auto-Merging (#02)

Why P0 first: Lowest complexity, highest expected ROI, validates test framework.

### Phase 3: Test P1 Approaches
Only if P0 approaches show promise or are insufficient.

### Phase 4: Test P2/P3 Approaches
Only if simpler approaches fail to meet success criteria.

---

## 9. Key Questions to Answer

| Question | How to Answer | Success Indicator |
|----------|---------------|-------------------|
| Is adaptation necessary at all? | Compare best adaptive vs "always heading" | Adaptive wins by ≥5% |
| Which approach best matches oracle? | Granularity match rate | ≥70% match |
| Is complexity justified? | Compare simple (rerank) vs complex (classifier) | Simple achieves 80%+ of complex gains |
| Which query types benefit most? | Per-type breakdown | Identify highest-impact categories |
| What's the quality/latency tradeoff? | Plot success rate vs latency | Find acceptable operating point |

---

## 10. Reporting Requirements

Each approach test must produce:

1. **Summary table:** All metrics vs baseline
2. **Per-query-type breakdown:** Success rate by category
3. **Failure analysis:** What queries failed and why
4. **Latency distribution:** Histogram of response times
5. **Recommendation:** Proceed / Reject / Needs modification
