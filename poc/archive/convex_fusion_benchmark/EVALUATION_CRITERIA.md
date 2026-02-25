# Convex Fusion vs RRF: Evaluation Criteria

**Goal**: Determine if score-based convex combination fusion outperforms rank-based RRF on technical terminology queries.

---

## 1. Problem Statement

Current PLM uses Reciprocal Rank Fusion (RRF) which only considers **document ranks**, ignoring actual relevance scores. This means:
- A document with BM25 score 15.2 (strong exact match) is treated the same as one with score 2.1 (weak match) if both are rank #3
- Score magnitude information is lost during fusion

**Hypothesis**: Using actual scores via convex combination (`alpha * BM25 + (1-alpha) * semantic`) will improve retrieval quality, especially for technical terminology queries where exact matches should be weighted heavily.

---

## 2. Baseline Measurements Required

Before testing convex fusion, establish these baselines:

### 2.1 Current RRF Performance

| Benchmark | Metric | Must Measure |
|-----------|--------|--------------|
| Informed (25 queries) | MRR@10 | Primary baseline |
| Informed (25 queries) | Recall@10 | Secondary |
| Informed (25 queries) | Hit@1 | Secondary |
| Needle (20 queries) | MRR@10 | Regression check |
| Realistic (50 sample) | MRR@10 | Regression check |

**Known values from previous benchmarks**:
- Informed MRR: 0.621 (PLM baseline without reranker)
- Needle MRR: 0.842
- Realistic MRR: 0.196

### 2.2 Raw Score Distributions

Must capture and analyze:

| Data Point | Why Needed |
|------------|------------|
| BM25 score range per query | Understand score magnitude variance |
| Semantic score range per query | Understand score magnitude variance |
| Score overlap between retrievers | How many docs appear in both? |
| Score correlation | Do high BM25 scores correlate with high semantic scores? |

### 2.3 Per-Query Breakdown

For each of the 25 informed queries, record:
- Query text
- Ground truth document ID(s)
- RRF rank of ground truth
- BM25 rank of ground truth
- Semantic rank of ground truth
- BM25 score of ground truth
- Semantic score of ground truth

This enables per-query analysis of where convex fusion helps/hurts.

---

## 3. Test Variables

### 3.1 Independent Variables (What We Manipulate)

| Variable | Values to Test | Rationale |
|----------|----------------|-----------|
| **Alpha** | 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 | Full sweep to find optimum |
| **Normalization** | Min-Max, Z-Score, Rank-Percentile | Different strategies may favor different score distributions |

**Alpha interpretation**:
- alpha = 0.0: Pure semantic (ignore BM25)
- alpha = 0.5: Equal weight
- alpha = 1.0: Pure BM25 (ignore semantic)

### 3.2 Dependent Variables (What We Measure)

| Variable | Definition | Why It Matters |
|----------|------------|----------------|
| MRR@10 | Mean Reciprocal Rank in top 10 | Primary quality metric |
| Recall@10 | Fraction of relevant docs in top 10 | Coverage metric |
| Hit@1 | Is rank 1 correct? | Precision metric |
| Rank delta | Change in ground truth rank vs RRF | Per-query improvement |

### 3.3 Control Variables (Held Constant)

| Variable | Value | Rationale |
|----------|-------|-----------|
| Top-k retrieval | 10 | Match existing benchmarks |
| Candidate pool | 50 per retriever | Match current PLM config |
| Embedding model | BGE-base-en-v1.5 | No model changes in this POC |
| BM25 implementation | Current bm25s | Isolate fusion variable |

---

## 4. Normalization Strategies to Test

Score normalization is critical because BM25 and semantic scores have different ranges.

### 4.1 Min-Max Normalization

Scales scores to [0, 1] based on observed min/max:
- Pros: Simple, preserves relative ordering
- Cons: Sensitive to outliers

### 4.2 Z-Score Normalization

Centers scores around mean with unit variance:
- Pros: Handles different distributions
- Cons: Can produce negative values, requires handling

### 4.3 Rank-Percentile Normalization

Converts scores to percentile ranks:
- Pros: Robust to outliers, comparable across retrievers
- Cons: Loses absolute score information

### 4.4 Evaluation

For each normalization strategy:
1. Apply to both BM25 and semantic scores
2. Run alpha sweep
3. Record best alpha and corresponding MRR
4. Compare best performance across strategies

---

## 5. Test Protocol

### Phase 1: Data Collection

**Objective**: Extract raw scores for all benchmark queries

**Steps**:
1. Load PLM retriever with existing index
2. For each informed query (25 total):
   - Run BM25 retrieval, capture top-50 with scores
   - Run semantic retrieval, capture top-50 with scores
   - Record ground truth document ID
3. Save all scores to structured format

**Acceptance**: Raw scores available for all 25 informed queries

### Phase 2: Baseline Establishment

**Objective**: Confirm RRF baseline metrics

**Steps**:
1. Compute RRF fusion on collected scores
2. Calculate MRR@10, Recall@10, Hit@1
3. Compare to known baseline (MRR = 0.621)
4. If >5% deviation, investigate before proceeding

**Acceptance**: RRF baseline matches expected values (within 2%)

### Phase 3: Normalization Comparison

**Objective**: Determine best normalization strategy

**Steps**:
1. For each normalization strategy:
   - Normalize all BM25 scores
   - Normalize all semantic scores
   - Run alpha sweep (0.0 to 1.0, step 0.1)
   - Record MRR@10 for each alpha
2. Identify best (normalization, alpha) combination
3. Document score distribution characteristics

**Acceptance**: Clear winner or documented tradeoffs between strategies

### Phase 4: Detailed Alpha Analysis

**Objective**: Fine-tune optimal alpha and understand behavior

**Steps**:
1. Using best normalization from Phase 3
2. Fine-grained alpha sweep around optimum (step 0.05)
3. Per-query analysis:
   - Which queries improve with convex fusion?
   - Which queries get worse?
   - What characterizes each group?
4. Statistical significance testing (bootstrap CI)

**Acceptance**: Optimal alpha identified with confidence interval

### Phase 5: Regression Testing

**Objective**: Ensure convex fusion doesn't hurt other query types

**Steps**:
1. Apply optimal (normalization, alpha) to Needle benchmark
2. Apply optimal (normalization, alpha) to Realistic sample (50 queries)
3. Calculate MRR@10 for each
4. Compare to RRF baselines

**Acceptance**: No regression >5% on either benchmark

---

## 6. Success Criteria

### 6.1 Primary Success (PASS)

All of the following must be true:

| Criterion | Threshold | Measurement |
|-----------|-----------|-------------|
| MRR improvement on Informed | >5% over RRF | (convex_mrr - 0.621) / 0.621 > 0.05 |
| No Needle regression | MRR >= 0.80 | Within 5% of 0.842 baseline |
| No Realistic regression | MRR >= 0.18 | Within 5% of 0.196 baseline |
| Statistical significance | p < 0.05 | Bootstrap confidence interval |

**If PASS**: Recommend convex fusion for production with identified optimal alpha.

### 6.2 Partial Success (PARTIAL)

Some improvement but with caveats:

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| 2-5% MRR improvement | MRR in [0.634, 0.652] | Marginal gain, consider implementation cost |
| Improvement with regression | Informed up, Needle down | Need query-adaptive alpha (future POC) |
| Alpha-sensitive | Narrow optimal range | Fragile, may not generalize |

**If PARTIAL**: Document findings, consider query-type detection for adaptive alpha.

### 6.3 Failure (FAIL)

Hypothesis rejected:

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| No alpha improves | All alphas <= RRF | Convex fusion not viable |
| Best improvement <2% | Marginal, not worth complexity | Stick with RRF |
| Regression on all benchmarks | Worse across the board | Fundamental issue with approach |

**If FAIL**: RRF is sufficient for current use case. Focus optimization efforts elsewhere (SPLADE, BM25F, etc.).

---

## 7. Analysis Questions

After completing tests, answer these questions:

### 7.1 Optimal Configuration

- What alpha value is optimal for informed queries?
- Is the optimal alpha consistent across normalization strategies?
- How sensitive is performance to alpha choice (flat vs peaked curve)?

### 7.2 Query Characteristics

- Do technical term queries (e.g., "SubjectAccessReview") favor semantic (low alpha)?
- Do phrase queries (e.g., "webhook token authenticator") favor BM25 (high alpha)?
- Can we predict optimal alpha from query characteristics?

### 7.3 Score Analysis

- What is the typical BM25 vs semantic score ratio for successful retrievals?
- Do failed queries show different score patterns?
- Is there a score threshold below which documents should be filtered?

### 7.4 Comparison to RRF

- On which specific queries does convex fusion outperform RRF?
- On which specific queries does RRF outperform convex fusion?
- What's the variance in per-query improvement?

---

## 8. Deliverables

### 8.1 Required Outputs

| Deliverable | Format | Purpose |
|-------------|--------|---------|
| Raw scores dataset | JSON | Reproducibility |
| Alpha sweep results | JSON + chart | Show optimization curve |
| Per-query analysis | Table | Understand failure modes |
| Statistical tests | Report | Confidence in results |
| Final recommendation | Markdown | Actionable conclusion |

### 8.2 Visualization Requirements

1. **Alpha vs MRR curve**: Line plot showing MRR at each alpha value
2. **Normalization comparison**: Grouped bar chart comparing strategies
3. **Per-query improvement**: Scatter plot of RRF rank vs convex rank
4. **Score distributions**: Histograms of BM25 and semantic scores

---

## 9. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Score extraction requires code changes | Medium | Delays POC | Plan retriever modifications upfront |
| Optimal alpha is query-dependent | High | Limits value | Document for future adaptive POC |
| Results don't generalize beyond benchmark | Medium | Limited value | Test on realistic queries too |
| BM25 tokenization issue confounds results | High | Misleading | Document that BM25 is broken for CamelCase |

---

## 10. Constraints

### 10.1 Must Do

- Test all 11 alpha values (0.0 to 1.0)
- Test all 3 normalization strategies
- Run on all 25 informed queries
- Validate on needle and realistic benchmarks
- Document per-query results

### 10.2 Must Not Do

- Skip alpha values based on early results
- Change benchmark queries mid-test
- Cherry-pick favorable queries
- Modify BM25 tokenization (separate concern)
- Use reranker (testing first-stage fusion only)

---

## 11. Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Data Collection | 0.5 days | Retriever access |
| Phase 2: Baseline | 0.5 days | Phase 1 |
| Phase 3: Normalization | 1 day | Phase 2 |
| Phase 4: Alpha Analysis | 1 day | Phase 3 |
| Phase 5: Regression | 0.5 days | Phase 4 |
| Documentation | 0.5 days | Phase 5 |
| **Total** | **4 days** | |

---

## 12. References

- Current RRF implementation: `src/plm/search/retriever.py` (lines 73-86)
- Benchmark data: `poc/plm_vs_rag_benchmark/`
- Research findings: `.sisyphus/plans/informed-query-improvements.md`
- Original paper: "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods" (Cormack et al., 2009)

---

*Document version: 1.0*
*Created: 2026-02-22*
