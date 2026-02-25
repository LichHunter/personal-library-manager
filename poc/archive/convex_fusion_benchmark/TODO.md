# Convex Fusion POC: Implementation TODO

**Estimated Duration**: 4 days  
**Priority**: High (POC #1 from informed-query-improvements plan)

---

## Phase 1: Project Setup (Day 1 - Morning)

### 1.1 Environment Setup
- [ ] Create `pyproject.toml` with dependencies:
  - numpy (score manipulation)
  - pandas (data analysis)
  - matplotlib (visualization)
  - scipy (statistical tests)
  - tqdm (progress bars)
- [ ] Create `.venv` and run `uv sync`
- [ ] Verify PLM package is importable from poc directory
- [ ] Create `artifacts/` subdirectory for outputs

### 1.2 Data Infrastructure
- [ ] Create symlink to `poc/plm_vs_rag_benchmark/test_db/` (existing PLM index)
- [ ] Create symlink to `poc/plm_vs_rag_benchmark/corpus/` (source documents)
- [ ] Locate and document path to benchmark query files:
  - Informed queries (25)
  - Needle queries (20)
  - Realistic queries (need to identify/sample 50)

### 1.3 Benchmark Query Verification
- [ ] Load informed benchmark queries
- [ ] Verify each query has ground truth document ID
- [ ] Confirm ground truth documents exist in index
- [ ] Document any missing or problematic queries

---

## Phase 2: Score Extraction (Day 1 - Afternoon)

### 2.1 Retriever Analysis
- [ ] Read `src/plm/search/retriever.py` to understand score flow
- [ ] Identify where BM25 raw scores are available
- [ ] Identify where semantic similarity scores are available
- [ ] Determine if retriever needs modification or if scores can be extracted externally

### 2.2 Score Extraction Implementation
- [ ] Create `score_extractor.py` module
- [ ] Implement function to get BM25 scores for query:
  - Input: query string
  - Output: dict of {chunk_id: bm25_score} for top-50 results
- [ ] Implement function to get semantic scores for query:
  - Input: query string
  - Output: dict of {chunk_id: cosine_similarity} for top-50 results
- [ ] Handle edge cases:
  - Query returns fewer than 50 results
  - Empty results
  - Score normalization issues

### 2.3 Score Dataset Creation
- [ ] Run score extraction on all 25 informed queries
- [ ] Run score extraction on all 20 needle queries
- [ ] Run score extraction on 50 sampled realistic queries
- [ ] Save to `artifacts/raw_scores.json` with structure:
  ```
  {
    "informed": [
      {"query_id": "...", "query": "...", "ground_truth": "...", 
       "bm25_scores": {...}, "semantic_scores": {...}},
      ...
    ],
    "needle": [...],
    "realistic": [...]
  }
  ```
- [ ] Validate: every query has both BM25 and semantic scores

---

## Phase 3: Baseline Establishment (Day 2 - Morning)

### 3.1 RRF Implementation Verification
- [ ] Create `fusion.py` module
- [ ] Implement RRF function matching current retriever behavior:
  - k parameter = 60 (current default)
  - BM25 weight = 1.0
  - Semantic weight = 1.0
- [ ] Verify implementation produces same rankings as live retriever

### 3.2 Metrics Implementation
- [ ] Create `metrics.py` module
- [ ] Implement MRR@k calculation
- [ ] Implement Recall@k calculation
- [ ] Implement Hit@k calculation
- [ ] Implement per-query rank extraction

### 3.3 Baseline Measurement
- [ ] Calculate RRF MRR@10 on informed queries
- [ ] Calculate RRF MRR@10 on needle queries
- [ ] Calculate RRF MRR@10 on realistic queries
- [ ] Compare to known baselines:
  - Informed: expected ~0.621
  - Needle: expected ~0.842
  - Realistic: expected ~0.196
- [ ] If >2% deviation, investigate and document cause
- [ ] Save baseline results to `artifacts/baseline_rrf.json`

---

## Phase 4: Normalization Implementation (Day 2 - Afternoon)

### 4.1 Normalization Functions
- [ ] Add to `fusion.py`:
- [ ] Implement min-max normalization:
  - Scale scores to [0, 1] based on observed min/max
  - Handle edge case: all scores identical
- [ ] Implement z-score normalization:
  - Center around mean, unit variance
  - Handle edge case: zero variance
  - Decide how to handle negative values (shift to positive?)
- [ ] Implement rank-percentile normalization:
  - Convert scores to percentile ranks [0, 1]
  - Handle tied scores

### 4.2 Normalization Validation
- [ ] For each normalization strategy:
  - Apply to sample BM25 scores
  - Apply to sample semantic scores
  - Verify output ranges are comparable
  - Verify ranking order preserved
- [ ] Document any anomalies or edge cases found

---

## Phase 5: Convex Fusion Implementation (Day 2 - Evening)

### 5.1 Core Fusion Function
- [ ] Implement convex_fusion in `fusion.py`:
  - Input: bm25_scores, semantic_scores, alpha, normalization_method
  - Output: combined_scores dict
- [ ] Handle documents appearing in only one retriever:
  - Document in BM25 only: semantic score = 0
  - Document in semantic only: BM25 score = 0
- [ ] Verify output is valid ranking (no ties, no missing docs)

### 5.2 Fusion Validation
- [ ] Test alpha = 0.0 produces semantic-only ranking
- [ ] Test alpha = 1.0 produces BM25-only ranking
- [ ] Test alpha = 0.5 produces blended ranking
- [ ] Verify convex fusion with alpha=0.5 differs from RRF
- [ ] Document any unexpected behaviors

---

## Phase 6: Alpha Sweep (Day 3 - Morning)

### 6.1 Sweep Infrastructure
- [ ] Create `alpha_sweep.py` script
- [ ] Define alpha values: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
- [ ] Define normalization strategies: ["min_max", "z_score", "rank_percentile"]
- [ ] Create results storage structure

### 6.2 Run Alpha Sweep on Informed Benchmark
- [ ] For each normalization strategy:
  - [ ] For each alpha value:
    - [ ] Apply convex fusion to all 25 informed queries
    - [ ] Calculate MRR@10
    - [ ] Calculate Recall@10
    - [ ] Calculate Hit@1
    - [ ] Record per-query ranks
- [ ] Save results to `artifacts/alpha_sweep_informed.json`
- [ ] Log progress (tqdm)

### 6.3 Initial Analysis
- [ ] Identify best (normalization, alpha) combination for informed queries
- [ ] Calculate improvement over RRF baseline
- [ ] Check if improvement meets >5% threshold
- [ ] Document if multiple combinations perform similarly

---

## Phase 7: Fine-Grained Alpha Tuning (Day 3 - Afternoon)

### 7.1 Narrow Sweep
- [ ] Using best normalization from Phase 6
- [ ] Define fine-grained alpha around optimum (step 0.02 or 0.05)
- [ ] Run sweep on informed queries
- [ ] Identify precise optimal alpha

### 7.2 Statistical Significance
- [ ] Implement bootstrap confidence interval for MRR
- [ ] Calculate 95% CI for optimal alpha MRR
- [ ] Calculate 95% CI for RRF baseline MRR
- [ ] Determine if confidence intervals overlap (significance)
- [ ] Document p-value or equivalent

### 7.3 Sensitivity Analysis
- [ ] Plot alpha vs MRR curve
- [ ] Determine if curve is flat (insensitive) or peaked (sensitive)
- [ ] Document robustness of optimal alpha choice

---

## Phase 8: Per-Query Analysis (Day 3 - Evening)

### 8.1 Query-Level Breakdown
- [ ] For each informed query:
  - [ ] Record RRF rank of ground truth
  - [ ] Record optimal convex rank of ground truth
  - [ ] Calculate rank improvement (positive = better)
  - [ ] Note query characteristics (CamelCase, phrase, single word)

### 8.2 Winner/Loser Analysis
- [ ] Identify queries where convex fusion improves rank significantly (>3 positions)
- [ ] Identify queries where convex fusion worsens rank (any regression)
- [ ] Look for patterns:
  - Do technical terms favor one method?
  - Do phrase queries favor one method?
  - Are there query length effects?

### 8.3 Score Pattern Analysis
- [ ] For improved queries: what are typical BM25 vs semantic score ratios?
- [ ] For worsened queries: what are typical score ratios?
- [ ] Is there a predictive pattern for which method works better?

---

## Phase 9: Regression Testing (Day 4 - Morning)

### 9.1 Apply Optimal Configuration to Other Benchmarks
- [ ] Using optimal (normalization, alpha) from Phase 7
- [ ] Run convex fusion on needle queries (20)
- [ ] Run convex fusion on realistic queries (50)
- [ ] Calculate MRR@10 for each

### 9.2 Regression Check
- [ ] Compare needle MRR to baseline (expected ~0.842)
- [ ] Compare realistic MRR to baseline (expected ~0.196)
- [ ] Check regression thresholds:
  - Needle: must be >= 0.80 (within 5% of baseline)
  - Realistic: must be >= 0.18 (within 5% of baseline)
- [ ] Document any regressions found

### 9.3 Cross-Benchmark Insights
- [ ] Does optimal alpha for informed work for needle?
- [ ] Does optimal alpha for informed work for realistic?
- [ ] If not, document per-benchmark optimal alphas

---

## Phase 10: Visualization (Day 4 - Afternoon)

### 10.1 Create Plots
- [ ] Create `visualize.py` script
- [ ] Plot 1: Alpha vs MRR curve (all normalizations on one plot)
- [ ] Plot 2: Normalization comparison (bar chart at optimal alpha)
- [ ] Plot 3: Per-query improvement scatter (RRF rank vs convex rank)
- [ ] Plot 4: Score distribution histograms (BM25 and semantic)
- [ ] Save all plots to `artifacts/plots/`

### 10.2 Summary Tables
- [ ] Generate alpha sweep results table (markdown format)
- [ ] Generate per-query results table
- [ ] Generate cross-benchmark comparison table

---

## Phase 11: Documentation (Day 4 - Evening)

### 11.1 Results Document
- [ ] Create `RESULTS.md` from template
- [ ] Fill in execution summary
- [ ] Fill in hypothesis verdicts with evidence
- [ ] Fill in all metrics tables
- [ ] Document key findings
- [ ] Document surprising results
- [ ] Document limitations
- [ ] Write recommendations

### 11.2 Artifact Finalization
- [ ] Ensure all JSON artifacts are complete
- [ ] Verify all plots are generated
- [ ] Create `artifacts/final_results.json` with summary
- [ ] Update README.md with findings summary

### 11.3 Code Cleanup
- [ ] Add docstrings to all functions
- [ ] Remove debug prints
- [ ] Verify all scripts run without errors
- [ ] Test full pipeline from scratch

---

## Phase 12: Final Verdict (Day 4 - End)

### 12.1 Success Determination
- [ ] Check primary success criteria:
  - [ ] MRR improvement >5% on informed? YES/NO
  - [ ] No needle regression >5%? YES/NO
  - [ ] No realistic regression >5%? YES/NO
  - [ ] Statistical significance? YES/NO
- [ ] Determine overall verdict: PASS / PARTIAL / FAIL

### 12.2 Recommendations
- [ ] If PASS: Document recommended alpha and normalization for production
- [ ] If PARTIAL: Document caveats and conditions
- [ ] If FAIL: Document why and recommend alternative approaches

### 12.3 Next Steps
- [ ] Update `.sisyphus/plans/informed-query-improvements.md` with results
- [ ] If successful, create integration task for production
- [ ] Identify follow-up POCs if needed (e.g., query-adaptive alpha)

---

## Checkpoints

| Checkpoint | Artifact | Acceptance Criteria |
|------------|----------|---------------------|
| End of Phase 2 | `artifacts/raw_scores.json` | Scores for all 95 queries |
| End of Phase 3 | `artifacts/baseline_rrf.json` | Baseline within 2% of expected |
| End of Phase 6 | `artifacts/alpha_sweep_informed.json` | All 33 combinations tested |
| End of Phase 9 | `artifacts/regression_check.json` | All benchmarks tested |
| End of Phase 11 | `RESULTS.md` | Complete, no TODOs |

---

## Risk Mitigation Checkpoints

- [ ] **After Phase 2**: If score extraction fails, document blocker and escalate
- [ ] **After Phase 3**: If baseline deviates >5%, investigate before proceeding
- [ ] **After Phase 6**: If no alpha beats RRF, consider early termination (document as FAIL)
- [ ] **After Phase 9**: If severe regression, consider query-adaptive approach

---

## Notes

- Do NOT modify production code in `src/plm/` during this POC
- All implementations should be self-contained in `poc/convex_fusion_benchmark/`
- If retriever modification needed, create local wrapper instead
- Document all assumptions and deviations from plan

---

*Created: 2026-02-22*
*Based on: EVALUATION_CRITERIA.md*
