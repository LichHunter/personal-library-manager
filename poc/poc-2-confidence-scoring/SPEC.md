# POC-2: Confidence Scoring and Threshold Determination

## TL;DR

> Validate that confidence signals correlate with extraction quality (r>0.6) and determine the optimal threshold for routing chunks between fast/slow extraction systems, with iterative improvement fallbacks if initial signals fail.

---

## 1. Research Question

**Primary Question**: Do the proposed confidence signals correlate with extraction quality, and what threshold should trigger slow processing?

**Sub-questions**:
- Which individual signal has the strongest correlation with quality?
- Does a combined signal outperform individual signals?
- What threshold achieves >80% classification accuracy while minimizing slow system load?
- How robust is the threshold across different document types?

---

## 2. Background

### 2.1 Why This Matters

Fast/slow routing depends on confidence scores. If scores don't reflect actual extraction quality, we'll either:
- **Miss bad extractions** (false high confidence) → poor term graph quality
- **Waste resources** re-processing good ones (false low confidence) → unnecessary LLM costs

### 2.2 Architecture Reference

- **Related Section**: `docs/architecture/RAG_PIPELINE_ARCHITECTURE.md` Section 4.2 (Fast System)
- **Design Decision**: D7 (Self-Improving Fast System) depends on correct routing

### 2.3 Current State

| Aspect | Current | Target |
|--------|---------|--------|
| Confidence signals | 1 simple formula | 4 validated signals |
| Threshold | Hardcoded 0.7 | Data-driven optimal |
| Classification accuracy | Unknown | >80% |
| Correlation with quality | Unknown | r > 0.6 |

---

## 3. Hypothesis

### 3.1 Primary Hypothesis

> **H1**: At least one of the four proposed confidence signals (Known Term Ratio, Coverage, Entity Density, Section Type Mismatch) will achieve correlation r > 0.6 with extraction quality grades.

### 3.2 Secondary Hypotheses

> **H2**: A combined signal (ensemble) will achieve classification accuracy >80% for GOOD vs POOR extractions.

> **H3**: A threshold exists that routes <30% of chunks to slow processing while catching >90% of POOR extractions.

### 3.3 Null Hypothesis

> **H0**: None of the proposed signals correlate meaningfully (r < 0.4) with extraction quality, indicating the routing strategy requires fundamental redesign.

---

## 4. Scope

### 4.1 IN SCOPE

| Item | Description |
|------|-------------|
| Signal Implementation | Implement 4 confidence signal calculators |
| Quality Grading | Grade 100 chunk extractions as GOOD/ACCEPTABLE/POOR |
| Correlation Analysis | Spearman correlation between signals and quality |
| Threshold Determination | ROC analysis to find optimal routing threshold |
| Iterative Improvement | Phases 4a-4d if initial signals fail |

### 4.2 OUT OF SCOPE

| Item | Reason | Deferred To |
|------|--------|-------------|
| Actual routing implementation | This POC validates; implementation is separate | Post-POC-2 |
| GLiNER integration | Rejected in POC-1c | N/A |
| Documentation corpus testing | SO NER used for speed | Future validation |
| Section Type signal validation | SO NER has no sections | Document as limitation |

### 4.3 Assumptions

| Assumption | If Violated |
|------------|-------------|
| SO NER ground truth quality is sufficient | Would need new annotated corpus |
| LLM grading agrees with human judgment | Fall back to manual grading |
| 100 samples provide statistical power | Increase to 150-200 |
| Fast extraction is deterministic | Add seed/reproducibility |

---

## 5. Test Design

### 5.1 Test Cases

#### TC-1: Signal-Quality Correlation

| Attribute | Value |
|-----------|-------|
| **Objective** | Validate individual signal correlation with extraction quality |
| **Input** | 100 chunks with extractions, quality grades, 4 signal values |
| **Procedure** | 1. Extract terms from each chunk using fast system<br>2. Grade each extraction (GOOD/ACCEPTABLE/POOR)<br>3. Calculate 4 confidence signals per chunk<br>4. Compute Spearman correlation for each signal vs quality |
| **Expected Outcome** | At least one signal achieves r > 0.6, p < 0.05 |
| **Pass Criteria** | max(r_signal1, r_signal2, r_signal3) >= 0.6 AND p < 0.05 |

#### TC-2: Combined Signal Classification

| Attribute | Value |
|-----------|-------|
| **Objective** | Validate ensemble signal for binary classification |
| **Input** | Same 100 chunks, combined confidence score |
| **Procedure** | 1. Use 70/30 train/validation split<br>2. Fit logistic regression on 70 samples<br>3. Evaluate on 30 hold-out samples<br>4. Report accuracy for GOOD vs POOR classification |
| **Expected Outcome** | Classification accuracy > 80% on validation set |
| **Pass Criteria** | Accuracy >= 0.80 AND AUC-ROC >= 0.85 |

#### TC-3: Threshold ROC Analysis

| Attribute | Value |
|-----------|-------|
| **Objective** | Find optimal threshold balancing precision/recall |
| **Input** | Combined signal scores, quality grades |
| **Procedure** | 1. Generate ROC curve<br>2. Find Youden's J point (max sensitivity + specificity - 1)<br>3. Calculate % routed to slow at this threshold<br>4. Report precision/recall for POOR detection |
| **Expected Outcome** | Threshold that catches >90% POOR with <30% slow routing |
| **Pass Criteria** | Recall(POOR) >= 0.90 AND SlowRouteRate <= 0.30 |

#### TC-4: Grader Reliability Validation

| Attribute | Value |
|-----------|-------|
| **Objective** | Validate LLM grading agrees with human judgment |
| **Input** | 30 randomly selected chunks with LLM grades |
| **Procedure** | 1. Human grades same 30 chunks blindly<br>2. Compute Cohen's kappa between LLM and human<br>3. If κ < 0.7, investigate disagreements |
| **Expected Outcome** | κ >= 0.7 (substantial agreement) |
| **Pass Criteria** | Cohen's κ >= 0.7 |

### 5.2 Test Data

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| SO NER test set | `poc/poc-1c-scalable-ner/artifacts/test_documents.json` | 249 docs | Extraction corpus |
| Term vocabulary | `data/vocabularies/term_index.json` | ~1000 terms | Known Term Ratio signal |
| POC sample | Random 100 from test set | 100 docs | Primary evaluation |
| Grader validation | Random 30 from sample | 30 docs | LLM grader validation |

**Data Requirements**:
- Minimum sample size: 100 (statistical power for r=0.3 at α=0.05, power=0.80)
- Diversity requirements: Include varied document lengths, tech domains
- Ground truth requirements: Use existing SO NER annotations

### 5.3 Variables

| Type | Variable | Values/Range |
|------|----------|--------------|
| **Independent** | Confidence signals (4) | 0.0 - 1.0 |
| **Dependent** | Quality grade | GOOD=2, ACCEPTABLE=1, POOR=0 |
| **Control** | Extraction method | Fast heuristic only |
| **Control** | Scoring function | v3_match from POC-1c |

---

## 6. Metrics

### 6.1 Primary Metrics (Determine Pass/Fail)

| Metric | Definition | Calculation | Target |
|--------|------------|-------------|--------|
| **Max Signal Correlation** | Highest Spearman r among 4 signals | `max(scipy.stats.spearmanr(signal, quality))` | r >= 0.6 |
| **Classification Accuracy** | Correct GOOD vs POOR predictions | `(TP + TN) / Total` on validation set | >= 80% |
| **POOR Recall** | % of POOR extractions caught by threshold | `TP_poor / (TP_poor + FN_poor)` | >= 90% |
| **Slow Route Rate** | % chunks routed to slow system | `N_below_threshold / N_total` | <= 30% |

### 6.2 Secondary Metrics (For Understanding)

| Metric | Definition | Purpose |
|--------|------------|---------|
| **AUC-ROC** | Area under ROC curve | Overall discriminative power |
| **Grader Agreement (κ)** | Cohen's kappa LLM vs human | Validate grading reliability |
| **Per-Signal Correlation** | Individual r for each signal | Identify best single predictor |
| **Threshold Sensitivity** | Accuracy change ±0.05 around optimal | Robustness check |

### 6.3 Metric Collection

| Metric | Collection Method | Frequency |
|--------|-------------------|-----------|
| Signal values | Computed per chunk | Per chunk |
| Quality grades | LLM + validation | Per chunk |
| Correlations | Aggregate after grading | Once per analysis |
| Classification metrics | Validation set evaluation | Once per threshold |

---

## 7. Success Criteria

### 7.1 PASS (All Must Be True)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Max signal correlation | r >= 0.6, p < 0.05 | Moderate-strong relationship required |
| Classification accuracy | >= 80% on validation | Acceptable error rate for routing |
| Grader reliability | κ >= 0.7 | Grades must be trustworthy |
| Threshold exists | POOR recall >= 90% AND slow rate <= 30% | Practical routing balance |

### 7.2 PARTIAL PASS (Hypothesis Partially Supported)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| Correlation r = 0.4-0.6 | One signal shows moderate correlation | Signals have potential, need improvement |
| Accuracy 70-80% | Some predictive power | May need more signals or ensemble |

### 7.3 FAIL (Hypothesis Rejected)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| All correlations r < 0.4 | No meaningful correlation | Signals don't capture quality; need new approach |
| Accuracy < 70% | Poor classification | Combined approach insufficient |
| Grader κ < 0.5 | Unreliable grades | Must use manual grading or different method |

**Note**: POC must ALWAYS run to completion. "FAIL" means hypothesis is disproven, NOT that POC was abandoned.

---

## 8. Execution Phases

### Phase 0: Signal Implementation

| Attribute | Value |
|-----------|-------|
| **Objective** | Implement the 4 confidence signal calculators |
| **Tasks** | 1. Create `signals.py`<br>2. Implement `known_term_ratio(terms, vocab)` function<br>3. Implement `coverage_score(terms, text)` function<br>4. Implement `entity_density(terms, text)` function<br>5. Implement `section_type_mismatch(terms, section_type)` function (stub for SO data)<br>6. Write unit tests for each signal |
| **Inputs** | Term vocabulary from `data/vocabularies/term_index.json` |
| **Outputs** | `signals.py` with 4 signal calculators |
| **Checkpoint Artifact** | `artifacts/phase-0-signals.json`<br>`artifacts/phase-0-summary.md` |
| **Acceptance Criteria** | All 4 signal functions exist and pass unit tests |

### Phase 1: Data Preparation

| Attribute | Value |
|-----------|-------|
| **Objective** | Prepare evaluation dataset with extractions and signal values |
| **Tasks** | 1. Load 100 random documents from SO NER test set<br>2. Run fast extraction on each document<br>3. Calculate ground truth metrics (P/R/F1/H) using POC-1c scoring<br>4. Calculate 4 confidence signals per document<br>5. Save as structured dataset |
| **Inputs** | `artifacts/test_documents.json` from POC-1c |
| **Outputs** | `artifacts/phase-1-dataset.json` with extractions + signals |
| **Checkpoint Artifact** | `artifacts/phase-1-dataset.json`<br>`artifacts/phase-1-summary.md` |
| **Acceptance Criteria** | 100 documents processed, all 4 signals calculated |

### Phase 2: Quality Grading

| Attribute | Value |
|-----------|-------|
| **Objective** | Assign quality grades to each extraction |
| **Tasks** | 1. Define grading rubric (F1-based)<br>2. Create LLM grading prompt<br>3. Grade all 100 extractions via LLM<br>4. Select 30 random for human validation<br>5. Compute Cohen's κ; if < 0.7, investigate and re-grade<br>6. Finalize grades |
| **Inputs** | `artifacts/phase-1-dataset.json` |
| **Outputs** | `artifacts/phase-2-grades.json` with quality grades |
| **Checkpoint Artifact** | `artifacts/phase-2-grades.json`<br>`artifacts/phase-2-kappa.json`<br>`artifacts/phase-2-summary.md` |
| **Acceptance Criteria** | All 100 graded, κ >= 0.7 on validation set |

### Phase 3: Correlation Analysis

| Attribute | Value |
|-----------|-------|
| **Objective** | Compute correlations and determine optimal threshold |
| **Tasks** | 1. Compute Spearman correlation for each signal vs grade<br>2. Test statistical significance (p < 0.05)<br>3. Fit logistic regression on 70% training set<br>4. Generate ROC curve for combined signal<br>5. Find optimal threshold using Youden's J<br>6. Evaluate on 30% validation set<br>7. Report classification accuracy and POOR recall |
| **Inputs** | `artifacts/phase-2-grades.json` |
| **Outputs** | `artifacts/phase-3-correlations.json`<br>`artifacts/phase-3-threshold.json`<br>`artifacts/phase-3-roc.png` |
| **Checkpoint Artifact** | All outputs above + `artifacts/phase-3-summary.md` |
| **Acceptance Criteria** | All correlations computed, threshold determined, validation complete |

---

## 9. Checkpoint Artifacts

### 9.1 Artifact Registry

| Phase | Artifact File | Format | Required Fields |
|-------|---------------|--------|-----------------|
| 0 | `phase-0-signals.json` | JSON | signal_names, function_signatures |
| 0 | `phase-0-summary.md` | Markdown | Objective, Approach, Results, Issues |
| 1 | `phase-1-dataset.json` | JSON | doc_id, text, gt_terms, extracted_terms, metrics, signals |
| 1 | `phase-1-summary.md` | Markdown | Statistics, sample_size, signal_ranges |
| 2 | `phase-2-grades.json` | JSON | doc_id, grade, reasoning |
| 2 | `phase-2-kappa.json` | JSON | kappa, p_value, agreement_matrix |
| 2 | `phase-2-summary.md` | Markdown | Grade distribution, validation results |
| 3 | `phase-3-correlations.json` | JSON | signals with r, p, significant |
| 3 | `phase-3-threshold.json` | JSON | optimal_threshold, accuracy, recall, auc_roc |
| 3 | `phase-3-roc.png` | PNG | ROC curve visualization |
| 3 | `phase-3-summary.md` | Markdown | Verdict, key findings, recommendations |

### 9.2 JSON Schema Requirements

```json
{
  "phase": "{phase_number}",
  "name": "{phase_name}",
  "completed_at": "{ISO 8601 timestamp}",
  "status": "complete|partial|blocked",
  "metrics": {
    "{metric_name}": "{value}"
  },
  "sample_size": "{number}",
  "notes": "{any observations}"
}
```

---

## 10. Dependencies

### 10.1 POC Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| POC-1c | Provides test data and scoring functions | Complete |

### 10.2 Infrastructure Dependencies

| Dependency | Purpose | Setup Instructions |
|------------|---------|-------------------|
| Python 3.11+ | Runtime | Available via nix develop |
| scipy | Correlation analysis | In pyproject.toml |
| scikit-learn | Classification, ROC | In pyproject.toml |
| rapidfuzz | String matching | In pyproject.toml |
| anthropic | LLM grading | In pyproject.toml, uses OAuth token |

### 10.3 Data Dependencies

| Data | Source | Availability |
|------|--------|--------------|
| SO NER test set | `poc/poc-1c-scalable-ner/artifacts/test_documents.json` | Available |
| Term vocabulary | `data/vocabularies/term_index.json` | Available |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| No signal correlates with quality | Medium | High | Iteration phases 4a-4d defined |
| LLM grading unreliable (κ < 0.5) | Low | Medium | Fall back to metric-based grading |
| SO NER data not representative | Medium | Medium | Document as limitation, validate on real docs later |
| Sample size insufficient | Low | Medium | Increase to 150-200 if needed |
| Threshold overfits to data | Medium | Medium | Use validation set, report sensitivity |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Known Term Ratio** | % of extracted terms found in existing vocabulary |
| **Coverage** | % of chunk text "covered" by extracted terms |
| **Entity Density** | Number of extracted terms per 100 tokens |
| **Section Type Mismatch** | Discrepancy between expected and actual extraction patterns |
| **Spearman r** | Rank correlation coefficient, robust to non-normality |
| **Cohen's κ** | Inter-rater agreement metric, κ > 0.7 = substantial |
| **Youden's J** | Threshold selection method maximizing sensitivity + specificity - 1 |

---

*Specification Version: 1.0*
*Created: 2026-02-16*
