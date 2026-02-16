# POC-2: Confidence Scoring and Threshold Determination

## TL;DR

> **Quick Summary**: Validate that confidence signals correlate with extraction quality (r>0.6) and determine the optimal threshold for routing chunks between fast/slow extraction systems, with iterative improvement fallbacks if initial signals fail.
> 
> **Deliverables**:
> - Implemented confidence signal calculators (4 signals)
> - Correlation analysis with extraction quality grades
> - Optimal threshold recommendation with ROC analysis
> - Decision tree for fast/slow routing
> 
> **Estimated Effort**: Medium (3-5 days)
> **Parallel Execution**: YES - 2 waves
> **Critical Path**: Phase 0 (signals) → Phase 1 (data) → Phase 2 (grading) → Phase 3 (analysis) → Phase 4x (iteration if needed)

---

## Context

### Original Request
Create a comprehensive work plan for POC-2: Confidence Scoring and Threshold Determination. The plan must include implementation, testing, expansion on results, and iterative improvement steps if initial results are poor.

### Interview Summary
**Key Discussions**:
- POC-2 is the logical next step after POC-1 (extraction) is complete
- Goal: Validate confidence signals correlate with extraction quality
- Determine optimal threshold for fast/slow routing
- User wants iterative test/consult/upgrade steps if results fail
- Must follow existing POC template structure from the project

**Research Findings**:
- Existing fast extraction code in `src/plm/extraction/fast/`
- Current confidence scoring is simple: `(source_score * 0.6) + (ratio_score * 0.4)`
- POC-1c has scoring functions and test data (249 docs with ground truth)
- Term vocabulary available in `data/vocabularies/`
- Architecture document specifies success criteria: r > 0.6 correlation, >80% accuracy

### Metis Review
**Identified Gaps** (addressed):
- Signal Implementation Gap: 3 of 4 signals don't exist in code → Added Phase 0
- Data Domain Decision: SO NER data ≠ documentation → Default: Use SO NER with documented limitations
- Grading Rubric Undefined → Default: F1-based thresholds defined
- Train/Validation Split → Added 70/30 split requirement
- Grader Validation → Added κ > 0.7 validation step

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

## 8. Verification Strategy (MANDATORY)

> **UNIVERSAL RULE: ZERO HUMAN INTERVENTION**
>
> ALL tasks in this plan MUST be verifiable WITHOUT any human action.
> This applies to EVERY task, regardless of test strategy.

### Test Decision
- **Infrastructure exists**: YES (pytest available)
- **Automated tests**: YES (Tests-after for analysis scripts)
- **Framework**: pytest

### Agent-Executed QA Scenarios (MANDATORY)

Each task will include specific verification scenarios using:
- **Bash (Python scripts)**: Run analysis scripts, verify outputs
- **Bash (pytest)**: Run unit tests for signal calculators
- **Bash (file checks)**: Verify artifact files exist and have correct structure

---

## 9. Execution Phases

### Phase 0: Signal Implementation

| Attribute | Value |
|-----------|-------|
| **Objective** | Implement the 4 confidence signal calculators |
| **Tasks** | 1. Create `poc/poc-2-confidence-scoring/signals.py`<br>2. Implement `known_term_ratio(terms, vocab)` function<br>3. Implement `coverage_score(terms, text)` function<br>4. Implement `entity_density(terms, text)` function<br>5. Implement `section_type_mismatch(terms, section_type)` function (stub for SO data)<br>6. Write unit tests for each signal |
| **Inputs** | Term vocabulary from `data/vocabularies/term_index.json` |
| **Outputs** | `signals.py` with 4 signal calculators |
| **Checkpoint Artifact** | `artifacts/phase-0-signals.json`<br>`artifacts/phase-0-summary.md` |
| **Acceptance Criteria** | All 4 signal functions exist and pass unit tests |

**Signal Implementations:**

```python
# Known Term Ratio: % of extracted terms found in vocabulary
def known_term_ratio(terms: list[str], vocab: set[str]) -> float:
    if not terms:
        return 0.0
    known = sum(1 for t in terms if t.lower() in vocab)
    return known / len(terms)

# Coverage: % of text "covered" by extracted terms (character-based)
def coverage_score(terms: list[str], text: str) -> float:
    if not text:
        return 0.0
    covered_chars = sum(len(t) * text.lower().count(t.lower()) for t in terms)
    return min(covered_chars / len(text), 1.0)

# Entity Density: terms per 100 tokens
def entity_density(terms: list[str], text: str) -> float:
    tokens = len(text.split())
    if tokens == 0:
        return 0.0
    return (len(terms) / tokens) * 100

# Section Type Mismatch: stub for SO data (no sections)
def section_type_mismatch(terms: list[str], section_type: str | None) -> float:
    # Returns 0.0 (no mismatch) for SO data; implement for docs later
    return 0.0
```

---

### Phase 1: Data Preparation

| Attribute | Value |
|-----------|-------|
| **Objective** | Prepare evaluation dataset with extractions and signal values |
| **Tasks** | 1. Load 100 random documents from SO NER test set<br>2. Run fast extraction on each document<br>3. Calculate ground truth metrics (P/R/F1/H) using POC-1c scoring<br>4. Calculate 4 confidence signals per document<br>5. Save as structured dataset |
| **Inputs** | `artifacts/test_documents.json` from POC-1c |
| **Outputs** | `artifacts/phase-1-dataset.json` with extractions + signals |
| **Checkpoint Artifact** | `artifacts/phase-1-dataset.json`<br>`artifacts/phase-1-summary.md` |
| **Acceptance Criteria** | 100 documents processed, all 4 signals calculated |

**Dataset Schema:**
```json
{
  "doc_id": "string",
  "text": "string",
  "gt_terms": ["string"],
  "extracted_terms": ["string"],
  "metrics": {"precision": 0.0, "recall": 0.0, "f1": 0.0, "hallucination": 0.0},
  "signals": {
    "known_term_ratio": 0.0,
    "coverage": 0.0,
    "entity_density": 0.0,
    "section_type_mismatch": 0.0
  }
}
```

---

### Phase 2: Quality Grading

| Attribute | Value |
|-----------|-------|
| **Objective** | Assign quality grades to each extraction |
| **Tasks** | 1. Define grading rubric (F1-based)<br>2. Create LLM grading prompt<br>3. Grade all 100 extractions via LLM<br>4. Select 30 random for human validation<br>5. Compute Cohen's κ; if < 0.7, investigate and re-grade<br>6. Finalize grades |
| **Inputs** | `artifacts/phase-1-dataset.json` |
| **Outputs** | `artifacts/phase-2-grades.json` with quality grades |
| **Checkpoint Artifact** | `artifacts/phase-2-grades.json`<br>`artifacts/phase-2-kappa.json`<br>`artifacts/phase-2-summary.md` |
| **Acceptance Criteria** | All 100 graded, κ >= 0.7 on validation set |

**Grading Rubric:**

| Grade | F1 Threshold | Hallucination Threshold | Numeric Value |
|-------|--------------|------------------------|---------------|
| **GOOD** | F1 >= 0.80 | H <= 15% | 2 |
| **ACCEPTABLE** | F1 >= 0.60 | H <= 30% | 1 |
| **POOR** | F1 < 0.60 | OR H > 30% | 0 |

**LLM Grading Prompt Template:**
```
Given the following extraction result, grade its quality:

Document text: {text}
Ground truth terms: {gt_terms}
Extracted terms: {extracted_terms}
Metrics: Precision={p}, Recall={r}, F1={f1}, Hallucination={h}

Based on these metrics, classify this extraction as:
- GOOD: F1 >= 0.80 AND Hallucination <= 15%
- ACCEPTABLE: F1 >= 0.60 AND Hallucination <= 30%
- POOR: F1 < 0.60 OR Hallucination > 30%

Grade: [GOOD|ACCEPTABLE|POOR]
Reasoning: [brief explanation]
```

---

### Phase 3: Correlation Analysis

| Attribute | Value |
|-----------|-------|
| **Objective** | Compute correlations and determine optimal threshold |
| **Tasks** | 1. Compute Spearman correlation for each signal vs grade<br>2. Test statistical significance (p < 0.05)<br>3. Fit logistic regression on 70% training set<br>4. Generate ROC curve for combined signal<br>5. Find optimal threshold using Youden's J<br>6. Evaluate on 30% validation set<br>7. Report classification accuracy and POOR recall |
| **Inputs** | `artifacts/phase-2-grades.json` |
| **Outputs** | `artifacts/phase-3-correlations.json`<br>`artifacts/phase-3-threshold.json`<br>`artifacts/phase-3-roc.png` |
| **Checkpoint Artifact** | All outputs above + `artifacts/phase-3-summary.md` |
| **Acceptance Criteria** | All correlations computed, threshold determined, validation complete |

**Analysis Outputs:**
```json
// phase-3-correlations.json
{
  "signals": {
    "known_term_ratio": {"r": 0.0, "p": 0.0, "significant": true/false},
    "coverage": {"r": 0.0, "p": 0.0, "significant": true/false},
    "entity_density": {"r": 0.0, "p": 0.0, "significant": true/false},
    "section_type_mismatch": {"r": 0.0, "p": 0.0, "significant": true/false}
  },
  "max_correlation": {"signal": "string", "r": 0.0}
}

// phase-3-threshold.json
{
  "optimal_threshold": 0.0,
  "method": "youden_j",
  "train_accuracy": 0.0,
  "validation_accuracy": 0.0,
  "poor_recall": 0.0,
  "slow_route_rate": 0.0,
  "auc_roc": 0.0
}
```

---

### Conditional Phases (If Initial Results Fail)

#### Phase 4a: Signal Engineering (If r < 0.6 for all signals)

| Attribute | Value |
|-----------|-------|
| **Trigger** | max(r) < 0.6 from Phase 3 |
| **Objective** | Design and test additional confidence signals |
| **Tasks** | 1. Analyze Phase 3 failures to identify patterns<br>2. Design 3 new candidate signals based on failure analysis<br>3. Implement and test new signals<br>4. Re-run correlation analysis<br>5. If still < 0.6 after 3 new signals, proceed to Phase 4b |
| **Exit Criteria** | r >= 0.6 achieved OR 3+ new signals tried with no r > 0.4 |

**Candidate New Signals:**
- **Term Rarity Score**: Average IDF of extracted terms (rare terms = higher confidence)
- **Pattern Diversity**: Number of distinct extraction patterns used (CamelCase, backtick, etc.)
- **Length Ratio**: Extracted term length vs document length
- **Capitalization Consistency**: % of terms following expected capitalization
- **Punctuation Context**: Terms near code indicators (`:`, `()`, `{}`)

---

#### Phase 4b: Ensemble Methods (If 4a fails)

| Attribute | Value |
|-----------|-------|
| **Trigger** | Phase 4a exhausted without r >= 0.6 |
| **Objective** | Try ensemble combinations of signals |
| **Tasks** | 1. Try weighted combinations of top 3 signals<br>2. Try Random Forest classifier with all signals<br>3. Try gradient boosting (XGBoost)<br>4. Evaluate each on validation set<br>5. If accuracy < 75% for all, proceed to Phase 4c |
| **Exit Criteria** | Accuracy >= 80% achieved OR all ensembles < 75% |

**Ensemble Approaches:**
1. **Weighted Sum**: `combined = w1*s1 + w2*s2 + w3*s3` with grid search
2. **Random Forest**: `sklearn.ensemble.RandomForestClassifier`
3. **XGBoost**: `xgboost.XGBClassifier`

---

#### Phase 4c: Oracle Consultation (If 4b fails)

| Attribute | Value |
|-----------|-------|
| **Trigger** | Phase 4b exhausted without accuracy >= 75% |
| **Objective** | Consult Oracle for architectural alternatives |
| **Tasks** | 1. Prepare summary of all attempts and failures<br>2. Invoke Oracle agent with full context<br>3. Document Oracle's recommendations<br>4. If Oracle suggests architectural change, proceed to Phase 4d |
| **Exit Criteria** | Oracle provides actionable recommendation |

**Oracle Prompt:**
```
POC-2 has failed to find confidence signals that correlate with extraction quality.

Attempts made:
- 4 original signals: max r = [value]
- 3 additional signals from Phase 4a: max r = [value]
- 3 ensemble methods from Phase 4b: max accuracy = [value]

The fundamental question: How can we determine which chunk extractions need slow-system review without signals that correlate with quality?

Please recommend:
1. Alternative routing strategies
2. Different quality proxies
3. Architectural changes to consider
4. Whether the fast/slow paradigm should be reconsidered
```

---

#### Phase 4d: Architectural Pivot (If Oracle recommends)

| Attribute | Value |
|-----------|-------|
| **Trigger** | Oracle recommends architectural change |
| **Objective** | Document pivot and update architecture |
| **Tasks** | 1. Document Oracle's recommendation<br>2. Create proposed architecture change document<br>3. Update RAG_PIPELINE_ARCHITECTURE.md with findings<br>4. Create follow-up POC if needed |
| **Exit Criteria** | Architecture updated, next steps defined |

---

## 10. TODOs

> Implementation + Test = ONE Task. EVERY task has: Recommended Agent Profile + Parallelization info.

- [x] 1. Setup POC-2 Directory Structure

  **What to do**:
  - Create `poc/poc-2-confidence-scoring/` directory
  - Initialize `pyproject.toml` with dependencies (scipy, sklearn, rapidfuzz, anthropic)
  - Create `README.md` from POC template
  - Create `SPEC.md` (copy this plan's spec sections)
  - Create `artifacts/` directory

  **Must NOT do**:
  - Install system-level dependencies
  - Modify existing POC directories

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple directory/file creation, no complex logic
  - **Skills**: []
    - No special skills needed for file scaffolding

  **Parallelization**:
  - **Can Run In Parallel**: NO (foundational task)
  - **Parallel Group**: Wave 1 (sequential start)
  - **Blocks**: Tasks 2, 3, 4, 5
  - **Blocked By**: None

  **References**:
  - `poc/POC_SPECIFICATION_TEMPLATE.md` - Structure template
  - `poc/poc-1c-scalable-ner/pyproject.toml` - Dependency example

  **Acceptance Criteria**:
  - [ ] Directory structure exists at `poc/poc-2-confidence-scoring/`
  - [ ] `pyproject.toml` contains required dependencies
  - [ ] `README.md` exists with basic setup instructions
  - [ ] `artifacts/` directory exists

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Directory structure created
    Tool: Bash
    Steps:
      1. ls -la poc/poc-2-confidence-scoring/
      2. Assert: pyproject.toml exists
      3. Assert: README.md exists
      4. Assert: artifacts/ directory exists
    Evidence: Terminal output captured

  Scenario: Dependencies specified
    Tool: Bash
    Steps:
      1. grep -E "scipy|sklearn|rapidfuzz" poc/poc-2-confidence-scoring/pyproject.toml
      2. Assert: All 3 dependencies found
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `feat(poc-2): initialize POC-2 directory structure`
  - Files: `poc/poc-2-confidence-scoring/*`
  - Pre-commit: `ls poc/poc-2-confidence-scoring/`

---

- [x] 2. Implement Confidence Signal Calculators (Phase 0)

  **What to do**:
  - Create `poc/poc-2-confidence-scoring/signals.py`
  - Implement `known_term_ratio(terms, vocab)` function
  - Implement `coverage_score(terms, text)` function
  - Implement `entity_density(terms, text)` function
  - Implement `section_type_mismatch(terms, section_type)` function (stub)
  - Create `test_signals.py` with unit tests for each function
  - Run tests and verify pass

  **Must NOT do**:
  - Implement complex NLP beyond simple calculations
  - Add dependencies beyond those in pyproject.toml

  **Recommended Agent Profile**:
  - **Category**: `quick`
    - Reason: Simple math functions, clear specs provided
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (after Task 1)
  - **Blocks**: Tasks 3, 4
  - **Blocked By**: Task 1

  **References**:
  - `src/plm/extraction/fast/confidence.py` - Existing confidence pattern
  - `poc/poc-1c-scalable-ner/scoring.py` - Scoring utilities
  - `data/vocabularies/term_index.json` - Vocabulary structure

  **Acceptance Criteria**:
  - [ ] `signals.py` contains all 4 signal functions
  - [ ] `test_signals.py` exists with tests for each function
  - [ ] `pytest test_signals.py` passes with 0 failures
  - [ ] `artifacts/phase-0-signals.json` created with function signatures
  - [ ] `artifacts/phase-0-summary.md` created

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Signal functions exist and work
    Tool: Bash
    Preconditions: Virtual env active
    Steps:
      1. cd poc/poc-2-confidence-scoring && python -c "from signals import known_term_ratio, coverage_score, entity_density, section_type_mismatch; print('OK')"
      2. Assert: Output is "OK"
    Evidence: Python output captured

  Scenario: Unit tests pass
    Tool: Bash
    Steps:
      1. cd poc/poc-2-confidence-scoring && pytest test_signals.py -v
      2. Assert: Exit code 0
      3. Assert: Output contains "passed"
    Evidence: pytest output captured

  Scenario: Edge case handling
    Tool: Bash
    Steps:
      1. python -c "from signals import known_term_ratio; print(known_term_ratio([], {'a'}))"
      2. Assert: Output is "0.0" (not error)
    Evidence: Output captured
  ```

  **Commit**: YES
  - Message: `feat(poc-2): implement confidence signal calculators`
  - Files: `poc/poc-2-confidence-scoring/signals.py`, `poc/poc-2-confidence-scoring/test_signals.py`
  - Pre-commit: `pytest poc/poc-2-confidence-scoring/test_signals.py`

---

- [x] 3. Prepare Evaluation Dataset (Phase 1)

  **What to do**:
  - Create `poc/poc-2-confidence-scoring/prepare_dataset.py`
  - Load 100 random documents from SO NER test set (seed=42 for reproducibility)
  - Run fast extraction on each document
  - Calculate ground truth metrics using POC-1c `many_to_many_score`
  - Calculate all 4 confidence signals per document
  - Save as `artifacts/phase-1-dataset.json`
  - Create `artifacts/phase-1-summary.md`

  **Must NOT do**:
  - Use slow extraction (LLM)
  - Modify source data files
  - Use more than 100 documents initially

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Data processing script, moderate complexity
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 1 (after Task 2)
  - **Blocks**: Tasks 4, 5
  - **Blocked By**: Tasks 1, 2

  **References**:
  - `poc/poc-1c-scalable-ner/artifacts/test_documents.json` - Source data
  - `poc/poc-1c-scalable-ner/scoring.py` - Scoring functions
  - `src/plm/extraction/fast/heuristic.py` - Fast extraction
  - `data/vocabularies/term_index.json` - Vocabulary for signals

  **Acceptance Criteria**:
  - [ ] `prepare_dataset.py` exists and runs without error
  - [ ] `artifacts/phase-1-dataset.json` contains exactly 100 entries
  - [ ] Each entry has: doc_id, text, gt_terms, extracted_terms, metrics, signals
  - [ ] All 4 signal values are floats between 0 and 1 (or density which can be >1)
  - [ ] `artifacts/phase-1-summary.md` contains statistics

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Dataset generated with correct count
    Tool: Bash
    Preconditions: Phase 0 complete
    Steps:
      1. cd poc/poc-2-confidence-scoring && python prepare_dataset.py
      2. python -c "import json; d=json.load(open('artifacts/phase-1-dataset.json')); print(len(d))"
      3. Assert: Output is "100"
    Evidence: Output captured

  Scenario: Dataset schema correct
    Tool: Bash
    Steps:
      1. python -c "import json; d=json.load(open('artifacts/phase-1-dataset.json'))[0]; assert 'signals' in d; assert 'known_term_ratio' in d['signals']; print('Schema OK')"
      2. Assert: Output is "Schema OK"
    Evidence: Output captured

  Scenario: Summary file exists
    Tool: Bash
    Steps:
      1. cat artifacts/phase-1-summary.md | head -20
      2. Assert: File contains statistics header
    Evidence: File content captured
  ```

  **Commit**: YES
  - Message: `feat(poc-2): create dataset preparation script`
  - Files: `poc/poc-2-confidence-scoring/prepare_dataset.py`, `artifacts/*`
  - Pre-commit: `python poc/poc-2-confidence-scoring/prepare_dataset.py`

---

- [x] 4. Implement Quality Grading (Phase 2)

  **What to do**:
  - Create `poc/poc-2-confidence-scoring/grading.py`
  - Implement F1-based automatic grading (GOOD/ACCEPTABLE/POOR)
  - Create LLM grading function for validation
  - Create human validation interface (CLI for 30 samples)
  - Grade all 100 samples automatically
  - Compute Cohen's κ between auto and LLM grades
  - Save `artifacts/phase-2-grades.json`
  - Save `artifacts/phase-2-kappa.json`
  - Create `artifacts/phase-2-summary.md`

  **Must NOT do**:
  - Require real-time human intervention
  - Modify grading thresholds after seeing results
  - Skip the κ validation step

  **Recommended Agent Profile**:
  - **Category**: `unspecified-low`
    - Reason: Moderate complexity, LLM integration
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 3)
  - **Blocks**: Task 5
  - **Blocked By**: Task 3

  **References**:
  - `poc/shared/README.md` - LLM provider usage
  - `poc/poc-1c-scalable-ner/utils/llm_provider.py` - LLM integration example
  - Grading rubric defined in Phase 2 section above

  **Acceptance Criteria**:
  - [ ] `grading.py` implements automatic F1-based grading
  - [ ] All 100 documents have grades in `artifacts/phase-2-grades.json`
  - [ ] Cohen's κ computed and saved in `artifacts/phase-2-kappa.json`
  - [ ] If κ < 0.7, disagreements documented
  - [ ] `artifacts/phase-2-summary.md` contains grade distribution

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: All documents graded
    Tool: Bash
    Steps:
      1. cd poc/poc-2-confidence-scoring && python grading.py
      2. python -c "import json; g=json.load(open('artifacts/phase-2-grades.json')); print(len(g))"
      3. Assert: Output is "100"
    Evidence: Output captured

  Scenario: Grade distribution reasonable
    Tool: Bash
    Steps:
      1. python -c "import json; g=json.load(open('artifacts/phase-2-grades.json')); from collections import Counter; print(Counter([x['grade'] for x in g]))"
      2. Assert: Output shows GOOD, ACCEPTABLE, POOR counts (not all one grade)
    Evidence: Distribution captured

  Scenario: Kappa computed
    Tool: Bash
    Steps:
      1. cat artifacts/phase-2-kappa.json
      2. Assert: Contains "kappa" key with numeric value
    Evidence: File content captured
  ```

  **Commit**: YES
  - Message: `feat(poc-2): implement quality grading with validation`
  - Files: `poc/poc-2-confidence-scoring/grading.py`, `artifacts/*`
  - Pre-commit: `python poc/poc-2-confidence-scoring/grading.py`

---

- [x] 5. Run Correlation Analysis and Threshold Determination (Phase 3)

  **What to do**:
  - Create `poc/poc-2-confidence-scoring/analysis.py`
  - Compute Spearman correlation for each signal vs quality grade
  - Report p-values and significance
  - Split data 70/30 for train/validation
  - Fit logistic regression on combined signals
  - Generate ROC curve and save as `artifacts/phase-3-roc.png`
  - Find optimal threshold using Youden's J
  - Evaluate on validation set
  - Save all results to `artifacts/phase-3-*.json`
  - Create comprehensive `artifacts/phase-3-summary.md`
  - Determine PASS/PARTIAL/FAIL based on success criteria

  **Must NOT do**:
  - Modify success criteria based on results
  - Cherry-pick best signal without proper validation
  - Skip statistical significance testing

  **Recommended Agent Profile**:
  - **Category**: `unspecified-high`
    - Reason: Statistical analysis, visualization, complex decision logic
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Wave 2 (after Task 4)
  - **Blocks**: Tasks 6, 7, 8 (conditional)
  - **Blocked By**: Task 4

  **References**:
  - `scipy.stats.spearmanr` - Correlation computation
  - `sklearn.linear_model.LogisticRegression` - Classification
  - `sklearn.metrics.roc_curve, auc` - ROC analysis
  - Success criteria defined in Section 7

  **Acceptance Criteria**:
  - [ ] `analysis.py` computes all correlations and threshold
  - [ ] `artifacts/phase-3-correlations.json` contains all signal correlations
  - [ ] `artifacts/phase-3-threshold.json` contains optimal threshold and metrics
  - [ ] `artifacts/phase-3-roc.png` visualizes ROC curve
  - [ ] `artifacts/phase-3-summary.md` contains PASS/PARTIAL/FAIL verdict
  - [ ] If PASS: Signal with r >= 0.6 identified, accuracy >= 80%
  - [ ] If PARTIAL or FAIL: Document which criteria failed

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: Analysis completes successfully
    Tool: Bash
    Preconditions: Phase 2 complete
    Steps:
      1. cd poc/poc-2-confidence-scoring && python analysis.py
      2. Assert: Exit code 0
    Evidence: Script output captured

  Scenario: All artifacts generated
    Tool: Bash
    Steps:
      1. ls artifacts/phase-3-*
      2. Assert: correlations.json, threshold.json, roc.png, summary.md all exist
    Evidence: File listing captured

  Scenario: Correlations computed correctly
    Tool: Bash
    Steps:
      1. python -c "import json; c=json.load(open('artifacts/phase-3-correlations.json')); print(c['max_correlation'])"
      2. Assert: Contains 'signal' and 'r' keys
    Evidence: JSON content captured

  Scenario: Verdict recorded
    Tool: Bash
    Steps:
      1. grep -E "PASS|PARTIAL|FAIL" artifacts/phase-3-summary.md
      2. Assert: One of PASS, PARTIAL, or FAIL found
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `feat(poc-2): implement correlation analysis and threshold determination`
  - Files: `poc/poc-2-confidence-scoring/analysis.py`, `artifacts/*`
  - Pre-commit: `python poc/poc-2-confidence-scoring/analysis.py`

---

- [x] 6. [CONDITIONAL] Signal Engineering Iteration (Phase 4a)

  **Trigger**: Phase 3 results show max(r) < 0.6

  **What to do**:
  - Analyze Phase 3 failure patterns
  - Design and implement 3 new candidate signals
  - Re-run correlation analysis with new signals
  - Document which signals were tried and their results
  - If r >= 0.6 achieved, proceed to Phase 3 analysis with new signals
  - If still < 0.6 after 3 new signals, proceed to Phase 4b

  **Must NOT do**:
  - Try more than 3 new signals before escalating
  - Modify original 4 signals (keep them for comparison)

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: Requires creative signal design based on failure analysis
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Conditional (only if Phase 3 fails)
  - **Blocks**: Task 7
  - **Blocked By**: Task 5 (with PARTIAL/FAIL result)

  **References**:
  - Phase 3 failure analysis in `artifacts/phase-3-summary.md`
  - Candidate signals listed in Phase 4a section above

  **Acceptance Criteria**:
  - [ ] At least 3 new signals designed and implemented
  - [ ] New correlations computed and documented
  - [ ] `artifacts/phase-4a-signals.json` contains new signal definitions
  - [ ] `artifacts/phase-4a-summary.md` documents results
  - [ ] Clear decision: proceed to Phase 3 (success) or Phase 4b (failure)

  **Commit**: YES (if executed)
  - Message: `feat(poc-2): add new confidence signals from iteration`
  - Files: `poc/poc-2-confidence-scoring/signals.py`, `artifacts/phase-4a-*`

---

- [x] 7. [CONDITIONAL] Ensemble Methods Iteration (Phase 4b)

  **Trigger**: Phase 4a exhausted without r >= 0.6

  **What to do**:
  - Implement ensemble combinations of all signals
  - Try weighted sum with grid search
  - Try Random Forest classifier
  - Try XGBoost classifier
  - Evaluate each on validation set
  - Document which ensembles were tried and their accuracy
  - If accuracy >= 80%, update threshold and proceed
  - If still < 75% for all, proceed to Phase 4c

  **Must NOT do**:
  - Overfit to validation set (use cross-validation)
  - Use test data during ensemble tuning

  **Recommended Agent Profile**:
  - **Category**: `ultrabrain`
    - Reason: ML model selection and tuning
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Conditional (only if Phase 4a fails)
  - **Blocks**: Task 8
  - **Blocked By**: Task 6 (with failure result)

  **References**:
  - `sklearn.ensemble.RandomForestClassifier`
  - `xgboost.XGBClassifier`
  - `sklearn.model_selection.GridSearchCV`

  **Acceptance Criteria**:
  - [ ] At least 3 ensemble methods tried
  - [ ] Cross-validation used for all ensembles
  - [ ] `artifacts/phase-4b-ensembles.json` contains all results
  - [ ] `artifacts/phase-4b-summary.md` documents best ensemble
  - [ ] Clear decision: success (accuracy >= 80%) or escalate (all < 75%)

  **Commit**: YES (if executed)
  - Message: `feat(poc-2): add ensemble methods for confidence scoring`
  - Files: `poc/poc-2-confidence-scoring/ensembles.py`, `artifacts/phase-4b-*`

---

- [x] 8. [CONDITIONAL] Oracle Consultation (Phase 4c) - SKIPPED (Phase 4b succeeded)

  **Trigger**: Phase 4b exhausted without accuracy >= 75%

  **What to do**:
  - Prepare comprehensive summary of all attempts
  - Invoke Oracle agent with full context
  - Document Oracle's recommendations
  - Create architecture update proposal if recommended
  - Determine next steps (new POC, architectural change, etc.)

  **Must NOT do**:
  - Ignore Oracle's recommendations
  - Make architectural changes without documenting

  **Recommended Agent Profile**:
  - **Category**: Use `oracle` subagent directly
    - Reason: High-IQ reasoning for architectural decisions
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Conditional (only if Phase 4b fails)
  - **Blocks**: Final report
  - **Blocked By**: Task 7 (with failure result)

  **References**:
  - All Phase 3, 4a, 4b artifacts
  - `docs/architecture/RAG_PIPELINE_ARCHITECTURE.md` Section 4.2

  **Acceptance Criteria**:
  - [ ] Oracle consultation completed
  - [ ] `artifacts/phase-4c-oracle.md` contains Oracle's response
  - [ ] `artifacts/phase-4c-recommendation.md` contains actionable next steps
  - [ ] If architectural change recommended, draft update created

  **Commit**: YES (if executed)
  - Message: `docs(poc-2): document Oracle consultation and recommendations`
  - Files: `artifacts/phase-4c-*`

---

- [x] 9. Create RESULTS.md and Final Report

  **What to do**:
  - Create comprehensive `RESULTS.md` following POC template
  - Summarize all phases executed (including conditional phases)
  - Document hypothesis verdicts with evidence
  - Report all primary and secondary metrics
  - List key findings and surprising results
  - Document limitations
  - Provide recommendations for architecture and next POCs
  - Update architecture document if needed

  **Must NOT do**:
  - Leave any section empty or TBD
  - Omit negative results
  - Modify success criteria in retrospect

  **Recommended Agent Profile**:
  - **Category**: `writing`
    - Reason: Documentation and technical writing
  - **Skills**: []
    - No special skills needed

  **Parallelization**:
  - **Can Run In Parallel**: NO
  - **Parallel Group**: Final (after all other tasks)
  - **Blocks**: None
  - **Blocked By**: Task 5 (or 6, 7, 8 if conditional phases executed)

  **References**:
  - `poc/POC_SPECIFICATION_TEMPLATE.md` Section 10 (Results Template)
  - All `artifacts/*.json` and `artifacts/*.md` files
  - `docs/architecture/RAG_PIPELINE_ARCHITECTURE.md` for context

  **Acceptance Criteria**:
  - [ ] `RESULTS.md` exists and follows template structure
  - [ ] All hypothesis verdicts documented with evidence
  - [ ] All metrics reported with actual values
  - [ ] Recommendations section complete
  - [ ] No "TBD" or empty sections

  **Agent-Executed QA Scenarios**:
  ```
  Scenario: RESULTS.md complete
    Tool: Bash
    Steps:
      1. wc -l poc/poc-2-confidence-scoring/RESULTS.md
      2. Assert: Line count > 100 (substantial content)
    Evidence: Line count captured

  Scenario: No empty sections
    Tool: Bash
    Steps:
      1. grep -c "TBD\|TODO\|PLACEHOLDER" poc/poc-2-confidence-scoring/RESULTS.md
      2. Assert: Count is 0
    Evidence: grep output captured

  Scenario: Verdict documented
    Tool: Bash
    Steps:
      1. grep -E "H1:.*SUPPORTED|H1:.*REJECTED" poc/poc-2-confidence-scoring/RESULTS.md
      2. Assert: Match found
    Evidence: grep output captured
  ```

  **Commit**: YES
  - Message: `docs(poc-2): complete POC-2 results documentation`
  - Files: `poc/poc-2-confidence-scoring/RESULTS.md`
  - Pre-commit: `grep -c "TBD" poc/poc-2-confidence-scoring/RESULTS.md | grep "^0$"`

---

## 11. Commit Strategy

| After Task | Message | Files | Verification |
|------------|---------|-------|--------------|
| 1 | `feat(poc-2): initialize POC-2 directory structure` | poc/poc-2-confidence-scoring/* | Directory exists |
| 2 | `feat(poc-2): implement confidence signal calculators` | signals.py, test_signals.py | pytest passes |
| 3 | `feat(poc-2): create dataset preparation script` | prepare_dataset.py, artifacts/* | 100 docs processed |
| 4 | `feat(poc-2): implement quality grading with validation` | grading.py, artifacts/* | κ computed |
| 5 | `feat(poc-2): implement correlation analysis and threshold` | analysis.py, artifacts/* | Verdict recorded |
| 6* | `feat(poc-2): add new confidence signals (iteration)` | signals.py, artifacts/* | Conditional |
| 7* | `feat(poc-2): add ensemble methods` | ensembles.py, artifacts/* | Conditional |
| 8* | `docs(poc-2): document Oracle consultation` | artifacts/* | Conditional |
| 9 | `docs(poc-2): complete POC-2 results documentation` | RESULTS.md | All sections filled |

*Tasks 6-8 are conditional based on Phase 3 results.

---

## 12. Success Criteria

### Verification Commands
```bash
# Run full POC pipeline
cd poc/poc-2-confidence-scoring
uv sync && source .venv/bin/activate
python prepare_dataset.py
python grading.py
python analysis.py

# Verify success criteria
python -c "
import json
c = json.load(open('artifacts/phase-3-correlations.json'))
t = json.load(open('artifacts/phase-3-threshold.json'))
print(f\"Max correlation: {c['max_correlation']['r']:.3f}\")
print(f\"Validation accuracy: {t['validation_accuracy']:.2%}\")
print(f\"POOR recall: {t['poor_recall']:.2%}\")
print(f\"Slow route rate: {t['slow_route_rate']:.2%}\")
print('PASS' if c['max_correlation']['r'] >= 0.6 and t['validation_accuracy'] >= 0.8 else 'CHECK CRITERIA')
"
```

### Final Checklist
- [x] All test cases (TC-1 through TC-4) completed
- [x] All checkpoint artifacts exist and are valid JSON/Markdown
- [x] RESULTS.md complete with no empty sections
- [x] If PASS: Threshold recommendation documented
- [x] If PARTIAL/FAIL: Iteration phases documented, next steps clear
- [x] Architecture implications documented

---

## 13. Dependencies

### 13.1 POC Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| POC-1c | Provides test data and scoring functions | Complete |

### 13.2 Infrastructure Dependencies

| Dependency | Purpose | Setup Instructions |
|------------|---------|-------------------|
| Python 3.11+ | Runtime | Available via nix develop |
| scipy | Correlation analysis | In pyproject.toml |
| scikit-learn | Classification, ROC | In pyproject.toml |
| rapidfuzz | String matching | In pyproject.toml |
| anthropic | LLM grading | In pyproject.toml, uses OAuth token |

### 13.3 Data Dependencies

| Data | Source | Availability |
|------|--------|--------------|
| SO NER test set | `poc/poc-1c-scalable-ner/artifacts/test_documents.json` | Available |
| Term vocabulary | `data/vocabularies/term_index.json` | Available |

---

## 14. Risks and Mitigations

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

## Appendix B: References

- `docs/architecture/RAG_PIPELINE_ARCHITECTURE.md` Section 4.2, 7.2
- `poc/POC_SPECIFICATION_TEMPLATE.md` - POC structure template
- `poc/poc-1c-scalable-ner/RESULTS.md` - Extraction pipeline results
- `src/plm/extraction/fast/confidence.py` - Current confidence implementation
- `src/plm/extraction/fast/heuristic.py` - Fast extraction patterns

---

*Plan Version: 1.0*
*Created: 2026-02-16*
*Last Updated: 2026-02-16*
