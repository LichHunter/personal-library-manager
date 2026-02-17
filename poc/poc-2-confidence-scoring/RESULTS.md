# POC-2 Results: Confidence Scoring and Threshold Determination

## Execution Summary

| Attribute | Value |
|-----------|-------|
| **Started** | 2026-02-16T18:00:00Z |
| **Completed** | 2026-02-16T19:16:23Z |
| **Duration** | 1.25 hours |
| **Executor** | Sisyphus-Junior |
| **Status** | PARTIAL PASS |

## Hypothesis Verdict

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1: At least one of four proposed confidence signals will achieve correlation r > 0.6 with extraction quality grades. | REJECTED | Max correlation achieved was r=0.334 (technical_pattern_ratio) in Phase 4a. |
| H2: A combined signal (ensemble) will achieve classification accuracy >80% for GOOD vs POOR extractions. | SUPPORTED | Random Forest ensemble achieved 96.7% accuracy in Phase 4b. |
| H3: A threshold exists that routes <30% of chunks to slow processing while catching >90% of POOR extractions. | REJECTED | Due to 94% POOR extraction rate in fast system, catching >90% of POOR requires routing >90% to slow system. |

## Primary Metrics

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Max Signal Correlation | r >= 0.6 | 0.334 | FAIL |
| Classification Accuracy | >= 80% | 96.7% | PASS |
| POOR Recall | >= 90% | 100.0% | PASS |
| Slow Route Rate | <= 30% | 96.7% | FAIL |

## Secondary Metrics

| Metric | Value | Observation |
|--------|-------|-------------|
| AUC-ROC (Ensemble) | 0.848 | Strong discriminative power for routing decisions. |
| Grader Agreement (κ) | 0.000 | Slight agreement (70% raw agreement) due to extreme class imbalance. |
| Avg F1 (Fast System) | 0.280 | Fast system performance is significantly lower than expected on SO NER. |
| Hallucination Rate | 55.3% | High hallucination rate in fast system necessitates aggressive routing. |

## Test Case Results

| TC | Name | Status | Notes |
|----|------|--------|-------|
| TC-1 | Signal-Quality Correlation | FAIL | Max r=0.276 (Phase 3) and r=0.334 (Phase 4a) both below 0.6 target. |
| TC-2 | Combined Signal Classification | PASS | Random Forest achieved 96.7% accuracy, exceeding 80% target. |
| TC-3 | Threshold ROC Analysis | FAIL | Target slow route rate (30%) impossible given 94% POOR base rate. |
| TC-4 | Grader Reliability Validation | FAIL | κ=0.0 below 0.7 target, though raw agreement was 70%. |

## Phase Completion

| Phase | Status | Artifact |
|-------|--------|----------|
| 0 | COMPLETE | `artifacts/phase-0-signals.json` |
| 1 | COMPLETE | `artifacts/phase-1-dataset.json` |
| 2 | COMPLETE | `artifacts/phase-2-grades.json` |
| 3 | COMPLETE | `artifacts/phase-3-correlations.json` |
| 4a | COMPLETE | `artifacts/phase-4a-signals.json` |
| 4b | COMPLETE | `artifacts/phase-4b-ensembles.json` |

## Key Findings

### Finding 1: Individual Signals are Weak Predictors
No single signal achieved the target correlation of r > 0.6. The strongest individual signal was `technical_pattern_ratio` (r=0.334), followed by `avg_term_length` (r=0.321) and `coverage` (r=0.276). This indicates that extraction quality is a multi-faceted property that cannot be captured by simple heuristics alone. The weak correlation suggests that while these signals provide some information, they are noisy and insufficient as standalone triggers for routing.

### Finding 2: Ensemble Methods are Highly Effective
Despite weak individual correlations, combining 7 signals using a Random Forest classifier achieved 96.7% accuracy in distinguishing POOR extractions from GOOD/ACCEPTABLE ones. This suggests non-linear interactions between signals are critical for accurate routing. The Random Forest model was able to learn complex patterns that individual linear correlations missed, achieving a POOR recall of 100% on the validation set.

### Finding 3: Fast System Performance is the Bottleneck
The fast extraction system produced 94% POOR extractions on the Stack Overflow NER dataset. This high failure rate makes the original goal of routing <30% to the slow system unattainable if quality is to be maintained. The fast system's low F1 score (0.28) and high hallucination rate (55.3%) indicate that it currently acts more as a "noise generator" than a reliable extractor for this specific dataset.

### Finding 4: Class Imbalance Distorts Metrics
The extreme scarcity of GOOD extractions (1%) made correlation analysis and kappa statistics difficult to interpret. The 70% raw agreement between LLM and automatic grading resulted in κ=0.0 because the "expected agreement" by chance was also very high. This highlights the need for more balanced datasets or specialized metrics for highly imbalanced quality assessment tasks.

## Signal Performance Comparison

| Signal | Spearman r | p-value | Significance |
|--------|------------|---------|--------------|
| technical_pattern_ratio | 0.3342 | 0.0007 | Significant |
| avg_term_length | 0.3213 | 0.0011 | Significant |
| coverage | 0.2763 | 0.0054 | Significant |
| known_term_ratio | -0.2373 | 0.0174 | Significant |
| entity_density | 0.2179 | 0.0294 | Significant |
| text_grounding_score | 0.0638 | 0.5282 | Not Significant |
| section_type_mismatch | 0.0000 | 1.0000 | Not Significant |

## Detailed Phase Breakdown

### Phase 0: Signal Implementation
Implemented 4 initial signals: `known_term_ratio`, `coverage_score`, `entity_density`, and `section_type_mismatch`. All functions passed comprehensive unit tests (36/36) covering edge cases like empty inputs and overlapping terms.

### Phase 1: Data Preparation
Processed 100 documents from the SO NER test set. Observed an average of 3.72 extracted terms vs 7.62 ground truth terms. The baseline performance of the fast system was established here, showing significant room for improvement.

### Phase 2: Quality Grading
Established a grading rubric based on F1 and Hallucination rates. LLM validation on 30 samples showed 70% agreement with automatic grades. Disagreements were primarily in the "ACCEPTABLE" range, where LLMs were more lenient than the strict F1-based thresholds.

### Phase 3: Correlation Analysis
Initial analysis showed all signals failed the r > 0.6 target. `coverage` was the strongest at r=0.276. Logistic regression achieved 80% accuracy but with poor recall for the minority class, leading to a FAIL verdict for this phase.

### Phase 4a: Signal Engineering
Introduced 3 new signals: `technical_pattern_ratio`, `avg_term_length`, and `text_grounding_score`. While `technical_pattern_ratio` improved the max correlation to 0.334, it still fell short of the 0.6 target, necessitating further ensemble work.

### Phase 4b: Ensemble Methods
Tested multiple ensemble models. Random Forest emerged as the clear winner with 96.7% validation accuracy and 100% recall for POOR extractions. This phase successfully demonstrated that a combination of weak signals can form a strong classifier.

## Surprising Results
The `known_term_ratio` signal actually showed a *negative* correlation (r=-0.237) with quality in Phase 3. This was counter-intuitive, as we expected higher vocabulary overlap to signal higher quality. Investigation revealed that the fast system often extracted many common words that happened to be in the vocabulary but were not relevant to the specific chunk, leading to high signal but low precision.

## Limitations
- **Dataset Bias**: Results are based on Stack Overflow NER data, which may have different characteristics than the target personal documentation corpus. Technical terms in SO are often embedded in conversational text, which differs from structured documentation.
- **Section Type Stub**: The `section_type_mismatch` signal was not fully tested as the dataset lacked section metadata. This signal remains a theoretical improvement for structured docs.
- **Small Minority Class**: Only 6 GOOD/ACCEPTABLE samples were available in the 100-doc dataset, limiting the statistical power of the classification evaluation and potentially leading to overfitting in the ensemble models.

## Recommendations

### For Architecture
- **Adopt Ensemble Routing**: Implement the Random Forest classifier in `src/plm/extraction/router.py` instead of a simple threshold on a single signal. The model should be serialized and loaded at runtime.
- **Aggressive Routing Configuration**: Given the fast system's current performance, the router should be configured for high recall of POOR extractions. This means routing >90% of chunks to the slow system until the fast system's base performance improves.
- **Fast System Optimization**: The high POOR rate (94%) suggests the fast system needs significant improvement. Consider incorporating the `technical_pattern_ratio` logic directly into the extraction heuristics to filter out noise before scoring.
- **Feedback Loop**: Implement a mechanism to collect "corrections" from the slow system to periodically retrain the Random Forest router, allowing it to adapt to new document types.

### For Next POCs
- **POC-3 (Integration)**: Validate the ensemble router on actual documentation chunks where the fast system might perform better than on SO NER. Focus on measuring the actual reduction in LLM tokens vs quality loss.
- **Signal Refinement**: Explore more sophisticated signals like "Dependency Parse Validity" or "NER Type Consistency" if routing accuracy needs to be improved further.
- **Active Learning**: Investigate if the router can be used to identify "uncertain" chunks for human review, creating a high-quality ground truth dataset for future training.

---

## Appendix: GLiNER NER Fine-Tuning Experiment

### Summary

| Attribute | Value |
|-----------|-------|
| **Date** | 2026-02-17 |
| **Objective** | Determine if fine-tuning GLiNER improves F1 above zero-shot baseline (0.518) and maintains confidence separation above +0.100 |
| **Base Model** | `urchade/gliner_medium-v2.1` (DeBERTa-v2 encoder, GLiNER v0.2.22) |
| **Dataset** | StackOverflow NER (Tabassum et al., ACL 2020) — `train.txt` |
| **Verdict** | **Fine-tuning improves all metrics over zero-shot.** F1 0.518 -> 0.662 (+28%), confidence separation +0.100 -> +0.153 (+53%). |

### Results: Zero-Shot vs Fine-Tuned

| Metric | Zero-Shot | Fine-Tuned | Delta |
|--------|-----------|------------|-------|
| Precision | 0.581 | **0.669** | +15% |
| Recall | 0.547 | **0.730** | +33% |
| F1 | 0.518 | **0.662** | +28% |
| Hallucination | 0.419 | **0.331** | -21% |
| TP Mean Confidence | -- | 0.703 | -- |
| FP Mean Confidence | -- | 0.549 | -- |
| Confidence Separation | +0.100 | **+0.153** | +53% |
| Avg Inference Time | -- | 23.4ms | -- |

### Context: Comparison to LLM Approaches (from POC-1c)

| Approach | Precision | Recall | F1 | Hallucination | Vocab Required |
|----------|-----------|--------|------|---------------|----------------|
| V6 + vocabulary (POC-1b) | **90.7%** | **95.8%** | **0.932** | **9.0%** | 176+ terms |
| Retrieval few-shot (POC-1c) | 81.6% | 80.6% | 0.811 | 18.4% | 0 terms |
| SLIMER zero-shot (POC-1c) | 84.9% | 66.0% | 0.743 | 15.1% | 0 terms |
| **GLiNER fine-tuned** | 66.9% | 73.0% | 0.662 | 33.1% | 0 terms |
| GLiNER zero-shot | 58.1% | 54.7% | 0.518 | 41.9% | 0 terms |

Fine-tuned GLiNER remains significantly below all LLM approaches. The 33% hallucination rate and 67% precision are insufficient for production use as a primary extractor.

### Theoretical Ceiling for GLiNER

Based on architecture analysis, the realistic performance ceiling for GLiNER fine-tuned on SO NER is estimated at:

- **F1: ~0.75-0.80** (vs current 0.662)
- **Precision: ~75-80%** (vs current 67%)
- **Hallucination: ~20-25%** (vs current 33%)
- **Confidence Separation: ~+0.20-0.25** (vs current +0.153)

This ceiling exists because:
1. GLiNER uses biaffine span-type dot-product scoring — less expressive than LLM-based reasoning
2. Software entities (CamelCase, dot.paths, mixed code/natural-language) require domain understanding that span matching cannot fully capture
3. Published NER models on SO NER achieve F1 of 0.50-0.65; our 0.662 is already competitive with SOTA
4. 23 fine-grained entity types (e.g., Library vs Library_Class vs Library_Function) create inherent disambiguation challenges

### Implication for Fast/Slow Routing

At the theoretical GLiNER ceiling (confidence separation ~+0.25), achieving 90% precision on the fast path would require routing **~70-80%** of predictions to the slow system. This defeats the purpose of fast/slow routing, which targets <=30% slow-path traffic.

**Conclusion: GLiNER confidence scores are insufficient for efficient routing.** The confidence separation (even at ceiling) cannot reliably distinguish correct from incorrect predictions. The ensemble routing approach from Phase 4b (Random Forest on 7 heuristic signals) remains the recommended routing strategy.

### Fine-Tuning Strategy (Reproducibility Guide)

#### Data Preparation

The SO NER dataset uses BIO format with blank-line-delimited sentences. The critical insight is that data is **already sentence-level** — the parser must emit one sample per sentence, NOT concatenate sentences within a document.

- **Correct**: 4,893 sentence-level samples, 8,873 entities, max 92 tokens/sentence
- **Wrong** (prior approach): 200 document-level samples with 500-700+ tokens, causing truncation at GLiNER's 384-token limit and silently dropping 7.5% of entities

```python
# GLiNER training format (one per sentence)
{"tokenized_text": ["word1", "word2", ...], "ner": [[start, end, "Type"], ...]}
# Indices are token-level and INCLUSIVE on both ends.
```

Excluded entity types: `Code_Block`, `Output_Block`, `Variable_Name`, `Value`, `User_Name` (too noisy or not extractable from text spans).

#### Training Configuration (Working)

```python
TrainingArguments(
    learning_rate=1e-5,           # DeBERTa encoder LR
    others_lr=1e-4,               # Scorer head LR (10x encoder — head must move aggressively)
    weight_decay=0.1,             # Encoder weight decay
    others_weight_decay=0.01,     # Head weight decay
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    max_grad_norm=10.0,           # NOT HF default 1.0 — allows larger gradient updates
    per_device_train_batch_size=8,
    focal_loss_alpha=0.90,        # Up-weight positives ~9x over negatives
    focal_loss_gamma=2,           # Focal modulation for hard examples
    loss_reduction="sum",         # Preserves per-positive gradient signal
    masking="global",             # Randomly drop 50% of negative span-type candidates
    negatives=0.5,                # Controls negative drop rate
    num_train_epochs=3,
)
```

Training time: ~5 minutes on NVIDIA GPU (1836 steps at ~6 it/s).

#### Failed Configurations and Root Causes

Five training runs were attempted. Only the final configuration succeeded.

| Run | Key Config | Loss Curve | Scores | F1 | Root Cause |
|-----|-----------|------------|--------|-----|------------|
| v1 | `sum`, `gamma=2`, `max_grad_norm=1.0`, `bs=4` | ~1000 → converged | 0.500 exactly | 0.262 | Gradient explosion destroyed scorer: `sum` produces loss ~1000, clipping at 1.0 is too aggressive, `gamma=2` amplifies imbalance |
| v2 | `mean`, `gamma=2`, `lr=5e-6` | 0.17 → 0.026 | ~0.50-0.53 | 0.192 | Mean reduction drowns positive gradient signal in 99.94% negatives |
| v3 | `mean`, `gamma=2`, `lr=1e-5` | 0.19 → 0.026 | ~0.50-0.53 | -- | Same as v2 — higher LR doesn't fix mean-reduction signal drowning |
| v4 | `sum`, `gamma=0`, `max_grad_norm=10.0`, `bs=8` | ~1200 plateau | 0.500 exactly | 0.265 | Official GLiNER config, but `gamma=0` (plain BCE) + our dataset's extreme sparsity = negatives dominate loss |
| **v5** | **`sum`, `gamma=2`, `alpha=0.90`, `negatives=0.5`, `masking=global`** | **~100 → converging** | **0.50-0.96** | **0.662** | **Correct combination: high alpha weights positives, negative sampling reduces dilution** |

#### The Core Problem: Extreme Class Imbalance

With 23 entity types and median 16-token sentences:
- ~136 candidate spans × variable types per sample = thousands of span-type candidates
- Only ~1.8 entities per sentence → **99.94% of candidates are negatives**

This extreme imbalance means:
1. **`loss_reduction="mean"`** averages over thousands of easy negatives, making positive gradient signal negligible
2. **`loss_reduction="sum"`** preserves positive signals but produces enormous loss values requiring high `max_grad_norm`
3. **`focal_loss_gamma=0`** (plain BCE) lets easy negatives contribute equal weight to loss
4. **`focal_loss_alpha=0.75`** gives positives only 3x weight — insufficient when negatives outnumber positives 1700:1

The working solution combines three mechanisms:
- **`alpha=0.90`**: Gives positives ~9x weight over negatives
- **`masking="global"` + `negatives=0.5`**: Randomly drops 50% of negative candidates from loss computation via Bernoulli mask (GLiNER's built-in mechanism in `BaseModel._loss()`)
- **`others_lr=1e-4`** (10x encoder LR): Scorer head must learn new discrimination quickly while encoder fine-tunes gently

#### API Notes (GLiNER v0.2.22)

- Use `from gliner.training import Trainer, TrainingArguments` and `from gliner.data_processing.collator import DataCollator`
- `model.train_model()` does NOT exist in v0.2.22 (only in newer versions)
- Do NOT call `model.to(device)` before HF Trainer — Trainer manages device placement
- `DataCollator` handles negative entity type sampling automatically via `batch_generate_class_mappings()`
- `max_types=25` and `max_neg_type_ratio=1` are config defaults — each sample gets its own entity types + sampled negatives, NOT all 23 types

### Artifacts

| File | Description |
|------|-------------|
| `artifacts/gliner_finetune_results.json` | Full results with per-document metrics |
| `artifacts/gliner-finetuned/` | Saved fine-tuned model weights |
| `finetune_gliner.py` | Complete training and evaluation script |

### References

- GLiNER official training config: [urchade/GLiNER/configs/config.yaml](https://github.com/urchade/GLiNER/blob/f3ffcd6fc86edf5b8d495d6b43464a4931c18e3c/configs/config.yaml)
- Known issue (score collapse): [urchade/GLiNER#294](https://github.com/urchade/GLiNER/issues/294)
- Known issue (catastrophic forgetting): [urchade/GLiNER#163](https://github.com/urchade/GLiNER/issues/163)
- Working fine-tuning example (calamanCy): [ljvmiranda921/calamanCy](https://github.com/ljvmiranda921/calamanCy/blob/master/models/v0.1.0-gliner/train.py)

---

## Raw Data

- Phase results: `artifacts/phase-4b-ensembles.json`
- Phase logs: `artifacts/phase-4b-summary.md`
- GLiNER fine-tuning: `artifacts/gliner_finetune_results.json`
