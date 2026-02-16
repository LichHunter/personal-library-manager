# Phase 3: Correlation Analysis and Threshold Determination

**Generated**: 2026-02-16 19:03:26

## Objective
Validate confidence signals correlate with extraction quality and determine optimal routing threshold.

## Approach
- Spearman correlation for each signal vs quality grade (GOOD=2, ACCEPTABLE=1, POOR=0)
- 70/30 train/validation split (seed=42)
- Logistic regression on combined signals (balanced class weights)
- ROC analysis with Youden's J statistic for threshold selection

## Results

### Correlation Analysis
- known_term_ratio: r=-0.2373, p=0.0174 (significant)
- coverage: r=0.2763, p=0.0054 (significant)
- entity_density: r=0.2179, p=0.0294 (significant)
- section_type_mismatch: r=0.0000, p=1.0000 (not significant) - No variance in signal
- **Max correlation**: coverage with r=0.2763

### Classification Performance
- Training accuracy: 81.4%
- Validation accuracy: 80.0%
- AUC-ROC: 0.8750
- Optimal threshold: 0.3474

**Confusion Matrix (Validation Set)**
|          | Pred POOR | Pred GOOD/ACC |
|----------|-----------|---------------|
| Act POOR |        21 |             7 |
| Act GOOD/ACC |         0 |             2 |


### Routing Performance
- POOR recall: 75.0%
- Slow route rate: 70.0%


## Verdict: **FAIL**

Failed criteria: all correlations weak (max r=0.276 < 0.4). Current signals do not correlate with extraction quality. Signal engineering required (Phase 4a).

### Criteria Checked
| Criterion | Result |
|-----------|--------|
| Correlation >= 0.6 | False |
| Accuracy >= 80% | True |
| Routing optimal | False |
| Correlation 0.4-0.6 (partial) | False |
| Accuracy 70-80% (partial) | False |

## Issues
- Severe class imbalance: 66/70 training samples are POOR (94.3%)
- Insufficient minority class in validation: only 2 GOOD/ACCEPTABLE samples

## Dataset Statistics
- Total samples: 100
- Training set: 70 samples
- Validation set: 30 samples
- Training distribution: {0: 66, 1: 4}
- Validation distribution: {0: 28, 1: 2}

## Next Steps
Proceed to Phase 4a (Signal Engineering). Current signals do not correlate with extraction quality. Consider:
  - Adding new signals (e.g., text length, term frequency)
  - Engineering signal combinations
  - Re-evaluating the grading criteria

## Artifacts Generated
- `phase-3-correlations.json`: Signal correlation results
- `phase-3-threshold.json`: Optimal threshold and classification metrics
- `phase-3-roc.png`: ROC curve visualization
- `phase-3-summary.md`: This summary document
