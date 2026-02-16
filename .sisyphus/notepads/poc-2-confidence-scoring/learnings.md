## Key Findings

**Cohen's κ: 0.000 (slight agreement)**

The low κ value indicates systematic disagreement between automatic and LLM grading. Analysis reveals:

1. **All disagreements are POOR→ACCEPTABLE**: LLM upgraded 9/30 samples (30%) from POOR to ACCEPTABLE
2. **Hallucination threshold too strict**: LLM accepts H=25-66% when F1 is moderate (0.40-0.56)
3. **Automatic rubric is conservative**: Current thresholds (F1≥0.60, H≤30%) reject borderline cases

**Recommendation**: The automatic grading rubric is working as designed. The disagreements reflect LLM's more lenient interpretation of 'acceptable' quality. For Phase 3 correlation analysis, we should proceed with automatic grades as they provide consistent, reproducible measurements.

**Grade Distribution**: 94% POOR, 5% ACCEPTABLE, 1% GOOD - reflects that fast extraction (heuristic-only) performs poorly on this dataset, validating the need for confidence-based routing.

## Phase 3 Findings: Correlation Analysis

**Key Result**: FAIL - Current signals do not adequately correlate with extraction quality.

### Correlation Results
| Signal | Spearman r | p-value | Significant |
|--------|-----------|---------|-------------|
| coverage | 0.276 | 0.005 | Yes |
| entity_density | 0.218 | 0.029 | Yes |
| known_term_ratio | -0.237 | 0.017 | Yes* |
| section_type_mismatch | 0.000 | 1.000 | No |

*Negative correlation is unexpected - higher known_term_ratio correlates with WORSE extraction quality.

### Classification Performance
- Validation accuracy: 80% (meets threshold)
- AUC-ROC: 0.875 (good discrimination)
- But POOR recall: 75% (< 90% target)
- Slow route rate: 70% (> 30% target)

### Critical Issue: Class Imbalance
- 94% POOR, 5% ACCEPTABLE, 1% GOOD
- Train: 66 POOR, 4 non-POOR
- Validation: 28 POOR, 2 non-POOR
- Insufficient minority class samples for reliable threshold determination

### Insights
1. **Coverage shows strongest (but weak) correlation**: Makes sense - more text covered = better recall
2. **known_term_ratio negative correlation unexpected**: May indicate heuristic extraction extracts wrong terms from vocabulary
3. **section_type_mismatch has no variance**: All samples have 0.0, signal not discriminative
4. **Accuracy vs Correlation disconnect**: Can achieve 80% accuracy due to class imbalance but signals don't truly predict quality

### Next Steps
Signal engineering (Phase 4a) required:
- Add text length signal
- Add term frequency signals
- Consider signal combinations/interactions
- Re-evaluate grading criteria to get more balanced dataset
## POC-2 Learnings
- Individual confidence signals are weak predictors of extraction quality (max r=0.334).
- Ensemble methods (Random Forest) are highly effective even with weak individual signals, achieving 96.7% accuracy.
- High class imbalance (94% POOR) makes traditional correlation and agreement metrics (like Cohen's kappa) difficult to interpret.
- Fast extraction systems may perform significantly worse on some datasets (like SO NER) than others, necessitating aggressive routing to slow systems to maintain quality.
