# Phase 4a: Signal Engineering Results

**Generated**: 2026-02-16 19:11:38

## Objective
Design and implement new confidence signals to improve correlation with extraction quality.

## New Signals Implemented

1. **technical_pattern_ratio**: Ratio of terms matching technical naming patterns (CamelCase, PascalCase, snake_case, dot.notation, CONSTANT_CASE, file paths)

2. **avg_term_length**: Average character length of extracted terms. Technical terms typically 5-20 chars; very short or very long terms indicate noise.

3. **text_grounding_score**: Ratio of terms found verbatim in original text. Low grounding suggests hallucination.

## Correlation Analysis (All 7 Signals)

| Signal | r | p | Significant | New? |
|--------|---|---|-------------|------|
| known_term_ratio | -0.2373 | 0.0174 | Yes | No |
| coverage | 0.2763 | 0.0054 | Yes | No |
| entity_density | 0.2179 | 0.0294 | Yes | No |
| section_type_mismatch | 0.0000 | 1.0000 | No | No |
| technical_pattern_ratio | 0.3342 | 0.0007 | Yes | Yes |
| avg_term_length | 0.3213 | 0.0011 | Yes | Yes |
| text_grounding_score | 0.0638 | 0.5282 | No | Yes |


**Maximum Correlation**: technical_pattern_ratio with r=0.3342, p=0.0007

## Classification Performance

| Metric | Value |
|--------|-------|
| Training Accuracy | 84.3% |
| Validation Accuracy | 80.0% |
| AUC-ROC | 0.9464 |
| Optimal Threshold | 0.6798 |
| POOR Recall | 78.6% |
| Slow Route Rate | 73.3% |

## Verdict: **FAILURE**

Max correlation r=0.334 < 0.4

### Exit Criteria Check
- SUCCESS (r >= 0.6): Not met
- PARTIAL (r >= 0.4 but < 0.6): Not met  
- FAILURE (r < 0.4): Met

## Analysis

### New Signal Performance
- **technical_pattern_ratio**: r=0.3342 (significant)
- **avg_term_length**: r=0.3213 (significant)
- **text_grounding_score**: r=0.0638 (not significant)

### Comparison with Phase 3
- Phase 3 max correlation: coverage (r=0.276)
- Phase 4a max correlation: technical_pattern_ratio (r=0.3342)
- Improvement: +0.0582

## Next Steps
New signals did not achieve target correlation. Proceed to Phase 4b or re-evaluate approach.

## Artifacts Generated
- `phase-4a-signals.json`: Signal definitions and correlation results
- `phase-4a-roc.png`: ROC curve visualization
- `phase-4a-summary.md`: This summary document
