# Phase 5 Summary: Analysis and Reporting

## Objective

Compute final metrics, run statistical tests, and generate the RESULTS.md report.

## Approach

1. Loaded 1080 raw extraction results
2. Aggregated metrics by model and prompt variant
3. Identified best configuration: claude-sonnet + D
4. Evaluated all hypotheses against success criteria
5. Generated comprehensive RESULTS.md

## Results

### Best Configuration
- Model: claude-sonnet
- Variant: D
- Precision: 81.0% (PASS)
- Recall: 63.7% (PASS)
- Hallucination: 16.8% (FAIL)

### Hypothesis Verdicts
| Hypothesis | Verdict |
|------------|---------|
| H1 (Full guardrails >80% P, <5% H) | REJECTED |
| H2 (Evidence reduces halluc >50%) | SUPPORTED |
| H3 (Sonnet > Haiku by >10%) | REJECTED |
| H4 (Local models >70%) | NOT_TESTED |

## Overall POC Status: PARTIAL

**Phase 5 Status: COMPLETE**
