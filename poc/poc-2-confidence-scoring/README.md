# POC-2: Confidence Scoring and Threshold Determination

## What

This POC validates that confidence signals correlate with extraction quality and determines the optimal threshold for routing chunks between fast (heuristic) and slow (LLM) extraction systems.

## Why

The fast/slow extraction routing strategy depends on confidence scores. If scores don't reflect actual extraction quality, we'll either:
- **Miss bad extractions** (false high confidence) → poor term graph quality
- **Waste resources** re-processing good ones (false low confidence) → unnecessary LLM costs

This POC answers: Do the proposed confidence signals correlate with extraction quality (r > 0.6)? What threshold achieves >80% classification accuracy?

## Hypothesis

**H1**: At least one of four proposed confidence signals (Known Term Ratio, Coverage, Entity Density, Section Type Mismatch) will achieve correlation r > 0.6 with extraction quality grades.

**H2**: A combined signal (ensemble) will achieve classification accuracy >80% for GOOD vs POOR extractions.

**H3**: A threshold exists that routes <30% of chunks to slow processing while catching >90% of POOR extractions.

## Setup

```bash
cd poc/poc-2-confidence-scoring
uv sync
source .venv/bin/activate
```

## Usage

The POC runs in phases:

```bash
# Phase 0: Signal implementation (already done)
# Phase 1: Prepare evaluation dataset
python prepare_dataset.py

# Phase 2: Grade extractions by quality
python grading.py

# Phase 3: Correlation analysis and threshold determination
python analysis.py

# View results
cat artifacts/phase-3-summary.md
```

## Results

See [RESULTS.md](./RESULTS.md) for complete results after execution.

## Key Metrics

| Metric | Target | Status |
|--------|--------|--------|
| Max signal correlation | r >= 0.6 | TBD |
| Classification accuracy | >= 80% | TBD |
| POOR extraction recall | >= 90% | TBD |
| Slow routing rate | <= 30% | TBD |

## Files

| File | Purpose |
|------|---------|
| `signals.py` | Confidence signal calculators (4 signals) |
| `test_signals.py` | Unit tests for signal functions |
| `prepare_dataset.py` | Load SO NER data, run fast extraction, calculate signals |
| `grading.py` | Grade extractions as GOOD/ACCEPTABLE/POOR |
| `analysis.py` | Correlation analysis, threshold determination, ROC analysis |
| `artifacts/` | Checkpoint files and results |
| `SPEC.md` | Full POC specification |
| `RESULTS.md` | Final results and conclusions |

## Architecture Reference

- **Related Section**: `docs/architecture/RAG_PIPELINE_ARCHITECTURE.md` Section 4.2 (Fast System)
- **Design Decision**: D7 (Self-Improving Fast System) depends on correct routing

## Next Steps

After POC-2 completion:
1. Implement routing logic in `src/plm/extraction/router.py`
2. Integrate confidence signals into fast extraction pipeline
3. Validate on actual documentation corpus (POC-3)

---

*POC-2 is Task 1 of 9 in the confidence scoring plan. See `.sisyphus/plans/poc-2-confidence-scoring.md` for full plan.*
