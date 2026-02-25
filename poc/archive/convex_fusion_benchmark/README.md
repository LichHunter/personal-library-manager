# Convex Fusion Benchmark POC

## What

Compare score-based convex combination fusion (`alpha * BM25 + (1-alpha) * semantic`) against rank-based RRF for hybrid retrieval.

## Why

Current PLM uses RRF which ignores score magnitudes. Hypothesis: using actual scores could improve retrieval for technical terminology queries where BM25 exact match signals are valuable.

## Hypothesis

Convex fusion with proper score normalization will outperform RRF by >5% MRR on informed (technical terminology) queries without regressing on other query types.

## Setup

```bash
cd /home/susano/Code/personal-library-manager
direnv allow  # or: nix develop

cd poc/convex_fusion_benchmark
# Uses main project's .venv (has all PLM dependencies)
```

## Usage

```bash
# Step 1: Extract raw scores from PLM index
python score_extractor.py --output artifacts/raw_scores.json

# Step 2: Run full benchmark (alpha sweep + regression testing)
python alpha_sweep.py

# Step 3: Generate visualizations
python visualize.py
```

## Results

**Verdict: PARTIAL**

| Metric | RRF | Convex (Best) | Change |
|--------|-----|---------------|--------|
| Informed MRR | 0.608 | 0.689 | **+13.3%** |
| Needle MRR | 0.842 | 0.699 | -17.0% |
| Realistic MRR | 0.243 | 0.185 | -5.6% |

**Best Config**: Z-score normalization, alpha=0.68

Convex fusion improves informed queries but causes unacceptable regression on other query types. The improvement is not statistically significant (p=0.149).

**Recommendation**: Do NOT replace RRF. Consider query-adaptive alpha as a future POC.

## Key Findings

1. **Optimal alpha is query-type dependent**
   - Technical terms → alpha ~0.7 (BM25-heavy)
   - Natural language → alpha ~0.3 (semantic-heavy)

2. **Z-score normalization performs best** for combining different score scales

3. **RRF is more robust** because rank-based fusion doesn't require tuning per query type

## Files

| File | Description |
|------|-------------|
| `score_extractor.py` | Extract raw BM25/semantic scores from PLM |
| `fusion.py` | RRF, normalization, convex fusion implementations |
| `metrics.py` | MRR, Recall, Hit@k, bootstrap CI |
| `alpha_sweep.py` | Main benchmark runner |
| `visualize.py` | Generate plots |
| `RESULTS.md` | Detailed results and analysis |
| `EVALUATION_CRITERIA.md` | Original evaluation criteria |
| `TODO.md` | Implementation checklist |

## Full Results

See [RESULTS.md](RESULTS.md) for complete analysis, plots, and recommendations.

---

*Completed: 2026-02-22*
