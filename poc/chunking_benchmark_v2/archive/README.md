# Archive - RAG Retrieval Benchmark

This directory contains archived files from the benchmark development process.

## Directory Structure

### FINAL_RESULTS/
Final strategy comparison results from 2026-01-26:
- `enriched_hybrid_llm_FINAL_GRADING.md` - Best strategy: 93% baseline, 68.7% edge cases
- `synthetic_variants_FINAL_GRADING.md` - Alternative: 93% baseline, 57% edge cases
- `adaptive_hybrid_FINAL_GRADING.md` - Not recommended: 91% baseline, 59% edge cases
- `BASELINE_STRATEGIES_GRADING_COMPARISON.md` - Full comparison report

### old_configs/
Configuration files used during development:
- Various YAML configs for different benchmark runs
- Kept for reference if needed to reproduce old results

### old_corpus/
Test corpora used during initial development:
- `realistic_documents/` - Original 5 synthetic test documents
- `minimal_test/` - Minimal test corpus
- Ground truth and metadata files

### old_docs/
Development documentation:
- `ENRICHMENT_PLAN.md` - Original enrichment strategy planning
- `PRECISION_RESEARCH.md` - Research on precision improvements
- `RETRIEVAL_GEMS_INVALID.md` - Invalid "gems" strategy experiments
- `AUTOMATED_VS_MANUAL_EVALUATION.md` - Evaluation methodology comparison
- `SMART_CHUNKING_VALIDATION_REPORT.md` - Chunking validation results

### old_results/
Intermediate benchmark results:
- Strategy comparison grading files
- Manual test results
- Gems strategy experiments (invalidated)

## Summary of Invalid Strategies

### "Gems" Strategies (INVALID)
The following strategies from `test_gems.py` were tested but found to be invalid:
- `adaptive_hybrid` - Adaptive weighting based on query analysis
- `negation_aware` - Special handling for negation queries
- `contextual` - Context-aware retrieval
- `bm25f_hybrid` - BM25F field boosting
- `hybrid_gems` - Combined approach

**Why Invalid**: These strategies showed no significant improvement over the baseline `enriched_hybrid_llm` strategy. The "gems" approach of dynamically selecting strategies based on query characteristics did not provide consistent benefits.

**Recommendation**: Use `enriched_hybrid_llm` for all queries. It provides:
- 93% accuracy on baseline queries
- 68.7% accuracy on edge case queries
- Consistent performance across query types
- LLM query rewriting for better semantic understanding

## Current Active Results

The following results are active and in the main `results/` directory:
- Needle-haystack benchmark results (90% baseline, 65% adversarial)
- These represent the validated benchmark for the `enriched_hybrid_llm` strategy

## Archive Date
2026-01-26
