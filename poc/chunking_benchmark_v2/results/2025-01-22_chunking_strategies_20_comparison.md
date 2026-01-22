# Chunking Benchmark V2.1 - Final Results

**Date**: 2025-01-22
**Corpus**: 16 CloudFlow documents, 30 queries, 133 key facts

## Executive Summary

Tested 20 chunking strategies with corrected metrics. Key finding: **retrieval is already excellent** - the apparent 92% ceiling was a metric limitation, not a retrieval problem.

| Evaluation Method | Best Strategy | k=5 | k=10 | Theoretical Max |
|-------------------|---------------|-----|------|-----------------|
| Exact string match | fixed_512_0pct | 93.2% | 94.0% | 94.0% |
| Fuzzy matching | fixed_512_0pct | **97.0%** | 97.7% | 97.7% |

## Full Strategy Rankings (Exact Matching, k=5)

| Rank | Strategy | Chunks | Avg Tokens | Key Facts | Recall@5 | Token Recall |
|------|----------|--------|------------|-----------|----------|--------------|
| 1 | fixed_512_0pct | 17 | 332 | 92.1% | 95.6% | 95.7% |
| 2 | fixed_512_15pct | 17 | 337 | 92.1% | 95.6% | 95.7% |
| 3 | fixed_400_0pct | 21 | 269 | 90.1% | 93.1% | 92.1% |
| 4 | recursive_1600_0pct | 26 | 217 | 89.0% | 95.6% | 89.2% |
| 5 | recursive_2000_0pct | 23 | 246 | 88.4% | 97.2% | 91.1% |
| 6 | recursive_1600_15pct | 26 | 227 | 88.3% | 93.9% | 88.7% |
| 7 | fixed_size_512 (V1) | 33 | 196 | 88.1% | 91.9% | 87.4% |
| 8 | fixed_300_0pct | 28 | 202 | 87.2% | 95.6% | 85.5% |
| 9 | semantic_400 | 22 | 257 | 86.4% | 91.9% | 89.8% |
| 10 | recursive_1200_0pct | 35 | 161 | 80.6% | 90.3% | 77.4% |
| 11 | recursive_800_0pct | 48 | 117 | 79.8% | 95.6% | 75.5% |
| 12 | fixed_200_0pct | 38 | 149 | 77.7% | 93.1% | 77.0% |
| 13 | paragraphs_50_256 (V1 winner) | 103 | 55 | 75.7% | 96.4% | 71.8% |
| 14 | semantic_200 | 45 | 125 | 71.9% | 92.8% | 68.6% |
| 15 | heading_limited_512 | 188 | 29 | 57.3% | 91.4% | 48.0% |
| 16 | paragraph_50_256_heading | 193 | 29 | 56.8% | 91.4% | 52.2% |
| 17 | heading_based_h3 | 220 | 25 | 54.7% | 91.4% | 51.1% |
| 18 | hierarchical_h4 | 257 | 22 | 49.0% | 91.4% | 44.8% |
| 19 | paragraph_50_256 | 193 | 25 | 46.7% | 91.9% | 39.7% |
| 20 | heading_paragraph_h3 | 511 | 22 | 45.8% | 91.4% | 33.4% |

## Top-K Analysis

How Key Facts Coverage changes with retrieval depth:

### Exact String Matching
| Strategy | k=5 | k=7 | k=10 | k=15 |
|----------|-----|-----|------|------|
| fixed_512_0pct | 93.2% | 93.2% | 94.0% | 94.0% |
| fixed_400_0pct | 91.0% | 93.2% | 94.0% | 94.0% |
| recursive_2000 | 89.5% | 91.0% | 93.2% | 94.0% |
| recursive_1600 | 90.2% | 92.5% | 93.2% | 94.0% |
| fixed_300_0pct | 88.7% | 88.7% | 90.2% | 94.0% |

### Fuzzy Matching (handles "version 15" -> "PostgreSQL 15")
| Strategy | k=5 | k=7 | k=10 | k=15 |
|----------|-----|-----|------|------|
| fixed_512_0pct | **97.0%** | 97.0% | 97.7% | 97.7% |
| fixed_400_0pct | 94.7% | 97.0% | 97.7% | 97.7% |
| recursive_2000 | 93.2% | 94.7% | 97.0% | 97.7% |
| recursive_1600 | 94.0% | 96.2% | 97.0% | 97.7% |
| fixed_300_0pct | 92.5% | 92.5% | 94.0% | 97.7% |

## Key Findings

### 1. V1 Winner Was Wrong
The V1 benchmark incorrectly identified `paragraphs_50_256` as winner based on Recall@5 (96.4%). With proper Key Facts evaluation, it ranks **13th** with only 75.7% coverage.

### 2. Larger Chunks = Better Coverage
| Chunk Size | Key Facts Coverage |
|------------|-------------------|
| ~25 tokens | 45-57% |
| ~55 tokens | 76% |
| ~150 tokens | 78% |
| ~250 tokens | 86-90% |
| **~330 tokens** | **92%** |

### 3. Exact vs Fuzzy Matching Gap
8 key facts don't exist as exact strings in the corpus:
- "version 15" (corpus has "PostgreSQL 15")
- "Standard tier", "1000 for Pro" (tier pricing details)
- "user_id FK", "workflow_id FK" (uses "REFERENCES" syntax)
- "transaction mode" (implied but not explicit)
- "team expertise" (phrased differently)
- "revision number" (described but not named)

An LLM would understand these semantically, so fuzzy matching better represents real-world performance.

### 4. Overlap Provides No Benefit
| Strategy | Overlap | Key Facts |
|----------|---------|-----------|
| fixed_512_0pct | 0% | 92.1% |
| fixed_512_15pct | 15% | 92.1% |

### 5. Semantic Chunking Not Worth It
| Strategy | Key Facts | Chunk Time |
|----------|-----------|------------|
| semantic_400 | 86.4% | 183ms |
| fixed_400_0pct | 90.1% | 0.4ms |

457x slower with worse results.

## Theoretical Limits

| Metric | Value | Notes |
|--------|-------|-------|
| Total key facts | 133 | From ground_truth_v2.json |
| Exact match max | 125 (94.0%) | 8 facts not literally in corpus |
| Fuzzy match max | 130 (97.7%) | 3 facts truly missing |

## Current Best Configuration

```python
# Chunking
FixedSizeStrategy(chunk_size=512, overlap=0)

# Retrieval
k = 5  # Top-5 chunks

# Expected Performance
# - Exact matching: 93.2%
# - Fuzzy matching: 97.0%
```

## Path to 98%+

The remaining 3% gap (97.0% -> 100%) requires:
1. **Hybrid retrieval** (semantic + BM25) for exact keyword matches
2. **Query expansion** to match different phrasings
3. **Cross-encoder reranking** for better top-k selection
4. **Larger/diverse test corpus** to validate findings

## Files Generated

```
poc/chunking_benchmark_v2/
├── run_comprehensive_benchmark.py  # All 20 strategies
├── test_top_k.py                   # k-value analysis
├── test_strategies_higher_k.py     # Strategy + k combinations
├── find_missing_facts.py           # Missing fact analysis
└── results/
    ├── comprehensive_benchmark.json
    ├── comprehensive_report.md
    └── BENCHMARK_RESULTS_V2.1.md   # This file
```
