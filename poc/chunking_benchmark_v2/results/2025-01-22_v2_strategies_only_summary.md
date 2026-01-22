# Chunking Benchmark V2.1 Results (FIXED METRICS)

**Corpus:** 16 documents, 30 test queries
**Device:** cuda

## Metric Fixes in V2.1

- Token metrics now calculated **per-document** then aggregated (was: incorrectly mixed across docs)
- Added **Key Facts Coverage** - % of key facts found in retrieved chunks (best RAG quality proxy)
- Token metrics calculated against **ALL expected docs** (penalizes missing docs)

## Summary

| Strategy | Chunks | Avg Tokens | Key Facts | Recall@5 | Token Precision | Token Recall |
|----------|--------|------------|-----------|----------|-----------------|--------------|
| fixed_512_0pct | 17 | 332 | 92.1% | 95.6% | 2.7% | 95.7% |
| fixed_512_15pct | 17 | 337 | 92.1% | 95.6% | 2.8% | 95.7% |
| fixed_400_0pct | 21 | 269 | 90.1% | 93.1% | 3.2% | 92.1% |
| recursive_1600_0pct | 26 | 217 | 89.0% | 95.6% | 3.9% | 89.2% |
| recursive_2000_0pct | 23 | 246 | 88.4% | 97.2% | 3.4% | 91.1% |
| recursive_1600_15pct | 26 | 227 | 88.3% | 93.9% | 3.8% | 88.7% |
| semantic_400 | 22 | 257 | 86.4% | 91.9% | 3.5% | 89.8% |
| recursive_800_0pct | 48 | 117 | 79.8% | 95.6% | 5.4% | 75.5% |
| fixed_200_0pct | 38 | 149 | 77.7% | 93.1% | 4.8% | 77.0% |
| semantic_200 | 45 | 125 | 71.9% | 92.8% | 6.1% | 68.6% |
| paragraph_50_256_heading | 193 | 29 | 56.8% | 91.4% | 14.9% | 52.2% |
| paragraph_50_256 | 193 | 25 | 46.7% | 91.9% | 12.6% | 39.7% |

## Recommendation

**Best strategy by Key Facts Coverage: fixed_512_0pct**

- Key Facts Coverage: 92.1%
- Recall@5: 95.6%
- Token Precision: 2.7%
- Token Recall: 95.7%
- 17 chunks averaging 332 tokens
