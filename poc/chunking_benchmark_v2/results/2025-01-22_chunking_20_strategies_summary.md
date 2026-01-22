# Comprehensive Chunking Benchmark Results

**Corpus:** 16 documents, 30 test queries
**Device:** cuda
**Strategies tested:** 20 (all V1 + V2 strategies)

## Evaluation Methodology

**Primary metric: Key Facts Coverage** - % of key facts found in retrieved chunks
- Directly measures: Can the LLM answer the question?
- Token metrics calculated per-document (fixed bug from V2.0)

## Results Ranked by Key Facts Coverage

| Rank | Strategy | Chunks | Avg Tokens | Key Facts | Recall@5 | Token Recall | Token Precision |
|------|----------|--------|------------|-----------|----------|--------------|-----------------|
| 1 | fixed_512_0pct | 17 | 332 | 92.1% | 95.6% | 95.7% | 2.7% |
| 2 | fixed_512_15pct | 17 | 337 | 92.1% | 95.6% | 95.7% | 2.8% |
| 3 | fixed_400_0pct | 21 | 269 | 90.1% | 93.1% | 92.1% | 3.2% |
| 4 | recursive_1600_0pct | 26 | 217 | 89.0% | 95.6% | 89.2% | 3.9% |
| 5 | recursive_2000_0pct | 23 | 246 | 88.4% | 97.2% | 91.1% | 3.4% |
| 6 | recursive_1600_15pct | 26 | 227 | 88.3% | 93.9% | 88.7% | 3.8% |
| 7 | fixed_size_512 | 33 | 196 | 88.1% | 91.9% | 87.4% | 4.4% |
| 8 | fixed_300_0pct | 28 | 202 | 87.2% | 95.6% | 85.5% | 3.9% |
| 9 | semantic_400 | 22 | 257 | 86.4% | 91.9% | 89.8% | 3.5% |
| 10 | recursive_1200_0pct | 35 | 161 | 80.6% | 90.3% | 77.4% | 4.4% |
| 11 | recursive_800_0pct | 48 | 117 | 79.8% | 95.6% | 75.5% | 5.4% |
| 12 | fixed_200_0pct | 38 | 149 | 77.7% | 93.1% | 77.0% | 4.8% |
| 13 | paragraphs_50_256 | 103 | 55 | 75.7% | 96.4% | 71.8% | 10.9% |
| 14 | semantic_200 | 45 | 125 | 71.9% | 92.8% | 68.6% | 6.1% |
| 15 | heading_limited_512 | 188 | 29 | 57.3% | 91.4% | 48.0% | 14.3% |
| 16 | paragraph_50_256_heading | 193 | 29 | 56.8% | 91.4% | 52.2% | 14.9% |
| 17 | heading_based_h3 | 220 | 25 | 54.7% | 91.4% | 51.1% | 14.7% |
| 18 | hierarchical_h4 | 257 | 22 | 49.0% | 91.4% | 44.8% | 14.9% |
| 19 | paragraph_50_256 | 193 | 25 | 46.7% | 91.9% | 39.7% | 12.6% |
| 20 | heading_paragraph_h3 | 511 | 22 | 45.8% | 91.4% | 33.4% | 19.5% |

## Winner Analysis

### Best Strategy: `fixed_512_0pct`

- **Key Facts Coverage:** 92.1%
- **Recall@5:** 95.6%
- **Token Recall:** 95.7%
- **Token Precision:** 2.7%
- **Number of chunks:** 17
- **Average tokens per chunk:** 332

## Analysis by Strategy Type

### Small Chunks (< 100 tokens avg)

Best: `paragraphs_50_256` - 75.7% Key Facts

### Medium Chunks (100-250 tokens avg)

Best: `recursive_1600_0pct` - 89.0% Key Facts

### Large Chunks (250+ tokens avg)

Best: `fixed_512_0pct` - 92.1% Key Facts

## Final Recommendation

Use **`fixed_512_0pct`** for the Personal Knowledge Assistant.

This strategy provides the best balance of:
- High Key Facts Coverage (92.1%)
- High Token Recall (95.7%)
- Reasonable chunk count (17 chunks)
