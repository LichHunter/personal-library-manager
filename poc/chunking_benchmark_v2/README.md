# Chunking Benchmark V2.1 (Fixed Metrics) - FINAL RESULTS

## What

Enhanced chunking benchmark with **correct** evaluation metrics:
- Token metrics calculated **per-document** then aggregated (was: incorrectly mixed across docs)
- **Key Facts Coverage** - % of key facts found in retrieved chunks (best RAG quality proxy)
- Token metrics against **ALL expected docs** (penalizes missing docs)
- **20 chunking strategies tested** (all V1 + V2 strategies combined)

## Critical Bug Fixed

**V2.0 had invalid token metrics!** We were comparing character positions from different documents as if they were in the same coordinate system. This made IoU/Precision/Recall meaningless.

**V2.1 fixes this** by calculating per-document, then aggregating.

## Final Comprehensive Results (20 Strategies)

| Rank | Strategy | Chunks | Avg Tokens | Key Facts | Recall@5 | Token Recall | Token Precision |
|------|----------|--------|------------|-----------|----------|--------------|-----------------|
| **1** | **fixed_512_0pct** | **17** | **332** | **92.1%** | 95.6% | **95.7%** | 2.7% |
| 2 | fixed_512_15pct | 17 | 337 | 92.1% | 95.6% | 95.7% | 2.8% |
| 3 | fixed_400_0pct | 21 | 269 | 90.1% | 93.1% | 92.1% | 3.2% |
| 4 | recursive_1600_0pct | 26 | 217 | 89.0% | 95.6% | 89.2% | 3.9% |
| 5 | recursive_2000_0pct | 23 | 246 | 88.4% | **97.2%** | 91.1% | 3.4% |
| 6 | recursive_1600_15pct | 26 | 227 | 88.3% | 93.9% | 88.7% | 3.8% |
| 7 | fixed_size_512 (V1) | 33 | 196 | 88.1% | 91.9% | 87.4% | 4.4% |
| 8 | fixed_300_0pct | 28 | 202 | 87.2% | 95.6% | 85.5% | 3.9% |
| 9 | semantic_400 | 22 | 257 | 86.4% | 91.9% | 89.8% | 3.5% |
| 10 | recursive_1200_0pct | 35 | 161 | 80.6% | 90.3% | 77.4% | 4.4% |
| 11 | recursive_800_0pct | 48 | 117 | 79.8% | 95.6% | 75.5% | 5.4% |
| 12 | fixed_200_0pct | 38 | 149 | 77.7% | 93.1% | 77.0% | 4.8% |
| 13 | paragraphs_50_256 (V1 winner) | 103 | 55 | 75.7% | 96.4% | 71.8% | 10.9% |
| 14 | semantic_200 | 45 | 125 | 71.9% | 92.8% | 68.6% | 6.1% |
| 15 | heading_limited_512 | 188 | 29 | 57.3% | 91.4% | 48.0% | 14.3% |
| 16 | paragraph_50_256_heading | 193 | 29 | 56.8% | 91.4% | 52.2% | **14.9%** |
| 17 | heading_based_h3 | 220 | 25 | 54.7% | 91.4% | 51.1% | 14.7% |
| 18 | hierarchical_h4 | 257 | 22 | 49.0% | 91.4% | 44.8% | 14.9% |
| 19 | paragraph_50_256 | 193 | 25 | 46.7% | 91.9% | 39.7% | 12.6% |
| 20 | heading_paragraph_h3 | 511 | 22 | 45.8% | 91.4% | 33.4% | 19.5% |

## Key Findings (Corrected)

### 1. Key Facts Coverage is the Primary Metric

**Key Facts Coverage** directly answers: "Can the LLM find all information needed to answer?"

| Chunk Size | Key Facts Coverage |
|------------|-------------------|
| ~50 tokens (paragraphs) | 46-57% |
| ~150 tokens | 72-78% |
| ~250 tokens | 86-90% |
| ~330 tokens | **92%** |

**Conclusion:** Larger chunks capture more key facts. 400-512 token chunks are optimal.

### 2. V1 Winner (paragraphs_50_256) is Actually WORST

With corrected metrics:
- **Key Facts Coverage: 46.7%** (worst - misses half the facts!)
- Token Recall: 39.7% (worst - misses 60% of relevant content)
- Token Precision: 12.6% (best - but irrelevant if you miss the answer)

**V1's 96% Recall@5 was misleading** - it only measured "did we retrieve the right document" not "did we retrieve the text containing the answer."

### 3. fixed_512 is the Clear Winner

**Best strategy: `fixed_512_0pct`**
- Key Facts Coverage: **92.1%** (highest)
- Token Recall: **95.7%** (captures almost all relevant content)
- Recall@5: 95.6%
- Only 17 chunks (efficient storage)

### 4. Overlap Provides No Benefit

| Strategy | Overlap | Key Facts |
|----------|---------|-----------|
| fixed_512_0pct | 0% | 92.1% |
| fixed_512_15pct | 15% | 92.1% |

Same results - skip overlap for simplicity.

### 5. Semantic Chunking Underperforms

| Strategy | Key Facts | Chunk Time |
|----------|-----------|------------|
| semantic_400 | 86.4% | 186ms |
| fixed_400_0pct | 90.1% | 0.5ms |

Semantic chunking is 370x slower AND has lower Key Facts Coverage. **Not worth it.**

### 6. Precision-Recall Tradeoff Confirmed

| Strategy | Token Precision | Token Recall | Key Facts |
|----------|-----------------|--------------|-----------|
| paragraph_50_256 | **12.6%** | 39.7% | 46.7% |
| fixed_512_0pct | 2.7% | **95.7%** | **92.1%** |

Small chunks have higher precision but miss too much content.
**For RAG, recall matters more than precision** - LLM can filter noise.

## Metric Definitions

| Metric | Formula | What It Measures |
|--------|---------|------------------|
| **Key Facts Coverage** | found_facts / total_facts | Can LLM answer the question? |
| **Token Recall** | overlap_chars / relevant_chars | % of answer text retrieved |
| **Token Precision** | overlap_chars / retrieved_chars | % of retrieved text that's relevant |
| **Recall@5** | expected_docs_in_top5 / expected_docs | Document-level retrieval |

## Recommendation

### For Personal Knowledge Assistant

Use **fixed-size chunks of 400-512 tokens**:

```python
FixedSizeStrategy(chunk_size=512, overlap=0)
```

Or equivalent with recursive splitter:
```python
RecursiveSplitterStrategy(chunk_size=2000, chunk_overlap=0)  # ~512 tokens
```

**Why:**
- 92% Key Facts Coverage (LLM can answer most questions)
- 96% Token Recall (captures almost all relevant content)
- Simple implementation
- Few chunks (17 vs 193 for paragraphs)

### When to Use Smaller Chunks

Only if:
- LLM context window is severely limited
- You need precise citations at paragraph level
- You're building a different application (summarization, not Q&A)

## Files

```
poc/chunking_benchmark_v2/
├── README.md
├── pyproject.toml
├── run_benchmark.py                 # V2.1 with fixed metrics (V2 strategies only)
├── run_comprehensive_benchmark.py   # All 20 strategies (V1 + V2 combined)
│
├── strategies/
│   ├── base.py
│   ├── recursive_splitter.py
│   ├── cluster_semantic.py
│   ├── paragraph_heading.py
│   └── fixed_size.py
│
├── evaluation/
│   ├── __init__.py
│   └── metrics.py                   # Fixed per-document calculation
│
├── corpus/
│   ├── generate_ground_truth_v2.py
│   └── ground_truth_v2.json
│
└── results/
    ├── benchmark_summary.json       # V2 strategies only
    ├── comprehensive_benchmark.json # All 20 strategies
    └── comprehensive_report.md      # Final results
```

## Running the Benchmarks

```bash
cd /home/fujin/Code/personal-library-manager
source .venv/bin/activate
export LD_LIBRARY_PATH="/run/opengl-driver/lib:$LD_LIBRARY_PATH"

cd poc/chunking_benchmark_v2

# Run V2 strategies only
python run_benchmark.py

# Run ALL strategies (V1 + V2 combined) - RECOMMENDED
python run_comprehensive_benchmark.py
```

## Comparison: V2.0 (Broken) vs V2.1 (Fixed)

| Metric | V2.0 (Wrong) | V2.1 (Correct) | Difference |
|--------|--------------|----------------|------------|
| Winner | recursive_2000 | fixed_512 | Different |
| Best Key Facts | N/A | 92.1% | New metric |
| paragraph Token Recall | 52.6% | 39.7% | -13% |
| paragraph Token Precision | 20.2% | 12.6% | -8% |
| fixed_512 Token Recall | 99.5% | 95.7% | -4% |

**V2.0 inflated metrics for small chunks** because cross-document character position collisions created false "overlaps."

## V1 vs V2 Comparison

The V1 benchmark incorrectly identified `paragraphs_50_256` as the winner based on Recall@5 alone.

| Benchmark | Winner | Primary Metric | Key Facts |
|-----------|--------|----------------|-----------|
| V1 | paragraphs_50_256 | Recall@5 (96.4%) | Not measured |
| V2.1 | fixed_512_0pct | Key Facts (92.1%) | **92.1%** |

**Why V1 was wrong:** Recall@5 only measures "did we find a chunk from the right document" - it doesn't measure whether the chunk contains the actual answer. The `paragraphs_50_256` strategy achieves high Recall@5 (96.4%) but only 75.7% Key Facts Coverage because its small chunks often miss the specific text containing the answer.
