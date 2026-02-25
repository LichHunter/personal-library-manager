# PLM vs Baseline RAG Benchmark Report

**Generated:** 2026-02-19  
**Corpus:** Kubernetes Documentation (1,569 docs, 20,801 chunks)

## Executive Summary

PLM's hybrid retrieval system provides **statistically significant improvement** on targeted needle questions but shows **marginal gains** on broader realistic queries. The improvement comes almost entirely from **BM25 + RRF fusion**, not from content enrichment.

| Benchmark | PLM MRR | Baseline MRR | Improvement | Significant? |
|-----------|---------|--------------|-------------|--------------|
| Needle (20 queries) | 0.842 | 0.713 | +18.1% | **Yes** |
| Realistic (400 queries) | 0.196 | 0.192 | +2.0% | No |

## Benchmark Configurations

### Three Variants Compared

| Variant | Description | Purpose |
|---------|-------------|---------|
| **PLM Full** | BM25 + Semantic + RRF fusion + Query expansion + Enriched embeddings | Full production system |
| **Baseline-Enriched** | FAISS + PLM's enriched content (keywords prepended) | Isolate enrichment contribution |
| **Baseline-Naive** | FAISS + Raw chunk content | Pure semantic baseline |

### Shared Configuration
- **Embedding model:** BAAI/bge-base-en-v1.5 (768-dim, normalized)
- **Chunk boundaries:** Identical across all variants (extracted from PLM)
- **k:** 5 results per query
- **Similarity:** Cosine (via IndexFlatIP after L2 normalization)

## Detailed Results

### Needle Benchmark (20 queries)

Targeted questions about Kubernetes Topology Manager. All queries have a single known target document.

| Metric | PLM | Enriched | Naive |
|--------|-----|----------|-------|
| **MRR** | **0.842** | 0.696 | 0.713 |
| **Hit@1** | **80%** | 60% | 60% |
| **Hit@3** | **90%** | 80% | 80% |
| **Hit@5** | **90%** | 85% | 85% |
| **Found** | **18/20** | 17/20 | 17/20 |
| **Avg Latency** | 430ms | 124ms | 89ms |

**95% Confidence Intervals (Bootstrap, n=1000):**
- PLM vs Naive: **[+0.033, +0.242]** — Significant (excludes 0)
- PLM vs Enriched: **[+0.050, +0.263]** — Significant
- Enriched vs Naive: [-0.163, +0.113] — Not significant

**Attribution Analysis:**
```
Total PLM improvement:      +0.129 MRR (+18.1%)
├── Enrichment contribution: -0.017 MRR (-2.3%)  ← Hurt performance!
└── RRF/BM25/expansion:      +0.146 MRR (+20.5%)
```

### Realistic Benchmark (400 queries)

Diverse queries across entire Kubernetes documentation, each with a target document.

| Metric | PLM | Enriched | Naive |
|--------|-----|----------|-------|
| **MRR** | **0.196** | 0.186 | 0.192 |
| **Hit@1** | 12.5% | **12.8%** | **12.8%** |
| **Hit@3** | **25.3%** | 22.0% | 23.3% |
| **Hit@5** | **32.0%** | 29.5% | 30.5% |
| **Found** | **128/400** | 118/400 | 122/400 |

**95% Confidence Intervals:**
- PLM vs Naive: [-0.031, +0.038] — Not significant (includes 0)
- PLM vs Enriched: [-0.018, +0.038] — Not significant
- Enriched vs Naive: [-0.030, +0.019] — Not significant

**Attribution Analysis:**
```
Total PLM improvement:      +0.004 MRR (+2.0%)
├── Enrichment contribution: -0.006 MRR (-3.1%)
└── RRF/BM25/expansion:      +0.010 MRR (+5.1%)
```

## Key Findings

### 1. PLM Excels on Targeted Queries

The needle benchmark shows PLM's strengths clearly:
- **+20% Hit@1** improvement (80% vs 60%)
- **18% MRR gain** over baselines
- Statistically significant at 95% confidence

PLM particularly excels on:
- Natural language problem descriptions ("topology affinity error")
- Questions requiring term matching ("NUMA nodes", "kubelet flag")
- Queries where BM25 lexical matching helps semantic search

### 2. Enrichment Provides No Benefit (or Hurts)

**Surprising finding:** Content enrichment (prepending keywords/entities) showed *negative* contribution in both benchmarks:
- Needle: -0.017 MRR (-2.3%)
- Realistic: -0.006 MRR (-3.1%)

**Hypothesis:** Enriched content may be creating noise for pure semantic search. The BGE model is already good at semantic understanding; prepending extracted keywords may dilute the embedding signal.

### 3. RRF + BM25 is the Key Differentiator

All PLM improvement comes from the hybrid search components:
- **BM25 lexical matching** catches exact term matches semantic misses
- **RRF fusion** combines signals effectively
- **Query expansion** (not isolated) likely helps broader coverage

### 4. Realistic Queries Are Harder

The low MRR across all variants on realistic queries (0.19-0.20) suggests:
- Many questions may not have ideal answers in the corpus
- Kubernetes documentation is large (1,569 docs) with overlapping content
- Some "realistic" questions may be ambiguous or too broad

### 5. Latency-Accuracy Tradeoff

| Variant | Avg Latency | Relative |
|---------|-------------|----------|
| PLM | 430ms | 1.0x |
| Enriched | 124ms | 0.29x |
| Naive | 89ms | 0.21x |

PLM is 3-5x slower due to:
- BM25 index lookup
- RRF score computation
- Query expansion (when enabled)

This is acceptable for search applications but may matter for high-throughput use cases.

## Recommendations

### Keep
- **BM25 + RRF fusion** — This is PLM's primary value add
- **Identical chunk boundaries** — Fair comparison methodology

### Reconsider
- **Content enrichment** — No measurable benefit for retrieval; may be useful only for final answer generation
- **Query expansion** — Not isolated in this benchmark; recommend separate A/B test

### Future Work
1. **Isolate query expansion** — Run PLM with/without expansion to quantify its contribution
2. **Test on different corpora** — Kubernetes may not be representative
3. **LLM grading** — Current metrics only measure retrieval; answer quality may differ
4. **Chunk size experiments** — Current chunks may not be optimal

## Methodology Notes

### Chunk Extraction
All chunks extracted from PLM's production SQLite database to ensure:
- Identical chunk boundaries across variants
- Same enriched content available for Baseline-Enriched
- No chunking strategy differences confounding results

### FAISS Index
- IndexFlatIP (inner product) used with normalized embeddings
- Equivalent to cosine similarity
- Cached to disk after first build (~63MB per variant)

### Statistical Testing
- Bootstrap resampling (n=1000) for confidence intervals
- 95% CI computed on paired MRR differences
- Significance determined by CI excluding 0

## Raw Data

See `needle_benchmark.json` and `realistic_benchmark.json` for per-query results.
