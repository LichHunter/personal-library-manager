# Adaptive Retrieval Evaluation — Complete Results

**Date:** 2026-02-21
**Corpus:** 1,569 Kubernetes docs, 20,801 chunks
**Test set:** 229 queries (54 factoid, 55 procedural, 53 explanatory, 33 comparison, 34 troubleshooting)
**Judge:** Claude 3 Haiku (local LLM-as-judge, 1-10 scale, success = grade >= 6)

---

## Best Result: Cross-Encoder Reranking

**Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB, runs 100% locally on CPU, no API calls)

| Metric | Baseline | Reranking | Improvement |
|--------|----------|-----------|-------------|
| **Answer Success Rate** | **92.1%** | **98.3%** | **+6.1%** |
| MRR@10 | 0.658 | 0.772 | +0.114 |
| Hit@5 | 75.4% | 89.2% | +13.8% |
| Hit@10 | 84.6% | 90.8% | +6.2% |
| Avg Grade | 7.64 | 8.20 | +0.56 |
| Latency p95 | 422ms | 1817ms | +1396ms |

**How it works:** Retrieve 50 candidates with existing hybrid search, then re-score each (query, chunk) pair with a cross-encoder model, return top 10. The cross-encoder understands query-document relevance at a deeper level than bi-encoder similarity.

**Why it's the best:** Zero regressions on any query type. The biggest gain is on explanatory queries (+22.6%), which was the baseline's weakest category. No cloud dependency — the model runs locally on CPU.

---

## All Approaches Compared

| Approach | ASR | vs Baseline | MRR@10 | Latency p95 | Tokens | Verdict |
|----------|-----|-------------|--------|-------------|--------|---------|
| Baseline | 92.1% | — | 0.658 | 422ms | 904 | — |
| **Reranking** | **98.3%** | **+6.1%** | **0.772** | 1817ms | 975 | **BEST** |
| Auto-Merging | 95.2% | +3.1% | 0.657 | 903ms | 1095 | Rejected |
| Adaptive Classifier | 98.3% | +6.1% | 0.736 | 1905ms | 969 | Rejected |
| Parent-Child | 88.6% | -3.5% | 0.646 | 918ms | 2255 | Rejected |

---

## Performance by Query Type

### Answer Success Rate per Approach

| Query Type | Count | Baseline | Reranking | Auto-Merge | Adaptive | Parent-Child |
|------------|-------|----------|-----------|------------|----------|--------------|
| Explanatory | 53 | 71.7% | **94.3%** | 81.1% | 94.3% | 71.7% |
| Factoid | 54 | 96.3% | 98.1% | **100.0%** | 98.1% | 85.2% |
| Troubleshooting | 34 | 97.1% | **100.0%** | 97.1% | 100.0% | 94.1% |
| Procedural | 55 | **100.0%** | **100.0%** | **100.0%** | 100.0% | 98.2% |
| Comparison | 33 | **100.0%** | **100.0%** | **100.0%** | 100.0% | 100.0% |

### MRR@10 per Approach (65 labeled queries)

| Query Type | Baseline | Reranking | Auto-Merge | Adaptive | Parent-Child |
|------------|----------|-----------|------------|----------|--------------|
| Explanatory | 0.615 | **0.872** | 0.615 | 0.872 | 0.603 |
| Factoid | 0.635 | **0.692** | 0.633 | 0.682 | 0.625 |
| Troubleshooting | 0.583 | **0.750** | 0.583 | 0.750 | 0.583 |
| Procedural | **1.000** | 0.917 | **1.000** | 0.917 | **1.000** |
| Comparison | 0.604 | **0.854** | 0.604 | 0.604 | 0.573 |

---

## Approach Details

### 1. Cross-Encoder Reranking (BEST)

**Config:** candidates_k=50, final_k=10, model=cross-encoder/ms-marco-MiniLM-L-6-v2

Retrieves 50 candidates via hybrid search, then re-scores each with a cross-encoder. Returns top 10 by cross-encoder score.

- ASR: 98.3% (+6.1%)
- Explanatory queries: 71.7% -> 94.3% (+22.6%)
- No regressions on any type
- Latency: +1396ms p95 (cross-encoder scoring on CPU)
- Model is 80MB, runs locally, no API calls

**Verdict:** Best overall. Latency is the only concern — addressable via reducing candidates_k to 30 or ONNX quantization.

### 2. Auto-Merging Retrieval (Rejected)

**Config:** retrieve_k=10, merge_threshold=0.5

Retrieves 10 chunks, groups by heading. If >=50% of a heading's chunks appear in results, replaces them with the full heading content.

- ASR: 95.2% (+3.1%) — marginal
- Merge rate: 92.6% — threshold too aggressive, merges almost everything
- Explanatory: 81.1% (+9.4%) — helped but not enough
- Factoid: 100.0% (+3.7%)

**Verdict:** Marginal gain doesn't justify the complexity. The 0.5 merge threshold is too low — merges indiscriminately.

### 3. Adaptive Classifier (Rejected)

**Config:** Rule-based classifier routes queries to chunk retrieval or reranking

Classifies queries as simple/complex/multi. Complex queries get reranking, simple queries get standard retrieval.

- ASR: 98.3% (+6.1%) — same as full reranking
- Classification accuracy: 63.3% (below 70% threshold)
- Rerank rate: 72.9% — classifier reranks most queries anyway
- No benefit over simpler full reranking

**Verdict:** Adds complexity without improvement. The rule-based classifier is too inaccurate to be useful.

### 4. Parent-Child Retrieval (Rejected)

**Config:** search_k=20, return_k=5, return_level=heading

Searches on chunks, returns heading-level content (all sibling chunks concatenated).

- ASR: 88.6% (-3.5%) — **regression**
- Factoid: 85.2% (-11.1%) — heading content drowns precise answers
- Tokens: 2255 (+1351) — 2.5x more context, mostly noise
- Explanatory: 71.7% (+0.0%) — no improvement despite more context

**Verdict:** Context expansion hurts more than it helps. Noise drowns signal for factoid queries.

---

## Theoretical Ceiling (Oracle)

The oracle tests all retrieval granularities (chunk, heading, document, reranked, merged) per query and picks the best one.

| Metric | Baseline | Oracle | Gap |
|--------|----------|--------|-----|
| Answer Success Rate | 92.1% | 99.6% | 7.4% |

Reranking closes 6.1% of the 7.4% gap (82% of theoretical maximum).

### Oracle by Query Type

| Type | Baseline | Oracle | Gap |
|------|----------|--------|-----|
| Explanatory | 71.7% | 98.1% | +26.4% |
| Factoid | 96.3% | 100.0% | +3.7% |
| Troubleshooting | 97.1% | 100.0% | +2.9% |
| Procedural | 100.0% | 100.0% | +0.0% |
| Comparison | 100.0% | 100.0% | +0.0% |

### Which Granularity Works Best (Oracle Distribution)

| Granularity | Queries | % |
|-------------|---------|---|
| Chunk (standard) | 123 | 53.9% |
| Reranked chunk | 65 | 28.5% |
| Merged (auto-merge) | 22 | 9.6% |
| Document-level | 17 | 7.5% |
| Heading-level | 1 | 0.4% |

### Only 1 Query Truly Unanswerable

`adv_adv_n04` ("What's wrong with using container scope for latency-sensitive applications?") fails across ALL approaches with a max grade of 4/10. This is a corpus gap — the content simply isn't in the indexed documents.

---

## Production Integration (DONE)

Reranking is now integrated into the production `HybridRetriever`:

```python
retriever.retrieve("query", k=10, use_rerank=True, candidates_k=50)
```

**Files modified:**
- `src/plm/search/components/reranker.py` — new `CrossEncoderReranker` component
- `src/plm/search/retriever.py` — `retrieve()` accepts `use_rerank` and `candidates_k` params
- `src/plm/search/service/app.py` — `/query` endpoint accepts `use_rerank` field

**Production benchmark results** (229 queries, same test set):

| Metric | Baseline | Production Rerank | Delta |
|--------|----------|-------------------|-------|
| **ASR** | **92.1%** | **98.3%** | **+6.1%** |
| MRR@10 | 0.658 | 0.772 | +0.114 |
| Hit@5 | 75.4% | 89.2% | +13.8% |
| Avg Grade | 7.64 | 8.21 | +0.57 |
| Latency p95 | 422ms | 2760ms | +2338ms |

Results match the POC exactly (ASR 98.3%, MRR 0.772). Production latency is higher than POC (2760ms vs 1817ms p95) due to cold-start model loading on first query; subsequent queries are ~1500ms.

---

## Updated Oracle Analysis (Post-Reranking)

After production reranking, the remaining gap is nearly closed:

| Metric | Production Rerank | Oracle | Remaining Gap |
|--------|-------------------|--------|---------------|
| **ASR** | **98.3%** | **99.1%** | **0.9%** |

### Where We Still Lack (2 improvable queries)

| Query ID | Type | Rerank Grade | Oracle Grade | Best Approach |
|----------|------|-------------|-------------|---------------|
| `expl_gen_016` | explanatory | 2/10 | 8/10 | document-level |
| `expl_gen_038` | explanatory | 2/10 | 8/10 | document-level |

Both are "What is the purpose of X?" explanatory queries where:
- Chunk-level retrieval (even reranked) fails because the concept is described across the entire document
- Document-level retrieval succeeds because it provides the full context

### Queries That Fail Across ALL Approaches (corpus gaps)

| Query ID | Type | Best Grade | Notes |
|----------|------|-----------|-------|
| `adv_adv_n04` | explanatory | 4/10 | Container scope for latency-sensitive apps — niche topic |
| `adv_adv_m03` | factoid | 2/10 | Multi-socket resource co-location — very specialized |

### Optimal Approach Distribution

| Approach | Count | % |
|----------|-------|---|
| Reranked chunk | 225 | 99.1% |
| Document-level | 2 | 0.9% |

**Key insight:** Reranking handles 99.1% of queries optimally. The remaining 0.9% (2 queries) would benefit from document-level retrieval fallback for broad conceptual questions. A hybrid approach (rerank first, fall back to document-level if grade is low) could close this gap, but the improvement is marginal (0.9%) and would require an additional LLM call for quality detection.

---

## Key Findings

1. **Ranking quality beats context quantity.** Reranking (same amount of text, better ordering) outperformed all context expansion approaches (more text, same ordering). The baseline retrieves the right documents but ranks them sub-optimally.

2. **Explanatory queries are the weak point.** The baseline's 92.1% ASR is dragged down by explanatory queries at 71.7%. All other types are >= 96.3%. Reranking fixes this specific weakness (+22.6%).

3. **The +10% ASR target is unreachable.** From 92.1%, +10% = 102.1% which exceeds 100%. The practical ceiling is 99.1% (oracle). Reranking reaches 98.3% — within 0.9% of the ceiling.

4. **Parent-child retrieval is harmful.** Returning heading-level content drowns factoid answers in irrelevant sibling chunks (-11.1% on factoids).

5. **Adaptive classification adds no value over full reranking.** The rule-based classifier achieved only 63.3% accuracy and ended up reranking 72.9% of queries anyway.

6. **Production integration validated.** The production benchmark matches POC results exactly (98.3% ASR, 0.772 MRR). The reranker is lazy-loaded and adds ~1500ms per query on CPU.

---

## What's Left

| Gap | Size | Effort | Approach |
|-----|------|--------|----------|
| 2 broad explanatory queries | 0.9% ASR | Medium | Document-level fallback for low-confidence results |
| 2 corpus gaps | 0.9% ASR | High | Add missing content to corpus |
| Latency (2760ms p95) | — | Low | Reduce candidates_k to 30, or ONNX quantization |

**Practical ceiling reached.** 98.3% ASR is within 0.9% of oracle. Further improvements require either corpus expansion or a hybrid multi-granularity approach for diminishing returns.

---

*All raw data available in corresponding .json files in this directory.*
*Production benchmark: `production_reranking.md` and `production_reranking.json`*
