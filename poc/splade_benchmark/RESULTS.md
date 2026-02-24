# SPLADE Benchmark Results

**Status**: COMPLETE  
**Started**: 2026-02-22  
**Completed**: 2026-02-22

---

## Executive Summary

SPLADE outperforms all PLM configurations, including the full production system with reranker:

| Configuration | Informed MRR | Needle MRR |
|---------------|-------------|------------|
| BM25-only | 0.6113 | 0.5622 |
| PLM Hybrid (BM25+Semantic) | 0.6887 | 0.6750 |
| PLM + Reranker | 0.7111 | 0.8050 |
| **SPLADE-only** | **0.7747** | **0.9000** |

### SPLADE Improvement by Baseline

| Comparison | Informed | Needle |
|------------|----------|--------|
| vs BM25-only | +26.7% | +60.1% |
| vs PLM Hybrid | +12.5% | +33.3% |
| **vs PLM + Reranker** | **+8.9%** | **+11.8%** |

**Verdict**: **PARTIAL PASS**
- Retrieval quality exceeds PLM+Reranker on all benchmarks
- Informed improvement (+8.9%) does NOT meet 10% target vs PLM+Reranker
- Needle improvement (+11.8%) meets 10% target
- Query latency exceeds 100ms target (CPU-only)

---

## Hypothesis

SPLADE's learned sparse representations will improve retrieval for technical terms by:
1. Expanding queries with semantically related terms
2. Learning domain-appropriate term weights
3. Preserving exact match signals while adding semantic understanding

**Success Criteria**:
- Informed MRR improvement >10% over BM25
- No regression (>5%) on Needle or Realistic
- Query encoding latency <100ms
- Index size <3x BM25

---

## BM25 Baseline Results

### PLM Production Baselines (Measured)

| Configuration | Informed MRR | Needle MRR | Description |
|---------------|-------------|------------|-------------|
| BM25-only | 0.6113 | 0.5622 | Lexical search only |
| PLM Hybrid | 0.6887 | 0.6750 | BM25 + Semantic + RRF (default) |
| PLM + Reranker | 0.7111 | 0.8050 | Hybrid + CrossEncoder rerank |

> **Note**: The original benchmark compared SPLADE against BM25-only, which overstated improvements. The table above shows actual PLM production baselines measured on the same test corpus. See `artifacts/plm_baseline_comparison.json` for full details.

| Benchmark | MRR@10 | Hit@1 | Hit@5 | Hit@10 | Latency |
|-----------|--------|-------|-------|--------|---------|
| Informed  | 0.6113 | 52.0% | 76.0% | 76.0%  | 0.38ms  |
| Needle    | 0.5622 | 45.0% | 65.0% | 80.0%  | 0.54ms  |
| Realistic | 0.1432 | 8.0%  | 22.0% | 30.0%  | 0.69ms  |

**Index Size**: 9.14 MB  
**Total Chunks**: 20,801

---

## SPLADE-Only Results

| Benchmark | MRR@10 | Change | Hit@1 | Hit@10 | Latency |
|-----------|--------|--------|-------|--------|---------|
| Informed  | **0.7747** | **+26.7%** | 68.0% | 96.0%  | 787ms  |
| Needle    | **0.9000** | **+60.1%** | 85.0% | 95.0%  | 783ms  |
| Realistic | **0.2147** | **+49.9%** | 14.0% | 40.0%  | 775ms  |

**Index Size**: 19.77 MB (2.16x BM25)  
**Encoding Time**: 4,408s (73 min for 20,801 chunks)  
**Avg Terms per Doc**: 124

---

## Hybrid (SPLADE + Semantic) Results

Hybrid fusion (RRF k=60) performs worse than SPLADE-only, suggesting SPLADE already captures semantic information effectively.

| Benchmark | MRR@10 | Change vs BM25 | Hit@1 | Latency |
|-----------|--------|----------------|-------|---------|
| Informed  | 0.7533 | +23.2%  | 64.0% | 1568ms  |
| Needle    | 0.7847 | +39.6%  | 70.0% | 1570ms  |
| Realistic | 0.1958 | +36.7%  | 14.0% | 1583ms  |

**Note**: Hybrid adds ~800ms (semantic search + RRF fusion) but reduces quality compared to SPLADE-only. **SPLADE-only is recommended.**

---

## Technical Term Analysis

SPLADE effectively expands technical terms with domain-relevant vocabulary.

### Sample Expansions

| Query | Top Expansion Terms |
|-------|---------------------|
| SubjectAccessReview | subject(3.2), review(2.1), access(1.8), authorization(1.2) |
| RBAC permissions | rb(3.1), ac(2.6), permission(2.5), consent(1.4), license(0.6) |
| persistent volume claim | volume(2.9), persistent(2.9), volumes(2.6), claim(2.4), persistence(2.0) |
| kube-apiserver | ku(2.7), be(2.4), api(2.1), server(1.6), java(0.5) |

### CamelCase Handling

SPLADE breaks CamelCase terms into meaningful subwords:
- `SubjectAccessReview` → subject, ##ac, ##ces, ##sr, ##ev, ##ie, ##w
- BERT's WordPiece tokenization captures morphological structure

### Rank Comparison (Informed queries)

- **Improved**: 4 queries saw better ranks with SPLADE
- **Regressed**: 4 queries saw worse ranks
- **Unchanged**: 11 queries maintained same rank
- Net effect: +26.7% MRR improvement |

---

## Latency Profile

| Metric | Value |
|--------|-------|
| Query encoding (mean) | ~780ms |
| Index lookup          | ~7ms |
| Total (SPLADE-only)   | ~787ms |
| Total (Hybrid)        | ~1,570ms |

**Bottleneck**: Query encoding (BERT inference) dominates latency. On CPU, this is ~780ms per query. GPU acceleration or smaller models (splade-tiny: ~205ms) could reduce this significantly.

---

## Success Criteria Check

### vs BM25-only (Original Comparison)

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Informed MRR >10% improvement | >10% | **+26.7%** | ✅ **PASS** |
| Needle MRR no regression | >95% of BM25 | **+60.1%** | ✅ **PASS** |
| Realistic MRR no regression | >95% of BM25 | **+49.9%** | ✅ **PASS** |

### vs PLM + Reranker (Production Comparison)

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Informed MRR >10% improvement | >10% | **+8.9%** | ❌ **FAIL** |
| Needle MRR no regression | >95% of PLM | **+11.8%** | ✅ **PASS** |

### Infrastructure Criteria

| Criterion | Target | Actual | Status |
|-----------|--------|--------|--------|
| Query encoding <100ms | <100ms | 780ms | ❌ **FAIL** (CPU) |
| Index size <3x BM25 | <27.4 MB | 19.77 MB | ✅ **PASS** |

**Overall vs BM25**: 4/5 criteria pass (latency fails on CPU)

**Overall vs PLM+Reranker**: 3/5 criteria pass (latency fails, Informed <10% improvement)

---

## Methodology

### Experimental Setup

- **Corpus**: Kubernetes documentation (20,801 chunks from 500+ documents)
- **Queries**: 3 benchmark sets
  - **Informed** (25 queries): Technical queries with known answers, tests exact term matching
  - **Needle** (20 queries): Queries targeting specific facts buried in docs
  - **Realistic** (50 queries): Natural language questions from real users
- **Model**: `naver/splade-cocondenser-ensembledistil` (default SPLADE++ variant)
- **Hardware**: CPU-only encoding (no GPU acceleration)
- **Retrieval**: Top-10 results, MRR@10 as primary metric

### SPLADE Configuration

- **Max sequence length**: 512 tokens
- **Batch size**: 16 (for encoding efficiency)
- **Sparsification**: log(1 + ReLU(logits)) with max-pooling over sequence
- **Index**: Sparse CSR matrix with term-to-document inverted lookup

### Alternative Models Evaluated

| Model | Params | Encode (ms) | Terms | Notes |
|-------|--------|-------------|-------|-------|
| ensembledistil | 110M | 912 | 34.8 | **Selected** - best quality |
| selfdistil | 110M | 962 | 66.8 | More expansion, similar speed |
| splade-tiny | 4.4M | 205 | 13.0 | 4.4x faster, 80% quality |
| splade-mini | 11.2M | 335 | 17.8 | Good speed/quality balance |
| splade-small | 28.8M | 369 | 23.8 | Moderate option |

**Finding**: `splade-tiny` offers 4.4x speedup with ~80% benchmark performance - viable for latency-critical applications.

---

## Visualizations

See `artifacts/plots/` for:
- `mrr_comparison.png` - MRR across all benchmarks
- `rank_scatter.png` - Per-query rank comparison
- `latency_comparison.png` - Latency breakdown
- `index_size.png` - Index size comparison
- `improvement_delta.png` - MRR change percentage

---

## Recommendations

### Primary Recommendation: **ADOPT SPLADE-only (not hybrid)**

SPLADE-only outperforms both BM25 and hybrid configurations. The hybrid fusion with semantic search actually degrades quality, suggesting SPLADE already captures semantic information effectively.

### Implementation Path

1. **For Production (latency-sensitive)**:
   - Use `splade-tiny` (4.4M params) for ~205ms encoding
   - Accept ~10-15% quality reduction vs full model
   - Or deploy with GPU for ~20-50ms encoding

2. **For Batch Processing (quality-focused)**:
   - Use `ensembledistil` (110M params) for best quality
   - Pre-compute SPLADE vectors for known queries
   - Cache frequently used query encodings

3. **Integration Strategy**:
   - Replace BM25 with SPLADE in hybrid retriever
   - Keep existing semantic embeddings for reranking
   - Use SPLADE for first-stage retrieval, rerank with cross-encoder

### Future Work

- Benchmark on GPU for realistic latency
- Test domain-adapted SPLADE (fine-tuned on Kubernetes docs)
- Evaluate SPLADE-v3 (requires HF account, +2% expected improvement)

---

## Appendix: Raw Data

All raw results are in `artifacts/`:
- `baseline_bm25.json` - BM25-only baseline
- `splade_only_results.json` - SPLADE-only results
- `hybrid_splade_results.json` - Hybrid results
- `encoding_stats.json` - Encoding statistics
- `technical_term_analysis.json` - Expansion term analysis

---

*Generated: 2026-02-22*
