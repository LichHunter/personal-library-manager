# SPLADE Benchmark POC

## Results Summary

**Status**: COMPLETE | **Verdict**: PARTIAL PASS

| Benchmark | BM25 MRR | SPLADE MRR | Improvement |
|-----------|----------|------------|-------------|
| Informed | 0.6113 | **0.7747** | **+26.7%** |
| Needle | 0.5622 | **0.9000** | **+60.1%** |
| Realistic | 0.1432 | **0.2147** | **+49.9%** |

**Key Findings**:
- SPLADE exceeds 10% improvement target by 2.6x on Informed queries
- Latency is high on CPU (~780ms) but index size is acceptable (2.16x BM25)
- Hybrid (SPLADE+Semantic) performs worse than SPLADE-only — **use SPLADE-only**
- `splade-tiny` offers 4.4x speedup with ~80% quality for latency-critical applications

**Full results**: See [RESULTS.md](./RESULTS.md)

---

## What is SPLADE?

**SPLADE** (SParse Lexical AnD Expansion) is a learned sparse retrieval model that bridges the gap between traditional lexical search (BM25) and dense semantic search (embeddings).

### The Problem SPLADE Solves

Traditional BM25 has a critical flaw for technical documentation:

| Query | Document Contains | BM25 Result |
|-------|-------------------|-------------|
| "subject access review" | "SubjectAccessReview" | **NO MATCH** |
| "k8s pod" | "Kubernetes Pod" | **NO MATCH** |
| "RBAC permissions" | "role-based access control" | **NO MATCH** |

BM25 only matches **exact tokens**. It cannot:
- Understand that "k8s" means "Kubernetes"
- Split CamelCase terms
- Expand acronyms
- Connect synonyms

Dense embeddings solve some of these but lose **exact match precision** - they might rank a document about "code review" highly for a query about "SubjectAccessReview" because both contain "review" semantically.

### How SPLADE Works

SPLADE uses BERT's Masked Language Model (MLM) head to learn **sparse term weights** for both queries and documents. Instead of a single dense vector, SPLADE produces a sparse vector over the entire vocabulary (~30,000 terms).

**Example**:
```
Input: "SubjectAccessReview webhook"

SPLADE Output (sparse vector):
{
  "SubjectAccessReview": 2.8,   ← Original term, high weight
  "webhook": 2.1,               ← Original term, high weight
  "RBAC": 1.4,                  ← Learned expansion
  "authorization": 1.2,         ← Learned expansion  
  "admission": 0.9,             ← Learned expansion
  "controller": 0.7,            ← Learned expansion
  "kubernetes": 0.5,            ← Learned expansion
  ... (most terms are 0)
}
```

**Key insight**: SPLADE learns to **expand** terms with semantically related words while **preserving** exact match signals. The original terms get high weights, and related terms get lower weights.

### Why SPLADE is Better Than BM25 + Dense Hybrid

| Approach | Exact Match | Semantic | Single Index | Learned Weights |
|----------|-------------|----------|--------------|-----------------|
| BM25 | Yes | No | Yes | No (TF-IDF) |
| Dense | No | Yes | Yes | Yes |
| BM25 + Dense (RRF) | Partial | Yes | No (two indexes) | Partial |
| **SPLADE** | **Yes** | **Yes** | **Yes** | **Yes** |

SPLADE advantages:
1. **Single inverted index** - Same infrastructure as BM25
2. **Learned term weights** - Better than TF-IDF for domain-specific terms
3. **Query/Document expansion** - Automatically adds related terms
4. **Exact match preservation** - Original terms get highest weights

### SPLADE Variants

| Model | Description | BEIR NDCG@10 |
|-------|-------------|--------------|
| SPLADE-max | Original, document expansion only | ~47% |
| SPLADE-doc | Symmetric expansion | ~48% |
| SPLADE++ | Improved training, ensemble distillation | ~50% |
| **SPLADE-v3** | Latest, best single model | **~49%** |
| Mistral-SPLADE | LLM-based backbone | ~52% |

For comparison: BM25 achieves ~41% on BEIR.

### Architecture Overview

```
Query: "SubjectAccessReview"
         │
         ▼
┌─────────────────────┐
│   BERT Encoder      │
│   (12 layers)       │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   MLM Head          │
│   (vocab logits)    │
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   ReLU + Log        │
│   (sparsification)  │
└─────────────────────┘
         │
         ▼
Sparse Vector: {term: weight, ...}
         │
         ▼
┌─────────────────────┐
│   Inverted Index    │
│   (same as BM25)    │
└─────────────────────┘
```

### Training Objectives

SPLADE is trained with:
1. **Contrastive loss** - Similar to dense retrieval training
2. **FLOPS regularization** - Encourages sparsity (fewer non-zero terms)
3. **Distillation** - Learning from cross-encoder teachers

The FLOPS regularization is key - it prevents SPLADE from activating too many terms, keeping the index efficient.

### Inference Characteristics

| Aspect | BM25 | SPLADE | Dense |
|--------|------|--------|-------|
| Query encoding | None | ~20-50ms (BERT) | ~10-20ms |
| Document encoding | None (at index) | ~20-50ms (at index) | ~10-20ms (at index) |
| Index type | Inverted | Inverted | HNSW/IVF |
| Index size | 1x | 2-3x | 10-50x |
| Retrieval | Fast | Fast | Fast |

**Key tradeoff**: SPLADE requires BERT inference for queries (20-50ms), but uses the same fast inverted index retrieval as BM25.

---

## Why SPLADE for This Project

### Current Problem

Our BM25 implementation uses naive tokenization:
```
"SubjectAccessReview" → ["subjectaccessreview"]  (single token)
```

This makes it impossible to match queries like "subject access review" or "access review".

### Why Not Just Fix Tokenization?

CamelCase splitting helps but doesn't solve:
- Acronym expansion (RBAC → role-based access control)
- Synonym matching (k8s → kubernetes)
- Semantic term associations (webhook → admission controller)

### What SPLADE Provides

1. **Automatic term expansion** - Learns domain-relevant expansions from training data
2. **Weighted exact matching** - Important terms get higher weights than expansions
3. **Compatible infrastructure** - Works with existing inverted index approach
4. **No per-query LLM calls** - BERT inference is fast and local

---

## Expected Outcomes

Based on BEIR benchmarks and similar technical documentation tasks:

| Metric | Current (BM25) | Expected (SPLADE) | Improvement |
|--------|----------------|-------------------|-------------|
| Informed MRR | ~0.62 | ~0.72-0.78 | +15-25% |
| Needle MRR | ~0.84 | ~0.85-0.90 | +1-7% |
| Realistic MRR | ~0.20 | ~0.22-0.26 | +10-30% |

### Risk Factors

1. **Domain mismatch** - Pre-trained SPLADE is trained on MS MARCO (web search), not technical docs
2. **Latency increase** - Query encoding adds 20-50ms
3. **Index size** - 2-3x larger than current BM25 index
4. **Kubernetes-specific terms** - May not be in SPLADE's training data

---

## References

### Papers
- **SPLADE** (2021): "SPLADE: Sparse Lexical and Expansion Model for First Stage Ranking" - Formal et al.
- **SPLADE v2** (2022): "SPLADE v2: Sparse Lexical and Expansion Model for Information Retrieval" - Formal et al.
- **SPLADE++** (2022): "From Distillation to Hard Negative Sampling: Making Sparse Neural IR Models More Effective" - Formal et al.
- **SPLADE-v3** (2024): "SPLADE-v3: New baselines for SPLADE" - Lassance et al. (arXiv:2403.06789)
- **Mistral-SPLADE** (2024): "LLMs for better Learned Sparse Retrieval" - Doshi et al. (arXiv:2408.11119)

### Implementations
- **Official**: `naver/splade` (GitHub)
- **HuggingFace**: `naver/splade-cocondenser-ensembledistil`
- **PyTerrier**: `pyterrier-splade` integration
- **Lightweight**: `light-splade` (pip package)

### Benchmarks
- **BEIR**: Benchmarking IR - 18 diverse retrieval datasets
- **MS MARCO**: Primary training dataset for SPLADE
- **TREC**: Traditional IR evaluation

---

## Setup & Usage

### Prerequisites

```bash
cd poc/splade_benchmark
uv sync
```

### Run Full Benchmark

```bash
.venv/bin/python run_all.py
```

This will:
1. Build BM25 baseline (if needed)
2. Build SPLADE index (~75 min on CPU)
3. Run SPLADE-only evaluation
4. Run hybrid SPLADE+Semantic evaluation
5. Run technical term analysis
6. Generate visualizations

### Run Individual Steps

```bash
# BM25 baseline only
.venv/bin/python baseline_bm25.py

# Build SPLADE index only
.venv/bin/python splade_index.py --batch-size 16

# SPLADE-only evaluation (requires index)
.venv/bin/python splade_benchmark.py

# Hybrid evaluation (requires index)
.venv/bin/python hybrid_splade.py

# Technical term analysis
.venv/bin/python technical_analysis.py --include-informed

# Visualizations
.venv/bin/python visualize.py
```

### Skip Index Building

If index already exists:
```bash
.venv/bin/python run_all.py --skip-index
```

---

## Files

| File | Purpose |
|------|---------|
| `splade_encoder.py` | SPLADE model wrapper |
| `splade_index.py` | Inverted index builder |
| `baseline_bm25.py` | BM25 baseline extraction |
| `splade_benchmark.py` | SPLADE-only evaluation |
| `hybrid_splade.py` | SPLADE + Semantic hybrid |
| `technical_analysis.py` | Term expansion analysis |
| `visualize.py` | Generate comparison charts |
| `run_all.py` | Orchestration script |
| `EVALUATION_CRITERIA.md` | Detailed evaluation criteria |
| `TODO.md` | Implementation checklist |
| `RESULTS.md` | Results template |

---

*Created: 2026-02-22*
*POC Priority: 2 (after Convex Fusion)*
*Status: COMPLETE - All benchmarks executed, results documented*
