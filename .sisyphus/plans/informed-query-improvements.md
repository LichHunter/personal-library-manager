# Informed Query Improvements: Research Findings & POC Plan

**Created**: 2026-02-22  
**Updated**: 2026-02-22  
**Goal**: Improve retrieval performance on technical terminology queries (informed benchmark)  
**Constraint**: No LLM-based solutions (no HyDE, no inference-time Claude calls)

---

## POC Status Summary

| # | POC | Status | Result | Key Finding |
|---|-----|--------|--------|-------------|
| 1 | Convex Fusion | **COMPLETE** | PARTIAL | +7.1% informed, but -12.4% needle regression |
| 2 | SPLADE | Pending | — | — |
| 3 | BM25F | Pending | — | — |
| 4 | RM3 Query Expansion | Pending | — | — |
| 5 | N-gram Phrase Indexing | Pending | — | — |
| 6 | E5-large-instruct | Pending | — | — |
| 7 | Term Graph | Pending | — | Investigation only |

### POC 1 Results: Convex Fusion (PARTIAL)

**Verdict**: Do NOT replace RRF with convex fusion.

| Benchmark | RRF | Convex (Best) | Change |
|-----------|-----|---------------|--------|
| Informed | 0.636 | 0.681 | +7.1% |
| Needle | 0.842 | 0.738 | **-12.4%** |
| Realistic | 0.262 | 0.190 | -3.0% |

**Key Insights**:
- Best config: Z-score normalization, alpha=0.36
- Improvement not statistically significant (p=0.385)
- RRF's adaptive per-query parameters outperform fixed alpha
- Consider adaptive convex fusion as future work (mirror RRF's expansion-triggered params)

**Full results**: `poc/convex_fusion_benchmark/RESULTS.md`

---

## Current State

### Benchmark Results (Before Improvements)

| Benchmark | Naive RAG | PLM Baseline | PLM + Reranker |
|-----------|-----------|--------------|----------------|
| Needle MRR | 0.713 | 0.842 | 0.817 |
| **Informed MRR** | **0.633** | **0.621** | **0.690** |
| Realistic MRR | 0.192 | 0.196 | 0.196 |

**Problem**: PLM loses to naive RAG on informed queries (-1.9% MRR).

### Root Cause Analysis

1. **BM25 Tokenization** (`bm25.py:48`): Uses `text.lower().split()` - CamelCase terms like `SubjectAccessReview` become single tokens, impossible to match with space-separated queries.

2. **Content Enrichment**: Prepending `"keywords | entities\n\n"` to content before embedding causes **-4.7% MRR degradation** on informed queries.

3. **RRF Weights When Expanded**: BM25 gets 3.0 weight vs semantic 0.3 - but BM25 is broken for technical terms, so this amplifies failures.

4. **Limited Query Expansion**: Only 14 hardcoded rules in `expander.py`, missing most Kubernetes API terms.

---

## Research Findings

### 1. Learned Sparse Retrieval (SPLADE)

**What**: Neural model that expands documents/queries into sparse weighted term vectors using BERT's MLM head.

**How it works**:
```
Input: "SubjectAccessReview webhook"
SPLADE output: {
  "SubjectAccessReview": 2.8,
  "webhook": 2.1,
  "RBAC": 1.4,          # learned expansion
  "authorization": 1.2,  # learned expansion
  "admission": 0.9       # learned expansion
}
```

**Expected improvement**: +12-16% on technical queries

**Benchmark (BEIR)**:
- SPLADE-v3: 49.2% NDCG@10
- BM25: 41.2% NDCG@10
- Improvement: +19%

**Implementation**:
```python
from transformers import AutoTokenizer, AutoModelForMaskedLM
model = AutoModelForMaskedLM.from_pretrained("naver/splade-cocondenser-ensembledistil")
# Compatible with inverted index (same infrastructure as BM25)
```

**Libraries**: 
- `naver/splade` (official)
- `light-splade` (lightweight wrapper)
- PyTerrier SPLADE integration

---

### 2. BM25F (Field-Weighted BM25)

**What**: BM25 variant that treats documents as multiple weighted fields instead of flat text.

**How it works**:
```python
field_weights = {
    'title': 3.0,       # Matches in title are 3x more important
    'heading': 2.0,     # Section headings
    'code': 1.5,        # Code blocks
    'api_path': 2.5,    # API paths (/api/v1/...)
    'body': 1.0         # Regular content
}

# Aggregate weighted term frequency across fields
weighted_tf = sum(weight[field] * tf[field] / norm[field] for field in doc)
```

**Expected improvement**: +20-30% on queries matching titles/headings

---

### 3. Convex Combination Fusion

**What**: Replace RRF with score-based fusion using tunable alpha.

**Current RRF**:
```python
score(d) = sum(1 / (k + rank(d)) for retriever in retrievers)
# Only uses ranks, ignores actual scores
```

**Convex Combination**:
```python
def convex_fusion(bm25_scores, semantic_scores, alpha=0.4):
    # Normalize to [0,1]
    bm25_norm = (bm25_scores - bm25_scores.min()) / (bm25_scores.ptp() + 1e-8)
    sem_norm = (semantic_scores - semantic_scores.min()) / (semantic_scores.ptp() + 1e-8)
    return alpha * bm25_norm + (1 - alpha) * sem_norm
```

**Expected improvement**: +10-15% MRR over RRF

**Key insight**: Alpha should be query-dependent. Technical queries need higher semantic weight (alpha < 0.5).

---

### 4. N-gram Phrase Matching

**What**: Index bigrams and trigrams alongside unigrams to capture multi-word technical terms.

**How it works**:
```python
def generate_ngrams(tokens, n=2):
    ngrams = []
    for i in range(len(tokens) - n + 1):
        ngram = '_'.join(tokens[i:i+n])
        ngrams.append(ngram)
    return ngrams

# Example
tokens = ['webhook', 'token', 'authenticator']
bigrams = ['webhook_token', 'token_authenticator']
trigrams = ['webhook_token_authenticator']

# Index all together
all_tokens = tokens + bigrams + trigrams
```

**Expected improvement**: +15-25% recall for multi-word technical terms

---

### 5. RM3 Pseudo-Relevance Feedback

**What**: Classical IR technique that expands queries using terms from top-retrieved documents.

**Classic RM3 Flow**:
```
1. Query: "SubjectAccessReview"
   ↓
2. Initial BM25 retrieval → top 10 docs
   ↓
3. Extract frequent terms from top 10:
   ["RBAC", "authorization", "kubectl", "auth", "webhook"]
   ↓
4. Expanded query: "SubjectAccessReview RBAC authorization kubectl auth"
   ↓
5. Re-run BM25 with expanded query
```

**Metadata-Based RM3 Variant** (use existing extraction):
```
1. Query: "SubjectAccessReview"
   ↓
2. Initial retrieval → top 10 docs
   ↓
3. Use existing metadata (entities, keywords) from those docs
   ↓
4. Aggregate and weight by frequency/relevance
   ↓
5. Expanded query
```

**Expected improvement**: +5-8% nDCG

**Implementation**: PyTerrier `pt.rewrite.RM3()` for classic, custom for metadata-based

---

### 6. E5-large-instruct Embedding Model

**What**: Instruction-tuned embedding model that uses task description to improve query understanding.

**Key difference from BGE**:

| Aspect | BGE-base-en-v1.5 (current) | E5-large-instruct |
|--------|---------------------------|-------------------|
| Parameters | 109M | 560M |
| Dimensions | 768 | 1024 |
| Query format | Raw text | Instruction + Query |
| Document format | Raw text | Raw text (NO instruction) |

**Asymmetric usage** (documents don't need instructions):
```python
# Documents: NO instruction (index once)
doc_embeddings = model.encode(documents)

# Queries: WITH instruction (every search)
query = f"Instruct: Retrieve Kubernetes API documentation\nQuery: {user_query}"
query_embedding = model.encode(query)
```

**Why instructions help technical terms**:
- Disambiguates: "review" in "SubjectAccessReview" ≠ "code review"
- Provides domain context without needing domain-specific fine-tuning

**Expected improvement**: +15-20% on technical retrieval

**Trade-offs**:
- 5x larger model (560M vs 109M)
- 1.3x slower inference
- Requires instruction engineering

---

### 7. Term Graph / Co-occurrence Matrix

**Approaches** (from simpler to complex):

| Approach | How | Maintenance | Quality |
|----------|-----|-------------|---------|
| **Synonym Dict** | Manual: `k8s ↔ kubernetes` | Manual | High precision |
| **Co-occurrence** | Mine from corpus: terms that appear together | Auto | Medium |
| **PMI-weighted** | Co-occurrence + statistical significance | Auto | Higher |
| **Entity Graph** | Use extracted entities as nodes | Semi-auto | High |

**Co-occurrence Implementation**:
```python
from collections import defaultdict
import math

class CooccurrenceMatrix:
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.cooccurrence = defaultdict(lambda: defaultdict(int))
        self.term_freq = defaultdict(int)
    
    def build(self, documents):
        for doc in documents:
            tokens = doc.lower().split()
            for i, term in enumerate(tokens):
                self.term_freq[term] += 1
                window = tokens[max(0, i-self.window_size):i+self.window_size+1]
                for other in window:
                    if other != term:
                        self.cooccurrence[term][other] += 1
    
    def get_related(self, term, top_k=10):
        """Get top-k related terms by PMI."""
        if term not in self.cooccurrence:
            return []
        
        total = sum(self.term_freq.values())
        results = []
        for other, count in self.cooccurrence[term].items():
            pmi = math.log2(
                (count * total) / 
                (self.term_freq[term] * self.term_freq[other] + 1e-10)
            )
            results.append((other, pmi))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
```

**Investigation needed**:
- What window size works best?
- Document-level vs chunk-level co-occurrence?
- How to combine with existing entity/keyword extraction?
- Can we use extracted entities as nodes instead of raw tokens?

---

## POC Specifications

### POC 1: Convex Fusion vs RRF

**Objective**: Compare score-based fusion against current RRF.

**Setup**:
1. Use informed benchmark queries (25 queries)
2. Get raw BM25 scores and semantic scores (before fusion)
3. Test alpha values: 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8

**Metrics**: MRR@10, Recall@10, Precision@10

**Implementation**:
```python
def convex_fusion(bm25_results, semantic_results, alpha):
    bm25_scores = normalize(bm25_results)
    sem_scores = normalize(semantic_results)
    
    combined = {}
    for doc_id in set(bm25_scores.keys()) | set(sem_scores.keys()):
        bm25_s = bm25_scores.get(doc_id, 0)
        sem_s = sem_scores.get(doc_id, 0)
        combined[doc_id] = alpha * bm25_s + (1 - alpha) * sem_s
    
    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
```

**Success criteria**: >5% MRR improvement over RRF at any alpha value.

---

### POC 2: BM25F Field Weighting

**Objective**: Test if field-weighted BM25 improves technical term matching.

**Setup**:
1. Parse chunks to extract fields: title, headings, code, body
2. Implement BM25F with configurable weights
3. Test weight configurations:
   - Baseline: all fields weight=1.0
   - Title-heavy: title=5.0, heading=2.0, code=1.5, body=1.0
   - Code-heavy: title=2.0, heading=1.5, code=3.0, body=1.0

**Metrics**: MRR@10 on informed queries, MRR@10 on needle (no regression)

**Success criteria**: >10% MRR improvement on informed queries.

---

### POC 3: SPLADE Integration

**Objective**: Test learned sparse retrieval as BM25 replacement.

**Setup**:
1. Use pre-trained: `naver/splade-cocondenser-ensembledistil`
2. Generate sparse vectors for all chunks
3. Build inverted index from sparse vectors
4. Compare against current BM25

**Metrics**: MRR@10 all benchmarks, latency, index size

**Implementation**:
```python
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

class SPLADEEncoder:
    def __init__(self, model_name="naver/splade-cocondenser-ensembledistil"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
    
    def encode(self, text):
        tokens = self.tokenizer(text, return_tensors="pt", truncation=True)
        with torch.no_grad():
            output = self.model(**tokens)
        
        logits = torch.max(
            torch.log(1 + torch.relu(output.logits)), 
            dim=1
        ).values[0]
        
        non_zero = torch.nonzero(logits).squeeze()
        sparse = {idx.item(): logits[idx].item() for idx in non_zero}
        return sparse
```

**Success criteria**: >10% MRR improvement AND latency < 100ms per query.

---

### POC 4: N-gram Phrase Indexing

**Objective**: Test bigram/trigram indexing for multi-word technical terms.

**Setup**:
1. Modify tokenization to generate n-grams (n=2,3)
2. Index unigrams + bigrams + trigrams
3. Test with phrase boost multiplier (2x, 3x, 5x)

**Test queries**:
- "webhook token authenticator"
- "horizontal pod autoscaler"
- "service account tokens"
- "network policy ingress"

**Metrics**: Recall@10, index size, query latency

**Success criteria**: >15% recall improvement on multi-word queries.

---

### POC 5: RM3 Query Expansion

**Objective**: Test pseudo-relevance feedback with two variants.

**Variants**:
1. **Classic RM3**: Extract terms from top-k retrieved documents
2. **Metadata RM3**: Use existing entities/keywords from top-k docs

**Parameters to test**:
- `fb_docs`: 5, 10, 20 (number of feedback documents)
- `fb_terms`: 10, 20, 50 (number of expansion terms)
- `original_weight`: 0.5, 0.7, 0.9 (weight of original query)

**Metrics**: MRR@10 improvement, query latency

**Implementation (Metadata RM3)**:
```python
def metadata_rm3_expand(query, top_docs, storage, fb_terms=20, original_weight=0.7):
    term_scores = defaultdict(float)
    
    for doc in top_docs:
        metadata = storage.get_chunk_metadata(doc['chunk_id'])
        entities = metadata.get('entities', [])
        keywords = metadata.get('keywords', [])
        
        rank_weight = 1.0 / (doc['rank'] + 1)
        
        for entity in entities:
            term_scores[entity.lower()] += rank_weight
        for keyword in keywords:
            term_scores[keyword.lower()] += rank_weight * 0.5
    
    sorted_terms = sorted(term_scores.items(), key=lambda x: x[1], reverse=True)
    expansion_terms = [t[0] for t in sorted_terms[:fb_terms]]
    
    expanded = f"{query} {' '.join(expansion_terms)}"
    return expanded
```

**Success criteria**: >5% MRR improvement with <100ms additional latency.

---

### POC 6: E5-large-instruct Embedding

**Objective**: Test instruction-tuned embeddings as BGE replacement.

**Setup**:
1. Load E5-large-instruct model
2. Re-embed all chunks (no instruction for docs)
3. Test instruction templates for queries

**Instruction templates to test**:
```python
templates = [
    "Instruct: Retrieve Kubernetes documentation\nQuery: {query}",
    "Instruct: Find technical documentation about\nQuery: {query}",
    "Instruct: Search for API reference\nQuery: {query}",
    "Instruct: Find relevant documentation\nQuery: {query}",
]
```

**Metrics**: MRR@10 all benchmarks, embedding latency, memory usage

**Success criteria**: >10% MRR improvement on informed queries.

---

### POC 7: Term Graph Investigation

**Objective**: Explore approaches for building term relationships without per-chunk LLM calls.

**Investigation areas**:

1. **Co-occurrence mining**:
   - Window-based (terms within N tokens)
   - Document-level (terms in same doc)
   - Chunk-level (terms in same chunk)
   - Compare PMI vs raw frequency

2. **Using existing extraction**:
   - Build graph from extracted entities
   - Use entity types as edge attributes
   - Cluster entities by co-occurrence

3. **Hybrid approaches**:
   - Core dictionary (curated, high-precision)
   - Statistical expansion (mined, high-recall)

**Output**: Technical report with:
- Recommended approach
- Implementation complexity estimate
- Expected coverage of technical terms
- Sample graph for Kubernetes domain

---

## Dismissed Approaches

### CamelCase Regex Splitting

**Reason**: User feedback - "regex is too complicated to update frequently"

**Note**: Valid quick fix, but user prefers more robust solutions.

---

## Implementation Priority

| Priority | POC | Expected Impact | Effort | Status |
|----------|-----|-----------------|--------|--------|
| ~~1~~ | ~~Convex Fusion~~ | ~~+10-15% MRR~~ | ~~1-2 days~~ | **DONE (PARTIAL)** |
| **2** | **SPLADE** | +12-16% MRR | 1-2 weeks | **NEXT** |
| 3 | BM25F | +20-30% MRR | 3-5 days | Pending |
| 4 | RM3 (both variants) | +5-8% MRR | 2-3 days | Pending |
| 5 | N-gram | +15-25% recall | 2-3 days | Pending |
| 6 | E5-instruct | +15-20% MRR | 1-2 days | Pending |
| 7 | Term Graph | Investigation | 1 week | Pending |

### Recommended Next POC: SPLADE

Based on convex fusion results, the core issue is **BM25's broken tokenization** for technical terms. SPLADE addresses this directly by learning semantic term expansions. Unlike convex fusion (which just reweights broken signals), SPLADE replaces the broken BM25 with learned sparse retrieval.

---

## Success Metrics

**Target**: Improve informed query MRR from 0.621 to >0.75 (>20% improvement)

**Measurement**:
- Use existing informed benchmark (25 queries)
- Compare against current PLM baseline
- Track both MRR and latency

**Constraint**: All solutions must work without inference-time LLM calls.

---

## References

### Papers
- SPLADE-v3: arXiv:2403.06789
- ColBERT-v2: arXiv:2112.01488
- E5-instruct: arXiv:2212.03533
- RM3: Lavrenko & Croft, "Relevance-Based Language Models" (SIGIR 2001)

### Libraries
- SPLADE: `naver/splade`, `light-splade`
- BM25: `bm25s`, `rank_bm25`
- PyTerrier: `pyterrier` (RM3, Bo1, etc.)
- E5: `intfloat/multilingual-e5-large-instruct`
