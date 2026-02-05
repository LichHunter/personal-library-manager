# Retrieval Improvement Research Notes

> Working document for deep research into query preparation and document filtering techniques.
> Focus: Real-world tested solutions, not theoretical corporate BS.

## Research Status

Started: 2026-01-30
Status: IN PROGRESS

## Problem Statement

- **Vocabulary gap**: 36% (realistic) vs 80% (informed) = 44pp difference
- **Root cause**: Users use natural language; docs use technical Kubernetes terminology
- **Rejected**: HyDE (brute-force hypothetical answer generation)

## Two Focus Areas

### 1. Query Preparation
- Reformatting user questions to be more concrete
- Adding/guessing technical terms
- Intent understanding

### 2. Document Filtering
- Pre-filtering using metadata before vector search
- Hierarchical retrieval (summaries → chunks)
- Routing to document subsets

---

## Research Threads

### Thread 1: Query Understanding (not generation)
- [ ] Intent classification
- [ ] Entity extraction for domain terms
- [ ] Query decomposition
- [ ] Faceted search query expansion

### Thread 2: Document Pre-filtering
- [ ] Metadata-based routing
- [ ] Hierarchical indexing (doc summaries → sections → chunks)
- [ ] Document clustering and cluster-first retrieval
- [ ] Learned sparse retrieval (SPLADE, etc.)

### Thread 3: Production Systems
- [ ] How does Algolia handle this?
- [ ] Elasticsearch Learning to Rank
- [ ] Google's production search techniques
- [ ] Perplexity/You.com architecture

### Thread 4: Adjacent Fields
- [ ] Traditional IR techniques (query expansion, relevance feedback)
- [ ] Recommendation systems (candidate generation → ranking)
- [ ] E-commerce search (facets, filters, query understanding)

### Thread 5: Academic but Tested
- [ ] ColBERT and late interaction models
- [ ] SPLADE (learned sparse representations)
- [ ] Query2Query (finding similar past queries)
- [ ] Conversational search patterns

---

## Findings

### Query Preparation Techniques (NO LLM per query)

#### 1. Intent Classification
| Technique | Latency | Accuracy | Best For |
|-----------|---------|----------|----------|
| FastText | 0.5-2ms | 85-92% | High-throughput, 10-100 classes |
| Embedding + Cosine | 2-3ms | 80-88% | Low latency, prototype-based |
| TinyBERT | 5-15ms | 88-94% | Better accuracy needed |
| Rule-based (spaCy Matcher) | <0.1ms | 95%+ | Structured/command queries |

**Key insight**: Embedding + cosine with precomputed prototypes is sweet spot for our use case.

#### 2. Entity Recognition for Domain Terms
| Technique | Latency | Accuracy | Best For |
|-----------|---------|----------|----------|
| spaCy EntityRuler | 1-5ms | 98%+ | Known vocabulary (K8s resources) |
| spaCy Custom NER | 10-50ms | 85-95% | Domain-specific entities |
| BERT-NER | 15-40ms | 90-96% | Ambiguous entities |

**Key insight**: EntityRuler for known K8s terms + minimal NER for ambiguous cases.

#### 3. Query Expansion (Classic IR)
| Technique | Latency | Impact | Best For |
|-----------|---------|--------|----------|
| Pseudo-Relevance Feedback (RM3) | 50-200ms | +10-24% NDCG | Offline acceptable |
| Co-occurrence lookup | 1-10ms | +8-12% | Domain corpus available |
| WordNet/ConceptNet | 5-20ms | +5-10% | Short queries |

**Key insight**: Co-occurrence is best latency/impact trade-off.

#### 4. Query-to-Query Matching
| Technique | Latency | Impact |
|-----------|---------|--------|
| Click-through logs | 1-5ms | +15-25% for popular queries |
| Embedding + ANN | 5-15ms | +10-20% |

### Document Filtering Techniques

#### 1. Hierarchical Indexing (RAPTOR)
- **How**: Recursively cluster + summarize chunks → tree structure
- **Benchmark**: +20-30% on multi-hop reasoning (QuALITY dataset)
- **Implementation**: LlamaIndex RaptorPack, NirDiamant/RAG_Techniques
- **Tradeoff**: Expensive indexing, good for long documents

#### 2. Metadata-Based Routing
- **Pinecone namespaces**: Up to 100k per index, queries within namespace are faster
- **Weaviate pre-filtering**: ACORN strategy for low-correlation filters
- **Pattern**: Filter → Vector Search → Re-rank

#### 3. Faceted Search
- Recommended facets for docs: type, version, language, topic, difficulty
- **Key benefit**: Filtering 1M→10K before vector search = 100x speedup

#### 4. IVF Clustering (Cluster-first retrieval)
- **FAISS IVF-PQ**: sqrt(n) clusters, search top-k clusters
- **Performance**: 90%+ recall with proper nprobe tuning, 4-5x compression

### Production System Patterns

#### 1. Two-Stage Retrieval (YouTube/Netflix pattern)
```
Stage 1: Bi-encoder + ANN → 500 candidates (<10ms)
Stage 2: Cross-encoder rerank → top 10 (~50-100ms)
Total: ~100ms for high-quality results
```

#### 2. Hybrid Search (2026 Standard)
```
Query → [BM25] → Results₁
     → [Dense] → Results₂  
     → RRF Fusion (k=60) → Combined
     → Cross-encoder rerank → Final
```
**Evidence**: +30-40% better than either alone (Elastic, Pinecone, Haystack)

#### 3. Production Query Understanding (Algolia)
- **Query Categorization**: ML model predicts category → auto-filter/boost
- **Result**: +22% conversion lift (Swiss retailer), +15% revenue (UK grocery)

#### 4. Phased Ranking (Vespa)
- Phase 1: Cheap scoring on ALL matches (BM25 + freshness)
- Phase 2: Expensive model on top-K per node (XGBoost)
- Phase 3: Global rerank on merged results (ONNX model)

### What Actually Works (Benchmarked)

#### Tier 1: High Evidence, Easy Implementation
1. **Hybrid Search (BM25 + Dense + RRF)** - +15-20% retrieval
2. **Cross-encoder reranking** - +10-15% additional
3. **Vocabulary expansion** (14→80+ terms) - +15-20% for vocabulary gap

#### Tier 2: High Evidence, Moderate Implementation
4. **ColBERT reranking** (via RAGatouille) - +5-10%
5. **SPLADE-v3** (learned sparse) - +10-15% vs BM25
6. **Doc2Query** (offline expansion) - +3-15%

#### Tier 3: Emerging/Complex
7. **RAPTOR** (hierarchical) - +20-30% on multi-hop
8. **LLM reranking** (expensive) - marginal over cross-encoder
9. **BGE-M3** (multi-vector) - SOTA but complex

---

## Key Questions ANSWERED

### 1. What do production search systems do for query understanding?
**Algolia**: ML-based query categorization → +22% CVR lift
**Elasticsearch**: Analyzers + BM25 tuning + LTR (XGBoost/LambdaMART)
**Vespa**: Phased ranking (cheap→expensive) + WAND pruning (97.5% reduction)

### 2. How do they handle vocabulary mismatch without LLM-per-query?
- **SetFit**: Few-shot classification, 8 examples/class, 85-90% accuracy, 4-10ms
- **Semantic caching**: Cache query→result mappings, 67% hit rate, 15-20ms
- **Co-occurrence expansion**: Pre-computed term co-occurrence, 1-10ms, +8-12%
- **RaFe**: Train small rewriter with reranker feedback (no annotations)

### 3. What pre-filtering techniques have measurable impact?
- **Metadata filtering**: doc_type, version, topic → 10-100x search space reduction
- **Intent → doc type routing**: troubleshooting→tasks/reference, how_to→tasks
- **Faceted search**: Pre-filter before vector search

### 4. Can we learn from e-commerce/recommendation systems?
**YES - Two-stage retrieval is the key pattern:**
- Stage 1: Bi-encoder + ANN → 500 candidates (<10ms)
- Stage 2: Cross-encoder rerank → top 10 (~50-100ms)

---

## SYNTHESIS: Recommended Architecture for Our Problem

### The Vocabulary Gap Problem
- **Realistic queries**: 36% Hit@5
- **Informed queries**: 80% Hit@5
- **Gap**: 44 percentage points (vocabulary mismatch)

### Solution Architecture (Multi-Tier, <100ms total)

```
User Query
     │
     ▼
┌────────────────────────────────────────┐
│ Tier 1: Rule-Based Extraction (<2ms)   │
│ - spaCy EntityRuler for K8s resources  │
│ - Intent patterns (troubleshooting,    │
│   how_to, what_is)                     │
│ - Extract: {resource, action, intent}  │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ Tier 2: Vocabulary Expansion (<5ms)    │
│ - DOMAIN_EXPANSIONS (80+ terms)        │
│ - Co-occurrence lookup                 │
│ - Synonym injection                    │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ Tier 3: Document Pre-filter (<5ms)     │
│ - Intent → doc_type routing            │
│   troubleshooting → [tasks, reference] │
│   how_to → [tasks]                     │
│   what_is → [concepts]                 │
│ - Reduce search space by 3-5x          │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ Tier 4: Hybrid Retrieval (~30ms)       │
│ - BM25 on filtered docs                │
│ - Dense retrieval on filtered docs     │
│ - RRF fusion (k=60)                    │
│ - Optional: VectorPRF refinement       │
└────────────────────────────────────────┘
     │
     ▼
┌────────────────────────────────────────┐
│ Tier 5: Cross-encoder Rerank (~50ms)   │
│ - ms-marco-MiniLM-L-6-v2               │
│ - Top 50 → Top 5                       │
└────────────────────────────────────────┘
     │
     ▼
Final Results
```

### Expected Impact by Component

| Component | Latency | Expected Impact | Difficulty |
|-----------|---------|-----------------|------------|
| Expand DOMAIN_EXPANSIONS (14→80+) | 0ms | +15-20% (36→55%) | Easy |
| Intent → doc_type routing | 2ms | +5-10% | Medium |
| Cross-encoder reranking | 50ms | +5-10% | Easy |
| VectorPRF (Rocchio-style) | 20ms | +5-10% | Medium |
| **Combined** | ~80ms | **+30-40% (36→70%)** | - |

### Implementation Priority

**Phase 1: Vocabulary Expansion (1-2 days)**
- Expand DOMAIN_EXPANSIONS from 14 to 80+ terms
- Focus on Kubernetes-specific mappings:
  - Problem symptoms → Technical components
  - User actions → kubectl commands
  - Natural language → API resources
- Expected: 36% → 55% Hit@5

**Phase 2: Intent-based Routing (2-3 days)**
- Add intent classification (rule-based first)
- Route to document type subsets
- Implement pre-filtering before retrieval
- Expected: +5-10% additional

**Phase 3: Cross-encoder Reranking (1 day)**
- Add ms-marco-MiniLM-L-6-v2 reranker
- Apply to top-50 results
- Expected: +5-10% additional

**Phase 4: Advanced (optional, 1 week)**
- VectorPRF (Rocchio-style query refinement)
- SetFit-based intent classifier (replace rules)
- Semantic caching for repeated queries

### Key Files to Modify

1. `components/query_expander.py` - Add 80+ K8s terms
2. `components/intent_classifier.py` - NEW: Intent detection
3. `modular_enriched_hybrid.py` - Add pre-filtering + reranking
4. `types.py` - Add IntentClassification type

### What NOT to Do

1. ❌ HyDE - Brute-force hypothetical generation
2. ❌ LLM-per-query rewriting - Too slow, marginal benefit
3. ❌ Over-engineering - Start simple, measure, iterate
4. ❌ RAPTOR hierarchical indexing - Overkill for our corpus size

---

## Key Research Papers (Validated)

1. **RaFe** (EMNLP 2024) - Train rewriter with reranker feedback
2. **ColBERT-PRF** (ICTIR 2021) - Neural PRF, +26% MAP
3. **SetFit** (HuggingFace) - Few-shot classification, 85%+ accuracy
4. **LLM-Assisted PRF** (ECIR 2026) - LLM filters before RM3

## Key Repositories

1. **NirDiamant/RAG_Techniques** - 24.5k stars, comprehensive
2. **huggingface/setfit** - Few-shot classification
3. **stanford-futuredata/ColBERT** - 3.8k stars, late interaction
4. **naver/splade** - 970 stars, learned sparse

---

## NEXT STEPS

1. ✅ Research complete
2. → Create work plan for implementation
3. → Start with vocabulary expansion
4. → Benchmark after each phase
