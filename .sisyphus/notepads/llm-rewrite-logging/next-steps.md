# Next Steps: Improving Retrieval Performance

**Date**: 2026-01-27  
**Current Performance**: 70% on adversarial questions (full corpus)  
**Goal**: 85%+ on adversarial questions

---

## Summary of Findings

### ✅ What's Working

1. **LLM Query Rewrite (Claude Haiku)** - 100% of rewrites are good-to-excellent
   - Expands acronyms (k8s → kubernetes, GA → general availability)
   - Transforms casual → technical language
   - Preserves intent in all cases
   - **Verdict**: Keep unchanged, do NOT modify

2. **Comparison Questions** - 80% pass rate on full corpus
   - Semantic understanding excels at comparing concepts
   - Only 1 failure due to retrieval noise

3. **Version Questions** - 80% pass rate on full corpus
   - Most version lookups work well
   - Only 1 consistent failure (frontmatter metadata)

### ⚠️ What's Struggling

1. **Vocabulary Questions** - 60% pass rate on full corpus (-40% from small corpus)
   - Extreme vocabulary mismatch: "IPC latency" vs "inter-NUMA overhead"
   - Generic docs outrank specific docs with more corpus noise
   - **Root cause**: Vocabulary gap + retrieval noise

2. **Negation Questions** - 60% pass rate on full corpus (-20% from small corpus)
   - "Why can't", "what's wrong" questions struggle
   - Semantic search doesn't capture "absence" or "limitation" well
   - **Root cause**: Semantic gap in understanding constraints

3. **Retrieval at Scale** - 20% drop when corpus expanded 7x
   - More documents = more noise
   - Generic docs push specific docs out of top-5
   - **Root cause**: Ranking/reranking not optimized for large corpora

---

## Prioritized Recommendations

### 🔴 HIGH PRIORITY: Improve Reranking for Large Corpora

**Problem**: 70% pass rate on full corpus (vs 90% on small corpus)

**Impact**: Could improve overall performance from 70% → 85%+

**Options**:

1. **ColBERT Reranking** (RECOMMENDED)
   - Late interaction model for better semantic matching
   - Proven to work well at scale
   - Implementation: Use `colbert-ai/colbertv2.0` model
   - Cost: ~100ms per query for reranking top-20 → top-5

2. **Cross-Encoder Reranking**
   - Rerank top-20 candidates with query-doc relevance model
   - Implementation: Use `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Cost: ~50ms per query for reranking top-20 → top-5

3. **Diversity-Aware Ranking**
   - Penalize redundant documents in top-K
   - Promote diverse results (different topics, different sections)
   - Implementation: MMR (Maximal Marginal Relevance) algorithm
   - Cost: Minimal (post-processing step)

**Next Action**: Benchmark all 3 options on full corpus

---

### 🟡 MEDIUM PRIORITY: Extract Frontmatter Metadata

**Problem**: VERSION questions fail when answer is in YAML frontmatter

**Impact**: Could improve VERSION from 80% → 100% (1 question)

**Solution**:
1. Parse YAML frontmatter during chunking
2. Extract fields: `min_version`, `feature_state`, `ga_version`
3. Add metadata to chunk object
4. Use metadata in BM25 indexing or as filter

**Implementation**:
```python
# In MarkdownSemanticStrategy.chunk()
import yaml

def extract_frontmatter(content: str) -> dict:
    if content.startswith('---'):
        parts = content.split('---', 2)
        if len(parts) >= 3:
            return yaml.safe_load(parts[1])
    return {}

# Add to chunk metadata
chunk.metadata['frontmatter'] = extract_frontmatter(doc.content)
```

**Next Action**: Implement frontmatter extraction and test on VERSION questions

---

### 🟡 MEDIUM PRIORITY: Vocabulary Expansion

**Problem**: VOCABULARY questions at 60% (worst category)

**Impact**: Could improve VOCABULARY from 60% → 80%

**Solution**:
1. Build domain synonym map from documentation
2. Expand query with synonyms before retrieval
3. Example mappings:
   - "CPU placement policy" → "topology manager policy"
   - "IPC latency" → "inter-NUMA communication overhead"
   - "resource co-location" → "topology aligned resource allocation"

**Implementation Options**:

1. **Manual Synonym Map** (Quick win)
   ```python
   SYNONYMS = {
       "cpu placement": ["topology manager", "numa alignment"],
       "ipc latency": ["inter-numa overhead", "cross-numa communication"],
       # ... more mappings
   }
   ```

2. **LLM-Generated Synonyms** (Better coverage)
   - Use Claude to generate domain synonyms for each query
   - Cache results for common terms
   - Cost: ~1s per query (can be parallelized with rewrite)

3. **Fine-Tuned Embeddings** (Best long-term)
   - Fine-tune BGE model on K8s documentation
   - Learn domain-specific vocabulary automatically
   - Cost: One-time training, no query-time overhead

**Next Action**: Start with manual synonym map for top 10 vocabulary mismatches

---

### 🟢 LOW PRIORITY: Comparison-Aware Retrieval

**Problem**: COMPARISON questions at 80% (only 1 failure)

**Impact**: Could improve COMPARISON from 80% → 100% (1 question)

**Solution**:
1. Detect comparison queries (pattern: "X vs Y", "difference between")
2. Boost documents that mention BOTH concepts
3. Use multi-vector retrieval (one vector per concept)

**Implementation**:
```python
def is_comparison_query(query: str) -> bool:
    patterns = ["vs", "versus", "difference between", "compare", "differ"]
    return any(p in query.lower() for p in patterns)

def extract_concepts(query: str) -> list[str]:
    # Extract concepts being compared
    # Example: "none vs best-effort" → ["none", "best-effort"]
    pass

def boost_multi_concept_docs(results, concepts):
    # Boost docs that mention ALL concepts
    pass
```

**Next Action**: Defer until higher-priority items complete

---

### 🟢 LOW PRIORITY: Negation-Aware Retrieval

**Problem**: NEGATION questions at 60%

**Impact**: Could improve NEGATION from 60% → 80%

**Solution**:
1. Detect negation queries (pattern: "why not", "what's wrong", "can't")
2. Boost documents with limitation/constraint language
3. Use contrastive embeddings (trained on positive/negative pairs)

**Implementation**:
```python
def is_negation_query(query: str) -> bool:
    patterns = ["why not", "what's wrong", "can't", "won't", "doesn't", "limitation"]
    return any(p in query.lower() for p in patterns)

def boost_limitation_docs(results):
    # Boost docs with words like "limitation", "constraint", "not supported"
    pass
```

**Next Action**: Defer until higher-priority items complete

---

## Recommended Execution Order

### Phase 1: Quick Wins (1-2 days)

1. ✅ **Document findings** (DONE - this file)
2. ⏭️ **Implement frontmatter extraction** (2-3 hours)
   - Test on VERSION questions
   - Expected: 80% → 100% (1 question improvement)

3. ⏭️ **Build manual synonym map** (2-3 hours)
   - Top 10 vocabulary mismatches
   - Test on VOCABULARY questions
   - Expected: 60% → 70% (1 question improvement)

### Phase 2: Reranking Benchmark (2-3 days)

4. ⏭️ **Implement ColBERT reranking** (1 day)
   - Integrate `colbert-ai/colbertv2.0`
   - Benchmark on full corpus
   - Measure: pass rate, latency, memory

5. ⏭️ **Implement cross-encoder reranking** (1 day)
   - Integrate `cross-encoder/ms-marco-MiniLM-L-6-v2`
   - Benchmark on full corpus
   - Compare with ColBERT

6. ⏭️ **Implement diversity-aware ranking** (0.5 day)
   - MMR algorithm
   - Benchmark on full corpus
   - Compare with ColBERT and cross-encoder

7. ⏭️ **Select best reranking strategy** (0.5 day)
   - Compare all 3 options
   - Choose based on: pass rate, latency, complexity
   - Expected: 70% → 85%+ overall

### Phase 3: Advanced Improvements (Future)

8. ⏭️ **LLM-generated synonyms** (if manual map works well)
9. ⏭️ **Comparison-aware retrieval** (if time permits)
10. ⏭️ **Negation-aware retrieval** (if time permits)

---

## Success Metrics

| Metric | Current | Phase 1 Target | Phase 2 Target |
|--------|---------|----------------|----------------|
| **Overall Pass Rate** | 70% | 75% | 85%+ |
| VERSION | 80% | 100% | 100% |
| COMPARISON | 80% | 80% | 90%+ |
| NEGATION | 60% | 60% | 75%+ |
| VOCABULARY | 60% | 70% | 80%+ |
| **Avg Latency** | 1,123ms | <1,200ms | <1,500ms |

---

## Key Decisions

### ✅ KEEP: LLM Query Rewrite

**Rationale**: All rewrites are excellent, clearly helping performance

**Evidence**: 90% pass on small corpus, all failures had good rewrites

**Action**: Do NOT modify the rewrite prompt or model

---

### ✅ FOCUS: Reranking at Scale

**Rationale**: 20% drop when corpus expanded 7x shows ranking is the bottleneck

**Evidence**: 4 new failures all show pattern of "generic docs outrank specific docs"

**Action**: Prioritize reranking improvements over query improvements

---

### ⏭️ DEFER: Negation and Comparison Improvements

**Rationale**: Only 2 questions affected, lower ROI than reranking

**Evidence**: COMPARISON 80%, NEGATION 60%, but reranking could fix both

**Action**: Wait until after reranking benchmark to see if still needed

---

## Questions to Answer (Next Session)

1. **Which reranking strategy performs best?**
   - ColBERT vs cross-encoder vs diversity-aware
   - Trade-off: accuracy vs latency vs complexity

2. **Does frontmatter extraction fix VERSION failures?**
   - Test on adv_v03 specifically
   - Measure impact on other VERSION questions

3. **What are the top 10 vocabulary mismatches?**
   - Analyze all VOCABULARY failures
   - Build synonym map from patterns

4. **Is 85% pass rate achievable?**
   - With reranking + frontmatter + synonyms
   - On adversarial questions (intentionally hard)

---

## Files to Reference

- **Failure analysis**: `.sisyphus/notepads/llm-rewrite-logging/failure-analysis.md`
- **Rewrite analysis**: `.sisyphus/notepads/llm-rewrite-logging/analysis.md`
- **Benchmark results**: `poc/chunking_benchmark_v2/results/needle_questions_adversarial_retrieval.json`
- **Rewrite logs**: `poc/chunking_benchmark_v2/rewrite_log_full_corpus.txt`
- **Questions**: `poc/chunking_benchmark_v2/corpus/needle_questions_adversarial.json`
