# Full Corpus Benchmark Results

**Date**: 2026-01-27 18:40  
**Corpus**: Full Kubernetes (1,569 docs → 7,269 chunks)  
**Questions**: 20 adversarial questions  
**Pass Rate**: 70% (14/20)

---

## Comparison: Small vs Full Corpus

| Metric | 200 Docs (1,030 chunks) | 1,569 Docs (7,269 chunks) | Change |
|--------|-------------------------|---------------------------|--------|
| **Pass Rate** | 90% (18/20) | 70% (14/20) | **-20%** ⬇️ |
| **Avg Latency** | 1,136ms | 1,123ms | -13ms (similar) |
| **Index Time** | 16.6s | 116.5s | +100s (7x more chunks) |

---

## Key Finding

**More documents = harder retrieval**, even with LLM rewrite.

The 20% drop in pass rate (90% → 70%) shows that:
1. **Noise increases with corpus size** - more irrelevant docs compete for top-5 slots
2. **LLM rewrite still helps** - but can't overcome the fundamental challenge of larger search space
3. **The 4 new failures** are likely due to relevant docs being pushed out of top-5 by noise

---

## Which Questions Failed?

Need to compare the two result files to see which 4 questions went from PASS → FAIL when corpus expanded.

**200-doc failures** (2/20):
- Q3: "When did prefer-closest-numa-nodes become GA?" (VERSION)
- Q14: "What's wrong with container scope for latency-sensitive apps?" (NEGATION)

**Full corpus failures** (6/20):
- Need to check which 4 additional questions failed

---

## Implications

1. **LLM rewrite is necessary but not sufficient** for large corpora
2. **Need better ranking/reranking** to handle noise at scale
3. **Semantic search struggles** when similar docs compete
4. **70% is still decent** for adversarial questions on full corpus

---

## Next Steps

1. Compare the two result files to identify the 4 new failures
2. Analyze if they're random or follow a pattern
3. Consider reranking strategies for large corpora
