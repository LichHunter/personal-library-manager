# Quick Reference: LLM Rewrite Analysis

**Date**: 2026-01-27  
**Status**: ✅ Analysis Complete

---

## One-Line Summary

**LLM rewrite is EXCELLENT (keep unchanged) - bottleneck is retrieval at scale (focus on reranking)**

---

## Performance Summary

| Metric | Small Corpus (200 docs) | Full Corpus (1,569 docs) | Change |
|--------|-------------------------|--------------------------|--------|
| **Pass Rate** | 90% (18/20) | 70% (14/20) | **-20%** ⬇️ |
| Avg Latency | 1,136ms | 1,123ms | -13ms |
| Index Time | 16.6s | 116.5s | +100s |

---

## Category Performance

| Category | Small Corpus | Full Corpus | Degradation |
|----------|--------------|-------------|-------------|
| VERSION | 80% (4/5) | 80% (4/5) | 0% |
| COMPARISON | 100% (5/5) | 80% (4/5) | -20% |
| NEGATION | 80% (4/5) | 60% (3/5) | -20% |
| VOCABULARY | 100% (5/5) | 60% (3/5) | **-40%** ⬇️ |

---

## Failed Questions

### Small Corpus (2 failures)

| ID | Category | Question | Rewrite Quality |
|----|----------|----------|-----------------|
| adv_v03 | VERSION | When did prefer-closest-numa-nodes become GA? | ✅ EXCELLENT |
| adv_n04 | NEGATION | What's wrong with container scope? | ✅ EXCELLENT |

### Full Corpus (6 failures)

| ID | Category | Question | Rewrite Quality | Root Cause |
|----|----------|----------|-----------------|------------|
| adv_v03 | VERSION | When did prefer-closest-numa-nodes become GA? | ✅ EXCELLENT | Frontmatter metadata |
| adv_c03 | COMPARISON | Compare none vs best-effort policy | ✅ EXCELLENT | Retrieval noise |
| adv_n03 | NEGATION | Why can't scheduler prevent failures? | ✅ GOOD | Retrieval noise |
| adv_n04 | NEGATION | What's wrong with container scope? | ✅ EXCELLENT | Semantic gap |
| adv_m01 | VOCABULARY | How to configure CPU placement? | ✅ EXCELLENT | Vocab mismatch + noise |
| adv_m05 | VOCABULARY | How to optimize IPC latency? | ✅ GOOD | Vocab mismatch + noise |

**Key Insight**: ALL failures had good-to-excellent rewrites. Problem is NOT the LLM rewrite.

---

## Root Causes

| Root Cause | Count | Examples |
|------------|-------|----------|
| **Retrieval noise** | 3 | adv_c03, adv_n03, adv_m01 |
| **Vocabulary mismatch + noise** | 2 | adv_m01, adv_m05 |
| **Frontmatter metadata** | 1 | adv_v03 |
| **Semantic gap** | 1 | adv_n04 |

---

## Rewrite Quality Assessment

| Quality | Count | Percentage |
|---------|-------|------------|
| ✅ EXCELLENT | 18/20 | 90% |
| ✅ GOOD | 2/20 | 10% |
| ❌ BAD | 0/20 | 0% |

**Verdict**: LLM rewrite is working perfectly. Do NOT modify.

---

## Prioritized Recommendations

| Priority | Action | Target | Expected Impact |
|----------|--------|--------|-----------------|
| 🔴 HIGH | Implement reranking (ColBERT/cross-encoder) | Overall | 70% → 85%+ |
| 🟡 MEDIUM | Extract frontmatter metadata | VERSION | 80% → 100% |
| 🟡 MEDIUM | Build vocabulary synonym map | VOCABULARY | 60% → 80% |
| 🟢 LOW | Comparison-aware retrieval | COMPARISON | 80% → 100% |
| 🟢 LOW | Negation-aware retrieval | NEGATION | 60% → 80% |

---

## Next Steps

### Phase 1: Quick Wins (1-2 days)
1. ✅ Document findings
2. ⏭️ Implement frontmatter extraction
3. ⏭️ Build manual synonym map

### Phase 2: Reranking Benchmark (2-3 days)
4. ⏭️ Implement ColBERT reranking
5. ⏭️ Implement cross-encoder reranking
6. ⏭️ Implement diversity-aware ranking
7. ⏭️ Select best strategy

### Phase 3: Advanced (Future)
8. ⏭️ LLM-generated synonyms
9. ⏭️ Comparison-aware retrieval
10. ⏭️ Negation-aware retrieval

---

## Key Files

| File | Purpose |
|------|---------|
| `ANALYSIS_SUMMARY.md` | Executive summary (this is the main file) |
| `failure-analysis.md` | Detailed breakdown of all 6 failures |
| `analysis.md` | All 20 query rewrites analyzed |
| `next-steps.md` | Implementation roadmap |
| `full-corpus-results.md` | Small vs full corpus comparison |
| `QUICK_REFERENCE.md` | This file (quick lookup) |

---

## Key Decisions

| Decision | Rationale |
|----------|-----------|
| ✅ KEEP LLM rewrite unchanged | All rewrites excellent, clearly helping |
| ✅ FOCUS on reranking | 20% drop shows ranking is bottleneck |
| ⏭️ DEFER negation/comparison | Only 2 questions, lower ROI |

---

## Example Rewrites

### ✅ Excellent Rewrite (Failed Due to Noise)

**Original**: "How do I optimize inter-process communication latency for pods?"

**Rewritten**: "kubernetes pod inter-process communication network performance optimization strategies"

**Quality**: ✅ EXCELLENT
- Added: "kubernetes", "network", "performance", "strategies"
- Preserved: "pod", "inter-process communication"
- Transformed: "how do I optimize" → "optimization strategies"

**Why it failed**: User says "IPC latency", docs say "inter-NUMA overhead". With 1,569 docs, generic networking docs outrank the specific NUMA solution.

**Root cause**: VOCABULARY MISMATCH + RETRIEVAL NOISE (not rewrite quality)

---

### ✅ Excellent Rewrite (Passed)

**Original**: "How does k8s coordinate resource co-location across multi-socket servers?"

**Rewritten**: "kubernetes multi-socket numa topology resource scheduling placement affinity"

**Quality**: ✅ EXCELLENT
- Expanded: "k8s" → "kubernetes"
- Transformed: "co-location" → "scheduling placement affinity"
- Added: "numa", "topology"

**Why it passed**: Rewrite added the KEY term "numa topology" that bridges the vocabulary gap.

**Success factor**: LLM rewrite successfully mapped user vocabulary to documentation vocabulary.

---

## Conclusion

**Question**: Is the LLM rewrite helping or hurting?

**Answer**: **HELPING - significantly.**

**Evidence**:
- 90% pass rate on small corpus
- 100% of rewrites are good-to-excellent
- All failures had good rewrites (problem is retrieval, not rewrite)

**Next Focus**: Reranking strategies for large corpora (ColBERT, cross-encoder, diversity-aware)

**Expected Outcome**: 70% → 85%+ on adversarial questions

---

**Analysis Complete** ✅
