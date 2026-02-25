# Phase 2 (P0) Review Summary

**Date:** 2026-02-21
**Approaches tested:** Reranking (#01), Parent-Child (#15), Auto-Merging (#02)
**Test set:** 229 queries (54 factoid, 55 procedural, 53 explanatory, 33 comparison, 34 troubleshooting)

---

## 1. Overall Comparison Table

| Metric | Baseline | Reranking | Parent-Child | Auto-Merge |
|--------|----------|-----------|--------------|------------|
| **Answer Success Rate** | 92.1% | **98.3%** (+6.1) | 88.6% (-3.5) | 95.2% (+3.1) |
| **MRR@10 (labeled)** | 0.658 | **0.772** (+0.114) | 0.646 (-0.012) | 0.657 (-0.001) |
| Hit@5 (labeled) | 75.4% | **89.2%** (+13.8) | 75.4% (+0.0) | 75.4% (+0.0) |
| Avg Grade | 7.64/10 | **8.20/10** (+0.56) | 7.52/10 (-0.12) | 8.03/10 (+0.39) |
| Latency p95 | 422ms | 1817ms (+1396) | 918ms (+496) | 903ms (+481) |
| Avg Tokens Retrieved | 904 | 975 (+71) | 2255 (+1351) | 1095 (+191) |
| Cannot Answer | 6.1% | **1.3%** (-4.8) | 7.4% (+1.3) | 2.2% (-3.9) |
| Incorrect | 1.7% | **0.4%** (-1.3) | 3.9% (+2.2) | 2.6% (+0.9) |

---

## 2. Per-Query-Type Best Approach

| Query Type | Baseline SR | Best P0 | Best SR | Delta |
|------------|-------------|---------|---------|-------|
| **Factoid** | 96.3% | Reranking / Auto-Merge | 98.1% / 100.0% | +1.9% / +3.7% |
| **Procedural** | 100.0% | Baseline (all match) | 100.0% | +0.0% |
| **Explanatory** | 71.7% | **Reranking** | **94.3%** | **+22.6%** |
| **Comparison** | 100.0% | Baseline (all match) | 100.0% | +0.0% |
| **Troubleshooting** | 97.1% | Reranking | 100.0% | +2.9% |

### Detailed Per-Type Comparison

| Type | Baseline | Reranking | Parent-Child | Auto-Merge |
|------|----------|-----------|--------------|------------|
| Factoid | 96.3% | 98.1% | 85.2% ❌ | **100.0%** |
| Procedural | 100.0% | 100.0% | 98.2% | 100.0% |
| Explanatory | 71.7% | **94.3%** | 71.7% | 81.1% |
| Comparison | 100.0% | 100.0% | 100.0% | 100.0% |
| Troubleshooting | 97.1% | **100.0%** | 94.1% | 97.1% |

---

## 3. Approach Verdicts

### Reranking (#01) — BEST P0 APPROACH

**Strengths:**
- Largest ASR improvement: +6.1% (92.1% → 98.3%)
- Largest MRR improvement: +0.114 (0.658 → 0.772)
- Massive explanatory improvement: +22.6% (71.7% → 94.3%)
- No regressions on any query type
- Minimal token increase (+71 avg)
- "Cannot Answer" rate dropped from 6.1% to 1.3%

**Weaknesses:**
- Latency: +1396ms p95 (1817ms total) — far exceeds 500ms limit
- CPU cross-encoder inference is the bottleneck (~1.1s per query for 50 candidates)

**Verdict:** NEEDS MODIFICATION — quality gains are excellent but latency is unacceptable. Options:
1. Reduce candidates_k from 50 to 20-30 (linear latency reduction)
2. GPU inference would reduce to ~50ms (but not available on target hardware)
3. Use distilled/quantized model

### Parent-Child (#15) — REJECTED

**Strengths:**
- Latency within acceptable range (+496ms p95)

**Weaknesses:**
- ASR regression: -3.5% (92.1% → 88.6%)
- Factoid queries regressed -11.1% (96.3% → 85.2%) — exceeds 5% regression threshold
- Heading-level content is too noisy for factoid queries (dilutes the answer)
- 2.5x more tokens with no quality benefit

**Verdict:** REJECT — heading-level content is too coarse for most queries. The noise-to-signal ratio hurts precision.

### Auto-Merging (#02) — MARGINAL, REJECTED

**Strengths:**
- Moderate ASR improvement: +3.1% (92.1% → 95.2%)
- No regressions on any query type
- Merge logic itself is very fast (3-4ms)
- Latency within acceptable range (+481ms p95)

**Weaknesses:**
- ASR improvement (+3.1%) well below +10% threshold
- Explanatory improvement (+9.4%) less than half of reranking's (+22.6%)
- 92.6% merge rate suggests over-merging (almost every query triggers merges)

**Verdict:** REJECT — marginal improvement doesn't justify the complexity. Reranking achieves 2x the gains.

---

## 4. Success Criteria Evaluation

| Criterion | Target | Reranking | Parent-Child | Auto-Merge |
|-----------|--------|-----------|--------------|------------|
| ASR ≥ +10% | +10.0% | +6.1% ❌ | -3.5% ❌ | +3.1% ❌ |
| Latency ≤ +500ms p95 | ≤500ms | +1396ms ❌ | +496ms ✅ | +481ms ✅ |
| No type regresses >5% | >-5% | ✅ | -11.1% ❌ | ✅ |

**No P0 approach meets all three success criteria.**

However, reranking is close on ASR (+6.1% vs +10% target) and could potentially reach the target if:
- Combined with another approach targeting explanatory queries
- Latency is reduced via fewer candidates or batched inference

---

## 5. Key Insights

1. **Explanatory queries are the key weakness.** Baseline: 71.7%. This is the category with the most room for improvement. Reranking brought it to 94.3%, but there are still 3 explanatory failures.

2. **Reranking works because it promotes better chunks, not because it adds context.** Token count barely changed (+71). The quality improvement comes from better ranking of the existing retrieval pool.

3. **Parent-child fails because heading-level context is too coarse.** Factoid queries suffer when drowned in irrelevant sibling content. The approach adds noise faster than signal.

4. **Auto-merging is the weakest improvement.** 92.6% merge rate means nearly all queries get expanded, diluting the benefit for queries that need precision.

5. **The remaining 4 failing queries** (in reranking) are consistently hard across approaches: `adv_adv_n04`, `adv_adv_m03`, `expl_gen_016`, `expl_gen_038`. These may be missing from the corpus entirely.

---

## 6. Decision

### **CONTINUE** — Proceed to P1 approaches

**Justification:**
- No P0 approach met the +10% ASR threshold
- Reranking achieved +6.1% ASR — strong but insufficient alone
- P1 approaches (especially adaptive classifier + reranking combination) could close the gap
- The explanatory query weakness is well-characterized and targetable

### Recommended P1 Focus

1. **Adaptive Classifier (#07):** Route explanatory/comparison queries to heading-level search + reranking, keep factoid/procedural at chunk-level. This should combine the best of both worlds.

2. **Iterative Expansion (#08):** Start with chunk retrieval + reranking, expand to heading-level only if LLM judges context insufficient. Could fix the remaining 3 explanatory failures.

3. **Reranking with reduced candidates (#01 variant):** Test candidates_k=20 to see if we can achieve similar quality with lower latency.

### What P1 Must Address

- Push ASR past 98% (ideally ≥ 100% = +10% over baseline of 92.1% → need ~100%)
- Keep latency under 500ms increase (reranking needs optimization)
- Maintain zero regressions on procedural/comparison/troubleshooting
- Target the remaining explanatory failures

---

*Generated from P0 results on 2026-02-21*
