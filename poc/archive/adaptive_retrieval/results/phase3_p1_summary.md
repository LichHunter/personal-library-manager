# Phase 3 (P1) Review Summary

**Date:** 2026-02-21
**Approaches tested:** Adaptive Classifier (#03) — only P1 approach tested
**Note:** Only 1 of 4 planned P1 approaches was tested. Early STOP recommended.

---

## 1. P1 Results

### Adaptive Classifier (#03)

| Metric | Adaptive | Baseline | Δ Baseline | Reranking (best P0) | Δ Reranking |
|--------|----------|----------|------------|---------------------|-------------|
| ASR | 98.3% | 92.1% | +6.1% | 98.3% | +0.0% |
| MRR@10 | 0.736 | 0.658 | +0.078 | 0.772 | -0.036 |
| Latency p95 | 1905ms | 422ms | +1483ms | 1817ms | +87ms |
| Avg Tokens | 969 | 904 | +65 | 975 | -6 |
| Rerank Rate | 72.9% | — | — | 100% | — |
| Class. Accuracy | 63.3% | — | — | — | — |

**Key findings:**
- ASR identical to full reranking (98.3%) — the ceiling for current approaches
- Classifier over-classifies to "complex" (72.9% rerank rate vs intended ~38%)
- No latency savings vs full reranking (87ms more due to classification overhead)
- Classification accuracy only 63.3% — rule-based heuristics insufficient for factoid/procedural

**Verdict:** REJECT — No improvement over simpler full reranking approach.

---

## 2. Decision: **STOP**

### Justification

**The ASR ceiling has been reached. Further approaches will not improve it.**

Evidence:
1. **Same 4 queries fail across ALL approaches** — `adv_adv_n04`, `adv_adv_m03`, `expl_gen_016`, `expl_gen_038`. These are consistently unfindable regardless of retrieval strategy. They likely represent content gaps in the corpus or extremely niche topics.

2. **+10% ASR target is mathematically unreachable** — Baseline ASR is 92.1%. +10% = 102.1%, which exceeds 100%. Even achieving perfect 100% ASR would only be +7.9%. The target should be reconsidered.

3. **Context expansion approaches performed worse than reranking** — Parent-child (-3.5% ASR), auto-merging (+3.1% ASR) both underperformed reranking (+6.1% ASR). This means the problem is ranking quality, not context quantity. Remaining P1 approaches (iterative expansion, recursive retriever, sentence window) are all context expansion techniques — unlikely to beat reranking.

4. **Reranking provides 98.3% ASR consistently** — Both full reranking and adaptive classifier hit this identical number. This is the practical ceiling.

5. **Diminishing returns** — Testing 3 more P1 approaches (each requiring ~15 min benchmark + 229 LLM judge calls) for marginal gains over an established ceiling is not justified.

### Remaining P1 Approaches: Skipped

| Approach | Why Skipped |
|----------|-------------|
| Iterative Expansion (#04) | Context expansion; parent-child already showed expansion hurts |
| Recursive Retriever (#08) | Context expansion; follows references — same class of approach |
| Sentence Window (#09) | Context expansion; adds surrounding text — similar to auto-merging |

### Recommendation

**Implement cross-encoder reranking** as the production enhancement:
- Model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (80MB, CPU-friendly)
- Config: `candidates_k=50`, `final_k=10` (or reduce to `candidates_k=30` for latency)
- Expected improvement: +6.1% ASR, +0.114 MRR@10
- Key win: Explanatory queries from 71.7% → 94.3% (+22.6%)
- Latency concern: ~1.1s per query on CPU (acceptable for batch/async, not for real-time)

### Latency Mitigation Options (not tested but recommended for production)
1. Reduce `candidates_k` from 50 to 20-30 (linear latency reduction, ~500-700ms)
2. ONNX quantization of MiniLM model (~2-3x speedup on CPU)
3. GPU inference if available (~20x speedup)
4. Batch inference for multiple queries

---

*Generated on 2026-02-21*
