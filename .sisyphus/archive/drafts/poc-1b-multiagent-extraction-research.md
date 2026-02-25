# POC-1b Research: Multi-Agent Term Extraction

## Status: Research In Progress
**Date**: 2026-02-05
**Target**: 95% Precision, 95% Recall, <5% Hallucination

---

## Current Best Results (None Hit Target)

| Strategy | Precision | Recall | Hallucination | Gap to Target |
|----------|-----------|--------|---------------|---------------|
| **ensemble_verified** | 89.3% | 88.9% | 10.7% | P:-5.7, R:-6.1, H:+5.7 |
| **vote_3** | 97.8% | 75.1% | 2.2% | P:+2.8, R:-19.9, H:-2.8 |
| **ensemble_sonnet_verify** | 94.4% | 70.9% | 5.6% | P:-0.6, R:-24.1, H:+0.6 |
| **exhaustive_sonnet** | 45.8% | 97.3% | 54.2% | P:-49.2, R:+2.3, H:+49.2 |

### The Fundamental Tradeoff Observed

```
HIGH RECALL (95%+) requires OVER-EXTRACTION → 40-50% hallucination
HIGH PRECISION (95%+) requires VERIFICATION → 65-75% recall

No approach achieved BOTH simultaneously.
```

---

## Strategies Tested in POC-1b

### From hybrid_results.txt (8 chunks):
- `exhaustive_haiku`: 56.5% P, 88.6% R, 43.5% H
- `exhaustive_sonnet`: 45.8% P, 97.3% R, 54.2% H
- `ensemble_sonnet_verify`: 94.4% P, 70.9% R, 5.6% H
- `sonnet_exh_haiku_verify`: 70.5% P, 79.3% R, 29.5% H
- `ensemble_vote_hybrid`: 64.1% P, 84.1% R, 35.9% H
- `multi_consensus`: 71.4% P, 80.9% R, 28.6% H
- `union_verified`: 88.3% P, 72.1% R, 11.7% H
- `exhaustive_double_verify`: 93.7% P, 65.2% R, 6.3% H

### From results_output.txt (5 chunks):
- `simple_haiku`: 69.0% P, 93.0% R, 31.0% H
- `quote_haiku`: 73.3% P, 61.3% R, 6.7% H
- `exhaustive_haiku`: 58.2% P, 95.7% R, 41.8% H
- `ensemble_verified`: 89.3% P, 88.9% R, 10.7% H (CLOSEST TO TARGET)
- `vote_2`: 65.2% P, 92.7% R, 34.8% H
- `vote_3`: 97.8% P, 75.1% R, 2.2% H
- `rated_important`: 64.9% P, 90.4% R, 35.1% H
- `sonnet_conservative`: 98.2% P, 59.0% R, 1.8% H
- `quote_verified`: 80.0% P, 59.8% R, 0.0% H
- `union_conservative`: 85.7% P, 72.8% R, 14.3% H

### From discrimination_results.json:
- `pattern_only`: 43.0% P, 33.2% R, 52.5% H
- `discrimination_no_cat`: 59.1% P, 18.6% R, 25.9% H
- `discrimination_full`: 64.0% P, 62.5% R, 31.0% H
- `discrimination_sonnet`: 70.7% P, 57.8% R, 19.3% H

---

## Why Current Multi-Agent Approaches Failed

**Current pattern**: Agent A extracts → Agent B FILTERS (destructive)

- Filtering is DESTRUCTIVE - once removed, a term is gone
- Each verification stage loses ~10-15% recall
- Double verification = 65% recall

**Key insight from user**: "multiple agents to check each other's work" is different from filtering

What we need:
- Agent A extracts
- Agent B extracts independently (different prompt/approach)
- Agent C **DELIBERATES on disagreements** (not just counts votes)

---

## Proposed New Approach: Sentence-Level Exhaustive + Deliberation

### Hypothesis

Oracle rejected sentence-splitting for EXTRACTION context. But what about EXHAUSTIVENESS?

**Current failure mode**: Haiku processes 200-word chunk → FORGETS some terms (attention/memory issue)

**New hypothesis**: If we force Haiku to process each sentence independently, it can't FORGET terms because each sentence is processed fresh.

### Proposed Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 1: Exhaustive Sentence-Level Extraction (Haiku × N)     │
│                                                                 │
│  For each sentence in chunk:                                    │
│    "What K8s terms appear in: '{sentence}'?"                    │
│    Output: term + 5-word explanation                            │
│                                                                 │
│  Why: Can't miss terms due to attention - each sentence fresh   │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 2: Cross-Reference & Confidence Scoring                  │
│                                                                 │
│  - Aggregate all terms from all sentences                       │
│  - Count: How many sentences mentioned this term?               │
│  - Multi-mention = HIGH confidence (keep without review)        │
│  - Single-mention = LOW confidence (needs deliberation)         │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 3: Deliberation on Edge Cases (Sonnet)                   │
│                                                                 │
│  ONLY for low-confidence terms:                                 │
│    Given: term, explanation, original sentence                  │
│    "Is '{term}' a K8s-specific technical term here?"            │
│                                                                 │
│  Why: Sonnet only reviews EDGE CASES, not everything            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│  PHASE 4: Final Assembly + Span Verification                    │
│                                                                 │
│  Keep: HIGH confidence + approved LOW confidence                │
│  Remove: rejected LOW confidence + span verification fails      │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Might Hit 95/95/5

1. **Sentence-level extraction prevents FORGETTING** - can't miss terms due to attention
2. **Multiple independent extractions** (one per sentence) = implicit voting
3. **Explanations provide signal** for downstream deliberation
4. **Sonnet deliberates, doesn't just filter** - reviews EDGE CASES with context
5. **High-confidence terms skip verification** - preserves recall

### Comparison to Failed Approaches

| Failed Approach | Why It Failed | New Approach |
|-----------------|---------------|---------------|
| Ensemble → Filter | Filter destroys recall | Deliberation on edge cases only |
| Vote_3 (3 must agree) | Too strict | Multi-mention = keep, single-mention = review |
| Double verify | Each step loses recall | Only low-confidence gets reviewed |
| Full-chunk extraction | Haiku forgets terms | Sentence-by-sentence can't forget |

---

## Open Questions to Explore

1. **Does sentence-level actually improve Haiku's exhaustiveness?**
   - Test: Same chunk, full vs sentence-by-sentence, measure recall

2. **What's the right confidence threshold?**
   - Terms in 2+ sentences = high confidence?
   - Or should explanation quality factor in?

3. **How should Sonnet deliberate?**
   - Binary yes/no per term?
   - Or see multiple low-confidence terms at once for consistency?

4. **Cost analysis**:
   - 200-word chunk ≈ 10 sentences × $0.0001 (Haiku) = $0.001
   - Plus Sonnet for ~30% of terms (edge cases) ≈ $0.002
   - Total ≈ $0.003/chunk

5. **Alternative: Structured table output**
   - User suggested JSON with explanations
   - Could help downstream filtering

---

## Key Files in POC-1b

| File | Purpose | Key Results |
|------|---------|-------------|
| `test_hybrid_final.py` | Combined strategies | ensemble_sonnet_verify: 94.4%P, 70.9%R |
| `test_fast_combined.py` | Voting + ensemble | vote_3: 97.8%P, 75.1%R |
| `test_discrimination_approach.py` | Pattern + LLM discrimination | 70.7%P, 57.8%R max |
| `test_quote_extract_multipass.py` | Multi-pass with categories | Not yet run fully |
| `test_small_chunk_extraction.py` | Small chunk baseline | 92%R possible but 48%H |
| `artifacts/small_chunk_ground_truth.json` | 15 chunks, 163 terms | Avg 10.9 terms/chunk |
| `artifacts/small_chunk_results.json` | Strategy comparison | |
| `hybrid_results.txt` | 8-chunk hybrid test | |
| `results_output.txt` | 5-chunk fast combined | |

---

## Ground Truth Stats

- **15 chunks** in small_chunk_ground_truth.json
- **163 total terms** (avg 10.9 per chunk)
- Term tiers: 1 (essential), 2 (important), 3 (nice-to-have)
- Created by Claude Opus with span verification

---

## Next Steps (To Discuss)

1. Design experiment for sentence-level exhaustive extraction hypothesis
2. Define exact prompts for each phase
3. Determine confidence threshold criteria
4. Test on subset before full run
5. Compare cost vs quality vs current best

---

*Draft saved: 2026-02-05*
*Status: Awaiting discussion to refine approach*
