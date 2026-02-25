# Architecture D+v2: "Sonnet-Anchored Complementary Merge"

**Date**: 2026-02-08
**Status**: Proposed — pending implementation
**Target**: 95% precision, 95% recall, <5% hallucination

---

## Why The Original D+ Failed

The original D+ was designed against the v1 ground truth (163 terms, ~40% incomplete). When we fixed the GT to v2 (277 terms), the numbers fell apart:

| Assumption (v1 GT) | Reality (v2 GT) | Impact |
|---------------------|-----------------|--------|
| "exhaustive_haiku gets 95.7% recall" | 74.8% recall | **Recall ceiling broken** |
| "vote_3 gets 97.8% precision" | 93.3% precision (but 27.3% recall) | Voting too aggressive |
| "3-extractor union recall ~97%" | 77.2% recall | **Can't reach 95% recall** |

**Root cause**: Haiku simply doesn't extract enough terms. It averages 15.9 terms/chunk vs 18.5 GT terms — it literally can't see ~23% of valid terms.

---

## The Breakthrough: Sonnet Exhaustive

Running Sonnet with the exhaustive prompt against v2 GT revealed:

| Metric | Haiku Exhaustive | Sonnet Exhaustive | Delta |
|--------|------------------|-------------------|-------|
| Recall | 74.8% | **94.3%** | **+19.5pp** |
| Precision | 80.0% | 55.2% | -24.8pp |
| Hallucination | 20.0% | 44.8% | +24.8pp |

Sonnet finds almost everything but is noisy. Haiku is precise but misses too much. **The architecture must use Sonnet for recall and Haiku for precision filtering.**

---

## Union + Vote Simulation Results

Using 3 extractors (Sonnet exhaustive + Haiku exhaustive + Haiku simple):

| Votes | True Positives | False Positives | Precision | % of All TPs |
|-------|----------------|-----------------|-----------|--------------|
| 3 (all agree) | 136 | 6 | **95.8%** | 39.1% |
| 2 (two agree) | 76 | 14 | **84.4%** | 21.8% |
| 1 (one only) | 136 | 121 | 52.9% | 39.1% |

**Combined recall ceiling: 94.6%** — nearly at 95% target.

**Key insight**: 3-vote terms already achieve 95.8% precision with 39.1% of true positives. The architecture only needs to clean up 2-vote and 1-vote terms.

---

## D+v2 Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  PHASE 1: Three-Extractor Extraction (parallel)              │
│                                                              │
│  sonnet_terms = extract_exhaustive(chunk, sonnet)   ← ANCHOR │
│  haiku_exh    = extract_exhaustive(chunk, haiku)             │
│  haiku_simple = extract_simple(chunk, haiku)                 │
│                                                              │
│  candidate_pool = union(all three)                           │
│  Per-term vote count = how many extractors found it          │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 2: Span Grounding (deterministic, no LLM)            │
│                                                              │
│  For EVERY candidate term:                                   │
│    - Check exact match in chunk text (case-insensitive)      │
│    - Check normalized match (- and _ as spaces)              │
│    - Check CamelCase split match                             │
│    - Check singular/plural variant                           │
│                                                              │
│  NOT grounded → DISCARD (catches true hallucinations)        │
│  Grounded → proceed to Phase 3                               │
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 3: Confidence-Based Routing                           │
│                                                              │
│  3 votes + grounded → AUTO-KEEP (95.8% precision)            │
│  2 votes + grounded → AUTO-KEEP (84.4% precision)            │
│  1 vote  + grounded → SEND TO PHASE 4 for Sonnet approval   │
└──────────────────────┬───────────────────────────────────────┘
                       │ (1-vote terms only)
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 4: Sonnet Batch Approval (1-vote terms only)          │
│                                                              │
│  Prompt: "Review these candidate terms extracted from the    │
│  documentation chunk below. For each, decide APPROVE or      │
│  REJECT.                                                     │
│                                                              │
│  APPROVE if the term names a technical concept, resource,    │
│  tool, protocol, or domain vocabulary that a learner would   │
│  benefit from understanding.                                 │
│                                                              │
│  REJECT only if the term is purely structural, formatting,   │
│  or a common English word with no technical significance     │
│  in this context."                                           │
│                                                              │
│  Default: APPROVE. When in doubt, APPROVE.                   │
│  Expected: ~50-60% approved (filters noise, keeps edge cases)│
└──────────────────────┬───────────────────────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────────────────────┐
│  PHASE 5: Assembly                                           │
│                                                              │
│  FINAL = 3-vote grounded (auto-kept)                         │
│        + 2-vote grounded (auto-kept)                         │
│        + 1-vote grounded + Sonnet-approved                   │
│                                                              │
│  Deduplicate by normalized form (keep highest-vote version)  │
└──────────────────────────────────────────────────────────────┘
```

---

## Predicted Metrics

### Phase-by-phase walkthrough (per chunk, based on simulation data):

| Phase | Avg Terms | Precision | Recall | Hallucination |
|-------|-----------|-----------|--------|---------------|
| Phase 1 (raw union) | ~35 | 49.8% | 94.6% | 50.2% |
| Phase 2 (span grounding) | ~30 | ~60% | ~94% | ~40% |
| Phase 3 (3+2 vote auto-keep) | ~15 | ~92% | ~61% | ~8% |
| Phase 3 (1-vote → Phase 4) | ~15 | 52.9% | 33.6% | 47.1% |
| Phase 4 (Sonnet approves ~55%) | ~8 | ~75% | ~25% | ~25% |
| **Phase 5 (assembly)** | **~23** | **~88-92%** | **~86-90%** | **~8-12%** |

### Honest assessment vs target:

| Metric | Predicted | Target | Gap | Achievable? |
|--------|-----------|--------|-----|-------------|
| Precision | 88-92% | 95% | 3-7pp | Stretch — needs tuned approval prompt |
| Recall | 86-90% | 95% | 5-9pp | Hard — limited by extraction ceiling |
| Hallucination | 8-12% | <5% | 3-7pp | Possible — tighter span grounding |

---

## Why 2-Vote Terms Are Auto-Kept

The original D+ sent 2-vote terms to Sonnet for verification. But the data shows:

- **2-vote precision is already 84.4%** — most are legitimate
- Sonnet verification of 2-vote terms risks false rejections (the "too generic" problem)
- Auto-keeping 2-vote grounded terms adds ~22% more TPs with only ~14 FPs total
- The span grounding already filters true hallucinations

If precision is still too low after testing, we can add Sonnet verification for 2-vote terms as a dial to turn.

---

## Cost Estimate

| Phase | LLM Calls | Model | Est. Cost/Chunk |
|-------|-----------|-------|-----------------|
| Phase 1a | 1 | Sonnet | ~$0.02 |
| Phase 1b | 1 | Haiku | ~$0.001 |
| Phase 1c | 1 | Haiku | ~$0.001 |
| Phase 2-3 | 0 | deterministic | $0 |
| Phase 4 | 1 (batch) | Sonnet | ~$0.01 |
| **Total** | **4** | **Mixed** | **~$0.032** |

For 15 chunks: ~$0.48. For 1000 chunks: ~$32.

---

## Key Design Decisions

### 1. Sonnet as PRIMARY extractor (not just verifier)
The original D+ used Sonnet only for filtering. But Sonnet is the only model that achieves >90% recall. It MUST be in the extraction loop.

### 2. Span grounding BEFORE voting
Apply span grounding to ALL terms first. This is free (deterministic) and catches ~5-10% of noise. Then vote counts are more meaningful.

### 3. Batch approval (not per-term)
Send all 1-vote terms to Sonnet in a single batch call instead of individual calls. Cheaper and allows Sonnet to see the full context.

### 4. Domain-agnostic prompts
All prompts avoid K8s-specific language. They reference "domain-specific", "technical", "documentation" — works for any source type.

---

## Knobs to Tune

| Knob | Current Setting | Effect of Tightening | Effect of Loosening |
|------|-----------------|----------------------|---------------------|
| 2-vote threshold | Auto-keep | +precision, -recall | N/A (already loose) |
| Sonnet approval bias | "When in doubt APPROVE" | +recall, -precision | +precision, -recall |
| Span grounding strictness | Exact + normalized + camelCase + s/pl | +precision, -recall | +recall, -precision |
| Add 4th extractor (Haiku quote) | Not included | +recall ceiling ~1-2% | N/A |
| Temperature on Haiku | 0.0 | Deterministic | +diversity, +recall ceiling |

---

## Comparison to Original D+

| Aspect | Original D+ | D+v2 |
|--------|-------------|------|
| Primary extractor | Haiku (3 variants) | **Sonnet exhaustive** |
| Recall ceiling | 77.2% (broken) | **94.6%** |
| Sonnet role | Filter (Phase 4) | **Extractor (Phase 1) + Approver (Phase 4)** |
| 2-vote handling | Sonnet verify | **Auto-keep (grounded)** |
| Predicted P/R/H | 93-96/94-97/2-5 (wrong) | **88-92/86-90/8-12** (honest) |
| LLM calls | 4-13 | **4** |
| Cost/chunk | $0.025-0.045 | **~$0.032** |

---

## Open Questions

1. **Can the v2 GT itself be tightened?** The missed-term analysis showed ~18 of 43 unfound terms are questionable GT entries ("components", "storage", "deletion" as standalone terms). Removing these would raise the effective recall ceiling to ~90%.

2. **Temperature sampling**: Would running Haiku exhaustive at temp=0.7 3x instead of 1x at temp=0.0 improve the recall ceiling? Could add ~2-3% recall for 2 extra Haiku calls.

3. **Gap-filling pass**: After Phase 1, show Sonnet what was found and ask "what did I miss?" Could rescue 2-5% recall but adds another Sonnet call (~$0.02/chunk).

---

*Draft: 2026-02-08*
*Based on: v2 GT (277 terms, 15 chunks), rebaseline results, exhaustive_sonnet results*
