# POC-1b: Untested Strategies & Missing Experiments

## Quick Reference: What Was Planned vs What Was Tested

### Strategies E, F, G, H, I Status

| Strategy | Code | Tested | Results | Notes |
|----------|------|--------|---------|-------|
| **E: Instructor Structured** | ✅ `run_experiment.py:351` | ❌ NO | None | Implemented but never executed |
| **F: Span Verification** | ✅ `run_experiment.py:366` | ❌ NO | None | Implemented but never executed |
| **G: Self-Consistency (N=10)** | ✅ `run_experiment.py:381` | ❌ NO | None | Implemented but never executed |
| **H: Multi-Pass (3 passes)** | ✅ `run_experiment.py:412` | ❌ NO | None | Implemented but never executed |
| **I: Combined (F+G+H)** | ✅ `run_experiment.py:446` | ❌ NO | None | Implemented but never executed |

---

## The Missing Main Experiment

### What Was Planned (SPEC.md Section 4.2)

```
45 chunks × 2 models × 5 strategies × 3 trials = 1,350 extractions
```

**Expected Output:** `artifacts/phase-3-raw-results.json`

### What Actually Happened

```bash
$ ls artifacts/phase*
# No results found
```

**The main experiment was NEVER RUN.**

---

## Specific Gaps

### 1. Self-Consistency Voting (Strategy G) - UNTESTED

**What was planned:**
- N=10 samples per chunk
- 70% agreement threshold
- Temperature 0.8 for diversity
- Expected: +25% precision, +15% recall

**What was implemented:**
```python
def strategy_g_self_consistency(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy G: Self-consistency voting with N=10 samples."""
    # Lines 381-409: Full implementation exists
    for i in range(SC_NUM_SAMPLES):  # SC_NUM_SAMPLES = 10
        response = call_llm(prompt, model=model, temperature=SC_TEMPERATURE, max_tokens=2000)
        # ... voting logic ...
    min_count = int(SC_NUM_SAMPLES * SC_AGREEMENT_THRESHOLD)  # 70% threshold
```

**What was tested:**
- ❌ NOTHING - Strategy G was never called in the main experiment
- ⚠️ Combined strategy (I) uses N=5 instead of N=10 (line 457)
- ⚠️ No ablation study on N values

**Missing evidence:**
- Does N=10 actually improve precision/recall?
- What's the optimal agreement threshold?
- How does temperature 0.8 affect diversity?
- Cost-benefit of 10 samples vs 5 vs 3?

---

### 2. Multi-Pass Extraction (Strategy H) - UNTESTED

**What was planned:**
- Pass 1: Initial extraction (deterministic, temp=0)
- Pass 2: "What did I miss?" (temp=0.5)
  - Focus on abbreviations, implicit refs, multi-word terms
- Pass 3: Category sweep (temp=0.5)
  - Resource types, API objects, commands, error states, config options
- Expected: +15-20% recall without increasing hallucination

**What was implemented:**
```python
def strategy_h_multi_pass(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy H: Multi-pass extraction with 'what did I miss?'"""
    # Lines 412-443: Full implementation with 3 passes
    
    # Pass 1: PROMPT_MULTI_PASS_1 (temp=0.3)
    # Pass 2: PROMPT_MULTI_PASS_2 (temp=0.5) - "What did I miss?"
    # Pass 3: PROMPT_MULTI_PASS_3 (temp=0.5) - Category sweep
```

**What was tested:**
- ❌ NOTHING - Strategy H was never called in the main experiment
- ⚠️ Pass 1 uses temp=0.3 instead of planned temp=0 (line 419)
- ⚠️ No measurement of recall improvement per pass
- ⚠️ No measurement of hallucination increase

**Missing evidence:**
- Does multi-pass actually improve recall?
- Which pass contributes most?
- Does hallucination increase with each pass?
- What's the cost-benefit vs single-pass?

---

### 3. Span Verification Effectiveness (Strategy F) - UNTESTED AT SCALE

**What was planned:**
- Pydantic field validators with `ValidationInfo.context`
- Exact substring matching in source text
- Reduce hallucination from 7.4% to <1%

**What was implemented:**
```python
class ExtractedTermVerified(BaseModel):
    """Extracted term with span verification."""
    
    @model_validator(mode="before")
    @classmethod
    def validate_span_exists(cls, values):
        """Verify span exists in source text."""
        # Lines 82-94: Simplified validation
        if span not in source_text:
            raise ValueError(f"Span not found in source text")
```

**What was tested:**
- ❌ NOTHING - No systematic comparison of E vs F
- ⚠️ Validation is simplified (not full Pydantic pattern from SPEC)
- ⚠️ No measurement of hallucination reduction
- ⚠️ No comparison of strict vs fuzzy matching

**Missing evidence:**
- How much does span verification reduce hallucination?
- What's the precision/recall tradeoff?
- Is exact matching too strict?
- Should we use fuzzy matching instead?

---

### 4. Combined Pipeline (Strategy I) - UNTESTED

**What was planned:**
- F (span verification) + G (self-consistency N=10) + H (multi-pass)
- Expected: 95%+ P, 85%+ R, <1% H

**What was implemented:**
```python
def strategy_i_combined(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy I: Combined pipeline (multi-pass + self-consistency + verification)."""
    # Lines 446-484: Full implementation
    
    # Step 1: Multi-pass for high recall
    multi_pass_terms, _ = strategy_h_multi_pass(chunk_text, model)
    
    # Step 2: Self-consistency voting (N=5, not N=10)
    for i in range(5):  # DIFFERENT FROM SPEC
        # ... voting logic ...
    
    # Step 3: Final span verification
    verified_terms = [t for t in combined_terms if t.lower() in chunk_text.lower()]
```

**What was tested:**
- ❌ NOTHING - Strategy I was never called in the main experiment
- ⚠️ Uses N=5 instead of N=10 (line 457)
- ⚠️ Uses 60% threshold instead of 70% (line 471)
- ⚠️ No measurement of combined effectiveness

**Missing evidence:**
- Does combining all techniques work?
- What's the optimal combination?
- Is N=5 sufficient or should we use N=10?
- What's the cost-benefit of combining?

---

## Hypotheses That Remain Unverified

**From SPEC.md Section 3.2:**

### H1: Instructor + Voting Achieves Targets
> LLM extraction with Instructor structured output + self-consistency voting (N=10, 70% agreement threshold) will achieve **95%+ precision, 85%+ recall, <1% hallucination**.

**Status:** ❌ UNTESTED
- No data on strategy G
- No data on strategy I
- No comparison with POC-1 baselines

### H2: Span Verification Reduces Hallucination
> Span verification validators will reduce hallucination from 7.4% to <1% while maintaining precision.

**Status:** ❌ UNTESTED
- No systematic comparison of E vs F
- No measurement of hallucination reduction
- No precision/recall tradeoff analysis

### H3: Multi-Pass Increases Recall
> Multi-pass extraction ("what did I miss?") will increase recall from 71% to 85%+ without increasing hallucination above 5%.

**Status:** ❌ UNTESTED
- No data on strategy H
- No measurement of recall improvement
- No measurement of hallucination increase

### H4: Self-Consistency Voting Achieves Targets
> Self-consistency voting with 70% agreement threshold will achieve 95%+ precision with <1% hallucination.

**Status:** ❌ UNTESTED
- No data on strategy G
- No comparison of agreement thresholds
- No measurement of precision/hallucination

### H5: Combined Techniques Achieve Target
> Combining all techniques will achieve the target: 95%+ P, 95%+ R, <1% H.

**Status:** ❌ UNTESTED
- No data on strategy I
- No comparison with individual strategies
- No verification of combined effectiveness

---

## Success Criteria Status

**From SPEC.md Section 6:**

### PASS Criteria (All Must Be True)

| Criterion | Target | Status |
|-----------|--------|--------|
| At least one strategy achieves P >95% | Required | ❌ UNKNOWN |
| At least one strategy achieves R >85% | Required | ❌ UNKNOWN |
| At least one strategy achieves H <1% | Required | ❌ UNKNOWN |
| Best config meets all three | Required | ❌ UNKNOWN |

### PARTIAL PASS Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| Best achieves P>90%, R>80%, H<3% | Acceptable | ❌ UNKNOWN |
| Significant improvement over POC-1 | +10% on any metric | ❌ UNKNOWN |

### FAIL Criteria

| Criterion | Target | Status |
|-----------|--------|--------|
| No strategy achieves P>85% | Below minimum | ❌ UNKNOWN |
| No strategy achieves H<5% | Unacceptable | ❌ UNKNOWN |
| No improvement over POC-1 | 0% gain | ❌ UNKNOWN |

**Overall:** CANNOT DETERMINE - experiment not executed.

---

## What WAS Tested Instead

The team created 11 ad-hoc test files exploring different approaches:

| File | Approach | Status |
|------|----------|--------|
| `test_small_chunk_extraction.py` | Small chunks (50-300 words) | ✅ 92% recall |
| `test_quote_extract_multipass.py` | Quote-based extraction | ✅ 78.7% P, 74.8% R |
| `test_quote_verify_approach.py` | Quote + verification | ✅ 40.8% P, 82.5% R |
| `test_high_recall_ensemble.py` | Ensemble voting | ✅ 51.5% P, 92% R |
| `test_hybrid_ner.py` | NER + LLM hybrid | ✅ Explored GLiNER |
| `test_discrimination_approach.py` | LLM discrimination | ✅ Ad-hoc testing |
| `test_advanced_strategies.py` | Quote-then-extract | ✅ Research-backed |
| `test_zero_hallucination.py` | Zero hallucination focus | ✅ Span verification |
| `test_pattern_plus_llm.py` | Pattern + LLM | ✅ Hybrid approach |
| `test_hybrid_final.py` | Final hybrid attempt | ✅ Combined approaches |
| `test_fast_combined.py` | Fast combined approach | ✅ Optimized pipeline |

**These are valuable explorations but NOT the planned systematic evaluation.**

---

## Key Findings from Ad-Hoc Testing

### Small Chunks Work Better
- **Finding:** 92% recall with small chunks (50-300 words)
- **Implication:** Chunk size matters more than extraction strategy?
- **Gap:** Not tested with strategies E-I

### Ensemble Achieves High Recall
- **Finding:** 92% recall with ensemble voting
- **Implication:** Diversity helps more than self-consistency?
- **Gap:** Not compared with strategy G (self-consistency)

### Quote-Based Extraction Balances Precision/Recall
- **Finding:** 78.7% precision, 74.8% recall
- **Implication:** Forcing quotes reduces hallucination?
- **Gap:** Not formalized as a strategy

### Span Verification Reduces True Hallucination
- **Finding:** True hallucination <5% with span verification
- **Implication:** Most "hallucinations" are valid terms not in ground truth?
- **Gap:** Not quantified in systematic experiment

---

## What's Needed to Complete POC-1b

### Immediate (To Execute Original Plan)

1. **Run the main experiment**
   ```bash
   python run_experiment.py
   ```
   - Generates `artifacts/phase-3-raw-results.json`
   - Tests all 1,350 conditions
   - Provides data for all hypotheses

2. **Analyze results**
   - Calculate metrics per strategy
   - Compare E vs F vs G vs H vs I
   - Verify hypotheses H1-H5

3. **Generate final report**
   - Create `artifacts/phase-4-final-metrics.json`
   - Document success/failure against criteria
   - Provide recommendations

### Short-Term (To Integrate Findings)

1. **Test strategies on small chunks**
   - Run E-I with 50-300 word chunks
   - Compare with full-document results
   - Measure chunk size impact

2. **Ablate self-consistency parameters**
   - Test N=3, 5, 7, 10, 15
   - Test thresholds: 50%, 60%, 70%, 80%
   - Find optimal configuration

3. **Ablate multi-pass configuration**
   - Test 1-pass vs 2-pass vs 3-pass
   - Measure recall improvement per pass
   - Measure hallucination increase per pass

4. **Formalize ensemble approach**
   - Compare ensemble vs self-consistency
   - Measure diversity benefits
   - Create Strategy J: Ensemble

### Medium-Term (To Production Readiness)

1. **Decide on chunking strategy**
   - Small chunks vs full documents
   - Semantic vs fixed-size chunking
   - Overlap and deduplication

2. **Choose extraction approach**
   - E, F, G, H, I, or ensemble?
   - Cost-benefit analysis
   - Latency requirements

3. **Set hallucination threshold**
   - Define acceptable rate
   - True hallucination vs valid terms
   - Human review queue sizing

4. **Implement in RAG pipeline**
   - Integrate best approach
   - Measure end-to-end performance
   - Monitor in production

---

## Summary

| Aspect | Status | Evidence |
|--------|--------|----------|
| **Strategies E-I implemented** | ✅ YES | Code in `run_experiment.py` |
| **Strategies E-I tested** | ❌ NO | No `phase-3-raw-results.json` |
| **Self-consistency voting tested** | ❌ NO | Strategy G never executed |
| **Multi-pass extraction tested** | ❌ NO | Strategy H never executed |
| **Span verification tested** | ❌ NO | No E vs F comparison |
| **Combined pipeline tested** | ❌ NO | Strategy I never executed |
| **Hypotheses H1-H5 verified** | ❌ NO | No data to verify |
| **Success criteria met** | ❌ UNKNOWN | Experiment not run |
| **Ad-hoc exploration** | ✅ YES | 11 test files created |
| **Promising findings** | ✅ YES | Small chunks, ensemble, quotes |

**POC-1b is 50% complete: Code exists but main experiment was never executed.**

---

*Document Generated: 2026-02-05*
*POC-1b Status: PARTIAL IMPLEMENTATION*
