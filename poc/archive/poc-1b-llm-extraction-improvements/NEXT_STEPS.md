# POC-1b: Next Steps to Complete the Experiment

## Current State

✅ **Code is complete** - All 5 strategies (E, F, G, H, I) are implemented in `run_experiment.py`
❌ **Experiment is not run** - Main experiment (1,350 conditions) was never executed
❌ **Results are missing** - No `phase-3-raw-results.json` or `phase-4-final-metrics.json`
❌ **Hypotheses are unverified** - H1-H5 remain untested

---

## Option 1: Complete the Original Plan (Recommended)

### Step 1: Execute the Main Experiment

**Command:**
```bash
cd /home/susano/Code/personal-library-manager/poc/poc-1b-llm-extraction-improvements
python run_experiment.py
```

**What it does:**
- Tests all 5 strategies (E, F, G, H, I)
- On all 45 chunks from POC-1
- With 2 models (Haiku, Sonnet)
- 3 trials each
- Total: 1,350 extractions

**Expected output:**
- `artifacts/phase-3-raw-results.json` - Raw results with all metrics
- Console output with progress and summary

**Time estimate:** 3-4 hours (with rate limiting)

**Cost estimate:** ~$50-100 in API calls (10 samples × 45 chunks × 2 models)

### Step 2: Analyze Results

**Create analysis script:**
```python
# analyze_results.py
import json
from pathlib import Path

results_path = Path("artifacts/phase-3-raw-results.json")
with open(results_path) as f:
    data = json.load(f)

# Group by strategy
for strategy_id in ["E", "F", "G", "H", "I"]:
    strategy_results = [r for r in data["results"] 
                       if r.get("strategy") == strategy_id and "metrics" in r]
    
    if strategy_results:
        avg_p = sum(r["metrics"]["precision"] for r in strategy_results) / len(strategy_results)
        avg_r = sum(r["metrics"]["recall"] for r in strategy_results) / len(strategy_results)
        avg_h = sum(r["metrics"]["hallucination_rate"] for r in strategy_results) / len(strategy_results)
        
        print(f"{strategy_id}: P={avg_p:.1%} R={avg_r:.1%} H={avg_h:.1%}")
```

### Step 3: Verify Hypotheses

**Check each hypothesis:**

| Hypothesis | Check | Pass Criteria |
|-----------|-------|---------------|
| H1 | Strategy I: P, R, H | P>95%, R>85%, H<1% |
| H2 | F vs E: H reduction | H(F) < H(E) and H(F) < 1% |
| H3 | H vs E: R improvement | R(H) > R(E) + 15% |
| H4 | G: P and H | P>95%, H<1% |
| H5 | I vs E,F,G,H | I meets all targets |

### Step 4: Generate Final Report

**Update RESULTS.md with:**
- Actual results from phase-3
- Hypothesis verdicts
- Comparison with POC-1 baselines
- Recommendations for production

---

## Option 2: Focused Testing (Faster, Lower Cost)

If full experiment is too expensive, test key strategies:

### Variant A: Test Self-Consistency Only

**Focus:** Does N=10 voting work?

```bash
# Modify run_experiment.py to test only strategy G
STRATEGIES = {
    "G": ("Self-Consistency", strategy_g_self_consistency),
}
MODELS = ["claude-haiku"]  # Single model
TRIALS = 1  # Single trial
# Result: 45 chunks × 1 model × 1 strategy × 1 trial = 45 extractions
```

**Time:** 30 minutes
**Cost:** ~$5

### Variant B: Test Multi-Pass Only

**Focus:** Does 3-pass extraction improve recall?

```bash
STRATEGIES = {
    "H": ("Multi-Pass", strategy_h_multi_pass),
}
MODELS = ["claude-haiku"]
TRIALS = 1
# Result: 45 extractions
```

**Time:** 30 minutes
**Cost:** ~$5

### Variant C: Test Combined Only

**Focus:** Does combining all techniques work?

```bash
STRATEGIES = {
    "I": ("Combined", strategy_i_combined),
}
MODELS = ["claude-haiku"]
TRIALS = 1
# Result: 45 extractions
```

**Time:** 1 hour (multi-pass + voting)
**Cost:** ~$10

### Variant D: Test on Subset

**Focus:** Quick validation before full run

```bash
# Modify run_experiment.py to test on first 10 chunks only
chunks = ground_truth["chunks"][:10]  # First 10 only
MODELS = ["claude-haiku"]
TRIALS = 1
# Result: 10 chunks × 1 model × 5 strategies × 1 trial = 50 extractions
```

**Time:** 15 minutes
**Cost:** ~$2

---

## Option 3: Integrate Ad-Hoc Findings (Alternative)

If you want to leverage the exploratory work already done:

### Step 1: Formalize Small-Chunk Approach

**Create Strategy J: Small Chunks**
```python
def strategy_j_small_chunks(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy J: Optimized for small chunks (50-300 words)."""
    # Use ensemble of simple + quote + exhaustive
    # From test_small_chunk_extraction.py
    pass
```

### Step 2: Formalize Ensemble Approach

**Create Strategy K: Ensemble Voting**
```python
def strategy_k_ensemble(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy K: Ensemble of multiple extraction methods."""
    # Union of simple + quote + exhaustive
    # From test_high_recall_ensemble.py
    pass
```

### Step 3: Formalize Quote-Based Approach

**Create Strategy L: Quote-Then-Extract**
```python
def strategy_l_quote_extract(chunk_text: str, model: str) -> tuple[list[str], float]:
    """Strategy L: Force LLM to quote source before extracting."""
    # From test_quote_extract_multipass.py
    pass
```

### Step 4: Run Comparative Experiment

```bash
# Test E, F, G, H, I, J, K, L
STRATEGIES = {
    "E": ("Structured Basic", strategy_e_structured),
    "F": ("Span Verification", strategy_f_span_verify),
    "G": ("Self-Consistency", strategy_g_self_consistency),
    "H": ("Multi-Pass", strategy_h_multi_pass),
    "I": ("Combined", strategy_i_combined),
    "J": ("Small Chunks", strategy_j_small_chunks),
    "K": ("Ensemble", strategy_k_ensemble),
    "L": ("Quote-Extract", strategy_l_quote_extract),
}
```

---

## Option 4: Hybrid Approach (Recommended for Speed)

### Phase 1: Quick Validation (1 hour, $5)

Test on first 10 chunks with all strategies:
```python
chunks = ground_truth["chunks"][:10]
MODELS = ["claude-haiku"]
TRIALS = 1
# 10 × 1 × 5 × 1 = 50 extractions
```

**Deliverable:** Quick metrics to validate approach

### Phase 2: Full Experiment (3-4 hours, $50-100)

If Phase 1 looks good, run full experiment:
```python
chunks = ground_truth["chunks"]  # All 45
MODELS = ["claude-haiku", "claude-sonnet"]
TRIALS = 3
# 45 × 2 × 5 × 3 = 1,350 extractions
```

**Deliverable:** Complete results with all hypotheses verified

---

## Recommended Path Forward

### For Immediate Completion (Next 4 hours)

1. **Run Phase 1 validation** (10 chunks, 1 model, 1 trial)
   - Verify code works
   - Check API connectivity
   - Estimate full cost

2. **If Phase 1 succeeds, run full experiment**
   - All 45 chunks
   - Both models
   - 3 trials each

3. **Analyze results**
   - Calculate metrics per strategy
   - Verify hypotheses H1-H5
   - Compare with POC-1 baselines

4. **Update RESULTS.md**
   - Document findings
   - Provide recommendations
   - Identify next steps

### For Production Readiness (Next 1-2 weeks)

1. **Integrate promising findings**
   - Small-chunk approach
   - Ensemble voting
   - Quote-based extraction

2. **Ablate key parameters**
   - Self-consistency: N=3,5,7,10,15
   - Multi-pass: 1,2,3 passes
   - Span verification: strict vs fuzzy

3. **Test on production data**
   - Real Kubernetes documentation
   - Measure end-to-end performance
   - Validate hallucination rate

4. **Implement in RAG pipeline**
   - Integrate best strategy
   - Monitor performance
   - Iterate based on feedback

---

## Troubleshooting

### If experiment fails to run

**Check 1: Dependencies**
```bash
cd poc/poc-1b-llm-extraction-improvements
uv sync
source .venv/bin/activate
python -c "import pydantic; import rapidfuzz; print('OK')"
```

**Check 2: Ground truth**
```bash
ls ../poc-1-llm-extraction-guardrails/artifacts/phase-2-ground-truth.json
# Should exist and be valid JSON
```

**Check 3: LLM provider**
```bash
python -c "from utils.llm_provider import call_llm; print(call_llm('Say OK', model='claude-haiku'))"
# Should return 'OK' or similar
```

**Check 4: API credentials**
```bash
echo $ANTHROPIC_API_KEY
# Should be set
```

### If experiment is too slow

**Reduce scope:**
```python
# Option 1: Fewer chunks
chunks = ground_truth["chunks"][:20]  # First 20 instead of 45

# Option 2: Fewer models
MODELS = ["claude-haiku"]  # Just Haiku, not Sonnet

# Option 3: Fewer trials
TRIALS = 1  # Just 1 trial instead of 3

# Option 4: Fewer strategies
STRATEGIES = {
    "E": ("Structured Basic", strategy_e_structured),
    "I": ("Combined", strategy_i_combined),
}
```

### If experiment is too expensive

**Reduce cost:**
```python
# Option 1: Use cheaper model
MODELS = ["claude-haiku"]  # Haiku is 10x cheaper than Sonnet

# Option 2: Reduce samples for self-consistency
SC_NUM_SAMPLES = 5  # Instead of 10

# Option 3: Reduce trials
TRIALS = 1  # Instead of 3

# Option 4: Test subset first
chunks = ground_truth["chunks"][:10]  # First 10 chunks
```

---

## Success Criteria

### Phase 1 (Validation)
- ✅ Code runs without errors
- ✅ All 5 strategies execute
- ✅ Metrics are calculated
- ✅ Results are saved

### Phase 2 (Full Experiment)
- ✅ All 1,350 conditions complete
- ✅ Results saved to `phase-3-raw-results.json`
- ✅ Metrics calculated per strategy
- ✅ Hypotheses H1-H5 verified

### Phase 3 (Analysis)
- ✅ At least one strategy achieves P>90%
- ✅ At least one strategy achieves R>80%
- ✅ At least one strategy achieves H<5%
- ✅ Comparison with POC-1 baselines
- ✅ Recommendations for production

---

## Timeline

| Phase | Task | Time | Cost |
|-------|------|------|------|
| 1 | Validation (10 chunks) | 15 min | $2 |
| 2 | Full experiment | 3-4 hrs | $50-100 |
| 3 | Analysis | 1 hr | $0 |
| 4 | Report | 1 hr | $0 |
| **Total** | | **5-6 hrs** | **$52-102** |

---

## Files to Update After Completion

1. **RESULTS.md** - Update with actual results
2. **GAP_ANALYSIS.md** - Mark experiments as complete
3. **UNTESTED_STRATEGIES.md** - Update status
4. **README.md** - Update with findings
5. **SPEC.md** - Document any deviations

---

## Questions to Answer

After running the experiment, answer these:

1. **Does self-consistency voting work?**
   - Compare G vs E: Does voting improve precision/recall?
   - What's the optimal N value?
   - What's the optimal agreement threshold?

2. **Does multi-pass extraction work?**
   - Compare H vs E: Does multi-pass improve recall?
   - Which pass contributes most?
   - Does hallucination increase?

3. **Does span verification work?**
   - Compare F vs E: How much does it reduce hallucination?
   - What's the precision/recall tradeoff?
   - Is exact matching too strict?

4. **Does combining all techniques work?**
   - Compare I vs E,F,G,H: Does combination improve all metrics?
   - Is the combination better than individual strategies?
   - What's the cost-benefit?

5. **How do we compare with POC-1?**
   - Is there improvement over POC-1 baselines?
   - Which strategy is best?
   - What are the recommendations for production?

---

*Next Steps Document Generated: 2026-02-05*
*POC-1b Status: Ready for execution*
