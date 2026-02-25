# Test Patterns Documentation Index

## 📋 Quick Navigation

### For Quick Answers
👉 **Start here**: [QUICK_REFERENCE.md](QUICK_REFERENCE.md) (7.4 KB)
- Copy-paste ready code snippets
- Essential functions
- Common prompts
- Key constants and patterns

### For Complete Understanding
👉 **Read this**: [TEST_PATTERNS_ANALYSIS.md](TEST_PATTERNS_ANALYSIS.md) (30 KB)
- Detailed explanation of each pattern
- Ground truth format and loading
- Shared utilities documentation
- Complete template with comments
- Summary table of all patterns

### For Project Overview
👉 **Check this**: [DELIVERABLES.md](DELIVERABLES.md) (8 KB)
- What was analyzed
- Key findings summary
- Files analyzed
- Next steps for new test file

---

## 📚 Document Contents

### TEST_PATTERNS_ANALYSIS.md (959 lines)

**Sections:**
1. **Test Structure Patterns** - File organization and imports
2. **Ground Truth Loading & Validation** - Format and structure
3. **Shared Utilities** - LLM provider, parsing, metrics
4. **Span Verification Pattern** - Hallucination filtering
5. **Metrics Calculation Pattern** - Precision, recall, F1
6. **Prompt Patterns** - Extraction and verification prompts
7. **Strategy Implementation Pattern** - Single, ensemble, voting
8. **Main Experiment Loop Pattern** - Per-chunk and aggregate evaluation
9. **Output & Logging Format** - Console and JSON output
10. **Results File Format** - JSON structure
11. **Template for New Test File** - Ready-to-use template
12. **Summary of Key Patterns** - Quick reference table

### QUICK_REFERENCE.md (230 lines)

**Sections:**
- File Structure (checklist)
- Essential Functions (copy-paste ready)
- Common Prompts (high recall, high precision, verification)
- Strategy Patterns (single, ensemble, voting)
- Key Imports
- Key Constants
- LLM Call Pattern
- Metrics Interpretation
- Target Thresholds
- File Locations

### DELIVERABLES.md (259 lines)

**Sections:**
- Summary of analysis
- Documents created
- Key findings
- Files analyzed
- Template usage steps
- Key insights
- Next steps
- Document map

---

## 🎯 Use Cases

### "I need to create a new test file"
1. Read: QUICK_REFERENCE.md (5 min)
2. Copy: Template from TEST_PATTERNS_ANALYSIS.md section 10
3. Customize: Change test name, prompts, strategies
4. Run: `python test_sentence_extraction.py`

### "I want to understand the metrics"
1. Read: TEST_PATTERNS_ANALYSIS.md section 5
2. Reference: QUICK_REFERENCE.md "Metrics Interpretation"
3. Check: Example results in `artifacts/fast_combined_results.json`

### "I need to implement a new strategy"
1. Read: TEST_PATTERNS_ANALYSIS.md section 7
2. Copy: Strategy pattern from QUICK_REFERENCE.md
3. Implement: Your extraction logic
4. Test: Run experiment on 5 chunks first

### "I want to understand the test structure"
1. Read: TEST_PATTERNS_ANALYSIS.md section 1
2. Reference: QUICK_REFERENCE.md "File Structure"
3. Compare: With existing test files

### "I need to write verification prompts"
1. Read: TEST_PATTERNS_ANALYSIS.md section 6
2. Copy: Prompt templates from QUICK_REFERENCE.md
3. Customize: For your specific needs

---

## 📊 Pattern Summary

### Test File Structure
```
Shebang + Docstring
    ↓
Imports
    ↓
Constants (paths, prompts)
    ↓
Utility Functions (parse, verify)
    ↓
Extraction Functions
    ↓
Metrics Functions
    ↓
Strategy Functions
    ↓
Main Experiment Loop
    ↓
if __name__ == "__main__"
```

### Key Patterns

| Pattern | Location | Purpose |
|---------|----------|---------|
| Ground Truth Loading | Section 2 | Load and validate test data |
| Span Verification | Section 4 | Filter hallucinations |
| Metrics Calculation | Section 5 | Compute precision/recall/F1 |
| Prompt Templates | Section 6 | LLM extraction instructions |
| Strategy Implementation | Section 7 | Define extraction approaches |
| Experiment Loop | Section 8 | Run and aggregate results |
| Output Format | Section 9 | Console and JSON output |

---

## 🔍 Files Analyzed

### Source Test Files
- **test_fast_combined.py** (454 lines)
  - 10 strategies tested
  - Best result: ensemble_verified (88.9% recall, 10.7% hallucination)

- **test_hybrid_final.py** (507 lines)
  - 8 strategies tested
  - Focus on hybrid approaches

### Shared Utilities
- **utils/llm_provider.py** (348 lines)
  - Anthropic OAuth provider
  - Automatic token refresh
  - Rate limiting with exponential backoff

### Ground Truth
- **artifacts/small_chunk_ground_truth.json**
  - 15 chunks
  - 163 total terms
  - Kubernetes documentation

---

## 💡 Key Insights

### What Works Well
✅ Ensemble + Verification (88.9% recall)
✅ Voting strategies (high precision)
✅ Span verification (filters hallucinations)
✅ Multi-model approaches (Haiku + Sonnet)

### Challenges
⚠️ Recall vs hallucination trade-off
⚠️ Generic word filtering
⚠️ Model consistency differences

### Best Practices
1. Always apply `strict_span_verify()` as final filter
2. Use `temperature=0` for deterministic extraction
3. Truncate content to 2500 chars
4. Use bipartite matching for metrics
5. Track both per-chunk and aggregate metrics
6. Save results to JSON for analysis

---

## 🚀 Getting Started

### Step 1: Understand the Patterns
```bash
# Read the quick reference (5 minutes)
cat QUICK_REFERENCE.md

# Read the full analysis (30 minutes)
cat TEST_PATTERNS_ANALYSIS.md
```

### Step 2: Copy the Template
```bash
# Copy template from TEST_PATTERNS_ANALYSIS.md section 10
# Or use the template in QUICK_REFERENCE.md
```

### Step 3: Customize for Your Strategy
```python
# Change test name
# Define your prompts
# Implement extraction functions
# Define strategies dict
```

### Step 4: Run the Test
```bash
python test_sentence_extraction.py
```

### Step 5: Check Results
```bash
# Console output shows metrics
# JSON file: artifacts/test_results.json
# Compare against targets: 95%+ recall, <10% hallucination
```

---

## 📖 Reading Guide

### For Beginners
1. QUICK_REFERENCE.md (overview)
2. TEST_PATTERNS_ANALYSIS.md sections 1-5 (basics)
3. Copy template and customize

### For Intermediate Users
1. QUICK_REFERENCE.md (refresh)
2. TEST_PATTERNS_ANALYSIS.md sections 6-8 (strategies)
3. Implement custom strategy

### For Advanced Users
1. TEST_PATTERNS_ANALYSIS.md sections 9-12 (advanced)
2. Analyze existing results
3. Design novel strategies

---

## 🔗 Related Files

### Ground Truth
- `artifacts/small_chunk_ground_truth.json` - Test data

### Shared Utilities
- `../poc-1-llm-extraction-guardrails/utils/llm_provider.py` - LLM provider

### Example Tests
- `test_fast_combined.py` - Combined strategies
- `test_hybrid_final.py` - Hybrid strategies

### Results
- `artifacts/fast_combined_results.json` - Combined results
- `artifacts/hybrid_final_results.json` - Hybrid results

---

## ✅ Checklist for New Test File

- [ ] Read QUICK_REFERENCE.md
- [ ] Copy template from TEST_PATTERNS_ANALYSIS.md
- [ ] Define test name and docstring
- [ ] Create extraction prompts
- [ ] Implement extraction functions
- [ ] Define strategies dict
- [ ] Run on 5 chunks first
- [ ] Check console output
- [ ] Verify JSON results file
- [ ] Compare against targets
- [ ] Iterate based on results

---

## 📞 Questions?

Refer to:
- **"How do I...?"** → QUICK_REFERENCE.md
- **"Why does...?"** → TEST_PATTERNS_ANALYSIS.md
- **"What was analyzed?"** → DELIVERABLES.md

---

**Last Updated**: 2026-02-05
**Status**: Complete and ready to use
**Files**: 3 documents, 1448 lines total
