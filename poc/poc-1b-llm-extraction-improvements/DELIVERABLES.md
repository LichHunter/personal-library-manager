# Test Patterns Analysis - Deliverables

## Summary

Comprehensive analysis of existing test file patterns in POC-1b to provide a template for creating new test files that match the codebase style.

## Documents Created

### 1. **TEST_PATTERNS_ANALYSIS.md** (30 KB) - MAIN DOCUMENT
Complete reference guide covering:
- **Test Structure Patterns**: File organization, imports, initialization
- **Ground Truth Loading**: Format, validation, structure
- **Shared Utilities**: LLM provider, parsing, metrics
- **Span Verification**: Hallucination filtering approach
- **Metrics Calculation**: Precision, recall, hallucination, F1
- **Prompt Patterns**: Extraction and verification prompts
- **Strategy Implementation**: Single, ensemble, voting patterns
- **Main Experiment Loop**: Per-chunk and aggregate evaluation
- **Output & Logging**: Console format and JSON results
- **Results File Format**: JSON structure for saving results
- **Complete Template**: Ready-to-use test file template
- **Summary Table**: Quick reference of all patterns

### 2. **QUICK_REFERENCE.md** (7.4 KB) - CHEAT SHEET
Fast lookup guide with:
- File structure checklist
- Essential functions (copy-paste ready)
- Common prompts
- Strategy patterns
- Key imports and constants
- LLM call pattern
- Metrics interpretation
- Target thresholds
- File locations

## Key Findings

### Test File Structure
```
1. Shebang + Docstring
2. Imports (stdlib → third-party → local)
3. Path setup + print header
4. load_ground_truth()
5. Parsing utilities
6. Prompts (CONSTANTS)
7. Extraction functions
8. Metrics functions
9. Strategy functions
10. run_experiment() main loop
11. if __name__ == "__main__"
```

### Common Patterns Identified

#### 1. Ground Truth Format
```json
{
  "chunks": [
    {
      "chunk_id": "...",
      "content": "...",
      "terms": [{"term": "...", "tier": 1}],
      "term_count": N
    }
  ]
}
```

#### 2. Shared Utilities
- **LLM Provider**: `call_llm(prompt, model, temperature, max_tokens)`
- **Parsing**: Custom `parse_terms()` handling markdown code blocks
- **Verification**: `strict_span_verify()` filters hallucinations
- **Metrics**: Bipartite matching for TP/FP/FN calculation

#### 3. Extraction Strategies
- **Simple**: Single prompt extraction
- **Exhaustive**: High-recall extraction with detailed instructions
- **Conservative**: High-precision extraction with strict criteria
- **Quote**: Extraction with exact quote locations
- **Ensemble**: Multiple extractions + LLM verification
- **Voting**: Multiple strategies with consensus threshold

#### 4. Metrics Calculation
- **Precision**: TP / (TP + FP) - % of extracted terms correct
- **Recall**: TP / (TP + FN) - % of ground truth found
- **Hallucination**: FP / extracted - % of false positives
- **F1**: Harmonic mean of precision and recall

#### 5. Output Format
- Per-chunk metrics with visual markers (✓/~/space)
- Aggregate table with averages
- Target check section (95%+ recall, <10% hallucination)
- JSON results file for analysis

### Prompt Patterns

#### High Recall (Exhaustive)
```
Extract ALL technical terms...
Be EXHAUSTIVE. Include: resources, components, concepts, feature gates...
```

#### High Precision (Conservative)
```
Extract ONLY the most important Kubernetes technical terms.
Be CONSERVATIVE. Only core resources, key components, essential concepts.
```

#### Verification
```
Filter this list of extracted terms...
Keep ONLY Kubernetes-specific technical terms. Remove generic English words.
```

### Span Verification Strategy
Filters hallucinations by checking terms exist in source content:
1. Exact match (case-insensitive)
2. Underscore/hyphen normalization
3. CamelCase expansion
4. Minimum 2-character length

### Metrics Matching
Bipartite matching algorithm:
1. Exact normalization match
2. Fuzzy match (85%+ similarity via rapidfuzz)
3. Token overlap (80%+ of ground truth tokens present)

## Files Analyzed

### Source Files
- `test_fast_combined.py` (454 lines)
  - 10 strategies tested
  - Focus on combined approaches
  - Results: ensemble_verified (88.9% recall, 10.7% hallucination)

- `test_hybrid_final.py` (507 lines)
  - 8 strategies tested
  - Focus on hybrid approaches
  - Results: ensemble_sonnet_verify (high precision)

### Shared Utilities
- `utils/llm_provider.py` (348 lines)
  - Anthropic OAuth provider
  - Automatic token refresh
  - Exponential backoff for rate limiting
  - Model aliases (haiku, sonnet, opus)

### Ground Truth
- `artifacts/small_chunk_ground_truth.json`
  - 15 chunks
  - 163 total terms
  - Kubernetes documentation

## Template Usage

### Step 1: Copy Template
Use the complete template from TEST_PATTERNS_ANALYSIS.md section 10

### Step 2: Customize
- Change test name in header and print statement
- Define extraction prompts for your strategy
- Implement extraction functions
- Define strategies dict with your approaches

### Step 3: Run
```bash
python test_sentence_extraction.py
```

### Step 4: Check Results
- Console output shows per-chunk and aggregate metrics
- JSON file saved to `artifacts/test_results.json`
- Compare against targets: 95%+ recall, <10% hallucination

## Key Insights

### What Works Well
1. **Ensemble + Verification**: High recall (88.9%) with reasonable hallucination (10.7%)
2. **Voting Strategies**: High precision (75%+) with low hallucination (2.2%)
3. **Span Verification**: Effective at filtering hallucinations
4. **Multi-model Approaches**: Combining Haiku and Sonnet improves results

### What Needs Improvement
1. **Recall vs Hallucination Trade-off**: Hard to achieve both 95%+ recall AND <10% hallucination
2. **Generic Word Filtering**: Verification prompts struggle with borderline terms
3. **Model Consistency**: Haiku and Sonnet have different strengths

### Best Practices
1. Always apply `strict_span_verify()` as final filter
2. Use `temperature=0` for deterministic extraction
3. Truncate content to 2500 chars to fit token limits
4. Use bipartite matching for metrics (each term matches at most once)
5. Track both per-chunk and aggregate metrics
6. Save results to JSON for analysis

## Next Steps for New Test File

1. **Copy template** from TEST_PATTERNS_ANALYSIS.md
2. **Define your strategy** (extraction approach)
3. **Create prompts** for your specific approach
4. **Implement extraction functions** following patterns
5. **Run experiment** on 5-10 chunks first
6. **Analyze results** against targets
7. **Iterate** based on findings

## Files to Reference

- **Ground Truth**: `artifacts/small_chunk_ground_truth.json`
- **LLM Provider**: `../poc-1-llm-extraction-guardrails/utils/llm_provider.py`
- **Example Tests**: `test_fast_combined.py`, `test_hybrid_final.py`
- **Results**: `artifacts/fast_combined_results.json`, `artifacts/hybrid_final_results.json`

## Document Map

```
TEST_PATTERNS_ANALYSIS.md (30 KB)
├── 1. Test Structure Patterns
├── 2. Ground Truth Loading & Validation
├── 3. Shared Utilities
├── 4. Span Verification Pattern
├── 5. Metrics Calculation Pattern
├── 6. Prompt Patterns
├── 7. Strategy Implementation Pattern
├── 8. Main Experiment Loop Pattern
├── 9. Output & Logging Format
├── 10. Results File Format
├── Template for New Test File
└── Summary of Key Patterns

QUICK_REFERENCE.md (7.4 KB)
├── File Structure Checklist
├── Essential Functions (copy-paste)
├── Common Prompts
├── Strategy Patterns
├── Key Imports & Constants
├── LLM Call Pattern
├── Metrics Interpretation
├── Target Thresholds
└── File Locations
```

## Success Criteria

✅ **Completed**:
- Analyzed 2 existing test files (test_fast_combined.py, test_hybrid_final.py)
- Extracted 10 common patterns
- Identified shared utilities (llm_provider, metrics, parsing)
- Documented ground truth loading and validation
- Created complete template structure
- Provided quick reference guide
- Included copy-paste ready code snippets

## Ready to Use

Both documents are ready for immediate use:
1. **TEST_PATTERNS_ANALYSIS.md**: Comprehensive reference for understanding patterns
2. **QUICK_REFERENCE.md**: Fast lookup for common code snippets

Start with QUICK_REFERENCE.md for quick answers, then refer to TEST_PATTERNS_ANALYSIS.md for detailed explanations.
