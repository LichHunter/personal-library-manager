# POC-1 Results: LLM Term Extraction Guardrails

> **Instructions**: Fill ALL sections during POC execution. No section should contain "TBD" or be left empty in the final version.

---

## Execution Summary

| Attribute | Value |
|-----------|-------|
| **Started** | {ISO 8601 timestamp} |
| **Completed** | {ISO 8601 timestamp} |
| **Duration** | {hours/days} |
| **Executor** | {human/model name} |
| **Status** | PASS / PARTIAL / FAIL |

---

## Hypothesis Verdict

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| **H1**: LLM extraction with full guardrails will achieve >80% precision and <5% hallucination | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence with numbers} |
| **H2**: Evidence citation reduces hallucination by >50% vs baseline | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence with numbers} |
| **H3**: Sonnet outperforms Haiku by >10%, Haiku still meets 80% | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence with numbers} |
| **H4**: Local models achieve >70% precision | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence with numbers} |

---

## Primary Metrics

### Best Configuration

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| Precision | >80% | {value}% | PASS/FAIL |
| Recall | >60% | {value}% | PASS/FAIL |
| Hallucination Rate | <5% | {value}% | PASS/FAIL |

**Best Model**: {model name}
**Best Prompt Variant**: {A/B/C/D}

### Results by Model (Prompt Variant D - Full Guardrails)

| Model | Precision | Recall | Hallucination | Latency (ms) |
|-------|-----------|--------|---------------|--------------|
| Claude Sonnet | {value}% | {value}% | {value}% | {value} |
| Claude Haiku | {value}% | {value}% | {value}% | {value} |
| Llama 3 8B | {value}% | {value}% | {value}% | {value} |
| Mistral 7B | {value}% | {value}% | {value}% | {value} |

### Results by Prompt Variant (Best Model)

| Variant | Precision | Recall | Hallucination | Description |
|---------|-----------|--------|---------------|-------------|
| A (Baseline) | {value}% | {value}% | {value}% | No guardrails |
| B (Evidence) | {value}% | {value}% | {value}% | Must cite spans |
| C (Constrained) | {value}% | {value}% | {value}% | Max 15, confidence |
| D (Full) | {value}% | {value}% | {value}% | Evidence + constraints |

---

## Secondary Metrics

| Metric | Value | Observation |
|--------|-------|-------------|
| Exact Match Precision | {value}% | {what this tells us} |
| Groundedness Score | {value}% | {what this tells us} |
| Confidence Calibration | {correlation} | {what this tells us} |
| Avg Tokens/Extraction | {value} | {cost implications} |

---

## Test Case Results

| TC | Name | Status | Key Finding |
|----|------|--------|-------------|
| TC-1 | Baseline Extraction | PASS/FAIL | {finding} |
| TC-2 | Evidence-Required | PASS/FAIL | {finding} |
| TC-3 | Constrained Output | PASS/FAIL | {finding} |
| TC-4 | Full Guardrails | PASS/FAIL | {finding} |
| TC-5 | Cross-Model Comparison | PASS/FAIL | {finding} |
| TC-6 | Variance Analysis | PASS/FAIL | {finding} |

---

## Phase Completion

| Phase | Status | Artifact | Notes |
|-------|--------|----------|-------|
| 1: Environment Setup | COMPLETE/INCOMPLETE | `artifacts/phase-1-*.json` | {any issues} |
| 2: Ground Truth | COMPLETE/INCOMPLETE | `artifacts/phase-2-*.json` | {any issues} |
| 3: Harness Implementation | COMPLETE/INCOMPLETE | `artifacts/phase-3-*.json` | {any issues} |
| 4: Experiment Execution | COMPLETE/INCOMPLETE | `artifacts/phase-4-*.json` | {any issues} |
| 5: Analysis | COMPLETE/INCOMPLETE | `artifacts/phase-5-*.json` | {any issues} |

---

## Statistical Analysis

### Model Comparisons

| Comparison | t-statistic | p-value | Significant? |
|------------|-------------|---------|--------------|
| Sonnet vs Haiku (Precision) | {value} | {value} | Yes/No |
| Sonnet vs Llama 3 (Precision) | {value} | {value} | Yes/No |
| Guardrails (D) vs Baseline (A) | {value} | {value} | Yes/No |

### Variance Analysis

| Model | Precision SD | Recall SD | Consistent? |
|-------|--------------|-----------|-------------|
| Claude Sonnet | {value} | {value} | Yes/No |
| Claude Haiku | {value} | {value} | Yes/No |
| Llama 3 8B | {value} | {value} | Yes/No |
| Mistral 7B | {value} | {value} | Yes/No |

---

## Key Findings

### Finding 1: {Title}

{Description with supporting data. Include specific numbers and examples.}

### Finding 2: {Title}

{Description with supporting data. Include specific numbers and examples.}

### Finding 3: {Title}

{Description with supporting data. Include specific numbers and examples.}

---

## Surprising Results

{Document anything unexpected, whether positive or negative. Include:}
- What was expected
- What actually happened
- Possible explanations

---

## Failure Analysis

### Terms Most Frequently Missed (False Negatives)

| Term | Ground Truth Tier | Miss Rate | Possible Reason |
|------|-------------------|-----------|-----------------|
| {term} | {tier} | {%} | {reason} |

### Terms Most Frequently Hallucinated (False Positives)

| Hallucinated Term | Frequency | Category | Possible Reason |
|-------------------|-----------|----------|-----------------|
| {term} | {count} | Fabrication/Drift/Partial | {reason} |

---

## Limitations

{What this POC did NOT prove or adequately test:}

1. {Limitation 1}
2. {Limitation 2}
3. {Limitation 3}

---

## Recommendations

### For RAG Pipeline Architecture

| Recommendation | Rationale | Impact |
|----------------|-----------|--------|
| {recommendation 1} | {based on finding X} | {what changes in architecture} |
| {recommendation 2} | {based on finding Y} | {what changes in architecture} |

### For Subsequent POCs

| POC | Recommendation | Reason |
|-----|----------------|--------|
| POC-2 (Confidence) | {recommendation} | {based on this POC's findings} |
| POC-6 (GLiNER) | {recommendation} | {based on this POC's findings} |
| POC-1b (Synonyms) | {recommendation} | {based on this POC's findings} |

### For Production Implementation

| Recommendation | Priority | Effort |
|----------------|----------|--------|
| {recommendation 1} | High/Medium/Low | {estimate} |
| {recommendation 2} | High/Medium/Low | {estimate} |

---

## Cost Analysis

| Model | Extractions | Input Tokens | Output Tokens | Est. Cost |
|-------|-------------|--------------|---------------|-----------|
| Claude Sonnet | 600 | {total} | {total} | ${value} |
| Claude Haiku | 600 | {total} | {total} | ${value} |
| Claude Opus (GT) | 100 | {total} | {total} | ${value} |
| **Total** | - | - | - | **${value}** |

---

## Raw Data References

| Data | Location | Description |
|------|----------|-------------|
| Ground Truth | `artifacts/phase-2-ground-truth.json` | 50 annotated chunks |
| Raw Results | `artifacts/phase-4-raw-results.json` | 2400 extraction results |
| Final Metrics | `artifacts/phase-5-final-metrics.json` | Aggregated analysis |
| Execution Log | `artifacts/execution-log.md` | Full execution trace |

---

## Appendix: Sample Extractions

### Example: High Quality Extraction

**Chunk**: {chunk text snippet}

**Ground Truth**: {list of terms}

**Model Output** ({model}, Variant D):
```json
{actual model output}
```

**Metrics**: Precision {X}%, Recall {Y}%, Hallucination {Z}%

### Example: Problematic Extraction

**Chunk**: {chunk text snippet}

**Ground Truth**: {list of terms}

**Model Output** ({model}, Variant {X}):
```json
{actual model output}
```

**Analysis**: {what went wrong and why}

---

*Results documented: {date}*
*Executor: {name/model}*
