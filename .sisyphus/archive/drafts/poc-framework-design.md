# Draft: POC Specification Framework Design

## User's Request

Create a strict POC documentation framework for 7 RAG pipeline POCs that:
- Provides strong guardrails for executing models
- Prevents lazy/incomplete execution
- Has clear success/failure criteria
- Is structured enough to be verifiable

## POCs to Cover (from RAG_PIPELINE_ARCHITECTURE.md)

| POC | Focus | Complexity |
|-----|-------|------------|
| POC-1 | LLM Term Extraction Guardrails | Medium |
| POC-2 | Confidence Scoring and Threshold | Medium |
| POC-3 | Topic Disambiguation (single query) | Medium |
| POC-4 | Summary-Based Search Narrowing | High |
| POC-5 | Query Enrichment Faithfulness | Medium |
| POC-6 | GLiNER Zero-Shot and Training | High |
| POC-7 | Embedding Model Selection | Low |

## Existing Infrastructure (can reuse)

- `poc/modular_retrieval_pipeline/benchmark.py` - benchmark runner
- `poc/modular_retrieval_pipeline/corpus/informed_questions.json` - 50 K8s questions
- `poc/modular_retrieval_pipeline/components/` - existing extractors
- `poc/test_data/` - test data generation
- Three-tier evaluation (deterministic → semantic → LLM-as-judge)

## Key Design Questions

1. **Specification vs Implementation**: POC spec describes WHAT to test + success criteria. Implementation is separate.
2. **Checkpoint mechanism**: How to ensure model doesn't skip steps?
3. **Evidence requirements**: What proof of completion is needed?
4. **Abort conditions**: When should POC be stopped early?

## User Decisions (Confirmed)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Location** | `poc/{poc-name}/SPEC.md` | Tightly coupled spec+implementation |
| **Detail Level** | Pure requirements | WHAT to test, success criteria. Executor decides HOW. |
| **Verification** | Both mechanisms | Checkpoint artifacts (per phase) + structured results template |
| **Early Termination** | No - always complete | Document all findings, even negative results |

---

## Proposed Framework Structure

### 1. Master Template Document
Location: `poc/POC_TEMPLATE.md`
Purpose: General template that ALL POC specs must follow

### 2. Individual POC Specs
Location: `poc/{poc-name}/SPEC.md`
Purpose: Specific requirements for each POC

### 3. Results Template
Location: `poc/{poc-name}/RESULTS.md`
Purpose: Structured template that executor must fill out

### 4. Checkpoint Artifacts
Location: `poc/{poc-name}/artifacts/`
Purpose: Each phase produces a specific file as proof of completion

---

## POC Spec Template Structure (Draft)

```markdown
# POC-{N}: {Title}

## TL;DR
> One-sentence summary of what this POC tests

## Research Question
The single question this POC answers (phrased as a question)

## Background
- Why this matters
- Link to architecture doc section
- Current state / baseline

## Hypothesis
What we expect to find (must be falsifiable)

## Scope
### IN SCOPE
- Explicit list of what this POC tests

### OUT OF SCOPE  
- Explicit list of what this POC does NOT test
- Things to defer to other POCs

## Test Design
### Test Cases
Numbered list of specific tests to run

### Test Data
- What data is needed
- Where it comes from
- How much (sample size)

### Variables
- Independent variables (what we change)
- Dependent variables (what we measure)
- Control variables (what stays constant)

## Metrics
### Primary Metrics
Metrics that determine success/failure

### Secondary Metrics
Metrics for deeper understanding (not pass/fail)

## Success Criteria
### PASS Criteria
Explicit thresholds that must ALL be met

### PARTIAL Criteria  
What counts as partial success

### FAIL Criteria
When we consider the hypothesis disproven

## Execution Phases
### Phase 1: {Name}
- Objective:
- Tasks:
- Checkpoint Artifact: `artifacts/{filename}`
- Acceptance: {what makes this phase complete}

### Phase 2: {Name}
...

## Checkpoint Artifacts
| Phase | Artifact | Format | Must Contain |
|-------|----------|--------|--------------|
| 1 | ... | ... | ... |

## Results Template
(What the RESULTS.md must contain - filled after execution)

## Dependencies
- Other POCs that must complete first
- Required infrastructure
- Required data

## Risks & Mitigations
- Known risks and how to handle them
```

---

## Additional Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| **Artifact Format** | Both: JSON (machine-verifiable) + Markdown summary |
| **Test Case Granularity** | Both: detailed scenarios + aggregate metrics |
| **First POC** | POC-1: LLM Extraction Guardrails |

---

## Final Framework Structure

### Documents to Create

1. **`poc/POC_SPECIFICATION_TEMPLATE.md`** - Master template
2. **`poc/poc-1-llm-extraction-guardrails/SPEC.md`** - First POC spec
3. **`poc/poc-1-llm-extraction-guardrails/RESULTS_TEMPLATE.md`** - Results template

### POC Directory Structure

```
poc/{poc-name}/
├── SPEC.md                    # POC specification (WHAT to test)
├── RESULTS_TEMPLATE.md        # Empty template (filled after execution)
├── RESULTS.md                 # Filled results (created during execution)
├── artifacts/                 # Checkpoint artifacts
│   ├── phase-1-*.json        # Machine-verifiable data
│   ├── phase-1-summary.md    # Human-readable summary
│   └── ...
├── pyproject.toml            # Dependencies
├── *.py                      # Implementation files
└── README.md                 # Setup/usage (how to run)
```

### Verification Flow

```
SPEC.md → Implementation → Checkpoint Artifacts → RESULTS.md
                              ↓
                    Each artifact validates:
                    1. JSON schema compliance
                    2. Required fields present
                    3. Sample sizes met
```

---

## Next Steps

1. Create `poc/POC_SPECIFICATION_TEMPLATE.md` (master template)
2. Create `poc/poc-1-llm-extraction-guardrails/SPEC.md` (first POC)
3. Create `poc/poc-1-llm-extraction-guardrails/RESULTS_TEMPLATE.md`

---

*All decisions finalized: 2026-02-03*

---

## POC-1 Design (Oracle Consultation)

### Key Recommendations from Oracle

**Sample Size**: 50 chunks minimum (30 gives ±12% CI, too wide)

**Content Diversity** (stratified sampling):
| Type | % | Why |
|------|---|-----|
| Prose (concepts) | 40% | Base case |
| Code/YAML | 20% | Literal extraction vs interpretation |
| Tables/lists | 15% | Structured data |
| Error messages | 15% | High-value domain terms |
| Mixed (API ref) | 10% | Stress test |

**Term Classification (4 Tiers)**:
| Tier | Definition | Examples | Include? |
|------|------------|----------|----------|
| 1: Domain-specific | Unique to K8s | CrashLoopBackOff, kube-apiserver | Always |
| 2: Domain-relevant | English with K8s meaning | pod, container, service | Yes |
| 3: Generic technical | Technical, not K8s-specific | API, server, configuration | Conditional |
| 4: Generic | No special meaning | the, component, using | No |

**Matching Strategy** (3-level):
1. Exact (after normalization)
2. Partial (≥80% token overlap)
3. Fuzzy (≥85% Levenshtein ratio)

**Hallucination Categories**:
| Category | Definition | Severity |
|----------|------------|----------|
| Fabrication | Term doesn't appear in chunk | Critical |
| Semantic drift | Term exists but meaning changed | High |
| Overclaim | Promotes Tier 3/4 to Tier 1 | Medium |
| Partial hallucination | Part of term fabricated | Medium |

**Prompt Variants** (2x2 design):
| Variant | Evidence Required | Output Constrained |
|---------|-------------------|-------------------|
| A: Baseline | No | No |
| B: Evidence | Must cite span | No |
| C: Constrained | No | Max 15 terms, confidence |
| D: Full guardrails | Must cite span | Max 15 terms, confidence |

**Statistical Design**:
- 3 trials per chunk-model combination (measure variance)
- Macro-average P/R (per chunk, then average)
- Hold out 20% chunks for validation after prompt tuning

**Models to Test**:
- Claude Haiku
- Claude Sonnet  
- Llama 3 8B (local)
- Mistral 7B (local)

**Estimated Effort**: ~20h total (3-4 days)

### Final Decisions (Confirmed)

| Decision | Choice |
|----------|--------|
| **Local model runtime** | Ollama |
| **Annotation approach** | Fully automated (LLM) + Opus review |
| **Synonym generation** | No - extraction only (defer to POC-1b) |

### Important Note on Annotation

Using Opus to create ground truth means:
- Opus should NOT be a test model (would be biased)
- Test models: Haiku, Sonnet, Llama 3 8B, Mistral 7B
- Ground truth created by: Opus (with human spot-check validation)

### Ready to Create POC-1 Spec
