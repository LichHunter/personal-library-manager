# POC Specification Template

> **Purpose**: This template defines the required structure for all POC specifications in this project. Every POC MUST follow this format to ensure consistency, completeness, and verifiability.

---

## How to Use This Template

1. Copy this template to `poc/{poc-name}/SPEC.md`
2. Fill in ALL sections (no section should be empty or contain only "TBD")
3. Create `RESULTS_TEMPLATE.md` based on the Results Template section
4. During execution, create checkpoint artifacts as specified
5. Fill `RESULTS.md` upon completion

---

## Required Directory Structure

```
poc/{poc-name}/
├── SPEC.md                    # POC specification (copy of this template, filled)
├── RESULTS_TEMPLATE.md        # Empty results template (created from SPEC)
├── RESULTS.md                 # Filled results (created during execution)
├── artifacts/                 # Checkpoint artifacts (created during execution)
│   ├── phase-{n}-{name}.json  # Machine-verifiable data
│   ├── phase-{n}-summary.md   # Human-readable summary
│   └── ...
├── pyproject.toml             # Dependencies (uv package manager)
├── *.py                       # Implementation files
└── README.md                  # Setup and usage instructions
```

---

# POC-{N}: {Title}

## TL;DR

> One sentence that captures what this POC tests and why it matters.

---

## 1. Research Question

**Primary Question**: {Single question this POC answers, phrased as a question ending with "?"}

**Sub-questions** (if applicable):
- {Sub-question 1}
- {Sub-question 2}

---

## 2. Background

### 2.1 Why This Matters

{2-3 sentences explaining the importance of this investigation. What decision does it inform? What risk does it mitigate?}

### 2.2 Architecture Reference

- **Related Section**: {Link to RAG_PIPELINE_ARCHITECTURE.md section}
- **Design Decision**: {Which design decision this validates}

### 2.3 Current State

| Aspect | Current | Target |
|--------|---------|--------|
| {Metric 1} | {value} | {value} |
| {Metric 2} | {value} | {value} |

---

## 3. Hypothesis

### 3.1 Primary Hypothesis

> **H1**: {Falsifiable statement. Example: "GLiNER zero-shot will achieve >60% precision on Kubernetes entity extraction."}

### 3.2 Secondary Hypotheses (if applicable)

> **H2**: {Additional falsifiable statement}

### 3.3 Null Hypothesis

> **H0**: {What we'd conclude if primary hypothesis is false. Example: "GLiNER requires domain-specific training to be useful."}

---

## 4. Scope

### 4.1 IN SCOPE

| Item | Description |
|------|-------------|
| {Item 1} | {What specifically is being tested} |
| {Item 2} | {What specifically is being tested} |

### 4.2 OUT OF SCOPE

| Item | Reason | Deferred To |
|------|--------|-------------|
| {Item 1} | {Why excluded} | {POC-N or "Future work"} |
| {Item 2} | {Why excluded} | {POC-N or "Future work"} |

### 4.3 Assumptions

| Assumption | If Violated |
|------------|-------------|
| {Assumption 1} | {Impact on POC validity} |
| {Assumption 2} | {Impact on POC validity} |

---

## 5. Test Design

### 5.1 Test Cases

#### TC-{N}: {Test Case Name}

| Attribute | Value |
|-----------|-------|
| **Objective** | {What this test validates} |
| **Input** | {Specific input data/conditions} |
| **Procedure** | {Steps to execute} |
| **Expected Outcome** | {What success looks like} |
| **Pass Criteria** | {Measurable threshold} |

#### TC-{N+1}: {Test Case Name}

{Repeat structure for each test case}

### 5.2 Test Data

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| {Dataset 1} | {Where it comes from} | {N samples} | {What it tests} |
| {Dataset 2} | {Where it comes from} | {N samples} | {What it tests} |

**Data Requirements**:
- Minimum sample size: {N}
- Diversity requirements: {e.g., "must include code blocks, tables, prose"}
- Ground truth requirements: {e.g., "manually labeled by domain expert"}

### 5.3 Variables

| Type | Variable | Values/Range |
|------|----------|--------------|
| **Independent** | {What we manipulate} | {Range of values} |
| **Dependent** | {What we measure} | {Expected range} |
| **Control** | {What stays constant} | {Fixed value} |

---

## 6. Metrics

### 6.1 Primary Metrics (Determine Pass/Fail)

| Metric | Definition | Calculation | Target |
|--------|------------|-------------|--------|
| {Metric 1} | {What it measures} | {Formula or method} | {Threshold} |
| {Metric 2} | {What it measures} | {Formula or method} | {Threshold} |

### 6.2 Secondary Metrics (For Understanding)

| Metric | Definition | Purpose |
|--------|------------|---------|
| {Metric 1} | {What it measures} | {Why we track it} |
| {Metric 2} | {What it measures} | {Why we track it} |

### 6.3 Metric Collection

| Metric | Collection Method | Frequency |
|--------|-------------------|-----------|
| {Metric 1} | {How collected} | {Per test case / aggregate} |

---

## 7. Success Criteria

### 7.1 PASS (All Must Be True)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| {Criterion 1} | {Measurable threshold} | {Why this threshold} |
| {Criterion 2} | {Measurable threshold} | {Why this threshold} |

### 7.2 PARTIAL PASS (Hypothesis Partially Supported)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| {Criterion 1} | {Range} | {What this means for architecture} |

### 7.3 FAIL (Hypothesis Rejected)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| {Criterion 1} | {Threshold} | {What this means for architecture} |

**Note**: POC must ALWAYS run to completion. "FAIL" means hypothesis is disproven, NOT that POC was abandoned.

---

## 8. Execution Phases

### Phase 1: {Phase Name}

| Attribute | Value |
|-----------|-------|
| **Objective** | {What this phase accomplishes} |
| **Tasks** | 1. {Task 1}<br>2. {Task 2}<br>3. {Task 3} |
| **Inputs** | {What's needed to start} |
| **Outputs** | {What this phase produces} |
| **Checkpoint Artifact** | `artifacts/phase-1-{name}.json`<br>`artifacts/phase-1-summary.md` |
| **Acceptance Criteria** | {What makes this phase complete} |

### Phase 2: {Phase Name}

{Repeat structure for each phase}

### Phase 3: {Phase Name}

{Repeat structure for each phase}

---

## 9. Checkpoint Artifacts

### 9.1 Artifact Registry

| Phase | Artifact File | Format | Required Fields |
|-------|---------------|--------|-----------------|
| 1 | `phase-1-{name}.json` | JSON | {List required keys} |
| 1 | `phase-1-summary.md` | Markdown | {List required sections} |
| 2 | `phase-2-{name}.json` | JSON | {List required keys} |
| ... | ... | ... | ... |

### 9.2 JSON Schema Requirements

```json
{
  "phase": "{phase_number}",
  "name": "{phase_name}",
  "completed_at": "{ISO 8601 timestamp}",
  "status": "complete|partial|blocked",
  "metrics": {
    "{metric_name}": "{value}"
  },
  "sample_size": "{number}",
  "notes": "{any observations}"
}
```

### 9.3 Markdown Summary Requirements

Each `phase-{n}-summary.md` MUST contain:
1. **Objective**: What this phase aimed to accomplish
2. **Approach**: How it was executed
3. **Results**: Key findings with numbers
4. **Issues**: Any problems encountered
5. **Next Phase Readiness**: Confirmation that next phase can proceed

---

## 10. Results Template

> The following structure MUST be used for `RESULTS.md` upon POC completion.

```markdown
# POC-{N} Results: {Title}

## Execution Summary

| Attribute | Value |
|-----------|-------|
| **Started** | {ISO 8601 timestamp} |
| **Completed** | {ISO 8601 timestamp} |
| **Duration** | {hours/days} |
| **Executor** | {human/model name} |
| **Status** | PASS / PARTIAL / FAIL |

## Hypothesis Verdict

| Hypothesis | Verdict | Evidence |
|------------|---------|----------|
| H1: {statement} | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence} |
| H2: {statement} | SUPPORTED / REJECTED / INCONCLUSIVE | {brief evidence} |

## Primary Metrics

| Metric | Target | Actual | Verdict |
|--------|--------|--------|---------|
| {Metric 1} | {target} | {actual} | PASS/FAIL |
| {Metric 2} | {target} | {actual} | PASS/FAIL |

## Secondary Metrics

| Metric | Value | Observation |
|--------|-------|-------------|
| {Metric 1} | {value} | {what this tells us} |

## Test Case Results

| TC | Name | Status | Notes |
|----|------|--------|-------|
| TC-1 | {name} | PASS/FAIL | {brief note} |
| TC-2 | {name} | PASS/FAIL | {brief note} |

## Phase Completion

| Phase | Status | Artifact |
|-------|--------|----------|
| 1 | COMPLETE | `artifacts/phase-1-*.json` |
| 2 | COMPLETE | `artifacts/phase-2-*.json` |

## Key Findings

### Finding 1: {Title}
{Description with supporting data}

### Finding 2: {Title}
{Description with supporting data}

## Surprising Results

{Anything unexpected, whether positive or negative}

## Limitations

{What this POC did NOT prove or test}

## Recommendations

### For Architecture
{How findings affect RAG_PIPELINE_ARCHITECTURE.md}

### For Next POCs
{What subsequent POCs should consider}

## Raw Data

- Full results: `artifacts/final-results.json`
- Logs: `artifacts/execution-log.md`
```

---

## 11. Dependencies

### 11.1 POC Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| {POC-N} | Must complete first | {Complete/Pending} |
| {POC-M} | Shares test data | {Complete/Pending} |

### 11.2 Infrastructure Dependencies

| Dependency | Purpose | Setup Instructions |
|------------|---------|-------------------|
| {Dependency 1} | {Why needed} | {How to set up} |

### 11.3 Data Dependencies

| Data | Source | Availability |
|------|--------|--------------|
| {Dataset 1} | {Where from} | {Available/Must create} |

---

## 12. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| {Risk 1} | High/Med/Low | High/Med/Low | {How to handle} |
| {Risk 2} | High/Med/Low | High/Med/Low | {How to handle} |

---

## 13. Execution Constraints

### 13.1 MUST DO (Non-Negotiable)

- [ ] Complete ALL test cases (no skipping)
- [ ] Produce ALL checkpoint artifacts
- [ ] Document ALL results (including failures)
- [ ] Fill RESULTS.md completely (no empty sections)
- [ ] Run to completion (no early termination)

### 13.2 MUST NOT DO

- [ ] Skip test cases due to initial poor results
- [ ] Modify success criteria mid-execution
- [ ] Leave artifacts incomplete
- [ ] Report "TBD" or "TODO" in final results
- [ ] Terminate early without documenting why

### 13.3 Quality Gates

| Gate | Check | Blocking? |
|------|-------|-----------|
| Phase completion | All artifacts exist and are valid | Yes |
| Sample size | Minimum N samples processed | Yes |
| Results completeness | All RESULTS.md sections filled | Yes |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| {Term 1} | {Definition} |
| {Term 2} | {Definition} |

## Appendix B: References

- {Reference 1}
- {Reference 2}

---

*Template Version: 1.0*
*Last Updated: 2026-02-03*
