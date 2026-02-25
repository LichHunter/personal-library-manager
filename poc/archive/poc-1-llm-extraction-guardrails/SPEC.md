# POC-1: LLM Term Extraction Guardrails

## TL;DR

> Test whether LLMs can reliably extract domain-specific terms from Kubernetes documentation with guardrail prompts, achieving >80% precision and <5% hallucination rate.

---

## 1. Research Question

**Primary Question**: How reliably can LLMs extract domain-specific terms from technical documentation when constrained by guardrail prompts?

**Sub-questions**:
- Which guardrail strategy (evidence citation, output constraints, or both) produces the best precision/recall tradeoff?
- How do different model sizes (Haiku vs Sonnet, 7B vs 8B) compare on this task?
- What is the baseline hallucination rate, and can guardrails reduce it below 5%?

---

## 2. Background

### 2.1 Why This Matters

The RAG pipeline's slow system relies on LLM extraction to discover novel domain terms that the fast system (YAKE + GLiNER) missed. If the LLM hallucination rate is too high, the manual review queue becomes unsustainable, defeating the purpose of automation.

**Critical dependency**: POC-2 (Confidence Threshold) and POC-6 (GLiNER Training) both depend on having a reliable slow system. If POC-1 fails, the entire term management architecture must be reconsidered.

### 2.2 Architecture Reference

- **Related Section**: [RAG_PIPELINE_ARCHITECTURE.md - Section 4.3: Slow System](../RAG_PIPELINE_ARCHITECTURE.md#43-slow-system-batch-processing)
- **Design Decision**: D5 (Tiered Review) assumes LLM can pre-filter with acceptable precision

### 2.3 Current State

| Aspect | Current | Target |
|--------|---------|--------|
| LLM Extraction Precision | Unknown | >80% |
| LLM Extraction Recall | Unknown | >60% |
| Hallucination Rate | Unknown | <5% |
| Guardrail Effectiveness | Unknown | Measurable improvement over baseline |

---

## 3. Hypothesis

### 3.1 Primary Hypothesis

> **H1**: LLM extraction with full guardrails (evidence citation + output constraints) will achieve >80% precision and <5% hallucination rate on Kubernetes term extraction.

### 3.2 Secondary Hypotheses

> **H2**: Evidence citation requirement will reduce hallucination rate by >50% compared to baseline prompt.

> **H3**: Claude Sonnet will outperform Claude Haiku by >10% precision, but Haiku will still meet the 80% threshold.

> **H4**: Local models (Llama 3 8B, Mistral 7B) will achieve >70% precision with guardrails, making them viable for cost-sensitive deployments.

### 3.3 Null Hypothesis

> **H0**: No guardrail configuration achieves both >80% precision AND <5% hallucination rate. LLM extraction is not reliable enough for the slow system without human review of ALL extractions.

---

## 4. Scope

### 4.1 IN SCOPE

| Item | Description |
|------|-------------|
| Term extraction | Identifying domain-specific terms that appear in chunk text |
| 4 prompt variants | Baseline, Evidence-required, Constrained, Full guardrails |
| 4 models | Claude Haiku, Claude Sonnet, Llama 3 8B (Ollama), Mistral 7B (Ollama) |
| 50 test chunks | Stratified sample from K8s documentation |
| 3 trials per condition | To measure within-model variance |
| Precision/Recall/Hallucination metrics | Primary evaluation metrics |

### 4.2 OUT OF SCOPE

| Item | Reason | Deferred To |
|------|--------|-------------|
| Synonym generation | Higher hallucination risk, separate capability | POC-1b |
| Term classification (error/resource/command) | Adds complexity, test extraction first | Future |
| Multi-domain testing | Start with K8s, generalize later | POC-6 |
| Cost optimization | Focus on quality first | Post-POC analysis |
| Prompt optimization beyond 4 variants | Avoid overfitting | Future iteration |

### 4.3 Assumptions

| Assumption | If Violated |
|------------|-------------|
| Ollama can run Llama 3 8B and Mistral 7B with acceptable latency | Skip local models, focus on Claude |
| 50 chunks is sufficient for statistical significance (±9% CI) | Document uncertainty, recommend follow-up with larger sample |
| Opus-generated ground truth is reliable | Bias in results; recommend human validation sample |
| K8s terminology is representative of technical domains | Results may not generalize to other domains |

---

## 5. Test Design

### 5.1 Ground Truth Creation

**Method**: Automated extraction with Opus review

#### Step 1: Sample Selection (50 chunks)

Select chunks from `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` with stratified sampling:

| Content Type | Count | Selection Criteria |
|--------------|-------|-------------------|
| Prose (concepts) | 20 | Files in `concepts_*/` directories |
| Code/YAML blocks | 10 | Files containing ``` or `yaml:` |
| Tables/lists | 8 | Files with `|` table syntax or `-` lists |
| Error messages | 7 | Files mentioning error states, troubleshooting |
| Mixed (API reference) | 5 | Files in `reference_kubernetes-api/` |

**Chunk size**: 200-500 tokens (use existing chunking from pipeline)

#### Step 2: Opus Ground Truth Generation

For each chunk, prompt Claude Opus to extract terms with this EXACT prompt:

```
You are creating ground truth annotations for evaluating term extraction models.

TASK: Extract ALL Kubernetes domain-specific terms from this chunk.

TERM CLASSIFICATION:
- Tier 1 (MUST INCLUDE): Terms unique to Kubernetes (CrashLoopBackOff, kube-apiserver, PodSpec)
- Tier 2 (MUST INCLUDE): English words with specific K8s meaning (pod, container, service, node, deployment)
- Tier 3 (CONDITIONAL): Technical terms not K8s-specific - include ONLY if used in K8s context (API, endpoint, namespace)
- Tier 4 (EXCLUDE): Generic words with no special K8s meaning (component, system, configuration)

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, quote the exact text span where it appears
3. Assign tier (1, 2, or 3)
4. Include multi-word terms (e.g., "Pod Security Policy" not just "Pod")

OUTPUT FORMAT (JSON):
{
  "chunk_id": "{chunk_id}",
  "terms": [
    {"term": "CrashLoopBackOff", "tier": 1, "span": "entered CrashLoopBackOff state"},
    {"term": "pod", "tier": 2, "span": "the pod will restart"},
    ...
  ],
  "total_terms": <count>
}

CHUNK:
---
{chunk_text}
---
```

#### Step 3: Opus Self-Review

After initial extraction, run a SECOND Opus pass with this prompt:

```
Review this ground truth annotation for completeness and accuracy.

ORIGINAL CHUNK:
---
{chunk_text}
---

EXTRACTED TERMS:
{opus_extraction_json}

CHECK:
1. Are there any Tier 1/2 terms in the chunk that were MISSED?
2. Are there any extracted terms that DON'T appear verbatim in the chunk?
3. Are tier assignments correct?

OUTPUT:
{
  "review_status": "APPROVED" | "CORRECTIONS_NEEDED",
  "missed_terms": [...],
  "false_extractions": [...],
  "tier_corrections": [...],
  "final_terms": [...]
}
```

#### Step 4: Human Spot-Check (10%)

Randomly select 5 chunks (10%) for human validation. Document any discrepancies.

### 5.2 Test Cases

#### TC-1: Baseline Extraction (No Guardrails)

| Attribute | Value |
|-----------|-------|
| **Objective** | Establish baseline performance without guardrails |
| **Prompt Variant** | A (Baseline) |
| **Input** | 50 chunks × 4 models × 3 trials = 600 extractions |
| **Procedure** | Run extraction with minimal prompt, no evidence or constraints |
| **Expected Outcome** | Lower precision, higher hallucination than guardrailed variants |
| **Pass Criteria** | Data collected for all 600 conditions |

**Baseline Prompt (Variant A)**:
```
Extract Kubernetes domain-specific terms from this text.

TEXT:
---
{chunk_text}
---

OUTPUT (JSON list of terms):
["term1", "term2", ...]
```

#### TC-2: Evidence-Required Extraction

| Attribute | Value |
|-----------|-------|
| **Objective** | Test whether requiring evidence citations reduces hallucination |
| **Prompt Variant** | B (Evidence Required) |
| **Input** | 50 chunks × 4 models × 3 trials = 600 extractions |
| **Procedure** | Require LLM to cite exact text span for each term |
| **Expected Outcome** | Lower hallucination rate, possibly lower recall |
| **Pass Criteria** | Hallucination rate <10% (50% reduction from baseline) |

**Evidence Prompt (Variant B)**:
```
Extract Kubernetes domain-specific terms from this text.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide the EXACT text span where it appears
3. If you cannot find the term verbatim, do NOT include it

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{
  "terms": [
    {"term": "...", "span": "...exact quote from text..."},
    ...
  ]
}
```

#### TC-3: Constrained Output Extraction

| Attribute | Value |
|-----------|-------|
| **Objective** | Test whether output constraints improve precision |
| **Prompt Variant** | C (Constrained) |
| **Input** | 50 chunks × 4 models × 3 trials = 600 extractions |
| **Procedure** | Limit to max 15 terms, require confidence scores |
| **Expected Outcome** | Higher precision (forced prioritization), similar recall |
| **Pass Criteria** | Precision >75% |

**Constrained Prompt (Variant C)**:
```
Extract Kubernetes domain-specific terms from this text.

RULES:
1. Maximum 15 terms per chunk
2. Prioritize terms that are MOST specific to Kubernetes
3. Assign confidence: HIGH (definitely K8s), MEDIUM (likely K8s), LOW (possibly generic)

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{
  "terms": [
    {"term": "...", "confidence": "HIGH|MEDIUM|LOW"},
    ...
  ]
}
```

#### TC-4: Full Guardrails Extraction

| Attribute | Value |
|-----------|-------|
| **Objective** | Test combined guardrails (evidence + constraints) |
| **Prompt Variant** | D (Full Guardrails) |
| **Input** | 50 chunks × 4 models × 3 trials = 600 extractions |
| **Procedure** | Combine evidence citation AND output constraints |
| **Expected Outcome** | Highest precision, lowest hallucination, acceptable recall |
| **Pass Criteria** | Precision >80%, Hallucination <5%, Recall >60% |

**Full Guardrails Prompt (Variant D)**:
```
Extract Kubernetes domain-specific terms from this documentation chunk.

RULES:
1. Only extract terms that appear VERBATIM in the text
2. For each term, provide the EXACT text span where it appears
3. Assign confidence: HIGH (definitely K8s-specific), MEDIUM (technical, likely K8s), LOW (possibly generic)
4. Maximum 15 terms per chunk
5. Prefer multi-word terms over single words when both exist (e.g., "Pod Security Policy" over "Pod")

TEXT:
---
{chunk_text}
---

OUTPUT (JSON):
{
  "terms": [
    {"term": "CrashLoopBackOff", "span": "the pod entered CrashLoopBackOff state", "confidence": "HIGH"},
    ...
  ]
}
```

#### TC-5: Cross-Model Comparison

| Attribute | Value |
|-----------|-------|
| **Objective** | Compare performance across models with best prompt variant |
| **Input** | Best prompt variant (likely D) across all 4 models |
| **Procedure** | Aggregate results from TC-1 through TC-4, compare by model |
| **Expected Outcome** | Sonnet > Haiku > Llama 3 > Mistral (hypothesis) |
| **Pass Criteria** | At least one model meets all success criteria |

#### TC-6: Variance Analysis

| Attribute | Value |
|-----------|-------|
| **Objective** | Measure within-model consistency |
| **Input** | 3 trials per chunk-model-prompt condition |
| **Procedure** | Calculate standard deviation of precision/recall across trials |
| **Expected Outcome** | SD < 5% for Claude models, SD < 10% for local models |
| **Pass Criteria** | Results are reproducible (low variance) |

### 5.3 Test Data

| Dataset | Source | Size | Purpose |
|---------|--------|------|---------|
| K8s Chunks | `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` | 50 chunks | Evaluation corpus |
| Ground Truth | Opus-generated + reviewed | 50 annotations | Gold standard labels |
| Validation Set | 10 held-out chunks (20%) | 10 chunks | Prevent prompt overfitting |

**Data Requirements**:
- Minimum sample size: 50 chunks (±9% CI at 95% confidence)
- Diversity: Stratified by content type (prose, code, tables, errors, mixed)
- Ground truth: Dual-pass Opus annotation with 10% human spot-check

### 5.4 Variables

| Type | Variable | Values/Range |
|------|----------|--------------|
| **Independent** | Prompt variant | A, B, C, D |
| **Independent** | Model | Haiku, Sonnet, Llama 3 8B, Mistral 7B |
| **Independent** | Trial number | 1, 2, 3 |
| **Dependent** | Precision | 0.0 - 1.0 |
| **Dependent** | Recall | 0.0 - 1.0 |
| **Dependent** | Hallucination rate | 0.0 - 1.0 |
| **Dependent** | Groundedness score | 0.0 - 1.0 |
| **Control** | Chunk selection | Fixed 50 chunks |
| **Control** | Temperature | 0.0 (deterministic) |
| **Control** | Max tokens | 2000 |

---

## 6. Metrics

### 6.1 Primary Metrics (Determine Pass/Fail)

| Metric | Definition | Calculation | Target |
|--------|------------|-------------|--------|
| **Precision** | % of extracted terms that are correct | (exact + partial matches) / total extracted | >80% |
| **Recall** | % of ground truth terms that were extracted | (exact + partial matches) / total ground truth | >60% |
| **Hallucination Rate** | % of extracted terms that are fabricated | fabrications / total extracted | <5% |

### 6.2 Secondary Metrics (For Understanding)

| Metric | Definition | Purpose |
|--------|------------|---------|
| **Exact Match Precision** | Precision using exact match only | Measure strict accuracy |
| **Groundedness Score** | % of cited spans that exist in text | Verify evidence quality |
| **Confidence Calibration** | Correlation between stated confidence and actual correctness | Assess self-awareness |
| **Latency** | Time per extraction (ms) | Cost/speed tradeoff |
| **Token Usage** | Input + output tokens per extraction | Cost estimation |

### 6.3 Matching Algorithm

```python
def normalize(term: str) -> str:
    """Normalize term for comparison."""
    return term.lower().strip().replace("-", " ").replace("_", " ")

def match_terms(extracted: str, ground_truth: str) -> str:
    """
    Three-level matching:
    - exact: Normalized strings match
    - partial: >=80% token overlap
    - fuzzy: >=85% Levenshtein similarity
    - no_match: None of the above
    """
    ext_norm = normalize(extracted)
    gt_norm = normalize(ground_truth)
    
    # Level 1: Exact match
    if ext_norm == gt_norm:
        return "exact"
    
    # Level 2: Token overlap
    ext_tokens = set(ext_norm.split())
    gt_tokens = set(gt_norm.split())
    if gt_tokens:  # Avoid division by zero
        overlap = len(ext_tokens & gt_tokens) / len(gt_tokens)
        if overlap >= 0.8:
            return "partial"
    
    # Level 3: Fuzzy match
    from rapidfuzz import fuzz
    if fuzz.ratio(ext_norm, gt_norm) >= 85:
        return "fuzzy"
    
    return "no_match"
```

### 6.4 Hallucination Categories

| Category | Definition | Example | Counted in Hallucination Rate? |
|----------|------------|---------|-------------------------------|
| **Fabrication** | Term doesn't appear anywhere in chunk | Chunk: "pod restart" -> Output: "StatefulSet" | YES |
| **Semantic Drift** | Term exists but meaning misinterpreted | "node" (K8s) -> interpreted as "NodeJS" | YES |
| **Partial Hallucination** | Part of term fabricated | "CrashLoopBackOff" -> "CrashLoopError" | YES (as fabrication) |
| **Overclaim** | Generic term claimed as domain-specific | "configuration" labeled HIGH confidence | NO (precision error, not hallucination) |

### 6.5 Aggregation Method

**Use Macro-Averaging**: Calculate precision/recall per chunk, then average across chunks.

```python
def macro_average_metrics(results: List[ChunkResult]) -> dict:
    """
    Macro-average: Each chunk weighted equally regardless of term count.
    """
    precisions = [r.precision for r in results]
    recalls = [r.recall for r in results]
    hallucination_rates = [r.hallucination_rate for r in results]
    
    return {
        "precision_mean": np.mean(precisions),
        "precision_std": np.std(precisions),
        "recall_mean": np.mean(recalls),
        "recall_std": np.std(recalls),
        "hallucination_mean": np.mean(hallucination_rates),
        "hallucination_std": np.std(hallucination_rates),
    }
```

---

## 7. Success Criteria

### 7.1 PASS (All Must Be True)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| At least one model+prompt achieves Precision >80% | Required | Minimum viable for automated triage |
| At least one model+prompt achieves Hallucination <5% | Required | Keeps manual review manageable |
| At least one model+prompt achieves Recall >60% | Required | Must catch majority of terms |
| Best configuration meets ALL THREE criteria | Required | Single viable configuration exists |

### 7.2 PARTIAL PASS (Hypothesis Partially Supported)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| Best config achieves 2 of 3 metrics | Precision >80% OR Hallucination <5% OR Recall >60% | May need prompt refinement or accept tradeoffs |
| Guardrails improve over baseline by >20% | Any primary metric | Guardrail approach is valid, thresholds need adjustment |
| At least Claude Sonnet meets criteria | Other models fail | Viable but higher cost |

### 7.3 FAIL (Hypothesis Rejected)

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| No model+prompt achieves Precision >70% | Below minimum | LLM extraction unreliable; need different approach |
| All configurations have Hallucination >10% | Unacceptable | Review burden too high; human-in-loop for ALL extractions |
| Guardrails show <10% improvement over baseline | Ineffective | Guardrails don't help; need structural changes |

**Note**: Even if FAIL, POC must run to completion and document all findings.

---

## 8. Execution Phases

### Phase 1: Environment Setup

| Attribute | Value |
|-----------|-------|
| **Objective** | Set up all required infrastructure and verify it works |
| **Tasks** | 1. Create POC directory structure<br>2. Install dependencies (anthropic, ollama, rapidfuzz, etc.)<br>3. Verify Claude API access<br>4. Install and verify Ollama with Llama 3 8B and Mistral 7B<br>5. Create evaluation harness skeleton |
| **Inputs** | POC spec, existing corpus |
| **Outputs** | Working environment, dependency list |
| **Checkpoint Artifact** | `artifacts/phase-1-setup.json`<br>`artifacts/phase-1-summary.md` |
| **Acceptance Criteria** | All 4 models respond to test prompt successfully |

### Phase 2: Ground Truth Creation

| Attribute | Value |
|-----------|-------|
| **Objective** | Create gold-standard annotations for 50 chunks |
| **Tasks** | 1. Select 50 chunks with stratified sampling<br>2. Run Opus extraction on all chunks<br>3. Run Opus self-review on all chunks<br>4. Human spot-check 5 random chunks<br>5. Compile final ground truth JSON |
| **Inputs** | K8s corpus, Opus API |
| **Outputs** | `ground_truth.json` with 50 annotated chunks |
| **Checkpoint Artifact** | `artifacts/phase-2-ground-truth.json`<br>`artifacts/phase-2-summary.md` |
| **Acceptance Criteria** | 50 chunks annotated, human spot-check completed, inter-annotator agreement documented |

### Phase 3: Evaluation Harness Implementation

| Attribute | Value |
|-----------|-------|
| **Objective** | Build the code to run extractions and compute metrics |
| **Tasks** | 1. Implement extraction runner for all 4 models<br>2. Implement 4 prompt templates<br>3. Implement matching algorithm (exact/partial/fuzzy)<br>4. Implement metric calculation (P/R/H)<br>5. Implement result aggregation<br>6. Test on 3 sample chunks |
| **Inputs** | Ground truth, model APIs |
| **Outputs** | Working evaluation code |
| **Checkpoint Artifact** | `artifacts/phase-3-harness-validation.json`<br>`artifacts/phase-3-summary.md` |
| **Acceptance Criteria** | Harness runs successfully on 3 test chunks with all 4 models |

### Phase 4: Main Experiment Execution

| Attribute | Value |
|-----------|-------|
| **Objective** | Run all 2400 extraction conditions (50 chunks x 4 models x 4 prompts x 3 trials) |
| **Tasks** | 1. Run Variant A (baseline) on all models<br>2. Run Variant B (evidence) on all models<br>3. Run Variant C (constrained) on all models<br>4. Run Variant D (full guardrails) on all models<br>5. Save raw outputs for all runs |
| **Inputs** | Ground truth, evaluation harness |
| **Outputs** | 2400 extraction results |
| **Checkpoint Artifact** | `artifacts/phase-4-raw-results.json`<br>`artifacts/phase-4-summary.md` |
| **Acceptance Criteria** | All 2400 conditions completed, no missing data |

### Phase 5: Analysis and Reporting

| Attribute | Value |
|-----------|-------|
| **Objective** | Compute metrics, statistical tests, generate final report |
| **Tasks** | 1. Calculate per-chunk metrics<br>2. Aggregate by model and prompt variant<br>3. Run statistical significance tests (t-test)<br>4. Generate visualizations (tables, charts)<br>5. Write RESULTS.md<br>6. Document recommendations |
| **Inputs** | Raw results |
| **Outputs** | RESULTS.md, final analysis |
| **Checkpoint Artifact** | `artifacts/phase-5-final-metrics.json`<br>`artifacts/phase-5-summary.md` |
| **Acceptance Criteria** | RESULTS.md complete with all sections filled, all metrics calculated |

---

## 9. Checkpoint Artifacts

### 9.1 Artifact Registry

| Phase | Artifact File | Format | Required Fields |
|-------|---------------|--------|-----------------|
| 1 | `phase-1-setup.json` | JSON | models_verified, dependencies, api_status |
| 1 | `phase-1-summary.md` | Markdown | Objective, Approach, Results, Issues, Next Phase Readiness |
| 2 | `phase-2-ground-truth.json` | JSON | chunks[], total_chunks, total_terms, content_type_distribution |
| 2 | `phase-2-summary.md` | Markdown | Sampling method, annotation process, spot-check results |
| 3 | `phase-3-harness-validation.json` | JSON | test_chunks[], models_tested, sample_metrics |
| 3 | `phase-3-summary.md` | Markdown | Implementation details, validation results |
| 4 | `phase-4-raw-results.json` | JSON | results[] (2400 entries), completion_stats |
| 4 | `phase-4-summary.md` | Markdown | Execution summary, any errors/retries |
| 5 | `phase-5-final-metrics.json` | JSON | metrics_by_model_prompt, statistical_tests, best_configuration |
| 5 | `phase-5-summary.md` | Markdown | Key findings, recommendations |

### 9.2 JSON Schema: Ground Truth (Phase 2)

```json
{
  "created_at": "2026-02-03T12:00:00Z",
  "created_by": "claude-opus",
  "total_chunks": 50,
  "total_terms": 342,
  "content_type_distribution": {
    "prose": 20,
    "code": 10,
    "tables": 8,
    "errors": 7,
    "mixed": 5
  },
  "chunks": [
    {
      "chunk_id": "chunk_001",
      "source_file": "concepts_architecture_controller.md",
      "content_type": "prose",
      "text": "...",
      "terms": [
        {"term": "controller", "tier": 2, "span": "the controller reconciles..."},
        {"term": "ReplicaSet", "tier": 1, "span": "creates a ReplicaSet"}
      ],
      "total_terms": 2,
      "human_validated": false
    }
  ],
  "human_spot_check": {
    "chunks_checked": 5,
    "discrepancies": [],
    "agreement_rate": 1.0
  }
}
```

### 9.3 JSON Schema: Raw Results (Phase 4)

```json
{
  "experiment_id": "poc-1-main",
  "started_at": "2026-02-03T14:00:00Z",
  "completed_at": "2026-02-03T18:00:00Z",
  "total_conditions": 2400,
  "completed_conditions": 2400,
  "failed_conditions": 0,
  "results": [
    {
      "chunk_id": "chunk_001",
      "model": "claude-haiku",
      "prompt_variant": "D",
      "trial": 1,
      "extracted_terms": [
        {"term": "controller", "span": "the controller reconciles", "confidence": "HIGH"}
      ],
      "metrics": {
        "precision": 0.85,
        "recall": 0.70,
        "hallucination_rate": 0.02,
        "groundedness_score": 0.98
      },
      "latency_ms": 1234,
      "tokens_used": {"input": 500, "output": 150}
    }
  ]
}
```

### 9.4 JSON Schema: Final Metrics (Phase 5)

```json
{
  "analysis_completed_at": "2026-02-03T19:00:00Z",
  "best_configuration": {
    "model": "claude-sonnet",
    "prompt_variant": "D",
    "precision": 0.84,
    "recall": 0.68,
    "hallucination_rate": 0.03
  },
  "metrics_by_model_prompt": {
    "claude-haiku": {
      "A": {"precision_mean": 0.65, "precision_std": 0.08},
      "B": {},
      "C": {},
      "D": {}
    },
    "claude-sonnet": {},
    "llama-3-8b": {},
    "mistral-7b": {}
  },
  "statistical_tests": {
    "sonnet_vs_haiku": {"t_stat": 2.34, "p_value": 0.02, "significant": true},
    "guardrails_vs_baseline": {"t_stat": 4.56, "p_value": 0.0001, "significant": true}
  },
  "hypothesis_verdicts": {
    "H1": {"verdict": "SUPPORTED", "evidence": "Sonnet+D achieves 84% precision, 3% hallucination"},
    "H2": {"verdict": "SUPPORTED", "evidence": "Evidence requirement reduced hallucination by 60%"},
    "H3": {"verdict": "PARTIAL", "evidence": "Sonnet +8% over Haiku, but Haiku at 78% (below 80%)"},
    "H4": {"verdict": "REJECTED", "evidence": "Llama 3 achieved 62% precision, below 70% threshold"}
  },
  "pass_fail_status": "PASS"
}
```

### 9.5 Markdown Summary Requirements

Each `phase-{n}-summary.md` MUST contain these sections:

```markdown
# Phase {N} Summary: {Phase Name}

## Objective
{What this phase aimed to accomplish}

## Approach
{How it was executed, key decisions made}

## Results
{Key findings with specific numbers}

## Issues Encountered
{Any problems and how they were resolved}

## Next Phase Readiness
{Confirmation that prerequisites for next phase are met}
- [ ] {Checklist item 1}
- [ ] {Checklist item 2}
```

---

## 10. Dependencies

### 10.1 POC Dependencies

| Dependency | Type | Status |
|------------|------|--------|
| None | N/A | POC-1 is foundational |

### 10.2 Infrastructure Dependencies

| Dependency | Purpose | Setup Instructions |
|------------|---------|-------------------|
| Claude API | Run Haiku, Sonnet, Opus | `export ANTHROPIC_API_KEY=...` |
| Ollama | Run local models | `curl -fsSL https://ollama.com/install.sh \| sh` |
| Llama 3 8B | Local model testing | `ollama pull llama3:8b` |
| Mistral 7B | Local model testing | `ollama pull mistral:7b` |
| Python 3.11+ | Runtime | Available in Nix shell |
| rapidfuzz | Fuzzy matching | `uv add rapidfuzz` |

### 10.3 Data Dependencies

| Data | Source | Availability |
|------|--------|--------------|
| K8s Documentation Chunks | `poc/chunking_benchmark_v2/corpus/kubernetes_sample_200/` | Available |
| Ground Truth | To be created in Phase 2 | Must create |

---

## 11. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Opus ground truth contains errors | Medium | High | Human spot-check 10%, document disagreements |
| Ollama models too slow | Medium | Medium | Set timeout, allow skip with documentation |
| API rate limits | Low | Medium | Implement retry with backoff, batch requests |
| Local models produce malformed JSON | High | Medium | Use structured output (Outlines) or fallback parsing |
| 50 chunks insufficient for significance | Medium | Medium | Document confidence intervals, recommend follow-up |
| Temperature variance in outputs | Low | Low | Set temperature=0, run 3 trials anyway |

---

## 12. Execution Constraints

### 12.1 MUST DO (Non-Negotiable)

- [ ] Complete ALL 2400 extraction conditions (no skipping)
- [ ] Produce ALL checkpoint artifacts for ALL phases
- [ ] Document ALL results including failures and errors
- [ ] Fill RESULTS.md completely (no empty sections, no "TBD")
- [ ] Run statistical significance tests on key comparisons
- [ ] Run to completion regardless of intermediate results

### 12.2 MUST NOT DO

- [ ] Skip test cases due to poor initial results
- [ ] Modify success criteria after seeing data
- [ ] Leave artifacts incomplete or missing required fields
- [ ] Report "TBD", "TODO", or placeholders in final results
- [ ] Terminate early without completing all phases
- [ ] Cherry-pick which chunks or models to report

### 12.3 Quality Gates

| Gate | Check | Blocking? |
|------|-------|-----------|
| Phase 1 complete | All 4 models respond to test prompt | Yes |
| Phase 2 complete | 50 chunks annotated, ground_truth.json valid | Yes |
| Phase 3 complete | Harness runs on 3 test chunks successfully | Yes |
| Phase 4 complete | 2400/2400 conditions completed | Yes |
| Phase 5 complete | RESULTS.md has all sections filled | Yes |

---

## Appendix A: Glossary

| Term | Definition |
|------|------------|
| **Precision** | True positives / (True positives + False positives) |
| **Recall** | True positives / (True positives + False negatives) |
| **Hallucination** | Model outputs content not grounded in input |
| **Fabrication** | Specific type of hallucination where term doesn't exist in source |
| **Groundedness** | Degree to which output is supported by input |
| **Guardrail** | Prompt constraint designed to reduce errors |
| **Macro-average** | Average of per-sample metrics (each sample weighted equally) |

## Appendix B: References

- [RAG_PIPELINE_ARCHITECTURE.md](../../RAG_PIPELINE_ARCHITECTURE.md) - System design
- [POC_SPECIFICATION_TEMPLATE.md](../POC_SPECIFICATION_TEMPLATE.md) - Template this spec follows
- [vocabulary-extraction-expansion-research.md](../../.sisyphus/research/vocabulary-extraction-expansion-research.md) - Background research

---

*Specification Version: 1.0*
*Created: 2026-02-03*
