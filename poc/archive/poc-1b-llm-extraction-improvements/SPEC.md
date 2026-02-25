# POC-1b: LLM Term Extraction Improvements

## TL;DR

> Test advanced LLM-only techniques (Instructor structured output, self-consistency voting, span verification, multi-pass extraction) to achieve **95%+ precision, 95%+ recall, <1% hallucination** for Kubernetes term extraction.

---

## 1. Research Question

**Primary Question**: Can LLM-only approaches achieve production-ready extraction quality (95%+ P/R, <1% hallucination) without NER hybrid systems?

**Sub-questions**:
- Does Instructor structured output with field validators reduce hallucination below 1%?
- Does self-consistency voting (N=5-10) improve both precision AND recall?
- Does multi-pass extraction ("what did I miss?") recover missed terms without increasing hallucination?
- Can span verification in Pydantic validators eliminate all fabricated terms?

---

## 2. Background

### 2.1 POC-1 Results (Baseline)

| Metric | Target | POC-1 Best | Gap |
|--------|--------|------------|-----|
| Precision | >95% | 81.0% (Sonnet+D) | -14% |
| Recall | >95% | 71.4% (Baseline) | -24% |
| Hallucination | <1% | 7.4% (Haiku+D) | +6.4% |

**Key Finding**: Variant D (full guardrails) achieved best precision/hallucination tradeoff, but recall dropped significantly. Evidence citation requirement reduced hallucination by 55% but made the model too conservative.

### 2.2 Why This Matters

The RAG pipeline uses NER as the fast path and LLM as the fallback for when NER isn't enough. The LLM must achieve near-perfect precision to avoid polluting the term graph with hallucinated terms, while maintaining high recall to catch domain-specific terms that NER misses.

### 2.3 Research-Backed Techniques

Based on extensive research, the following techniques show promise:

| Technique | Expected Impact | Evidence |
|-----------|----------------|----------|
| **Instructor + Pydantic validators** | Hallucination <1% | Citation validation forces grounding in source |
| **Self-consistency voting (N=10)** | +25% precision, +15% recall | MATH-500 benchmark: 3.4x accuracy with N=10 |
| **Multi-pass extraction** | +15-20% recall | "What did I miss?" prompting reduces conservatism |
| **Span verification** | Hallucination <1% | Exact substring matching catches fabrications |
| **Temperature 0.8 + diversity** | +10% recall | High temp enables self-consistency diversity |

---

## 3. Hypothesis

### 3.1 Primary Hypothesis

> **H1**: LLM extraction with Instructor structured output + self-consistency voting (N=10, 70% agreement threshold) will achieve **95%+ precision, 85%+ recall, <1% hallucination** on Kubernetes term extraction.

### 3.2 Secondary Hypotheses

> **H2**: Span verification validators will reduce hallucination from 7.4% to <1% while maintaining precision.

> **H3**: Multi-pass extraction ("what did I miss?") will increase recall from 71% to 85%+ without increasing hallucination above 5%.

> **H4**: Self-consistency voting with 70% agreement threshold will achieve 95%+ precision with <1% hallucination.

> **H5**: Combining all techniques will achieve the target: 95%+ P, 95%+ R, <1% H.

---

## 4. Experiment Design

### 4.1 Strategies to Test

| Strategy | Description | Key Parameters |
|----------|-------------|----------------|
| **E (Instructor)** | Structured output with Pydantic field validators | max_retries=3, validation_context |
| **F (Span Verify)** | Instructor + span verification validator | Exact substring check |
| **G (Self-Consistency)** | N=10 samples, 70% agreement voting | temp=0.8, top_p=0.95 |
| **H (Multi-Pass)** | 3-pass extraction with "what did I miss?" | Pass 1: extract, Pass 2: review, Pass 3: category sweep |
| **I (Combined)** | F + G + H (all techniques) | Full pipeline |

### 4.2 Test Matrix

- **Chunks**: Same 45 chunks from POC-1 ground truth
- **Models**: Claude Haiku, Claude Sonnet (skip local models for speed)
- **Strategies**: E, F, G, H, I (5 strategies)
- **Trials**: 3 per condition
- **Total**: 45 chunks × 2 models × 5 strategies × 3 trials = **1,350 extractions**

### 4.3 Comparison Baselines

From POC-1:
- **Baseline A**: 69.4% P, 63.2% R, 24.7% H (Haiku)
- **Best D**: 79.3% P, 45.4% R, 7.4% H (Haiku)
- **Best D**: 81.0% P, 63.7% R, 16.8% H (Sonnet)

---

## 5. Implementation Details

### 5.1 Strategy E: Instructor Structured Output

```python
import instructor
from pydantic import BaseModel, Field, field_validator, ValidationInfo
from typing import List

client = instructor.from_anthropic(
    Anthropic(),
    mode=instructor.Mode.ANTHROPIC_TOOLS,
    max_retries=3
)

class ExtractedTerm(BaseModel):
    term: str = Field(description="Exact term from source text")
    span: str = Field(description="Exact quote (20-50 words) containing the term")
    confidence: str = Field(pattern="^(HIGH|MEDIUM|LOW)$")

class ExtractionResult(BaseModel):
    terms: List[ExtractedTerm] = Field(max_length=15)

# Usage
result = client.messages.create(
    model="claude-3-5-haiku-latest",
    max_tokens=2000,
    temperature=0,
    messages=[...],
    response_model=ExtractionResult
)
```

### 5.2 Strategy F: Span Verification Validator

```python
class ExtractedTerm(BaseModel):
    term: str
    span: str
    confidence: str
    
    @field_validator('span')
    @classmethod
    def span_must_exist(cls, v: str, info: ValidationInfo):
        source = info.context.get('source_text', '')
        if v not in source:
            raise ValueError(f"Span not found in source. Extract EXACT substrings only.")
        return v
    
    @field_validator('term')
    @classmethod
    def term_must_be_in_span(cls, v: str, info: ValidationInfo):
        span = info.data.get('span', '')
        if v.lower() not in span.lower():
            raise ValueError(f"Term '{v}' not found in its span.")
        return v

# Usage with validation_context
result = client.messages.create(
    ...,
    response_model=ExtractionResult,
    validation_context={"source_text": chunk_text}
)
```

### 5.3 Strategy G: Self-Consistency Voting

```python
from collections import Counter

def self_consistency_extract(chunk_text: str, n_samples: int = 10, 
                            agreement_threshold: float = 0.7):
    all_terms = []
    
    for i in range(n_samples):
        result = client.messages.create(
            model="claude-3-5-haiku-latest",
            temperature=0.8,  # High for diversity
            top_p=0.95,
            messages=[...],
            response_model=ExtractionResult,
            validation_context={"source_text": chunk_text}
        )
        terms = {t.term.lower() for t in result.terms}
        all_terms.append(terms)
    
    # Count occurrences
    term_counts = Counter()
    for term_set in all_terms:
        term_counts.update(term_set)
    
    # Keep terms with >= threshold agreement
    min_count = int(n_samples * agreement_threshold)
    high_confidence = {term for term, count in term_counts.items() 
                       if count >= min_count}
    
    return high_confidence
```

### 5.4 Strategy H: Multi-Pass Extraction

```python
def multi_pass_extract(chunk_text: str):
    all_terms = set()
    
    # Pass 1: Initial extraction
    result_1 = extract_with_prompt(chunk_text, PROMPT_PASS_1)
    all_terms.update(result_1)
    
    # Pass 2: "What did I miss?"
    prompt_2 = f"""Previously extracted: {list(all_terms)}
    
    Review the text again. What Kubernetes terms did I MISS?
    Focus on:
    - Abbreviations and acronyms
    - Implicit references
    - Multi-word terms
    - Technical jargon
    
    Text: {chunk_text}
    
    Additional terms only:"""
    
    result_2 = extract_with_prompt(chunk_text, prompt_2)
    all_terms.update(result_2)
    
    # Pass 3: Category sweep
    prompt_3 = f"""Already found: {list(all_terms)}
    
    Final sweep for specific categories:
    - Resource types (Pod, Service, Deployment, etc.)
    - API objects and fields
    - Commands and tools (kubectl, kubeadm)
    - Error states and status codes
    - Configuration options
    
    Text: {chunk_text}
    
    Any additional terms in these categories?"""
    
    result_3 = extract_with_prompt(chunk_text, prompt_3)
    all_terms.update(result_3)
    
    return all_terms
```

### 5.5 Strategy I: Combined Pipeline

```python
def combined_extract(chunk_text: str):
    """
    Full pipeline: Multi-pass + Self-consistency + Span verification
    """
    # Step 1: Multi-pass extraction (high recall)
    multi_pass_terms = multi_pass_extract(chunk_text)
    
    # Step 2: Self-consistency voting on multi-pass results
    # Run N=10 extractions with span verification
    voted_terms = self_consistency_extract(
        chunk_text, 
        n_samples=10, 
        agreement_threshold=0.5  # Lower threshold since multi-pass already filtered
    )
    
    # Step 3: Final span verification on voted terms
    verified_terms = []
    for term in voted_terms:
        if term.lower() in chunk_text.lower():
            verified_terms.append(term)
    
    return verified_terms
```

---

## 6. Success Criteria

### 6.1 PASS (All Must Be True)

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| At least one strategy achieves Precision **>95%** | Required | Production-ready quality |
| At least one strategy achieves Recall **>85%** | Required | Must catch most terms |
| At least one strategy achieves Hallucination **<1%** | Required | Near-zero fabrication |
| Best configuration meets **all three** | Required | Single viable configuration |

### 6.2 PARTIAL PASS

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| Best achieves P>90%, R>80%, H<3% | Acceptable | Close to target, refine further |
| Significant improvement over POC-1 | +10% on any metric | Techniques are directionally correct |

### 6.3 FAIL

| Criterion | Threshold | Implication |
|-----------|-----------|-------------|
| No strategy achieves P>85% | Below minimum | LLM extraction unreliable |
| No strategy achieves H<5% | Unacceptable | Span verification ineffective |
| No improvement over POC-1 | 0% gain | Techniques don't work |

---

## 7. Execution Phases

### Phase 1: Environment Setup
- Install instructor library
- Copy POC-1 ground truth and utilities
- Verify Instructor + Anthropic integration

### Phase 2: Strategy Implementation
- Implement all 5 strategies (E, F, G, H, I)
- Unit test each strategy on 3 sample chunks

### Phase 3: Main Experiment
- Run 1,350 extractions
- Save raw results with full metadata

### Phase 4: Analysis
- Calculate metrics per strategy
- Statistical comparison vs POC-1 baselines
- Generate RESULTS.md

---

## 8. Checkpoint Artifacts

| Phase | Artifact | Required Fields |
|-------|----------|-----------------|
| 1 | `phase-1-setup.json` | instructor_version, models_verified |
| 2 | `phase-2-implementation.json` | strategies_implemented, unit_test_results |
| 3 | `phase-3-raw-results.json` | results[] (1350 entries) |
| 4 | `phase-4-final-metrics.json` | metrics_by_strategy, hypothesis_verdicts |

---

## 9. Dependencies

| Dependency | Purpose | Installation |
|------------|---------|--------------|
| instructor | Structured output | `uv add instructor` |
| anthropic | Claude API | `uv add anthropic` |
| pydantic | Validation | (included with instructor) |
| rapidfuzz | Fuzzy matching | `uv add rapidfuzz` |

---

## 10. Risk Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Self-consistency too slow | High | Early stopping when >50% majority |
| Instructor retries exceed budget | Medium | Cap at max_retries=3 |
| Span verification too strict | Medium | Test fuzzy matching fallback |
| Multi-pass increases hallucination | Medium | Apply span verification to all passes |

---

*Specification Version: 1.0*
*Created: 2026-02-03*
*Builds on: POC-1 LLM Term Extraction Guardrails*
