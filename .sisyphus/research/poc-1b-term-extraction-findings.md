# POC-1b: LLM Term Extraction - Findings and Term Definition

> **Status**: Investigation complete
> **Date**: 2026-02-05
> **Related**: poc/poc-1b-llm-extraction-improvements/

## Executive Summary

POC-1b investigated LLM-based term extraction for the RAG pipeline's slow system. The key finding is that **the original targets (95%+ recall, <10% hallucination) are harder to achieve than expected**, but we discovered critical insights about what "hallucination" actually means and how to define "terms" properly.

## Key Findings

### 1. Small Chunks Are Essential

Extraction on small semantic chunks (50-300 words) **dramatically outperforms** full-document extraction:

| Approach | Recall | Notes |
|----------|--------|-------|
| Full document extraction | 53-68% | Attention dilution across long text |
| Small chunk extraction | 88-97% | Focused extraction per semantic unit |

### 2. The Precision-Recall Tradeoff

No single strategy achieved both 95%+ recall AND <10% hallucination:

| Strategy | Recall | "Hallucination" | Notes |
|----------|--------|-----------------|-------|
| exhaustive_sonnet | 97.3% | 54.2% | Highest recall |
| ensemble_verified | 88.9% | 10.7% | Best balance |
| ensemble_sonnet_verify | 70.9% | 5.6% | Very low "hallucination" |
| vote_3 | 75.1% | 2.2% | Highest precision |

### 3. "Hallucination" Is Misleading

**Critical insight**: 53% of measured "hallucinations" are actually **valid technical terms** that the ground truth missed.

Analysis showed:
- Ground truth created by Opus was too conservative
- Many "false positives" like `Cluster Architecture`, `memory`, `fault-tolerance` ARE valid
- True hallucination (fabricated terms) is much lower due to strict span verification
- Real hallucination rate ≈ reported rate × 0.47

### 4. Term Definition Was Undefined

Root cause of inconsistent results: **We never defined what a "term" actually is**.

## Proposed Term Definition

### Core Definition

> **A term is a word or phrase that has "bridging potential"** - meaning it either:
> 1. Is something users search for (user language)
> 2. Is what documentation uses for what users search for (technical language)  
> 3. Connects these two worlds through synonym relationships

### The Key Test

> *"Would a user searching for this term (or its synonyms) expect to find documentation about a specific technical concept, resource, or behavior?"*

### Term Types

| Type | Description | Priority | Examples |
|------|-------------|----------|----------|
| **Technical Entity** | Named resources, components, APIs, config fields | HIGH | `Pod`, `kubelet`, `CrashLoopBackOff`, `replicas` |
| **Domain Concept** | Abstract ideas with domain-specific meaning | HIGH | `control plane`, `service discovery`, `namespace` |
| **Symptom/Behavior** | Observable states, errors, behaviors | HIGHEST | `keeps restarting`, `OOMKilled`, `pending` |
| **Contextual Term** | Generic terms WITH domain-relevant meaning | LOW | `memory` (→ resource limits), `Linux` (→ prerequisites) |

### What to EXCLUDE

1. **Structural/metadata**: `content_type`, `overview`, `section_2`, YAML keys not relevant to search
2. **Document headings AS headings**: "Cluster Architecture" is a heading, not a term (but `cluster` IS a term)
3. **Stop words**: `the`, `is`, `using`, `how to`
4. **Overly generic without domain meaning**: `system`, `application` (unless contextually specific)
5. **Complete sentences or instructional phrases**

## Recommended Production Configuration

Based on findings, for the slow system:

```python
def extract_terms_slow_system(chunk_content: str) -> list[str]:
    """
    High-recall extraction with LLM verification.
    
    Expected performance (with proper term definition):
    - Recall: ~90%
    - True hallucination: <5%
    """
    # Step 1: High-recall ensemble extraction
    terms = set()
    terms.update(extract_simple(chunk_content, "claude-sonnet"))
    terms.update(extract_exhaustive(chunk_content, "claude-haiku"))
    
    # Step 2: Strict span verification (blocks fabricated terms)
    terms = [t for t in terms if exists_in_text(t, chunk_content)]
    
    # Step 3: LLM verification against term definition
    terms = verify_with_llm(terms, chunk_content, TERM_DEFINITION_PROMPT)
    
    return terms
```

## Next Steps

1. **Create calibration set**: 20-30 terms manually labeled with the new definition
2. **Regenerate ground truth**: Using proper term definition (not conservative Opus)
3. **Rerun experiments**: Measure against properly-defined ground truth
4. **Integrate into pipeline**: Once validated, integrate into slow system

## Files Created

```
poc/poc-1b-llm-extraction-improvements/
├── test_small_chunk_extraction.py    # Small chunk experiment
├── test_combined_strategies.py       # Combined strategy tests
├── test_fast_combined.py             # Optimized combined tests
├── test_hybrid_final.py              # Hybrid strategy tests
├── analyze_false_positives.py        # False positive analysis
├── artifacts/
│   ├── small_chunk_ground_truth.json
│   ├── small_chunk_results.json
│   ├── fast_combined_results.json
│   └── hybrid_final_results.json
└── RESULTS.md
```

## Lessons Learned

1. **Define terms before measuring**: Can't measure extraction quality without clear definition
2. **Ground truth quality matters**: Conservative GT inflates "hallucination" metrics
3. **Span verification is essential**: Blocks true fabrication effectively
4. **Small chunks unlock higher recall**: 50-300 words is the sweet spot
5. **Multiple strategies help**: Ensemble approaches maximize recall
