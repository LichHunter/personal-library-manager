# Investigation Draft: POC-1c Scalable NER

## Problem Statement
Current v6_spanfix strategy relies on 176+ manually curated vocabulary terms that don't scale.

## Key Research Findings

### 1. Where is the bottleneck?
- **FILTERING causes 80% of FPs** (Phase 7 Must-extract seeding)
- **EXTRACTION causes 33% of FNs** (missed unusual formats)
- **Context validation is working correctly** (100% accuracy, 0 false negatives)

The vocabulary lists (MUST_EXTRACT_SEEDS, CONTEXT_VALIDATION_BYPASS) are the FP factory.

### 2. Data Leakage Concern — RESOLVED
The SO NER dataset has proper train/dev/test splits:
- **train.txt**: 741 documents (use for retrieval)
- **test.txt**: 249 documents (use for evaluation)
- **Zero overlap** by design (different question IDs)

Safe to use train.txt as retrieval corpus without leakage.

### 3. Alternative Approaches Identified

| Approach | Zero-Shot? | Needs Examples? | Research |
|----------|------------|-----------------|----------|
| **Retrieval-augmented** | No | Yes (train.txt) | BANER, GEIC |
| **SLIMER structured prompting** | Yes | No | ACL 2024 |
| Self-consistency | Yes | No | Various |
| Confidence calibration | Partial | ~2K calibration | NAACL |
| Fine-tuning small model | No | Yes | Standard |

### 4. Chosen Approaches for POC-1c

**Approach A: Retrieval-augmented few-shot**
- Embed 741 train.txt documents
- For each test doc, retrieve top-5 similar train docs
- Use their (text, GT_entities) as few-shot examples
- Include Jaccard filtering to prevent near-duplicate contamination

**Approach B: SLIMER structured prompting**
- Rich entity type definitions (8 categories)
- Detailed annotation guidelines
- Chain-of-thought reasoning
- Zero external examples needed

### 5. Expected Outcomes

**Hypothesis**:
- Retrieval may have higher recall (sees actual GT patterns)
- SLIMER may have higher precision (no overfitting to examples)
- Both should scale better than vocabulary lists

**Success criteria**:
- Match or beat iter 29 baseline (P≥91%, R≥91%, H≤9%)
- Zero vocabulary lists in implementation
- Clear winner identified

## Decision Log

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Retrieval corpus | train.txt (741 docs) | Proper separation, no leakage |
| Embedding model | all-MiniLM-L6-v2 | Fast, good quality |
| Vector search | FAISS (IndexFlatIP) | Simple, exact search |
| Near-duplicate filter | Jaccard (n=13, threshold=0.8) | Standard approach |
| SLIMER entity types | 8 categories | Based on SO NER types |

## Open Questions

None — ready to proceed with implementation.

## Files to Create

See `.sisyphus/plans/poc-1c-scalable-ner.md` for complete plan.
