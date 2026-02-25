# Problem Statement: Keyword and Entity Extraction for RAG Pipeline

## 1. Core Task

**Goal**: Extract meaningful keywords and entities from any piece of text to improve retrieval in a RAG (Retrieval-Augmented Generation) pipeline.

**Domain**: Kubernetes documentation

**Current Retrieval Performance**: 36% Hit@5

---

## 2. The Vocabulary Mismatch Problem

Users search using **colloquial language**, but documentation uses **technical terminology**.

| User Query | Document Contains |
|------------|-------------------|
| "pod keeps restarting" | "CrashLoopBackOff" |
| "out of memory" | "OOMKilled" |
| "can't pull image" | "ImagePullBackOff" |

**Result**: Queries fail to match relevant documents because the words don't overlap.

---

## 3. Current Extraction Setup

| Tool | Purpose | Limitation |
|------|---------|------------|
| **YAKE** | Keyword extraction | Statistical - doesn't know what a "domain term" is |
| **spaCy** | Entity extraction | General NER - trained on news, not K8s terminology |

---

## 4. Specific Problems with Current Setup

### Problem A: YAKE Extracts Statistically Significant Phrases, Not Domain Terms

YAKE uses statistical features (frequency, position, capitalization) to rank phrases. It has no concept of "this is a Kubernetes term."

**Consequence**: 
- May extract "the container" (common phrase) but miss "CrashLoopBackOff" (single mention, weird position)
- Cannot distinguish domain-relevant terms from generic phrases

### Problem B: spaCy Recognizes Generic Entity Types, Not Domain-Specific Ones

spaCy's NER model recognizes PERSON, ORG, GPE, DATE, etc. It does not recognize:
- K8s resource types (Pod, Deployment, Service)
- K8s error states (CrashLoopBackOff, OOMKilled)
- K8s commands (kubectl apply, helm install)
- Configuration fields (spec.containers.resources.limits)

**Consequence**: Domain-critical entities are missed entirely.

### Problem C: No Synonym/Alias Mapping

Even when terms ARE extracted correctly, there's no mapping between:
- Technical term -> colloquial equivalents
- Colloquial query -> technical term

**Consequence**: Extracted terms don't bridge the vocabulary gap.

### Problem D: No Mechanism to Handle Novel Content

If a document contains terminology not seen before, there's no way to:
1. Detect that this content is "new"
2. Trigger additional processing to understand it
3. Learn from it for future documents

**Consequence**: System cannot adapt or improve over time.

---

## 5. Requirements for a Solution

Any solution must address:

| Requirement | Description |
|-------------|-------------|
| **R1** | Extract domain-relevant terms (not just statistically significant phrases) |
| **R2** | Recognize domain-specific entity types (K8s resources, errors, commands) |
| **R3** | Map technical terms to colloquial equivalents (and vice versa) |
| **R4** | Detect when content contains unknown/novel terminology |
| **R5** | Be reliable (low hallucination, verifiable outputs) |
| **R6** | Be efficient (fast enough for batch processing, ideally no GPU required) |

---

## 6. Constraints

| Constraint | Description |
|------------|-------------|
| **C1** | One-time ingestion (no re-ingestion planned) |
| **C2** | No user behavior data available (no query logs, clicks) |
| **C3** | Solution should be maintainable (not a black box) |

---

## 7. Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Hit@5 | 36% | 55%+ |

---

## 8. Open Questions (To Be Answered by Solution)

1. How to determine if an extracted term is domain-relevant vs generic?
2. How to extract domain-specific entities without training a custom NER model?
3. How to generate/acquire term-alias mappings at scale?
4. How to detect "novel" content when there's no reference corpus?
5. How to validate extraction quality without manual review of every document?

---

*This document defines the problem. Solutions will be evaluated against requirements R1-R6 and constraints C1-C3.*
