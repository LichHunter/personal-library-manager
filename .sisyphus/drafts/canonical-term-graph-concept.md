# Canonical Term Graph: Concept Formalization

**Date**: 2026-01-27
**Status**: Concept draft - needs research validation
**Purpose**: Define a vocabulary mapping system where synonyms link to canonical "base words"

---

## Core Concept

### The Problem with Flat Synonym Lists

Traditional synonym systems treat all terms as equivalent:
```
[HPA, horizontal pod autoscaler, pod autoscaler, autoscaler] ← flat, no anchor
```

**Issues:**
- No "ground truth" term to anchor the cluster
- Hard to merge overlapping clusters
- No hierarchy when terms have different specificity levels
- Context-dependent meanings not handled

### Proposed Solution: Canonical Term Graph

Every synonym cluster has a **canonical base word** (ground truth) that all variants link to:

```
                    ┌─────────────────┐
                    │  CANONICAL TERM │
                    │  "autoscaling"  │
                    │  (base word)    │
                    └────────┬────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
   ┌───────────┐      ┌───────────┐      ┌───────────┐
   │   "HPA"   │      │"scale pods"│     │"autoscaler"│
   │ [k8s,abbr]│      │ [user,verb]│     │ [k8s,noun] │
   └───────────┘      └───────────┘      └───────────┘
```

### Key Properties

1. **Canonical Term**: The "base word" that represents the concept
   - Selected during ingestion (LLM-assisted)
   - Typically the most formal/complete term from the documentation
   - Example: "HorizontalPodAutoscaler" is canonical, "HPA" is alias

2. **Synonyms**: Terms that map TO the canonical term
   - User terms (how people search)
   - Abbreviations (HPA, k8s, etc.)
   - Alternate phrasings ("scale pods automatically")

3. **Context Tags**: Disambiguate terms with multiple meanings
   - Domain tags: `[k8s]`, `[aws]`, `[general]`
   - Type tags: `[abbr]`, `[verb]`, `[noun]`, `[symptom]`
   - Specificity: `[broad]`, `[narrow]`, `[exact]`

4. **Multi-Level Hierarchy**: When clusters overlap, create meta-canonical terms
   ```
                        "scaling"           ← Level 2 (meta-canonical)
                           │
              ┌────────────┼────────────┐
              │            │            │
        "autoscaling"  "manual-scaling" "cluster-scaling"  ← Level 1 (canonical)
              │
      ┌───────┼───────┐
      │       │       │
    "HPA"  "VPA"  "KEDA"  ← Level 0 (synonyms/specifics)
   ```

---

## Data Sources

### 1. Ingested Documents (Primary)
- Extract terms during document ingestion
- LLM identifies canonical terms vs variants
- Build graph from YOUR corpus, not generic thesauri
- **Rationale**: Users search for terms that appear in your docs

### 2. Training Data Questions (For Known Domains)
- Use question datasets (kubefix, Stack Overflow, etc.)
- Extract user vocabulary from questions
- Map to document terms found in answers
- **Rationale**: Questions reveal how users actually phrase things

### 3. Failed Search Log (Continuous Learning)
- Store queries that returned poor results
- Periodically analyze with LLM (Opus-level for quality)
- Add missing synonyms to graph
- **Rationale**: Real user behavior reveals vocabulary gaps

---

## Graph Construction Process

### Phase 1: Document Ingestion (Per-Document)

```
Document → LLM Extraction → Term Candidates
                               │
                               ▼
                    ┌─────────────────────┐
                    │ For each term:      │
                    │ • Is it canonical?  │
                    │ • Is it a synonym?  │
                    │ • What's the base?  │
                    │ • What tags apply?  │
                    └─────────────────────┘
                               │
                               ▼
                    Add to Graph (merge with existing)
```

**LLM Prompt Concept:**
```
Given this document chunk about Kubernetes:
"{chunk_text}"

Extract technical terms and classify them:

For each term, determine:
1. Is this a CANONICAL term (the formal/official name)?
2. Or is it a SYNONYM/VARIANT of another term?
3. If synonym, what is its canonical base?
4. What context tags apply? (domain, type, specificity)

Output as structured JSON.
```

### Phase 2: Cluster Merging (Cross-Document)

**Problem**: Different documents may identify different canonical terms for the same concept.

**Solution**: Meta-canonical selection

```
Document A says: "HorizontalPodAutoscaler" is canonical
Document B says: "HPA" is canonical (abbreviation-heavy doc)

Merge Process:
1. Detect overlap (both have similar synonym sets)
2. LLM decides which is "more canonical"
3. Winner becomes canonical, loser becomes synonym
4. OR: Create hierarchy if they represent different specificity levels
```

### Phase 3: Hierarchy Construction

When multiple canonical terms overlap:

```
Detected canonical terms:
- "HorizontalPodAutoscaler" (specific K8s resource)
- "autoscaling" (general concept)
- "scaling" (even more general)

LLM determines hierarchy:
scaling (L2)
  └── autoscaling (L1)
        └── HorizontalPodAutoscaler (L0, most specific)
```

### Phase 4: Continuous Learning

```
Failed Search: "pods won't grow"
                    │
                    ▼
            Store in Failed Log
                    │
                    ▼
        Periodic Opus Analysis:
        "pods won't grow" → likely means "autoscaling not working"
                    │
                    ▼
        Add to graph: "pods won't grow" → SYMPTOM_OF → "autoscaling"
```

---

## Graph Schema (Conceptual)

### Node Types

```typescript
interface CanonicalTerm {
  id: string;
  term: string;
  level: number;  // 0 = most specific, higher = more general
  tags: string[];  // ["k8s", "resource", "noun"]
  definition?: string;  // from docs
  doc_refs: string[];  // which docs mention this
}

interface SynonymTerm {
  id: string;
  term: string;
  canonical_id: string;  // links to CanonicalTerm
  tags: string[];  // ["user", "abbr", "verb"]
  confidence: number;  // 0.0-1.0
  source: "ingestion" | "training_data" | "failed_search";
}
```

### Edge Types

```typescript
interface TermRelation {
  source_id: string;
  target_id: string;
  relation_type: 
    | "SYNONYM_OF"      // synonym → canonical
    | "BROADER_THAN"    // canonical → more-general-canonical
    | "NARROWER_THAN"   // canonical → more-specific-canonical
    | "SYMPTOM_OF"      // user problem description → canonical concept
    | "RELATED_TO";     // associated but not synonym
  weight: number;  // 0.0-1.0
  tags: string[];  // context tags for this specific relation
}
```

### Example Graph State

```json
{
  "canonical_terms": [
    {
      "id": "c_001",
      "term": "HorizontalPodAutoscaler",
      "level": 0,
      "tags": ["k8s", "resource", "noun"],
      "definition": "Automatically scales workload resources...",
      "doc_refs": ["docs/hpa.md", "docs/autoscaling.md"]
    },
    {
      "id": "c_002", 
      "term": "autoscaling",
      "level": 1,
      "tags": ["k8s", "concept", "noun"],
      "doc_refs": ["docs/autoscaling.md"]
    }
  ],
  "synonyms": [
    {
      "id": "s_001",
      "term": "HPA",
      "canonical_id": "c_001",
      "tags": ["k8s", "abbr"],
      "confidence": 0.99,
      "source": "ingestion"
    },
    {
      "id": "s_002",
      "term": "scale pods automatically",
      "canonical_id": "c_001",
      "tags": ["user", "verb_phrase"],
      "confidence": 0.85,
      "source": "training_data"
    },
    {
      "id": "s_003",
      "term": "pods won't scale",
      "canonical_id": "c_001",
      "tags": ["user", "symptom"],
      "confidence": 0.80,
      "source": "failed_search"
    }
  ],
  "relations": [
    {
      "source_id": "c_001",
      "target_id": "c_002",
      "relation_type": "NARROWER_THAN",
      "weight": 0.95,
      "tags": ["k8s"]
    }
  ]
}
```

---

## Query Expansion Using the Graph

### At Search Time

```
User Query: "pods won't scale"
     │
     ▼
┌─────────────────────────────────────┐
│ 1. GRAPH LOOKUP                     │
│    "pods won't scale" found as      │
│    synonym of "HorizontalPodAutoscaler" │
│    with tags: [user, symptom]       │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 2. EXPAND TO CANONICAL + SIBLINGS   │
│    canonical: "HorizontalPodAutoscaler" │
│    siblings: "HPA", "pod autoscaler"│
│    parent: "autoscaling"            │
└─────────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────────┐
│ 3. CONTEXT-AWARE FILTERING          │
│    User context suggests k8s domain │
│    Filter to tags containing [k8s]  │
└─────────────────────────────────────┘
     │
     ▼
Expanded Query: 
  "pods won't scale" OR "HorizontalPodAutoscaler" OR "HPA" 
  OR "autoscaling"
  (with appropriate weights)
```

---

## Open Questions (For Research)

1. **Canonical Selection Algorithm**
   - How does LLM decide which term is "most canonical"?
   - What heuristics help? (frequency in docs, formality, completeness)
   - How to handle ties?

2. **Cluster Merging Strategy**
   - When do we merge vs create hierarchy?
   - How to detect that two canonical terms are actually the same concept?
   - Threshold for similarity?

3. **Tag Taxonomy**
   - What tags are most useful for context disambiguation?
   - Should tags be hierarchical? (domain/subdomain)
   - How many tags per term is practical?

4. **Failed Search Analysis**
   - How often to run Opus analysis? (daily? weekly?)
   - Minimum failed search count before analysis?
   - Human review before adding to graph?

5. **Multi-Level Hierarchy Depth**
   - How many levels are useful? (2? 3? more?)
   - When does hierarchy add value vs noise?
   - How to prevent over-generalization?

6. **Performance Considerations**
   - Graph size limits for in-memory operation?
   - Caching strategy for frequent lookups?
   - Incremental update vs full rebuild?

---

## Next Steps

1. **Research**: Validate concept against existing approaches
   - Are there systems that use canonical term anchoring?
   - How do knowledge graphs handle "preferred label" selection?
   - What does SKOS say about `skos:prefLabel` selection?

2. **Prototype**: Build minimal viable graph
   - ~50 K8s terms manually curated
   - Test expansion quality
   - Measure Hit@5 improvement

3. **LLM Prompt Engineering**: Design extraction prompts
   - Test on sample documents
   - Iterate on canonical selection criteria
   - Validate tag assignments

4. **Integration Design**: Plan pipeline integration
   - Where in ingestion does extraction happen?
   - How does graph merge with existing enrichment?
   - Query expansion integration point
