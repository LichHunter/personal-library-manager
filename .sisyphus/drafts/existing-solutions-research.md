# Existing Solutions Research: Canonical Term Graph

**Date**: 2026-01-27
**Purpose**: Identify existing systems that solve parts of the canonical term graph problem
**Status**: Research complete - ready for synthesis

---

## Executive Summary

Your concept of a **Canonical Term Graph** with base words, context tags, and multi-level hierarchy maps closely to several established domains:

| Domain | Key Insight | Relevance |
|--------|-------------|-----------|
| **Terminology Management (TBX/ISO)** | "Preferred term" per concept is THE standard | Direct match to your canonical term idea |
| **Entity Linking (BLINK, spaCy)** | Surface form → canonical entity mapping | Exactly your synonym → base word mapping |
| **SKOS Thesauri** | prefLabel/altLabel + broader/narrower | Multi-level hierarchy + synonyms |
| **Automatic Term Extraction** | LLM-based extraction from corpus | Building graph FROM documents |
| **Failed Query Learning** | Mining reformulations for synonyms | Your continuous learning idea |

**Key Finding**: Your concept is NOT novel in isolation - it's a SYNTHESIS of well-established patterns. This is GOOD - we can leverage existing standards and tools.

---

## 1. TERMINOLOGY MANAGEMENT SYSTEMS

### The "Preferred Term" Concept (ISO 704, TBX)

**This is exactly your "canonical base word" idea.**

**ISO 704/1087 Standard Terminology:**
- **Preferred designation**: The primary term for a concept
- **Admitted designation**: Acceptable synonym
- **Deprecated designation**: Discouraged term

**TBX (TermBase eXchange) Implementation:**
```xml
<conceptEntry id="c42">
  <langSec xml:lang="en">
    <termSec>
      <term>HorizontalPodAutoscaler</term>
      <termNote type="normativeAuthorization">preferredTerm-admn-sts</termNote>
    </termSec>
    <termSec>
      <term>HPA</term>
      <termNote type="normativeAuthorization">admittedTerm-admn-sts</termNote>
    </termSec>
    <termSec>
      <term>pod autoscaler</term>
      <termNote type="normativeAuthorization">admittedTerm-admn-sts</termNote>
    </termSec>
  </langSec>
</conceptEntry>
```

**Relevance to Your Design:**
- `conceptEntry` = Your canonical concept
- `preferredTerm` = Your base word
- `admittedTerm` = Your synonyms
- `deprecatedTerm` = Could be used for failed search terms that shouldn't expand

### Automatic Term Extraction Tools

**TermSuite** (Java, open source):
- Extracts terms from corpus
- Detects **term variants** (morphological, syntactic, semantic)
- Groups variants automatically
- **Key feature**: Variant gathering algorithm

**LlamATE / LLM-based ATE (2025-2026)**:
- Use LLMs to extract terms from documents
- Domain-adaptive with in-context learning
- Reduces need for manual annotation
- **Matches your "extract during ingestion" approach**

### Recommendation
**Adopt TBX/ISO 704 concepts** for your data model. Use `preferred/admitted/deprecated` status.

---

## 2. ENTITY LINKING SYSTEMS

### The Problem They Solve
Map text mentions (surface forms) → canonical entities in a knowledge base.

**This is EXACTLY your problem**: user query terms → canonical document terms.

### Key Systems Analyzed

| System | Architecture | Custom KB Support | Learning |
|--------|--------------|-------------------|----------|
| **spaCy EntityLinker** | Candidate gen + neural ranker | YES - InMemoryLookupKB | Needs pre-built KB |
| **BLINK (Facebook)** | Bi-encoder + cross-encoder | YES - JSONL entities | Needs pre-built KB |
| **GENRE** | Autoregressive generation | YES - just need names | Needs entity list |
| **Gilda** | String matching + ML disambiguation | YES - custom grounders | Biomedical focus |

### spaCy InMemoryLookupKB (Most Relevant)

**Data Model:**
```python
from spacy.kb import InMemoryLookupKB

kb = InMemoryLookupKB(vocab, entity_vector_length=300)

# Add canonical entity
kb.add_entity(
    entity="HPA",  # Canonical ID
    freq=100,      # Prior probability
    entity_vector=embedding  # For disambiguation
)

# Add aliases (surface forms)
kb.add_alias(
    alias="horizontal pod autoscaler",
    entities=["HPA"],
    probabilities=[1.0]
)
kb.add_alias(
    alias="pod autoscaler", 
    entities=["HPA"],
    probabilities=[1.0]
)
```

**Relevance:**
- Entity = Your canonical term
- Alias = Your synonyms
- Probabilities = Your weights
- Entity vector = Context-aware disambiguation

### BLINK Two-Stage Architecture

```
User Query: "pods won't scale"
     │
     ▼
┌─────────────────────────────────┐
│ Stage 1: BI-ENCODER (Fast)      │
│ Embed query, find k=100 similar │
│ entities by vector similarity   │
└─────────────────────────────────┘
     │
     ▼
┌─────────────────────────────────┐
│ Stage 2: CROSS-ENCODER (Precise)│
│ Score each candidate with full  │
│ context → select best match     │
└─────────────────────────────────┘
     │
     ▼
Canonical Entity: "HorizontalPodAutoscaler"
```

### Recommendation
**Use spaCy's KnowledgeBase pattern** for your term graph. It's production-ready and handles:
- Canonical entities with IDs
- Multiple aliases per entity
- Prior probabilities (weights)
- Embedding-based disambiguation

---

## 3. SKOS THESAURI (W3C Standard)

### Perfect Match for Your Multi-Level Hierarchy

**SKOS Properties:**
```turtle
:hpa a skos:Concept ;
    skos:prefLabel "HorizontalPodAutoscaler"@en ;  # Canonical
    skos:altLabel "HPA"@en ;                        # Synonym
    skos:altLabel "pod autoscaler"@en ;             # Synonym
    skos:hiddenLabel "scale pods"@en ;              # User term (hidden from display)
    skos:broader :autoscaling ;                     # Parent concept
    skos:narrower :hpa-v2 ;                         # Child concept
    skos:related :metrics-server ;                  # Associated
    skos:inScheme :kubernetes-vocabulary ;          # Domain tag
    skos:definition "Automatically scales..."@en .
```

**Label Hierarchy:**
- `skos:prefLabel`: Your canonical base word (ONE per language)
- `skos:altLabel`: Your synonyms (multiple allowed)
- `skos:hiddenLabel`: User terms for search only (not displayed)

**Concept Hierarchy:**
- `skos:broader`: Parent concept (more general)
- `skos:narrower`: Child concept (more specific)
- `skos:related`: Associated but not hierarchical

**Domain/Context:**
- `skos:inScheme`: Links concept to a ConceptScheme (domain)
- Multiple schemes = multiple contexts

### Your Multi-Level Hierarchy in SKOS

```
                    skos:ConceptScheme
                    "kubernetes-vocabulary"
                           │
                           │ inScheme
                           ▼
                    ┌─────────────┐
                    │  "scaling"  │  Level 2 (broad)
                    │  prefLabel  │
                    └──────┬──────┘
                           │ narrower
                           ▼
                    ┌─────────────┐
                    │"autoscaling"│  Level 1 (mid)
                    │  prefLabel  │
                    │ altLabel:   │
                    │ "auto-scale"│
                    └──────┬──────┘
                           │ narrower
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌─────────┐  ┌─────────┐  ┌─────────┐
        │  "HPA"  │  │  "VPA"  │  │ "KEDA"  │  Level 0 (specific)
        │prefLabel│  │prefLabel│  │prefLabel│
        │altLabel:│  │         │  │         │
        │"pod     │  │         │  │         │
        │autoscaler"│ │         │  │         │
        └─────────┘  └─────────┘  └─────────┘
```

### Recommendation
**Use SKOS as your data model foundation.** It provides:
- Canonical term (prefLabel)
- Synonyms (altLabel)
- Hidden search terms (hiddenLabel)
- Multi-level hierarchy (broader/narrower)
- Domain separation (inScheme)

---

## 4. WIKIDATA MODEL

### How Wikidata Handles Exactly Your Problem

**Structure per entity:**
- **Q-number**: Unique canonical ID (e.g., Q42)
- **Label**: Primary name per language
- **Aliases**: Multiple alternative names per language
- **Description**: Disambiguating phrase

**Example:**
```json
{
  "id": "Q12345",
  "labels": {
    "en": "HorizontalPodAutoscaler"
  },
  "aliases": {
    "en": ["HPA", "pod autoscaler", "horizontal pod autoscaler"]
  },
  "descriptions": {
    "en": "Kubernetes resource that automatically scales workloads"
  }
}
```

**Disambiguation:**
- Same label, different entities → distinguished by description
- "Mercury" has Q308 (planet), Q925 (element), Q193 (deity)
- Each has unique description for disambiguation

**Relevance:**
- Q-number = Your canonical ID
- Label = Your base word
- Aliases = Your synonyms
- Description = Context for disambiguation

---

## 5. AUTOMATIC TAXONOMY CONSTRUCTION

### Building the Graph FROM Documents

**Hearst Patterns (Pattern-Based Extraction):**
```
"NP such as NP" → hypernym(NP₁, NP₂)
"NP, including NP" → hypernym(NP₁, NP₂)
"NP is a NP" → hypernym(NP₂, NP₁)
```

**Example from K8s docs:**
```
"Controllers such as HPA, VPA, and cluster autoscaler..."
→ hypernym("Controllers", "HPA")
→ hypernym("Controllers", "VPA")
→ hypernym("Controllers", "cluster autoscaler")
```

**LLM-Based Extraction (State of Art 2026):**

**Chain-of-Layer (CoL) Framework:**
```
Input: Document corpus
Step 1: Extract Level 0 terms (most specific)
Step 2: LLM generates broader Level 1 terms
Step 3: LLM generates even broader Level 2 terms
Step 4: Ensemble ranking to reduce hallucinations
Output: Multi-level taxonomy
```

**EDC Framework (Extract-Define-Canonicalize):**
```
Phase 1: Open information extraction
Phase 2: Schema definition (LLM-generated)
Phase 3: Post-hoc canonicalization
```

**Relevance:**
- Hearst patterns: Bootstrap initial hierarchy
- LLM extraction: Identify canonical vs synonym terms
- CoL: Build your multi-level structure automatically

### Recommendation
**Hybrid approach:**
1. Hearst patterns for high-confidence IS-A relations
2. LLM for synonym identification and canonicalization
3. Human review for validation

---

## 6. FAILED QUERY LEARNING

### Systems That Learn from Search Failures

**Query Reformulation Mining:**
```
Session: 
  Query 1: "pods won't scale" → 0 clicks
  Query 2: "HPA not working" → 3 clicks

Inference:
  "pods won't scale" → SYNONYM_OF → "HPA"
```

**Click-Through Analysis:**
```
Query A: "scale pods" → clicks doc123
Query B: "HPA config" → clicks doc123

Inference:
  "scale pods" RELATED_TO "HPA config"
```

**Commercial Implementations:**
- **Amazon A9**: Learns synonyms from purchase patterns
- **Algolia AI**: Suggests synonyms from query refinements
- **Coveo**: Continuous learning from click-through

### Recommendation
**Implement query log analysis:**
1. Store failed queries (low-click or reformulated)
2. Periodic LLM analysis of patterns
3. Suggest new synonyms for human review
4. Add validated synonyms to graph

---

## 7. SYNTHESIS: YOUR CANONICAL TERM GRAPH

### Mapping Your Concept to Existing Standards

| Your Concept | SKOS Equivalent | TBX Equivalent | spaCy Equivalent |
|--------------|-----------------|----------------|------------------|
| Canonical base word | `skos:prefLabel` | `preferredTerm` | Entity ID |
| Synonym | `skos:altLabel` | `admittedTerm` | Alias |
| User search term | `skos:hiddenLabel` | (custom) | Alias |
| Deprecated term | (custom property) | `deprecatedTerm` | (removed) |
| Parent concept | `skos:broader` | (concept relations) | (not native) |
| Child concept | `skos:narrower` | (concept relations) | (not native) |
| Domain tag | `skos:inScheme` | `subjectField` | (custom) |
| Weight | (custom property) | (custom) | Probability |

### Recommended Data Model

```python
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

class TermStatus(Enum):
    PREFERRED = "preferred"    # Canonical base word
    ADMITTED = "admitted"      # Valid synonym
    HIDDEN = "hidden"          # User term (search only)
    DEPRECATED = "deprecated"  # Discouraged term

@dataclass
class Term:
    text: str
    status: TermStatus
    confidence: float  # 0.0-1.0
    source: str  # "ingestion", "training_data", "failed_search"
    tags: List[str]  # Context tags

@dataclass
class Concept:
    id: str  # Unique canonical ID
    level: int  # 0=specific, 1=mid, 2=broad
    domain: str  # e.g., "kubernetes"
    definition: Optional[str]
    terms: List[Term]  # All terms for this concept
    broader: Optional[str]  # Parent concept ID
    narrower: List[str]  # Child concept IDs
    related: List[str]  # Associated concept IDs
    doc_refs: List[str]  # Documents mentioning this concept

    @property
    def preferred_term(self) -> Optional[Term]:
        """Get the canonical base word."""
        return next(
            (t for t in self.terms if t.status == TermStatus.PREFERRED),
            None
        )
    
    @property
    def synonyms(self) -> List[Term]:
        """Get all synonyms (admitted + hidden)."""
        return [t for t in self.terms 
                if t.status in (TermStatus.ADMITTED, TermStatus.HIDDEN)]
```

### SKOS/Turtle Export

```python
def to_skos(concept: Concept) -> str:
    """Export concept to SKOS Turtle format."""
    lines = [f":{concept.id} a skos:Concept ;"]
    
    for term in concept.terms:
        if term.status == TermStatus.PREFERRED:
            lines.append(f'    skos:prefLabel "{term.text}"@en ;')
        elif term.status == TermStatus.ADMITTED:
            lines.append(f'    skos:altLabel "{term.text}"@en ;')
        elif term.status == TermStatus.HIDDEN:
            lines.append(f'    skos:hiddenLabel "{term.text}"@en ;')
    
    if concept.broader:
        lines.append(f'    skos:broader :{concept.broader} ;')
    
    for narrower in concept.narrower:
        lines.append(f'    skos:narrower :{narrower} ;')
    
    lines.append(f'    skos:inScheme :{concept.domain} .')
    
    return '\n'.join(lines)
```

---

## 8. TOOLS TO LEVERAGE

### For Building the Graph

| Tool | Use Case | Integration |
|------|----------|-------------|
| **TermSuite** | Term extraction from corpus | Java, can run as CLI |
| **spaCy + EntityLinker** | Entity linking infrastructure | Python, native |
| **Gilda** | Grounding with custom ontologies | Python, pip install |
| **Protege** | Manual ontology editing | Desktop app |
| **VocBench** | Collaborative SKOS editing | Web-based |

### For Storage

| Option | Pros | Cons |
|--------|------|------|
| **JSON + NetworkX** | Simple, portable, no deps | In-memory only |
| **SQLite + relations** | Persistent, portable | Manual graph queries |
| **RDFLib + Turtle** | Standard SKOS format | Query via SPARQL |
| **Neo4j** | Native graph, rich queries | External service |

### For Query Expansion

| Tool | Integration |
|------|-------------|
| **spaCy KnowledgeBase** | Direct Python API |
| **Elasticsearch synonym_graph** | Query-time expansion |
| **SPARQL** | Query SKOS graph directly |

---

## 9. RECOMMENDED IMPLEMENTATION PATH

### Phase 1: Bootstrap (Week 1-2)
1. **Define SKOS-based data model** (see above)
2. **Extract terms from K8s docs** using TermSuite or LLM
3. **Manual curation** of top ~100 concepts
4. **Store as JSON**, load into NetworkX for queries

### Phase 2: LLM-Assisted Canonicalization (Week 3-4)
1. **LLM prompt** to identify preferred vs admitted terms
2. **LLM prompt** to identify hierarchical relationships
3. **Human review** of LLM suggestions
4. **Expand to ~500 concepts**

### Phase 3: Integration (Week 5-6)
1. **Build spaCy KnowledgeBase** from graph
2. **Integrate with retrieval pipeline**
3. **Measure Hit@5 improvement**
4. **Iterate based on results**

### Phase 4: Continuous Learning (Ongoing)
1. **Log failed queries**
2. **Periodic LLM analysis** of patterns
3. **Human review** of suggested additions
4. **Monthly graph updates**

---

## 10. KEY INSIGHTS

1. **Your concept is well-grounded** in existing standards (TBX, SKOS, ISO 704)
2. **Entity linking provides the architecture** for surface → canonical mapping
3. **SKOS gives you the hierarchy** (broader/narrower) for multi-level concepts
4. **LLM extraction is state-of-art** for building from documents (2025-2026)
5. **Failed query learning is proven** in e-commerce (Amazon, Algolia)

**The innovation in your approach is the COMBINATION:**
- TBX concept of preferred/admitted terms
- SKOS hierarchy for levels
- Entity linking for disambiguation
- LLM extraction from YOUR corpus
- Failed query learning for continuous improvement

**This is not "reinventing the wheel" - it's assembling the right wheels.**

---

## 11. FAILED QUERY LEARNING (Your Continuous Improvement Idea)

### This Is a Proven Pattern

**Amazon's Approach:**
- **QUEEN**: Neural query rewriting (6% improvement in recall)
- **RLQR**: RL-based reformulation (28.6% increase in product coverage)
- **Unsupervised Synonym Extraction**: 85.5% quality with NO human labels

**eBay's Approach:**
- Query reformulation mining from logs
- Intent-aware reformulation (Same/Similar/Inspired Intent)
- **Predicts reformulation BEFORE showing results**

### Data to Capture

| Signal | Description | Confidence |
|--------|-------------|------------|
| **Zero Results** | Query returned nothing | High (vocab gap) |
| **No Clicks** | Results shown, none clicked | Medium (relevance issue) |
| **Query Reformulation** | User rephrased immediately | High (synonym signal) |
| **Co-Click** | Different queries → same doc | Very High (synonym) |
| **Session Co-occurrence** | Queries in same session | Medium (related) |
| **Purchase Overlap** | Queries → same purchase | Very High (synonym) |

### Extraction Methods

**Automatic (Amazon-style):**
```
Query Logs → Behavioral Analysis → Synonym Extraction → Quality Filter (>85%) → Deploy
```

**Semi-Automatic (Your Opus review idea):**
```
Query Logs → Pattern Mining → LLM Analysis (Opus) → Human Review → Add to Graph
```

### Open Source Tools

| Tool | Purpose | GitHub |
|------|---------|--------|
| **RL-Query-Reformulation** | Query rewriting | PraveenSH/RL-Query-Reformulation |
| **SynonymNet** | Entity synonym discovery | czhang99/SynonymNet |
| **markovclick** | Clickstream modeling | ismailuddin/markovclick |
| **Retentioneering** | User path analysis | PyPI: retentioneering |

### Recommended Implementation

**Phase 1: Capture (Immediate)**
```python
@dataclass
class FailedQuery:
    query: str
    timestamp: datetime
    session_id: str
    results_count: int  # 0 = zero results
    clicked: bool
    reformulated_to: Optional[str]  # Next query in session
```

**Phase 2: Analyze (Weekly)**
```python
def analyze_failed_queries(queries: List[FailedQuery]) -> List[SynonymCandidate]:
    candidates = []
    
    # Pattern 1: Reformulation → Original query relates to reformulated
    for q in queries:
        if q.reformulated_to and not q.clicked:
            candidates.append(SynonymCandidate(
                source=q.query,
                target=q.reformulated_to,
                confidence=0.7,
                pattern="reformulation"
            ))
    
    # Pattern 2: Co-click → Queries clicking same docs
    # ... (analyze click patterns)
    
    return candidates
```

**Phase 3: Validate (Monthly)**
```python
def llm_validate(candidates: List[SynonymCandidate], llm="opus") -> List[ValidatedSynonym]:
    prompt = f"""
    Review these potential synonym pairs for Kubernetes documentation:
    
    {format_candidates(candidates)}
    
    For each pair, determine:
    1. Is this a valid synonym in the K8s context?
    2. Which term should be canonical (base word)?
    3. Confidence (0.0-1.0)?
    
    Output as JSON.
    """
    return call_llm(prompt, model=llm)
```

---

## 12. COMPLETE SOLUTION MAPPING

### Your Concept → Existing Standards

| Your Idea | Existing Standard | Tool/Implementation |
|-----------|-------------------|---------------------|
| Canonical base word | SKOS prefLabel, TBX preferredTerm | spaCy EntityLinker |
| Synonyms | SKOS altLabel, TBX admittedTerm | Elasticsearch synonyms |
| User search terms | SKOS hiddenLabel | (search only, not displayed) |
| Context tags | SKOS inScheme, TBX subjectField | Multiple concept schemes |
| Multi-level hierarchy | SKOS broader/narrower | RDFLib, Neo4j |
| Build from docs | Automatic Term Extraction | TermSuite, LLM extraction |
| Bootstrap from Q&A | Training data mining | Amazon QUEEN approach |
| Learn from failures | Query log analysis | Reformulation mining |
| LLM canonicalization | EDC Framework | LangChain, custom prompts |

### Recommended Tech Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    CANONICAL TERM GRAPH                     │
├─────────────────────────────────────────────────────────────┤
│  Data Model: SKOS (prefLabel/altLabel/broader/narrower)     │
│  Storage: JSON → NetworkX (prototype) → Neo4j (production)  │
│  Entity Linking: spaCy KnowledgeBase pattern                │
│  Term Extraction: LLM (Claude) + TermSuite patterns         │
│  Query Expansion: Graph lookup + embedding fallback         │
│  Failed Query Learning: Log → Analyze → Opus review → Add   │
│  Export: SKOS/Turtle for interoperability                   │
└─────────────────────────────────────────────────────────────┘
```

---

## 13. NEXT STEPS

The research is complete. Your concept is **well-validated** by existing standards and implementations. The next steps are:

1. **Finalize data model** (adopt SKOS pattern)
2. **Prototype with ~100 K8s terms** (manual + LLM-assisted)
3. **Integrate with retrieval pipeline** (spaCy KB or custom)
4. **Implement failed query logging** (capture signals)
5. **Measure improvement** (Hit@5 benchmark)
6. **Iterate** based on results
