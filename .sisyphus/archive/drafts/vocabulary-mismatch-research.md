# Vocabulary Mismatch Solutions: Comprehensive Research

**Date**: 2026-01-27
**Purpose**: Research synthesis for addressing the 59% failure rate in realistic questions benchmark
**Scope**: RAG, classical IR, machine translation, e-commerce, medical/legal, conversational AI

---

## Selected Strategies (User Analysis)

Based on research review, the following strategies were identified as promising:

### High Priority (Implement)

| Strategy | Why Selected | Core Mechanism |
|----------|--------------|----------------|
| **LLM-based Query Expansion** | Flexible, no manual maintenance | Maps user language → technical terms via LLM |
| **Synonym Dictionary** | Precise, predictable | Explicit user-term → doc-term mappings |
| **Intent Classification** | Route to right strategy | Classify query type → apply best retrieval |
| **Metadata Enrichment** | Enables filtering | Extract attributes for faceted search |

### Medium Priority (Consider Later)

| Strategy | Why Deferred | Notes |
|----------|--------------|-------|
| **PRF (Pseudo-Relevance Feedback)** | Unclear immediate value | Could help but adds complexity |
| **Hierarchical Navigation** | Similar to existing LOD | Reimplement LOD with this structure |
| **Learning Synonyms from Behavior** | Background task | Requires usage data, continuous process |

### Low Priority (Expensive/Brute Force)

| Strategy | Why Deferred | Notes |
|----------|--------------|-------|
| **Document Expansion (Doc2Query)** | Brute force, indexing cost | +46% MRR but heavy |
| **Contextual Retrieval** | LLM call per chunk | Effective but expensive |
| **ColBERT** | Infrastructure cost | 3-5x index size, complex |

---

## Key Insight: The Common Thread

**User's Intuition**: "Most of them try to build a map of words/synonyms and then make input somewhat standardized"

This is exactly right. Let's analyze:

### The Underlying Pattern

All effective vocabulary mismatch solutions fundamentally do ONE thing:

```
USER VOCABULARY  ──────────────────►  DOCUMENT VOCABULARY
   (natural)         [MAPPING]           (technical)
```

The difference is HOW and WHEN this mapping happens:

| Strategy | When | How | Mapping Type |
|----------|------|-----|--------------|
| **Synonym Dictionary** | Query time | Explicit rules | User term → Doc terms (1:N) |
| **LLM Query Expansion** | Query time | LLM generation | User query → Multiple variants |
| **Intent Classification** | Query time | Classification | Query → Intent → Strategy |
| **Doc2Query** | Index time | Model generation | Doc → Possible user queries |
| **Contextual Retrieval** | Index time | LLM enrichment | Doc → User-friendly context |
| **SPLADE** | Both | Learned weights | Term → Related terms (automatic) |
| **Fine-tuned Embeddings** | Encoding | Learned space | Both vocabularies → Shared space |

### The Standardization Insight

Your intuition about "standardization" reveals the core problem:

```
PROBLEM:
  User says: "why won't my pods scale automatically"
  Doc says:  "HorizontalPodAutoscaler configuration"
  
  These are the SAME CONCEPT but DIFFERENT VOCABULARY

SOLUTION APPROACHES:

1. QUERY NORMALIZATION (standardize input)
   "why won't my pods scale automatically"
   → "HorizontalPodAutoscaler" OR "HPA" OR "pod autoscaling"
   
2. DOCUMENT NORMALIZATION (standardize index)
   "HorizontalPodAutoscaler configuration"
   → Add: "auto scaling", "scale automatically", "dynamic pods"
   
3. SHARED REPRESENTATION (standardize both)
   Both map to same point in embedding space
   (This is what fine-tuned embeddings do)
```

### Why Synonym Dictionary + LLM Expansion Work Together

```
SYNONYM DICTIONARY (Precision - Known Mappings):
┌─────────────────────────────────────────────┐
│ "scale pods"     → [HPA, autoscaler, HorizontalPodAutoscaler]
│ "auto scaling"   → [HPA, autoscaler, HorizontalPodAutoscaler]  
│ "dynamic replicas" → [HPA, autoscaler, ReplicaSet]
└─────────────────────────────────────────────┘
  ✓ Fast, predictable
  ✓ High precision for known terms
  ✗ Limited coverage (only what's in dictionary)

LLM QUERY EXPANSION (Recall - Unknown Mappings):
┌─────────────────────────────────────────────┐
│ "why won't my pods scale automatically"
│   → "horizontal pod autoscaler not working"
│   → "HPA troubleshooting"  
│   → "kubernetes autoscaling issues"
│   → "pod replica count stuck"
└─────────────────────────────────────────────┘
  ✓ Handles novel phrasings
  ✓ Generates contextual expansions
  ✗ Slower (LLM call)
  ✗ Less predictable

COMBINED (Best of Both):
┌─────────────────────────────────────────────┐
│ 1. Check synonym dictionary first (fast, precise)
│ 2. If low confidence → LLM expansion (slower, broader)
│ 3. Merge results
└─────────────────────────────────────────────┘
```

### Intent Classification's Role

Intent classification doesn't directly solve vocabulary mismatch, but it **optimizes which solution to apply**:

```
INTENT: "troubleshooting"
  → Weight: error messages, symptoms, solutions
  → Expand with: common error patterns
  
INTENT: "how-to"  
  → Weight: tutorials, step-by-step guides
  → Expand with: action verbs, configuration terms
  
INTENT: "reference"
  → Weight: API docs, configuration options
  → Expand with: parameter names, exact terms
  
INTENT: "conceptual"
  → Weight: overview docs, explanations
  → Expand with: broader concepts, analogies
```

---

## Unified Mental Model

### The Vocabulary Bridge

Think of it as building a **bridge** between two languages:

```
┌─────────────────┐                    ┌─────────────────┐
│  USER LANGUAGE  │                    │  DOC LANGUAGE   │
│                 │                    │                 │
│ "scale pods"    │                    │ "HPA"           │
│ "auto scaling"  │    ═══════════     │ "autoscaler"    │
│ "more replicas" │     BRIDGE         │ "replica count" │
│ "handle traffic"│                    │ "scaling policy"│
└─────────────────┘                    └─────────────────┘

BRIDGE COMPONENTS:
├── Synonym Dictionary (explicit mappings)
├── LLM Expansion (dynamic mappings)  
├── Intent Router (context-aware selection)
└── Shared Embeddings (learned mappings)
```

### Why This Matters for Implementation

If the core problem is "vocabulary mapping", then:

1. **Synonym Dictionary** = Manual, high-quality mappings for known terms
2. **LLM Expansion** = Automatic, flexible mappings for unknown terms
3. **Intent Classification** = Context to choose right mapping strategy
4. **Metadata/Hierarchy** = Structured navigation when search fails

These aren't separate solutions - they're **layers of the same solution**:

```
Query: "why won't my pods scale"
         │
         ▼
┌─────────────────────────────────────┐
│ Layer 1: INTENT CLASSIFICATION      │
│ → Detected: "troubleshooting"       │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Layer 2: SYNONYM DICTIONARY         │
│ → "scale" → [HPA, autoscaler, ...]  │
│ → "pods" → [pod, workload, ...]     │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Layer 3: LLM EXPANSION              │
│ → "HPA not scaling"                 │
│ → "autoscaler stuck"                │
│ → "replica count not increasing"    │
└─────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────┐
│ Layer 4: RETRIEVAL (Hybrid)         │
│ → BM25 with expanded terms          │
│ → Dense with original + expanded    │
│ → RRF fusion                        │
└─────────────────────────────────────┘
```

---

## Research Questions to Explore

Based on this unified model, key questions:

1. **Synonym Dictionary Construction**
   - How to bootstrap? (LLM-generated + human validation?)
   - What's the minimum viable dictionary size?
   - How to handle multi-word expressions?

2. **LLM Expansion Optimization**
   - Which model? (Haiku for speed vs Sonnet for quality)
   - How many expansions? (3? 5? 10?)
   - How to prevent noise/drift?

3. **Intent Classification**
   - How many intent categories?
   - Rule-based vs ML-based?
   - How does intent affect retrieval weights?

4. **Integration with Existing Pipeline**
   - Where does expansion happen in current code?
   - How to A/B test different strategies?
   - Latency budget for each layer?

---

## NEW RESEARCH: External Vocabulary Structures (2026-01-27)

### User Requirement (Critical Constraint)

**"Never update input documentation"** - Documents are mutable (users update them), so we need:
- External knowledge structure that lives SEPARATELY from documents
- A "mindmap" or graph that keeps relations between words
- Something like what humans do when building mental vocabulary maps

### Research Categories Explored

1. Knowledge Graphs for IR
2. Synonym Databases & Thesauri (Elasticsearch, Solr)
3. Graph Databases for NLP (Neo4j, NetworkX)
4. Automatic Thesaurus Construction
5. Query Expansion Techniques

---

## Approach 1: Graph-Based Term Relationships

### Core Idea
Build a **term relationship graph** that sits outside documents and maps user vocabulary → document vocabulary.

```
┌─────────────────────────────────────────────────────────────┐
│                    TERM RELATIONSHIP GRAPH                  │
│                                                             │
│  "scale pods" ──SYNONYM(0.9)──► "HPA"                      │
│       │                           │                         │
│       │                      ACRONYM_OF                     │
│       │                           │                         │
│       └──RELATED(0.7)──► "HorizontalPodAutoscaler"         │
│                                   │                         │
│                              BROADER                        │
│                                   │                         │
│                              "autoscaler"                   │
│                                   │                         │
│                              BROADER                        │
│                                   │                         │
│                              "controller"                   │
└─────────────────────────────────────────────────────────────┘

At query time:
  User: "scale pods" 
  → Graph lookup → [HPA, HorizontalPodAutoscaler, autoscaler]
  → Expanded query for retrieval
```

### Relationship Types (from research)

| Relation | Description | Weight Range | Example |
|----------|-------------|--------------|---------|
| `SYNONYM` | Same meaning | 0.85-1.0 | ML ↔ machine learning |
| `ACRONYM_OF` | Abbreviation | 0.95-1.0 | HPA → HorizontalPodAutoscaler |
| `BROADER` | Hypernym | 0.6-0.8 | HPA → autoscaler → controller |
| `NARROWER` | Hyponym | 0.6-0.8 | controller → autoscaler → HPA |
| `RELATED` | Associated | 0.3-0.7 | HPA ↔ metrics-server |
| `SYMPTOM_OF` | Problem→Cause | 0.5-0.8 | "pods not scaling" → HPA misconfiguration |

### Implementation Options

#### Option A: NetworkX (Simple, Python-native)
```python
import networkx as nx

G = nx.DiGraph()
G.add_node("scale pods", type="user_term")
G.add_node("HPA", type="doc_term")
G.add_edge("scale pods", "HPA", relation="SYNONYM", weight=0.9)

def expand_term(G, term, max_hops=2, min_weight=0.5):
    """BFS expansion with weight threshold"""
    ...
```
**Pros**: Easy to prototype, pure Python, no external deps
**Cons**: In-memory only, no persistence, limited scale

#### Option B: Neo4j (Production-grade)
```cypher
// Schema
CREATE (t:Term {name: "scale pods", type: "user"})
CREATE (d:Term {name: "HPA", type: "doc"})
CREATE (t)-[:SYNONYM {weight: 0.9}]->(d)

// Query expansion
MATCH (t:Term {name: $query})-[r*1..2]-(related:Term)
WHERE ALL(rel IN r WHERE rel.weight > 0.5)
RETURN related.name, reduce(w=1.0, rel IN r | w*rel.weight) AS score
ORDER BY score DESC LIMIT 10
```
**Pros**: Scalable, persistent, rich queries, graph algorithms
**Cons**: External service, operational overhead

#### Option C: SQLite + JSON (Lightweight persistent)
```sql
CREATE TABLE terms (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE,
    type TEXT,  -- 'user' or 'doc'
    metadata JSON
);

CREATE TABLE relations (
    source_id INTEGER,
    target_id INTEGER,
    relation_type TEXT,
    weight REAL,
    FOREIGN KEY (source_id) REFERENCES terms(id),
    FOREIGN KEY (target_id) REFERENCES terms(id)
);
```
**Pros**: Persistent, portable, no external deps
**Cons**: Manual graph traversal, no built-in algorithms

---

## Approach 2: Elasticsearch/OpenSearch Synonym Graph

### How It Works
- Define synonyms in a file or via API
- Applied at **query time** (search analyzer)
- Supports multi-word synonyms via `synonym_graph` token filter

### Formats Supported

**Solr Format** (simple):
```
# Equivalent synonyms
HPA, horizontal pod autoscaler, pod autoscaler

# Explicit mapping
scale pods => HPA, horizontal pod autoscaler
```

**WordNet Format** (structured):
```
s(100000001,1,'HPA',n,1,0).
s(100000001,2,'horizontal pod autoscaler',n,1,0).
s(100000001,3,'autoscaler',n,1,0).
```

### Limitations
- **Flat structure**: No hierarchy, no weighted relationships
- **No inference**: Can't discover transitive synonyms
- **Maintenance**: Manual updates required

---

## Approach 3: Semantic Knowledge Bases

### WordNet
- **Structure**: Synsets (synonym sets) organized by meaning
- **Relations**: Hypernyms, hyponyms, meronyms, antonyms
- **Coverage**: General English, weak on technical domains
- **Python**: `nltk.corpus.wordnet`

### ConceptNet
- **Structure**: Commonsense knowledge graph
- **Relations**: 34 relation types (IsA, PartOf, UsedFor, etc.)
- **Coverage**: Broader but still general-purpose
- **API**: REST API or local dump

### SKOS (Simple Knowledge Organization System)
- **Standard**: W3C vocabulary for thesauri/taxonomies
- **Structure**: Concepts with prefLabel, altLabel, broader, narrower, related
- **Format**: RDF/SPARQL
- **Python**: `skosprovider`, `rdflib`

### Limitation: Domain Gap
All these are general-purpose. For Kubernetes:
- "HPA" not in WordNet
- "pod autoscaling" not in ConceptNet
- We need **domain-specific** vocabulary

---

## Approach 4: Automatic Thesaurus Construction

### Methods to Build Domain Vocabulary

#### 1. Distributional Similarity (Word2Vec, FastText)
- Train embeddings on K8s docs
- Similar words = similar vectors
- Find synonyms via cosine similarity

```python
from gensim.models import Word2Vec

# Train on K8s corpus
model = Word2Vec(sentences, vector_size=100, window=5)

# Find similar terms
model.wv.most_similar("autoscaling", topn=10)
# → [('HPA', 0.87), ('horizontal-pod-autoscaler', 0.82), ...]
```

**Quality**: ~60-70% precision without human review

#### 2. Co-occurrence Analysis
- Terms appearing together frequently are related
- Build graph from co-occurrence matrix

```python
# Count co-occurrences in sliding window
for doc in corpus:
    for i, term in enumerate(doc):
        for j in range(i-window, i+window):
            if i != j:
                cooccurrence[(term, doc[j])] += 1

# Filter by frequency threshold
related_pairs = [(a, b) for (a,b), count in cooccurrence.items() 
                 if count > threshold]
```

#### 3. LLM-Based Extraction
- Ask LLM to identify synonyms/related terms from docs
- Higher quality but expensive

```python
prompt = """
Given this Kubernetes documentation excerpt:
{doc_chunk}

Extract:
1. Technical terms mentioned
2. For each term, list synonyms or alternative names users might search for
3. Hierarchical relationships (broader/narrower concepts)

Output as JSON.
"""
```

**Quality**: ~80-90% with good prompting

#### 4. Hybrid: Automatic Bootstrap + Human Curation
1. **Bootstrap**: Use Word2Vec/LLM to generate candidates
2. **Validate**: Human review removes false positives
3. **Iterate**: Add new terms from search logs

---

## Approach 5: Query-Time Expansion Without Graph

### Embedding-Based Expansion (k-NN)
- Embed user query
- Find k nearest terms in vocabulary embedding space
- Add to query

```python
query_embedding = embed("scale pods")
similar_terms = ann_index.search(query_embedding, k=5)
# → ["HPA", "autoscaler", "replica scaling", ...]
expanded_query = original_query + " " + " ".join(similar_terms)
```

**Advantage**: No explicit graph needed
**Disadvantage**: May add noise, less interpretable

### Pseudo-Relevance Feedback (PRF)
- Run initial retrieval
- Extract terms from top-k results
- Add to query, re-retrieve

```python
initial_results = retrieve(query, k=10)
expansion_terms = extract_key_terms(initial_results)
expanded_query = query + " " + " ".join(expansion_terms)
final_results = retrieve(expanded_query, k=5)
```

**Advantage**: Adapts to corpus vocabulary automatically
**Disadvantage**: Error propagation if initial results bad

---

## Synthesis: What Makes Sense for Our Use Case?

### Requirements Recap
1. **External structure** - Don't modify source docs
2. **Domain-specific** - K8s terminology
3. **Maintainable** - Easy to update as docs change
4. **Query-time application** - Fast lookup during retrieval

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  VOCABULARY KNOWLEDGE LAYER                 │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  TERM GRAPH     │    │  TERM EMBEDDINGS │                │
│  │  (NetworkX)     │    │  (Word2Vec/SBERT)│                │
│  │                 │    │                  │                │
│  │  Explicit       │    │  Implicit        │                │
│  │  relationships  │    │  similarity      │                │
│  └────────┬────────┘    └────────┬─────────┘                │
│           │                      │                          │
│           └──────────┬───────────┘                          │
│                      │                                      │
│              QUERY EXPANSION                                │
│                      │                                      │
└──────────────────────┼──────────────────────────────────────┘
                       │
                       ▼
              ┌─────────────────┐
              │   RETRIEVAL     │
              │   (BM25+Dense)  │
              └─────────────────┘
```

### Phased Approach

**Phase 1: Simple Graph (NetworkX + JSON)**
- Build initial K8s term graph manually (~100-200 key terms)
- Store as JSON, load into NetworkX
- Integrate with existing query expansion

**Phase 2: Automatic Expansion**
- Train Word2Vec on K8s corpus
- Use to suggest new terms/relationships
- Human review before adding to graph

**Phase 3: Embedding Hybrid**
- Add term embeddings alongside graph
- Use both for expansion (graph for known terms, embeddings for novel)

---

## Key Decisions Needed

1. **Graph storage**: NetworkX+JSON vs SQLite vs Neo4j?
2. **Bootstrap method**: Manual curation vs LLM extraction vs Word2Vec?
3. **Relationship granularity**: Simple (synonym/related) vs rich (SKOS-like)?
4. **Update process**: Manual only vs semi-automatic with review?

---

## RESEARCH SYNTHESIS (Completed 2026-01-27)

### Summary of All Approaches Researched

| Approach | Structure | Quality | Speed | Cost | Best For |
|----------|-----------|---------|-------|------|----------|
| **WordNet/ConceptNet** | Pre-built graph | General 65-75% | Fast | Free | General English, fallback |
| **SKOS Thesaurus** | RDF graph | High (curated) | Fast | Free | Hierarchical vocabularies |
| **Elasticsearch synonyms** | Flat file + graph filter | Depends on curation | Very fast | Free | Simple equivalences |
| **Distributional (Word2Vec)** | Embedding space | 65-75% | Fast train, fast query | Low | Large corpus, domain-specific |
| **Embedding clustering** | Vector clusters | 70-85% | Medium | Low-Med | Synonym discovery |
| **LLM extraction** | Structured JSON/graph | 85-95% | Slow | High | High-quality, domain-specific |
| **Co-occurrence** | Adjacency matrix → graph | 50-60% synonyms, 75-85% related | Very fast | Very low | Related terms, keyword extraction |
| **Knowledge Graph (GraphRAG)** | Entity-relationship graph | 83-87% complex queries | Slower | High setup | Multi-hop reasoning |
| **Pseudo-relevance feedback** | No pre-built structure | Variable | Fast | Low | Adaptive, corpus-driven |
| **Hybrid (auto + human)** | Any | 90-95% | Medium | Medium | Production systems |

### Critical Findings

#### 1. Query Expansion Can HURT Performance
- **Strong retrievers don't need it**: Expansion harms cross-encoders and strong neural rankers
- **Precision vs Recall trade-off**: Expansion increases recall but often decreases precision
- **Selective expansion**: Use model comparison to predict when expansion will fail
- **UMLS study**: ALL types of thesaurus expansion **degraded aggregate performance** (though 40% of individual queries improved)

#### 2. Domain-Specific is Essential
- General-purpose thesauri (WordNet) introduce noise in specialized domains
- Pre-trained embeddings may not capture domain-specific semantics
- **Recommendation**: Build K8s-specific vocabulary, don't rely on general resources

#### 3. Multi-Word Expressions Need Special Handling
- Standard tokenizers split "horizontal pod autoscaler" before synonym filter sees it
- **Solution**: Use graph token filters (ES/Solr) that maintain `positionLength`
- **Alternative**: AutoPhrasing to convert phrases to single tokens first

#### 4. Context-Dependent Synonyms are Unsolved
- "apple" means different things in tech vs food contexts
- **Current workarounds**: Domain-specific synonym sets, field-specific analyzers, query classification
- **Research direction**: Word embeddings + sense clustering (not production-ready)

#### 5. Recommended Parameter Ranges

| Technique | # Expansion Terms | Original Weight | Expansion Weight | Threshold |
|-----------|-------------------|-----------------|------------------|-----------|
| Thesaurus | 2-5 synonyms | 1.0 | 0.8-0.9 | N/A |
| Ontology | 3-10 concepts | 1.0 | 0.3-0.7 (by level) | 1-2 hops |
| Embeddings | 3-10 similar | 1.0 (or 5x boost) | 0.2-1.0 | cosine > 0.5-0.7 |
| PRF (Rocchio) | 10-50 terms | α=1.0 | β=0.6 | Top-10 docs |
| Knowledge Graph | 5-20 entities | 1.0 | 0.2-0.8 (by hop) | 2-3 hops |

### Recommended Architecture for K8s RAG

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VOCABULARY KNOWLEDGE LAYER                       │
│                    (External, Independent of Docs)                  │
│                                                                     │
│  ┌──────────────────────┐     ┌──────────────────────┐            │
│  │  TERM RELATIONSHIP   │     │  TERM EMBEDDINGS     │            │
│  │  GRAPH (NetworkX)    │     │  (Sentence-BERT)     │            │
│  │                      │     │                      │            │
│  │  • Explicit synonyms │     │  • Implicit similarity│            │
│  │  • Acronym mappings  │     │  • Novel term handling│            │
│  │  • Hierarchies       │     │  • Fallback for unknowns│          │
│  │  • Symptom→Cause     │     │                      │            │
│  └──────────┬───────────┘     └──────────┬───────────┘            │
│             │                            │                         │
│             └────────────┬───────────────┘                         │
│                          │                                         │
│                  QUERY EXPANSION LAYER                             │
│                          │                                         │
│         ┌────────────────┼────────────────┐                       │
│         │                │                │                        │
│         ▼                ▼                ▼                        │
│   Graph lookup     Embedding k-NN    LLM rewrite                  │
│   (known terms)    (novel terms)     (complex queries)            │
│         │                │                │                        │
│         └────────────────┼────────────────┘                       │
│                          │                                         │
│                  MERGED EXPANSION                                  │
│                          │                                         │
└──────────────────────────┼──────────────────────────────────────────┘
                           │
                           ▼
                   ┌───────────────┐
                   │   RETRIEVAL   │
                   │  (BM25+Dense) │
                   └───────────────┘
```

### Implementation Phases

**Phase 1: Bootstrap Term Graph (Week 1-2)**
- Extract key K8s terms from docs using NER/YAKE
- Use Sentence-BERT to cluster similar terms
- Generate synonym candidates automatically
- LLM validation of top candidates (GPT-4 for quality)
- Store as JSON, load into NetworkX

**Phase 2: Integrate with Retrieval (Week 3)**
- Add graph lookup to query expansion pipeline
- Implement weighted expansion (original terms boosted)
- A/B test vs current LLM-only rewriting
- Measure Hit@5 improvement

**Phase 3: Continuous Learning (Ongoing)**
- Monitor query logs for failed searches
- Extract reformulation patterns (user's own synonyms)
- Human review of suggested additions
- Monthly graph updates

### Storage Format Recommendation

**Primary: JSON + NetworkX (Simple, Portable)**
```json
{
  "terms": {
    "HPA": {
      "type": "doc_term",
      "pos": "NOUN",
      "aliases": ["horizontal pod autoscaler", "Horizontal Pod Autoscaler"],
      "embedding_id": "emb_001"
    },
    "scale pods": {
      "type": "user_term", 
      "pos": "VERB_PHRASE"
    }
  },
  "relations": [
    {"source": "scale pods", "target": "HPA", "type": "SYNONYM", "weight": 0.9},
    {"source": "HPA", "target": "autoscaler", "type": "BROADER", "weight": 0.7},
    {"source": "pods not scaling", "target": "HPA", "type": "SYMPTOM_OF", "weight": 0.8}
  ]
}
```

**Why this format:**
- Human-readable and editable
- Version-controllable (git diff friendly)
- Easy to load into NetworkX for graph operations
- Can export to Neo4j later if needed
- No external dependencies for basic usage

### Quality Expectations

| Phase | Expected Hit@5 | Method |
|-------|---------------|--------|
| Current (LLM rewrite only) | 40.75% | Baseline |
| + Term graph (synonyms) | 50-55% | Known term expansion |
| + Embedding fallback | 55-60% | Novel term handling |
| + Symptom→Cause relations | 60-65% | Troubleshooting queries |
| + Continuous learning | 65-70% | User-driven refinement |

### Key Risks

1. **Over-expansion noise**: Too many terms dilute relevance
   - Mitigation: Conservative weights (0.8-0.9 for synonyms), limit to 5 terms

2. **Domain drift**: Generic terms creep in
   - Mitigation: K8s-specific validation, LLM review

3. **Maintenance burden**: Graph becomes stale
   - Mitigation: Automatic candidate generation, monthly human review

4. **Latency impact**: Graph lookup adds time
   - Mitigation: In-memory graph (NetworkX), cache frequent lookups

---

## Next Steps (Updated)

1. **Prototype term graph** - Start with ~100 high-value K8s terms
2. **Benchmark expansion impact** - Measure Hit@5 with/without graph
3. **Integrate with existing pipeline** - Add graph lookup before LLM rewrite
4. **Design continuous learning** - Query log analysis for new synonyms
