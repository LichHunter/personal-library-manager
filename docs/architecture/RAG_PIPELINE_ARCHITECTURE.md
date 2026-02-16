# RAG Pipeline Architecture

> **Status**: DESIGN DOCUMENT - Under refinement
> **Initial Domain**: Kubernetes Documentation (expandable to any domain)
> **Current Performance**: 36% Hit@5
> **Target Performance**: 55%+ Hit@5
> **Validated**: 40% → 80% Hit@5 with manual term dictionary

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [Architectural Overview](#2-architectural-overview)
3. [Ingestion Pipeline](#3-ingestion-pipeline)
4. [Term Management System](#4-term-management-system)
5. [Retrieval Pipeline](#5-retrieval-pipeline)
6. [Data Model](#6-data-model)
7. [PoC Investigation Areas](#7-poc-investigation-areas)
8. [Design Decisions & Tradeoffs](#8-design-decisions--tradeoffs)
9. [Open Questions](#9-open-questions)

---

## 1. Problem Statement

### 1.1 The Vocabulary Mismatch Problem

Users search using natural language, but technical documentation uses domain-specific terminology. This creates a fundamental retrieval gap where semantically relevant documents are not lexically matched.

**Example Mismatches**:

| What User Types | What Documentation Contains | Why Match Fails |
|-----------------|----------------------------|-----------------|
| "pod keeps restarting" | "CrashLoopBackOff" | No lexical overlap |
| "out of memory" | "OOMKilled" | Acronym vs. description |
| "can't pull image" | "ImagePullBackOff" | Colloquial vs. technical |
| "container won't start" | "ErrImagePull, CreateContainerError" | Multiple possible causes |

### 1.2 Why Current Approaches Fail

**BM25 (Lexical Search)**: Requires exact or stemmed word matches. "restarting" doesn't match "CrashLoopBackOff".

**Semantic Search (Embeddings)**: Better at conceptual similarity, but general-purpose embedding models weren't trained on domain-specific terminology. The embedding space doesn't place "pod keeps restarting" near "CrashLoopBackOff".

**Current Hybrid Approach**: Combines both but doesn't solve the fundamental vocabulary gap—it just averages two incomplete solutions.

### 1.3 Solution Hypothesis

**Validated**: Manual testing showed 40% → 80% Hit@5 improvement when using a domain terminology dictionary.

Bridge the vocabulary gap at TWO points:

1. **Ingestion Time**: Enrich documents with alternative phrasings (technical terms → user language)
2. **Query Time**: Enrich queries with domain terminology (user language → technical terms)

Additionally, use **document/section summaries** to narrow search scope when a query could match many documents but the user has a specific intent (e.g., "debugging" vs. "understanding concepts").

### 1.4 Multi-Domain Design

This system is designed to handle **thousands of documents across multiple domains**, not just Kubernetes. The architecture must:

- Support any domain without domain-specific hardcoding
- Learn new terminology automatically as documents are ingested
- Improve extraction quality over time through feedback loops

---

## 2. Architectural Overview

### 2.1 System Boundaries

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              RAG SYSTEM                                      │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     INGESTION SUBSYSTEM                              │   │
│  │                                                                      │   │
│  │  Documents → Chunking → Summarization → Term Extraction → Indexing  │   │
│  │                                              │                       │   │
│  │                                              ▼                       │   │
│  │                                    ┌─────────────────┐               │   │
│  │                                    │  TERM GRAPH     │               │   │
│  │                                    │  (shared state) │               │   │
│  │                                    └─────────────────┘               │   │
│  │                                              ▲                       │   │
│  │                                              │                       │   │
│  └──────────────────────────────────────────────┼───────────────────────┘   │
│                                                 │                           │
│  ┌──────────────────────────────────────────────┼───────────────────────┐   │
│  │                     RETRIEVAL SUBSYSTEM      │                       │   │
│  │                                              │                       │   │
│  │  Query → Query Processing → Search → Ranking │→ Results              │   │
│  │                   │                          │                       │   │
│  │                   └──────────────────────────┘                       │   │
│  │                        (term lookup)                                 │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                  TERM MANAGEMENT SUBSYSTEM                           │   │
│  │                                                                      │   │
│  │  Low-Confidence Queue → LLM Review → Manual Review → Graph Update   │   │
│  │                              │                              │        │   │
│  │                              └──────► Fast System Training ◄┘        │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Core Principles

**Principle 1: Enrich at Both Ends**
- Documents are enriched with user-facing language at ingestion
- Queries are enriched with technical terminology at retrieval
- The term graph serves as the bridge between both vocabularies

**Principle 2: Fast Path with Quality Escape Hatch**
- Most documents processed quickly using statistical extraction (YAKE) and trainable NER (GLiNER)
- Low-confidence extractions queued for deeper processing
- LLM review tier before human review for scalability

**Principle 3: Self-Improving Systems**
- Fast and slow systems are designed to improve over time
- Corrections from slow system become training data for fast system
- NER model is periodically retrained on accumulated feedback
- Every mistake is an opportunity to improve

**Principle 4: Summaries for Scope Narrowing**
- When a query matches many documents, summaries help identify the RIGHT documents
- Summaries capture document/section INTENT (debugging vs. conceptual vs. reference)
- Two-stage retrieval: narrow by summary, then search within scope

**Principle 5: LLM with Guardrails**
- LLM used for query processing, slow-path extraction, and review queue triage
- Never trusted blindly—always validated against term graph or human review
- Guardrails prevent hallucination and intent drift
- Fallback to local models or original query when LLM unavailable

---

## 3. Ingestion Pipeline

### 3.1 Document Chunking

**Purpose**: Split documents into retrievable units that are semantically coherent and appropriately sized for embedding.

**Strategy**: Heading-aware paragraph chunking

**How It Works**:
- Documents are split along structural boundaries (headings, sections)
- Within sections, paragraphs form the base chunking unit
- Small consecutive paragraphs (under minimum threshold) are merged
- Large paragraphs (over maximum threshold) are split at sentence boundaries
- Each chunk retains its section context (heading path)

**Chunk Properties**:
- **Size Range**: 50-256 tokens (validated via benchmark)
- **Context Preservation**: Each chunk knows its heading path (e.g., "Troubleshooting > Pods > Restart Issues")
- **Boundary Respect**: Never splits mid-sentence or mid-paragraph when avoidable

**Why This Strategy**:
- Heading-based splitting respects document structure
- Paragraph boundaries are natural semantic units
- Size constraints ensure chunks are neither too fragmented nor too large for embedding
- Benchmark showed 96.4% Recall@5 with this approach

### 3.2 Summarization

**Purpose**: Generate condensed representations of documents and sections that capture their INTENT and SCOPE, enabling search narrowing.

#### 3.2.1 Document Summaries

**What They Capture**:
- Overall document purpose (tutorial, reference, troubleshooting guide, etc.)
- Main topics covered
- Target audience and prerequisites
- Key concepts introduced

**How They're Used**:
- First-stage filtering when query is broad
- Identifying which documents are relevant before expensive chunk search
- Disambiguation when same topic appears in multiple documents with different purposes

#### 3.2.2 Section Summaries

**What They Capture**:
- Section's specific focus within the document
- What questions this section answers
- Relationship to other sections

**How They're Used**:
- Narrowing search within a document when multiple sections might match
- Understanding section intent without reading all chunks
- Linking related information across sections

#### 3.2.3 Summary Storage Architecture

**Design Decision**: Summaries are stored SEPARATELY from chunks, not concatenated or embedded together.

**Rationale**:
- Summaries serve a different retrieval purpose (scope narrowing vs. content matching)
- Concatenating would dilute chunk embeddings with summary language
- Separate storage enables two-stage search pattern
- Summaries can be updated without re-embedding all chunks

**Relationships**:
- Document summary links to document ID
- Section summary links to document ID + section identifier
- Section summary references the chunk IDs it covers

### 3.3 Term Extraction

**Purpose**: Identify domain-specific terminology in each chunk for vocabulary bridging and confidence scoring.

#### 3.3.1 Fast Extraction System (Ingest Time)

**Design Goal**: Quick, general-purpose extraction that improves over time through training.

**Components**:

1. **YAKE Keyword Extractor**
   - Statistical extraction based on term frequency, position, co-occurrence
   - Fast (~100-200ms per document)
   - Language-agnostic, no training required
   - Outputs ranked list of significant phrases

2. **GLiNER Entity Recognizer** (replaces spaCy)
   - Zero-shot NER that works with ANY entity type without domain-specific training
   - Define entity categories at runtime (e.g., "error_state", "resource_type", "command")
   - General-purpose: works across all domains, not hardcoded to specific terminology
   - **Trainable**: Can be incrementally fine-tuned as corrections accumulate
   - Fast enough for batch processing (~100-300ms per document)

3. **Term Graph Matcher**
   - Looks up extracted terms against known vocabulary
   - Case-insensitive, handles minor variations
   - Returns matched terms with their synonyms

4. **Confidence Calculator**
   - Combines signals from extraction and matching
   - Outputs confidence score and review flag

**Why GLiNER Instead of spaCy**:
- spaCy's general NER (en_core_web_sm) recognizes PERSON, ORG, GPE but NOT domain-specific terms
- GLiNER is zero-shot: works immediately on any domain without training
- GLiNER is trainable: slow system corrections become training data
- GLiNER supports dynamic entity types: add new categories without retraining

#### 3.3.2 Confidence Scoring

The extraction confidence indicates how well we understood the chunk's terminology.

**Signals**:

| Signal | What It Measures | Low Confidence Indicator |
|--------|------------------|-------------------------|
| **Known Term Ratio** | % of extracted terms found in term graph | < 30% of terms known |
| **Coverage** | % of chunk content captured by extracted terms | < 10% coverage |
| **Entity Density** | Entities per 100 tokens | Abnormally low or high |
| **Section Type Mismatch** | Expected vs. actual extraction patterns | Code block with no code entities |

**Confidence Threshold**: Determined by POC-2. Chunks scoring below threshold are queued for slow processing.

**What Fast Extraction Does NOT Do**:
- Does NOT use LLM (too slow for batch ingestion)
- Does NOT discover new domain terms (only matches against known vocabulary)
- Does NOT generate synonyms (only looks up existing mappings)

#### 3.3.3 Output of Term Extraction

For each chunk, term extraction produces:
- **Extracted Terms**: Raw terms found by YAKE + GLiNER
- **Matched Terms**: Terms that exist in the term graph
- **Enrichment Terms**: Synonyms/aliases from the term graph for matched terms
- **Confidence Score**: Numeric score (0-1) indicating extraction quality
- **Needs Review Flag**: Boolean indicating if chunk should go to slow processing

### 3.4 Embedding and Indexing

**Purpose**: Create searchable representations of chunks for hybrid retrieval.

#### 3.4.1 Embedding Strategy

**Input Composition**: Chunk text + enrichment terms from term graph

The embedding input is constructed as:
- Primary content: Original chunk text
- Enrichment suffix: Terms and synonyms from term graph, formatted as metadata

**Why Enrich Before Embedding**:
- Embedding model learns that technical terms and user phrasings belong to the same chunk
- Creates implicit vocabulary bridging in the embedding space
- User query about colloquial phrasing will have higher similarity to enriched chunk

**Model**: To be determined by POC-7 (candidates: BAAI/bge-base-en-v1.5, e5-large-v2, instruction-tuned models)

#### 3.4.2 Index Structures

**Vector Index**: For semantic similarity search
- Stores chunk embeddings
- Supports approximate nearest neighbor queries

**Full-Text Index**: For BM25 lexical search
- Indexes chunk text + enrichment terms
- Supports keyword matching and phrase queries

**Summary Index**: For summary-based filtering
- Stores document and section summary embeddings
- Used in first-stage scope narrowing

**Metadata Index**: For filtering and faceting
- Document ID, section path, confidence score, review status
- Enables filtered searches (e.g., "only chunks from troubleshooting sections")

### 3.5 Two-Pass Ingestion (Bootstrap Mode)

**Problem**: On first corpus ingestion, the term graph only has seed terms. Most extracted terms won't match, leading to artificially low confidence scores and overwhelming the slow system.

**Solution**: Two-pass ingestion for initial corpus load.

**Pass 1: Term Discovery**
- Extract all terms from all chunks using YAKE + GLiNER
- No graph lookup, no confidence scoring
- Build frequency table of extracted terms
- Auto-add high-frequency terms to graph (threshold: appears in >N chunks)

**Pass 2: Normal Processing**
- Re-process chunks with enriched term graph
- Known Term Ratio now meaningful
- Queue only genuinely problematic chunks for slow processing

**When to Use**:
- Initial corpus ingestion (hundreds/thousands of documents)
- Adding a new domain with many documents

**Incremental Ingestion**:
- Single-pass processing (term graph already populated)
- Normal confidence scoring and routing

**Note**: Initial ingestion will be slow due to two-pass processing, but this is acceptable as a one-time cost. Later, with a populated term graph, slow processing will be needed less frequently.

---

## 4. Term Management System

### 4.1 Term Graph Structure

**Purpose**: Central knowledge base mapping technical terminology to user-facing language.

#### 4.1.1 Term Entities

Each term in the graph represents a domain concept:

**Core Attributes**:
- **Canonical Form**: The official/preferred term (e.g., "CrashLoopBackOff")
- **Term Type**: Classification (error_state, resource_type, command, concept, configuration)
- **Domain**: Topic area (kubernetes, docker, networking, storage, etc.)
- **Source**: How the term was added (seed, llm_extracted, manual_review)
- **Confidence**: Trust level (1.0 for seed/manual, variable for LLM-extracted)

**Statistical Attributes**:
- **Document Count**: How many documents contain this term
- **Chunk Count**: How many chunks contain this term
- **Query Count**: How often this term appears in user queries (future)

#### 4.1.2 Synonym Relationships

Synonyms connect technical terms to user-facing language:

**Synonym Attributes**:
- **Synonym Text**: The alternative phrasing
- **Direction**: Whether this is user→technical or technical→user or bidirectional
- **Source**: How the synonym was discovered
- **Confidence**: Trust level for this mapping

**Example**:
```
CrashLoopBackOff (canonical)
├── "pod keeps restarting" (user→technical, high confidence)
├── "restart loop" (user→technical, high confidence)
├── "container crash loop" (user→technical, medium confidence)
└── "pod won't stay up" (user→technical, medium confidence)
```

**Design Decision**: Synonyms only, no complex relationships (causes, part_of, etc.). Simpler to maintain, and co-occurrence in documents captures implicit relationships.

### 4.2 Fast System (Ingest-Time Extraction)

**Purpose**: Quickly extract and match terms during document ingestion without LLM overhead.

**Components**:

1. **YAKE Keyword Extractor**
   - Statistical extraction based on term frequency, position, co-occurrence
   - Fast (~100-200ms per document)
   - Language-agnostic, no training required
   - Outputs ranked list of significant phrases

2. **GLiNER Entity Recognizer**
   - Zero-shot NER for any domain
   - Trainable on accumulated feedback
   - Fast (~100-300ms per document)
   - Outputs entities with types and confidence

3. **Term Graph Matcher**
   - Looks up extracted terms against known vocabulary
   - Case-insensitive, handles minor variations
   - Returns matched terms with their synonyms

4. **Confidence Calculator**
   - Combines signals from extraction and matching
   - Outputs confidence score and review flag

**Process Flow**:

```
Chunk Text
    │
    ├─────────────────┬─────────────────┐
    ▼                 ▼                 │
YAKE Keywords    GLiNER Entities       │
    │                 │                 │
    └────────┬────────┘                 │
             ▼                          │
      Merged Term List                  │
             │                          │
             ▼                          │
    Term Graph Lookup ◄─────────────────┘
             │                     (original text for
             │                      coverage calculation)
    ┌────────┴────────┐
    ▼                 ▼
Matched Terms    Unmatched Terms
    │                 │
    ▼                 ▼
Get Synonyms    Count for
from Graph      Confidence
    │                 │
    └────────┬────────┘
             ▼
    Confidence Score
             │
             ▼
    ┌────────┴────────┐
    │                 │
  ≥ threshold     < threshold
    │                 │
    ▼                 ▼
 Index with      Queue for
 Enrichment      Slow System
```

**What Fast System Produces**:
- Enriched chunk (text + synonym terms) for embedding
- Metadata (extracted terms, matched terms, confidence)
- Routing decision (index directly vs. queue for review)

**Expandability**: The fast system improves through:
- GLiNER retraining on slow system corrections
- Term graph expansion from approved terms
- Confidence threshold tuning based on observed quality

### 4.3 Slow System (Batch Processing)

**Purpose**: Deep processing of low-confidence chunks using LLM, with tiered review for quality control.

**Trigger**: Chunks with extraction confidence below threshold (determined by POC-2)

**Why These Chunks Need Special Handling**:
- May contain novel terminology not in the term graph
- May have unusual structure (code-heavy, table-heavy)
- May be poorly written or ambiguous
- Statistical extraction may have failed for some reason

#### 4.3.1 LLM-Based Deep Extraction

**When Invoked**: Periodically (e.g., daily batch) or on-demand

**What LLM Does**:
1. **Term Discovery**: Identify domain-specific terms that YAKE/GLiNER missed
2. **Synonym Generation**: Suggest user-facing phrasings for technical terms
3. **Context Understanding**: Explain why certain terms are significant in this chunk
4. **Confidence Assessment**: Rate its own certainty about extractions

**Guardrails**:
- LLM must cite specific text spans as evidence for each term
- Extracted terms are validated against document context
- Novel terms flagged for review, not auto-added to graph
- Synonyms validated against existing usage patterns

**Fallback**: If LLM unavailable (rate limit, outage), use local model (e.g., Llama 3 8B) or keep in queue for retry

**Output**: Candidate terms and synonyms with evidence, passed to review tier

#### 4.3.2 Tiered Review Process

**Tier 1: LLM Review (Automated)**

Purpose: Filter obvious cases before human review

Process:
- LLM (e.g., Claude Opus/Sonnet) reviews extraction candidates
- High-confidence approvals: Auto-add to term graph
- High-confidence rejections: Auto-reject with logging
- Uncertain cases: Escalate to human review

Benefits:
- Scales to large review queues
- Handles obvious cases automatically
- Reduces human review burden
- Humans focus on genuinely uncertain cases

**Tier 2: Human Review (Manual)**

Input: Uncertain cases from LLM review tier

Reviewer Actions:

| Action | Effect |
|--------|--------|
| **Approve Term** | Add to term graph with source="manual_review" |
| **Approve Synonym** | Link synonym to existing term |
| **Reject** | Mark as invalid, add to rejection log for training |
| **Modify** | Edit term/synonym before approval |

Review Interface Requirements:
- Show chunk context where term was found
- Show LLM's evidence/reasoning
- Show similar existing terms in graph
- Enable bulk actions for efficiency

#### 4.3.3 Feedback Loop to Fast System

**Critical Design**: Slow system corrections improve fast system over time.

**Training Data Collection**:
- Approved terms/synonyms become positive examples
- Rejected extractions become negative examples
- Accumulated examples stored for periodic retraining

**GLiNER Retraining**:
- Trigger: When accumulated examples reach threshold (e.g., 100+ examples)
- Frequency: Monthly or on-demand
- Method: Fine-tune on new examples, optionally freeze encoder for speed
- Validation: Test on held-out set before deploying

**Term Graph Update**:
- Approved terms/synonyms added immediately
- Affected chunks identified for re-enrichment (future enhancement)

### 4.4 Seed Terminology Map

**Purpose**: Bootstrap the term graph with high-quality initial vocabulary for initial domain.

**Scope**: ~100-200 high-impact terms for starting domain, covering:
- Common error states
- Core resources/concepts
- Key commands
- Important terminology

**Quality Requirements**:
- Each term has 3-5 verified user-facing synonyms
- Synonyms sourced from: Stack Overflow questions, GitHub issues, forum posts
- Manual verification of each mapping

**Maintenance**: Seed map is the foundation; slow system + manual review expand it over time.

**Multi-Domain**: Each new domain can have its own seed map, or rely on zero-shot extraction + slow system learning.

---

## 5. Retrieval Pipeline

### 5.1 Query Processing

**Purpose**: Transform user queries into enriched queries that bridge vocabulary gap.

#### 5.1.1 LLM Query Processor

**Design Decision**: LLM processes ALL queries, not just "bad" ones.

**Rationale**:
- Consistent processing path (no heuristic "is this query good enough?" check)
- Can extract useful signals even from well-formed queries
- Can identify when query is already specific and needs no changes
- Simpler architecture than conditional branching

**What LLM Determines**:

1. **Specificity Assessment**:
   - Is the query already using technical terminology?
   - Does it need enrichment or is it search-ready?

2. **Enriched Query**:
   - Original query augmented with relevant technical terms
   - Terms sourced from term graph lookup
   - Must preserve original intent (guardrail)

3. **Domain Terms**:
   - List of terms relevant to this query
   - Used for downstream filtering and highlighting

#### 5.1.2 Guardrails for Query Processing

**Intent Preservation**:
- Enriched query must be semantically equivalent to original
- If LLM is uncertain, return original query unchanged
- Log cases where enrichment significantly changed meaning for review

**Term Relevance**:
- Only add terms that are DIRECTLY relevant to the query
- Prefer terms from seed vocabulary (high confidence)
- Limit number of added terms to prevent topic drift

**Faithfulness**:
- Enriched query must be answerable by documents that would answer original
- No speculative expansion ("maybe they also want to know about X")

**Fallback**:
- If LLM unavailable: Use dictionary-based expansion from term graph
- If LLM fails: Return original query + log for investigation

### 5.2 Summary-Based Search Narrowing

**Purpose**: Use summaries to identify relevant documents/sections BEFORE expensive chunk search.

#### 5.2.1 When to Use Summary Narrowing

**Applied to ALL queries** (LLM determines relevance):
- LLM decides if query would benefit from scope narrowing
- Summary search always available as a tool

#### 5.2.2 Summary Search Process

**Stage 1: Document Summary Search**

Purpose: Identify which documents are relevant to the query

Process:
1. Embed the (enriched) query
2. Search document summary embeddings for similarity
3. Return top-N candidate documents (e.g., top 5-10)
4. Filter out documents whose summaries indicate wrong intent

**Stage 2: Section Summary Search (Optional)**

Purpose: Further narrow within selected documents

Process:
1. For each candidate document, search section summaries
2. Identify sections most relevant to query intent
3. Limit subsequent chunk search to those sections

#### 5.2.3 Filtered Chunk Search

After summary narrowing:
1. Chunk search is restricted to chunks within selected documents/sections
2. Reduces search space, improving relevance and speed
3. Metadata filter applied to limit search scope

### 5.3 Hybrid Search

**Purpose**: Combine lexical (BM25) and semantic (embedding) search for robust retrieval.

#### 5.3.1 Search Components

**BM25 Search**:
- Searches chunk text + enrichment terms
- Strong for exact keyword matches
- Handles rare terms well (high IDF)
- Fast, no embedding computation at query time

**Semantic Search**:
- Searches chunk embeddings
- Strong for conceptual similarity
- Handles paraphrasing and synonyms
- Requires query embedding computation

#### 5.3.2 Result Fusion

**Method**: Reciprocal Rank Fusion (RRF)

**How RRF Works**:
- Each search method produces a ranked list
- RRF score for each result = sum of 1/(k + rank) across all lists
- k is a constant (typically 60) that prevents high-ranked items from dominating
- Results sorted by combined RRF score

**Why RRF**:
- No tuning required (unlike weighted averaging)
- Robust to different score scales
- Naturally balances contributions from both methods

#### 5.3.3 Search Parameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| **Initial candidates** | 50 per method | Balance between coverage and speed |
| **RRF k constant** | 60 | Standard value, prevents rank domination |
| **Post-fusion candidates** | 50 | Input to reranker |

### 5.4 Reranking

**Purpose**: Reorder candidates using a more sophisticated model that considers query-document interaction.

**Method**: Cross-encoder model

**How Cross-Encoders Work**:
- Takes (query, document) pair as input
- Computes interaction between query and document tokens
- Outputs relevance score
- More accurate than bi-encoder (embedding) similarity but slower

**Model**: bge-reranker-base (or similar)

**Process**:
1. Take top 50 candidates from RRF fusion
2. Score each (query, chunk) pair with cross-encoder
3. Return top K results (typically 5-10)

**Why Reranking**:
- Catches cases where embedding similarity missed relevance
- Handles nuanced relevance distinctions
- Significantly improves precision at top ranks

**Performance Note**: Start with quality (slower), optimize for speed later if needed.

### 5.5 Topic Disambiguation

**Problem**: Some queries could relate to multiple domains (e.g., "container restart" could be Docker or Kubernetes).

**Approach**: Topic-first classification with single enriched query

#### 5.5.1 Topic Classification

**Signals for Classification**:
- Terms in query that are domain-specific
- Term graph matches per domain
- Query structure patterns typical of each domain

**Classification Output**:
- Primary domain (e.g., "kubernetes")
- Confidence level (high/medium/low)
- Secondary domains if ambiguous

#### 5.5.2 Enrichment Based on Classification

**High Confidence (>70%)**:
- Enrich query with terms from primary domain only
- Use domain-specific synonyms

**Low Confidence (<70%)**:
- Enrich conservatively (only very generic terms)
- Let retrieval results disambiguate
- Possibly return results from multiple domains with labels

**Future Consideration**: As corpus grows to multiple domains, topic disambiguation becomes more important.

---

## 6. Data Model

### 6.1 Document Store

**Purpose**: Store source documents and their metadata.

**Document Entity**:
- Unique identifier
- File path or source URL
- Document title
- Content hash (for change detection)
- Document-level summary
- Domain classification
- Indexing timestamp
- Update timestamp

**Relationships**: One document has many chunks, one document summary.

### 6.2 Chunk Store

**Purpose**: Store retrievable text units with their metadata.

**Chunk Entity**:
- Unique identifier
- Parent document reference
- Chunk text content
- Section heading
- Full heading path (breadcrumb)
- Character offsets in source document
- Token count
- Extraction confidence score
- Needs-review flag
- Processing status (indexed, pending_review, reviewed)

**Relationships**: One chunk belongs to one document, one chunk belongs to one section, one chunk has many extracted terms.

### 6.3 Summary Store

**Purpose**: Store document and section summaries separately from chunks.

**Document Summary Entity**:
- Unique identifier
- Parent document reference
- Summary text
- Summary embedding

**Section Summary Entity**:
- Unique identifier
- Parent document reference
- Section identifier (heading path)
- Summary text
- Summary embedding
- References to chunk IDs covered by this section

**Design Note**: Summaries stored separately (not concatenated with chunks) to enable independent summary search and avoid diluting chunk embeddings.

### 6.4 Embedding Store

**Purpose**: Store vector embeddings for similarity search.

**Chunk Embedding Entity**:
- Chunk reference
- Embedding vector (dimensions depend on model)

**Summary Embedding Entities**:
- Document summary embeddings
- Section summary embeddings

**Index Requirements**: Support approximate nearest neighbor search.

### 6.5 Full-Text Index

**Purpose**: Support BM25 lexical search.

**Indexed Content**:
- Chunk text
- Enrichment terms (from term graph)
- Section headings

**Index Requirements**: Support BM25 scoring, phrase queries.

### 6.6 Term Graph Store

**Purpose**: Store vocabulary knowledge base.

**Term Entity**:
- Unique identifier
- Canonical form (official term)
- Term type (error, resource, command, concept)
- Domain (kubernetes, docker, general, etc.)
- Source (seed, llm_extracted, manual_review)
- Confidence score
- Document count (how many docs contain this term)
- Creation and update timestamps

**Synonym Entity**:
- Parent term reference
- Synonym text
- Direction (user→technical, technical→user, bidirectional)
- Source
- Confidence score

**Chunk-Term Association**:
- Chunk reference
- Term reference
- Extraction method (fast, slow, manual)

### 6.7 Processing Queues

**Purpose**: Track chunks requiring additional processing.

**Review Queue Entity**:
- Unique identifier
- Chunk reference
- Confidence score that triggered queueing
- Extracted terms (what fast system found)
- LLM candidates (what slow system suggested)
- LLM review result (approved, rejected, uncertain)
- Status (pending_llm, pending_human, completed, rejected)
- Reviewer notes
- Timestamps (created, llm_reviewed, human_reviewed)

### 6.8 Training Data Store

**Purpose**: Store examples for fast system retraining.

**Training Example Entity**:
- Unique identifier
- Chunk text (or reference)
- Extracted entities with labels
- Source (llm_approved, human_approved, human_corrected)
- Timestamp

**Rejection Log Entity**:
- Unique identifier
- Chunk reference
- Rejected term/synonym
- Rejection reason
- Source (llm_rejected, human_rejected)
- Timestamp

---

## 7. PoC Investigation Areas

### POC-1: LLM Term Extraction Guardrails ✅ COMPLETE

**Status**: Complete (POC-1, POC-1b, POC-1c)

**Question**: How reliably can LLM extract domain terms from unknown content with guardrails?

**Why This Matters**: The slow system relies on LLM extraction for novel terms. If hallucination rate is too high, manual review burden becomes unsustainable.

**Original Target**: P>95%, R>95%, H<5%

**Final Results**:

| Approach | Precision | Recall | Hallucination | F1 | Notes |
|----------|-----------|--------|---------------|-----|-------|
| V6 (with vocab) | 90.7% | 95.8% | 9.3% | 0.932 | Best with manual vocabulary (176 terms) |
| V6 @ 50 docs | 84.2% | 92.8% | 15.8% | 0.883 | Scale degradation |
| Retrieval few-shot | 81.6% | 80.6% | 18.4% | 0.811 | **Zero vocabulary maintenance** |
| SLIMER zero-shot | 84.9% | 66.0% | 15.1% | 0.743 | Definition-based |

**Key Findings**:
1. **95/95/5 NOT achievable** — benchmark ceiling is ~P=94%, R=96%, H=6% due to GT annotation gaps
2. **Vocabulary-free max: ~80/90/20** — retrieval few-shot eliminates maintenance with ~10% F1 drop
3. **GLiNER rejected** — produces garbage results for software entities, no usable signal
4. **Heuristic extraction viable** — CamelCase, backticks, ALL_CAPS patterns work well for fast system

**Recommendation**: Use retrieval few-shot for production (zero vocab maintenance). Accept the precision/hallucination tradeoff — for RAG, recall matters more than matching benchmark conventions.

**Artifacts**:
- `src/plm/extraction/` — Production extraction package (fast heuristic + slow V6)
- `poc/poc-1c-scalable-ner/RESULTS.md` — Full analysis
- `poc/poc-1c-scalable-ner/docs/V6_RESULTS.md` — V6 detailed breakdown

---

### POC-2: Confidence Scoring and Threshold Determination

**Question**: Do the proposed confidence signals correlate with extraction quality? What threshold should trigger slow processing?

**Why This Matters**: Fast/slow routing depends on confidence scores. If scores don't reflect actual quality, we'll either miss bad extractions or waste resources re-processing good ones.

**Test Design**:
- Extract from 100 chunks using fast system
- Manually grade extraction quality for each (good/acceptable/poor)
- Correlate grades with confidence signals
- Determine optimal threshold for slow system routing

**Signals to Validate**:
- Known term ratio
- Coverage
- Entity density
- Section type expectations

**Success Criteria**:
- Strong correlation (r > 0.6) between at least one signal and quality
- Combined signal achieves >80% accuracy at classifying good vs. poor
- Identified threshold that balances quality vs. slow system load

---

### POC-3: Topic Disambiguation with Single Query

**Question**: Can topic-first classification + single enriched query handle ambiguous terms effectively?

**Why This Matters**: Alternative is parallel multi-domain queries, which adds complexity and latency.

**Test Design**:
- Download Docker docs + Kubernetes docs into test corpus
- Create 20 questions with ambiguous terms (e.g., "container restart", "image pull error")
- For each: classify topic → enrich → search → evaluate
- Compare with baseline: parallel queries (one per domain) → merge results

**Success Criteria**:
- Single enriched query achieves >85% of parallel query Hit@5
- Classification accuracy >80%

---

### POC-4: Summary-Based Search Narrowing

**Question**: Does two-stage search (summaries → chunks) improve relevance for ambiguous queries?

**Why This Matters**: Summary generation is expensive. Need to validate ROI before generating summaries for entire corpus.

**Test Design**:
- Generate summaries for subset of corpus (20-30 documents)
- Create 20 queries that match multiple documents
- Compare: direct chunk search vs. summary-filtered chunk search

**Metrics**:
- Hit@5 improvement
- Precision@5 improvement
- Latency increase

**Success Criteria**:
- Hit@5 improvement >10%
- Validates investment in summary generation

---

### POC-5: Query Enrichment Faithfulness

**Question**: Does LLM query enrichment preserve user intent?

**Why This Matters**: If enrichment changes query meaning, we'll retrieve wrong documents even with good vocabulary bridging.

**Test Design**:
- Create 50 user queries (mix of specific and vague)
- LLM enriches each query
- Human evaluators rate: "Does enriched query have same intent as original?"
- Also test retrieval: compare results for original vs. enriched

**Success Criteria**:
- Intent preservation rated "same" or "very similar" for >90% of queries
- Retrieval relevance improves (not degrades) for >80% of queries

---

### POC-6: GLiNER Zero-Shot and Training Performance

**Question**: How well does GLiNER perform zero-shot? How much does fine-tuning improve it?

**Why This Matters**: GLiNER is proposed as the trainable NER component. Need to validate it works for general-purpose extraction.

**Test Design**:
- Test GLiNER zero-shot on diverse document types (technical docs, different domains)
- Collect 100-200 training examples from manual corrections
- Fine-tune GLiNER and measure improvement

**Metrics**:
- Zero-shot precision/recall on general entities
- Post-training precision/recall improvement
- Training time and effort

**Success Criteria**:
- Zero-shot precision > 60% on general entities
- Fine-tuning improves precision by >15%

---

### POC-7: Embedding Model Selection

**Question**: Which embedding model works best for technical documentation retrieval?

**Why This Matters**: Embedding model significantly impacts semantic search quality.

**Test Design**:
- Compare 3-4 embedding models on retrieval benchmark
- Test on diverse query types (specific, vague, technical, colloquial)

**Models to Test**:
- BAAI/bge-base-en-v1.5
- e5-large-v2
- Instruction-tuned models (e5-instruct)

**Success Criteria**:
- Identify model with best Hit@5 on benchmark
- Document quality vs. speed tradeoffs

---

## 8. Design Decisions & Tradeoffs

### D1: LLM for All Queries vs. Selective Processing

**Decision**: LLM processes every query.

**Rationale**:
- Avoids heuristic complexity ("is this query good enough?")
- Extracts useful signals even from well-formed queries
- Consistent code path

**Tradeoff**: Latency cost for every query.

**Mitigation**: 
- Use fast model (Haiku)
- Cache frequent queries
- Fallback to dictionary expansion if LLM unavailable

---

### D2: Summaries in Separate Store vs. Concatenated with Chunks

**Decision**: Store summaries separately.

**Rationale**:
- Different retrieval purpose (scope narrowing vs. content matching)
- Avoids diluting chunk embeddings
- Enables two-stage search pattern
- Independent update cycle

**Tradeoff**: May lose some reranking signal (reranker doesn't see summary context).

---

### D3: GLiNER vs. spaCy for NER

**Decision**: Use GLiNER instead of spaCy.

**Rationale**:
- spaCy's general NER doesn't recognize domain-specific entities
- GLiNER is zero-shot (works on any domain without training)
- GLiNER is trainable (improves with slow system corrections)
- Supports dynamic entity types

**Tradeoff**: Slightly slower than spaCy, requires more memory.

---

### D4: Synonyms Only vs. Term Relationships

**Decision**: Synonyms only, no complex relationships (causes, part_of).

**Rationale**:
- Simpler to maintain
- Relationship extraction is hard and error-prone
- Co-occurrence in documents captures implicit relationships
- 80% of value with 20% of complexity

**Tradeoff**: May miss some query expansion opportunities.

**Alternative**: Mine co-occurrence from document structure if relationship-based expansion needed later.

---

### D5: Tiered Review (LLM + Human) vs. Human-Only

**Decision**: LLM review tier before human review.

**Rationale**:
- Scales to large review queues
- Handles obvious cases automatically
- Reduces human review burden
- Humans focus on genuinely uncertain cases

**Tradeoff**: LLM may make mistakes (mitigated by conservative thresholds).

---

### D6: Two-Pass Ingestion for Bootstrap

**Decision**: Use two-pass ingestion for initial corpus load.

**Rationale**:
- Single-pass would flood slow system with false low-confidence flags
- Two-pass builds term graph before confidence scoring
- Later ingestion uses single-pass normally

**Tradeoff**: Initial ingestion takes longer (acceptable for one-time cost).

---

### D7: Self-Improving Fast System

**Decision**: Fast system (GLiNER) is retrained on slow system corrections.

**Rationale**:
- Every correction improves future extraction
- Reduces slow system load over time
- System gets smarter with use

**Tradeoff**: Requires training infrastructure and periodic retraining effort.

---

## 9. Open Questions

### Architecture Questions

1. **Batch vs. Streaming Slow Processing**: Should slow system process chunks in daily batches or as they're queued?

2. **Summary Generation Timing**: Generate summaries during initial ingestion or as separate batch process?

3. **Multi-Domain Support**: How to handle queries that legitimately span multiple domains? Return results from both with labels?

### Technology Questions

4. **LLM Model Selection**: Haiku vs. Sonnet for query processing? Opus vs. Sonnet for LLM review tier? Cost vs. quality tradeoff.

5. **Storage Backend**: SQLite vs. PostgreSQL vs. dedicated vector database?

6. **Cache Strategy**: What to cache beyond embeddings? Query results? LLM responses?

### Operational Questions

7. **GLiNER Retraining Frequency**: Monthly? Weekly? On-demand when examples accumulate?

8. **Review Queue Prioritization**: When queue is large, which chunks to process first?

9. **Quality Monitoring**: How to detect when extraction quality degrades over time?

---

*Document Version: 3.0*
*Last Updated: 2026-02-03*
