# Draft: PLM as MCP Server Strategy

## Current System Analysis

### Existing Architecture
- **FastAPI service** with `/query`, `/health`, `/status` endpoints
- **Response format**: QueryResult with chunk_id, doc_id, content, enriched_content, score, heading, start_char, end_char
- **Hierarchical retrieval**: chunks, headings, documents levels
- **MCP SERVER ALREADY EXISTS** at `src/plm/search/mcp/server.py` ✅

### MCP Server Status (PRODUCTION READY)
- **Entry point**: `plm-mcp` command
- **Framework**: FastMCP
- **5 Tools exposed**:
  1. `search(query, k, use_rewrite)` - Hybrid chunk-level search
  2. `get_status()` - Index health check
  3. `get_chunk(chunk_id)` - Retrieve full chunk details
  4. `search_headings(query, k)` - Section-level semantic search
  5. `search_documents(query, k)` - Document-level semantic search

### Current Response Shape (for humans)
```python
QueryResult:
  chunk_id: str
  doc_id: str
  content: str        # Raw chunk text
  enriched_content: str  # With keywords/entities
  score: float
  heading: str | None
  start_char: int | None
  end_char: int | None
```

## User Requirements (Confirmed)

### Primary Use Case
**General-Purpose Knowledge Base / Institutional Memory**
- NOT just debugging docs - general notes, official docs (k8s), etc.
- Documents likely written by LLMs (well-structured markdown with headings)
- Query patterns unknown - could be anything from LLM or user

### Target Consumer
- **Claude via OpenCode/API** - native MCP support, 200k context

### Token Strategy
- **Return as much as needed** - full top-k chunks
- No aggressive truncation - let the LLM handle context

### Key Insight
This is **LLM-to-LLM** knowledge transfer:
- Documents often written by LLMs → well-structured, consistent
- Queries often from LLMs → potentially more precise than human queries
- Both producer and consumer are LLMs → different optimization targets

### Scope & Scale
- **Corpus size**: Thousands+ documents (large scale)
- **Growth**: Continuously adding
- **Write path**: Not in scope now (retrieval focus)

### Document Usage Patterns (all apply)
1. **Cite verbatim** - copy-paste with attribution
2. **Synthesize/adapt** - rephrase for current situation
3. **Execute steps** - follow how-to instructions

---

## Design Decisions (Confirmed)

### 1. Query Strategy: INTELLIGENT PLM (Key Decision)
**PLM should decide what level of detail to return, not the calling LLM**
- Current: 3 separate tools (search, search_headings, search_documents)
- Desired: Single smart search that auto-selects appropriate granularity
- Rationale: "Can't trust outer LLM to know what and how to search"

### 2. Context Expansion: YES
- Get siblings (adjacent chunks)
- Get parent (heading section)
- Enable drilling down and expanding

### 3. Metadata & Filtering: ESSENTIAL
- Filter by document type (e.g., "only k8s docs")
- Filter by date/recency
- Filter by tags/categories

### 4. Trust Signals: RRF Score is Sufficient
- Documents in library are assumed ground truth
- No need for "uncertainty" signaling
- RRF score indicates match quality, not content trustworthiness

---

## Existing Building Blocks

| Component | Location | Status |
|-----------|----------|--------|
| **QueryRewriter** | `src/plm/search/components/query_rewriter.py` | Production |
| **classify_query()** | `poc/adaptive_retrieval/scripts/run_adaptive_classifier.py` | POC |
| **LOD-LLM** | `poc/chunking_benchmark_v2/retrieval/lod_llm.py` | POC |
| **Hierarchical Storage** | `src/plm/search/storage/sqlite.py` | Production |

## Research Findings: Query Classification/Routing

### What Was Tried and REJECTED
| Approach | Result | Why Rejected |
|----------|--------|--------------|
| **Rule-based regex** | 63.3% accuracy | Too inaccurate, over-routes to complex |
| **LOD LLM routing** | Works but slow | 30s+ per query (Ollama) |

### What Was DESIGNED but Never Implemented
From `docs/RESEARCH.md` — **2-stage pipeline with small models**:

```
Stage 1: cnmoro/granite-question-classifier (30M, ~5ms)
  └── generic → document-level
  └── directed → Stage 2

Stage 2: MoritzLaurer/xtremedistil (13M, ~10ms)
  ├── factual → chunk-level
  ├── conceptual → heading-level
  ├── troubleshooting → heading + chunk
  └── procedural → document → expand

Total: ~43M params, ~15ms CPU, ~80MB RAM
```

### Small Model Options (Not Yet Tried)
| Model | Params | Type | Speed | Notes |
|-------|--------|------|-------|-------|
| `MoritzLaurer/xtremedistil-l6-h256-zeroshot` | **13M** | Zero-shot | ⚡⚡⚡ | Any labels at runtime |
| `cnmoro/granite-question-classifier` | **30M** | Binary | ⚡⚡⚡ | 94% accuracy |
| `Danswer/intent-model` | 67M | Fixed labels | ⚡⚡ | Production-proven |
| Fine-tune `bert-mini` | **11M** | Trained | ⚡⚡⚡ | Best long-term |

### THE GAP
**Small model classification was never implemented/tested**. This is the unexplored path.

---

## External Research: Small Models for Query Routing (2026 SOTA)

### Fastest Approaches (Production-Ready)

| Approach | Latency | Accuracy | Notes |
|----------|---------|----------|-------|
| **Semantic Router + MiniLM** | **15ms** | 80-85% | `pip install semantic-router` |
| **SetFit (few-shot)** | 20ms | 85-92% | 8-32 examples per class |
| **DistilBERT fine-tuned** | 30ms | 90%+ | Standard classification |
| **Phi-3-mini (3.8B)** | 100ms | 93% | LLM fallback |

### Recommended Hybrid Strategy
```
Query → Rule pre-filter (1ms) 
      → Semantic Router (15ms) [handles 80%]
      → LLM fallback (200ms) [complex cases]

Expected: 80% @ <20ms, 15% @ 20-50ms, 5% @ 200ms
Average: ~30ms
```

### Key Libraries
- `semantic-router` (Aurelio Labs) — embedding-based routing
- `setfit` (HuggingFace) — few-shot classification
- `sentence-transformers` — MiniLM embeddings

## Design Decisions (Confirmed)

### Performance Constraints
| Dimension | Decision | Notes |
|-----------|----------|-------|
| **Latency** | < 5 seconds | Quality over speed, multi-step OK |
| **Cost** | Prefer local LLM/rules | Haiku acceptable if needed |
| **Context Expansion** | Automatic | PLM decides optimal amount |
| **Filtering** | Natural language + explicit | Both supported |

### Target Architecture

```
ask(question, filters?) → PLM Intelligence → Optimal Response
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│                   PLM INTELLIGENCE LAYER                     │
├─────────────────────────────────────────────────────────────┤
│  1. Parse Query                                              │
│     - Extract explicit filters (if any)                      │
│     - Extract implicit filters from NL                       │
│     - Classify query complexity                              │
│                                                              │
│  2. Route to Retrieval Strategy                              │
│     - SIMPLE → chunk search (k=5-10)                        │
│     - COMPLEX → heading search + expand to chunks           │
│     - BROAD → document search + drill down                  │
│                                                              │
│  3. Auto-Expand Context                                      │
│     - Get siblings if chunk boundary seems mid-sentence     │
│     - Include parent heading for context                    │
│     - Aggregate if multiple chunks from same section        │
│                                                              │
│  4. Format Response                                          │
│     - Structured for LLM consumption                        │
│     - Include citations (doc, heading, offsets)             │
│     - Include relevance scores                               │
└─────────────────────────────────────────────────────────────┘
```

### Validation Strategy
- Benchmark against SOTA local RAG tools (LlamaIndex, RAGFlow, Haystack, etc.)
- Find comparable tool, run same queries, compare results
- Research phase needed to identify best comparison target

### Implementation Priority
1. **Unified `ask()` tool** - the intelligent interface
2. Internal routing logic
3. Context expansion
4. Filtering (source/type only for now)

### Filter Schema (Initial)
- Source file path patterns (`*.md`, `k8s/*`, etc.)
- Keep simple, extend later

---

## Tools to Incorporate (Candidates for POC)

### Query Classification/Routing
| Tool | Type | Notes |
|------|------|-------|
| `semantic-router` (Aurelio Labs) | Embedding-based | ~15ms, 80-85% accuracy, production-ready |
| `MoritzLaurer/xtremedistil-l6-h256-zeroshot` | Zero-shot classifier | 13M params, any labels at runtime |
| `cnmoro/granite-question-classifier` | Binary gate | 30M params, 94% accuracy on generic/directed |
| `Danswer/intent-model` | Fixed labels | 67M params, production-proven in Danswer RAG |
| `SetFit` | Few-shot | 8-32 examples per class, 85-92% accuracy |

### Reranking
| Tool | Type | Notes |
|------|------|-------|
| `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder | Already integrated |
| `BAAI/bge-reranker-base` | Cross-encoder | Alternative option |
| `Cohere Rerank` | API | High quality, but adds cost |

### Retrieval Enhancement
| Tool | Type | Notes |
|------|------|-------|
| `SPLADE` | Learned sparse | Already have it |
| `ColBERT` | Late interaction | Token-level matching |
| `HyDE` | Query expansion | Generate hypothetical doc, then search |

### Agentic/Self-Correction
| Tool | Type | Notes |
|------|------|-------|
| `Self-RAG` | Reflection tokens | LLM decides when to retrieve |
| `CRAG` | Corrective RAG | Evaluate retrieval quality, fallback to web |

### Knowledge Graphs (Future)
| Tool | Type | Notes |
|------|------|-------|
| `neo4j` | Graph DB | Entity relationships |
| `GraphRAG` (Microsoft) | LLM-extracted | Knowledge graph from docs |

---

## Open Source Solutions to Investigate

### MCP RAG Servers (Direct Competitors)
| Project | Stars | Key Features | Why Investigate |
|---------|-------|--------------|-----------------|
| `qdrant/mcp-server-qdrant` | 1.2k | Semantic search, FastEmbed, multi-transport | Most popular MCP RAG server |
| `RAG-C` (Aparnap2) | - | Hybrid + RRF + reranking | Closest to our architecture |
| `RagDocs` (heltonteixeira) | - | Qdrant + Ollama/OpenAI | Simple reference |
| `DocuMCP` | - | Privacy-focused, local-first | Same philosophy as us |
| `Minima` (dmayboroda) | - | Local RAG for files | Minimal implementation |

### Full RAG Systems
| Project | Key Features | Why Investigate |
|---------|--------------|-----------------|
| `R2R` (SciPhi) | Agentic RAG, knowledge graphs | Enterprise-grade patterns |
| `Danswer` | Open source enterprise search | Uses intent-model we're considering |
| `Verba` (Weaviate) | RAG chatbot | Weaviate integration patterns |
| `RAGFlow` | Document parsing quality | Chunking strategies |
| `Haystack` (deepset) | Composable pipelines, routers | Router architecture |

### NotebookLM Clones
| Project | Key Features | Why Investigate |
|---------|--------------|-----------------|
| `notebooklm-mastra` | Agent orchestration | Architecture patterns |
| `Quivr` | "Second brain" with RAG | UX patterns |

### Query Understanding
| Project | Key Features | Why Investigate |
|---------|--------------|-----------------|
| `Adaptive-RAG` (starsuzi) | Query complexity classifier | Auto-labeled training approach |

---

## Competitive Differentiation

### What Existing MCP Servers Do
- Pure semantic search (Qdrant MCP)
- Hybrid retrieval without intelligence (RAG-C)
- No query classification or routing
- Caller must orchestrate granularity

### What We Add
- **Query intent classification** → auto-route to optimal level
- **Granularity selection** → chunk vs heading vs document
- **Context expansion** → siblings + parent heading
- **Unified `ask()` interface** → PLM decides, not caller
- **NL filter parsing** → "only k8s docs" works

### Our Niche
**Intelligent MCP knowledge server** — not a dumb pipe, but an intermediary that understands queries and optimizes retrieval automatically.

---

## OSS Research Learnings (2026-02-24)

### From Semantic-Router (Aurelio Labs)

| Pattern | How We Apply It |
|---------|-----------------|
| **4-stage pipeline** (encode→retrieve→score→filter) | Our flow: Classify → Route → Retrieve → Rerank → Expand |
| **Per-route thresholds** | Each intent class gets its own confidence threshold, not global |
| **fit() optimization** | Train thresholds on labeled examples; random search beats manual tuning |
| **Lazy LLM execution** | LLM called ONLY when classifier confidence is low or synthesis needed |
| **Score aggregation** | When multiple routes match, aggregate (mean/max) then pick best |
| **Include negative examples** | Train with `y=None` cases to prevent over-triggering |
| **Graceful failure** | Return empty choice, not exception — let caller decide fallback |

### From Danswer (Onyx)

| Danswer Lesson | Our Decision |
|----------------|--------------|
| **Deprecated intent model** — "let search handle it" | ⚠️ Classification is OPTIONAL — fallback to retrieve-all when uncertain |
| **Time decay for relevance** | ✅ ADOPT: `score * exp(-decay_rate * age_days)` |
| **Multi-vector per document** | ✅ Already have: chunks + headings + documents all embedded |
| **Unified hybrid search** | ✅ Already have: SPLADE + semantic + RRF in one retriever |

### From Haystack (deepset)

| Pattern | How We Apply It |
|---------|-----------------|
| **@component decorator** | Wrap our components for auto-validation, introspection |
| **Warm-up pattern** | Lazy-load models on first use, not at import |
| **Jinja2-based routing** | Serializable rules (config-driven, not code-driven) |
| **Pipeline serialization** | YAML config for pipeline variants |

### From R2R (SciPhi)

| Pattern | How We Apply It |
|---------|-----------------|
| **Observability** | Log every decision: query → classification → retrieval → rerank scores |
| **Retry mechanisms** | If retrieval fails, retry with broader granularity |
| **Batch processing** | Support bulk queries with progress tracking |
| **Configurable prompts** | Tool descriptions in YAML, not hardcoded |
| **Postgres + pgvector is enough** | We use SQLite + FAISS — same principle: no separate graph DB |

### From Qdrant MCP

| Pattern | How We Apply It |
|---------|-----------------|
| **FastMCP architecture** | Already using FastMCP ✅ |
| **Environment-only config** | All settings via env vars, no CLI args |
| **Dynamic tool registration** | Hide parameters based on config |
| **Custom tool descriptions** | `PLM_SEARCH_DESCRIPTION` env var for different use cases |
| **Return None, not []** | LLMs handle `None` better than empty list |

### From RAG Best Practices

| Practice | Our Status |
|----------|------------|
| **1. Hybrid search** | ✅ SPLADE + semantic + RRF |
| **2. Cross-encoder reranking** | ✅ ms-marco-MiniLM integrated |
| **3. Query classification** | ⚠️ OPTIONAL — fallback when uncertain |
| **4. Systematic evaluation** | ❌ Need to build test set |
| **5. User feedback loop** | ❌ Need to add thumbs up/down |

---

## Why No Unified RAG Solution Exists (Research Findings)

### The Latency-Accuracy-Cost Trilemma

You **cannot** optimize all three simultaneously:
- **Speed** → Cache-Augmented Generation (2.33s vs RAG's 94s)
- **Accuracy** → Agentic RAG (+50% F1) but 10x cost
- **Cost** → Vanilla RAG but compromised quality

### Our Choice: Accuracy > Cost > Latency

**User directive**: "Latency can be as big as needed. Optimize for accuracy first, cost second."

### Production Pitfalls to Avoid

| Pitfall | What Happens | Our Mitigation |
|---------|--------------|----------------|
| **Latency Explosion** | 7 layers × 100ms = 700ms before LLM | Accept it — accuracy over speed |
| **Cost Explosion** | All optimizations = 7x baseline | Local models only, lazy LLM calls |
| **Complexity Trap** | Debugging becomes guesswork | Observability from day 1 |
| **Generalization Fallacy** | Optimized for nothing | Focus on our corpus type (LLM-written markdown) |

---

## Final Design Decisions

### Confirmed Choices

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Classification** | **NONE** — unified ranking handles it implicitly | Vespa/Danswer lesson: "let search handle it" — ranking adapts to query type automatically |
| **Filter parsing** | **COMPLEX** — small model for NL → structured | User preference; "recent kubernetes docs" → `{source: "k8s", recency: 30d}` |
| **Ranking strategy** | **UNIFIED** — Vespa-style single scoring function | BM25 + semantic + time decay in one expression, query-adaptive alpha |
| **Latency budget** | **Unlimited** | Quality over speed |
| **Cost strategy** | **Local models only** | No API calls for routing; filter parser is only model |
| **Time decay** | `score * exp(-0.1 * age_days)` | Danswer pattern, built into unified ranking |
| **Warm-up pattern** | Lazy-load models on first query | Haystack pattern |
| **Config** | Env vars only, no CLI args | Qdrant MCP pattern |
| **Tool descriptions** | Configurable via `PLM_*_DESCRIPTION` | Qdrant MCP pattern |
| **Return None not []** | When no results | Qdrant MCP pattern |
| **Observability** | Log retrieval + rerank scores | R2R pattern |
| **Feedback loop** | Capture thumbs up/down | RAG best practices |

---

## Vespa-Inspired Unified Ranking (Key Design)

### Why No Explicit Classifier

**Danswer's lesson**: They deprecated their DistilBERT intent classifier after migrating to Vespa. The unified ranking expression made classification unnecessary.

**Vespa's insight**: Instead of routing queries to different systems, evaluate ALL signals (BM25 + semantic) in one formula and let the math decide.

### Our Approach: Apply Vespa Philosophy to Existing Stack

**Keep SQLite + FAISS** (no migration), but implement Vespa-style ranking:

```python
def unified_rank(query: str, candidates: List[Document]) -> List[Document]:
    """Vespa-style unified ranking — classification is implicit"""
    
    query_embedding = embed(query)
    query_terms = tokenize(query)
    
    for doc in candidates:
        # Component scores
        bm25_score = compute_bm25(query_terms, doc)
        vector_score = cosine_similarity(query_embedding, doc.embedding)
        time_score = exp(-0.001 * doc.age_days)
        
        # Query-adaptive alpha (implicit classification)
        # Short queries → favor BM25 (keyword lookup)
        # Long queries → favor semantic (conceptual search)
        query_length = len(query.split())
        alpha = 0.7 - 0.1 * min(query_length, 4)  # Range: 0.3 to 0.7
        
        # Unified score
        doc.score = (
            alpha * bm25_score + 
            (1 - alpha) * vector_score + 
            0.1 * time_score
        )
    
    return sorted(candidates, key=lambda d: d.score, reverse=True)
```

### How It Handles Different Query Types

| Query Type | Example | What Happens |
|------------|---------|--------------|
| **Keyword lookup** | "HPA config" | Short query → high alpha → BM25 dominates |
| **Conceptual** | "why won't my pods autoscale when load increases" | Long query → low alpha → semantic dominates |
| **Mixed** | "kubernetes autoscaling troubleshooting" | Medium query → balanced weighting |

**No classifier model needed** — the ranking function adapts automatically.

### Benefits Over Explicit Classification

| Aspect | Explicit Classifier | Unified Ranking |
|--------|---------------------|-----------------|
| **Models to maintain** | SetFit + retriever + reranker | Retriever + reranker only |
| **Training data needed** | Labeled intent examples | None (self-adapting) |
| **Failure modes** | Misclassification → wrong retrieval | Graceful degradation |
| **Complexity** | 3-stage pipeline | 2-stage pipeline |
| **Danswer's verdict** | "Deprecated" | "Finally achieved accuracy we wanted"

---

## Revised Component Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  MCP Interface (FastMCP)                                    │
│  └── ask(query, filters?) → Response                        │
│  └── Custom descriptions via PLM_*_DESCRIPTION env vars     │
├─────────────────────────────────────────────────────────────┤
│  Filter Parser (only model in pipeline)                     │
│  └── Small model for NL → structured filters                │
│      "recent k8s docs" → {source: "k8s", recency: 30d}      │
├─────────────────────────────────────────────────────────────┤
│  Unified Retrieval Engine (Vespa-style)                     │
│  ├── Multi-level: chunks + headings + documents             │
│  ├── SPLADE + semantic (parallel retrieval)                 │
│  ├── Unified ranking: alpha*BM25 + (1-α)*semantic + decay   │
│  ├── Query-adaptive alpha (implicit classification)         │
│  ├── Time decay: score * exp(-decay * age)                  │
│  └── Cross-encoder reranker (second phase)                  │
├─────────────────────────────────────────────────────────────┤
│  Context Expansion                                          │
│  ├── Heading-aware sibling inclusion                        │
│  ├── Parent context when chunk is top result                │
│  └── Token budget management                                │
├─────────────────────────────────────────────────────────────┤
│  Production Features                                        │
│  ├── Observability: log scores at each stage                │
│  ├── Retry: broader granularity on failure                  │
│  ├── Batch: bulk queries with progress                      │
│  └── Feedback: thumbs up/down capture                       │
├─────────────────────────────────────────────────────────────┤
│  Storage (existing — no changes needed)                     │
│  └── SQLite + FAISS                                         │
└─────────────────────────────────────────────────────────────┘
```

**Key simplification**: No classifier model. Query type adaptation is implicit in the unified ranking function.

---

## Implicit Query Adaptation (No Explicit Classification)

**How unified ranking handles different query types**:

| Query Pattern | Alpha Value | Dominant Signal | Result |
|---------------|-------------|-----------------|--------|
| Short (1-2 words) | 0.6-0.7 | BM25 (keyword) | Exact term matches surface |
| Medium (3-5 words) | 0.4-0.5 | Balanced | Both signals contribute |
| Long (6+ words) | 0.3-0.4 | Semantic | Conceptual matches surface |

**Granularity strategy**:
- Retrieve at ALL levels (chunk + heading + document)
- Let cross-encoder reranker sort by relevance
- Context expansion adds surrounding content as needed

**Why this works**:
- Short queries ("HPA config") are typically keyword lookups → BM25 handles it
- Long queries ("why pods don't scale under load") need semantic understanding → vector handles it
- Mixed queries get balanced treatment automatically

**No explicit intent labels needed** — the math adapts.

---

## What We're NOT Building (Anti-Patterns)

| Anti-Pattern | Source | Our Avoidance |
|--------------|--------|---------------|
| Separate search engines | Danswer | Unified SPLADE + semantic |
| Explicit intent classifier | Danswer/Vespa | Unified ranking handles it implicitly |
| Hardcoded prompts | R2R | YAML config for prompts |
| Eager model loading | Haystack | warm_up() pattern |
| Complex graph DB | R2R | SQLite + FAISS sufficient |
| Knowledge/term graphs | Industry research | Maintenance burden not justified |
| CLI args | Qdrant MCP | Env vars only |
| Vespa (the database) | Our analysis | Overkill for our scale; apply philosophy instead |

---

## Implementation Phases

### Phase 1: Foundation
- Single `ask()` MCP endpoint replacing 5 tools
- Unified ranking function (Vespa-style)
- Query-adaptive alpha weighting
- Time decay in scoring
- Retrieve at all levels, let reranker sort

### Phase 2: Intelligence
- Filter parser (small model for NL → structured)
- Context expansion (heading-aware siblings)
- Token budget management

### Phase 3: Production
- Observability layer (log scores at each stage)
- Feedback capture (thumbs up/down)
- Batch processing
- Retry mechanisms

---

## Open Items (Investigation Phase)

1. **Filter parser model selection** — which small model for NL → structured?
2. **Alpha tuning** — optimal query-length-to-alpha mapping for our corpus
3. **Evaluation test set** — 100+ queries with expected results
4. **Time decay curve** — what decay rate fits our corpus?
5. **Context expansion heuristics** — when to include siblings/parents?

---

## Research Sources

### Vespa Investigation (2026-02-24)
- **Why Vespa eliminates classification**: Unified ranking expression evaluates BM25 + semantic in same formula
- **Danswer's experience**: Deprecated DistilBERT classifier after Vespa migration
- **Key quote**: "Vespa allows for easy normalization across multiple search types"
- **Our approach**: Apply Vespa's philosophy to existing SQLite + FAISS stack (no migration needed)
- **Trade-off**: Vespa is overkill for thousands of docs; we get same benefit with simpler stack

---
*Draft updated: 2026-02-24*
