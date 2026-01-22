# Personal Knowledge Assistant - Research & Design Document

> A NotebookLM-like system with TUI interface for document-grounded Q&A with citations

## Project Vision

Build a local-first, TUI-based knowledge assistant that:
- Ingests documents (PDF, markdown, text) into a structured knowledge base
- Answers questions with **90-100% accuracy** grounded in source material
- Provides **exact citations** with references to source documents
- Uses hierarchical document indexing (LOD) for efficient retrieval
- Employs multi-agent orchestration (smart coordinator + specialized workers)

---

## Research Findings

### 1. Google NotebookLM Architecture

**How it works:**
```
Document Upload → Parse → Segment into passages → Embed (Gemini) → Vector Index
                                                                        ↓
User Query → Embed query → Cosine similarity search → Top-K retrieval
                                                                        ↓
                              Retrieved passages + Query → Gemini → Grounded response with citations
```

**Key design principles:**
- **Source grounding**: Every claim MUST cite a source passage
- **No hallucination by design**: Model can ONLY use uploaded documents
- **Inline citations**: References back to exact source location

**Sources:**
- [NotebookLM: An LLM with RAG (arXiv)](https://arxiv.org/html/2504.09720v2)
- [Breaking Down NotebookLM (Medium)](https://deepganteam.medium.com/breaking-down-notebooklm-fa0bcb1526d9)

---

### 2. Hierarchical Document Indexing (LOD Concept)

Our original idea of Level-of-Detail indexing aligns with **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval).

#### RAPTOR Architecture
```
                    [Document Summary]           ← Level 3 (most abstract)
                          ↓
            [Section 1 Summary] [Section 2 Summary]  ← Level 2
                    ↓                   ↓
        [Para summaries...]    [Para summaries...]   ← Level 1
              ↓                       ↓
    [Original chunks + embeddings]                   ← Level 0 (most detailed)
```

**How RAPTOR builds the tree:**
1. Chunk document into segments
2. Embed all chunks
3. Cluster similar chunks (using GMM or k-means)
4. Summarize each cluster → creates parent node
5. Recursively repeat until single root summary

**Query strategies:**
- **Tree traversal**: Start at root, descend to relevant branches
- **Collapsed tree**: Search ALL levels simultaneously, let relevance decide

**Benefits for our use case:**
- Broad questions → match high-level summaries
- Specific questions → match detailed chunks
- Reduces context window usage (don't load full documents)

**Sources:**
- [RAPTOR Paper (arXiv)](https://arxiv.org/abs/2401.18059)
- [Official Implementation](https://github.com/parthsarthi03/raptor)
- [LlamaIndex HierarchicalNodeParser](https://docs.llamaindex.ai/en/stable/examples/node_parsers/hierarchical/)

---

### 2.1 RAPTOR Deep Analysis: Benefits & Drawbacks

#### What Problem RAPTOR Solves

Traditional RAG suffers from **context fragmentation**:
```
Traditional RAG:
  Query: "What is the main thesis of this paper?"
  → Retrieves random chunks mentioning keywords
  → Misses the big picture

RAPTOR:
  → Creates hierarchy of abstractions
  → Root level captures thesis
  → Leaf level has details
  → Query matches appropriate level
```

#### How the Clustering Works

RAPTOR uses **soft clustering** via UMAP + GMM:

```
1. UMAP: Reduce embedding dimensions (1536 → 10)
   - Adaptive n_neighbors = sqrt(n-1)
   - Preserves local + global structure

2. GMM: Find optimal clusters via BIC (Bayesian Information Criterion)
   - Auto-selects number of clusters
   - BIC balances fit vs complexity

3. Soft Assignment: Node can belong to MULTIPLE clusters
   - threshold=0.1: If P(cluster|node) > 0.1, node joins cluster
   - "ML in healthcare" → belongs to ML cluster AND healthcare cluster

4. Recursive Split: If cluster > max_tokens, re-cluster
```

#### Two Retrieval Strategies

| Strategy | How It Works | Best For |
|----------|--------------|----------|
| **Collapsed** (default) | Search ALL nodes as flat set | Broad queries: "What is this about?" |
| **Tree Traversal** | Start at root, descend through children | Specific: "What does section 3 say?" |

#### Benefits

1. **Multi-Scale Understanding**
   - Broad questions match summaries
   - Specific questions match leaves
   - Same index serves both

2. **Better Semantic Grouping**
   - Clustering creates natural topics
   - Soft clustering preserves overlapping concepts
   - Cross-document themes emerge at higher levels

3. **Token Efficiency**
   - Without hierarchy: 10 chunks → 4000 tokens
   - With hierarchy: 3 summaries + 2 leaves → 2000 tokens (same info density)

4. **Clean Extension Points**
   ```python
   # Easy to swap components
   class MyEmbedding(BaseEmbeddingModel): ...
   class MySummarizer(BaseSummarizationModel): ...
   class MyClustering(ClusteringAlgorithm): ...
   ```

#### Drawbacks

1. **API Cost Scales Badly**
   
   | Doc Size | Chunks | Embedding Calls | Summary Calls | Cost |
   |----------|--------|-----------------|---------------|------|
   | 5K tok | 50 | ~65 | ~13 | ~$0.05 |
   | 50K tok | 500 | ~650 | ~130 | ~$0.50 |
   | 500K tok | 5,000 | ~6,500 | ~1,300 | ~$5.00 |
   | 5M tok | 50,000 | ~65,000 | ~13,000 | ~$50.00 |

2. **Indexing is Slow**
   - 100 chunks ≈ 1-2 minutes
   - 10,000 chunks ≈ multi-hour

3. **Memory: UMAP O(n²) Problem**
   - 1,000 nodes: ~8MB
   - 10,000 nodes: ~800MB  
   - 100,000 nodes: ~80GB ← Breaks

4. **Code Quality: 5.5/10**
   - Missing `self` parameter bug
   - Race condition in threading
   - Exception objects returned as answers
   - No tests

5. **No Incremental Updates**
   - Add 1 document → Rebuild ENTIRE tree
   - Need: Update only affected branches

6. **Summarization Quality Dependency**
   - Bad summary at layer 1 → propagates up
   - No automatic quality verification

7. **Threshold Sensitivity**
   - 0.05: Lots of overlap, expensive
   - 0.50: May miss connections
   - No principled way to choose

#### Scaling Limits

| Scale | Feasibility | Notes |
|-------|-------------|-------|
| ~100 docs | ✅ Easy | Works fine |
| ~1,000 docs | ⚠️ Feasible | 1-2 hours, ~$5 |
| ~10,000 docs | ⛔ Challenging | 10-20 hours, ~$50 |
| ~100,000 docs | ❌ Impractical | Architecture changes needed |

#### What to Adopt vs Change for Our Implementation

**ADOPT from RAPTOR:**
- ✅ Hierarchical tree structure concept
- ✅ Multi-level retrieval (collapsed + traversal)
- ✅ Clean base class extension points
- ✅ Soft clustering idea (content → multiple groups)

**CHANGE/IMPROVE:**
- ❌ UMAP+GMM → Consider HDBSCAN (faster, no K needed)
- ❌ Per-call API → Batch embedding, aggressive caching
- ❌ Full rebuild → Design for incremental updates from start
- ❌ Research code → Production-grade from day 1
- ❌ No persistence → SQLite + sqlite-vec from start
- ❌ Pure abstractive → Consider extractive + abstractive hybrid

**Open Questions for Our Design:**
1. Do we need soft clustering? (Adds complexity)
2. Can we use local models to eliminate API costs?
3. How do we handle incremental updates efficiently?
4. What's our target scale? (100 docs? 10K docs?)

---

### 3. Multi-Agent Orchestration Patterns

#### Pattern: Orchestrator + Specialist Workers

```
┌────────────────────────────────────────────────────────────┐
│                  ORCHESTRATOR AGENT                         │
│              (Large model: Claude Sonnet)                   │
│                                                              │
│  Responsibilities:                                          │
│  • Analyze query complexity                                 │
│  • Decompose into sub-tasks                                 │
│  • Route to specialist agents                               │
│  • Synthesize final answer with citations                   │
└──────────────────────┬─────────────────────────────────────┘
                       │
    ┌──────────────────┼──────────────────┐
    ▼                  ▼                  ▼
┌─────────┐     ┌─────────────┐    ┌─────────────┐
│ ROUTER  │     │  RETRIEVER  │    │  VERIFIER   │
│ (Small) │     │   (Small)   │    │   (Small)   │
└─────────┘     └─────────────┘    └─────────────┘
Decides LOD     Extracts facts     Cross-checks
level to        from passages      citations
search
```

#### Key Lesson from Mastra's NotebookLM Clone

> "The success of an orchestrator agent depends heavily on **INCREDIBLY DETAILED prompts**. Without explicit phases and tool definitions, agents skip steps or hallucinate capabilities."

**Their orchestrator phases:**
1. Validate sources available
2. Query summaries and chunks
3. Identify key insights
4. Generate outline
5. Generate script
6. Review output

**Sources:**
- [Mastra NotebookLM Clone Blog](https://mastra.ai/blog/notebooklm-clone-with-agent-orchestration)
- [MA-RAG Paper](https://arxiv.org/html/2505.20096v2)
- [MAO-ARAG Paper](https://arxiv.org/abs/2508.01005)

---

### 4. Existing Open-Source Implementations

| Project | Stack | Features | Notes |
|---------|-------|----------|-------|
| [mastra-ai/notebooklm-mastra](https://github.com/mastra-ai/notebooklm-mastra) | TS, PgVector, Claude | Agent orchestrator, podcast gen | Good reference for orchestration |
| [patchy631/notebook-lm-clone](https://github.com/patchy631/ai-engineering-hub/tree/main/notebook-lm-clone) | Python, RAG | Citations, multi-format | Simpler architecture |
| [parthsarthi03/raptor](https://github.com/parthsarthi03/raptor) | Python | Hierarchical indexing | Core LOD implementation |
| [TeacherOp/NoobBook](https://github.com/TeacherOp/NoobBook) | React | Educational, 3-panel UI | UI reference |

---

## Design Decisions

### Decision Log

| ID | Decision | Rationale | Status | Date |
|----|----------|-----------|--------|------|
| D001 | Use RAPTOR-style hierarchical indexing | Matches our LOD concept, proven effective | **ACCEPTED** | 2025-01-21 |
| D002 | TUI interface (not web) | Unique differentiation, local-first, keyboard-driven | **ACCEPTED** | 2025-01-21 |
| D003 | Mandatory citations in all responses | Core requirement for 90-100% accuracy verification | **ACCEPTED** | 2025-01-21 |
| D004 | Start with single-agent RAG, evolve to multi-agent | Avoid over-engineering; add complexity when needed | **PROPOSED** | 2025-01-21 |
| D005 | SQLite + sqlite-vss for storage | Local-first, no server dependency, good enough perf | **PROPOSED** | 2025-01-21 |
| D006 | Python for backend | RAPTOR/LlamaIndex ecosystem, rapid prototyping | **PROPOSED** | 2025-01-21 |
| D007 | Textual for TUI framework | Modern, async, rich widgets, Python native | **PROPOSED** | 2025-01-21 |

---

## Proposed Architecture

### High-Level Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         KNOWLEDGE ASSISTANT                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                     TUI (Textual)                           │     │
│  │                                                             │     │
│  │  ┌─────────────┐  ┌─────────────────────┐  ┌────────────┐  │     │
│  │  │  Sources    │  │      Chat           │  │  Details   │  │     │
│  │  │  Panel      │  │      Panel          │  │  Panel     │  │     │
│  │  │             │  │                     │  │            │  │     │
│  │  │ - doc1.pdf  │  │ > What is X?        │  │ Citation   │  │     │
│  │  │ - doc2.md   │  │                     │  │ preview    │  │     │
│  │  │ - notes/    │  │ X is... [1][2]      │  │            │  │     │
│  │  │             │  │                     │  │ [1] p.23   │  │     │
│  │  │             │  │ Sources:            │  │ "exact..." │  │     │
│  │  │             │  │ [1] doc1.pdf:23     │  │            │  │     │
│  │  └─────────────┘  └─────────────────────┘  └────────────┘  │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                │                                     │
│                                ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                    QUERY ENGINE                             │     │
│  │                                                             │     │
│  │  Query → [Router] → [Retriever] → [Reranker] → [Synthesizer]│     │
│  │              │            │            │             │      │     │
│  │          (decides     (searches    (scores      (generates  │     │
│  │           LOD level)   tree)       results)     + cites)    │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                │                                     │
│                                ▼                                     │
│  ┌────────────────────────────────────────────────────────────┐     │
│  │                   KNOWLEDGE BASE                            │     │
│  │                                                             │     │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │     │
│  │  │   Documents  │    │  RAPTOR Tree │    │   Vectors    │  │     │
│  │  │   (SQLite)   │    │   (SQLite)   │    │ (sqlite-vss) │  │     │
│  │  └──────────────┘    └──────────────┘    └──────────────┘  │     │
│  │                                                             │     │
│  │  Schema:                                                    │     │
│  │  - documents(id, path, title, content, created_at)         │     │
│  │  - nodes(id, doc_id, parent_id, level, content, summary)   │     │
│  │  - embeddings(node_id, vector)                             │     │
│  └────────────────────────────────────────────────────────────┘     │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Model (LOD Structure)

```
Document
├── id: UUID
├── path: str (file path)
├── title: str
├── content: str (full text)
├── summary: str (L3 - document-level summary)
├── created_at: datetime
└── nodes: List[Node]

Node (RAPTOR tree node)
├── id: UUID
├── document_id: FK
├── parent_id: FK (nullable, for tree structure)
├── level: int (0=chunk, 1=paragraph summary, 2=section, 3=document)
├── content: str (original text or summary)
├── embedding: vector
├── children: List[Node]
└── metadata: JSON (page number, section title, etc.)
```

### Query Flow

```
1. User enters question in TUI
                ↓
2. Router Agent (small LLM) analyzes question:
   - Broad/conceptual → search L2-L3
   - Specific/detailed → search L0-L1
   - Multi-part → decompose into sub-queries
                ↓
3. Retriever searches selected LOD levels:
   - Embed query
   - Vector similarity search
   - BM25 keyword search (hybrid)
   - Return top-K nodes with scores
                ↓
4. Reranker (optional) re-scores results:
   - Cross-encoder model
   - Filter low-confidence matches
                ↓
5. Synthesizer (large LLM) generates response:
   - Context = retrieved nodes
   - MUST cite sources: [doc_name:page/section]
   - If insufficient evidence: "I cannot find information about X in the sources"
                ↓
6. TUI displays:
   - Answer with inline citations
   - Expandable source references
   - Confidence indicator
```

---

## Tech Stack (Proposed)

| Layer | Technology | Alternatives Considered |
|-------|------------|------------------------|
| **TUI** | Textual (Python) | Ratatui (Rust), Bubbletea (Go) |
| **Backend** | Python 3.11+ | - |
| **Database** | SQLite + sqlite-vss | PostgreSQL + pgvector, ChromaDB |
| **Embeddings** | sentence-transformers (local) | OpenAI ada-002, Cohere |
| **Router LLM** | Claude Haiku / GPT-4o-mini | Llama 3.1 8B (local) |
| **Synthesizer LLM** | Claude Sonnet | GPT-4o, Gemini Pro |
| **Document Parsing** | pypdf, markdown-it | LlamaParse, Unstructured |
| **Framework** | Custom (or LlamaIndex) | LangChain, DSPy |

---

## Open Questions

1. **Local vs API LLMs**: Should we support fully local operation with Ollama/llama.cpp?
2. **Incremental indexing**: How to handle document updates without full re-index?
3. **Multi-notebook support**: One knowledge base or multiple isolated collections?
4. **Export formats**: Should answers be exportable (markdown, PDF)?
5. **Conversation memory**: How much chat history to maintain for follow-up questions?

---

## Detailed Documentation

### RAPTOR Deep Dive
Comprehensive reverse-engineering of the RAPTOR implementation (~10,500 lines of documentation):

**[Full RAPTOR Documentation →](docs/research/raptor/README.md)**

| Category | Documents |
|----------|-----------|
| **Discovery** | [Codebase Inventory](docs/research/raptor/01-discovery/codebase-inventory.md), [Dependencies](docs/research/raptor/01-discovery/dependency-analysis.md), [API Surface](docs/research/raptor/01-discovery/api-surface.md) |
| **Structure** | [Components](docs/research/raptor/02-structure/component-diagram.md), [Classes](docs/research/raptor/02-structure/class-diagram.md), [Data Model](docs/research/raptor/02-structure/data-model.md), [Packages](docs/research/raptor/02-structure/package-diagram.md) |
| **Behavior** | [Sequences](docs/research/raptor/03-behavior/sequence-diagrams.md), [States](docs/research/raptor/03-behavior/state-diagrams.md), [Flowcharts](docs/research/raptor/03-behavior/activity-flowcharts.md), [Data Flow](docs/research/raptor/03-behavior/data-flow.md) |
| **Algorithms** | [Clustering](docs/research/raptor/04-algorithms/clustering.md), [Tree Building](docs/research/raptor/04-algorithms/tree-building.md), [Retrieval](docs/research/raptor/04-algorithms/retrieval.md), [Complexity](docs/research/raptor/04-algorithms/complexity-analysis.md) |
| **Integration** | [Extension Points](docs/research/raptor/05-integration/extension-points.md), [Configuration](docs/research/raptor/05-integration/configuration.md), [Error Handling](docs/research/raptor/05-integration/error-handling.md) |
| **Assessment** | [Code Quality](docs/research/raptor/06-assessment/code-quality.md) (5.5/10), [Test Coverage](docs/research/raptor/06-assessment/test-coverage.md) (0%), [Performance](docs/research/raptor/06-assessment/performance.md) |

---

## Benchmark Results (Local Models)

POC benchmark run on 2026-01-21 with local models. See `poc/raptor_test/` for code.

### Configuration
- **Document**: ~1,500 words about Python programming language history
- **Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384 dims, local)
- **LLM**: `llama3.2:3b` via Ollama (local)
- **Chunk Size**: 50 tokens
- **Clustering**: PCA + GMM (UMAP skipped due to dependency issues)

### Results

| Metric | Value |
|--------|-------|
| **Chunks Created** | 21 |
| **Total Nodes** | 26 (21 leaves + 5 summaries) |
| **Tree Layers** | 1 |
| **Total Indexing Time** | 40.86s |

#### Timing Breakdown

| Phase | Time | % of Total |
|-------|------|------------|
| Chunking | 0.001s | 0.0% |
| Embedding | 0.17s | 0.4% |
| Clustering | 0.34s | 0.8% |
| **Summarization** | **40.28s** | **98.6%** |

#### Accuracy

| Metric | Score |
|--------|-------|
| Retrieval Accuracy | 80% (8/10) |
| QA Accuracy | 70-90% (7-9/10, varies by run) |

### Key Findings

1. **Summarization is the bottleneck**: 98.6% of indexing time spent on LLM summarization calls
   - 5 summaries took ~40 seconds = ~8 seconds per summary with llama3.2:3b
   - This confirms RAPTOR's documented O(N) LLM call scaling issue

2. **Local embeddings are fast**: 0.17s for 21 embeddings = ~8ms each
   - SBERT on CPU is plenty fast for our scale
   - No need for API embeddings

3. **Accuracy is acceptable**: 70-90% QA accuracy with local 3B model
   - Retrieval works well (80%)
   - QA errors mostly from LLM quality, not retrieval failures

4. **Hierarchy helps**: With hierarchy (26 nodes) vs flat (21 chunks):
   - Summaries provide topic-level context
   - But requires LLM calls which dominate time

### Implications for Our Design

| Finding | Implication |
|---------|-------------|
| Summarization is 98% of time | Consider lazy summarization, extractive summaries, or skip hierarchy for small docs |
| Local embeddings work | No need for embedding API - use SBERT by default |
| Local LLM is viable | 3B model gives 70-90% accuracy - acceptable for personal use |
| PCA works as UMAP fallback | Don't need UMAP complexity for small-medium docs |

---

## Next Steps

- [x] ~~Reverse-engineer RAPTOR architecture~~
- [x] ~~Analyze RAPTOR benefits & drawbacks~~
- [x] ~~Benchmark with local models~~
- [ ] **Decision needed**: Finalize clustering approach (UMAP+GMM vs alternatives)
- [ ] **Decision needed**: Local vs API models (cost vs quality tradeoff)
- [ ] **Decision needed**: Target scale (affects architecture)
- [ ] Finalize tech stack decisions
- [ ] Design detailed database schema (with incremental update support)
- [ ] Prototype tree builder (our improved version)
- [ ] Build basic TUI shell
- [ ] Implement document ingestion pipeline
- [ ] Add vector search
- [ ] Implement citation extraction
- [ ] Add multi-agent orchestration (phase 2)

---

## References

### Papers
- [RAPTOR: Recursive Abstractive Processing for Tree-Organized Retrieval](https://arxiv.org/abs/2401.18059)
- [MA-RAG: Multi-Agent RAG via Collaborative Chain-of-Thought](https://arxiv.org/html/2505.20096v2)
- [MAO-ARAG: Multi-Agent Orchestration for Adaptive RAG](https://arxiv.org/abs/2508.01005)

### Implementations
- [RAPTOR Official](https://github.com/parthsarthi03/raptor)
- [Mastra NotebookLM Clone](https://github.com/mastra-ai/notebooklm-mastra)
- [LlamaIndex Hierarchical Parsing](https://docs.llamaindex.ai/en/stable/examples/node_parsers/hierarchical/)

### Articles
- [Building NotebookLM Clone with Agent Orchestration](https://mastra.ai/blog/notebooklm-clone-with-agent-orchestration)
- [Implementing RAPTOR in LangChain](https://medium.com/the-ai-forum/implementing-advanced-rag-in-langchain-using-raptor-258a51c503c6)
