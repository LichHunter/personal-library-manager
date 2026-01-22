# Personal Knowledge Assistant - Architecture DRAFT

> ⚠️ **STATUS: DRAFT** - Some decisions confirmed via benchmarks, others pending testing.

## Overview

Local-first, NotebookLM-like system for querying 1000+ documents with grounded, cited answers.

**Confirmed Decisions (via benchmarks):**
- Hybrid retrieval (BM25 + semantic) instead of LLM routing (small models fail at routing) — `poc/retrieval_benchmark/`
- **Paragraph-based chunking** (50-256 tokens) — `poc/chunking_benchmark/`
- Single SQLite database with sqlite-vec + FTS5

**Pending Testing:**
- Cross-encoder reranker model selection
- LLM selection (llama3:8b vs alternatives)
- Citation extraction/verification approach
- Query classification heuristics

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              API LAYER                                       │
│         POST /index    POST /query    GET/POST /session/:id                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
┌──────────────────┐    ┌─────────────────────────────┐    ┌──────────────────┐
│  INDEX SERVICE   │    │       QUERY PIPELINE        │    │ SESSION SERVICE  │
│                  │    │                             │    │                  │
│ • Parse/convert  │    │  ┌─────────────────────┐   │    │ • Conv. state    │
│ • Chunk          │    │  │   QUERY ANALYZER    │   │    │ • Follow-up ctx  │
│ • Embed          │    │  │ • Heuristic classify│   │    │ • History        │
│ • BM25 index     │    │  │ • Decompose complex │   │    │                  │
│ • Hash tracking  │    │  └──────────┬──────────┘   │    └──────────────────┘
└──────────────────┘    │             ▼              │
                        │  ┌─────────────────────┐   │
                        │  │  HYBRID RETRIEVER   │   │
                        │  │ • BM25 (FTS5)       │   │
                        │  │ • Vector (sqlite-vec)│  │
                        │  │ • Reciprocal Rank   │   │
                        │  │   Fusion → top-50   │   │
                        │  └──────────┬──────────┘   │
                        │             ▼              │
                        │  ┌─────────────────────┐   │
                        │  │     RERANKER        │   │
                        │  │ • Cross-encoder     │   │
                        │  │ • bge-reranker-base │   │
                        │  │ • → top-10 final    │   │
                        │  └──────────┬──────────┘   │
                        │             ▼              │
                        │  ┌─────────────────────┐   │
                        │  │    SYNTHESIZER      │   │
                        │  │ • LLM generation    │   │
                        │  │ • Inline citations  │   │
                        │  │ • Conflict handling │   │
                        │  │ • "Not found" cases │   │
                        │  └──────────┬──────────┘   │
                        │             ▼              │
                        │  ┌─────────────────────┐   │
                        │  │      CITATOR        │   │
                        │  │ • Verify citations  │   │
                        │  │ • Extract quotes    │   │
                        │  │ • Flag unsupported  │   │
                        │  │ • Confidence score  │   │
                        │  └─────────────────────┘   │
                        └─────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                        STORAGE LAYER (Single SQLite)                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │  Documents +    │  │  sqlite-vec     │  │      FTS5                   │  │
│  │  Chunks +       │  │  (embeddings)   │  │  (BM25 full-text)           │  │
│  │  Sessions       │  │                 │  │                             │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MODEL LAYER                                        │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────────┐  │
│  │ Embedding       │  │  Reranker       │  │          LLM                │  │
│  │ all-MiniLM-L6   │  │ bge-reranker-   │  │  llama3:8b (local)          │  │
│  │ (384 dims)      │  │ base            │  │  or API fallback            │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Index Service

**Responsibilities:**
- Parse documents (Markdown as canonical format)
- Chunk using **paragraph-based strategy** (see below)
- Generate embeddings (all-MiniLM-L6-v2, 384 dimensions)
- Build BM25 index (FTS5)
- Track document hashes for change detection

**Chunking Strategy (Confirmed via `poc/chunking_benchmark/`):**
```python
ParagraphStrategy(min_tokens=50, max_tokens=256)
```
- Split on paragraph boundaries (`\n\n`)
- Merge consecutive paragraphs under 50 tokens
- Split paragraphs over 256 tokens at sentence boundaries
- Prepend section heading to each chunk for context

**Benchmark Results:**
| Strategy | Recall@5 | MRR |
|----------|----------|-----|
| **paragraphs_50_256** | **96.4%** | **0.940** |
| heading_based | 91.4% | 0.912 |
| fixed_size_512 | 88.6% | 0.908 |

**Indexing Flow:**
```
File → Parse → Paragraph Chunk → Embed → Store (SQLite + sqlite-vec + FTS5)
```

### 2. Query Pipeline

#### 2.1 Query Analyzer
- **Heuristic complexity classification** (no LLM)
- Patterns detected:
  - Simple: single entity lookup ("What does X do?")
  - Complex: comparison, exhaustive, system understanding, decision queries
- Decompose complex queries into sub-queries

```python
complex_patterns = [
    r'\b(compare|contrast|difference|vs)\b',      # Comparison
    r'\b(all|every|each|list)\b',                  # Exhaustive
    r'\b(how does .* work|architecture|flow)\b',   # System understanding
    r'\b(why|decision|rationale)\b',               # Decision queries
]
```

#### 2.2 Hybrid Retriever
- **BM25** via FTS5 (keyword matching)
- **Vector search** via sqlite-vec (semantic similarity)
- **Reciprocal Rank Fusion** to combine results
- Returns top-50 candidates

#### 2.3 Reranker
- Cross-encoder model (bge-reranker-base)
- Re-scores top-50 candidates against query
- Returns top-10 for synthesis

#### 2.4 Synthesizer
- LLM generates answer with inline citations
- Handles conflicts ("Source [1] says X, while [2] says Y")
- Explicit "not found" when information missing

**Synthesis Prompt:**
```
Answer using ONLY the provided context.
For each claim, cite using [1], [2], etc.

Rules:
1. If not found, say "I could not find information about X"
2. If sources conflict, present both: "Source [1] says X, while [2] says Y"
3. Every factual claim MUST have a citation
4. Use exact quotes when possible

Context:
{chunks_with_ids}

Question: {query}
```

#### 2.5 Citator
- Extract citations from generated answer
- Verify each claim against source chunk
- Flag unsupported claims
- Generate confidence score

### 3. Session Service

**Responsibilities:**
- Maintain conversation state
- Provide context for follow-up questions
- Store message history with citations

### 4. Adaptive Effort System

```python
def adaptive_search(query, mode="auto"):
    if mode == "quick":
        return quick_search(query)    # 5-10s, top-20 retrieval
    elif mode == "deep":
        return deep_search(query)     # up to 5 min, decomposition + exhaustive
    else:
        # Auto: classify and potentially escalate
        complexity = classify_complexity(query)
        
        if complexity == SIMPLE:
            return quick_search(query)
        
        result = moderate_search(query)
        
        # Escalate if insufficient
        if result.confidence < 0.7 or len(result.sources) < 2:
            return deep_search(query)
        
        return result
```

---

## Data Model

Based on chunking benchmark results, using paragraph-based chunks with heading context.

```sql
CREATE TABLE documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL,
    title TEXT,
    content_hash TEXT NOT NULL,      -- For change detection
    metadata JSON,
    indexed_at TIMESTAMP,
    updated_at TIMESTAMP
);

CREATE TABLE chunks (
    id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL REFERENCES documents(id),
    content TEXT NOT NULL,           -- Full chunk content (50-256 tokens)
    heading TEXT,                    -- Section heading for context
    heading_path TEXT,               -- e.g., "Architecture > Database > Schema"
    start_char INTEGER,
    end_char INTEGER,
    token_count INTEGER,
    metadata JSON
);
```

**Design Choice:** Store chunk content in DB (not just file references) because:
1. Chunks need content for embedding and retrieval display
2. Heading context is prepended during chunking (not in original file)
3. Single SQLite file is easier to backup/sync

### Common Tables

```sql
-- Vector embeddings (sqlite-vec)
CREATE VIRTUAL TABLE chunk_embeddings USING vec0(
    chunk_id TEXT PRIMARY KEY,
    embedding FLOAT[384]
);

-- Full-text search (FTS5 for BM25)
CREATE VIRTUAL TABLE chunks_fts USING fts5(
    content,
    content='chunks'
);

-- Sessions
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP,
    last_active TIMESTAMP
);

CREATE TABLE session_messages (
    id TEXT PRIMARY KEY,
    session_id TEXT REFERENCES sessions(id),
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    citations JSON,
    created_at TIMESTAMP
);
```

---

## Model Selection

| Component | Model | Size | Notes |
|-----------|-------|------|-------|
| Embedding | all-MiniLM-L6-v2 | 80MB | 384 dims, fast, good quality |
| Reranker | bge-reranker-base | 278MB | Cross-encoder, high precision |
| LLM | llama3:8b | ~4.7GB | Fits 8GB VRAM with quantization |

**Fallback:** API-based models (OpenAI, Anthropic) for better quality when available.

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Index time (1000 docs) | < 30 minutes |
| Quick search | 5-10 seconds |
| Deep search | up to 5 minutes |
| Reranking latency | 100-200ms for top-50 |

---

## Key Design Rationale

### Why Hybrid Retrieval (not LOD/RAPTOR)?
Benchmark results showed:
- LOD-LLM: 0% section recall with llama3.2:3b (small models can't route)
- RAPTOR: 11x slower indexing, same accuracy as flat search
- Flat embedding: 100% doc recall, fast

**Conclusion:** Use simple retrieval + smart synthesis, not complex routing.

### Why Cross-encoder Reranking?
- Significantly improves precision without LLM overhead
- bge-reranker-base is small enough to run locally
- Filters noise before expensive LLM synthesis

### Why Post-hoc Citation Verification?
- Critical requirement: no hallucinations
- Generate-then-verify catches errors without slowing generation
- Can flag low-confidence claims to user

### Why Single SQLite?
- Simpler deployment (single file)
- ACID transactions
- sqlite-vec + FTS5 sufficient for 1000 docs scale
- Easy backup/restore

---

## Implementation Phases

| Phase | Duration | Deliverables |
|-------|----------|--------------|
| **Phase 1: Core** | 1 week | SQLite schema, ingestion, chunking, hybrid retrieval, basic synthesis |
| **Phase 2: Quality** | 1 week | Reranking, citation verification, confidence scoring, conflict handling |
| **Phase 3: Adaptive** | 3-4 days | Query classification, decomposition, quick/deep modes, sessions |
| **Phase 4: Polish** | 3-4 days | API endpoints, dev scripts, evaluation, docs |

---

## Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Small model hallucination | High | Critical | Post-hoc citation verification |
| Reranker latency | Medium | Medium | Batch reranking, limit to top-50 |
| Query decomposition quality | Medium | Medium | Heuristics first, LLM only for complex |
| Citation extraction errors | Medium | High | Structured format + regex + LLM hybrid |

---

## Open Questions (Candidates for POC Testing)

### ✅ Resolved
1. ~~Document storage strategy~~ → Store chunks in DB with heading context
2. ~~Chunk size optimization~~ → 50-256 token paragraphs (96.4% Recall@5)

### 🔬 Candidates for Testing

#### 1. Reranker Model Selection
**Question:** bge-reranker-base vs bge-reranker-v2-m3 vs ms-marco-MiniLM?

**Test approach:**
- Use chunking benchmark corpus + queries
- Compare reranker precision (Precision@5, Precision@10)
- Measure latency for top-50 reranking
- Check VRAM usage

**Expected outcome:** Determine best quality/speed tradeoff for local inference.

#### 2. LLM Selection for Synthesis
**Question:** llama3:8b vs mistral:7b vs phi-3 vs qwen2.5:7b?

**Test approach:**
- Create 20 synthesis test cases with ground-truth answers
- Measure: answer correctness, citation accuracy, hallucination rate
- Compare latency and VRAM usage

**Expected outcome:** Find model that follows citation instructions reliably.

#### 3. Citation Extraction Accuracy
**Question:** Regex-based vs LLM-based vs hybrid citation extraction?

**Test approach:**
- Generate 50 sample answers with citations
- Compare extraction methods for precision/recall
- Measure false positives (hallucinated citations)

**Expected outcome:** Determine if simple regex suffices or needs LLM verification.

#### 4. BM25 vs Vector vs Hybrid Retrieval Weights
**Question:** What's the optimal fusion weight for BM25 + vector search?

**Test approach:**
- Test fusion weights: 0.3/0.7, 0.5/0.5, 0.7/0.3 (BM25/vector)
- Measure Recall@K on chunking benchmark queries
- Check if some query types favor BM25 vs vector

**Expected outcome:** Optimal fusion weights, potentially query-type-aware.

### 💭 Lower Priority (Defer Until MVP Working)
5. Query decomposition effectiveness
6. Confidence score calibration
7. Session context window size
