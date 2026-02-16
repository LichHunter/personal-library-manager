# System Design Plan

## Progress Tracker

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Architecture Design | 🔶 In Progress | Draft v1 created, blocked on data model decision |
| 2. Core Algorithm Design | 🔲 Pending | |
| 3. Data Model Design | 🔶 Blocked | Waiting for chunking benchmark |
| 4. Technology Selection | 🔲 Pending | |
| 5. API Design | 🔲 Pending | |
| 6. Evaluation Framework | 🔲 Pending | |

## Active POCs

| POC | Status | Location | Blocking |
|-----|--------|----------|----------|
| Chunking Strategy Benchmark | 🔶 In Design | `poc/chunking_benchmark/` | Architecture (data model) |

---

## Phase 1: Architecture Design

### Goals
- [ ] High-level component diagram
- [ ] Data flow for indexing operation
- [ ] Data flow for query operation
- [ ] Data flow for follow-up questions
- [ ] Define component boundaries and responsibilities

### Decisions Made
- Hybrid retrieval (BM25 + semantic) instead of LLM routing
- Cross-encoder reranking for precision
- Post-hoc citation verification
- Single SQLite with sqlite-vec + FTS5
- Heuristic query classification (no LLM for routing)
- Query pipeline split into: Analyzer → Retriever → Reranker → Synthesizer → Citator

### Open Questions (Blocking)
- **How to chunk/store documents?** → Requires chunking benchmark POC
- What granularity to index (section vs paragraph vs fixed)?
- Store full content in DB or reference files on disk?

---

## Phase 2: Core Algorithm Design

### Goals
- [ ] Indexing pipeline (ingest → chunk → embed → store)
- [ ] Query analysis & decomposition
- [ ] Retrieval strategy (adaptive effort)
- [ ] Synthesis with grounding
- [ ] Citation generation & verification

### Decisions Made
_(to be filled)_

### Open Questions
_(to be filled)_

---

## Phase 3: Data Model Design

### Goals
- [ ] Document storage schema
- [ ] Chunk schema with metadata
- [ ] Index structures
- [ ] Cross-reference tracking
- [ ] Session/conversation state

### Decisions Made
_(to be filled)_

### Open Questions
_(to be filled)_

---

## Phase 4: Technology Selection

### Goals
- [ ] Embedding model selection
- [ ] LLM selection (local 7B/13B)
- [ ] Vector store selection
- [ ] Framework selection
- [ ] Other dependencies

### Decisions Made
_(to be filled)_

### Open Questions
_(to be filled)_

---

## Phase 5: API Design

### Goals
- [ ] Indexing endpoints
- [ ] Query endpoints
- [ ] Response format specification
- [ ] Error handling
- [ ] Dev scripts for local testing

### Decisions Made
_(to be filled)_

### Open Questions
_(to be filled)_

---

## Phase 6: Evaluation Framework

### Goals
- [ ] Accuracy metrics definition
- [ ] Hallucination detection approach
- [ ] Benchmark test cases
- [ ] Success/failure criteria measurement

### Decisions Made
_(to be filled)_

### Open Questions
_(to be filled)_

---

## Backlog (Deferred Topics)

1. Document change tracking & relations
2. Image/diagram understanding
3. Stronger hardware optimization
4. Multi-user support
5. Web UI / TUI interfaces

---

## Session Log

### Session 1
- Completed requirements gathering
- Created REQUIREMENTS.md
- Started Phase 1: Architecture Design
- Created ARCHITECTURE_V1.md (DRAFT - pending data model decision)
- Key insight from previous POC: small models fail at LLM routing, use hybrid retrieval instead
- Architecture BLOCKED on: how to chunk/store documents
- Created POC design for chunking benchmark (poc/chunking_benchmark/DESIGN.md)
- 6 chunking strategies to test:
  1. Fixed-size chunks (baseline)
  2. Heading-based sections
  3. Heading + size limit
  4. Hierarchical (parent-child)
  5. Semantic paragraphs
  6. Heading + paragraph hybrid

### Next Steps
1. Run chunking benchmark POC
2. Based on results, finalize data model
3. Update ARCHITECTURE_V1.md with final data model
4. Then proceed to Phase 2 (Core Algorithm Design)
