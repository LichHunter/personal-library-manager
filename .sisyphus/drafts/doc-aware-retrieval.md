# Draft: Doc-Aware Retrieval

## Requirements (confirmed)
- Improve retrieval by using doc_type metadata that already exists on chunks
- Q8 from manual grading failed (3/10) because architecture question retrieved user_guide chunks
- Current retrieval does NOT use doc_type metadata at all

## Current State

### Metadata Available on Chunks
From metadata enrichment work, chunks now have:
- `doc_type`: "api", "architecture", "operations", "troubleshooting", "user"
- `doc_title`: Document title
- `heading_path_str`: Full breadcrumb
- `chunk_index`, `total_chunks`, `chunk_position`

### Current Retrieval Flow (enriched_hybrid_llm.py)
1. Query rewrite via LLM
2. Domain expansion (hardcoded dictionary)
3. Encode query (semantic)
4. BM25 scoring (sparse)
5. RRF fusion of semantic + BM25 scores
6. Return top-k

**Problem**: No step uses chunk.metadata.doc_type

### Corpus Document Types
| doc_type | Document | Questions About |
|----------|----------|-----------------|
| api | API Reference | Rate limits, JWT, auth methods |
| architecture | Architecture Overview | Database stack, resources, capacity, latency |
| operations | Deployment Guide | Kubernetes, Helm, health endpoints, HPA |
| troubleshooting | Troubleshooting Guide | Error diagnosis, retry logic |
| user | User Guide | Workflow timeout, scheduling |

### Q8 Failure Analysis
- **Question**: "What is CloudFlow's workflow execution capacity?"
- **Expected doc_type**: architecture
- **Retrieved doc_type**: user (wrong!)
- **Root cause**: Query about "capacity" semantically similar to "limits" in user guide

## Technical Decisions

### Approach Options

1. **Query Classification + Hard Filter**
   - Classify query → detect doc_type intent
   - Filter chunks to only matching doc_type
   - **Risk**: Too restrictive, may miss cross-document answers

2. **Query Classification + Soft Boost**
   - Classify query → detect doc_type intent  
   - Add boost to RRF score for matching doc_type
   - **Better**: Allows cross-document retrieval with preference

3. **Self-Querying Retriever (LangChain pattern)**
   - LLM generates metadata filters from query
   - **Expensive**: Adds LLM call per query
   - **Complex**: Need to parse LLM output

4. **Keyword-Based Heuristic Classification**
   - Pattern match keywords to doc_types
   - No LLM call, fast
   - **Simple**: Easy to implement and debug

### Recommended: Hybrid Approach
- Use keyword heuristics for clear signals
- Add soft boost (not hard filter) to RRF scores
- Allow override via explicit user signal

## Open Questions
- [ ] What boost factor to use? (1.5x? 2x? 3x?)
- [ ] Should doc_type boost be configurable?
- [ ] How to handle queries that span multiple doc_types?

## Scope Boundaries
- INCLUDE: Add doc_type aware scoring to retrieval
- INCLUDE: Keyword-based query classification
- INCLUDE: Configurable boost factor
- EXCLUDE: Full self-querying retriever (too complex for now)
- EXCLUDE: LLM-based query classification (adds latency)
- EXCLUDE: Changes to chunking strategy (already done)

## Research Findings

### Query → Doc Type Mapping Heuristics
Based on ground_truth_realistic.json:

| Keywords | Likely doc_type |
|----------|----------------|
| "capacity", "resource", "architecture", "pod", "database", "stack", "P99", "latency", "RPO", "RTO", "Redis", "Kafka" | architecture |
| "rate limit", "JWT", "token", "auth", "API key", "OAuth", "algorithm" | api |
| "deploy", "helm", "kubernetes", "namespace", "HPA", "health", "PgBouncer" | operations |
| "error", "fix", "diagnose", "failing", "troubleshoot", "retry" | troubleshooting |
| "workflow", "schedule", "timeout", "trigger" | user |

### Test Strategy Decision
- **Infrastructure exists**: YES (benchmark framework)
- **User wants tests**: Manual verification via benchmark
- **QA approach**: Run benchmark before/after, compare Q8 specifically
