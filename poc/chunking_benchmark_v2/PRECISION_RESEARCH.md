# Chunking Benchmark V2 Precision Investigation

**Investigation Started**: 2026-01-25T09:15:02.699Z  
**Session ID**: ses_40b953165ffeNCpBNux9VncnUI  
**Investigator**: Atlas (Orchestrator)

---

## Executive Summary

This document tracks the investigation into why the chunking benchmark v2 loses approximately 25% precision (measured as key facts coverage). The best-performing strategy (enriched_hybrid_fast) achieves only 77.4% coverage on original queries, with significant degradation on certain query types.

**Current Status**: Investigation in progress

---

## Investigation Log

### 2026-01-25 09:15 - Investigation Initiated

**Objective**: Identify root causes of ~25% precision loss in chunking benchmark v2

**Scope**:
- Analyze FastEnricher (YAKE + spaCy) enrichment process
- Examine text splitting and chunk boundaries
- Investigate metadata and keyword extraction
- Test hypotheses with empirical data
- Consult Oracle for solution validation

**Out of Scope**:
- Different embedding models
- Reranker optimization
- Implementation of solutions (validation only)

---

## Baseline Metrics

### Current Performance (2026-01-24)

**Test Corpus**:
- Documents: 5 CloudFlow technical documents (~3400 words each)
- Queries: 20 queries with 53 total key facts
- Chunking: Fixed-size 512 tokens, 0% overlap
- Chunks created: 51 total

**Coverage by Strategy**:

| Strategy | Original | Synonym | Problem | Casual | Contextual | Negation |
|----------|----------|---------|---------|--------|------------|----------|
| semantic | 67.9% | 60.4% | 54.7% | 54.7% | 62.3% | 54.7% |
| hybrid | 75.5% | 69.8% | 54.7% | 64.2% | 60.4% | 52.8% |
| **enriched_hybrid_fast** | **77.4%** | **71.7%** | **62.3%** | **64.2%** | **67.9%** | **54.7%** |

**Key Observations**:
- Best strategy misses **22.6%** of key facts on original queries
- **Negation queries** show worst degradation: -26.5% vs original
- **Problem queries** show -22.2% degradation vs original
- Enrichment provides only **+1.9%** improvement over hybrid baseline

### Degradation Analysis

From summary.json degradation_analysis:

| Dimension | Avg Coverage | Delta from Original | % Degradation |
|-----------|--------------|---------------------|---------------|
| negation | 54.1% | -19.5% | **-26.5%** |
| problem | 57.2% | -16.4% | **-22.2%** |
| casual | 61.0% | -12.6% | -17.1% |
| contextual | 63.5% | -10.1% | -13.7% |
| synonym | 67.3% | -6.3% | -8.5% |

---

## Hypotheses

### Hypothesis 1: Chunk Boundary Issues
**Description**: Key facts may be split across chunk boundaries, making them impossible to retrieve in a single chunk.

**Rationale**: Fixed-size chunking (512 tokens) doesn't respect semantic boundaries. A fact like "100 requests per minute per authenticated user" could be split if it spans a chunk boundary.

**Test Plan**: 
- Add logging to show chunk boundaries
- Identify if missed facts are split across chunks
- Test with smaller chunk size (256 tokens)

**Status**: DISPROVEN - See Finding 2

---

### Hypothesis 2: Enrichment Prefix Dilution
**Description**: Keywords/entities added by FastEnricher may not match query terms, diluting BM25 relevance.

**Rationale**: FastEnricher prepends "keywords | entities" to chunks. If these don't match the query, they add noise to BM25 scoring.

**Example**: 
- Original chunk: "CloudFlow enforces rate limits..."
- Enriched: "Rate Limiting, Rate Limit, Rate | CloudFlow, API\n\nCloudFlow enforces rate limits..."
- Query: "What is the API rate limit per minute?"

**Test Plan**:
- Log enrichment prefixes vs query terms
- Compare BM25 scores with/without enrichment
- Test retrieval without enrichment prefix

**Status**: Pending

---

### Hypothesis 3: BM25 vs Semantic Misalignment
**Description**: BM25 and semantic embeddings may rank chunks differently, causing RRF to miss relevant chunks.

**Rationale**: Hybrid retrieval uses Reciprocal Rank Fusion (RRF) to combine BM25 and semantic scores. If they disagree significantly, relevant chunks may rank lower.

**Test Plan**:
- Log BM25 rank vs semantic rank for each chunk
- Identify cases where BM25 and semantic disagree
- Test BM25-only vs semantic-only vs hybrid

**Status**: Pending

---

### Hypothesis 4: Semantic Embedding Shift
**Description**: Enrichment prefix may shift the semantic embedding away from the original content meaning.

**Rationale**: Adding keywords/entities changes the text that gets embedded. This could move the embedding in vector space away from the query.

**Test Plan**:
- Compare embeddings of original vs enriched chunks
- Measure cosine similarity shift
- Test if enrichment helps or hurts semantic retrieval

**Status**: Pending

---

### Hypothesis 5: Chunk Size Suboptimal
**Description**: 512-token chunks may be too large, including irrelevant content that dilutes relevance.

**Rationale**: Larger chunks have more content, which can dilute the signal for specific facts.

**Test Plan**:
- Test with 256-token chunks
- Test with 1024-token chunks
- Compare coverage across chunk sizes

**Status**: Pending

---

## Findings

### Finding 1: Missed Facts Analysis (2026-01-25)

**Date**: 2026-01-25 10:30  
**Hypothesis Tested**: General analysis of what facts are being missed  
**Result**: 69 out of 120 query-dimension combinations have missed facts

**Key Patterns Identified**:

1. **Facts exist in chunks but chunks not retrieved (RETRIEVAL FAILURE)**
   - Example: Query "database stack" should retrieve chunk with "PostgreSQL 15.4, Redis 7.2, Apache Kafka 3.6"
   - These facts are in `architecture_overview_fix_1` (chunk 1, chars 3231-5982)
   - But retrieved chunks were: architecture_overview_fix_12, deployment_guide_fix_3, user_guide_fix_7
   - **Root cause**: Semantic/keyword mismatch between query and relevant chunk

2. **Specific facts consistently missed across all dimensions**:
   - `TTL: 1 hour` and `Workflow Definitions: TTL: 1 hour` - 0% coverage across ALL dimensions
   - `RPO/RTO` facts - 0% coverage on most dimensions
   - `max_db_connections = 100` - 75% coverage (found in some but not all)

3. **Query dimension impact**:
   - **Negation queries** perform worst (-26.5% from original)
   - Example: "Is CloudFlow using MySQL or something else?" retrieves troubleshooting/API docs instead of architecture

**Missed Facts Summary Table**:

| Query ID | Missed Fact | In Chunk | Retrieved Instead |
|----------|-------------|----------|-------------------|
| realistic_004 | PostgreSQL 15.4, Redis 7.2, Kafka 3.6 | arch_fix_1 | arch_fix_12, deploy_fix_3 |
| realistic_010 | RPO 1 hour, RTO 4 hours | arch_fix_12 | Various other chunks |
| realistic_013 | TTL: 1 hour | arch_fix_6 | Other architecture chunks |
| realistic_003 | All tokens expire after 3600s | api_fix_1 | troubleshooting chunks |

**Evidence**: Benchmark results from 2026-01-25_102205

**Conclusion**: The primary issue is NOT chunk boundary splitting. Facts exist intact within chunks. The problem is **retrieval ranking** - the correct chunks are not ranked in top-5.

---

### Finding 2: Chunk Boundaries NOT Causing Splits

**Date**: 2026-01-25 10:35  
**Hypothesis Tested**: Hypothesis 1 - Chunk Boundary Issues

**Result**: DISPROVEN - Facts are NOT split across boundaries

**Evidence**:
- PostgreSQL 15.4 at pos 3375 -> CHUNK 1 (complete)
- Redis 7.2 at pos 3392 -> CHUNK 1 (complete)
- Apache Kafka 3.6 at pos 3424 -> CHUNK 1 (complete)
- All technology stack facts are in the SAME chunk (arch_fix_1)

**Conclusion**: Chunk boundaries are not the cause of precision loss. All key facts are contained completely within single chunks.

---

## Decision Log

### Decision 1: [To be populated after Oracle consultation]

**Date**: TBD  
**Context**: TBD  
**Options Considered**: TBD  
**Decision**: TBD  
**Rationale**: TBD  
**Oracle Feedback**: TBD

---

## Solutions

### Proposed Solutions

*This section will be populated after hypothesis testing and Oracle consultation.*

**Criteria for Solutions**:
- Must be validated by empirical testing
- Must be approved by Oracle
- Must show measurable improvement in coverage
- Must not significantly increase latency or complexity

---

## External Research

### Sources

*To be populated during research phase (Task 8)*

---

## Next Steps

1. ✅ Create research file structure
2. ⏳ Add detailed logging to FastEnricher
3. ⏳ Add detailed logging to EnrichedHybridRetrieval
4. ⏳ Add chunk boundary logging
5. ⏳ Run baseline benchmark with full logging
6. ⏳ Analyze missed facts per query
7. ⏳ Test hypotheses with experiments
8. ⏳ Research solutions online
9. ⏳ Consult Oracle for validation
10. ⏳ Document final conclusions

---

## Appendix

### Benchmark Configuration

**Embedding Model**: BAAI/bge-base-en-v1.5 (with prefix)  
**Chunking Strategy**: Fixed-size 512 tokens, 0% overlap  
**Retrieval Strategy**: Enriched Hybrid (BM25 + Semantic + FastEnricher)  
**Enrichment**: YAKE keywords (top 10) + spaCy NER  
**RRF Parameter**: k=60  
**Candidate Multiplier**: 10x  
**Top-k**: 5 chunks retrieved per query

### Key Files

- Benchmark runner: `run_benchmark.py`
- FastEnricher: `enrichment/fast.py`
- EnrichedHybridRetrieval: `retrieval/enriched_hybrid.py`
- FixedSizeStrategy: `strategies/fixed_size.py`
- Logger: `logger.py`
- Config: `config_fast_enrichment.yaml`
- Ground truth: `corpus/ground_truth_realistic.json`
- Latest results: `results/2026-01-24_234851/`

### Glossary

- **Coverage**: Percentage of key facts found in retrieved chunks (substring match)
- **Precision**: In this context, refers to coverage (not the traditional precision metric)
- **Key Facts**: Specific facts that must be present to answer a query (e.g., "100 requests per minute")
- **Enrichment**: Adding keywords/entities to chunks before embedding
- **RRF**: Reciprocal Rank Fusion - method to combine BM25 and semantic rankings
- **BM25**: Statistical ranking function for keyword matching
- **YAKE**: Yet Another Keyword Extractor - unsupervised keyword extraction
- **spaCy NER**: Named Entity Recognition using spaCy library
