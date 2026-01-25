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

---

## Detailed Missed Facts Analysis (2026-01-25 Baseline Investigation)

**Source**: `baseline_investigation.log` (semantic strategy section, lines 102-1213)
**Baseline Strategy**: semantic (BAAI/bge-base-en-v1.5, fixed_512_0pct, k=5)
**Coverage**: 67.9% (36/53 facts found)
**Missed Facts**: 17 unique facts (across 120 query variations)

### Complete Missed Facts Inventory

#### Fact 1: Monitor X-RateLimit-Remaining header values
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | Not in top-5 (fact exists in corpus but not retrieved) |
| **BM25 Rank** | N/A (semantic-only baseline) |
| **Semantic Rank** | >5 (outside retrieval window) |
| **Root Cause** | SEMANTIC_GAP - Monitoring advice lacks semantic similarity to rate limit queries |
| **Queries Affected** | 429 error queries, rate limit handling queries |

#### Fact 2: Implement exponential backoff when receiving 429 responses
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | CHUNK_BOUNDARY - Best practice buried in chunk with rate limit info |
| **Queries Affected** | "My requests keep failing with 429 status code" |

#### Fact 3: Retry-After
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | troubleshooting_guide_fix_6 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | CROSS_DOCUMENT - Header info in troubleshooting, not API reference |
| **Queries Affected** | Rate limit handling queries |

#### Fact 4: max 3600 seconds from iat
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_0 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | TECHNICAL_JARGON - "iat" (issued-at) is JWT-specific terminology |
| **Queries Affected** | All 6 JWT token expiration queries |

#### Fact 5: All tokens expire after 3600 seconds
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_0 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | REDUNDANT_FACT - Same info as "3600 seconds" but phrased differently |
| **Queries Affected** | Token expiration queries |

#### Fact 6: PostgreSQL 15.4 / Redis 7.2 / Apache Kafka 3.6
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - Query "database stack" doesn't match "PostgreSQL" semantically |
| **Queries Affected** | "database stack" (casual query) |

#### Fact 7: 2 vCPU, 4GB RAM per pod
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "api gateway resources" doesn't embed close to resource specs |
| **Queries Affected** | "api gateway resources" (casual query) |

#### Fact 8: /ready
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_2 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | PARTIAL_RETRIEVAL - /health found but /ready in different chunk section |
| **Queries Affected** | 6 health endpoint queries |

#### Fact 9: targetCPUUtilizationPercentage: 70
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | GRANULARITY - Specific config value not semantically linked to HPA queries |
| **Queries Affected** | 6 HPA/autoscaling queries |

#### Fact 10: minReplicas: 3 / maxReplicas: 10
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | INDIRECT_QUERY - "Why do we have 3 replicas even with low traffic?" |
| **Queries Affected** | Negation-style query about replicas |

#### Fact 11: P99 latency: < 200ms / average P99 latency of 180ms
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0, architecture_overview_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "api latency target" doesn't match P99 terminology |
| **Queries Affected** | 4-6 latency target queries |

#### Fact 12: RPO (Recovery Point Objective): 1 hour / RTO: 4 hours
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_10 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | ACRONYM_GAP - RPO/RTO acronyms don't embed well with disaster recovery queries |
| **Queries Affected** | All 6 disaster recovery queries (100% miss rate) |

#### Fact 13: exceeded maximum execution time of 3600 seconds
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | troubleshooting_guide_fix_4 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | CROSS_DOCUMENT - Error message in troubleshooting, timeout value in user guide |
| **Queries Affected** | 6 workflow timeout queries |

#### Fact 14: RS256 signing algorithm
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_0 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | REDUNDANT_FACT - "RS256" found but "RS256 signing algorithm" phrasing missed |
| **Queries Affected** | JWT algorithm queries |

#### Fact 15: Workflow Definitions: TTL: 1 hour
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_6 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | INDIRECT_QUERY - "My workflow updates aren't reflecting immediately" |
| **Queries Affected** | 3 cache TTL queries (problem/contextual variations) |

#### Fact 16: Jaeger for distributed tracing
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "monitoring stack" doesn't match Jaeger semantically |
| **Queries Affected** | "monitoring stack" (casual), some monitoring queries |

#### Fact 17: helm repo add cloudflow https://charts.cloudflow.io
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_1 |
| **BM25 Rank** | N/A |
| **Semantic Rank** | >5 |
| **Root Cause** | COMMAND_SYNTAX - Helm command syntax doesn't embed well with natural language |
| **Queries Affected** | 5/6 Helm repo queries (83% miss rate) |

---

### Root Cause Category Summary

| Category | Count | Description | Example |
|----------|-------|-------------|---------|
| **VOCABULARY_MISMATCH** | 5 | Query terms don't semantically match document terms | "database stack" vs "PostgreSQL 15.4" |
| **ACRONYM_GAP** | 1 | Technical acronyms poorly embedded | "RPO RTO" query misses disaster recovery content |
| **CROSS_DOCUMENT** | 2 | Related info split across documents | Retry-After in troubleshooting, not API ref |
| **INDIRECT_QUERY** | 2 | Problem-style queries don't match factual content | "Why do we have 3 replicas..." |
| **CHUNK_BOUNDARY** | 1 | Fact buried in chunk with other dominant content | Exponential backoff in rate limit chunk |
| **PARTIAL_RETRIEVAL** | 1 | Related fact found, sibling fact missed | /health found, /ready missed |
| **TECHNICAL_JARGON** | 1 | Domain-specific terms poorly embedded | "iat" (JWT issued-at claim) |
| **REDUNDANT_FACT** | 2 | Same info phrased differently counted as miss | "3600 seconds" vs "All tokens expire after 3600 seconds" |
| **GRANULARITY** | 1 | Specific config values not linked to concept queries | targetCPUUtilizationPercentage: 70 |
| **COMMAND_SYNTAX** | 1 | CLI commands don't embed well | helm repo add command |

---

### Query Type Performance Analysis

| Query Type | Coverage | Missed Facts Pattern |
|------------|----------|---------------------|
| **original** | 67.9% | Baseline - direct questions |
| **synonym** | 60.4% | Vocabulary mismatch amplified |
| **problem** | 54.7% | Indirect queries fail most |
| **casual** | 54.7% | Short queries lack context |
| **contextual** | 62.3% | Better than problem/casual |
| **negation** | 54.7% | "Why doesn't X work" pattern fails |

---

### Improvement Recommendations

#### High Impact (Address 5+ missed facts)
1. **Hybrid Retrieval (BM25 + Semantic)** - Addresses VOCABULARY_MISMATCH, ACRONYM_GAP
2. **Query Expansion** - Addresses INDIRECT_QUERY, VOCABULARY_MISMATCH
3. **Chunk Overlap** - Addresses CHUNK_BOUNDARY, PARTIAL_RETRIEVAL

#### Medium Impact (Address 2-4 missed facts)
4. **Acronym Expansion** - Pre-process queries to expand RPO/RTO, JWT, etc.
5. **Cross-Document Linking** - Metadata linking related chunks across docs

#### Low Impact (Address 1 missed fact each)
6. **Command Indexing** - Special handling for CLI command syntax
7. **Fact Deduplication** - Consolidate redundant fact variations

---

### Chunk Location Summary

| Chunk ID | Document | Missed Facts |
|----------|----------|--------------|
| api_reference_fix_0 | api_reference | max 3600 seconds from iat, All tokens expire, RS256 signing |
| api_reference_fix_1 | api_reference | Implement exponential backoff |
| architecture_overview_fix_0 | architecture_overview | PostgreSQL/Redis/Kafka, P99 180ms, Jaeger |
| architecture_overview_fix_1 | architecture_overview | 2 vCPU 4GB RAM, P99 < 200ms |
| architecture_overview_fix_6 | architecture_overview | Workflow Definitions TTL |
| architecture_overview_fix_10 | architecture_overview | RPO 1 hour, RTO 4 hours |
| deployment_guide_fix_1 | deployment_guide | targetCPUUtilization, min/maxReplicas, helm repo |
| deployment_guide_fix_2 | deployment_guide | /ready endpoint |
| troubleshooting_guide_fix_4 | troubleshooting_guide | exceeded maximum execution time |
| troubleshooting_guide_fix_6 | troubleshooting_guide | Retry-After header |

---

### Finding 3: Root Cause Analysis - Enriched Hybrid Fast (2026-01-25)

## Enriched Hybrid Fast Analysis (2026-01-25 - PRIMARY BASELINE)

**Source**: `baseline_investigation.log` (enriched_hybrid_fast section, lines 2382-9291)
**Strategy**: enriched_hybrid_fast (BAAI/bge-base-en-v1.5, fixed_512_0pct, k=5, RRF fusion)
**Coverage**: 77.4% (41/53 facts found)
**Missed Facts**: 12 unique facts (across 120 query variations)

### Enriched Hybrid Fast - Missed Facts Inventory

The enriched_hybrid_fast strategy uses BM25 + Semantic with RRF (Reciprocal Rank Fusion) and YAKE/spaCy enrichment. Below are the 12 unique facts that remain missed even with this improved strategy.

#### Fact 1: Monitor X-RateLimit-Remaining header values
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_1 (contains fact but not surfaced) |
| **BM25 Rank** | Varies by query (2-4 typically) |
| **Semantic Rank** | Varies by query (0-9 typically) |
| **RRF Final Rank** | >5 (outside top-5 window) |
| **Root Cause** | FACT_BURIED - Monitoring advice buried in rate limit chunk; RRF doesn't boost it enough |
| **Queries Affected** | All 6 rate limit queries (100% miss rate for this specific fact) |

#### Fact 2: max 3600 seconds from iat
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_0 |
| **BM25 Rank** | 1-3 (good BM25 match on "JWT", "token") |
| **Semantic Rank** | 9+ (poor semantic match) |
| **RRF Final Rank** | 4-5 (borderline) |
| **Root Cause** | TECHNICAL_JARGON - "iat" (issued-at) is JWT-specific; semantic embedding doesn't capture it |
| **Queries Affected** | 4-6 JWT token expiration queries |

#### Fact 3: All tokens expire after 3600 seconds
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | api_reference_fix_0 |
| **BM25 Rank** | 1-3 |
| **Semantic Rank** | 9+ |
| **RRF Final Rank** | 4-5 |
| **Root Cause** | REDUNDANT_PHRASING - "3600 seconds" found but this exact phrasing missed |
| **Queries Affected** | Token expiration queries (problem/contextual variations) |

#### Fact 4: PostgreSQL 15.4 / Redis 7.2 / Apache Kafka 3.6
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0 |
| **BM25 Rank** | >10 (no keyword match for "database stack") |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "database stack" doesn't match specific technology names |
| **Queries Affected** | "database stack" (casual), "Is CloudFlow using MySQL" (negation) |

#### Fact 5: 2 vCPU, 4GB RAM per pod
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_1 |
| **BM25 Rank** | >10 for casual queries |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "api gateway resources" doesn't match resource specifications |
| **Queries Affected** | "api gateway resources" (casual), 1-2 resource queries |

#### Fact 6: targetCPUUtilizationPercentage: 70
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_1 |
| **BM25 Rank** | 5-10 |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | GRANULARITY - Specific YAML config value not semantically linked to HPA concept |
| **Queries Affected** | 5-6 HPA/autoscaling queries |

#### Fact 7: minReplicas: 3 / maxReplicas: 10
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | deployment_guide_fix_1 |
| **BM25 Rank** | >10 for negation query |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | INDIRECT_QUERY - "Why do we have 3 replicas even with low traffic?" doesn't match config |
| **Queries Affected** | Negation-style autoscaling queries |

#### Fact 8: P99 latency: < 200ms / average P99 latency of 180ms
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0, architecture_overview_fix_1 |
| **BM25 Rank** | >10 for "api latency target" |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "api latency target" doesn't match P99/percentile terminology |
| **Queries Affected** | 4-6 latency target queries |

#### Fact 9: RPO (Recovery Point Objective): 1 hour / RTO: 4 hours
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_10 |
| **BM25 Rank** | >10 (acronyms don't match well) |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | ACRONYM_GAP - RPO/RTO acronyms poorly embedded; disaster recovery queries miss this chunk |
| **Queries Affected** | All 6 disaster recovery queries (100% miss rate) |

#### Fact 10: TTL: 1 hour / Workflow Definitions: TTL: 1 hour
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_6 |
| **BM25 Rank** | 4-10 |
| **Semantic Rank** | 0-5 (good for direct queries) |
| **RRF Final Rank** | Varies (found in some, missed in others) |
| **Root Cause** | INDIRECT_QUERY - "My workflow updates aren't reflecting immediately" doesn't match TTL |
| **Queries Affected** | Problem/contextual cache queries |

#### Fact 11: Jaeger for distributed tracing
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | architecture_overview_fix_0 |
| **BM25 Rank** | >10 for "monitoring stack" |
| **Semantic Rank** | >10 |
| **RRF Final Rank** | >5 |
| **Root Cause** | VOCABULARY_MISMATCH - "monitoring stack" doesn't match "Jaeger" or "distributed tracing" |
| **Queries Affected** | "monitoring stack" (casual), some monitoring queries |

#### Fact 12: minimum scheduling interval is 1 minute
| Attribute | Value |
|-----------|-------|
| **Chunk ID** | user_guide_fix_4 |
| **BM25 Rank** | Varies |
| **Semantic Rank** | Varies |
| **RRF Final Rank** | >5 for original query |
| **Root Cause** | PHRASING_MISMATCH - "minimum scheduling interval is 1 minute" vs "The minimum scheduling interval is **1 minute**" |
| **Queries Affected** | Original scheduling interval query |

---

### Enriched Hybrid Fast - Root Cause Summary

| Category | Count | Description | Improvement from Semantic Baseline |
|----------|-------|-------------|-----------------------------------|
| **VOCABULARY_MISMATCH** | 4 | Query terms don't match document terms | Reduced from 5 (hybrid helps some) |
| **ACRONYM_GAP** | 1 | RPO/RTO acronyms poorly embedded | No improvement |
| **INDIRECT_QUERY** | 2 | Problem-style queries don't match facts | No improvement |
| **TECHNICAL_JARGON** | 1 | "iat" JWT terminology | No improvement |
| **GRANULARITY** | 1 | Specific config values not linked | No improvement |
| **FACT_BURIED** | 1 | Fact in retrieved chunk but not surfaced | New category for hybrid |
| **REDUNDANT_PHRASING** | 1 | Same info, different phrasing | Reduced from 2 |
| **PHRASING_MISMATCH** | 1 | Markdown formatting differences | New category |

---

### Enriched Hybrid Fast - Query Type Performance

| Query Type | Coverage | Delta from Original | Improvement over Semantic |
|------------|----------|---------------------|---------------------------|
| **original** | 77.4% | - | +9.5% |
| **synonym** | 71.7% | -5.7% | +11.3% |
| **problem** | 62.3% | -15.1% | +7.6% |
| **casual** | 64.2% | -13.2% | +9.5% |
| **contextual** | 67.9% | -9.5% | +5.6% |
| **negation** | 54.7% | -22.7% | +0% (no improvement) |

**Key Insight**: Negation queries show NO improvement with enriched hybrid. These queries like "Why doesn't X work?" or "Is CloudFlow using MySQL or something else?" require understanding of negation semantics that neither BM25 nor semantic embeddings capture well.

---

### Enriched Hybrid Fast - RRF Fusion Analysis

From the log traces, we can see how RRF combines BM25 and semantic rankings:

**Example: Query "How do I fix 429 Too Many Requests errors?"**
```
RRF[0] idx=4  total=0.0333 (sem_rank=0 + bm25_rank=0) -> api_reference_fix_4 (FOUND)
RRF[1] idx=34 total=0.0328 (sem_rank=1 + bm25_rank=1) -> troubleshooting_guide_fix_6 (FOUND)
RRF[2] idx=1  total=0.0323 (sem_rank=2 + bm25_rank=2) -> api_reference_fix_1 (FOUND)
RRF[3] idx=35 total=0.0308 (sem_rank=3 + bm25_rank=7) -> troubleshooting_guide_fix_7 (FOUND)
RRF[4] idx=46 total=0.0302 (sem_rank=10 + bm25_rank=3) -> user_guide_fix_6 (FOUND)
```
Result: 4/5 facts found. Missed: "Monitor X-RateLimit-Remaining header values" (in api_reference_fix_1 but not extracted)

**Example: Query "database stack" (casual)**
```
BM25 Top-10: No architecture_overview_fix_0 (contains PostgreSQL/Redis/Kafka)
Semantic Top-10: No architecture_overview_fix_0
RRF Result: Misses all 3 database technology facts
```
Result: 0/3 facts found. Root cause: VOCABULARY_MISMATCH

---

### Recommendations for Remaining 12 Missed Facts

#### High Priority (Address 4+ facts)
1. **Query Expansion/Rewriting** - Expand "database stack" -> "PostgreSQL Redis Kafka database"
2. **Acronym Dictionary** - Expand RPO/RTO, JWT, HPA before retrieval

#### Medium Priority (Address 2-3 facts)
3. **Negation-Aware Retrieval** - Special handling for "Why doesn't X" patterns
4. **Fact Extraction Post-Processing** - Extract specific facts from retrieved chunks

#### Low Priority (Address 1 fact each)
5. **Markdown Normalization** - Strip formatting before fact matching
6. **Synonym Injection** - Add "monitoring" -> "Prometheus Grafana Jaeger" mappings

---

## Oracle Consultation (2026-01-25)

**Session**: Task 3.1 - Validate Root Cause Analysis

### Oracle Feedback Summary

**Diagnosis Validation**: ✅ Sound and well-structured

**Key Refinement**: VOCABULARY_MISMATCH and ACRONYM_GAP are the **same root cause** - query-document vocabulary gaps. Both "database stack" → "PostgreSQL Redis Kafka" and "RPO RTO" → "Recovery Point Objective" are vocabulary mismatches.

### Recommended Solution Priority

#### Priority 1: Query Expansion with Domain Dictionary
- **Effort**: Quick (2-4 hours)
- **Addresses**: 5 facts (VOCABULARY_MISMATCH + ACRONYM_GAP)
- **Expected Coverage**: 86.8% (from 77.4%)
- **Implementation**: Add domain expansion dictionary to retrieval layer
- **Rationale**: Highest ROI - 5 facts with ~50 lines of code, no model changes

**Example Implementation**:
```python
DOMAIN_EXPANSIONS = {
    "rpo": "recovery point objective RPO data loss",
    "rto": "recovery time objective RTO downtime recovery",
    "database stack": "PostgreSQL Redis Kafka database storage",
    "monitoring": "Prometheus Grafana Jaeger observability metrics",
}
```

#### Priority 2: Negation-Aware Query Rewriting
- **Effort**: Short (4-8 hours)
- **Addresses**: 2 facts (INDIRECT_QUERY)
- **Expected Coverage**: 90.6% (from 86.8%)
- **Implementation**: Pattern-based query rewriting + multi-query retrieval
- **Rationale**: Handles "Why doesn't X work?" patterns

#### Priority 3: Deferred
- FACT_BURIED (1 fact): Extraction problem, not retrieval
- GRANULARITY (1 fact): Solved by Priority 1 dictionary
- PHRASING_MISMATCH (1 fact): Low value for 1 fact

### Testing Strategy

**Incremental Stacking** (recommended over isolated testing):
1. Baseline (current): 77.4%
2. Query expansion only: Expected 86.8%
3. Query expansion + negation: Expected 90.6%

**Rationale**: Solutions are additive, stacking shows cumulative progress

### Acceptable Thresholds

| Coverage | Assessment | Notes |
|----------|------------|-------|
| **90%+** | Excellent | Production-ready, edge cases only |
| **85-90%** | Good | Acceptable gaps, systematic coverage |
| **80-85%** | Minimum viable | Below 80%, users notice missing info |

### Action Plan

| Step | Task | Effort | Expected Gain |
|------|------|--------|---------------|
| 1 | Implement domain expansion dictionary | 2h | +5 facts → 86.8% |
| 2 | Test expansion in isolation | 1h | Validate approach |
| 3 | Add negation query rewriting | 4h | +2 facts → 90.6% |
| 4 | Run full benchmark | 30m | Confirm stacked results |
| 5 | Analyze remaining misses | 1h | Decide if worth pursuing |

**Estimated Total**: 1 day to reach ~90% coverage

### Watch Out For

1. **Over-expansion**: Aggressive dictionary may retrieve irrelevant chunks
2. **Negation rewriting false positives**: "Why doesn't X work?" might want troubleshooting, not feature docs
3. **Benchmark overfitting**: 20 queries is small sample, consider adding 10-20 more

### Escalation Triggers

Consider complex solutions (HyDE, fine-tuned embeddings, LLM query rewriting) only if:
- Query expansion + negation handling yields < 85% coverage
- New systematic failure pattern affecting 3+ facts identified
- Production user feedback shows retrieval failures not in benchmark

---

